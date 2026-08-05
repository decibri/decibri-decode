//! RIFF/WAVE and RF64: the first container this crate reads, and the first
//! bytes it parses that it did not produce.
//!
//! # Dispatch is on content, and on the pair
//!
//! `wFormatTag` alone does not say what the payload is. Tag 1 is "integer PCM"
//! at 8, 16, 24 or 32 bits, and reading a 24-bit file as 16-bit produces
//! plausible-looking noise at the wrong stride rather than an error. So every
//! decision here is on the **`(tag, bits)` pair**, in [`WavCodec::resolve`],
//! and never on the tag alone.
//!
//! Nothing in this crate looks at a file name. A `.wav` extension is a claim
//! made by whoever renamed the file; the first four bytes are a claim made by
//! whoever wrote it, and only the second one is checkable.
//!
//! # The two defects this reader was written not to repeat
//!
//! Both were measured in decibri's own `parse_wav` during an earlier pass and
//! left in place there under a read-only rule.
//!
//! **A partial trailing frame is an error, not a rounding.** decibri's reader
//! hands its `data` payload to a `chunks_exact` conversion and never compares
//! the payload length against the frame size, so a `data` chunk holding two and
//! a half stereo frames decodes as two and the half frame disappears. Silent
//! truncation of audio is the failure class this crate exists to avoid.
//! [`WavReader::new`] rejects it with [`DecodeError::Truncated`].
//!
//! **Size arithmetic is checked and 64-bit.** decibri's reader computes
//! `body + size > bytes.len()` in `usize` (`crates/decibri/src/file.rs:880`).
//! On a 32-bit target that sum wraps for a crafted chunk size, the range check
//! passes, and the slice that follows panics. Everything here is `u64` and
//! compares `size > available`, which cannot wrap on any target; see
//! [`riff`](crate::riff).
//!
//! # A size in a file is a claim, not a fact
//!
//! No allocation anywhere in this module is proportional to a number the file
//! declared. A `data` chunk announcing four gigabytes inside a two-kilobyte
//! file is rejected by the chunk walk before a byte is reserved, and the
//! whole-file decode reserves from the payload slice's real length. The
//! streaming reader buffers only chunk headers and the two chunk bodies it has
//! to read (`fmt ` and `ds64`), both of which are capped at a size the format
//! itself bounds.
//!
//! # RF64
//!
//! RIFF's 32-bit size fields stop at four gigabytes, which is about six and a
//! half hours of 16-bit stereo at 44.1 kHz. RF64 (EBU Tech 3306) keeps RIFF's
//! structure and moves the oversized numbers into a `ds64` chunk, leaving
//! `0xFFFFFFFF` behind in the fields it replaced. Both reading and writing are
//! supported here.
//!
//! # Two readers, and why they differ
//!
//! [`WavReader`] holds the whole file and can therefore accept the chunks in
//! any order, `data` before `fmt ` included. [`WavStreamDecoder`] cannot: a
//! payload that arrives before the header describing it would have to be
//! buffered in full to be decoded later, and buffering an unbounded payload is
//! the thing this module refuses to do. So the streaming reader requires
//! `fmt ` first and says so with a typed error when it does not get it. That is
//! the one behavioural difference between the two paths, and it is a property
//! of streaming rather than a shortcut.

use crate::audio::{AudioBuffer, AudioSpec};
use crate::codec::{CodecId, FourCc};
use crate::error::DecodeError;
use crate::g711::G711Law;
use crate::payload::Payload;
use crate::riff::{self, ChunkWalker};
use crate::sample::SampleFormat;
use crate::source::StreamSource;

/// `WAVE_FORMAT_PCM`.
const TAG_PCM: u16 = 1;
/// `WAVE_FORMAT_IEEE_FLOAT`.
const TAG_FLOAT: u16 = 3;
/// `WAVE_FORMAT_ALAW`.
const TAG_ALAW: u16 = 6;
/// `WAVE_FORMAT_MULAW`.
const TAG_MULAW: u16 = 7;
/// `WAVE_FORMAT_EXTENSIBLE`: the real format is in the SubFormat GUID.
const TAG_EXTENSIBLE: u16 = 0xFFFE;

/// The fourteen bytes every `KSDATAFORMAT_SUBTYPE_*` GUID ends with.
///
/// The GUIDs Microsoft assigned to the `WAVEFORMATEX` tags are the tag itself
/// in the first little-endian word followed by this fixed tail, so
/// `KSDATAFORMAT_SUBTYPE_PCM` is `00000001-0000-0010-8000-00aa00389b71`. A GUID
/// carrying this tail names a `wFormatTag`; a GUID that does not is a format
/// with no tag at all, and is reported as the GUID it was.
const SUBFORMAT_TAIL: [u8; 14] = [
    0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
];

/// The smallest `fmt ` chunk `WAVEFORMATEX` defines.
const FMT_BYTES_PLAIN: u64 = 16;

/// The `fmt ` chunk a `WAVE_FORMAT_EXTENSIBLE` file carries: the sixteen fixed
/// bytes, `cbSize`, and the twenty-two-byte extension it counts.
const FMT_BYTES_EXTENSIBLE: u64 = 40;

/// The largest `fmt ` chunk `WAVEFORMATEX` can describe.
///
/// The sixteen fixed bytes, the `cbSize` field, and the extension `cbSize`
/// counts, and `cbSize` is a `u16`, so 65,553 bytes is the format's own
/// ceiling rather than a number picked here. It matters because `fmt ` is one
/// of the two chunk bodies the streaming reader buffers instead of skipping.
const MAX_FMT_BYTES: u64 = 18 + u16::MAX as u64;

/// The `ds64` body this crate writes: sixteen bytes of sizes, eight of sample
/// count, four of table length, and no table.
const DS64_BODY_BYTES: u64 = 28;

/// How many decoded samples the streaming reader holds before it stops taking
/// bytes.
///
/// The same figure [`PcmDecoder`](crate::PcmDecoder) and
/// [`G711Decoder`](crate::G711Decoder) hold, for the same
/// reason: a caller handing over a whole file in one call gets back-pressure
/// and a bounded reader rather than a buffer the size of the file.
const READY_LIMIT: usize = 65_536;

// -- What a WAV file can carry ------------------------------------------------

/// An encoding a WAV file can carry and this crate can decode.
///
/// Deliberately not "a [`SampleFormat`] plus a law". WAV is little-endian and
/// carries six of the twelve `SampleFormat`s, so a type parameterised on
/// `SampleFormat` would be able to name `I16Be`, a file that cannot exist,
/// and every function taking one would need a failure case for it. The eight
/// variants here are exactly the eight `(wFormatTag, wBitsPerSample)` pairs
/// this crate reads and writes, so [`format_tag`](Self::format_tag),
/// [`bits_per_sample`](Self::bits_per_sample) and the writer are total.
///
/// `#[non_exhaustive]`: a consumer matching on it needs a `_` arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum WavCodec {
    /// Unsigned 8-bit PCM, offset by 128. `wFormatTag` 1, 8 bits.
    PcmU8,
    /// Signed 16-bit little-endian PCM. `wFormatTag` 1, 16 bits.
    PcmI16,
    /// Signed 24-bit little-endian PCM, packed into three bytes.
    /// `wFormatTag` 1, 24 bits.
    PcmI24,
    /// Signed 32-bit little-endian PCM. `wFormatTag` 1, 32 bits.
    PcmI32,
    /// IEEE 754 binary32, little-endian. `wFormatTag` 3, 32 bits.
    Float32,
    /// IEEE 754 binary64, little-endian. `wFormatTag` 3, 64 bits.
    Float64,
    /// ITU-T G.711 A-law. `wFormatTag` 6, 8 bits.
    ALaw,
    /// ITU-T G.711 mu-law. `wFormatTag` 7, 8 bits.
    MuLaw,
}

impl WavCodec {
    /// The encoding a `(wFormatTag, wBitsPerSample)` pair names.
    ///
    /// **On the pair, never on the tag.** Tag 1 covers four widths and tag 3
    /// covers two; a reader that dispatched on the tag alone would decode a
    /// 24-bit file at a 16-bit stride and produce noise rather than an error.
    /// The step-0 audit found decibri already doing this correctly and named it
    /// the thing most implementations get wrong.
    ///
    /// ```
    /// use decibri_decode::WavCodec;
    ///
    /// assert_eq!(WavCodec::resolve(1, 24).unwrap(), WavCodec::PcmI24);
    /// assert_eq!(WavCodec::resolve(7, 8).unwrap(), WavCodec::MuLaw);
    /// // Same tag, unsupported width: rejected, and the rejection says which.
    /// assert!(WavCodec::resolve(1, 20).is_err());
    /// ```
    ///
    /// # Errors
    ///
    /// [`DecodeError::UnsupportedSampleFormat`] for a tag this crate carries at
    /// a width it does not, and [`DecodeError::UnsupportedCodec`] for a tag it
    /// does not carry at all. The two are separate because they tell a caller
    /// different things: the first says the file is PCM and this build cannot
    /// read that width, the second says the file is not PCM.
    pub fn resolve(tag: u16, bits_per_sample: u16) -> Result<Self, DecodeError> {
        match (tag, bits_per_sample) {
            (TAG_PCM, 8) => Ok(Self::PcmU8),
            (TAG_PCM, 16) => Ok(Self::PcmI16),
            (TAG_PCM, 24) => Ok(Self::PcmI24),
            (TAG_PCM, 32) => Ok(Self::PcmI32),
            (TAG_FLOAT, 32) => Ok(Self::Float32),
            (TAG_FLOAT, 64) => Ok(Self::Float64),
            (TAG_ALAW, 8) => Ok(Self::ALaw),
            (TAG_MULAW, 8) => Ok(Self::MuLaw),
            (TAG_PCM | TAG_FLOAT | TAG_ALAW | TAG_MULAW, _) => {
                Err(DecodeError::UnsupportedSampleFormat {
                    format: CodecId::WaveFormatTag(tag),
                    bits_per_sample,
                })
            }
            _ => Err(DecodeError::UnsupportedCodec {
                codec: CodecId::WaveFormatTag(tag),
            }),
        }
    }

    /// The `wFormatTag` a file carrying this encoding declares.
    ///
    /// Total: every variant is one a WAV file can name and this crate can
    /// write.
    pub const fn format_tag(self) -> u16 {
        match self {
            Self::PcmU8 | Self::PcmI16 | Self::PcmI24 | Self::PcmI32 => TAG_PCM,
            Self::Float32 | Self::Float64 => TAG_FLOAT,
            Self::ALaw => TAG_ALAW,
            Self::MuLaw => TAG_MULAW,
        }
    }

    /// The `wBitsPerSample` a file carrying this encoding declares.
    pub const fn bits_per_sample(self) -> u16 {
        match self {
            Self::PcmU8 | Self::ALaw | Self::MuLaw => 8,
            Self::PcmI16 => 16,
            Self::PcmI24 => 24,
            Self::PcmI32 | Self::Float32 => 32,
            Self::Float64 => 64,
        }
    }

    /// How many bytes one sample occupies in the payload.
    pub const fn bytes_per_sample(self) -> usize {
        match self {
            Self::PcmU8 | Self::ALaw | Self::MuLaw => 1,
            Self::PcmI16 => 2,
            Self::PcmI24 => 3,
            Self::PcmI32 | Self::Float32 => 4,
            Self::Float64 => 8,
        }
    }

    /// The linear PCM format this encoding is, or `None` for the companded
    /// ones.
    pub const fn sample_format(self) -> Option<SampleFormat> {
        match self {
            Self::PcmU8 => Some(SampleFormat::U8),
            Self::PcmI16 => Some(SampleFormat::I16Le),
            Self::PcmI24 => Some(SampleFormat::I24Le),
            Self::PcmI32 => Some(SampleFormat::I32Le),
            Self::Float32 => Some(SampleFormat::F32Le),
            Self::Float64 => Some(SampleFormat::F64Le),
            Self::ALaw | Self::MuLaw => None,
        }
    }

    /// The G.711 companding law this encoding is, or `None` for the linear
    /// ones.
    pub const fn law(self) -> Option<G711Law> {
        match self {
            Self::ALaw => Some(G711Law::ALaw),
            Self::MuLaw => Some(G711Law::MuLaw),
            _ => None,
        }
    }

    /// Appends every whole sample in `bytes` to `output`, and returns how many
    /// it appended.
    ///
    /// Straight through to [`SampleFormat::decode`] or [`G711Law::decode`]:
    /// there is no WAV-specific conversion anywhere in this crate, so a file
    /// decoded through the container and a headerless stream decoded through
    /// [`PcmDecoder`](crate::PcmDecoder) give the same samples by construction rather than by
    /// agreement.
    pub fn decode(self, bytes: &[u8], output: &mut Vec<f32>) -> usize {
        match (self.sample_format(), self.law()) {
            (Some(format), _) => format.decode(bytes, output),
            (_, Some(law)) => law.decode(bytes, output),
            // Unreachable: every variant is one or the other, and the match in
            // `sample_format` and `law` is exhaustive over the enum.
            _ => 0,
        }
    }

    /// Appends `samples` to `output` in this encoding, and returns how many
    /// bytes it appended.
    pub fn encode(self, samples: &[f32], output: &mut Vec<u8>) -> usize {
        match (self.sample_format(), self.law()) {
            (Some(format), _) => format.encode(samples, output),
            (_, Some(law)) => law.encode(samples, output),
            _ => 0,
        }
    }
}

/// A resolved `fmt ` chunk: what the file declared, and what it resolved to.
///
/// The declared fields are kept beside the resolution rather than discarded,
/// because a caller reporting on a file wants what the file said and a caller
/// decoding it wants what that means.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct WavFormat {
    /// The rate and layout the payload decodes to.
    pub spec: AudioSpec,
    /// What the payload is encoded as, from the `(tag, bits)` pair.
    pub codec: WavCodec,
    /// `wFormatTag` exactly as the file declared it: `0xFFFE` for an
    /// extensible header, whatever its SubFormat GUID resolved to.
    pub format_tag: u16,
    /// `wBitsPerSample` as declared.
    pub bits_per_sample: u16,
    /// `nBlockAlign` as declared, in bytes per frame.
    ///
    /// **Advisory.** The frame size this crate decodes at comes from the
    /// resolved codec and the channel count, never from this field. Writers
    /// that get it wrong are common, and a file whose audio is perfectly
    /// decodable is not worth rejecting over a redundant field; a file whose
    /// `nBlockAlign` disagrees with `nChannels * wBitsPerSample / 8` still
    /// decodes, and the disagreement is visible here to anyone who wants it.
    pub block_align: u16,
    /// `nAvgBytesPerSec` as declared, in bytes per second. Advisory, as
    /// [`block_align`](Self::block_align) is.
    pub byte_rate: u32,
}

impl WavFormat {
    /// How many bytes one whole frame occupies in the payload.
    ///
    /// At least one: the channel count is rejected at zero, so this never
    /// divides by nothing.
    pub const fn frame_bytes(&self) -> usize {
        self.spec.channels as usize * self.codec.bytes_per_sample()
    }

    /// `true` when the file declared `WAVE_FORMAT_EXTENSIBLE`.
    pub const fn is_extensible(&self) -> bool {
        self.format_tag == TAG_EXTENSIBLE
    }
}

// -- Reading the `fmt ` chunk -------------------------------------------------

/// Rejects a `fmt ` chunk whose declared size the format cannot mean.
///
/// Called on both paths with the same number so the whole-file and streaming
/// readers agree: on the whole-file path the body is already bounded by the
/// input, and on the streaming path this is what bounds the buffer.
fn check_fmt_size(size: u64, offset: u64) -> Result<(), DecodeError> {
    if size < FMT_BYTES_PLAIN {
        return Err(DecodeError::Malformed {
            expected: "a fmt chunk of at least 16 bytes",
            offset,
        });
    }
    if size > MAX_FMT_BYTES {
        return Err(DecodeError::Malformed {
            expected: "a fmt chunk of at most 65553 bytes, which is all cbSize can count",
            offset,
        });
    }
    Ok(())
}

/// A GUID in the text form Microsoft writes it in, for a rejection to name.
///
/// The first three groups are little-endian and the last two are byte order as
/// stored, which is why the rendering is written out rather than done with a
/// hex formatter over the sixteen bytes.
fn format_guid(guid: &[u8; 16]) -> String {
    const HEX: [u8; 16] = *b"0123456789abcdef";
    let data1 = u32::from_le_bytes([guid[0], guid[1], guid[2], guid[3]]);
    let data2 = u16::from_le_bytes([guid[4], guid[5]]);
    let data3 = u16::from_le_bytes([guid[6], guid[7]]);
    let mut text = format!("{data1:08x}-{data2:04x}-{data3:04x}-");
    for (index, byte) in guid[8..].iter().enumerate() {
        if index == 2 {
            text.push('-');
        }
        text.push(HEX[usize::from(byte >> 4)] as char);
        text.push(HEX[usize::from(byte & 0x0F)] as char);
    }
    text
}

/// Reads a `fmt ` chunk body that has already been checked with
/// [`check_fmt_size`].
///
/// `offset` is where the chunk *header* sits in the input, so the offsets in
/// any [`DecodeError::Malformed`] this returns are absolute.
fn parse_fmt(body: &[u8], offset: u64) -> Result<WavFormat, DecodeError> {
    let body_at = offset + riff::CHUNK_HEADER_BYTES;
    let (
        Some(declared_tag),
        Some(channels),
        Some(sample_rate),
        Some(byte_rate),
        Some(block_align),
        Some(bits_per_sample),
    ) = (
        riff::u16_at(body, 0),
        riff::u16_at(body, 2),
        riff::u32_at(body, 4),
        riff::u32_at(body, 8),
        riff::u16_at(body, 12),
        riff::u16_at(body, 14),
    )
    else {
        return Err(DecodeError::Malformed {
            expected: "a fmt chunk of at least 16 bytes",
            offset,
        });
    };

    if channels == 0 {
        return Err(DecodeError::UnsupportedChannelLayout { channels });
    }
    if sample_rate == 0 {
        // No `UnsupportedSampleRate` variant exists and `DecodeError` is
        // closed, so this is a structural rejection: a rate of zero is not a
        // rate this file could have meant.
        return Err(DecodeError::Malformed {
            expected: "a non-zero sample rate",
            offset: body_at + 4,
        });
    }

    // WAVE_FORMAT_EXTENSIBLE names the real encoding in a GUID, and the tag
    // only says to go and look.
    let tag = if declared_tag == TAG_EXTENSIBLE {
        resolve_subformat(body, body_at)?
    } else {
        declared_tag
    };

    Ok(WavFormat {
        spec: AudioSpec::new(sample_rate, channels),
        codec: WavCodec::resolve(tag, bits_per_sample)?,
        format_tag: declared_tag,
        bits_per_sample,
        block_align,
        byte_rate,
    })
}

/// The `wFormatTag` a `WAVE_FORMAT_EXTENSIBLE` header's SubFormat GUID names.
///
/// The extension has to be there and has to be big enough: `cbSize` under 22,
/// or a body without room for the twenty-two bytes it counts, is a header that
/// says "the format is in the extension" and then does not carry one.
fn resolve_subformat(body: &[u8], body_at: u64) -> Result<u16, DecodeError> {
    const MIN_EXTENSION: u16 = 22;
    let inconsistent = DecodeError::Malformed {
        expected: "a 22-byte WAVE_FORMAT_EXTENSIBLE extension after cbSize",
        offset: body_at + 16,
    };
    let Some(cb_size) = riff::u16_at(body, 16) else {
        return Err(inconsistent);
    };
    if cb_size < MIN_EXTENSION {
        return Err(inconsistent);
    }
    let Some(guid) = body.get(24..40) else {
        return Err(inconsistent);
    };
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(guid);
    if bytes[2..] == SUBFORMAT_TAIL {
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    } else {
        // A GUID outside the `WAVEFORMATEX` family names a codec that has no
        // tag at all, so the rejection carries the GUID rather than inventing
        // a number for it.
        Err(DecodeError::UnsupportedCodec {
            codec: CodecId::Name(format_guid(&bytes)),
        })
    }
}

/// Rejects a `data` chunk whose length is not a whole number of frames.
///
/// The first of the two decibri defects this reader exists not to repeat.
/// `chunks_exact` drops the remainder silently, so a payload of two and a half
/// stereo frames decodes as two frames and the caller is never told.
fn check_whole_frames(data_len: u64, frame_bytes: usize) -> Result<(), DecodeError> {
    let remainder = data_len % frame_bytes as u64;
    if remainder != 0 {
        return Err(DecodeError::Truncated {
            expected: frame_bytes as u64,
            available: remainder,
        });
    }
    Ok(())
}

// -- The whole-file reader ----------------------------------------------------

/// A WAV file held whole in memory.
///
/// [`new`](Self::new) does all the parsing and all the rejecting; a reader that
/// exists is a file that has been fully validated, which is why
/// [`decode_to_end`](Self::decode_to_end) returns an [`AudioBuffer`] rather
/// than a `Result`. That is a property of the two payload encodings this crate
/// carries (linear PCM and G.711 both decode every byte sequence to a sample)
/// and is stated here rather than assumed: a codec that can fail part-way
/// through its payload does not go behind this API.
///
/// # Example
///
/// ```
/// use decibri_decode::{AudioSpec, WavCodec, WavReader, WavWriter};
///
/// let bytes = WavWriter::new(AudioSpec::new(8_000, 2), WavCodec::MuLaw)
///     .to_bytes(&[0.0, 0.5, -0.5, 0.25])?;
///
/// let reader = WavReader::new(&bytes)?;
/// assert_eq!(reader.format().codec, WavCodec::MuLaw);
/// assert_eq!(reader.spec(), AudioSpec::new(8_000, 2));
/// assert_eq!(reader.frames(), 2);
/// assert_eq!(reader.decode_to_end().samples().len(), 4);
/// # Ok::<(), decibri_decode::DecodeError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WavReader<'a> {
    format: WavFormat,
    data: &'a [u8],
    rf64: bool,
}

impl<'a> WavReader<'a> {
    /// Parses `bytes` as a RIFF/WAVE or RF64 file.
    ///
    /// Chunks may arrive in any order and unknown chunks anywhere are skipped,
    /// including between `fmt ` and `data`. The walk stops once both have been
    /// found, so trailing chunks are not examined and a file with junk after
    /// its audio still reads.
    ///
    /// **"Any order" includes `data` before `fmt `, which
    /// [`WavStreamDecoder`] cannot accept.** A file that opens here can
    /// therefore fail when streamed. That difference is forced, not chosen: a
    /// streaming reader meeting the payload before the header describing it
    /// would have to buffer the whole payload to decode it later, and an
    /// unbounded buffer is the thing the streaming reader exists to avoid.
    /// This reader holds the whole file already, so it has nothing to buffer
    /// and accepts either order.
    ///
    /// # Errors
    ///
    /// - [`DecodeError::Truncated`] for an input under twelve bytes, a chunk
    ///   declaring more than the input holds, and a `data` chunk whose length
    ///   is not a whole number of frames.
    /// - [`DecodeError::UnsupportedContainer`] for a magic that is neither
    ///   `RIFF` nor `RF64`, and for a form type other than `WAVE`.
    /// - [`DecodeError::Malformed`] for a missing or structurally wrong
    ///   `fmt `, a missing `data`, a zero sample rate, and an inconsistent
    ///   `WAVE_FORMAT_EXTENSIBLE` extension.
    /// - [`DecodeError::UnsupportedChannelLayout`] for a zero channel count.
    /// - [`DecodeError::UnsupportedCodec`] and
    ///   [`DecodeError::UnsupportedSampleFormat`] from
    ///   [`WavCodec::resolve`].
    pub fn new(bytes: &'a [u8]) -> Result<Self, DecodeError> {
        let header = riff::read_riff_header(bytes)?;
        if header.form != riff::WAVE {
            return Err(DecodeError::UnsupportedContainer { tag: header.form });
        }

        let mut walker = ChunkWalker::new(bytes, riff::ByteOrder::Little);
        let rf64 = header.magic == riff::RF64;
        if rf64 {
            // EBU Tech 3306 puts `ds64` immediately after the form type, and it
            // has to be there: every oversized field in the file is a sentinel
            // pointing at it.
            let Some(first) = walker.next().transpose()? else {
                return Err(DecodeError::Truncated {
                    expected: riff::HEADER_BYTES + riff::CHUNK_HEADER_BYTES,
                    available: bytes.len() as u64,
                });
            };
            if first.id != riff::DS64 {
                return Err(DecodeError::Malformed {
                    expected: "a ds64 chunk immediately after the RF64 form type",
                    offset: first.offset,
                });
            }
            if first.body.len() as u64 > riff::MAX_DS64_BYTES {
                return Err(DecodeError::Malformed {
                    expected: "a ds64 chunk of at most 65536 bytes",
                    offset: first.offset,
                });
            }
            walker.set_overrides(riff::parse_ds64(first.body, first.offset)?);
        }

        let mut format: Option<WavFormat> = None;
        let mut data: Option<&[u8]> = None;
        while format.is_none() || data.is_none() {
            let Some(chunk) = walker.next().transpose()? else {
                break;
            };
            if chunk.id == riff::FMT && format.is_none() {
                check_fmt_size(chunk.body.len() as u64, chunk.offset)?;
                format = Some(parse_fmt(chunk.body, chunk.offset)?);
            } else if chunk.id == riff::DATA && data.is_none() {
                data = Some(chunk.body);
            }
        }

        let Some(format) = format else {
            return Err(DecodeError::Malformed {
                expected: "a fmt chunk",
                offset: riff::HEADER_BYTES,
            });
        };
        let Some(data) = data else {
            return Err(DecodeError::Malformed {
                expected: "a data chunk",
                offset: riff::HEADER_BYTES,
            });
        };
        check_whole_frames(data.len() as u64, format.frame_bytes())?;

        Ok(Self { format, data, rf64 })
    }

    /// What the `fmt ` chunk declared and resolved to.
    pub const fn format(&self) -> &WavFormat {
        &self.format
    }

    /// The rate and layout the payload decodes to.
    pub const fn spec(&self) -> AudioSpec {
        self.format.spec
    }

    /// The `data` payload, undecoded.
    pub const fn data(&self) -> &'a [u8] {
        self.data
    }

    /// How many whole frames the payload holds.
    ///
    /// Exact rather than declared: the payload is a real subslice of the input
    /// and its length was checked against the frame size at parse time, so this
    /// is a count of frames that are present, not a count the file claimed.
    pub const fn frames(&self) -> u64 {
        self.data.len() as u64 / self.format.frame_bytes() as u64
    }

    /// `true` when the file was RF64 rather than plain RIFF.
    pub const fn is_rf64(&self) -> bool {
        self.rf64
    }

    /// Appends every interleaved sample of the payload to `output`, and
    /// returns how many it appended.
    ///
    /// Interleaved samples, frames times channels, not the frames
    /// [`frames`](Self::frames) counts.
    pub fn decode(&self, output: &mut Vec<f32>) -> usize {
        self.format.codec.decode(self.data, output)
    }

    /// Decodes the whole payload, bound to the spec that describes it.
    ///
    /// The reservation is from the payload's real length, never from a size the
    /// file declared.
    ///
    /// The channel count of the returned buffer is never zero: a file
    /// declaring none is refused by [`new`](Self::new) with
    /// [`DecodeError::UnsupportedChannelLayout`].
    pub fn decode_to_end(&self) -> AudioBuffer {
        let mut samples =
            Vec::with_capacity(self.data.len() / self.format.codec.bytes_per_sample());
        self.decode(&mut samples);
        AudioBuffer::from_samples(self.format.spec, samples)
    }
}

// -- The streaming reader -----------------------------------------------------

/// Where the streaming reader is in the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Waiting for the twelve-byte RIFF header.
    Riff,
    /// Waiting for an eight-byte chunk header.
    ChunkHeader,
    /// Buffering the body of a chunk that has to be read: `fmt ` or `ds64`.
    Body {
        id: FourCc,
        size: usize,
        offset: u64,
    },
    /// Discarding the body of a chunk that does not have to be read. Nothing is
    /// buffered here, which is what keeps an enormous `LIST` chunk free.
    Skip { size: u64, left: u64 },
    /// Discarding the one pad byte after an odd-length chunk.
    Pad,
    /// Streaming the `data` payload.
    Data,
    /// Past the end of `data`. Everything after it is discarded.
    Done,
}

/// Reads a WAV file that arrives in pieces.
///
/// The [`StreamSource`] half of this module. Bytes are pushed in whatever sizes
/// they turn up in, samples are pulled out, and nothing is buffered in
/// proportion to a size the file declared: chunk headers and the `fmt ` and
/// `ds64` bodies are buffered, every other chunk body is discarded as it flows
/// past, and the payload is decoded straight into a bounded ready buffer.
///
/// # `fmt ` has to come first
///
/// [`WavReader`] accepts `data` before `fmt `; this does not, and answers
/// [`DecodeError::Malformed`] when it meets one. Decoding a payload that
/// arrives before the header describing it would mean holding all of it, and an
/// unbounded buffer is exactly what this type is arranged to avoid. Every other
/// chunk order, and unknown chunks in any position, are the same on both paths.
///
/// # Example
///
/// ```
/// use decibri_decode::{AudioSpec, StreamSource, WavCodec, WavStreamDecoder, WavWriter};
///
/// let file = WavWriter::new(AudioSpec::mono(16_000), WavCodec::PcmI16)
///     .to_bytes(&[0.0, 0.5, -0.5])?;
///
/// let mut stream = WavStreamDecoder::new();
/// let mut samples = Vec::new();
/// for piece in file.chunks(7) {
///     let mut offset = 0;
///     while offset < piece.len() {
///         offset += stream.push(&piece[offset..])?;
///         while stream.pull(&mut samples, usize::MAX)? > 0 {}
///     }
/// }
/// stream.finish(&mut samples)?;
///
/// assert_eq!(stream.spec(), Some(AudioSpec::mono(16_000)));
/// assert_eq!(samples, [0.0, 0.5, -0.5]);
/// # Ok::<(), decibri_decode::DecodeError>(())
/// ```
#[derive(Debug)]
pub struct WavStreamDecoder {
    state: State,
    /// Bytes of a header or of a `fmt `/`ds64` body that has not fully
    /// arrived. Bounded by [`MAX_FMT_BYTES`] and
    /// [`riff::MAX_DS64_BYTES`](crate::riff).
    pending: Vec<u8>,
    /// Byte offset of the next byte to arrive, for error reporting.
    offset: u64,
    /// Sizes a `ds64` chunk overrides.
    overrides: Vec<(FourCc, u64)>,
    /// Set while the RF64 header's mandatory `ds64` chunk is still expected.
    expect_ds64: bool,
    format: Option<WavFormat>,
    payload: Option<Payload>,
    /// Decoded samples not yet pulled, and how far into them the caller is.
    ready: Vec<f32>,
    ready_read: usize,
    /// The `data` chunk's size, and how much of it is still to arrive.
    data_size: u64,
    data_left: u64,
    finished: bool,
}

impl Default for WavStreamDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl WavStreamDecoder {
    /// A reader waiting for the first byte of a file.
    pub fn new() -> Self {
        Self {
            state: State::Riff,
            pending: Vec::new(),
            offset: 0,
            overrides: Vec::new(),
            expect_ds64: false,
            format: None,
            payload: None,
            ready: Vec::new(),
            ready_read: 0,
            data_size: 0,
            data_left: 0,
            finished: false,
        }
    }

    /// What the `fmt ` chunk declared, once it has arrived.
    pub const fn format(&self) -> Option<WavFormat> {
        self.format
    }

    /// How many interleaved samples are decoded and waiting to be pulled:
    /// frames times channels, which is the length a caller's output vector
    /// grows by when they are.
    ///
    /// Interleaved samples, not the frames [`pull`](StreamSource::pull)
    /// counts; the two figures differ by the channel count.
    pub fn ready_samples(&self) -> usize {
        self.ready.len() - self.ready_read
    }

    /// Buffers `bytes` into `pending` until it holds `need` of them, and says
    /// whether it now does.
    fn accumulate(&mut self, bytes: &[u8], need: usize, taken: &mut usize) -> bool {
        let want = need.saturating_sub(self.pending.len()).min(bytes.len());
        self.pending.extend_from_slice(&bytes[..want]);
        *taken += want;
        self.offset += want as u64;
        self.pending.len() >= need
    }

    /// Reads the twelve-byte RIFF header out of `pending`.
    fn start_file(&mut self) -> Result<(), DecodeError> {
        let header = riff::read_riff_header(&self.pending)?;
        if header.form != riff::WAVE {
            return Err(DecodeError::UnsupportedContainer { tag: header.form });
        }
        self.expect_ds64 = header.magic == riff::RF64;
        self.pending.clear();
        self.state = State::ChunkHeader;
        Ok(())
    }

    /// Reads an eight-byte chunk header out of `pending` and decides what to do
    /// with the body.
    fn start_chunk(&mut self) -> Result<(), DecodeError> {
        let offset = self.offset - riff::CHUNK_HEADER_BYTES;
        let (Some(id), Some(declared)) = (
            riff::four_cc_at(&self.pending, 0),
            riff::u32_at(&self.pending, 4),
        ) else {
            return Err(DecodeError::Truncated {
                expected: riff::CHUNK_HEADER_BYTES,
                available: self.pending.len() as u64,
            });
        };
        self.pending.clear();
        let size = riff::resolve_size(&self.overrides, id, declared);

        if self.expect_ds64 && id != riff::DS64 {
            return Err(DecodeError::Malformed {
                expected: "a ds64 chunk immediately after the RF64 form type",
                offset,
            });
        }

        if id == riff::DS64 && self.expect_ds64 {
            if size > riff::MAX_DS64_BYTES {
                return Err(DecodeError::Malformed {
                    expected: "a ds64 chunk of at most 65536 bytes",
                    offset,
                });
            }
            self.state = State::Body {
                id,
                size: size as usize,
                offset,
            };
            return Ok(());
        }

        if id == riff::FMT && self.format.is_none() {
            check_fmt_size(size, offset)?;
            self.state = State::Body {
                id,
                size: size as usize,
                offset,
            };
            return Ok(());
        }

        if id == riff::DATA {
            let Some(format) = self.format else {
                // The one thing the streaming path cannot do that the
                // whole-file path can.
                return Err(DecodeError::Malformed {
                    expected: "a fmt chunk before the data chunk",
                    offset,
                });
            };
            check_whole_frames(size, format.frame_bytes())?;
            self.payload = Some(Payload::from_parts(
                format.codec.sample_format(),
                format.codec.law(),
                format.spec,
            ));
            self.data_size = size;
            self.data_left = size;
            self.state = if size == 0 { State::Done } else { State::Data };
            return Ok(());
        }

        self.state = if size == 0 {
            State::ChunkHeader
        } else {
            State::Skip { size, left: size }
        };
        Ok(())
    }

    /// Reads a buffered `fmt ` or `ds64` body out of `pending`.
    fn finish_body(&mut self, id: FourCc, size: usize, offset: u64) -> Result<(), DecodeError> {
        if id == riff::FMT {
            self.format = Some(parse_fmt(&self.pending, offset)?);
        } else {
            self.overrides = riff::parse_ds64(&self.pending, offset)?;
            self.expect_ds64 = false;
        }
        self.pending.clear();
        self.state = if riff::pad_len(size as u64) == 1 {
            State::Pad
        } else {
            State::ChunkHeader
        };
        Ok(())
    }

    /// Moves decoded samples out of the payload decoder and keeps the ready
    /// buffer from growing without bound.
    fn drain_payload(&mut self) -> Result<(), DecodeError> {
        if let Some(payload) = self.payload.as_mut() {
            payload.decode(&mut self.ready)?;
        }
        Ok(())
    }
}

impl StreamSource for WavStreamDecoder {
    fn push(&mut self, bytes: &[u8]) -> Result<usize, DecodeError> {
        if self.finished {
            return Ok(0);
        }
        let mut taken = 0;
        let result = self.push_inner(bytes, &mut taken);
        if result.is_err() {
            // A stream that has failed structurally is over. Leaving it live
            // would mean a caller who keeps pushing gets a second, different
            // answer from the same file.
            self.finished = true;
        }
        result.map(|()| taken)
    }

    fn pull(&mut self, output: &mut Vec<f32>, max_frames: usize) -> Result<usize, DecodeError> {
        let Some(format) = self.format else {
            return Ok(0);
        };
        let channels = usize::from(format.spec.channels);
        let available = self.ready.len() - self.ready_read;
        let frames = (available / channels).min(max_frames);
        let count = frames * channels;
        output.extend_from_slice(&self.ready[self.ready_read..self.ready_read + count]);
        self.ready_read += count;
        // Amortised compaction: the consumed prefix is dropped once it is at
        // least half the buffer, so a caller pulling one frame at a time does
        // not leave the whole file's worth of consumed samples behind it.
        if self.ready_read == self.ready.len() {
            self.ready.clear();
            self.ready_read = 0;
        } else if self.ready_read >= self.ready.len() / 2 {
            self.ready.drain(..self.ready_read);
            self.ready_read = 0;
        }
        Ok(frames)
    }

    fn spec(&self) -> Option<AudioSpec> {
        self.format.map(|format| format.spec)
    }

    fn buffered_bytes(&self) -> usize {
        self.pending.len()
            + self
                .payload
                .as_ref()
                .map_or(0, |payload| payload.buffered_bytes())
    }

    fn finish(&mut self, output: &mut Vec<f32>) -> Result<usize, DecodeError> {
        if self.finished {
            return Ok(0);
        }
        self.finished = true;

        // Structural completeness first: an item half-arrived is Truncated, and
        // a file that ended on a chunk boundary without ever naming its audio
        // is Malformed.
        match self.state {
            State::Done => {}
            State::Riff => {
                return Err(DecodeError::Truncated {
                    expected: riff::HEADER_BYTES,
                    available: self.pending.len() as u64,
                })
            }
            // A stream that ended on a chunk boundary never named its audio.
            // The missing pad byte after a final odd-length chunk lands here
            // too, and for the same reason: the chunk was complete, the file
            // was not.
            State::Pad => {
                return Err(DecodeError::Malformed {
                    expected: "a data chunk",
                    offset: self.offset,
                })
            }
            State::ChunkHeader if self.pending.is_empty() => {
                return Err(DecodeError::Malformed {
                    expected: "a data chunk",
                    offset: self.offset,
                })
            }
            State::ChunkHeader => {
                return Err(DecodeError::Truncated {
                    expected: riff::CHUNK_HEADER_BYTES,
                    available: self.pending.len() as u64,
                })
            }
            State::Body { size, .. } => {
                return Err(DecodeError::Truncated {
                    expected: size as u64,
                    available: self.pending.len() as u64,
                })
            }
            State::Skip { size, left } => {
                return Err(DecodeError::Truncated {
                    expected: size,
                    available: size - left,
                })
            }
            State::Data => {
                return Err(DecodeError::Truncated {
                    expected: self.data_size,
                    available: self.data_size - self.data_left,
                })
            }
        }

        if let Some(payload) = self.payload.as_mut() {
            payload.flush(&mut self.ready)?;
        }
        let Some(format) = self.format else {
            return Ok(0);
        };
        let channels = usize::from(format.spec.channels);
        let frames = (self.ready.len() - self.ready_read) / channels;
        output.extend_from_slice(&self.ready[self.ready_read..self.ready_read + frames * channels]);
        self.ready.clear();
        self.ready_read = 0;
        Ok(frames)
    }

    fn reset(&mut self) {
        *self = Self::new();
    }
}

impl WavStreamDecoder {
    /// The body of [`push`](StreamSource::push), split out so a failure can set
    /// the finished flag in one place.
    fn push_inner(&mut self, bytes: &[u8], taken: &mut usize) -> Result<(), DecodeError> {
        while *taken < bytes.len() {
            let rest = &bytes[*taken..];
            match self.state {
                State::Riff => {
                    if !self.accumulate(rest, riff::HEADER_BYTES as usize, taken) {
                        break;
                    }
                    self.start_file()?;
                }
                State::ChunkHeader => {
                    if !self.accumulate(rest, riff::CHUNK_HEADER_BYTES as usize, taken) {
                        break;
                    }
                    self.start_chunk()?;
                }
                State::Body { id, size, offset } => {
                    if !self.accumulate(rest, size, taken) {
                        break;
                    }
                    self.finish_body(id, size, offset)?;
                }
                State::Skip { size, left } => {
                    let step = left.min(rest.len() as u64);
                    *taken += step as usize;
                    self.offset += step;
                    let left = left - step;
                    self.state = if left == 0 {
                        if riff::pad_len(size) == 1 {
                            State::Pad
                        } else {
                            State::ChunkHeader
                        }
                    } else {
                        State::Skip { size, left }
                    };
                }
                State::Pad => {
                    *taken += 1;
                    self.offset += 1;
                    self.state = State::ChunkHeader;
                }
                State::Data => {
                    if self.ready.len() - self.ready_read >= READY_LIMIT {
                        // Back-pressure: the caller has to pull before the
                        // reader will take more.
                        break;
                    }
                    let limit = self.data_left.min(rest.len() as u64) as usize;
                    let fed = match self.payload.as_mut() {
                        Some(payload) => payload.feed(&rest[..limit])?,
                        None => 0,
                    };
                    self.drain_payload()?;
                    *taken += fed;
                    self.offset += fed as u64;
                    self.data_left -= fed as u64;
                    if self.data_left == 0 {
                        self.state = State::Done;
                    } else if fed == 0 {
                        break;
                    }
                }
                State::Done => {
                    // Chunks after `data` are not this reader's business, and
                    // discarding them is what makes a file with metadata at the
                    // end readable rather than an error.
                    self.offset += (bytes.len() - *taken) as u64;
                    *taken = bytes.len();
                }
            }
        }
        Ok(())
    }
}

// -- The writer ---------------------------------------------------------------

/// How a written file names its format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum WavHeaderStyle {
    /// The plain `wFormatTag`, 1, 3, 6 or 7, in a sixteen-byte `fmt ` chunk.
    ///
    /// What every reader understands, and the default for that reason.
    #[default]
    Plain,
    /// `wFormatTag` 0xFFFE with a forty-byte `fmt ` chunk naming the format in
    /// a SubFormat GUID.
    ///
    /// What Windows writes for anything above two channels or sixteen bits.
    Extensible,
}

/// Which RIFF flavour a written file uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum RiffFlavour {
    /// Plain `RIFF`, upgrading to `RF64` only when the file would not fit in
    /// RIFF's 32-bit size fields.
    ///
    /// There is no third option where the writer refuses: a caller with five
    /// gigabytes of audio wants a file, and `RF64` is what a file of that size
    /// is. Keeping the upgrade automatic is what lets
    /// [`WavWriter::write`] fail only on things the caller can fix.
    #[default]
    Automatic,
    /// `RF64` with a `ds64` chunk whatever the size.
    Rf64,
}

/// Whether a file of this shape needs RF64's 64-bit sizes.
///
/// Split out as a function of two numbers so the threshold can be tested
/// without materialising four gigabytes to test it with.
fn needs_rf64(fmt_bytes: u64, data_bytes: u64) -> bool {
    riff_size(fmt_bytes, data_bytes, false).is_none_or(|size| size > u64::from(u32::MAX))
}

/// The value of the RIFF header's size field: everything after the first eight
/// bytes. `None` when that does not fit in 64 bits, which no caller can reach
/// but the arithmetic should not pretend about.
fn riff_size(fmt_bytes: u64, data_bytes: u64, ds64: bool) -> Option<u64> {
    let mut size: u64 = 4; // the form type
    if ds64 {
        size = size.checked_add(riff::CHUNK_HEADER_BYTES + DS64_BODY_BYTES)?;
    }
    size = size
        .checked_add(riff::CHUNK_HEADER_BYTES)?
        .checked_add(fmt_bytes)?;
    size = size
        .checked_add(riff::CHUNK_HEADER_BYTES)?
        .checked_add(data_bytes)?
        // The pad byte belongs to the enclosing form even though it is outside
        // the `data` chunk's own declared size.
        .checked_add(data_bytes & 1)?;
    Some(size)
}

/// Writes RIFF/WAVE and RF64 files.
///
/// The writer exists in the same step as the reader because round-trip identity
/// is the cheapest strong gate available for a container, and it needs both
/// halves. It writes every encoding the reader accepts, and [`WavCodec`] has
/// no variant that is readable and not writable, with the pad
/// byte on an odd-length `data` chunk and a correct `RIFF` size field.
///
/// # Samples outside full scale, and samples that are not finite
///
/// **The integer encodings clamp and the float encodings do not.** A sample
/// outside `-1.0..=1.0` written to an integer encoding becomes the extreme
/// value rather than wrapping, an infinity becomes the same extreme, and a
/// NaN becomes silence. A sample written to [`WavCodec::Float32`] or
/// [`WavCodec::Float64`] is written through as it is, overshoot, infinities
/// and NaN included. A count of clipped samples is therefore a number about
/// an integer encoding and not about a float one.
///
/// Read back through this crate, a NaN written to [`WavCodec::Float32`]
/// returns the same bit pattern, and a NaN written to [`WavCodec::Float64`]
/// returns silence, because narrowing a NaN from `f64` to `f32` normalises
/// it. Both infinities return unchanged at either width.
///
/// # Example
///
/// ```
/// use decibri_decode::{AudioSpec, WavCodec, WavHeaderStyle, WavReader, WavWriter};
///
/// let file = WavWriter::new(AudioSpec::new(44_100, 2), WavCodec::PcmI24)
///     .with_header_style(WavHeaderStyle::Extensible)
///     .to_bytes(&[0.5, -0.5, 0.25, -0.25])?;
///
/// let reader = WavReader::new(&file)?;
/// assert!(reader.format().is_extensible());
/// assert_eq!(reader.format().codec, WavCodec::PcmI24);
/// assert_eq!(reader.decode_to_end().samples(), [0.5, -0.5, 0.25, -0.25]);
/// # Ok::<(), decibri_decode::DecodeError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WavWriter {
    spec: AudioSpec,
    codec: WavCodec,
    header: WavHeaderStyle,
    flavour: RiffFlavour,
}

impl WavWriter {
    /// A writer for `codec` audio at `spec`, in a plain `RIFF` header.
    pub const fn new(spec: AudioSpec, codec: WavCodec) -> Self {
        Self {
            spec,
            codec,
            header: WavHeaderStyle::Plain,
            flavour: RiffFlavour::Automatic,
        }
    }

    /// Sets whether the `fmt ` chunk is plain or `WAVE_FORMAT_EXTENSIBLE`.
    pub const fn with_header_style(mut self, header: WavHeaderStyle) -> Self {
        self.header = header;
        self
    }

    /// Sets whether the file is `RIFF` or `RF64`.
    pub const fn with_flavour(mut self, flavour: RiffFlavour) -> Self {
        self.flavour = flavour;
        self
    }

    /// The rate and layout written into the header.
    pub const fn spec(&self) -> AudioSpec {
        self.spec
    }

    /// The encoding the payload is written in.
    pub const fn codec(&self) -> WavCodec {
        self.codec
    }

    /// Appends a complete file to `output`, and returns how many bytes it
    /// appended.
    ///
    /// `output` is appended to, never cleared, the same convention as
    /// everything else in this crate that writes into a caller's buffer.
    ///
    /// # Errors
    ///
    /// - [`DecodeError::UnsupportedChannelLayout`] for no channels, and for a
    ///   layout whose frame does not fit `nBlockAlign`. That second one is a
    ///   limit of `WAVEFORMATEX` rather than of this crate: the field is a
    ///   `u16`, so 8,191 channels of `f64` is the widest frame a WAV file can
    ///   describe. The reader accepts such a file if one turns up, because
    ///   `nBlockAlign` is advisory there; the writer will not produce one,
    ///   because it would have to write a field it knows is wrong.
    /// - [`DecodeError::Malformed`] for a zero sample rate.
    /// - [`DecodeError::Truncated`] when `samples` does not divide into whole
    ///   frames. Writing a partial trailing frame would produce a file this
    ///   crate's own reader rejects, and dropping it would be the silent
    ///   truncation the reader exists to catch.
    pub fn write(&self, samples: &[f32], output: &mut Vec<u8>) -> Result<usize, DecodeError> {
        let channels = usize::from(self.spec.channels);
        if channels == 0 {
            return Err(DecodeError::UnsupportedChannelLayout {
                channels: self.spec.channels,
            });
        }
        let frame_bytes = channels * self.codec.bytes_per_sample();
        let Ok(block_align) = u16::try_from(frame_bytes) else {
            return Err(DecodeError::UnsupportedChannelLayout {
                channels: self.spec.channels,
            });
        };
        if self.spec.sample_rate == 0 {
            return Err(DecodeError::Malformed {
                expected: "a non-zero sample rate",
                offset: riff::HEADER_BYTES + riff::CHUNK_HEADER_BYTES + 4,
            });
        }
        let leftover = samples.len() % channels;
        if leftover != 0 {
            return Err(DecodeError::Truncated {
                expected: frame_bytes as u64,
                available: (leftover * self.codec.bytes_per_sample()) as u64,
            });
        }

        let fmt_bytes = match self.header {
            WavHeaderStyle::Plain => FMT_BYTES_PLAIN,
            WavHeaderStyle::Extensible => FMT_BYTES_EXTENSIBLE,
        };
        let data_bytes = (samples.len() * self.codec.bytes_per_sample()) as u64;
        let ds64 = match self.flavour {
            RiffFlavour::Rf64 => true,
            RiffFlavour::Automatic => needs_rf64(fmt_bytes, data_bytes),
        };
        let declared_riff = riff_size(fmt_bytes, data_bytes, ds64);

        let start = output.len();
        output.extend_from_slice(if ds64 {
            riff::RF64.as_bytes()
        } else {
            riff::RIFF.as_bytes()
        });
        // RF64 leaves every oversized field at the sentinel and states the real
        // number in `ds64`, so that a plain RIFF reader meeting the file fails
        // on the size rather than reading four gigabytes of nothing.
        let riff_field = match (ds64, declared_riff) {
            (true, _) => riff::SIZE_SENTINEL,
            (false, Some(size)) => size as u32,
            (false, None) => riff::SIZE_SENTINEL,
        };
        output.extend_from_slice(&riff_field.to_le_bytes());
        output.extend_from_slice(riff::WAVE.as_bytes());

        if ds64 {
            output.extend_from_slice(riff::DS64.as_bytes());
            output.extend_from_slice(&(DS64_BODY_BYTES as u32).to_le_bytes());
            output.extend_from_slice(&declared_riff.unwrap_or(u64::MAX).to_le_bytes());
            output.extend_from_slice(&data_bytes.to_le_bytes());
            output.extend_from_slice(&((samples.len() / channels) as u64).to_le_bytes());
            output.extend_from_slice(&0u32.to_le_bytes()); // no override table
        }

        output.extend_from_slice(riff::FMT.as_bytes());
        output.extend_from_slice(&(fmt_bytes as u32).to_le_bytes());
        let tag = match self.header {
            WavHeaderStyle::Extensible => TAG_EXTENSIBLE,
            WavHeaderStyle::Plain => self.codec.format_tag(),
        };
        let bits = self.codec.bits_per_sample();
        output.extend_from_slice(&tag.to_le_bytes());
        output.extend_from_slice(&self.spec.channels.to_le_bytes());
        output.extend_from_slice(&self.spec.sample_rate.to_le_bytes());
        // `nAvgBytesPerSec` is a `u32` and the product need not fit one at an
        // absurd rate. Saturating keeps the field advisory rather than making
        // the writer fail on a number nothing reads.
        let byte_rate = u64::from(self.spec.sample_rate) * u64::from(block_align);
        output.extend_from_slice(&(byte_rate.min(u64::from(u32::MAX)) as u32).to_le_bytes());
        output.extend_from_slice(&block_align.to_le_bytes());
        output.extend_from_slice(&bits.to_le_bytes());
        if matches!(self.header, WavHeaderStyle::Extensible) {
            output.extend_from_slice(&22u16.to_le_bytes()); // cbSize
            output.extend_from_slice(&bits.to_le_bytes()); // wValidBitsPerSample
                                                           // A channel mask of zero says the channels have no assigned
                                                           // speaker positions, which is the truth: nothing told this writer
                                                           // where they go, and inventing a layout is how a rear channel ends
                                                           // up labelled as a centre one.
            output.extend_from_slice(&0u32.to_le_bytes());
            output.extend_from_slice(&self.codec.format_tag().to_le_bytes());
            output.extend_from_slice(&SUBFORMAT_TAIL);
        }

        output.extend_from_slice(riff::DATA.as_bytes());
        let data_field = if ds64 {
            riff::SIZE_SENTINEL
        } else {
            data_bytes as u32
        };
        output.extend_from_slice(&data_field.to_le_bytes());
        self.codec.encode(samples, output);
        if riff::pad_len(data_bytes) == 1 {
            // RIFF pads an odd body to an even boundary. The pad is outside the
            // chunk's declared size and inside the form's.
            output.push(0);
        }
        Ok(output.len() - start)
    }

    /// A complete file as a new `Vec<u8>`.
    ///
    /// # Errors
    ///
    /// As [`write`](Self::write).
    pub fn to_bytes(&self, samples: &[f32]) -> Result<Vec<u8>, DecodeError> {
        let mut output = Vec::new();
        self.write(samples, &mut output)?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every encoding, for the tests that have to cover all of them rather than
    /// the one that was being thought about.
    pub(crate) const ALL: [WavCodec; 8] = [
        WavCodec::PcmU8,
        WavCodec::PcmI16,
        WavCodec::PcmI24,
        WavCodec::PcmI32,
        WavCodec::Float32,
        WavCodec::Float64,
        WavCodec::ALaw,
        WavCodec::MuLaw,
    ];

    /// The exhaustive match is the assertion: a variant added to [`WavCodec`]
    /// without a line in [`ALL`] fails to compile here, so no later encoding
    /// can be added and quietly left out of every "every codec" gate in the
    /// suite.
    #[test]
    fn all_lists_every_codec() {
        for codec in ALL {
            match codec {
                WavCodec::PcmU8
                | WavCodec::PcmI16
                | WavCodec::PcmI24
                | WavCodec::PcmI32
                | WavCodec::Float32
                | WavCodec::Float64
                | WavCodec::ALaw
                | WavCodec::MuLaw => {}
            }
        }
        assert_eq!(ALL.len(), 8);
    }

    // -- Dispatch, on the pair --------------------------------------------

    /// The whole `(tag, bits)` space this crate accepts, and the shape of every
    /// rejection outside it.
    #[test]
    fn dispatch_is_on_the_tag_and_the_width_together() {
        let accepted: [(u16, u16, WavCodec); 8] = [
            (1, 8, WavCodec::PcmU8),
            (1, 16, WavCodec::PcmI16),
            (1, 24, WavCodec::PcmI24),
            (1, 32, WavCodec::PcmI32),
            (3, 32, WavCodec::Float32),
            (3, 64, WavCodec::Float64),
            (6, 8, WavCodec::ALaw),
            (7, 8, WavCodec::MuLaw),
        ];
        for (tag, bits, codec) in accepted {
            assert_eq!(WavCodec::resolve(tag, bits).expect("accepted"), codec);
            // And the codec names the pair it came from, so the writer cannot
            // disagree with the reader about what it just wrote.
            assert_eq!(codec.format_tag(), tag);
            assert_eq!(codec.bits_per_sample(), bits);
        }

        // A carried tag at a width this build does not have is a sample-format
        // rejection, and it names the width.
        for (tag, bits) in [
            (1u16, 4u16),
            (1, 12),
            (1, 20),
            (1, 64),
            (3, 16),
            (6, 16),
            (7, 4),
        ] {
            let error = WavCodec::resolve(tag, bits).expect_err("must reject");
            assert!(
                matches!(
                    &error,
                    DecodeError::UnsupportedSampleFormat { format, bits_per_sample }
                        if *format == CodecId::WaveFormatTag(tag) && *bits_per_sample == bits
                ),
                "({tag}, {bits}): unexpected error: {error}"
            );
        }

        // A tag this build does not carry is a codec rejection, and it names
        // the tag.
        for tag in [0u16, 2, 0x11, 0x31, 0x50, 0x161, 0xFFFE] {
            let error = WavCodec::resolve(tag, 16).expect_err("must reject");
            assert!(
                matches!(&error, DecodeError::UnsupportedCodec { codec }
                    if *codec == CodecId::WaveFormatTag(tag)),
                "tag {tag:#x}: unexpected error: {error}"
            );
        }
    }

    /// Every width tag 1 carries decodes at its own stride. This is the
    /// assertion a reader dispatching on the tag alone fails: it would read all
    /// four of these as whichever width it picked.
    #[test]
    fn each_width_of_tag_one_has_its_own_stride() {
        let widths: [(u16, usize); 4] = [(8, 1), (16, 2), (24, 3), (32, 4)];
        for (bits, bytes) in widths {
            let codec = WavCodec::resolve(1, bits).expect("accepted");
            assert_eq!(codec.bytes_per_sample(), bytes, "{bits} bits");
            let mut out = Vec::new();
            assert_eq!(codec.decode(&[0u8; 24], &mut out), 24 / bytes);
        }
    }

    #[test]
    fn every_codec_is_linear_or_companded_and_never_both() {
        for codec in ALL {
            assert_ne!(
                codec.sample_format().is_some(),
                codec.law().is_some(),
                "{codec:?} is on both sides or neither"
            );
            if let Some(format) = codec.sample_format() {
                assert_eq!(format.bytes_per_sample(), codec.bytes_per_sample());
                assert_eq!(format.bits_per_sample(), codec.bits_per_sample());
            } else {
                // G.711 is one byte per sample and eight bits per sample, and
                // the two statements are about different things: the code is a
                // byte, the sample it stands for is 13 or 14 bits wide.
                assert_eq!(codec.bytes_per_sample(), 1);
            }
        }
    }

    // -- The extensible header --------------------------------------------

    /// The GUID text form, against the two values Microsoft publishes.
    #[test]
    fn a_guid_is_reported_in_the_form_it_is_documented_in() {
        let mut pcm = [0u8; 16];
        pcm[0] = 1;
        pcm[2..].copy_from_slice(&SUBFORMAT_TAIL);
        assert_eq!(format_guid(&pcm), "00000001-0000-0010-8000-00aa00389b71");

        let ambisonic = [
            0x01, 0x00, 0x00, 0x00, 0x21, 0x07, 0xd3, 0x11, 0x86, 0x44, 0xc8, 0xc1, 0xca, 0x00,
            0x00, 0x00,
        ];
        assert_eq!(
            format_guid(&ambisonic),
            "00000001-0721-11d3-8644-c8c1ca000000"
        );
    }

    /// A `fmt ` body for the tests below: sixteen fixed bytes plus whatever
    /// extension is named. Written here rather than reached for from the
    /// writer, so parsing is tested against bytes the writer did not produce.
    fn fmt_body(tag: u16, channels: u16, rate: u32, bits: u16, extension: &[u8]) -> Vec<u8> {
        let block_align = channels * bits.div_ceil(8);
        let mut body = Vec::new();
        body.extend_from_slice(&tag.to_le_bytes());
        body.extend_from_slice(&channels.to_le_bytes());
        body.extend_from_slice(&rate.to_le_bytes());
        body.extend_from_slice(&(rate * u32::from(block_align)).to_le_bytes());
        body.extend_from_slice(&block_align.to_le_bytes());
        body.extend_from_slice(&bits.to_le_bytes());
        body.extend_from_slice(extension);
        body
    }

    /// A `WAVE_FORMAT_EXTENSIBLE` extension naming `tag`, with `cb_size` as
    /// declared so an inconsistent one can be built deliberately.
    fn extension(tag: u16, cb_size: u16) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&cb_size.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes()); // wValidBitsPerSample
        bytes.extend_from_slice(&0u32.to_le_bytes()); // dwChannelMask
        bytes.extend_from_slice(&tag.to_le_bytes());
        bytes.extend_from_slice(&SUBFORMAT_TAIL);
        bytes
    }

    #[test]
    fn an_extensible_header_resolves_through_its_subformat_guid() {
        for (tag, bits, codec) in [
            (1u16, 16u16, WavCodec::PcmI16),
            (1, 24, WavCodec::PcmI24),
            (3, 32, WavCodec::Float32),
            (6, 8, WavCodec::ALaw),
            (7, 8, WavCodec::MuLaw),
        ] {
            let body = fmt_body(TAG_EXTENSIBLE, 2, 48_000, bits, &extension(tag, 22));
            let format = parse_fmt(&body, 12).expect("extensible header");
            assert_eq!(format.codec, codec);
            // The declared tag is kept as the file wrote it.
            assert_eq!(format.format_tag, TAG_EXTENSIBLE);
            assert!(format.is_extensible());
        }
    }

    #[test]
    fn an_inconsistent_cb_size_is_malformed_rather_than_read_past() {
        // cbSize under 22 says there is no room for a GUID.
        for cb_size in [0u16, 1, 21] {
            let body = fmt_body(TAG_EXTENSIBLE, 1, 8_000, 16, &extension(1, cb_size));
            let error = parse_fmt(&body, 12).expect_err("must reject");
            assert!(
                matches!(
                    error,
                    DecodeError::Malformed {
                        expected: "a 22-byte WAVE_FORMAT_EXTENSIBLE extension after cbSize",
                        ..
                    }
                ),
                "cbSize {cb_size}: unexpected error: {error}"
            );
        }
        // cbSize says 22 and the body does not carry them.
        for truncate_to in [16usize, 18, 24, 39] {
            let mut body = fmt_body(TAG_EXTENSIBLE, 1, 8_000, 16, &extension(1, 22));
            body.truncate(truncate_to);
            let error = parse_fmt(&body, 12).expect_err("must reject");
            assert!(
                matches!(error, DecodeError::Malformed { .. }),
                "{truncate_to} bytes: unexpected error: {error}"
            );
        }
    }

    #[test]
    fn an_unrecognised_subformat_guid_is_reported_as_the_guid_it_was() {
        let mut ext = extension(1, 22);
        // Break the first byte of the fixed tail, so the GUID is outside the
        // WAVEFORMATEX family and names a codec with no tag at all.
        ext[10] = 0xAB;
        let body = fmt_body(TAG_EXTENSIBLE, 1, 8_000, 16, &ext);
        let error = parse_fmt(&body, 12).expect_err("must reject");
        match error {
            DecodeError::UnsupportedCodec {
                codec: CodecId::Name(name),
            } => assert_eq!(name, "00ab0001-0000-0010-8000-00aa00389b71"),
            other => panic!("unexpected error: {other}"),
        }
    }

    /// An extensible header whose GUID names a codec this build does not carry
    /// reports the tag the GUID named, not `0xFFFE`. `0xFFFE` says only "look
    /// in the extension", so reporting it back tells a caller nothing.
    #[test]
    fn an_extensible_header_naming_an_unsupported_codec_reports_the_resolved_tag() {
        let body = fmt_body(TAG_EXTENSIBLE, 1, 8_000, 4, &extension(0x0011, 22));
        let error = parse_fmt(&body, 12).expect_err("must reject");
        assert!(
            matches!(&error, DecodeError::UnsupportedCodec { codec }
                if *codec == CodecId::WaveFormatTag(0x0011)),
            "unexpected error: {error}"
        );
    }

    // -- `fmt ` validation ------------------------------------------------

    #[test]
    fn a_zero_channel_count_is_an_unsupported_layout() {
        let body = fmt_body(1, 0, 8_000, 16, &[]);
        let error = parse_fmt(&body, 12).expect_err("must reject");
        assert!(
            matches!(error, DecodeError::UnsupportedChannelLayout { channels: 0 }),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn a_zero_sample_rate_is_malformed_and_names_where_it_was() {
        let body = fmt_body(1, 1, 0, 16, &[]);
        let error = parse_fmt(&body, 12).expect_err("must reject");
        assert!(
            matches!(
                error,
                DecodeError::Malformed {
                    expected: "a non-zero sample rate",
                    offset: 24
                }
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn a_fmt_chunk_size_outside_what_waveformatex_can_mean_is_malformed() {
        for size in [0u64, 1, 15] {
            assert!(check_fmt_size(size, 12).is_err(), "{size} bytes accepted");
        }
        for size in [16u64, 18, 40, MAX_FMT_BYTES] {
            assert!(check_fmt_size(size, 12).is_ok(), "{size} bytes rejected");
        }
        // cbSize is a u16, so this is the format's ceiling and not a number
        // chosen here.
        assert_eq!(MAX_FMT_BYTES, 65_553);
        assert!(check_fmt_size(MAX_FMT_BYTES + 1, 12).is_err());
    }

    #[test]
    fn a_block_align_that_disagrees_with_the_format_does_not_stop_the_decode() {
        let mut body = fmt_body(1, 2, 44_100, 16, &[]);
        body[12..14].copy_from_slice(&999u16.to_le_bytes());
        let format = parse_fmt(&body, 12).expect("advisory field");
        assert_eq!(format.block_align, 999, "the declaration is kept");
        assert_eq!(format.frame_bytes(), 4, "and the frame size is computed");
    }

    // -- Whole frames -----------------------------------------------------

    #[test]
    fn a_data_length_that_is_not_whole_frames_is_truncated() {
        // Four-byte frames: two channels of sixteen-bit.
        assert!(check_whole_frames(8, 4).is_ok());
        for (len, remainder) in [(9u64, 1u64), (10, 2), (11, 3), (1, 1)] {
            let error = check_whole_frames(len, 4).expect_err("must reject");
            assert!(
                matches!(error, DecodeError::Truncated { expected: 4, available }
                    if available == remainder),
                "{len} bytes: unexpected error: {error}"
            );
        }
        // Zero frames is a whole number of frames.
        assert!(check_whole_frames(0, 4).is_ok());
    }

    // -- The RF64 threshold -----------------------------------------------

    /// The upgrade point, tested as arithmetic rather than by materialising
    /// four gigabytes to cross it with.
    #[test]
    fn the_rf64_upgrade_happens_exactly_at_the_riff_size_limit() {
        // A plain RIFF header over a sixteen-byte `fmt `: 4 for the form type,
        // 8 + 16 for the `fmt ` chunk and 8 for the `data` header, 36 bytes
        // the payload has to share `u32::MAX` with.
        const OVERHEAD: u64 =
            4 + riff::CHUNK_HEADER_BYTES + FMT_BYTES_PLAIN + riff::CHUNK_HEADER_BYTES;
        assert_eq!(OVERHEAD, 36);
        let ceiling = u64::from(u32::MAX);

        // `u32::MAX - 36` is odd, so its pad byte is what tips it over: the
        // largest payload that fits is one less than that, and even.
        let largest = ceiling - OVERHEAD - 1;
        assert_eq!(largest % 2, 0);
        assert_eq!(
            riff_size(FMT_BYTES_PLAIN, largest, false),
            Some(ceiling - 1)
        );
        assert!(
            !needs_rf64(FMT_BYTES_PLAIN, largest),
            "the last fitting size"
        );
        assert!(needs_rf64(FMT_BYTES_PLAIN, largest + 1), "one byte past it");
        assert!(needs_rf64(FMT_BYTES_PLAIN, largest + 2));

        // The pad byte is genuinely counted: the odd payload one byte smaller
        // fits only because 36 + payload + 1 still lands on the ceiling.
        assert_eq!(
            riff_size(FMT_BYTES_PLAIN, largest - 1, false),
            Some(ceiling - 1)
        );
        assert!(!needs_rf64(FMT_BYTES_PLAIN, largest - 1));

        // A larger `fmt ` chunk brings the limit down by exactly its extra
        // twenty-four bytes.
        assert!(needs_rf64(FMT_BYTES_EXTENSIBLE, largest));
        assert!(!needs_rf64(FMT_BYTES_EXTENSIBLE, largest - 24));
        assert!(needs_rf64(FMT_BYTES_EXTENSIBLE, largest - 23));

        // And nothing small is upgraded.
        assert!(!needs_rf64(FMT_BYTES_PLAIN, 0));
        assert!(!needs_rf64(FMT_BYTES_PLAIN, 1_000_000));
    }

    #[test]
    fn the_riff_size_field_counts_everything_after_the_first_eight_bytes() {
        // 4 (WAVE) + 8 + 16 (fmt) + 8 + 4 (data) = 40.
        assert_eq!(riff_size(FMT_BYTES_PLAIN, 4, false), Some(40));
        // An odd payload adds its pad byte to the form's size.
        assert_eq!(riff_size(FMT_BYTES_PLAIN, 5, false), Some(42));
        // And ds64 adds its own header and body.
        assert_eq!(riff_size(FMT_BYTES_PLAIN, 4, true), Some(76));
    }

    // -- The writer's rejections ------------------------------------------

    #[test]
    fn the_writer_rejects_a_partial_trailing_frame() {
        let writer = WavWriter::new(AudioSpec::new(48_000, 3), WavCodec::PcmI16);
        let error = writer.to_bytes(&[0.0; 7]).expect_err("must reject");
        assert!(
            matches!(
                error,
                DecodeError::Truncated {
                    expected: 6,
                    available: 2
                }
            ),
            "unexpected error: {error}"
        );
        // Seven is one sample past two frames; six and nine are whole.
        assert!(writer.to_bytes(&[0.0; 6]).is_ok());
        assert!(writer.to_bytes(&[0.0; 9]).is_ok());
    }

    #[test]
    fn the_writer_rejects_what_the_header_cannot_express() {
        let no_channels = WavWriter::new(AudioSpec::new(48_000, 0), WavCodec::PcmI16);
        assert!(matches!(
            no_channels.to_bytes(&[]),
            Err(DecodeError::UnsupportedChannelLayout { channels: 0 })
        ));

        let no_rate = WavWriter::new(AudioSpec::mono(0), WavCodec::PcmI16);
        assert!(matches!(
            no_rate.to_bytes(&[]),
            Err(DecodeError::Malformed {
                expected: "a non-zero sample rate",
                ..
            })
        ));

        // nBlockAlign is a u16, so a frame wider than 65,535 bytes cannot be
        // described. 8,192 channels of f64 is 65,536.
        let too_wide = WavWriter::new(AudioSpec::new(48_000, 8_192), WavCodec::Float64);
        assert!(matches!(
            too_wide.to_bytes(&[]),
            Err(DecodeError::UnsupportedChannelLayout { channels: 8_192 })
        ));
        // And one channel fewer fits exactly.
        let widest = WavWriter::new(AudioSpec::new(48_000, 8_191), WavCodec::Float64);
        assert!(widest.to_bytes(&[]).is_ok());
    }

    #[test]
    fn the_writer_appends_rather_than_clearing() {
        let writer = WavWriter::new(AudioSpec::mono(8_000), WavCodec::PcmU8);
        let mut output = vec![0xAA];
        let written = writer.write(&[0.0, 0.0], &mut output).expect("write");
        assert_eq!(output[0], 0xAA);
        assert_eq!(output.len(), 1 + written);
        assert_eq!(written, 12 + 8 + 16 + 8 + 2);
    }
}
