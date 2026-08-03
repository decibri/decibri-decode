//! AIFF and AIFF-C: the second container this crate reads and writes, and the
//! first big-endian one. [`AiffWriter`] writes every encoding the readers
//! accept, including the 80-bit extended-float *encoding* side of the sample
//! rate, so no format here is supported in one direction only.
//!
//! # AIFF is not WAV with the bytes swapped
//!
//! The chunk walk really is shared (EA IFF 85 chunks are RIFF chunks with a
//! big-endian size field, and [`ChunkWalker`] takes the byte order as an
//! argument) but everything the chunks *mean* differs, and five differences
//! in particular are traps that produce silently wrong audio rather than an
//! error:
//!
//! - **Eight-bit samples are signed.** The exact inverse of WAV's unsigned
//!   convention, and not negotiable: it is what the two specifications say.
//!   Getting it wrong offsets the audio by half full scale. See
//!   [`SampleFormat::I8`]. The one exception is AIFF-C's `raw `, which names
//!   offset-binary unsigned 8-bit and is [`SampleFormat::U8`].
//! - **The sample rate is an 80-bit IEEE 754 extended-precision float.** Sign
//!   bit, 15-bit exponent, and a 64-bit significand whose integer bit is
//!   explicit, unlike a double's. No Rust primitive holds it, so it is parsed
//!   by hand in [`extended_to_f64`]. Rates are integers in practice and the
//!   field is a float regardless; the parser handles fractional values
//!   correctly and the reader then rejects a non-integral rate with a typed
//!   error rather than rounding it, because a rounded rate is a wrong rate
//!   delivered silently.
//! - **`sowt` is little-endian data inside a big-endian container.** AIFF-C
//!   names its encoding in a compression four-CC: `NONE` and `twos` are
//!   big-endian two's complement, and `sowt`, `twos` spelled backwards, is
//!   little-endian, which Apple used heavily. The container's byte order says
//!   nothing about the payload's.
//! - **`SSND` does not start with samples.** Its body opens with two 32-bit
//!   fields, `offset` and `blockSize`, and the sample data begins `offset`
//!   bytes after them. Both are almost always zero, which is exactly why a
//!   parser that ignores them passes every test until it meets a file where
//!   they are not.
//! - **There are two sources of truth for length.** `COMM` declares
//!   `numSampleFrames` and `SSND` has a chunk size, and they can disagree.
//!   This reader requires them to agree exactly and answers a typed error
//!   when they do not: [`DecodeError::Truncated`] when `SSND` holds fewer
//!   bytes than the declared frames need, [`DecodeError::Malformed`] when it
//!   holds more. Silently preferring either source is how audio gets
//!   truncated or extended without anybody noticing, which is the failure
//!   class this crate exists to avoid.
//!
//! # Dispatch is on the form type and the compression four-CC
//!
//! A `FORM` of type `AIFF` is uncompressed big-endian PCM by definition; a
//! `FORM` of type `AIFC` names its encoding in `COMM`'s compression type.
//! `alaw`/`ALAW` and `ulaw`/`ULAW` dispatch to the same G.711 tables WAV uses;
//! `fl32`/`FL32` and `fl64`/`FL64` are IEEE float; `raw ` is unsigned 8-bit
//! linear PCM. Every other compression type (`ima4`, `MAC3`, `MAC6`, `QDMC`
//! and the rest) is [`DecodeError::UnsupportedCodec`], reported as the
//! four-CC it was. Nothing anywhere looks at a file name.
//!
//! For the linear widths, dispatch is on the **(compression, sampleSize)
//! pair**, the same rule as WAV's `(tag, bits)`: `NONE` covers four widths and
//! reading one at another's stride produces noise rather than an error. The
//! companded types are the exception, deliberately: G.711's wire stride is one
//! byte whatever `sampleSize` claims, and AIFF-C writers disagree about
//! whether the field should carry 8 (the wire width) or 16 (the decoded
//! width), so for `alaw` and `ulaw` the field is recorded and not dispatched
//! on.
//!
//! # A size in a file is a claim, not a fact
//!
//! The same discipline as [`wav`](crate::wav): malformed input is a typed
//! [`DecodeError`], never a panic, and no allocation is proportional to a
//! declared size. The streaming reader buffers chunk headers, the `COMM` body
//! (capped at [`MAX_COMM_BYTES`], the format's own ceiling) and `SSND`'s
//! eight-byte prefix, and nothing else. All chunk arithmetic is 64-bit and
//! checked.
//!
//! # Two readers, and the one difference between them
//!
//! [`AiffReader`] holds the whole file and accepts `COMM` and `SSND` in
//! either order. [`AiffStreamDecoder`] requires `COMM` first, for the reason
//! recorded on [`WavStreamDecoder`](crate::WavStreamDecoder): a payload that
//! arrives before the header describing it would have to be buffered in full,
//! and an unbounded buffer is the thing the streaming reader exists to avoid.
//! A file that opens fine can therefore fail when streamed; that is forced by
//! streaming, not chosen.

use crate::audio::{AudioBuffer, AudioSpec};
use crate::codec::{CodecId, FourCc};
use crate::error::DecodeError;
use crate::g711::G711Law;
use crate::payload::Payload;
use crate::riff::{self, ByteOrder, ChunkWalker};
use crate::sample::SampleFormat;
use crate::source::StreamSource;

/// The magic of every EA IFF 85 file.
///
/// Visible to the crate because [`probe`](crate::probe) has to tell a `FORM`
/// that is an AIFF from one that is an `8SVX`, and the three form four-CCs are
/// written down once for both readers of them.
pub(crate) const FORM: FourCc = FourCc(*b"FORM");

/// The form type of a plain AIFF file: uncompressed big-endian PCM.
pub(crate) const AIFF: FourCc = FourCc(*b"AIFF");

/// The form type of an AIFF-C file: the encoding is in `COMM`.
pub(crate) const AIFC: FourCc = FourCc(*b"AIFC");

/// The chunk describing the payload: channels, frames, width, rate and, in
/// AIFF-C, the compression type.
const COMM: FourCc = FourCc(*b"COMM");

/// The chunk carrying the payload, behind its `offset` and `blockSize`
/// fields.
const SSND: FourCc = FourCc(*b"SSND");

/// AIFF-C's `FVER` chunk. The *readers* deliberately treat it as unknown,
/// read past rather than validated, because it has carried the same timestamp
/// since 1991 and a file whose audio parses is not worth rejecting over a
/// bookkeeping field. The writer writes it, because the AIFF-C specification
/// requires one and omitting it costs compatibility for nothing.
const FVER: FourCc = FourCc(*b"FVER");

/// The one `FVER` version timestamp that has ever existed: AIFC Version 1,
/// May 23, 1990, as an Apple timestamp.
const FVER_TIMESTAMP: u32 = 0xA280_5140;

/// `NONE`: uncompressed big-endian two's-complement PCM.
const NONE: FourCc = FourCc(*b"NONE");

/// `twos`: the same thing `NONE` is, under the name QuickTime gave it.
const TWOS: FourCc = FourCc(*b"twos");

/// `sowt`: `twos` backwards, and little-endian two's-complement PCM.
const SOWT: FourCc = FourCc(*b"sowt");

/// `raw `: offset-binary, meaning unsigned, 8-bit linear PCM. The trailing
/// space is part of the four-CC.
const RAW: FourCc = FourCc(*b"raw ");

/// The two spellings of IEEE 754 binary32.
const FL32: [FourCc; 2] = [FourCc(*b"fl32"), FourCc(*b"FL32")];

/// The two spellings of IEEE 754 binary64.
const FL64: [FourCc; 2] = [FourCc(*b"fl64"), FourCc(*b"FL64")];

/// The two spellings of G.711 A-law.
const ALAW: [FourCc; 2] = [FourCc(*b"alaw"), FourCc(*b"ALAW")];

/// The two spellings of G.711 mu-law.
const ULAW: [FourCc; 2] = [FourCc(*b"ulaw"), FourCc(*b"ULAW")];

/// The `COMM` body a plain AIFF file carries: channels, frames, width, rate.
const COMM_BYTES_AIFF: u64 = 18;

/// The smallest `COMM` body an AIFF-C file can carry: the AIFF fields plus
/// the compression four-CC.
///
/// The specification also requires a compressionName pascal string after the
/// four-CC, but writers that omit it entirely exist and the audio in their
/// files is complete, so a 22-byte body is read rather than rejected.
const COMM_BYTES_AIFC: u64 = 22;

/// The largest `COMM` body this crate will read.
///
/// The format's own ceiling rather than a number picked here: the
/// compressionName is a pascal string whose length byte bounds it at 255
/// characters, padded to an even total, so 22 + 256 is the most a conforming
/// `COMM` can occupy. It matters because `COMM` is the one chunk body the
/// streaming reader buffers rather than skips, so its declared size governs a
/// buffer.
const MAX_COMM_BYTES: u64 = COMM_BYTES_AIFC + 256;

/// The two fields at the start of every `SSND` body, before `offset` bytes of
/// alignment padding and then the sample data.
const SSND_PREFIX_BYTES: u64 = 8;

/// How many decoded samples the streaming reader holds before it stops taking
/// bytes. The same figure, for the same reason, as every other bounded reader
/// in the crate.
const READY_LIMIT: usize = 65_536;

// -- The 80-bit extended-precision sample rate --------------------------------

/// Parses the 80-bit IEEE 754 extended-precision float `COMM` stores its
/// sample rate in.
///
/// Sign bit, 15-bit exponent biased by 16383, then a 64-bit significand whose
/// integer bit is **explicit**, unlike a double, where it is implied. The
/// value is `(-1)^sign * significand * 2^(exponent - 16383 - 63)`.
///
/// The conversion is exact wherever `f64` can hold the answer: the
/// significand-to-`f64` conversion rounds to nearest (ties to even) only when
/// more than 53 bits are set, and the scaling multiplies by exact powers of
/// two. Values beyond `f64`'s range become infinity or zero, which the caller
/// rejects the same way it rejects any other impossible rate. Deterministic
/// on every target: nothing here depends on the platform's `long double`.
fn extended_to_f64(bytes: [u8; 10]) -> f64 {
    let negative = bytes[0] & 0x80 != 0;
    let exponent = (u16::from(bytes[0] & 0x7F) << 8) | u16::from(bytes[1]);
    let significand = u64::from_be_bytes([
        bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7], bytes[8], bytes[9],
    ]);
    let sign = if negative { -1.0 } else { 1.0 };

    if exponent == 0x7FFF {
        // Infinity when the fraction below the integer bit is clear, NaN
        // otherwise. Neither is a rate, and the caller says so.
        return if significand << 1 == 0 {
            sign * f64::INFINITY
        } else {
            f64::NAN
        };
    }
    if significand == 0 {
        return sign * 0.0;
    }
    // A zero exponent with a non-zero significand is the denormal range,
    // whose values are 2^-16446 at the largest: zero in f64 and zero as a
    // sample rate. The `max(1)` gives them the exponent the format defines
    // for them and the scaling below underflows them to zero deterministically.
    let shift = i32::from(exponent.max(1)) - 16383 - 63;
    sign * scale_by_power_of_two(significand as f64, shift)
}

/// `value * 2^shift`, for shifts far beyond what a single `f64` multiply can
/// express.
///
/// Stepping by at most 1000 keeps every intermediate scale factor a normal,
/// exactly-representable power of two, so each multiplication is exact until
/// the result itself overflows to infinity or underflows to zero, both of
/// which are the correct destination for an 80-bit exponent `f64` cannot
/// hold.
fn scale_by_power_of_two(value: f64, shift: i32) -> f64 {
    const STEP: i32 = 1000;
    let mut value = value;
    let mut shift = shift;
    while shift != 0 && value != 0.0 && value.is_finite() {
        let step = shift.clamp(-STEP, STEP);
        // 2^step as bits: biased exponent in the top eleven bits of an f64.
        value *= f64::from_bits(((step + 1023) as u64) << 52);
        shift -= step;
    }
    value
}

/// The declared rate as the `u32` an [`AudioSpec`] carries, or a typed
/// rejection.
///
/// A rate of zero, a negative rate, infinity, NaN, a fraction and a value
/// past `u32::MAX` are all rejected with [`DecodeError::Malformed`] naming
/// the field's offset. Rounding a fractional rate instead would deliver a
/// wrong rate silently, which is the exact failure [`AudioSpec`] exists to
/// prevent; no real writer produces one, so the rejection costs nothing.
fn rate_to_spec_rate(rate: f64, offset: u64) -> Result<u32, DecodeError> {
    // NaN fails the first two comparisons and differs from its own
    // truncation, so it takes the same door as every other impossible rate.
    if rate < 1.0 || rate > f64::from(u32::MAX) || rate != rate.trunc() {
        return Err(DecodeError::Malformed {
            expected: "a positive integral sample rate that fits in 32 bits",
            offset,
        });
    }
    Ok(rate as u32)
}

/// Encodes an integer sample rate as the 80-bit extended-precision float
/// `COMM` stores.
///
/// The encode half of [`extended_to_f64`]: sign bit clear, the exponent
/// biased by 16383, and the significand normalised so its **explicit**
/// integer bit, the bit a double leaves implied, is set. Normalising means
/// shifting the rate's leading one up to bit 63 and recording how far below
/// 63 it started, so every non-zero `u32` is represented exactly: the
/// significand holds the rate's own bits and nothing is rounded. `rate` is
/// never zero here, since the writer rejects a zero rate before encoding it,
/// and zero is the one value this normalisation cannot spell.
fn rate_to_extended(rate: u32) -> [u8; 10] {
    debug_assert_ne!(rate, 0, "a zero rate has no normalised form");
    let high_bit = 31 - rate.leading_zeros();
    let exponent = 16383 + high_bit as u16;
    let significand = u64::from(rate) << (63 - high_bit);
    let mut bytes = [0u8; 10];
    bytes[0] = (exponent >> 8) as u8;
    bytes[1] = exponent as u8;
    bytes[2..].copy_from_slice(&significand.to_be_bytes());
    bytes
}

// -- What an AIFF file can carry ----------------------------------------------

/// Which of the two form types the file declared.
///
/// `#[non_exhaustive]` for the same reason the other container enums are: a
/// consumer matching on it keeps a `_` arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AiffForm {
    /// A plain `AIFF` form: uncompressed big-endian PCM, no compression
    /// field.
    Aiff,
    /// An `AIFC` form: the encoding is named by `COMM`'s compression four-CC.
    Aifc,
}

/// An encoding an AIFF or AIFF-C file can carry and this crate can decode.
///
/// The variants are exactly the `(compression, sampleSize)` pairs this crate
/// reads, the same shape as [`WavCodec`](crate::WavCodec) and for the same
/// reason: a type that carried an arbitrary [`SampleFormat`] could name files
/// that cannot exist. The three `...Sowt` variants are the same integer widths
/// as their big-endian twins with the bytes the other way round: `sowt` data
/// inside the big-endian container.
///
/// `#[non_exhaustive]`: a consumer matching on it needs a `_` arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AiffCodec {
    /// Signed 8-bit PCM, signed unlike WAV's. `NONE`, `twos` or `sowt` at
    /// 8 bits: one byte has no byte order.
    PcmI8,
    /// Unsigned 8-bit PCM, offset binary. `raw ` at 8 bits, which the
    /// compression field makes an AIFF-C encoding only. The same convention
    /// WAV's 8-bit carries, under a four-CC that names it explicitly rather
    /// than the implicit signedness of `NONE`.
    PcmU8,
    /// Signed 16-bit big-endian PCM. `NONE` or `twos` at 16 bits.
    PcmI16,
    /// Signed 24-bit big-endian PCM. `NONE` or `twos` at 24 bits.
    PcmI24,
    /// Signed 32-bit big-endian PCM. `NONE` or `twos` at 32 bits.
    PcmI32,
    /// Signed 16-bit **little-endian** PCM. `sowt` at 16 bits.
    PcmI16Sowt,
    /// Signed 24-bit **little-endian** PCM. `sowt` at 24 bits.
    PcmI24Sowt,
    /// Signed 32-bit **little-endian** PCM. `sowt` at 32 bits.
    PcmI32Sowt,
    /// IEEE 754 binary32, big-endian. `fl32` or `FL32`.
    Float32,
    /// IEEE 754 binary64, big-endian. `fl64` or `FL64`.
    Float64,
    /// ITU-T G.711 A-law. `alaw` or `ALAW`.
    ALaw,
    /// ITU-T G.711 mu-law. `ulaw` or `ULAW`.
    MuLaw,
}

impl AiffCodec {
    /// The encoding a `(compression, sampleSize)` pair names.
    ///
    /// A plain `AIFF` form has no compression field; its `NONE` is implicit,
    /// and the `COMM` parser passes it explicitly so a width rejection can
    /// name it. That is also what confines the compression-only encodings to
    /// `AIFC`: a plain `AIFF` form resolves as `NONE` whatever bytes follow
    /// the four fixed `COMM` fields, so `raw `, `sowt`, the floats and G.711
    /// are reachable from an `AIFC` form and from no other.
    ///
    /// For the linear types the dispatch is on the pair, never the four-CC
    /// alone. `NONE` covers four widths, and decoding one at another's
    /// stride is noise rather than an error. For the companded types the
    /// four-CC alone decides: G.711's wire stride is one byte whatever
    /// `sampleSize` says, and real writers put both 8 and 16 in the field.
    ///
    /// ```
    /// use decibri_decode::{AiffCodec, FourCc};
    ///
    /// assert_eq!(AiffCodec::resolve(FourCc(*b"NONE"), 24).unwrap(), AiffCodec::PcmI24);
    /// assert_eq!(AiffCodec::resolve(FourCc(*b"sowt"), 16).unwrap(), AiffCodec::PcmI16Sowt);
    /// // `raw ` is unsigned 8-bit, and only at 8 bits.
    /// assert_eq!(AiffCodec::resolve(FourCc(*b"raw "), 8).unwrap(), AiffCodec::PcmU8);
    /// assert!(AiffCodec::resolve(FourCc(*b"raw "), 16).is_err());
    /// // Same compression, unsupported width: rejected, and it says which.
    /// assert!(AiffCodec::resolve(FourCc(*b"NONE"), 20).is_err());
    /// ```
    ///
    /// # Errors
    ///
    /// [`DecodeError::UnsupportedSampleFormat`] for a carried compression
    /// type at a width this crate does not read, and
    /// [`DecodeError::UnsupportedCodec`] for a compression type it does not
    /// carry at all, `ima4`, `MAC3` and the rest.
    pub fn resolve(compression: FourCc, bits_per_sample: u16) -> Result<Self, DecodeError> {
        if compression == NONE || compression == TWOS || compression == SOWT {
            let sowt = compression == SOWT;
            return match (bits_per_sample, sowt) {
                (8, _) => Ok(Self::PcmI8),
                (16, false) => Ok(Self::PcmI16),
                (24, false) => Ok(Self::PcmI24),
                (32, false) => Ok(Self::PcmI32),
                (16, true) => Ok(Self::PcmI16Sowt),
                (24, true) => Ok(Self::PcmI24Sowt),
                (32, true) => Ok(Self::PcmI32Sowt),
                _ => Err(DecodeError::UnsupportedSampleFormat {
                    format: CodecId::FourCc(compression),
                    bits_per_sample,
                }),
            };
        }
        if compression == RAW {
            return if bits_per_sample == 8 {
                Ok(Self::PcmU8)
            } else {
                Err(DecodeError::UnsupportedSampleFormat {
                    format: CodecId::FourCc(compression),
                    bits_per_sample,
                })
            };
        }
        if FL32.contains(&compression) {
            return if bits_per_sample == 32 {
                Ok(Self::Float32)
            } else {
                Err(DecodeError::UnsupportedSampleFormat {
                    format: CodecId::FourCc(compression),
                    bits_per_sample,
                })
            };
        }
        if FL64.contains(&compression) {
            return if bits_per_sample == 64 {
                Ok(Self::Float64)
            } else {
                Err(DecodeError::UnsupportedSampleFormat {
                    format: CodecId::FourCc(compression),
                    bits_per_sample,
                })
            };
        }
        if ALAW.contains(&compression) {
            return Ok(Self::ALaw);
        }
        if ULAW.contains(&compression) {
            return Ok(Self::MuLaw);
        }
        Err(DecodeError::UnsupportedCodec {
            codec: CodecId::FourCc(compression),
        })
    }

    /// How many bytes one sample occupies in the payload.
    pub const fn bytes_per_sample(self) -> usize {
        match self {
            Self::PcmI8 | Self::PcmU8 | Self::ALaw | Self::MuLaw => 1,
            Self::PcmI16 | Self::PcmI16Sowt => 2,
            Self::PcmI24 | Self::PcmI24Sowt => 3,
            Self::PcmI32 | Self::PcmI32Sowt | Self::Float32 => 4,
            Self::Float64 => 8,
        }
    }

    /// The linear PCM format this encoding is, or `None` for the companded
    /// ones.
    pub const fn sample_format(self) -> Option<SampleFormat> {
        match self {
            Self::PcmI8 => Some(SampleFormat::I8),
            Self::PcmU8 => Some(SampleFormat::U8),
            Self::PcmI16 => Some(SampleFormat::I16Be),
            Self::PcmI24 => Some(SampleFormat::I24Be),
            Self::PcmI32 => Some(SampleFormat::I32Be),
            Self::PcmI16Sowt => Some(SampleFormat::I16Le),
            Self::PcmI24Sowt => Some(SampleFormat::I24Le),
            Self::PcmI32Sowt => Some(SampleFormat::I32Le),
            Self::Float32 => Some(SampleFormat::F32Be),
            Self::Float64 => Some(SampleFormat::F64Be),
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

    /// The `sampleSize` a file written with this encoding declares.
    ///
    /// For the linear and float types this is the real sample width. For the
    /// companded types it is 16, the decoded width, which is how the AIFF-C
    /// specification's compression registry describes G.711, while the wire
    /// stride stays one byte. Real writers put both 8 and 16 in the field and
    /// the readers here dispatch on the four-CC alone, so either spelling
    /// reads back; 16 is written because it is the specification's own.
    pub const fn bits_per_sample(self) -> u16 {
        match self {
            Self::PcmI8 | Self::PcmU8 => 8,
            Self::PcmI16 | Self::PcmI16Sowt | Self::ALaw | Self::MuLaw => 16,
            Self::PcmI24 | Self::PcmI24Sowt => 24,
            Self::PcmI32 | Self::PcmI32Sowt | Self::Float32 => 32,
            Self::Float64 => 64,
        }
    }

    /// The compression four-CC a written AIFF-C file carrying this encoding
    /// declares.
    ///
    /// One canonical spelling per encoding: the readers accept `twos` for
    /// big-endian PCM and the uppercase float and G.711 spellings as well,
    /// but a writer that varied its spelling would produce two different
    /// files for one encoding, and determinism is the crate's first claim.
    /// The big-endian PCM types answer `NONE`, which is also what a plain
    /// `AIFF` form means implicitly. See [`form`](Self::form) for why those
    /// types are not written as `AIFC` at all.
    pub const fn compression_type(self) -> FourCc {
        match self {
            Self::PcmI8 | Self::PcmI16 | Self::PcmI24 | Self::PcmI32 => NONE,
            Self::PcmU8 => RAW,
            Self::PcmI16Sowt | Self::PcmI24Sowt | Self::PcmI32Sowt => SOWT,
            Self::Float32 => FL32[0],
            Self::Float64 => FL64[0],
            Self::ALaw => ALAW[0],
            Self::MuLaw => ULAW[0],
        }
    }

    /// The form type a file written with this encoding uses.
    ///
    /// This is the writer's whole selection rule: plain `AIFF` carries only
    /// uncompressed big-endian two's-complement PCM, so the four codecs that
    /// *are* that, [`PcmI8`](Self::PcmI8) through [`PcmI32`](Self::PcmI32),
    /// are written as plain `AIFF`, the more widely readable of the two
    /// forms. Everything else, `raw `, `sowt`, the floats and G.711, needs
    /// the compression field that only `AIFC` has, and is written as `AIFC`
    /// with the `FVER` chunk the specification requires.
    pub const fn form(self) -> AiffForm {
        match self {
            Self::PcmI8 | Self::PcmI16 | Self::PcmI24 | Self::PcmI32 => AiffForm::Aiff,
            Self::PcmU8
            | Self::PcmI16Sowt
            | Self::PcmI24Sowt
            | Self::PcmI32Sowt
            | Self::Float32
            | Self::Float64
            | Self::ALaw
            | Self::MuLaw => AiffForm::Aifc,
        }
    }

    /// Appends every whole sample in `bytes` to `output`, and returns how
    /// many it appended.
    ///
    /// Straight through to [`SampleFormat::decode`] or [`G711Law::decode`]:
    /// there is no AIFF-specific conversion anywhere in this crate, so the
    /// same payload decoded through this container and through WAV gives the
    /// same samples by construction rather than by agreement.
    pub fn decode(self, bytes: &[u8], output: &mut Vec<f32>) -> usize {
        match (self.sample_format(), self.law()) {
            (Some(format), _) => format.decode(bytes, output),
            (_, Some(law)) => law.decode(bytes, output),
            // Unreachable: every variant is one or the other.
            _ => 0,
        }
    }

    /// Appends `samples` to `output` in this encoding, and returns how many
    /// bytes it appended.
    ///
    /// The encode half of [`decode`](Self::decode), through the same two
    /// layers, so the writer cannot have a conversion the reader lacks.
    pub fn encode(self, samples: &[f32], output: &mut Vec<u8>) -> usize {
        match (self.sample_format(), self.law()) {
            (Some(format), _) => format.encode(samples, output),
            (_, Some(law)) => law.encode(samples, output),
            _ => 0,
        }
    }
}

/// A resolved `COMM` chunk: what the file declared, and what it resolved to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct AiffFormat {
    /// The rate and layout the payload decodes to.
    pub spec: AudioSpec,
    /// What the payload is encoded as, from the form type, the compression
    /// four-CC and `sampleSize`.
    pub codec: AiffCodec,
    /// Which form type the file declared: `AIFF` or `AIFC`.
    pub form: AiffForm,
    /// The compression four-CC exactly as declared, `NONE` for a plain
    /// `AIFF` form, which has no field to read it from.
    pub compression: FourCc,
    /// `sampleSize` as declared.
    ///
    /// For the companded types this is recorded and not dispatched on:
    /// writers put both the wire width and the decoded width here, and the
    /// stride is one byte either way.
    pub bits_per_sample: u16,
    /// `numSampleFrames` as declared.
    ///
    /// Not advisory: the `SSND` chunk is required to hold exactly this many
    /// frames, and a file where the two disagree is rejected rather than
    /// silently resolved in either direction.
    pub sample_frames: u32,
}

impl AiffFormat {
    /// How many bytes one whole frame occupies in the payload.
    ///
    /// At least one: the channel count is rejected at zero, so this never
    /// divides by nothing.
    pub const fn frame_bytes(&self) -> usize {
        self.spec.channels as usize * self.codec.bytes_per_sample()
    }

    /// How many bytes of sample data `numSampleFrames` requires.
    fn data_bytes(&self) -> u64 {
        u64::from(self.sample_frames) * self.frame_bytes() as u64
    }
}

// -- Reading the `COMM` chunk -------------------------------------------------

/// Rejects a `COMM` chunk whose declared size the format cannot mean.
///
/// Called on both paths with the same numbers so the whole-file and streaming
/// readers agree: on the whole-file path the body is already bounded by the
/// input, and on the streaming path this is what bounds the buffer.
fn check_comm_size(form: AiffForm, size: u64, offset: u64) -> Result<(), DecodeError> {
    let minimum = match form {
        AiffForm::Aiff => COMM_BYTES_AIFF,
        AiffForm::Aifc => COMM_BYTES_AIFC,
    };
    if size < minimum {
        return Err(DecodeError::Malformed {
            expected: "a COMM chunk of at least 18 bytes, 22 in AIFF-C",
            offset,
        });
    }
    if size > MAX_COMM_BYTES {
        return Err(DecodeError::Malformed {
            expected: "a COMM chunk of at most 278 bytes, which is all a compressionName can need",
            offset,
        });
    }
    Ok(())
}

/// Reads a `COMM` chunk body that has already been checked with
/// [`check_comm_size`].
///
/// `offset` is where the chunk *header* sits in the input, so the offsets in
/// any [`DecodeError::Malformed`] this returns are absolute.
fn parse_comm(body: &[u8], offset: u64, form: AiffForm) -> Result<AiffFormat, DecodeError> {
    let body_at = offset + riff::CHUNK_HEADER_BYTES;
    let (Some(channels), Some(sample_frames), Some(bits_per_sample), Some(rate_field)) = (
        riff::u16_be_at(body, 0),
        riff::u32_be_at(body, 2),
        riff::u16_be_at(body, 6),
        body.get(8..18),
    ) else {
        return Err(DecodeError::Malformed {
            expected: "a COMM chunk of at least 18 bytes, 22 in AIFF-C",
            offset,
        });
    };

    if channels == 0 {
        return Err(DecodeError::UnsupportedChannelLayout { channels });
    }
    let mut rate_bytes = [0u8; 10];
    rate_bytes.copy_from_slice(rate_field);
    let sample_rate = rate_to_spec_rate(extended_to_f64(rate_bytes), body_at + 8)?;

    let compression = match form {
        AiffForm::Aiff => NONE,
        AiffForm::Aifc => {
            let Some(compression) = riff::four_cc_at(body, 18) else {
                return Err(DecodeError::Malformed {
                    expected: "a COMM chunk of at least 18 bytes, 22 in AIFF-C",
                    offset,
                });
            };
            // The compressionName pascal string: a length byte, that many
            // characters, padded to an even total. It is the last field in
            // the chunk, so nothing needs its value, but a length byte
            // counting past the chunk's end is a miscounted chunk, and a
            // parser that shrugged here would be trusting a length it never
            // checked.
            if let Some(&name_len) = body.get(22) {
                if 23 + usize::from(name_len) > body.len() {
                    return Err(DecodeError::Malformed {
                        expected: "a compressionName that fits its COMM chunk",
                        offset: body_at + 22,
                    });
                }
            }
            compression
        }
    };

    Ok(AiffFormat {
        spec: AudioSpec::new(sample_rate, channels),
        codec: AiffCodec::resolve(compression, bits_per_sample)?,
        form,
        compression,
        bits_per_sample,
        sample_frames,
    })
}

/// Reads the twelve-byte `FORM` header and answers which form type it is.
///
/// # Errors
///
/// [`DecodeError::Truncated`] for an input shorter than twelve bytes, and
/// [`DecodeError::UnsupportedContainer`] naming the bytes seen for a magic
/// that is not `FORM` or a form type that is neither `AIFF` nor `AIFC`.
fn read_form_header(bytes: &[u8]) -> Result<AiffForm, DecodeError> {
    let (Some(magic), Some(form)) = (riff::four_cc_at(bytes, 0), riff::four_cc_at(bytes, 8)) else {
        return Err(DecodeError::Truncated {
            expected: riff::HEADER_BYTES,
            available: bytes.len() as u64,
        });
    };
    if magic != FORM {
        return Err(DecodeError::UnsupportedContainer { tag: magic });
    }
    match form {
        _ if form == AIFF => Ok(AiffForm::Aiff),
        _ if form == AIFC => Ok(AiffForm::Aifc),
        _ => Err(DecodeError::UnsupportedContainer { tag: form }),
    }
}

/// Validates an `SSND` body against the `COMM` that describes it, and returns
/// the sample data region's start within the body.
///
/// This is where the two sources of truth meet. `available` is what the chunk
/// actually holds past its prefix and its `offset` padding; `needed` is what
/// `numSampleFrames` requires. They must be equal: fewer is
/// [`DecodeError::Truncated`] naming both numbers, more is
/// [`DecodeError::Malformed`], and neither is silently resolved.
fn check_ssnd(
    format: &AiffFormat,
    body_len: u64,
    data_offset: u32,
    chunk_offset: u64,
) -> Result<(), DecodeError> {
    if body_len < SSND_PREFIX_BYTES {
        return Err(DecodeError::Malformed {
            expected: "an SSND chunk of at least 8 bytes",
            offset: chunk_offset,
        });
    }
    // u64 throughout: offset is a u32 the file chose, and adding it to the
    // prefix in u32 could wrap on a crafted value.
    let data_start = SSND_PREFIX_BYTES + u64::from(data_offset);
    if data_start > body_len {
        return Err(DecodeError::Malformed {
            expected: "an SSND offset that fits inside the SSND chunk",
            offset: chunk_offset,
        });
    }
    let available = body_len - data_start;
    let needed = format.data_bytes();
    if available < needed {
        return Err(DecodeError::Truncated {
            expected: needed,
            available,
        });
    }
    if available > needed {
        return Err(DecodeError::Malformed {
            expected: "an SSND chunk holding exactly numSampleFrames frames",
            offset: chunk_offset,
        });
    }
    Ok(())
}

// -- The whole-file reader ----------------------------------------------------

/// An AIFF or AIFF-C file held whole in memory.
///
/// [`new`](Self::new) does all the parsing and all the rejecting; a reader
/// that exists is a file that has been fully validated, which is why
/// [`decode_to_end`](Self::decode_to_end) returns an [`AudioBuffer`] rather
/// than a `Result`, the same property, for the same payload encodings, as
/// [`WavReader`](crate::WavReader).
///
/// # Example
///
/// ```
/// use decibri_decode::{AiffCodec, AiffReader, AudioSpec};
///
/// // A minimal AIFF built by hand: 8 kHz mono signed 8-bit, three frames.
/// let mut file = Vec::new();
/// file.extend_from_slice(b"FORM");
/// file.extend_from_slice(&50u32.to_be_bytes());
/// file.extend_from_slice(b"AIFF");
/// file.extend_from_slice(b"COMM");
/// file.extend_from_slice(&18u32.to_be_bytes());
/// file.extend_from_slice(&1u16.to_be_bytes()); // channels
/// file.extend_from_slice(&3u32.to_be_bytes()); // frames
/// file.extend_from_slice(&8u16.to_be_bytes()); // bits
/// // 8000 as an 80-bit extended float.
/// file.extend_from_slice(&[0x40, 0x0B, 0xFA, 0, 0, 0, 0, 0, 0, 0]);
/// file.extend_from_slice(b"SSND");
/// file.extend_from_slice(&11u32.to_be_bytes());
/// file.extend_from_slice(&[0; 8]); // offset, blockSize
/// file.extend_from_slice(&[0x00, 0x40, 0x80]); // 0.0, 0.5, -1.0
/// file.push(0); // the pad byte after an odd chunk
///
/// let reader = AiffReader::new(&file)?;
/// assert_eq!(reader.format().codec, AiffCodec::PcmI8);
/// assert_eq!(reader.spec(), AudioSpec::mono(8_000));
/// assert_eq!(reader.frames(), 3);
/// assert_eq!(reader.decode_to_end().samples(), [0.0, 0.5, -1.0]);
/// # Ok::<(), decibri_decode::DecodeError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AiffReader<'a> {
    format: AiffFormat,
    data: &'a [u8],
}

impl<'a> AiffReader<'a> {
    /// Parses `bytes` as an AIFF or AIFF-C file.
    ///
    /// `COMM` and `SSND` may arrive in either order and unknown chunks
    /// anywhere are skipped, `FVER` included. The walk stops once both have
    /// been found, so trailing chunks are not examined.
    ///
    /// **"Either order" includes `SSND` before `COMM`, which
    /// [`AiffStreamDecoder`] cannot accept**: a file that opens here can
    /// fail when streamed, for the reason recorded on the module.
    ///
    /// A file whose `COMM` declares zero frames may omit `SSND` entirely, as
    /// the specification allows, and decodes to nothing.
    ///
    /// # Errors
    ///
    /// - [`DecodeError::Truncated`] for an input under twelve bytes, a chunk
    ///   declaring more than the input holds, and an `SSND` holding fewer
    ///   bytes than `numSampleFrames` requires.
    /// - [`DecodeError::UnsupportedContainer`] for a magic that is not
    ///   `FORM` and a form type that is neither `AIFF` nor `AIFC`.
    /// - [`DecodeError::Malformed`] for a missing or structurally wrong
    ///   `COMM`, a missing `SSND` when frames are declared, an `SSND` whose
    ///   `offset` does not fit its chunk or which holds more than
    ///   `numSampleFrames` requires, a zero or non-integral sample rate, and
    ///   a compressionName that overruns its chunk.
    /// - [`DecodeError::UnsupportedChannelLayout`] for a zero channel count.
    /// - [`DecodeError::UnsupportedCodec`] and
    ///   [`DecodeError::UnsupportedSampleFormat`] from
    ///   [`AiffCodec::resolve`].
    pub fn new(bytes: &'a [u8]) -> Result<Self, DecodeError> {
        let form = read_form_header(bytes)?;

        let mut walker = ChunkWalker::new(bytes, ByteOrder::Big);
        let mut comm: Option<(AiffFormat, u64)> = None;
        let mut ssnd: Option<(&[u8], u64)> = None;
        while comm.is_none() || ssnd.is_none() {
            let Some(chunk) = walker.next().transpose()? else {
                break;
            };
            if chunk.id == COMM && comm.is_none() {
                check_comm_size(form, chunk.body.len() as u64, chunk.offset)?;
                comm = Some((parse_comm(chunk.body, chunk.offset, form)?, chunk.offset));
            } else if chunk.id == SSND && ssnd.is_none() {
                ssnd = Some((chunk.body, chunk.offset));
            }
        }

        let Some((format, _)) = comm else {
            return Err(DecodeError::Malformed {
                expected: "a COMM chunk",
                offset: riff::HEADER_BYTES,
            });
        };
        let Some((body, chunk_offset)) = ssnd else {
            // The specification lets a zero-frame file omit SSND, and a
            // zero-frame file with no SSND holds exactly the audio it
            // declares: none.
            if format.sample_frames == 0 {
                return Ok(Self { format, data: &[] });
            }
            return Err(DecodeError::Malformed {
                expected: "an SSND chunk",
                offset: riff::HEADER_BYTES,
            });
        };

        let data_offset = riff::u32_be_at(body, 0).unwrap_or(0);
        check_ssnd(&format, body.len() as u64, data_offset, chunk_offset)?;
        let data_start = SSND_PREFIX_BYTES as usize + data_offset as usize;
        Ok(Self {
            format,
            data: &body[data_start..],
        })
    }

    /// What the `COMM` chunk declared and resolved to.
    pub const fn format(&self) -> &AiffFormat {
        &self.format
    }

    /// The rate and layout the payload decodes to.
    pub const fn spec(&self) -> AudioSpec {
        self.format.spec
    }

    /// The sample data, undecoded: the `SSND` body past its prefix and its
    /// `offset` padding.
    pub const fn data(&self) -> &'a [u8] {
        self.data
    }

    /// How many whole frames the payload holds.
    ///
    /// Exact and doubly attested: the payload's real length was required to
    /// equal `numSampleFrames * frame size` at parse time, so the declaration
    /// and the bytes agree by construction.
    pub const fn frames(&self) -> u64 {
        self.data.len() as u64 / self.format.frame_bytes() as u64
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
    /// The reservation is from the payload's real length, never from a size
    /// the file declared.
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
    /// Waiting for the twelve-byte `FORM` header.
    Form,
    /// Waiting for an eight-byte chunk header.
    ChunkHeader,
    /// Buffering the body of the `COMM` chunk, the one body that has to be
    /// read.
    CommBody { size: usize, offset: u64 },
    /// Buffering `SSND`'s eight-byte `offset`/`blockSize` prefix.
    SsndPrefix { size: u64, offset: u64 },
    /// Discarding the `offset` bytes of alignment padding before the sample
    /// data. Nothing is buffered here.
    SsndGap { size: u64, left: u64 },
    /// Discarding the body of a chunk that does not have to be read.
    Skip { size: u64, left: u64 },
    /// Discarding the one pad byte after an odd-length chunk.
    Pad,
    /// Streaming the sample data.
    Data,
    /// Past the end of the audio. Everything after it is discarded.
    Done,
}

/// Reads an AIFF or AIFF-C file that arrives in pieces.
///
/// The [`StreamSource`] half of this module, with the same shape and the same
/// bounds as [`WavStreamDecoder`](crate::WavStreamDecoder): bytes are pushed
/// in whatever sizes they turn up in, samples are pulled out, and nothing is
/// buffered in proportion to a size the file declared. Chunk headers, the
/// `COMM` body (capped at the format's own 278-byte ceiling) and `SSND`'s
/// eight-byte prefix are buffered, every other chunk body is discarded as it
/// flows past, and the payload decodes into a bounded ready buffer.
///
/// # `COMM` has to come first
///
/// [`AiffReader`] accepts `SSND` before `COMM`; this does not, and answers
/// [`DecodeError::Malformed`] when it meets one, for the reason recorded on
/// the module: decoding a payload that arrives before the header describing
/// it would mean holding all of it. Every other chunk order, and unknown
/// chunks in any position, are the same on both paths.
///
/// # Example
///
/// ```
/// use decibri_decode::{AiffStreamDecoder, AudioSpec, StreamSource};
///
/// // The same minimal file as the `AiffReader` example.
/// let mut file = Vec::new();
/// file.extend_from_slice(b"FORM");
/// file.extend_from_slice(&50u32.to_be_bytes());
/// file.extend_from_slice(b"AIFF");
/// file.extend_from_slice(b"COMM");
/// file.extend_from_slice(&18u32.to_be_bytes());
/// file.extend_from_slice(&1u16.to_be_bytes());
/// file.extend_from_slice(&3u32.to_be_bytes());
/// file.extend_from_slice(&8u16.to_be_bytes());
/// file.extend_from_slice(&[0x40, 0x0B, 0xFA, 0, 0, 0, 0, 0, 0, 0]);
/// file.extend_from_slice(b"SSND");
/// file.extend_from_slice(&11u32.to_be_bytes());
/// file.extend_from_slice(&[0; 8]);
/// file.extend_from_slice(&[0x00, 0x40, 0x80]);
/// file.push(0);
///
/// let mut stream = AiffStreamDecoder::new();
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
/// assert_eq!(stream.spec(), Some(AudioSpec::mono(8_000)));
/// assert_eq!(samples, [0.0, 0.5, -1.0]);
/// # Ok::<(), decibri_decode::DecodeError>(())
/// ```
#[derive(Debug)]
pub struct AiffStreamDecoder {
    state: State,
    /// Bytes of a header, the `COMM` body or the `SSND` prefix that have not
    /// fully arrived. Bounded by [`MAX_COMM_BYTES`].
    pending: Vec<u8>,
    /// Byte offset of the next byte to arrive, for error reporting.
    offset: u64,
    /// The form type, once the header has arrived.
    form: Option<AiffForm>,
    format: Option<AiffFormat>,
    payload: Option<Payload>,
    /// Decoded samples not yet pulled, and how far into them the caller is.
    ready: Vec<f32>,
    ready_read: usize,
    /// The sample data's size, and how much of it is still to arrive.
    data_size: u64,
    data_left: u64,
    finished: bool,
}

impl Default for AiffStreamDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl AiffStreamDecoder {
    /// A reader waiting for the first byte of a file.
    pub fn new() -> Self {
        Self {
            state: State::Form,
            pending: Vec::new(),
            offset: 0,
            form: None,
            format: None,
            payload: None,
            ready: Vec::new(),
            ready_read: 0,
            data_size: 0,
            data_left: 0,
            finished: false,
        }
    }

    /// What the `COMM` chunk declared, once it has arrived.
    pub const fn format(&self) -> Option<AiffFormat> {
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

    /// Reads the twelve-byte `FORM` header out of `pending`.
    fn start_file(&mut self) -> Result<(), DecodeError> {
        self.form = Some(read_form_header(&self.pending)?);
        self.pending.clear();
        self.state = State::ChunkHeader;
        Ok(())
    }

    /// Reads an eight-byte chunk header out of `pending` and decides what to
    /// do with the body.
    fn start_chunk(&mut self) -> Result<(), DecodeError> {
        let offset = self.offset - riff::CHUNK_HEADER_BYTES;
        let (Some(id), Some(declared)) = (
            riff::four_cc_at(&self.pending, 0),
            riff::u32_be_at(&self.pending, 4),
        ) else {
            return Err(DecodeError::Truncated {
                expected: riff::CHUNK_HEADER_BYTES,
                available: self.pending.len() as u64,
            });
        };
        self.pending.clear();
        let size = u64::from(declared);
        // Set at construction and at every header since, so this is always
        // present here; answered rather than unwrapped because a panic on
        // untrusted input is the failure class this crate refuses.
        let form = self.form.unwrap_or(AiffForm::Aiff);

        if id == COMM && self.format.is_none() {
            check_comm_size(form, size, offset)?;
            self.state = State::CommBody {
                size: size as usize,
                offset,
            };
            return Ok(());
        }

        if id == SSND {
            if self.format.is_none() {
                // The one thing the streaming path cannot do that the
                // whole-file path can.
                return Err(DecodeError::Malformed {
                    expected: "a COMM chunk before the SSND chunk",
                    offset,
                });
            }
            if size < SSND_PREFIX_BYTES {
                return Err(DecodeError::Malformed {
                    expected: "an SSND chunk of at least 8 bytes",
                    offset,
                });
            }
            self.state = State::SsndPrefix { size, offset };
            return Ok(());
        }

        // FVER and every unknown chunk: skipped, on both paths alike.
        self.state = if size == 0 {
            State::ChunkHeader
        } else {
            State::Skip { size, left: size }
        };
        Ok(())
    }

    /// Reads a buffered `COMM` body out of `pending`.
    fn finish_comm(&mut self, size: usize, offset: u64) -> Result<(), DecodeError> {
        let form = self.form.unwrap_or(AiffForm::Aiff);
        self.format = Some(parse_comm(&self.pending, offset, form)?);
        self.pending.clear();
        self.state = if riff::pad_len(size as u64) == 1 {
            State::Pad
        } else {
            State::ChunkHeader
        };
        Ok(())
    }

    /// Reads `SSND`'s buffered eight-byte prefix out of `pending` and lines
    /// up the gap, the data and the decoder.
    fn finish_ssnd_prefix(&mut self, size: u64, offset: u64) -> Result<(), DecodeError> {
        let Some(format) = self.format else {
            // Guarded at start_chunk; kept as an answer rather than a panic.
            return Err(DecodeError::Malformed {
                expected: "a COMM chunk before the SSND chunk",
                offset,
            });
        };
        let data_offset = riff::u32_be_at(&self.pending, 0).unwrap_or(0);
        self.pending.clear();
        check_ssnd(&format, size, data_offset, offset)?;

        self.payload = Some(Payload::from_parts(
            format.codec.sample_format(),
            format.codec.law(),
            format.spec,
        ));
        self.data_size = format.data_bytes();
        self.data_left = self.data_size;

        let gap = u64::from(data_offset);
        self.state = if gap > 0 {
            State::SsndGap {
                size: gap,
                left: gap,
            }
        } else if self.data_size == 0 {
            State::Done
        } else {
            State::Data
        };
        Ok(())
    }

    /// Moves decoded samples out of the payload decoder.
    fn drain_payload(&mut self) -> Result<(), DecodeError> {
        if let Some(payload) = self.payload.as_mut() {
            payload.decode(&mut self.ready)?;
        }
        Ok(())
    }
}

impl StreamSource for AiffStreamDecoder {
    fn push(&mut self, bytes: &[u8]) -> Result<usize, DecodeError> {
        if self.finished {
            return Ok(0);
        }
        let mut taken = 0;
        let result = self.push_inner(bytes, &mut taken);
        if result.is_err() {
            // A stream that has failed structurally is over, for the reason
            // recorded on the WAV reader: a caller who keeps pushing must not
            // get a second, different answer from the same file.
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
        // Amortised compaction, exactly as the WAV reader does it.
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

        // Structural completeness first: an item half-arrived is Truncated,
        // and a file that ended on a chunk boundary without carrying the
        // audio it declared is Malformed. The one clean early end is a
        // zero-frame file, whose SSND the specification lets it omit.
        match self.state {
            State::Done => {}
            State::ChunkHeader
                if self.pending.is_empty()
                    && self.format.is_some_and(|format| format.sample_frames == 0) => {}
            State::Form => {
                return Err(DecodeError::Truncated {
                    expected: riff::HEADER_BYTES,
                    available: self.pending.len() as u64,
                })
            }
            State::Pad | State::ChunkHeader if self.pending.is_empty() => {
                return Err(DecodeError::Malformed {
                    expected: if self.format.is_some() {
                        "an SSND chunk"
                    } else {
                        "a COMM chunk"
                    },
                    offset: self.offset,
                })
            }
            State::ChunkHeader | State::Pad => {
                return Err(DecodeError::Truncated {
                    expected: riff::CHUNK_HEADER_BYTES,
                    available: self.pending.len() as u64,
                })
            }
            State::CommBody { size, .. } => {
                return Err(DecodeError::Truncated {
                    expected: size as u64,
                    available: self.pending.len() as u64,
                })
            }
            State::SsndPrefix { .. } => {
                return Err(DecodeError::Truncated {
                    expected: SSND_PREFIX_BYTES,
                    available: self.pending.len() as u64,
                })
            }
            State::SsndGap { size, left } | State::Skip { size, left } => {
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

impl AiffStreamDecoder {
    /// The body of [`push`](StreamSource::push), split out so a failure can
    /// set the finished flag in one place.
    fn push_inner(&mut self, bytes: &[u8], taken: &mut usize) -> Result<(), DecodeError> {
        while *taken < bytes.len() {
            let rest = &bytes[*taken..];
            match self.state {
                State::Form => {
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
                State::CommBody { size, offset } => {
                    if !self.accumulate(rest, size, taken) {
                        break;
                    }
                    self.finish_comm(size, offset)?;
                }
                State::SsndPrefix { size, offset } => {
                    if !self.accumulate(rest, SSND_PREFIX_BYTES as usize, taken) {
                        break;
                    }
                    self.finish_ssnd_prefix(size, offset)?;
                }
                State::SsndGap { size, left } => {
                    let step = left.min(rest.len() as u64);
                    *taken += step as usize;
                    self.offset += step;
                    let left = left - step;
                    self.state = if left == 0 {
                        if self.data_size == 0 {
                            State::Done
                        } else {
                            State::Data
                        }
                    } else {
                        State::SsndGap { size, left }
                    };
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
                    // Trailing chunks are not this reader's business; the
                    // audio it was promised is complete.
                    self.offset += (bytes.len() - *taken) as u64;
                    *taken = bytes.len();
                }
            }
        }
        Ok(())
    }
}

// -- The writer ---------------------------------------------------------------

/// The `COMM` body a written plain `AIFF` file carries: the four fixed
/// fields, nothing else, because the form has no compression field.
const COMM_WRITE_AIFF: u64 = COMM_BYTES_AIFF;

/// The `COMM` body a written `AIFC` file carries: the fixed fields, the
/// compression four-CC, and a zero-length compressionName pascal string:
/// its length byte plus the one pad byte that takes the string to an even
/// total. The name is empty because it is display text the specification
/// does not constrain, and an empty name is one fewer thing to keep
/// byte-identical across releases.
const COMM_WRITE_AIFC: u64 = COMM_BYTES_AIFC + 2;

/// The whole `FVER` chunk a written `AIFC` file carries: header plus the
/// four-byte timestamp.
const FVER_CHUNK_BYTES: u64 = riff::CHUNK_HEADER_BYTES + 4;

/// The value of the `FORM` header's size field for a written file:
/// everything after the first eight bytes. `None` when that does not fit in
/// 64 bits, which no caller can reach but the arithmetic should not pretend
/// about, the same posture as the WAV writer's `riff_size`.
fn aiff_form_size(form: AiffForm, data_bytes: u64) -> Option<u64> {
    let mut size: u64 = 4; // the form type
    let comm_bytes = match form {
        AiffForm::Aiff => COMM_WRITE_AIFF,
        AiffForm::Aifc => {
            size = size.checked_add(FVER_CHUNK_BYTES)?;
            COMM_WRITE_AIFC
        }
    };
    size = size
        .checked_add(riff::CHUNK_HEADER_BYTES)?
        .checked_add(comm_bytes)?;
    let ssnd_body = SSND_PREFIX_BYTES.checked_add(data_bytes)?;
    size = size
        .checked_add(riff::CHUNK_HEADER_BYTES)?
        .checked_add(ssnd_body)?
        // The pad byte after an odd SSND body belongs to the enclosing form
        // even though it is outside the chunk's own declared size.
        .checked_add(riff::pad_len(ssnd_body))?;
    Some(size)
}

/// Writes AIFF and AIFF-C files.
///
/// The same shape as [`WavWriter`](crate::WavWriter): construct with a spec
/// and a codec, then [`write`](Self::write) or [`to_bytes`](Self::to_bytes)
/// so that the two are learnable as a pair. It writes every encoding
/// [`AiffReader`] accepts: [`AiffCodec`] has no variant that is readable and
/// not writable.
///
/// # Which form type a file gets
///
/// The rule lives on [`AiffCodec::form`] and the writer follows it: plain
/// `AIFF` for big-endian two's-complement PCM, which is the only thing that
/// form can carry and the more widely readable of the two; `AIFC`, with the
/// `FVER` chunk and compression field the specification requires, for
/// everything else. There is no override, because forcing `AIFF` around an
/// encoding it cannot name would write a file that lies about its payload.
///
/// # No RF64 equivalent
///
/// RIFF outgrows its 32-bit size fields into RF64; AIFF has no such escape
/// hatch. A write whose `FORM` size would not fit 32 bits is refused with a
/// typed error rather than written with wrapped sizes, unlike
/// [`WavWriter`](crate::WavWriter), which upgrades instead of refusing,
/// because it has somewhere to upgrade to.
///
/// # Samples outside full scale, and samples that are not finite
///
/// **The integer encodings clamp and the float encodings do not.** A sample
/// outside `-1.0..=1.0` written to an integer or G.711 encoding becomes the
/// extreme value rather than wrapping, an infinity becomes the same extreme,
/// and a NaN becomes silence. A sample written to [`AiffCodec::Float32`] or
/// [`AiffCodec::Float64`] is written through as it is, overshoot, infinities
/// and NaN included. A count of clipped samples is therefore a number about
/// an integer encoding and not about a float one.
///
/// Read back through this crate, a NaN written to [`AiffCodec::Float32`]
/// returns the same bit pattern, and a NaN written to [`AiffCodec::Float64`]
/// returns silence, because narrowing a NaN from `f64` to `f32` normalises
/// it. Both infinities return unchanged at either width.
///
/// # Example
///
/// ```
/// use decibri_decode::{AiffCodec, AiffReader, AiffWriter, AudioSpec};
///
/// let file = AiffWriter::new(AudioSpec::new(44_100, 2), AiffCodec::PcmI24)
///     .to_bytes(&[0.5, -0.5, 0.25, -0.25])?;
///
/// let reader = AiffReader::new(&file)?;
/// assert_eq!(reader.format().codec, AiffCodec::PcmI24);
/// assert_eq!(reader.frames(), 2);
/// assert_eq!(reader.decode_to_end().samples(), [0.5, -0.5, 0.25, -0.25]);
/// # Ok::<(), decibri_decode::DecodeError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AiffWriter {
    spec: AudioSpec,
    codec: AiffCodec,
}

impl AiffWriter {
    /// A writer for `codec` audio at `spec`.
    pub const fn new(spec: AudioSpec, codec: AiffCodec) -> Self {
        Self { spec, codec }
    }

    /// The rate and layout written into the `COMM` chunk.
    pub const fn spec(&self) -> AudioSpec {
        self.spec
    }

    /// The encoding the payload is written in.
    pub const fn codec(&self) -> AiffCodec {
        self.codec
    }

    /// The form type this writer's files declare, from [`AiffCodec::form`].
    ///
    /// Exposed so a caller can see which of the two forms an encoding
    /// selects without writing a file to find out.
    pub const fn form(&self) -> AiffForm {
        self.codec.form()
    }

    /// Appends a complete file to `output`, and returns how many bytes it
    /// appended.
    ///
    /// `output` is appended to, never cleared, the same convention as
    /// everything else in this crate that writes into a caller's buffer.
    ///
    /// `numSampleFrames` and the `SSND` length agree by construction: both
    /// are computed from the one frame count `samples` implies, so the
    /// agreement the readers enforce cannot be broken from here. `SSND`'s
    /// `offset` and `blockSize` are written, as zero: present because the
    /// fields are not optional, zero because nothing here needs block
    /// alignment.
    ///
    /// # Errors
    ///
    /// - [`DecodeError::UnsupportedChannelLayout`] for no channels.
    /// - [`DecodeError::Malformed`] for a zero sample rate, and for audio
    ///   whose file would exceed AIFF's 32-bit size fields. The `FORM` size
    ///   is checked in 64-bit arithmetic and a file that would wrap it is
    ///   refused, because AIFF has no RF64 to upgrade into. That one check
    ///   also bounds `numSampleFrames`: the payload holds at least one byte
    ///   per frame, so a frame count past `u32::MAX` cannot reach the field
    ///   that would truncate it.
    /// - [`DecodeError::Truncated`] when `samples` does not divide into
    ///   whole frames, for the reason recorded on
    ///   [`WavWriter::write`](crate::WavWriter::write): writing a partial
    ///   frame would produce a file this crate's own readers reject.
    pub fn write(&self, samples: &[f32], output: &mut Vec<u8>) -> Result<usize, DecodeError> {
        let channels = usize::from(self.spec.channels);
        if channels == 0 {
            return Err(DecodeError::UnsupportedChannelLayout {
                channels: self.spec.channels,
            });
        }
        let form = self.codec.form();
        if self.spec.sample_rate == 0 {
            // The offset names where the unencodable rate field would sit:
            // after the header, the AIFC form's FVER chunk, the COMM header
            // and COMM's first eight bytes.
            let comm_at = riff::HEADER_BYTES
                + match form {
                    AiffForm::Aiff => 0,
                    AiffForm::Aifc => FVER_CHUNK_BYTES,
                };
            return Err(DecodeError::Malformed {
                expected: "a non-zero sample rate",
                offset: comm_at + riff::CHUNK_HEADER_BYTES + 8,
            });
        }
        let bytes_per_sample = self.codec.bytes_per_sample();
        let frame_bytes = channels * bytes_per_sample;
        let leftover = samples.len() % channels;
        if leftover != 0 {
            return Err(DecodeError::Truncated {
                expected: frame_bytes as u64,
                available: (leftover * bytes_per_sample) as u64,
            });
        }

        let data_bytes = (samples.len() as u64).checked_mul(bytes_per_sample as u64);
        let declared_form = data_bytes.and_then(|data| aiff_form_size(form, data));
        let (Some(data_bytes), Some(declared_form)) = (data_bytes, declared_form) else {
            return Err(size_refusal());
        };
        if declared_form > u64::from(u32::MAX) {
            return Err(size_refusal());
        }
        // In range now: the payload holds at least one byte per frame, so
        // `frames <= data_bytes <= declared_form <= u32::MAX`.
        let frames = (samples.len() / channels) as u32;

        let start = output.len();
        output.extend_from_slice(FORM.as_bytes());
        output.extend_from_slice(&(declared_form as u32).to_be_bytes());
        let comm_bytes = match form {
            AiffForm::Aiff => {
                output.extend_from_slice(AIFF.as_bytes());
                COMM_WRITE_AIFF
            }
            AiffForm::Aifc => {
                output.extend_from_slice(AIFC.as_bytes());
                output.extend_from_slice(FVER.as_bytes());
                output.extend_from_slice(&4u32.to_be_bytes());
                output.extend_from_slice(&FVER_TIMESTAMP.to_be_bytes());
                COMM_WRITE_AIFC
            }
        };

        output.extend_from_slice(COMM.as_bytes());
        output.extend_from_slice(&(comm_bytes as u32).to_be_bytes());
        output.extend_from_slice(&self.spec.channels.to_be_bytes());
        output.extend_from_slice(&frames.to_be_bytes());
        output.extend_from_slice(&self.codec.bits_per_sample().to_be_bytes());
        output.extend_from_slice(&rate_to_extended(self.spec.sample_rate));
        if form == AiffForm::Aifc {
            output.extend_from_slice(self.codec.compression_type().as_bytes());
            // The zero-length compressionName: a length byte of zero, then
            // the pad byte that takes the pascal string to an even total.
            output.extend_from_slice(&[0, 0]);
        }

        output.extend_from_slice(SSND.as_bytes());
        output.extend_from_slice(&((SSND_PREFIX_BYTES + data_bytes) as u32).to_be_bytes());
        output.extend_from_slice(&0u32.to_be_bytes()); // offset
        output.extend_from_slice(&0u32.to_be_bytes()); // blockSize
        self.codec.encode(samples, output);
        if riff::pad_len(SSND_PREFIX_BYTES + data_bytes) == 1 {
            // The IFF pad byte after an odd chunk body, outside the chunk's
            // declared size and inside the form's.
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

/// The one refusal for audio AIFF's 32-bit size fields cannot hold.
///
/// A function rather than a constant expression at each site so the message
/// exists once; the offset names the `FORM` size field the number would not
/// fit.
fn size_refusal() -> DecodeError {
    DecodeError::Malformed {
        expected: "audio that fits within AIFF's 32-bit size fields",
        offset: 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Gate 7: the 80-bit extended float, against hand-computed bytes ---

    /// Every byte string here was worked out by hand from the format
    /// definition, sign, 15-bit biased exponent, explicit-integer-bit
    /// significand, not produced by an encoder that could share a mistake
    /// with the parser.
    #[test]
    fn the_extended_float_parser_matches_hand_computed_values() {
        // 8000 = 0.9765625 * 2^13: exponent 16383+12 = 0x400B with the
        // significand normalised to its integer bit, 8000 << 51 = 0xFA00...
        let cases: [([u8; 10], f64); 7] = [
            ([0x40, 0x0B, 0xFA, 0, 0, 0, 0, 0, 0, 0], 8_000.0),
            ([0x40, 0x0C, 0xFA, 0, 0, 0, 0, 0, 0, 0], 16_000.0),
            // 44100 = 0xAC44 * 2^0; normalised: exponent 16383+15, 0xAC44 << 48.
            ([0x40, 0x0E, 0xAC, 0x44, 0, 0, 0, 0, 0, 0], 44_100.0),
            ([0x40, 0x0E, 0xBB, 0x80, 0, 0, 0, 0, 0, 0], 48_000.0),
            ([0x40, 0x0D, 0xAC, 0x44, 0, 0, 0, 0, 0, 0], 22_050.0),
            // A non-zero fractional part: 11025.5 = 22051 * 2^-1, and 22051
            // = 0x5623 normalises to 0xAC46 << 48 at exponent 16383+13.
            ([0x40, 0x0C, 0xAC, 0x46, 0, 0, 0, 0, 0, 0], 11_025.5),
            // One: integer bit alone at exponent 16383.
            ([0x3F, 0xFF, 0x80, 0, 0, 0, 0, 0, 0, 0], 1.0),
        ];
        for (bytes, expected) in cases {
            assert_eq!(
                extended_to_f64(bytes),
                expected,
                "{bytes:02X?} did not parse to {expected}"
            );
        }
    }

    #[test]
    fn the_extended_float_parser_handles_the_edges_deterministically() {
        // Zero, and negative zero.
        assert_eq!(extended_to_f64([0; 10]), 0.0);
        assert_eq!(extended_to_f64([0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 0.0);
        // A negative rate parses as negative; rejecting it is the caller's
        // job, not the parser's.
        assert_eq!(
            extended_to_f64([0xC0, 0x0B, 0xFA, 0, 0, 0, 0, 0, 0, 0]),
            -8_000.0
        );
        // Infinity: the 0x7FFF exponent with a clear fraction.
        assert_eq!(
            extended_to_f64([0x7F, 0xFF, 0x80, 0, 0, 0, 0, 0, 0, 0]),
            f64::INFINITY
        );
        assert_eq!(
            extended_to_f64([0xFF, 0xFF, 0x80, 0, 0, 0, 0, 0, 0, 0]),
            f64::NEG_INFINITY
        );
        // NaN: the same exponent with fraction bits set.
        assert!(extended_to_f64([0x7F, 0xFF, 0xC0, 0, 0, 0, 0, 0, 0, 1]).is_nan());
        // An exponent past f64's range overflows to infinity rather than
        // wrapping or panicking.
        assert_eq!(
            extended_to_f64([0x7F, 0xFE, 0x80, 0, 0, 0, 0, 0, 0, 0]),
            f64::INFINITY
        );
        // And a denormal underflows to zero rather than misparsing.
        assert_eq!(extended_to_f64([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), 0.0);
    }

    #[test]
    fn a_rate_the_spec_cannot_carry_is_rejected_with_its_offset() {
        for wrong in [
            0.0,
            -8_000.0,
            11_025.5,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            f64::from(u32::MAX) + 1.0,
            0.25,
        ] {
            let error = rate_to_spec_rate(wrong, 20).expect_err("must reject");
            assert!(
                matches!(
                    error,
                    DecodeError::Malformed {
                        expected: "a positive integral sample rate that fits in 32 bits",
                        offset: 20
                    }
                ),
                "{wrong}: unexpected error: {error}"
            );
        }
        for right in [1.0, 8_000.0, 44_100.0, 192_000.0, f64::from(u32::MAX)] {
            assert_eq!(rate_to_spec_rate(right, 20).expect("accept"), right as u32);
        }
    }

    // -- Dispatch on the (compression, sampleSize) pair -------------------

    #[test]
    fn dispatch_is_on_the_compression_and_the_width_together() {
        use AiffCodec::*;
        let accepted: [(&[u8; 4], u16, AiffCodec); 16] = [
            (b"NONE", 8, PcmI8),
            (b"raw ", 8, PcmU8),
            (b"NONE", 16, PcmI16),
            (b"NONE", 24, PcmI24),
            (b"NONE", 32, PcmI32),
            (b"twos", 8, PcmI8),
            (b"twos", 16, PcmI16),
            (b"sowt", 8, PcmI8),
            (b"sowt", 16, PcmI16Sowt),
            (b"sowt", 24, PcmI24Sowt),
            (b"sowt", 32, PcmI32Sowt),
            (b"fl32", 32, Float32),
            (b"FL32", 32, Float32),
            (b"fl64", 64, Float64),
            (b"FL64", 64, Float64),
            (b"alaw", 16, ALaw),
        ];
        for (compression, bits, codec) in accepted {
            assert_eq!(
                AiffCodec::resolve(FourCc(*compression), bits).expect("accepted"),
                codec,
                "{compression:?} at {bits} bits"
            );
        }

        // The companded types dispatch on the four-CC alone: both the wire
        // width and the decoded width appear in real files.
        for law_cc in [b"alaw", b"ALAW"] {
            for bits in [8u16, 16] {
                assert_eq!(
                    AiffCodec::resolve(FourCc(*law_cc), bits).expect("alaw"),
                    ALaw
                );
            }
        }
        for law_cc in [b"ulaw", b"ULAW"] {
            for bits in [8u16, 16] {
                assert_eq!(
                    AiffCodec::resolve(FourCc(*law_cc), bits).expect("ulaw"),
                    MuLaw
                );
            }
        }

        // A carried compression at a width this crate does not read names
        // both halves of the rejection.
        for (compression, bits) in [
            (b"NONE", 12u16),
            (b"NONE", 64),
            (b"sowt", 20),
            (b"fl32", 16),
            (b"fl64", 32),
            // `raw ` is the one unsigned encoding and exists at one width.
            // Any other width is a width rejection, never a silent accept.
            (b"raw ", 1),
            (b"raw ", 16),
            (b"raw ", 24),
            (b"raw ", 32),
        ] {
            let error = AiffCodec::resolve(FourCc(*compression), bits).expect_err("must reject");
            assert!(
                matches!(
                    &error,
                    DecodeError::UnsupportedSampleFormat { format, bits_per_sample }
                        if *format == CodecId::FourCc(FourCc(*compression))
                            && *bits_per_sample == bits
                ),
                "{compression:?} at {bits}: unexpected error: {error}"
            );
        }

        // A compression this crate does not carry names the four-CC it was.
        // Each of these five is a compression scheme, so carrying one would
        // mean a decoder for it. `raw ` was in this list until 0.1.2 and is
        // not a compression scheme; it is asserted above, accepted at 8 bits
        // and rejected on width everywhere else.
        for compression in [b"ima4", b"MAC3", b"MAC6", b"QDMC", b"GSM "] {
            let error = AiffCodec::resolve(FourCc(*compression), 16).expect_err("must reject");
            assert!(
                matches!(&error, DecodeError::UnsupportedCodec { codec }
                    if *codec == CodecId::FourCc(FourCc(*compression))),
                "{compression:?}: unexpected error: {error}"
            );
        }
    }

    #[test]
    fn every_codec_is_linear_or_companded_and_never_both() {
        const ALL: [AiffCodec; 12] = [
            AiffCodec::PcmI8,
            AiffCodec::PcmU8,
            AiffCodec::PcmI16,
            AiffCodec::PcmI24,
            AiffCodec::PcmI32,
            AiffCodec::PcmI16Sowt,
            AiffCodec::PcmI24Sowt,
            AiffCodec::PcmI32Sowt,
            AiffCodec::Float32,
            AiffCodec::Float64,
            AiffCodec::ALaw,
            AiffCodec::MuLaw,
        ];
        // The exhaustive match is the assertion: a variant added to the enum
        // without a line in ALL fails to compile here.
        for codec in ALL {
            match codec {
                AiffCodec::PcmI8
                | AiffCodec::PcmU8
                | AiffCodec::PcmI16
                | AiffCodec::PcmI24
                | AiffCodec::PcmI32
                | AiffCodec::PcmI16Sowt
                | AiffCodec::PcmI24Sowt
                | AiffCodec::PcmI32Sowt
                | AiffCodec::Float32
                | AiffCodec::Float64
                | AiffCodec::ALaw
                | AiffCodec::MuLaw => {}
            }
            assert_ne!(
                codec.sample_format().is_some(),
                codec.law().is_some(),
                "{codec:?} is on both sides or neither"
            );
            if let Some(format) = codec.sample_format() {
                assert_eq!(format.bytes_per_sample(), codec.bytes_per_sample());
            } else {
                assert_eq!(codec.bytes_per_sample(), 1);
            }
        }
    }

    // -- COMM parsing -----------------------------------------------------

    fn comm_aiff(channels: u16, frames: u32, bits: u16, rate: [u8; 10]) -> Vec<u8> {
        let mut body = Vec::new();
        body.extend_from_slice(&channels.to_be_bytes());
        body.extend_from_slice(&frames.to_be_bytes());
        body.extend_from_slice(&bits.to_be_bytes());
        body.extend_from_slice(&rate);
        body
    }

    const RATE_8K: [u8; 10] = [0x40, 0x0B, 0xFA, 0, 0, 0, 0, 0, 0, 0];

    #[test]
    fn a_comm_chunk_parses_to_what_it_declared() {
        let body = comm_aiff(2, 100, 16, RATE_8K);
        let format = parse_comm(&body, 12, AiffForm::Aiff).expect("parse");
        assert_eq!(format.spec, AudioSpec::new(8_000, 2));
        assert_eq!(format.codec, AiffCodec::PcmI16);
        assert_eq!(format.form, AiffForm::Aiff);
        assert_eq!(format.compression, NONE);
        assert_eq!(format.bits_per_sample, 16);
        assert_eq!(format.sample_frames, 100);
        assert_eq!(format.frame_bytes(), 4);
    }

    #[test]
    fn a_compression_name_that_overruns_its_chunk_is_malformed() {
        let mut body = comm_aiff(1, 4, 16, RATE_8K);
        body.extend_from_slice(b"sowt");
        // The length byte claims eleven characters; only four follow.
        body.push(11);
        body.extend_from_slice(b"four");
        let error = parse_comm(&body, 12, AiffForm::Aifc).expect_err("must reject");
        assert!(
            matches!(
                error,
                DecodeError::Malformed {
                    expected: "a compressionName that fits its COMM chunk",
                    ..
                }
            ),
            "unexpected error: {error}"
        );

        // The same name, counted truthfully, parses.
        let mut body = comm_aiff(1, 4, 16, RATE_8K);
        body.extend_from_slice(b"sowt");
        body.push(4);
        body.extend_from_slice(b"four");
        body.push(0); // the pstring's own even-total pad
        let format = parse_comm(&body, 12, AiffForm::Aifc).expect("parse");
        assert_eq!(format.codec, AiffCodec::PcmI16Sowt);
    }

    #[test]
    fn zero_channels_and_impossible_rates_are_typed_rejections() {
        let body = comm_aiff(0, 4, 16, RATE_8K);
        assert!(matches!(
            parse_comm(&body, 12, AiffForm::Aiff),
            Err(DecodeError::UnsupportedChannelLayout { channels: 0 })
        ));

        for bad_rate in [
            [0u8; 10],                                  // zero
            [0xC0, 0x0B, 0xFA, 0, 0, 0, 0, 0, 0, 0],    // negative
            [0x40, 0x0C, 0xAC, 0x46, 0, 0, 0, 0, 0, 0], // 11025.5
            [0x7F, 0xFF, 0x80, 0, 0, 0, 0, 0, 0, 0],    // infinity
            [0x7F, 0xFF, 0xC0, 0, 0, 0, 0, 0, 0, 1],    // NaN
        ] {
            let body = comm_aiff(1, 4, 16, bad_rate);
            let error = parse_comm(&body, 12, AiffForm::Aiff).expect_err("must reject");
            assert!(
                matches!(
                    error,
                    DecodeError::Malformed {
                        expected: "a positive integral sample rate that fits in 32 bits",
                        offset: 28
                    }
                ),
                "{bad_rate:02X?}: unexpected error: {error}"
            );
        }
    }

    // -- The SSND agreement rule ------------------------------------------

    #[test]
    fn the_ssnd_length_must_equal_what_comm_declared() {
        let format = parse_comm(&comm_aiff(2, 10, 16, RATE_8K), 12, AiffForm::Aiff).expect("parse");
        // 10 frames * 4 bytes: the body needs exactly 8 + 40 bytes.
        assert!(check_ssnd(&format, 48, 0, 30).is_ok());
        // Under: Truncated, naming both numbers.
        let error = check_ssnd(&format, 40, 0, 30).expect_err("under");
        assert!(
            matches!(
                error,
                DecodeError::Truncated {
                    expected: 40,
                    available: 32
                }
            ),
            "unexpected error: {error}"
        );
        // Over: Malformed, not a silent preference for either source.
        let error = check_ssnd(&format, 50, 0, 30).expect_err("over");
        assert!(
            matches!(
                error,
                DecodeError::Malformed {
                    expected: "an SSND chunk holding exactly numSampleFrames frames",
                    offset: 30
                }
            ),
            "unexpected error: {error}"
        );
        // A non-zero offset moves the data and the arithmetic follows it.
        assert!(check_ssnd(&format, 52, 4, 30).is_ok());
        // An offset past the chunk cannot wrap the arithmetic into a pass.
        let error = check_ssnd(&format, 48, u32::MAX, 30).expect_err("offset");
        assert!(
            matches!(
                error,
                DecodeError::Malformed {
                    expected: "an SSND offset that fits inside the SSND chunk",
                    ..
                }
            ),
            "unexpected error: {error}"
        );
    }

    // -- The 80-bit extended float, on the encode side --------------------

    /// Every byte string here was worked out by hand from the format
    /// definition, the same derivations as the parser's gate above, so the
    /// two directions are anchored to the same independent bytes rather
    /// than to each other.
    #[test]
    fn the_extended_float_encoder_matches_hand_computed_values() {
        let cases: [(u32, [u8; 10]); 5] = [
            // 8000 = 0x1F40, leading one at bit 12: exponent 16383+12 =
            // 0x400B, significand 8000 << 51 = 0xFA00...
            (8_000, [0x40, 0x0B, 0xFA, 0, 0, 0, 0, 0, 0, 0]),
            // 16000: one bit higher, same significand pattern.
            (16_000, [0x40, 0x0C, 0xFA, 0, 0, 0, 0, 0, 0, 0]),
            // 22050 = 0x5622, leading one at bit 14: exponent 16383+14 =
            // 0x400D, significand 0x5622 << 49 = 0xAC44...
            (22_050, [0x40, 0x0D, 0xAC, 0x44, 0, 0, 0, 0, 0, 0]),
            // 44100 = 0xAC44, leading one at bit 15: exponent 0x400E,
            // significand 0xAC44 << 48.
            (44_100, [0x40, 0x0E, 0xAC, 0x44, 0, 0, 0, 0, 0, 0]),
            // 48000 = 0xBB80, leading one at bit 15: exponent 0x400E,
            // significand 0xBB80 << 48.
            (48_000, [0x40, 0x0E, 0xBB, 0x80, 0, 0, 0, 0, 0, 0]),
        ];
        for (rate, expected) in cases {
            assert_eq!(
                rate_to_extended(rate),
                expected,
                "{rate} did not encode to {expected:02X?}"
            );
        }
    }

    /// The encoder round-trips through the existing parser for every rate
    /// shape: the common rates, a dense low sweep, every power of two, and
    /// the extremes of the field.
    #[test]
    fn the_extended_float_encoder_round_trips_through_the_parser() {
        let common = [
            8_000u32, 11_025, 16_000, 22_050, 32_000, 44_100, 48_000, 88_200, 96_000, 176_400,
            192_000,
        ];
        let sweep = 1..=1_000;
        let powers = (0..32).map(|shift| 1u32 << shift);
        let extremes = [3u32, u32::MAX - 1, u32::MAX];
        for rate in common
            .into_iter()
            .chain(sweep)
            .chain(powers)
            .chain(extremes)
        {
            let parsed = extended_to_f64(rate_to_extended(rate));
            assert_eq!(parsed, f64::from(rate), "{rate} did not round-trip");
            // And the parsed value passes the same acceptance the reader
            // applies, landing on the same u32.
            assert_eq!(rate_to_spec_rate(parsed, 0).expect("integral"), rate);
        }
    }

    // -- The writer's size arithmetic -------------------------------------

    /// The refusal point, tested as arithmetic rather than by materialising
    /// four gigabytes to cross it with, the same posture as the WAV
    /// writer's RF64 threshold test.
    #[test]
    fn the_size_refusal_happens_exactly_at_the_32_bit_form_limit() {
        let ceiling = u64::from(u32::MAX);

        // Plain AIFF over an 18-byte COMM: 4 (form type) + 8 + 18 (COMM)
        // + 8 (SSND header) + 8 (SSND prefix) = 46 bytes the payload has to
        // share u32::MAX with.
        const AIFF_OVERHEAD: u64 = 46;
        assert_eq!(aiff_form_size(AiffForm::Aiff, 0), Some(AIFF_OVERHEAD));
        // `u32::MAX - 46` is odd, so its pad byte is what tips it over: the
        // largest fitting payload is one byte smaller and even.
        let largest = ceiling - AIFF_OVERHEAD - 1;
        assert_eq!(largest % 2, 0);
        assert_eq!(aiff_form_size(AiffForm::Aiff, largest), Some(ceiling - 1));
        assert!(aiff_form_size(AiffForm::Aiff, largest).unwrap() <= ceiling);
        // One byte more is odd, pads, and lands one past the ceiling.
        assert_eq!(
            aiff_form_size(AiffForm::Aiff, largest + 1),
            Some(ceiling + 1)
        );
        // The odd payload one byte *smaller* fits only because its pad byte
        // still lands on ceiling - 1: the pad is genuinely counted.
        assert_eq!(
            aiff_form_size(AiffForm::Aiff, largest - 1),
            Some(ceiling - 1)
        );

        // AIFC adds FVER (12) and the longer COMM (24 + 8 against 18 + 8):
        // 64 bytes of overhead.
        const AIFC_OVERHEAD: u64 = 64;
        assert_eq!(aiff_form_size(AiffForm::Aifc, 0), Some(AIFC_OVERHEAD));
        let largest = ceiling - AIFC_OVERHEAD - 1;
        assert_eq!(largest % 2, 0);
        assert_eq!(aiff_form_size(AiffForm::Aifc, largest), Some(ceiling - 1));
        assert_eq!(
            aiff_form_size(AiffForm::Aifc, largest + 1),
            Some(ceiling + 1)
        );

        // And the arithmetic answers None rather than wrapping at the top of
        // u64 itself.
        assert_eq!(aiff_form_size(AiffForm::Aiff, u64::MAX), None);
        assert_eq!(aiff_form_size(AiffForm::Aifc, u64::MAX - 20), None);

        // The refusal the writer hands back for everything past the limit.
        let error = size_refusal();
        assert!(
            matches!(
                error,
                DecodeError::Malformed {
                    expected: "audio that fits within AIFF's 32-bit size fields",
                    offset: 4
                }
            ),
            "unexpected error: {error}"
        );
    }

    // -- The writer's rejections ------------------------------------------

    #[test]
    fn the_writer_rejects_a_partial_trailing_frame() {
        let writer = AiffWriter::new(AudioSpec::new(48_000, 3), AiffCodec::PcmI16);
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
        let no_channels = AiffWriter::new(AudioSpec::new(48_000, 0), AiffCodec::PcmI16);
        assert!(matches!(
            no_channels.to_bytes(&[]),
            Err(DecodeError::UnsupportedChannelLayout { channels: 0 })
        ));

        // A zero rate is refused before the encoder that cannot spell it,
        // and the offset names the rate field of the form being written:
        // byte 28 in plain AIFF, byte 40 in AIFC behind FVER.
        for (codec, rate_at) in [(AiffCodec::PcmI16, 28), (AiffCodec::Float32, 40)] {
            let no_rate = AiffWriter::new(AudioSpec::mono(0), codec);
            assert!(matches!(
                no_rate.to_bytes(&[]),
                Err(DecodeError::Malformed {
                    expected: "a non-zero sample rate",
                    offset
                }) if offset == rate_at
            ));
        }
    }

    #[test]
    fn the_writer_appends_rather_than_clearing() {
        let writer = AiffWriter::new(AudioSpec::mono(8_000), AiffCodec::PcmI8);
        let mut output = vec![0xAA];
        let written = writer.write(&[0.0, 0.0], &mut output).expect("write");
        assert_eq!(output[0], 0xAA);
        assert_eq!(output.len(), 1 + written);
        // Header, COMM, SSND header and prefix, two bytes of payload.
        assert_eq!(written, 12 + 8 + 18 + 8 + 8 + 2);
    }

    // -- The AIFF against AIFC selection rule, every branch ---------------

    /// Each codec lands on the form the rule states, and the written bytes
    /// agree with the answer: the form type at bytes 8..12, and FVER present
    /// exactly when the form is AIFC.
    #[test]
    fn each_codec_selects_the_form_the_rule_states() {
        let cases: [(AiffCodec, AiffForm); 12] = [
            (AiffCodec::PcmI8, AiffForm::Aiff),
            (AiffCodec::PcmU8, AiffForm::Aifc),
            (AiffCodec::PcmI16, AiffForm::Aiff),
            (AiffCodec::PcmI24, AiffForm::Aiff),
            (AiffCodec::PcmI32, AiffForm::Aiff),
            (AiffCodec::PcmI16Sowt, AiffForm::Aifc),
            (AiffCodec::PcmI24Sowt, AiffForm::Aifc),
            (AiffCodec::PcmI32Sowt, AiffForm::Aifc),
            (AiffCodec::Float32, AiffForm::Aifc),
            (AiffCodec::Float64, AiffForm::Aifc),
            (AiffCodec::ALaw, AiffForm::Aifc),
            (AiffCodec::MuLaw, AiffForm::Aifc),
        ];
        for (codec, expected_form) in cases {
            let writer = AiffWriter::new(AudioSpec::new(16_000, 2), codec);
            assert_eq!(writer.form(), expected_form, "{codec:?}");
            let bytes = writer.to_bytes(&[0.0, 0.0]).expect("write");
            let form_type = &bytes[8..12];
            let has_fver = bytes.windows(4).any(|window| window == b"FVER");
            match expected_form {
                AiffForm::Aiff => {
                    assert_eq!(form_type, b"AIFF", "{codec:?}");
                    assert!(!has_fver, "{codec:?} wrote FVER into a plain AIFF");
                }
                AiffForm::Aifc => {
                    assert_eq!(form_type, b"AIFC", "{codec:?}");
                    assert!(has_fver, "{codec:?} omitted the FVER chunk");
                }
            }
            // And the file reads back under the codec it was written as.
            let reader = AiffReader::new(&bytes).expect("read back");
            assert_eq!(reader.format().codec, codec, "{codec:?}");
            assert_eq!(reader.format().form, expected_form, "{codec:?}");
        }
    }
}
