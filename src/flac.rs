//! FLAC: the third container this crate reads, and the first that is a codec
//! as well as a container.
//!
//! Written from [RFC 9639](https://www.rfc-editor.org/rfc/rfc9639), which
//! specifies the format in full, including the numerical considerations that
//! decide the integer widths below and worked decoding examples that this
//! crate carries as literal test vectors. No other implementation's source
//! was consulted, deliberately: the point of a specification complete enough
//! to implement from is that the resulting code has no provenance question
//! attached to it.
//!
//! # The oracle is inside the data
//!
//! Every FLAC file carries an MD5 of its own unencoded audio in the
//! streaminfo metadata block. That makes FLAC the one format in this crate
//! whose correctness can be checked on **every file the decoder ever sees**,
//! not only on the ones in a test suite, and this crate checks it: a
//! whole-file decode hashes as it goes and a streaming decode hashes
//! incrementally, and both compare at the end. A mismatch is
//! [`DecodeError::Malformed`] naming the checksum field. The specification
//! permits an all-zero checksum, which means "not known"; that is not a
//! failure and is not treated as one.
//!
//! The hash is MD5 because the format says MD5. See [`md5`](crate::md5) for
//! why substituting something stronger would remove the check rather than
//! improve it.
//!
//! # The five ways a FLAC decoder is silently wrong
//!
//! Roughly in order of how likely each is to produce wrong audio rather than
//! an error, which is the failure class this crate exists to avoid:
//!
//! - **Wasted bits.** A subframe may declare that every sample has some
//!   number of low-order zero bits removed. Parsing the field and forgetting
//!   to shift them back produces audio quieter by a power of two, with no
//!   error anywhere. It is applied in [`decode_subframe`], **before** stereo
//!   decorrelation is undone, which is the order RFC 9639 section 9.2.2
//!   requires.
//! - **Stereo decorrelation.** Left/side, right/side and mid/side each
//!   reconstruct differently, and mid/side carries an odd-sample correction
//!   that is easy to drop: an odd side sample means the mid sample lost a
//!   least-significant bit to the right shift, and it is restored from the
//!   side channel's low bit. See [`undo_decorrelation`].
//! - **Integer width.** The format permits 32-bit audio, a side channel
//!   needs 33 bits before reconstruction, and a linear prediction sums up to
//!   32 products of a 33-bit sample and a 15-bit coefficient, 53 bits by
//!   RFC 9639 appendix A.3's arithmetic. Every sample value and every
//!   intermediate here is `i64`. A 32-bit intermediate would overflow only
//!   on loud passages of deep material, which no small test file contains.
//! - **The Rice escape code.** An all-ones Rice parameter means the
//!   partition holds unencoded residuals at a stated width rather than
//!   Rice-coded ones. Treating it as an ordinary parameter produces garbage
//!   for that partition alone. See [`decode_residual`].
//! - **The coded number.** A variable-length code that looks like UTF-8 and
//!   is extended to 36 bits, so a UTF-8 routine cannot read it. It is
//!   decoded by hand in [`read_coded_number`].
//!
//! # Both CRCs are verified
//!
//! The frame header carries a CRC-8 and the whole frame a CRC-16. Both are
//! checked before any sample from that frame is delivered, and a mismatch is
//! [`DecodeError::Malformed`] rather than audio. A decoder that decodes
//! through a bad CRC is guessing.
//!
//! # A size in a file is a claim, not a fact
//!
//! The same discipline as [`wav`](crate::wav) and [`aiff`](crate::aiff), with
//! one honest difference recorded rather than glossed. A FLAC frame declares
//! its own sample count and a decoder must hold that many samples to
//! reconstruct them, so unlike the two PCM containers there is an allocation
//! that follows a number from the file. It is bounded twice: by the format,
//! whose largest legal frame is 65535 samples in 8 channels, and by the
//! streaminfo maximum block size, which RFC 9639 section 8.2 requires every
//! frame to respect and which this reader enforces. Nothing is ever
//! allocated from the declared *frame* size, the metadata block sizes or the
//! total sample count.
//!
//! # No rate policy, and no property changes mid-stream
//!
//! Each frame header restates the sample rate, channel count and bit depth,
//! and the format allows them to differ from frame to frame. This crate
//! rejects that with a typed error rather than decoding it, because
//! [`AudioBuffer`] carries exactly one [`AudioSpec`] for the samples it
//! holds and there is no honest way to describe a buffer whose rate changed
//! half way through. RFC 9639 section 9 explicitly permits a decoder to stop
//! on such a change and recommends a real container for such streams.
//!
//! # Two readers, and the one difference between them
//!
//! [`FlacReader`] holds the whole file. [`FlacStreamDecoder`] reads one that
//! arrives in pieces. Both drive the same frame decoder over the same bytes,
//! so their output is identical by construction rather than by agreement.
//! The difference is where the work happens: [`FlacReader::new`] validates
//! the metadata only, and the audio frames are decoded by
//! [`decode_to_end`](FlacReader::decode_to_end), which therefore returns a
//! `Result` where [`WavReader`](crate::WavReader) and
//! [`AiffReader`](crate::AiffReader) return the buffer directly. Those two
//! carry linear PCM, which cannot fail once its header has parsed; a FLAC
//! frame can.
//!
//! # And one writer
//!
//! [`FlacWriter`] writes native FLAC streams at compression levels 0
//! through 8. It lives in this module because it must: residuals are
//! computed by subtracting the very prediction functions the decoder adds,
//! [`fixed_prediction`] and [`lpc_prediction`], so that the two directions
//! cannot disagree about the arithmetic. What the writer searches and what
//! it guarantees are recorded on the type itself.

use std::ops::Range;

use crate::audio::{AudioBuffer, AudioSpec};
use crate::codec::{CodecId, FourCc};
use crate::error::DecodeError;
use crate::md5::Md5;
use crate::sample::quantize;
use crate::source::StreamSource;

/// The four bytes every FLAC stream starts with.
///
/// Taken from [`probe`](crate::probe) rather than written again here, because
/// the probe has to recognise a FLAC stream in a build that does not compile
/// this module and the signature is one fact.
const MAGIC: FourCc = crate::probe::FLAC_MAGIC;

/// How many bytes a metadata block header occupies: one flag-and-type byte
/// and a 24-bit big-endian length.
const METADATA_HEADER_BYTES: usize = 4;

/// The streaminfo metadata block type.
const BLOCK_STREAMINFO: u8 = 0;

/// The metadata block type RFC 9639 section 8.1 forbids, so that a block
/// header can never be mistaken for a frame sync code.
const BLOCK_FORBIDDEN: u8 = 127;

/// How many bytes a streaminfo block body occupies. Fixed by the format, so
/// a block declaring any other length is malformed rather than merely
/// unexpected.
const STREAMINFO_BYTES: usize = 34;

/// The smallest block size the format allows, except in a stream's last
/// frame.
const MIN_BLOCK_SIZE: u32 = 16;

/// The largest block size the format allows.
const MAX_BLOCK_SIZE: u32 = 65_535;

/// The most channels a FLAC frame can carry.
const MAX_CHANNELS: u8 = 8;

/// The highest linear predictor order the format defines.
const MAX_LPC_ORDER: usize = 32;

/// The smallest frame header, in bytes: four fixed bytes, a one-byte coded
/// number and the CRC-8.
const MIN_FRAME_HEADER_BYTES: usize = 6;

/// The unencoded size of the largest frame the format permits, in bytes.
///
/// 65535 samples in 8 channels at 32 bits each, stored verbatim. A stereo
/// frame's side subframe carries one extra bit, but two channels at 33 bits
/// is far short of eight at 32, so this is the maximum across every legal
/// shape.
const MAX_VERBATIM_FRAME_BYTES: usize = MAX_BLOCK_SIZE as usize * MAX_CHANNELS as usize * (32 / 8);

/// The largest frame this crate will read, in bytes.
///
/// [`MAX_VERBATIM_FRAME_BYTES`] plus four kilobytes, which is forty times
/// what a frame header, eight subframe headers and the two-byte footer can
/// occupy together. A frame larger than this cannot be one an encoder should
/// have written: RFC 9639 section 9.2.7.3 requires an encoder that cannot
/// code a subframe within the residual range to fall back to a verbatim
/// subframe, so a conforming frame never exceeds its own unencoded size by
/// more than its headers.
///
/// This is what bounds the streaming reader's buffer. It is a number stated
/// here, derived from the format's own limits, and never read from a file.
const MAX_FRAME_BYTES: usize = MAX_VERBATIM_FRAME_BYTES + 4_096;

/// How many decoded samples the streaming reader holds before it stops
/// taking bytes. The same figure, for the same reason, as every other
/// bounded reader in the crate.
const READY_LIMIT: usize = 65_536;

/// How many bytes of interleaved samples are staged before being handed to
/// the MD5 computation.
///
/// A fixed buffer rather than one frame's worth, so that hashing costs the
/// same on a file with 65535-sample frames as on one with 16-sample frames
/// and adds nothing to the reader's allocation ceiling.
const MD5_STAGE_BYTES: usize = 4_096;

// -- CRC ----------------------------------------------------------------------

/// The CRC-8 lookup table for the polynomial `x^8 + x^2 + x^1 + x^0`, which
/// RFC 9639 section 9.1.8 specifies for the frame header.
const CRC8: [u8; 256] = crc8_table();

/// The CRC-16 lookup table for the polynomial `x^16 + x^15 + x^2 + x^0`,
/// which RFC 9639 section 9.3 specifies for the whole frame.
const CRC16: [u16; 256] = crc16_table();

/// Builds [`CRC8`] at compile time from the polynomial itself.
///
/// The table is derived rather than transcribed so there is one place the
/// polynomial appears and no 256 constants for a typo to hide in.
const fn crc8_table() -> [u8; 256] {
    let mut table = [0u8; 256];
    let mut index = 0;
    while index < 256 {
        let mut crc = index as u8;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 0x80 != 0 {
                (crc << 1) ^ 0x07
            } else {
                crc << 1
            };
            bit += 1;
        }
        table[index] = crc;
        index += 1;
    }
    table
}

/// Builds [`CRC16`] at compile time from the polynomial itself.
const fn crc16_table() -> [u16; 256] {
    let mut table = [0u16; 256];
    let mut index = 0;
    while index < 256 {
        let mut crc = (index as u16) << 8;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 0x8000 != 0 {
                (crc << 1) ^ 0x8005
            } else {
                crc << 1
            };
            bit += 1;
        }
        table[index] = crc;
        index += 1;
    }
    table
}

/// The CRC-8 of `bytes`, initialised with zero.
fn crc8(bytes: &[u8]) -> u8 {
    bytes
        .iter()
        .fold(0u8, |crc, byte| CRC8[usize::from(crc ^ byte)])
}

/// The CRC-16 of `bytes`, initialised with zero.
fn crc16(bytes: &[u8]) -> u16 {
    bytes.iter().fold(0u16, |crc, byte| {
        (crc << 8) ^ CRC16[usize::from((crc >> 8) as u8 ^ byte)]
    })
}

// -- The bit reader -----------------------------------------------------------

/// How many bits of the next 64 a window is guaranteed to expose.
///
/// A window is eight bytes read big-endian and shifted left by the bit
/// offset within the first of them, so up to seven bits fall off the top.
/// Every read below is of at most 33 bits, comfortably inside this.
const WINDOW_BITS: u32 = 57;

/// A most-significant-bit-first reader over a frame's bytes.
///
/// FLAC subframes are not byte aligned, and only the first subframe in a
/// frame starts on a byte boundary, so everything past the frame header is read
/// bit by bit. Every read is bounds-checked against the bytes that actually
/// arrived, never against a length the file declared.
struct BitReader<'a> {
    /// The frame's bytes, starting at its first byte.
    bytes: &'a [u8],
    /// How many bits have been consumed from the start of `bytes`.
    bit: usize,
    /// Where `bytes[0]` sits in the whole input, so a rejection can name an
    /// absolute offset.
    base: u64,
}

impl<'a> BitReader<'a> {
    /// A reader positioned at the first bit of `bytes`.
    fn new(bytes: &'a [u8], base: u64) -> Self {
        Self {
            bytes,
            bit: 0,
            base,
        }
    }

    /// How many bits have not been consumed.
    fn bits_left(&self) -> usize {
        (self.bytes.len() * 8).saturating_sub(self.bit)
    }

    /// The next 64 bits, left-aligned and zero-padded past the end of the
    /// input.
    ///
    /// Only the top [`WINDOW_BITS`] are meaningful; the rest may have been
    /// shifted out. Padding past the end is safe because every caller checks
    /// [`bits_left`](Self::bits_left) first.
    fn window(&self) -> u64 {
        let byte = self.bit >> 3;
        let shift = (self.bit & 7) as u32;
        let mut buffer = [0u8; 8];
        let take = self.bytes.len().saturating_sub(byte).min(8);
        buffer[..take].copy_from_slice(&self.bytes[byte..byte + take]);
        u64::from_be_bytes(buffer) << shift
    }

    /// The next `count` bits as an unsigned number. `count` is at most
    /// [`WINDOW_BITS`].
    fn read_bits(&mut self, count: u32) -> Result<u64, DecodeError> {
        debug_assert!((1..=WINDOW_BITS).contains(&count));
        if self.bits_left() < count as usize {
            return Err(self.truncated(count as usize));
        }
        let value = self.window() >> (64 - count);
        self.bit += count as usize;
        Ok(value)
    }

    /// The next `count` bits as a two's-complement signed number.
    fn read_signed(&mut self, count: u32) -> Result<i64, DecodeError> {
        let raw = self.read_bits(count)?;
        // Landing the value in the top of an i64 and shifting back
        // arithmetically sign-extends it without a branch or a mask.
        Ok(((raw << (64 - count)) as i64) >> (64 - count))
    }

    /// How many zero bits precede the next one bit, consuming both.
    fn read_unary(&mut self) -> Result<u32, DecodeError> {
        let mut zeros = 0u32;
        loop {
            let window = self.window();
            if window == 0 {
                // No one bit within reach. Step over the zeros that are
                // really there and look again.
                let step = self.bits_left().min(WINDOW_BITS as usize);
                if step == 0 {
                    return Err(self.truncated(1));
                }
                zeros = zeros.saturating_add(step as u32);
                self.bit += step;
                continue;
            }
            let run = window.leading_zeros() as usize;
            if run >= self.bits_left() {
                // The one bit is in the zero padding past the end, which
                // means it has not arrived.
                return Err(self.truncated(run + 1));
            }
            self.bit += run + 1;
            return Ok(zeros.saturating_add(run as u32));
        }
    }

    /// Steps forward to the next byte boundary, which RFC 9639 section 9.3
    /// requires before the frame footer.
    fn align_to_byte(&mut self) {
        self.bit = (self.bit + 7) & !7;
    }

    /// How many whole bytes have been consumed. Only meaningful when the
    /// reader is byte aligned.
    fn byte_position(&self) -> usize {
        self.bit >> 3
    }

    /// A rejection saying the frame needed more bytes than arrived.
    ///
    /// `expected` counts from the **start of the frame** rather than the
    /// start of the input, because the incomplete item is the frame. The
    /// streaming reader uses the number directly as "do not try again until
    /// this many bytes of this frame are held".
    fn truncated(&self, more_bits: usize) -> DecodeError {
        DecodeError::Truncated {
            expected: self.bit.saturating_add(more_bits).div_ceil(8) as u64,
            available: self.bytes.len() as u64,
        }
    }

    /// A rejection naming what the format required at the current position.
    fn malformed(&self, expected: &'static str) -> DecodeError {
        DecodeError::Malformed {
            expected,
            offset: self.base + (self.bit >> 3) as u64,
        }
    }
}

// -- Streaminfo ---------------------------------------------------------------

/// What a FLAC stream's streaminfo metadata block declares.
///
/// RFC 9639 section 8.2's fields, read as they were written. Two of them are
/// `Option` rather than the zero the format stores, because zero means "not
/// known" for both, and a caller that could not tell an unknown length from
/// an empty stream would be facing exactly the ambiguity the `Option` exists
/// to avoid: `Some(0)` is an empty stream, `None` is a length the stream did
/// not state.
///
/// `#[non_exhaustive]`: a consumer matching on it keeps a `..` rest pattern,
/// and building one from a block a container carried goes through
/// [`from_block`](Self::from_block).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct FlacStreamInfo {
    /// The smallest block size in the stream, in interchannel samples,
    /// excluding the last block.
    pub min_block_size: u16,
    /// The largest block size in the stream, in interchannel samples.
    ///
    /// Not advisory: RFC 9639 section 8.2 requires every frame to respect
    /// it, and this crate rejects a frame that does not, which is also what
    /// keeps the decode buffer bounded by something the format guarantees.
    pub max_block_size: u16,
    /// The smallest frame size in the stream, in bytes, or `None` when the
    /// field is the zero that means unknown.
    pub min_frame_size: Option<u32>,
    /// The largest frame size in the stream, in bytes, or `None` when the
    /// field is the zero that means unknown.
    ///
    /// Read and reported, and deliberately never allocated against.
    pub max_frame_size: Option<u32>,
    /// The rate and layout the stream decodes to.
    pub spec: AudioSpec,
    /// Bits per sample, between 4 and 32.
    pub bits_per_sample: u8,
    /// Total interchannel samples in the stream, or `None` when the field is
    /// the zero that means unknown.
    pub total_samples: Option<u64>,
    /// The MD5 of the stream's unencoded audio, or `None` when the field is
    /// the all-zero value that means unset.
    ///
    /// `None` is not a failure and does not weaken anything else: a file may
    /// legitimately omit it, in which case there is simply no oracle for that
    /// file and the CRCs remain.
    pub md5: Option<[u8; 16]>,
}

impl FlacStreamInfo {
    /// Parses the 34-byte **body** of a streaminfo metadata block, which is
    /// how a container supplies one out of band.
    ///
    /// This is the constructor for the Ogg and Matroska shape: the container
    /// carries the streaminfo block in its own codec-private header and hands
    /// over bare frames, so the block has to become a [`FlacStreamInfo`] for
    /// [`FlacFrameReader::with_stream_info`],
    /// [`FlacStreamDecoder::frames_with_stream_info`] or
    /// [`FlacRecovery::with_stream_info`] to take. The struct is
    /// `#[non_exhaustive]`, so this is the one way to build one without a
    /// whole stream in hand.
    ///
    /// The 34 bytes are RFC 9639 section 8.2's block body alone, **without**
    /// the four-byte metadata block header (`0x00` or `0x80`, then a 24-bit
    /// length of 34) in front of it. Where a caller finds them:
    ///
    /// - **Matroska** (the WebM and MKV FLAC mapping): `CodecPrivate` holds
    ///   what a FLAC file starts with, the `fLaC` signature and then the
    ///   metadata blocks, streaminfo required first. The body is bytes
    ///   `8..42` of `CodecPrivate`.
    /// - **Ogg** (the Ogg FLAC mapping): the first packet is a nine-byte
    ///   prologue (`0x7F`, `FLAC`, two version bytes, a two-byte header
    ///   packet count), the `fLaC` signature, then the streaminfo block. The
    ///   body is bytes `17..51`, the last 34 of the 51-byte packet.
    ///
    /// # Errors
    ///
    /// [`DecodeError::Malformed`] when the slice is not exactly 34 bytes,
    /// when either block size is below the format's minimum of 16 or the
    /// minimum exceeds the maximum, when the sample rate is zero, or when
    /// the bit depth is outside 4 through 32. Offsets in the error count
    /// from the start of the body.
    ///
    /// # Example
    ///
    /// RFC 9639 appendix D.3's worked example taken apart the way Matroska
    /// carries it: the streaminfo body from what would be `CodecPrivate`,
    /// the bare frame from the packets.
    ///
    /// ```
    /// use decibri_decode::{FlacFrameReader, FlacStreamInfo, Md5Check};
    ///
    /// # let file: [u8; 73] = [
    /// #     0x66, 0x4c, 0x61, 0x43, 0x80, 0x00, 0x00, 0x22, 0x10, 0x00, 0x10, 0x00,
    /// #     0x00, 0x00, 0x1f, 0x00, 0x00, 0x1f, 0x07, 0xd0, 0x00, 0x70, 0x00, 0x00,
    /// #     0x00, 0x18, 0xf8, 0xf9, 0xe3, 0x96, 0xf5, 0xcb, 0xcf, 0xc6, 0xdc, 0x80,
    /// #     0x7f, 0x99, 0x77, 0x90, 0x6b, 0x32, 0xff, 0xf8, 0x68, 0x02, 0x00, 0x17,
    /// #     0xe9, 0x44, 0x00, 0x4f, 0x6f, 0x31, 0x3d, 0x10, 0x47, 0xd2, 0x27, 0xcb,
    /// #     0x6d, 0x09, 0x08, 0x31, 0x45, 0x2b, 0xdc, 0x28, 0x22, 0x22, 0x80, 0x57,
    /// #     0xa3,
    /// # ];
    /// let info = FlacStreamInfo::from_block(&file[8..42])?;
    /// assert_eq!(info.spec.sample_rate, 32_000);
    /// assert_eq!(info.total_samples, Some(24));
    ///
    /// let mut samples = Vec::new();
    /// let report = FlacFrameReader::with_stream_info(&file[42..], info).decode(&mut samples)?;
    /// assert_eq!(report.samples, 24);
    /// // The block carries the MD5, so the bare frames were verified.
    /// assert_eq!(report.md5, Md5Check::Verified);
    /// # Ok::<(), decibri_decode::DecodeError>(())
    /// ```
    pub fn from_block(body: &[u8]) -> Result<Self, DecodeError> {
        parse_streaminfo(body, 0)
    }

    /// How many bytes one sample occupies once sign-extended to a whole
    /// number of bytes, which is the form RFC 9639 section 8.2 hashes.
    fn md5_bytes_per_sample(&self) -> usize {
        usize::from(self.bits_per_sample).div_ceil(8)
    }
}

/// Reads a streaminfo block body, which is always exactly
/// [`STREAMINFO_BYTES`] long.
///
/// `offset` is where the body sits in the input, so rejections name an
/// absolute position.
fn parse_streaminfo(body: &[u8], offset: u64) -> Result<FlacStreamInfo, DecodeError> {
    let malformed = |expected| DecodeError::Malformed { expected, offset };
    if body.len() != STREAMINFO_BYTES {
        return Err(malformed("a streaminfo block of exactly 34 bytes"));
    }
    let mut reader = BitReader::new(body, offset);

    let min_block_size = reader.read_bits(16)? as u16;
    let max_block_size = reader.read_bits(16)? as u16;
    let min_frame_size = reader.read_bits(24)? as u32;
    let max_frame_size = reader.read_bits(24)? as u32;
    let sample_rate = reader.read_bits(20)? as u32;
    let channels = reader.read_bits(3)? as u16 + 1;
    let bits_per_sample = reader.read_bits(5)? as u8 + 1;
    let total_samples = reader.read_bits(36)?;
    let mut md5 = [0u8; 16];
    md5.copy_from_slice(&body[18..34]);

    // RFC 9639 section 8.2: both block sizes are in 16..=65535 and the
    // minimum does not exceed the maximum.
    if u32::from(min_block_size) < MIN_BLOCK_SIZE || u32::from(max_block_size) < MIN_BLOCK_SIZE {
        return Err(malformed("a streaminfo block size of at least 16 samples"));
    }
    if min_block_size > max_block_size {
        return Err(malformed(
            "a streaminfo minimum block size no larger than the maximum",
        ));
    }
    // A rate of zero is legal in the format and means the content is not
    // audio along a time axis. This crate decodes audio, and an AudioSpec
    // carrying a rate of zero would be a wrong rate travelling with real
    // samples, the failure AudioSpec exists to prevent. Rejected, named.
    if sample_rate == 0 {
        return Err(malformed("a streaminfo sample rate above zero"));
    }
    if !(4..=32).contains(&bits_per_sample) {
        return Err(malformed("a streaminfo bit depth between 4 and 32"));
    }

    Ok(FlacStreamInfo {
        min_block_size,
        max_block_size,
        min_frame_size: (min_frame_size != 0).then_some(min_frame_size),
        max_frame_size: (max_frame_size != 0).then_some(max_frame_size),
        spec: AudioSpec::new(sample_rate, channels),
        bits_per_sample,
        total_samples: (total_samples != 0).then_some(total_samples),
        md5: (md5 != [0u8; 16]).then_some(md5),
    })
}

// -- The frame header ---------------------------------------------------------

/// How a stream divides its samples into frames.
///
/// RFC 9639 section 9.1: the bit that says which MUST NOT change during a
/// stream, and this crate enforces that.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Blocking {
    /// Every frame but the last holds the same number of samples, and the
    /// coded number is a frame number.
    Fixed,
    /// Frames may hold different numbers of samples, and the coded number is
    /// a sample number.
    Variable,
}

/// How a frame's subframes map onto its channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChannelAssignment {
    /// Every channel is coded on its own. The only legal form above two
    /// channels.
    Independent(u8),
    /// Two channels stored as left and (left - right).
    LeftSide,
    /// Two channels stored as (left - right) and right.
    SideRight,
    /// Two channels stored as ((left + right) >> 1) and (left - right).
    MidSide,
}

impl ChannelAssignment {
    /// How many channels the frame carries.
    const fn count(self) -> u8 {
        match self {
            Self::Independent(channels) => channels,
            Self::LeftSide | Self::SideRight | Self::MidSide => 2,
        }
    }

    /// Which subframe, if any, carries the side channel and therefore one
    /// extra bit of depth.
    const fn side_channel(self) -> Option<usize> {
        match self {
            Self::Independent(_) => None,
            Self::LeftSide | Self::MidSide => Some(1),
            Self::SideRight => Some(0),
        }
    }
}

/// A parsed frame header.
#[derive(Debug, Clone, Copy)]
struct FrameHeader {
    blocking: Blocking,
    /// Samples per channel in this frame.
    block_size: u32,
    /// The rate this frame declares, or `None` when it defers to streaminfo.
    sample_rate: Option<u32>,
    channels: ChannelAssignment,
    /// The bit depth this frame declares, or `None` when it defers to
    /// streaminfo.
    bits_per_sample: Option<u8>,
    /// The frame or sample number this frame carries, which
    /// [`Blocking`] says which of the two it is.
    ///
    /// Nothing in an ordinary decode reads it, since the samples arrive in order
    /// and a decoder that trusted it would be trusting a number in a file.
    /// [`FlacRecovery`] reads it, because after a gap it is the only thing in
    /// the stream that says *where* decoding resumed.
    coded_number: u64,
    /// How many bytes the header occupies, including the CRC-8.
    header_bytes: usize,
}

/// Reads the frame or sample number, which RFC 9639 section 9.1.5 codes in a
/// UTF-8-like variable-length form extended to 36 bits.
///
/// **Not UTF-8**, and not decodable by a UTF-8 routine: the encoding runs to
/// seven octets and codes numbers rather than characters. Written out by
/// hand for exactly that reason.
///
/// Overlong encodings, a value spelled in more octets than it needs, are
/// accepted. The specification's table gives a range per length but does not
/// say a decoder must reject a value outside its range, the number is not
/// used for anything here, and rejecting real files over a bookkeeping field
/// costs compatibility for nothing.
fn read_coded_number(reader: &mut BitReader<'_>) -> Result<u64, DecodeError> {
    let first = reader.read_bits(8)? as u8;
    let (continuations, mut value) = match first.leading_ones() {
        0 => (0, u64::from(first)),
        // 0b10xxxxxx is a continuation octet and cannot start a code.
        1 => return Err(reader.malformed("a coded number not starting with a continuation octet")),
        leading @ 2..=7 => {
            let payload = 7 - leading;
            (
                leading - 1,
                u64::from(first & ((1u8 << payload).wrapping_sub(1))),
            )
        }
        // 0b11111111 is not a length this encoding defines.
        _ => return Err(reader.malformed("a coded number of at most seven octets")),
    };
    for _ in 0..continuations {
        let octet = reader.read_bits(8)? as u8;
        if octet & 0xC0 != 0x80 {
            return Err(reader.malformed("a coded number continuation octet"));
        }
        value = (value << 6) | u64::from(octet & 0x3F);
    }
    Ok(value)
}

/// Reads a frame header from the start of `bytes` and verifies its CRC-8.
///
/// Every rejection here is structural: the sync code, a reserved value, the
/// reserved bit, or the CRC. Nothing is checked against streaminfo yet, so
/// this function is usable on a frame in isolation.
fn parse_frame_header(bytes: &[u8], base: u64) -> Result<FrameHeader, DecodeError> {
    if bytes.len() < MIN_FRAME_HEADER_BYTES {
        return Err(DecodeError::Truncated {
            expected: MIN_FRAME_HEADER_BYTES as u64,
            available: bytes.len() as u64,
        });
    }
    let mut reader = BitReader::new(bytes, base);

    // The 15-bit sync code, then the blocking strategy bit.
    if reader.read_bits(15)? != 0b111_1111_1111_1100 {
        return Err(reader.malformed("a frame sync code"));
    }
    let blocking = if reader.read_bits(1)? == 0 {
        Blocking::Fixed
    } else {
        Blocking::Variable
    };

    let block_size_bits = reader.read_bits(4)? as u8;
    let sample_rate_bits = reader.read_bits(4)? as u8;
    let channel_bits = reader.read_bits(4)? as u8;
    let bit_depth_bits = reader.read_bits(3)? as u8;
    if reader.read_bits(1)? != 0 {
        return Err(reader.malformed("a cleared reserved bit in the frame header"));
    }

    let channels = match channel_bits {
        0..=7 => ChannelAssignment::Independent(channel_bits + 1),
        8 => ChannelAssignment::LeftSide,
        9 => ChannelAssignment::SideRight,
        10 => ChannelAssignment::MidSide,
        _ => return Err(reader.malformed("a defined channel assignment")),
    };
    let bits_per_sample = match bit_depth_bits {
        0 => None,
        1 => Some(8),
        2 => Some(12),
        4 => Some(16),
        5 => Some(20),
        6 => Some(24),
        7 => Some(32),
        // 0b011 is reserved.
        _ => return Err(reader.malformed("a defined bit depth")),
    };
    let block_size = match block_size_bits {
        0 => return Err(reader.malformed("a block size other than the reserved 0b0000")),
        1 => Some(192),
        2..=5 => Some(144u32 << block_size_bits),
        // 0b0110 and 0b0111 store the real value after the coded number.
        6 | 7 => None,
        _ => Some(1u32 << block_size_bits),
    };

    // The coded number comes before the uncommon block size and sample rate,
    // which is the one ordering trap in the header layout.
    let coded_number = read_coded_number(&mut reader)?;

    let block_size = match block_size {
        Some(size) => size,
        None => {
            let width = if block_size_bits == 6 { 8 } else { 16 };
            let stored = reader.read_bits(width)? as u32;
            // 65535 codes for a block size of 65536, which streaminfo cannot
            // represent, so RFC 9639 section 9.1.6 forbids it.
            if stored == 65_535 {
                return Err(reader.malformed("a block size representable in streaminfo"));
            }
            stored + 1
        }
    };

    let sample_rate = match sample_rate_bits {
        0 => None,
        1 => Some(88_200),
        2 => Some(176_400),
        3 => Some(192_000),
        4 => Some(8_000),
        5 => Some(16_000),
        6 => Some(22_050),
        7 => Some(24_000),
        8 => Some(32_000),
        9 => Some(44_100),
        10 => Some(48_000),
        11 => Some(96_000),
        12 => Some(reader.read_bits(8)? as u32 * 1_000),
        13 => Some(reader.read_bits(16)? as u32),
        14 => Some(reader.read_bits(16)? as u32 * 10),
        _ => return Err(reader.malformed("a sample rate other than the forbidden 0b1111")),
    };

    let header_bytes = reader.byte_position();
    let stored_crc = reader.read_bits(8)? as u8;
    if crc8(&bytes[..header_bytes]) != stored_crc {
        return Err(DecodeError::Malformed {
            expected: "a frame header whose CRC-8 matches its contents",
            offset: base + header_bytes as u64,
        });
    }

    Ok(FrameHeader {
        blocking,
        block_size,
        sample_rate,
        channels,
        bits_per_sample,
        coded_number,
        header_bytes: header_bytes + 1,
    })
}

/// What a frame header codes when it says a field comes from streaminfo, and
/// there is no streaminfo.
///
/// RFC 9639 section 9.1.4's sample rate code `0b0000` and section 9.1.3's bit
/// depth code `0b000` both mean "take this from the streaminfo block". A bare
/// frame stream has no streaminfo block, so neither is resolvable and neither
/// is guessed: the strings below are what the rejection names.
const UNRESOLVED_RATE: &str =
    "a frame sample rate, or a streaminfo block to resolve the sample rate code 0b0000";
/// The bit depth half of [`UNRESOLVED_RATE`].
const UNRESOLVED_DEPTH: &str =
    "a frame bit depth, or a streaminfo block to resolve the bit depth code 0b000";

/// Builds the stream description a bare frame stream never carried, out of
/// the first frame's own header.
///
/// FLAC frames are self-describing: each header restates the sample rate,
/// channel assignment and bit depth, so for a stream whose first frame states
/// all three this is a read rather than a guess. The two escape codes above
/// are the exception and are rejected by name.
///
/// Two fields cannot be derived and are not invented. `max_block_size` is the
/// format's own 65,535 rather than anything this stream promises, because a
/// bare stream makes no promise and a later frame may be larger than the
/// first; that is what bounds the decode buffer here, and it is the one place
/// in this crate where a ceiling follows a specification limit rather than
/// the bytes that arrived. `total_samples` and `md5` are `None`, which is
/// what makes [`Md5Check::NoStreamInfo`] the honest answer for this path.
fn derive_stream_info(header: &FrameHeader, base: u64) -> Result<FlacStreamInfo, DecodeError> {
    let Some(sample_rate) = header.sample_rate else {
        return Err(DecodeError::Malformed {
            expected: UNRESOLVED_RATE,
            offset: base,
        });
    };
    let Some(bits_per_sample) = header.bits_per_sample else {
        return Err(DecodeError::Malformed {
            expected: UNRESOLVED_DEPTH,
            offset: base,
        });
    };
    Ok(FlacStreamInfo {
        min_block_size: MIN_BLOCK_SIZE as u16,
        max_block_size: MAX_BLOCK_SIZE as u16,
        min_frame_size: None,
        max_frame_size: None,
        spec: AudioSpec::new(sample_rate, u16::from(header.channels.count())),
        bits_per_sample,
        total_samples: None,
        md5: None,
    })
}

// -- Prediction arithmetic, shared by the decoder and the encoder -------------

/// The fixed predictor of `order` applied to the samples before `prior`'s
/// end: what RFC 9639 section 9.2.5 says the next sample is expected to be.
///
/// **This is the one statement of the fixed prediction arithmetic in the
/// crate, and both directions go through it.** The decoder adds a residual to
/// this value; the encoder subtracts this value from a sample. If the two
/// used separate arithmetic, an encoder bug could produce streams this
/// crate's own decoder reconstructs incorrectly yet round-trips perfectly,
/// which is coverage lesson 3 exactly: one rule, two implementations, and a
/// control on either says nothing about the other.
///
/// The predictors are written out per order rather than folded into a
/// coefficient loop: they are five fixed polynomials, and a table of them
/// would be a table nobody could check against the specification at a
/// glance.
///
/// Wrapping for the reason recorded on [`decode_fixed`]: a corrupted stream
/// can drive the recursion without limit, and the CRC-16 that rejects the
/// frame is checked after the subframes are decoded.
fn fixed_prediction(prior: &[i64], order: usize) -> i64 {
    let at = prior.len();
    let previous = |back: usize| prior[at - back];
    match order {
        0 => 0,
        1 => previous(1),
        2 => previous(1).wrapping_mul(2).wrapping_sub(previous(2)),
        3 => previous(1)
            .wrapping_mul(3)
            .wrapping_sub(previous(2).wrapping_mul(3))
            .wrapping_add(previous(3)),
        _ => previous(1)
            .wrapping_mul(4)
            .wrapping_sub(previous(2).wrapping_mul(6))
            .wrapping_add(previous(3).wrapping_mul(4))
            .wrapping_sub(previous(4)),
    }
}

/// The linear prediction of the sample after `prior`'s end, from quantised
/// `coefficients` and their right `shift`: RFC 9639 section 9.2.6's sum,
/// shifted.
///
/// **The one statement of the LPC arithmetic in the crate**, for the same
/// reason as [`fixed_prediction`], and the more important of the two: an
/// encoder that predicted in floating point and quantised afterwards would
/// produce streams that decode to something slightly different, passing on
/// quiet material and failing on loud material at high bit depths, which no
/// small test file catches.
///
/// The coefficients are ordered as the bitstream orders them: the first
/// belongs to the sample immediately before the one being predicted.
/// i64 throughout: a 33-bit side sample times a 15-bit coefficient, summed
/// 32 times, needs 53 bits by RFC 9639 appendix A.3. Wrapping for the reason
/// recorded on [`decode_fixed`].
fn lpc_prediction(coefficients: &[i64], prior: &[i64], shift: u32) -> i64 {
    let at = prior.len();
    let mut prediction = 0i64;
    for (step, coefficient) in coefficients.iter().enumerate() {
        prediction = prediction.wrapping_add(coefficient.wrapping_mul(prior[at - 1 - step]));
    }
    prediction >> shift
}

// -- Subframes ----------------------------------------------------------------

/// Decodes one subframe's `dst.len()` samples at `bits` bits of depth.
///
/// `bits` is the frame's bit depth, already increased by one where this
/// subframe carries the side channel. Wasted bits are subtracted from it for
/// the coded values and shifted back on before returning, which is the order
/// RFC 9639 section 9.2.2 requires and the one that keeps stereo
/// reconstruction lossless.
fn decode_subframe(
    reader: &mut BitReader<'_>,
    bits: u32,
    dst: &mut [i64],
) -> Result<(), DecodeError> {
    if reader.read_bits(1)? != 0 {
        return Err(reader.malformed("a cleared leading bit in the subframe header"));
    }
    let subframe_type = reader.read_bits(6)? as u8;
    let wasted = if reader.read_bits(1)? == 0 {
        0
    } else {
        // The count minus one, in unary.
        reader.read_unary()?.saturating_add(1)
    };
    if wasted >= bits {
        return Err(reader.malformed("fewer wasted bits than the subframe's bit depth"));
    }
    let coded_bits = bits - wasted;

    match subframe_type {
        0 => {
            let value = reader.read_signed(coded_bits)?;
            dst.fill(value);
        }
        1 => {
            for sample in dst.iter_mut() {
                *sample = reader.read_signed(coded_bits)?;
            }
        }
        8..=12 => decode_fixed(reader, coded_bits, usize::from(subframe_type - 8), dst)?,
        32..=63 => decode_lpc(reader, coded_bits, usize::from(subframe_type - 31), dst)?,
        _ => return Err(reader.malformed("a defined subframe type")),
    }

    if wasted > 0 {
        for sample in dst.iter_mut() {
            *sample <<= wasted;
        }
    }
    Ok(())
}

/// Decodes a fixed-predictor subframe of `order`, which is 0 through 4.
///
/// The prediction itself is [`fixed_prediction`], the one statement of that
/// arithmetic in the crate, which the encoder also subtracts through.
///
/// # Why the arithmetic wraps
///
/// For a conforming stream it cannot: RFC 9639 appendix A.3 bounds a
/// fourth-order prediction over 32-bit audio at 36 bits, and every value
/// here is `i64`. For a *corrupted* one it can, because a frame whose
/// residuals were scrambled in transit drives the recursion without limit, and
/// CRC-16 that rejects the frame is checked after the subframes are decoded,
/// not before. Wrapping is defined behaviour that the CRC then catches;
/// a checked `+` would be a panic on untrusted input, which is the one
/// failure this crate refuses outright. The same reasoning applies to
/// [`decode_lpc`] and [`undo_decorrelation`].
fn decode_fixed(
    reader: &mut BitReader<'_>,
    bits: u32,
    order: usize,
    dst: &mut [i64],
) -> Result<(), DecodeError> {
    if order > dst.len() {
        return Err(reader.malformed("a predictor order no larger than the block size"));
    }
    for sample in dst.iter_mut().take(order) {
        *sample = reader.read_signed(bits)?;
    }
    decode_residual(reader, order, dst)?;

    for index in order..dst.len() {
        let prediction = fixed_prediction(&dst[..index], order);
        dst[index] = dst[index].wrapping_add(prediction);
    }
    Ok(())
}

/// Decodes a linear-predictor subframe of `order`, which is 1 through 32.
///
/// The coefficients appear in the bitstream in the order of the *past
/// samples* they multiply, which is the reverse of chronological order: the
/// first coefficient belongs to the sample immediately before the one being
/// predicted. Getting that backwards produces plausible-looking audio that
/// is wrong everywhere, so it is worth stating twice.
fn decode_lpc(
    reader: &mut BitReader<'_>,
    bits: u32,
    order: usize,
    dst: &mut [i64],
) -> Result<(), DecodeError> {
    if order > dst.len() {
        return Err(reader.malformed("a predictor order no larger than the block size"));
    }
    for sample in dst.iter_mut().take(order) {
        *sample = reader.read_signed(bits)?;
    }

    let precision = reader.read_bits(4)? as u32 + 1;
    if precision > 15 {
        return Err(reader.malformed("a coefficient precision other than the forbidden 0b1111"));
    }
    // The shift is stored two's complement and RFC 9639 appendix B.4 says it
    // must not be negative. A negative shift would be a left shift, which no
    // encoder since the field was defined has written.
    let shift = reader.read_signed(5)?;
    if shift < 0 {
        return Err(reader.malformed("a non-negative predictor right shift"));
    }
    let shift = shift as u32;

    let mut coefficients = [0i64; MAX_LPC_ORDER];
    for coefficient in coefficients.iter_mut().take(order) {
        *coefficient = reader.read_signed(precision)?;
    }

    decode_residual(reader, order, dst)?;

    for index in order..dst.len() {
        let prediction = lpc_prediction(&coefficients[..order], &dst[..index], shift);
        dst[index] = dst[index].wrapping_add(prediction);
    }
    Ok(())
}

/// Decodes the coded residual into `dst[order..]`.
///
/// RFC 9639 section 9.2.7. The partitioning arithmetic is the part that has
/// to be exact: the first partition is short by the predictor order, because
/// those samples were stored unencoded as warm-up.
fn decode_residual(
    reader: &mut BitReader<'_>,
    order: usize,
    dst: &mut [i64],
) -> Result<(), DecodeError> {
    let parameter_bits = match reader.read_bits(2)? {
        0 => 4u32,
        1 => 5,
        _ => return Err(reader.malformed("a defined residual coding method")),
    };
    let escape = (1u64 << parameter_bits) - 1;
    let partition_order = reader.read_bits(4)? as u32;
    let partitions = 1usize << partition_order;

    let block_size = dst.len();
    if !block_size.is_multiple_of(partitions) {
        return Err(reader.malformed("a block size divisible by the partition count"));
    }
    let per_partition = block_size >> partition_order;
    if per_partition <= order {
        return Err(reader.malformed("a partition longer than the predictor order"));
    }

    let mut at = order;
    for partition in 0..partitions {
        let count = if partition == 0 {
            per_partition - order
        } else {
            per_partition
        };
        let parameter = reader.read_bits(parameter_bits)?;

        if parameter == escape {
            // An escaped partition holds unencoded residuals at a stated
            // width. A width of zero means every residual in it is zero and
            // no bits at all are stored.
            let width = reader.read_bits(5)? as u32;
            if width == 0 {
                dst[at..at + count].fill(0);
            } else {
                for sample in &mut dst[at..at + count] {
                    *sample = reader.read_signed(width)?;
                }
            }
        } else {
            let parameter = parameter as u32;
            for sample in &mut dst[at..at + count] {
                let high = u64::from(reader.read_unary()?);
                let folded = if parameter == 0 {
                    high
                } else {
                    (high << parameter) | reader.read_bits(parameter)?
                };
                // RFC 9639 section 9.2.7.3 bounds every residual to a 32-bit
                // one's-complement range, so the folded value fits 32 bits
                // and cannot be the one that folds to -2^31.
                if folded > u64::from(u32::MAX) - 1 {
                    return Err(reader.malformed("a residual within the format's 32-bit range"));
                }
                // Zigzag: even folds to itself halved, odd folds to the
                // negative of half the next value up.
                *sample = ((folded >> 1) as i64) ^ -((folded & 1) as i64);
            }
        }
        at += count;
    }
    Ok(())
}

// -- Stereo decorrelation -----------------------------------------------------

/// Restores left and right from whichever pair of channels the frame stored.
///
/// `left` and `right` are the two subframes as decoded, in bitstream order,
/// and are replaced in place. RFC 9639 section 4.2.
///
/// The mid/side case is the one that is commonly got wrong. A mid sample is
/// `(left + right) >> 1`, so it has lost a least-significant bit, but only
/// when `left + right` was odd, which is exactly when `left - right` is odd
/// too. So the bit is recoverable from the side channel, and the correction
/// below is what makes mid/side lossless rather than nearly so.
fn undo_decorrelation(assignment: ChannelAssignment, left: &mut [i64], right: &mut [i64]) {
    match assignment {
        ChannelAssignment::Independent(_) => {}
        ChannelAssignment::LeftSide => {
            // left is left, right is the side channel.
            for (channel, side) in left.iter().zip(right.iter_mut()) {
                *side = channel.wrapping_sub(*side);
            }
        }
        ChannelAssignment::SideRight => {
            // left is the side channel, right is right.
            for (side, channel) in left.iter_mut().zip(right.iter()) {
                *side = side.wrapping_add(*channel);
            }
        }
        ChannelAssignment::MidSide => {
            for (mid, side) in left.iter_mut().zip(right.iter_mut()) {
                let restored = (mid.wrapping_shl(1)) | (*side & 1);
                let difference = *side;
                *mid = restored.wrapping_add(difference) >> 1;
                *side = restored.wrapping_sub(difference) >> 1;
            }
        }
    }
}

// -- The frame decoder --------------------------------------------------------

/// Decodes frames of one stream, holding the scratch both readers reuse.
///
/// It owns the MD5 computation, so the two readers cannot disagree about
/// what was hashed: they feed the same function the same frames.
#[derive(Debug)]
struct FrameDecoder {
    info: FlacStreamInfo,
    /// Decoded integer samples, one channel after another, `block_size` of
    /// each. Grown to the largest frame actually met, never to a declared
    /// size.
    channels: Vec<i64>,
    /// The MD5 of the unencoded audio, computed as decoding proceeds.
    hasher: Md5,
    /// Interleaved bytes staged for the MD5, and how many are filled.
    stage: [u8; MD5_STAGE_BYTES],
    staged: usize,
    /// Interchannel samples produced so far.
    samples_done: u64,
    /// Whether the previous frame was short enough that the format only
    /// allows it as the stream's last.
    last_frame_seen: bool,
    /// The blocking strategy the first frame declared, which the rest must
    /// match.
    blocking: Option<Blocking>,
    /// Whether `info` came from a streaminfo block, one read out of the
    /// stream or one the caller supplied out of band, rather than being
    /// derived from the first frame's own header.
    ///
    /// It decides one thing: whether a frame that defers its sample rate or
    /// bit depth to streaminfo can be resolved. Derived properties are not a
    /// streaminfo block and are not allowed to stand in for one, because a
    /// later frame's escape code refers to a block that does not exist and
    /// answering it from the first frame would be a guess.
    streaminfo: bool,
    /// `sample as f32` divides by this, per the crate's scaling rule.
    divisor: f32,
}

impl FrameDecoder {
    /// A decoder for a stream whose streaminfo block says `info`.
    fn new(info: FlacStreamInfo) -> Self {
        Self::build(info, true)
    }

    /// A decoder for a bare frame stream, whose `info` was derived from its
    /// first frame by [`derive_stream_info`].
    fn from_first_frame(info: FlacStreamInfo) -> Self {
        Self::build(info, false)
    }

    /// The two constructors' shared body.
    ///
    /// Construction, then [`reset_to`](Self::reset_to), so the starting state
    /// is stated once rather than once per way of reaching it.
    fn build(info: FlacStreamInfo, streaminfo: bool) -> Self {
        let mut decoder = Self {
            info,
            channels: Vec::new(),
            hasher: Md5::new(),
            stage: [0; MD5_STAGE_BYTES],
            staged: 0,
            samples_done: 0,
            last_frame_seen: false,
            blocking: None,
            streaminfo,
            divisor: 0.0,
        };
        decoder.reset_to(info, streaminfo);
        decoder
    }

    /// Returns this decoder to its just-constructed state for `info`, keeping
    /// the sample buffer's allocation.
    ///
    /// **This is the one statement of what a fresh decoder's state is**, and
    /// [`build`](Self::build) goes through it, so the two cannot drift into
    /// disagreeing about what "fresh" means.
    ///
    /// It exists because [`FlacRecovery`] trial-decodes a candidate frame at
    /// every plausible sync point, and a frame header may legitimately
    /// declare the format's largest block: 65,535 samples in 8 channels is a
    /// four-megabyte buffer. Building a decoder per candidate meant a byte
    /// run of such headers cost one four-megabyte allocation, and one
    /// four-megabyte zeroing, per candidate position, which is time
    /// proportional to input length times the format's maximum frame rather
    /// than to the input. Reusing the buffer is a change to what the scan
    /// *costs* and to nothing it *decides*: every field a decision reads is
    /// reset here to exactly what construction sets.
    fn reset_to(&mut self, info: FlacStreamInfo, streaminfo: bool) {
        self.info = info;
        self.channels.clear();
        self.hasher = Md5::new();
        self.staged = 0;
        self.samples_done = 0;
        self.last_frame_seen = false;
        self.blocking = None;
        self.streaminfo = streaminfo;
        // The crate's scaling rule, generalised to FLAC's 4-to-32-bit range:
        // an integer sample of n bits divides by 2^(n-1). At 8, 16, 24 and 32
        // bits this is exactly the divisor `sample.rs` uses, which
        // `flac_scaling_agrees_with_sample_rs` pins.
        self.divisor = (1u64 << (info.bits_per_sample - 1)) as f32;
    }

    /// Decodes one frame from the start of `bytes`, appends its interleaved
    /// samples to `output` and returns how many bytes the frame occupied.
    ///
    /// `base` is where `bytes[0]` sits in the whole input.
    fn decode_frame(
        &mut self,
        bytes: &[u8],
        base: u64,
        output: &mut Vec<f32>,
    ) -> Result<usize, DecodeError> {
        let header = parse_frame_header(bytes, base)?;
        self.check_against_stream(&header, base)?;
        // Recorded after the check, so the first frame sets the strategy and
        // every frame after it has to match.
        self.blocking = Some(header.blocking);

        let block_size = header.block_size as usize;
        let channel_count = usize::from(header.channels.count());
        let bits = u32::from(self.info.bits_per_sample);

        // The one allocation that follows a number from the file, bounded by
        // the streaminfo maximum block size checked above and by the
        // format's own 65535-by-8 ceiling.
        self.channels.clear();
        self.channels.resize(block_size * channel_count, 0);

        let mut reader = BitReader::new(bytes, base);
        reader.bit = header.header_bytes * 8;
        for channel in 0..channel_count {
            let side = header.channels.side_channel() == Some(channel);
            let start = channel * block_size;
            decode_subframe(
                &mut reader,
                bits + u32::from(side),
                &mut self.channels[start..start + block_size],
            )?;
        }

        reader.align_to_byte();
        let frame_bytes = reader.byte_position();
        let stored_crc = reader.read_bits(16)? as u16;
        if crc16(&bytes[..frame_bytes]) != stored_crc {
            return Err(DecodeError::Malformed {
                expected: "a frame whose CRC-16 matches its contents",
                offset: base + frame_bytes as u64,
            });
        }

        if channel_count == 2 {
            let (left, right) = self.channels.split_at_mut(block_size);
            undo_decorrelation(header.channels, left, right);
        }

        self.emit(block_size, channel_count, output);
        self.samples_done += header.block_size as u64;
        self.last_frame_seen = header.block_size < MIN_BLOCK_SIZE;
        Ok(frame_bytes + 2)
    }

    /// Rejects a frame whose properties do not match the stream's.
    ///
    /// Every check here is a place where decoding on would produce audio
    /// that is wrong rather than an error: a channel count or bit depth that
    /// changed cannot be described by one [`AudioSpec`], and a block size
    /// past the streaminfo maximum is both a format violation and the only
    /// bound on this decoder's buffer.
    fn check_against_stream(&self, header: &FrameHeader, base: u64) -> Result<(), DecodeError> {
        let malformed = |expected| DecodeError::Malformed {
            expected,
            offset: base,
        };
        if let Some(blocking) = self.blocking {
            if blocking != header.blocking {
                return Err(malformed("an unchanging blocking strategy"));
            }
        }
        if self.last_frame_seen {
            return Err(malformed(
                "no frame after one shorter than the minimum block size",
            ));
        }
        if u32::from(header.channels.count()) != u32::from(self.info.spec.channels) {
            return Err(malformed("a channel count matching streaminfo"));
        }
        match header.bits_per_sample {
            Some(bits) if bits != self.info.bits_per_sample => {
                return Err(malformed("a bit depth matching streaminfo"))
            }
            // The escape code, in a stream that has no streaminfo to escape
            // to. Rejected by name rather than answered from the first frame.
            None if !self.streaminfo => return Err(malformed(UNRESOLVED_DEPTH)),
            _ => {}
        }
        match header.sample_rate {
            Some(rate) if rate != self.info.spec.sample_rate => {
                return Err(malformed("a sample rate matching streaminfo"))
            }
            None if !self.streaminfo => return Err(malformed(UNRESOLVED_RATE)),
            _ => {}
        }
        if header.block_size > u32::from(self.info.max_block_size) {
            return Err(malformed(
                "a block size no larger than the streaminfo maximum",
            ));
        }
        Ok(())
    }

    /// Interleaves the decoded channels into `output` as `f32`, and feeds the
    /// same samples to the MD5 in the byte-aligned little-endian form
    /// RFC 9639 section 8.2 specifies.
    fn emit(&mut self, block_size: usize, channel_count: usize, output: &mut Vec<f32>) {
        let width = self.info.md5_bytes_per_sample();
        output.reserve(block_size * channel_count);
        for index in 0..block_size {
            for channel in 0..channel_count {
                let sample = self.channels[channel * block_size + index];
                output.push(sample as f32 / self.divisor);
                if self.staged + width > MD5_STAGE_BYTES {
                    self.hasher.update(&self.stage[..self.staged]);
                    self.staged = 0;
                }
                // Little-endian, sign-extended to a whole number of bytes:
                // the low `width` bytes of the two's-complement value are
                // exactly that, because the value already fits the depth.
                self.stage[self.staged..self.staged + width]
                    .copy_from_slice(&sample.to_le_bytes()[..width]);
                self.staged += width;
            }
        }
    }

    /// Closes the stream: checks the sample count against streaminfo and the
    /// MD5 against the decoded audio.
    ///
    /// `md5_offset` is where the checksum field sits in the input, so the
    /// rejection names it rather than the file.
    fn finish(mut self, md5_offset: u64) -> Result<(), DecodeError> {
        if let Some(total) = self.info.total_samples {
            if self.samples_done != total {
                return Err(if self.samples_done < total {
                    DecodeError::Truncated {
                        expected: total,
                        available: self.samples_done,
                    }
                } else {
                    DecodeError::Malformed {
                        expected: "a stream holding exactly the declared total number of samples",
                        offset: md5_offset,
                    }
                });
            }
        }
        let Some(declared) = self.info.md5 else {
            // An all-zero checksum means "not known", which the format
            // permits. There is simply no oracle for this file.
            return Ok(());
        };
        self.hasher.update(&self.stage[..self.staged]);
        self.staged = 0;
        if self.hasher.finish() != declared {
            return Err(DecodeError::Malformed {
                expected: "decoded audio matching the streaminfo MD5 checksum",
                offset: md5_offset,
            });
        }
        Ok(())
    }
}

// -- Metadata -----------------------------------------------------------------

/// What a metadata walk established about a stream's header.
struct Metadata {
    info: FlacStreamInfo,
    /// Where the MD5 field sits in the input.
    md5_offset: u64,
    /// Where the first audio frame starts.
    audio_offset: usize,
}

/// Reads the `fLaC` signature and every metadata block, and answers where the
/// audio starts.
///
/// RFC 9639 section 8 requires the first metadata block to be streaminfo, so
/// a file that puts anything else there is rejected rather than searched.
/// Every other block type is stepped over without being parsed: this crate
/// decodes audio, and a vorbis comment it never reads cannot mislead it.
fn read_metadata(bytes: &[u8]) -> Result<Metadata, DecodeError> {
    let magic = bytes
        .get(..4)
        .map(|head| FourCc([head[0], head[1], head[2], head[3]]));
    match magic {
        Some(tag) if tag == MAGIC => {}
        Some(tag) => return Err(DecodeError::UnsupportedContainer { tag }),
        None => {
            return Err(DecodeError::Truncated {
                expected: 4,
                available: bytes.len() as u64,
            })
        }
    }

    let mut at = 4usize;
    let mut info: Option<(FlacStreamInfo, u64)> = None;
    loop {
        let Some(header) = bytes.get(at..at + METADATA_HEADER_BYTES) else {
            return Err(DecodeError::Truncated {
                expected: (at + METADATA_HEADER_BYTES) as u64,
                available: bytes.len() as u64,
            });
        };
        let last = header[0] & 0x80 != 0;
        let kind = header[0] & 0x7F;
        // u64 throughout: `at + length` in usize wraps on a 32-bit target for
        // a crafted length, which turns a range check into a pass.
        let length =
            (u64::from(header[1]) << 16) | (u64::from(header[2]) << 8) | u64::from(header[3]);
        let body_at = at as u64 + METADATA_HEADER_BYTES as u64;
        if kind == BLOCK_FORBIDDEN {
            return Err(DecodeError::Malformed {
                expected: "a metadata block type other than the forbidden 127",
                offset: at as u64,
            });
        }
        if body_at + length > bytes.len() as u64 {
            return Err(DecodeError::Truncated {
                expected: body_at + length,
                available: bytes.len() as u64,
            });
        }
        let body_at = body_at as usize;
        let body_end = body_at + length as usize;

        if kind == BLOCK_STREAMINFO {
            if info.is_some() {
                return Err(DecodeError::Malformed {
                    expected: "exactly one streaminfo metadata block",
                    offset: at as u64,
                });
            }
            info = Some((
                parse_streaminfo(&bytes[body_at..body_end], body_at as u64)?,
                body_at as u64 + 18,
            ));
        } else if info.is_none() {
            return Err(DecodeError::Malformed {
                expected: "a streaminfo block first among the metadata blocks",
                offset: at as u64,
            });
        }

        at = body_end;
        if last {
            break;
        }
    }

    let Some((info, md5_offset)) = info else {
        return Err(DecodeError::Malformed {
            expected: "a streaminfo metadata block",
            offset: 4,
        });
    };
    Ok(Metadata {
        info,
        md5_offset,
        audio_offset: at,
    })
}

/// `true` when `bytes` opens with something that could be a frame header.
///
/// Used only to tell a stream that carries more audio than it declared from
/// one that carries a trailing tag, which is common in the wild and harmless.
fn looks_like_a_frame(bytes: &[u8]) -> bool {
    matches!(bytes, [0xFF, second, ..] if second & 0xFE == 0xF8)
}

/// Decodes every frame in `audio` into `output` and answers how many bytes
/// were consumed.
///
/// `base` is where `audio[0]` sits in the whole input, so rejections name
/// absolute offsets. The MD5 is *not* closed here: the caller decides whether
/// [`FrameDecoder::finish`] is the right thing to do with what it produced.
///
/// One function rather than one per reader, for the reason recorded in
/// `CLAUDE.md`'s third coverage lesson: a rule with two implementations means
/// a negative control on the first says nothing about the second. The
/// whole-file reader, the bare frame reader and the recovering reader all
/// walk frames through this.
fn decode_frames(
    decoder: &mut FrameDecoder,
    audio: &[u8],
    base: u64,
    output: &mut Vec<f32>,
) -> Result<usize, DecodeError> {
    let total = decoder.info.total_samples;
    let mut at = 0usize;
    while at < audio.len() {
        if total.is_some_and(|total| decoder.samples_done >= total) {
            break;
        }
        at += decoder.decode_frame(&audio[at..], base + at as u64, output)?;
    }

    // A stream that declared its length and then carried more audio is
    // internally inconsistent, and preferring either source silently is the
    // failure this crate refuses. A trailing tag is not audio and is left
    // alone.
    if total.is_some() && looks_like_a_frame(&audio[at..]) {
        return Err(DecodeError::Malformed {
            expected: "a stream holding exactly the declared total number of samples",
            offset: base + at as u64,
        });
    }
    Ok(at)
}

// -- What a decode did about the checksum -------------------------------------

/// Whether a decode checked its output against a streaminfo MD5, and why not
/// when it did not.
///
/// The whole-file reader has no need of this: a caller holding a
/// [`FlacReader`] can read [`FlacStreamInfo::md5`] and knows that a decode
/// which returned `Ok` matched it. The three paths that take their streaminfo
/// from somewhere other than the stream can each end up with no checksum to
/// check, for three different reasons, and none of them is visible in the
/// samples. Silently not verifying is how a decoder ends up credited with a
/// guarantee it did not provide, so it is reported instead.
///
/// There is no "failed" state. A mismatch is [`DecodeError::Malformed`] and
/// no samples travel with it.
///
/// `#[non_exhaustive]`: a consumer matching on it keeps a `_ =>` arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Md5Check {
    /// The decoded audio was hashed and matched the streaminfo checksum.
    Verified,
    /// No streaminfo was available, so the stream's properties were derived
    /// from its first frame and there was no checksum to compare against.
    NoStreamInfo,
    /// Streaminfo was available and its checksum field is the all-zero value
    /// that RFC 9639 section 8.2 defines as "not known". Permitted by the
    /// format and not a failure.
    ChecksumUnset,
    /// Audio was skipped, so a hash over what was decoded is a hash over a
    /// different set of samples than the checksum describes. Only
    /// [`FlacRecovery`] can reach this.
    AudioIncomplete,
}

impl Md5Check {
    /// Whether the audio was actually checked against a checksum.
    ///
    /// The one-line form of the enum above, for a caller that wants to refuse
    /// audio nothing verified.
    pub const fn is_verified(self) -> bool {
        matches!(self, Self::Verified)
    }
}

// -- The whole-file reader ----------------------------------------------------

/// A FLAC file held whole in memory.
///
/// [`new`](Self::new) validates the signature and every metadata block;
/// [`decode_to_end`](Self::decode_to_end) decodes the audio frames and
/// verifies the streaminfo MD5 over what it produced. The split is why this
/// reader's decode returns a `Result` where the two PCM containers' do not:
/// their payload cannot fail once the header has parsed, and a FLAC frame
/// can.
///
/// # Example
///
/// ```
/// use decibri_decode::FlacReader;
///
/// // RFC 9639 appendix D.3's worked example: 8-bit mono, 24 samples.
/// let file: [u8; 73] = [
///     0x66, 0x4c, 0x61, 0x43, 0x80, 0x00, 0x00, 0x22, 0x10, 0x00, 0x10, 0x00,
///     0x00, 0x00, 0x1f, 0x00, 0x00, 0x1f, 0x07, 0xd0, 0x00, 0x70, 0x00, 0x00,
///     0x00, 0x18, 0xf8, 0xf9, 0xe3, 0x96, 0xf5, 0xcb, 0xcf, 0xc6, 0xdc, 0x80,
///     0x7f, 0x99, 0x77, 0x90, 0x6b, 0x32, 0xff, 0xf8, 0x68, 0x02, 0x00, 0x17,
///     0xe9, 0x44, 0x00, 0x4f, 0x6f, 0x31, 0x3d, 0x10, 0x47, 0xd2, 0x27, 0xcb,
///     0x6d, 0x09, 0x08, 0x31, 0x45, 0x2b, 0xdc, 0x28, 0x22, 0x22, 0x80, 0x57,
///     0xa3,
/// ];
///
/// let reader = FlacReader::new(&file)?;
/// assert_eq!(reader.spec().sample_rate, 32_000);
/// assert_eq!(reader.stream_info().bits_per_sample, 8);
/// assert_eq!(reader.frames(), Some(24));
///
/// let decoded = reader.decode_to_end()?;
/// assert_eq!(decoded.frames(), 24);
/// // The first four samples are 0, 79, 111 and 78, scaled by 2^7.
/// assert_eq!(&decoded.samples()[..4], &[0.0, 79.0 / 128.0, 111.0 / 128.0, 78.0 / 128.0]);
/// # Ok::<(), decibri_decode::DecodeError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlacReader<'a> {
    info: FlacStreamInfo,
    md5_offset: u64,
    /// The audio frames, from the first frame byte to the end of the input.
    audio: &'a [u8],
    /// Where `audio[0]` sits in the input, so rejections name absolute
    /// offsets.
    audio_offset: u64,
}

impl<'a> FlacReader<'a> {
    /// Parses `bytes` as a FLAC stream, reading the signature and metadata.
    ///
    /// The audio frames are not touched here, so this is cheap on a large
    /// file and a reader that exists is a file whose *header* is valid.
    ///
    /// # Errors
    ///
    /// - [`DecodeError::UnsupportedContainer`] when the input does not start
    ///   with `fLaC`, naming the four bytes it did start with.
    /// - [`DecodeError::Truncated`] when the input ends inside the signature,
    ///   a metadata block header or a metadata block body.
    /// - [`DecodeError::Malformed`] when the first metadata block is not
    ///   streaminfo, a second streaminfo appears, a block declares the
    ///   forbidden type 127, or streaminfo carries a block size, sample rate
    ///   or bit depth the format does not allow.
    pub fn new(bytes: &'a [u8]) -> Result<Self, DecodeError> {
        let metadata = read_metadata(bytes)?;
        Ok(Self {
            info: metadata.info,
            md5_offset: metadata.md5_offset,
            audio: &bytes[metadata.audio_offset..],
            audio_offset: metadata.audio_offset as u64,
        })
    }

    /// What the streaminfo metadata block declared.
    pub const fn stream_info(&self) -> &FlacStreamInfo {
        &self.info
    }

    /// The rate and layout the stream decodes to.
    pub const fn spec(&self) -> AudioSpec {
        self.info.spec
    }

    /// How many interchannel samples the stream declares, or `None` when the
    /// streaminfo field holds the zero that means unknown.
    ///
    /// `None` rather than `0` because a stream of unknown length and an
    /// empty one are different things, and `0` would conflate them.
    pub const fn frames(&self) -> Option<u64> {
        self.info.total_samples
    }

    /// The audio frames, undecoded.
    pub const fn frame_data(&self) -> &'a [u8] {
        self.audio
    }

    /// Decodes every frame, appends the interleaved samples to `output` and
    /// verifies the streaminfo MD5 over them.
    ///
    /// Returns how many interleaved samples were appended, frames times
    /// channels, which is the length `output` grew by.
    ///
    /// # Errors
    ///
    /// - [`DecodeError::Malformed`] for a frame whose CRC-8 or CRC-16 does
    ///   not match, a reserved or forbidden field value, a frame whose
    ///   properties disagree with streaminfo, and, as the check that covers
    ///   everything else, decoded audio whose MD5 is not the one streaminfo
    ///   carries.
    /// - [`DecodeError::Truncated`] when the input ends inside a frame or
    ///   holds fewer samples than streaminfo declares.
    pub fn decode(&self, output: &mut Vec<f32>) -> Result<usize, DecodeError> {
        let before = output.len();
        let mut decoder = FrameDecoder::new(self.info);
        decode_frames(&mut decoder, self.audio, self.audio_offset, output)?;
        decoder.finish(self.md5_offset)?;
        Ok(output.len() - before)
    }

    /// Decodes the whole stream, bound to the spec that describes it.
    ///
    /// # Errors
    ///
    /// As [`decode`](Self::decode).
    pub fn decode_to_end(&self) -> Result<AudioBuffer, DecodeError> {
        let mut samples = Vec::new();
        self.decode(&mut samples)?;
        Ok(AudioBuffer::from_samples(self.info.spec, samples))
    }
}

// -- The bare frame reader ----------------------------------------------------

/// What a bare frame stream decode did.
///
/// `#[non_exhaustive]`: a consumer constructing or matching it exhaustively
/// would break on a field added later.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct FlacFrameReport {
    /// How many interleaved samples were appended to the output.
    pub samples: usize,
    /// Whether the audio was checked against a streaminfo MD5.
    pub md5: Md5Check,
}

/// A bare FLAC frame stream held whole in memory: no `fLaC` signature, no
/// metadata, the first byte is the first byte of a frame header.
///
/// # Why this is decodable at all
///
/// A FLAC frame is self-describing. Its header restates the block size,
/// sample rate, channel assignment and bit depth, carries a CRC-8 over
/// itself and is followed by a CRC-16 over the whole frame. So a run of
/// frames with nothing in front of it is a complete stream description, and
/// the corpus carries two files that are exactly that.
///
/// # Where it sits
///
/// This is the third headerless decoder in the crate, beside
/// [`PcmDecoder`](crate::PcmDecoder) and [`G711Decoder`](crate::G711Decoder),
/// and it takes the same position they do: what the container would have said
/// arrives out of band, or not at all. The difference is that PCM and G.711
/// bytes carry no claim, so their `new` takes an
/// [`AudioSpec`](crate::AudioSpec) the caller asserts, whereas FLAC frames do
/// carry one, so [`new`](Self::new) reads the properties rather than being
/// told them, and [`with_stream_info`](Self::with_stream_info) is for the
/// caller who has a real streaminfo block from somewhere else.
///
/// # The two escape codes
///
/// Sample rate code `0b0000` and bit depth code `0b000` mean "take this field
/// from the streaminfo block". [`new`](Self::new) has no streaminfo block, so
/// a frame using either is [`DecodeError::Malformed`] naming which of the two
/// could not be resolved. Nothing is defaulted and nothing is carried over
/// from a neighbouring frame. [`with_stream_info`](Self::with_stream_info)
/// resolves both.
///
/// # Example
///
/// ```
/// use decibri_decode::{FlacFrameReader, Md5Check};
///
/// # let file: [u8; 73] = [
/// #     0x66, 0x4c, 0x61, 0x43, 0x80, 0x00, 0x00, 0x22, 0x10, 0x00, 0x10, 0x00,
/// #     0x00, 0x00, 0x1f, 0x00, 0x00, 0x1f, 0x07, 0xd0, 0x00, 0x70, 0x00, 0x00,
/// #     0x00, 0x18, 0xf8, 0xf9, 0xe3, 0x96, 0xf5, 0xcb, 0xcf, 0xc6, 0xdc, 0x80,
/// #     0x7f, 0x99, 0x77, 0x90, 0x6b, 0x32, 0xff, 0xf8, 0x68, 0x02, 0x00, 0x17,
/// #     0xe9, 0x44, 0x00, 0x4f, 0x6f, 0x31, 0x3d, 0x10, 0x47, 0xd2, 0x27, 0xcb,
/// #     0x6d, 0x09, 0x08, 0x31, 0x45, 0x2b, 0xdc, 0x28, 0x22, 0x22, 0x80, 0x57,
/// #     0xa3,
/// # ];
/// // RFC 9639 appendix D.3's worked example with its 42-byte header removed,
/// // which is what a container handing over bare frames would deliver.
/// let frames = &file[42..];
///
/// let reader = FlacFrameReader::new(frames)?;
/// assert_eq!(reader.spec().sample_rate, 32_000);
/// assert_eq!(reader.stream_info().bits_per_sample, 8);
/// // Nothing declares a length, so there is none to report.
/// assert_eq!(reader.frames(), None);
///
/// let mut samples = Vec::new();
/// let report = reader.decode(&mut samples)?;
/// assert_eq!(report.samples, 24);
/// // No streaminfo block came with the frames, so nothing verified them.
/// assert_eq!(report.md5, Md5Check::NoStreamInfo);
/// assert!(!report.md5.is_verified());
/// # Ok::<(), decibri_decode::DecodeError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlacFrameReader<'a> {
    /// The frames, from the first frame byte to the end of the input.
    audio: &'a [u8],
    info: FlacStreamInfo,
    /// Whether `info` came from a streaminfo block rather than from the first
    /// frame's header.
    streaminfo: bool,
}

impl<'a> FlacFrameReader<'a> {
    /// Reads a bare frame stream, taking the stream's properties from its
    /// first frame header.
    ///
    /// Only the first header is parsed here, so this is cheap on a large
    /// input and a reader that exists is a stream whose *first frame header*
    /// is valid.
    ///
    /// # Errors
    ///
    /// - [`DecodeError::Truncated`] when the input is shorter than the
    ///   smallest possible frame header.
    /// - [`DecodeError::Malformed`] when the input does not begin with a
    ///   frame sync code, when the header's CRC-8 does not match, when a
    ///   reserved value is used, and, as the case this path adds, when the
    ///   header defers its sample rate or bit depth to a streaminfo block
    ///   that does not exist, which the error names.
    pub fn new(bytes: &'a [u8]) -> Result<Self, DecodeError> {
        let header = parse_frame_header(bytes, 0)?;
        Ok(Self {
            audio: bytes,
            info: derive_stream_info(&header, 0)?,
            streaminfo: false,
        })
    }

    /// Reads a bare frame stream described by a streaminfo block the caller
    /// obtained elsewhere.
    ///
    /// This is the shape Ogg and Matroska use: the streaminfo block travels
    /// in the container's own codec-private header and the packets that
    /// follow are bare frames. The block's bytes become an `info` through
    /// [`FlacStreamInfo::from_block`]. Supplying it resolves the two escape
    /// codes, restores the declared sample count as a check, and, when the
    /// block carries one, restores the MD5 that a bare stream otherwise has
    /// no way to be checked against.
    ///
    /// Nothing is parsed here, so nothing can fail; a stream whose frames
    /// disagree with `info` is rejected by [`decode`](Self::decode), naming
    /// the field that disagreed.
    pub const fn with_stream_info(bytes: &'a [u8], info: FlacStreamInfo) -> Self {
        Self {
            audio: bytes,
            info,
            streaminfo: true,
        }
    }

    /// The stream's properties: supplied by the caller, or read out of the
    /// first frame header.
    ///
    /// A derived description is not a streaminfo block and does not pretend
    /// to be one. Its `max_block_size` is the format's own 65,535 rather than
    /// a promise this stream made, and `total_samples`, `md5`,
    /// `min_frame_size` and `max_frame_size` are all `None`.
    pub const fn stream_info(&self) -> &FlacStreamInfo {
        &self.info
    }

    /// The rate and layout the stream decodes to.
    pub const fn spec(&self) -> AudioSpec {
        self.info.spec
    }

    /// How many interchannel samples the stream declares, which for a derived
    /// description is always `None` because nothing declared anything.
    pub const fn frames(&self) -> Option<u64> {
        self.info.total_samples
    }

    /// Decodes every frame and appends the interleaved samples to `output`.
    ///
    /// # Errors
    ///
    /// - [`DecodeError::Malformed`] for a frame whose CRC-8 or CRC-16 does
    ///   not match, a reserved or forbidden field value, a frame whose
    ///   properties disagree with the stream's, a frame deferring a field to
    ///   a streaminfo block that is not there, and, when a checksum was
    ///   supplied, decoded audio that does not match it.
    /// - [`DecodeError::Truncated`] when the input ends inside a frame. A
    ///   partial trailing frame is an error here, exactly as it is
    ///   everywhere else in this crate; recovering what can be read from a
    ///   damaged run is [`FlacRecovery`], and is asked for explicitly.
    pub fn decode(&self, output: &mut Vec<f32>) -> Result<FlacFrameReport, DecodeError> {
        let before = output.len();
        let mut decoder = if self.streaminfo {
            FrameDecoder::new(self.info)
        } else {
            FrameDecoder::from_first_frame(self.info)
        };
        decode_frames(&mut decoder, self.audio, 0, output)?;
        // No streaminfo block sits in this input, so a checksum rejection
        // cannot name where the checksum came from. It names where the decode
        // concluded, which is the last byte it read.
        decoder.finish(self.audio.len() as u64)?;
        Ok(FlacFrameReport {
            samples: output.len() - before,
            md5: self.md5_outcome(),
        })
    }

    /// Decodes the whole stream, bound to the spec that describes it.
    ///
    /// # Errors
    ///
    /// As [`decode`](Self::decode).
    pub fn decode_to_end(&self) -> Result<(AudioBuffer, FlacFrameReport), DecodeError> {
        let mut samples = Vec::new();
        let report = self.decode(&mut samples)?;
        Ok((AudioBuffer::from_samples(self.info.spec, samples), report))
    }

    /// What a decode that returned `Ok` did about the checksum.
    fn md5_outcome(&self) -> Md5Check {
        if !self.streaminfo {
            Md5Check::NoStreamInfo
        } else if self.info.md5.is_none() {
            Md5Check::ChecksumUnset
        } else {
            Md5Check::Verified
        }
    }
}

// -- Recovery -----------------------------------------------------------------

/// Why a range of input produced no audio.
///
/// `#[non_exhaustive]`: a consumer matching on it keeps a `_ =>` arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FlacSkipReason {
    /// No sync point could be established anywhere in the range. Leading
    /// garbage, a run that begins part-way through a frame, and anything
    /// after the last frame all arrive here.
    NoSyncPoint,
    /// A frame at a known boundary was rejected, by a CRC that did not match
    /// or a header that disagreed with the stream, and the search for the
    /// next sync point began after it.
    FrameRejected,
}

/// One range of input that produced no audio.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct FlacSkip {
    /// The half-open byte range of the input, `[start, end)`, that was
    /// skipped.
    pub bytes: Range<u64>,
    /// The frames this gap cost, as a half-open range of the stream's own
    /// interchannel sample positions, or `None` when the stream does not say
    /// what they were.
    ///
    /// A frame is one interchannel sample, one value per channel, the unit
    /// AIFF calls a sample frame and FLAC's own frame counts use. A stereo
    /// gap of `0..10` is ten frames and twenty interleaved samples, so this
    /// range and [`FlacRecoveryReport::samples`] count in different units,
    /// and the names say which is which.
    ///
    /// The positions are **the stream's own**, taken from the coded numbers
    /// that surviving frame headers carry, not offsets into the output. A
    /// variable-blocksize stream codes the sample number directly; a
    /// fixed-blocksize stream codes a frame number, which is multiplied by
    /// the largest block size any recovered frame carried, the stream's
    /// constant block size, unless the only frames recovered were the
    /// stream's short last one.
    ///
    /// A gap between two recovered frames always resolves, and its width is
    /// exactly what was lost however the stream numbers itself. The two ends
    /// of the input are the cases that may not:
    ///
    /// - A gap **before the first recovered frame** resolves only when that
    ///   frame is the stream's first, or when a streaminfo block was supplied
    ///   and the input is therefore known to start where the stream does. A
    ///   bare frame stream may begin anywhere, and the conformance corpus has
    ///   one whose first surviving frame is number 12,927, so answering
    ///   "12,927 frames were lost" for a run that never held them would be an
    ///   invention. [`FlacRecoveryReport::first_frame`] is the thing that is
    ///   always known there.
    /// - A gap **after the last recovered frame** resolves only when a
    ///   declared total sample count says how far the stream ran.
    ///
    /// `None` is not zero and must not be read as it. An empty range is the
    /// meaningful zero: it says this gap cost no audio, which is what garbage
    /// in front of a stream's first frame amounts to.
    pub frames: Option<Range<u64>>,
    /// Why this range produced nothing.
    pub reason: FlacSkipReason,
}

/// What a recovering decode found and what it did not.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct FlacRecoveryReport {
    /// How many interleaved samples were appended to the output: frames
    /// times channels, which is the length the output vector grew by.
    ///
    /// The gaps in [`skipped`](Self::skipped) count frames, not interleaved
    /// samples; the two units are named apart because confusing them is a
    /// silent factor of the channel count.
    pub samples: usize,
    /// Whether the audio was checked against a streaminfo MD5. Any lost or
    /// unaccounted audio makes this [`Md5Check::AudioIncomplete`], because a
    /// hash over a stream with a hole in it is not the hash the checksum
    /// describes.
    pub md5: Md5Check,
    /// Every range of input that produced no audio, in input order.
    ///
    /// Empty means the whole input decoded, which is a thing recovery is
    /// allowed to report and is the case for a clean bare frame stream.
    pub skipped: Vec<FlacSkip>,
    /// Where the first frame of the output sits in the stream's own
    /// numbering, or `None` when the stream's blocking strategy leaves it
    /// underdetermined.
    ///
    /// A position in the unit [`FlacSkip::frames`] counts: one interchannel
    /// sample, one value per channel, and not the interleaved samples
    /// [`samples`](Self::samples) counts. On a stereo stream `Some(10)` is
    /// the eleventh frame and the twenty-first interleaved sample. RFC 9639
    /// calls this position the sample number, its name for an interchannel
    /// position, so its sample number and this frame position are the same
    /// field under the two vocabularies.
    ///
    /// This is the datum a leading gap cannot always give: it is read
    /// straight off the coded number of the first frame recovered, so it is
    /// right whether or not the input begins where the stream does. `Some(0)`
    /// says the output starts at the stream's first frame and nothing at the
    /// front is missing.
    pub first_frame: Option<u64>,
    /// What the stream turned out to be: the streaminfo the caller supplied,
    /// or the properties derived from the first frame that was recovered.
    pub stream_info: FlacStreamInfo,
}

impl FlacRecoveryReport {
    /// How many frames were lost, in [`FlacSkip::frames`]'s unit of one
    /// interchannel sample per frame, or `None` when any skipped range's
    /// extent is unknown.
    ///
    /// Frames, not the interleaved samples [`samples`](Self::samples)
    /// counts. The two differ by the channel count, and the names are what
    /// say which is which: subtracting this from `samples` is subtracting
    /// two different units, and the names are there so that a reader of the
    /// call site can see it.
    ///
    /// `Some(0)` is the meaningful success case: bytes were skipped and none
    /// of them carried audio. `None` is not zero and must not be read as it:
    /// it says the decoder cannot account for what it did not get.
    pub fn frames_lost(&self) -> Option<u64> {
        let mut total = 0u64;
        for skip in &self.skipped {
            let range = skip.frames.as_ref()?;
            total += range.end.saturating_sub(range.start);
        }
        Some(total)
    }
}

/// Recovers whatever audio a run of bytes holds, which may begin part-way
/// through a frame, hold garbage, or be damaged in the middle.
///
/// # This is opt-in, and it is a separate type on purpose
///
/// Everywhere else in this crate a partial or damaged frame is an error
/// rather than quietly shortened audio, and recovery is where that promise
/// would be easiest to lose. So it is not a flag on the other readers and
/// there is no way to reach it by accident: a caller who wants recovery names
/// this type, and every decode through it returns a [`FlacRecoveryReport`]
/// that says exactly which bytes produced nothing and which sample positions
/// were lost. A recovering decoder that returned what it found and said
/// nothing about the rest would contradict the rest of the crate.
///
/// # Chained validation, and why one header is not evidence
///
/// The frame sync code is 14 bits and the header CRC-8 is 8, so about one
/// position in 256 of random data carries something that parses as a frame
/// header. A single validating header is therefore no evidence at all. A sync
/// point is accepted only when the frame at it decodes with a matching
/// CRC-16 **and** a further frame header parses at the offset that frame
/// predicts. The one case with no successor, a frame that ends exactly at
/// the end of the input, is accepted on its own CRC-16, because requiring a
/// successor there would discard the last frame of every stream.
///
/// # What it cannot recover
///
/// Nothing before the first sync point it can establish, nothing in a frame
/// whose CRC-16 fails, and, without a streaminfo block, nothing in a frame
/// that defers its sample rate or bit depth to one. Those frames are skipped
/// and reported, not guessed at.
///
/// # Recovery is whole-buffer only
///
/// This type takes a byte slice and works over all of it. **There is no
/// streaming form of it**, so a live stream that meets corruption part-way
/// through cannot be resynchronised in place: [`FlacStreamDecoder`] reports
/// the damage as a typed error and stops, and recovering what the rest of the
/// stream held means collecting the bytes and handing them here.
///
/// That is a limitation rather than a defect, and it is a consequence of how
/// a sync point is established above. Accepting one requires decoding the
/// frame at it *and* parsing a header where that frame ends, so the decision
/// is only makeable with the bytes after the candidate already in hand. A
/// streaming resync would have to buffer forward by an unbounded amount to
/// reach the same standard of evidence, or lower the standard, and a 14-bit
/// sync code with an 8-bit header CRC is not evidence on its own.
///
/// # Example
///
/// ```
/// use decibri_decode::{FlacRecovery, FlacSkipReason};
///
/// # let file: [u8; 73] = [
/// #     0x66, 0x4c, 0x61, 0x43, 0x80, 0x00, 0x00, 0x22, 0x10, 0x00, 0x10, 0x00,
/// #     0x00, 0x00, 0x1f, 0x00, 0x00, 0x1f, 0x07, 0xd0, 0x00, 0x70, 0x00, 0x00,
/// #     0x00, 0x18, 0xf8, 0xf9, 0xe3, 0x96, 0xf5, 0xcb, 0xcf, 0xc6, 0xdc, 0x80,
/// #     0x7f, 0x99, 0x77, 0x90, 0x6b, 0x32, 0xff, 0xf8, 0x68, 0x02, 0x00, 0x17,
/// #     0xe9, 0x44, 0x00, 0x4f, 0x6f, 0x31, 0x3d, 0x10, 0x47, 0xd2, 0x27, 0xcb,
/// #     0x6d, 0x09, 0x08, 0x31, 0x45, 0x2b, 0xdc, 0x28, 0x22, 0x22, 0x80, 0x57,
/// #     0xa3,
/// # ];
/// // Five bytes of junk, then RFC 9639 appendix D.3's frames.
/// let mut damaged = vec![0xFF, 0xF8, 0x00, 0x11, 0x22];
/// damaged.extend_from_slice(&file[42..]);
///
/// let mut samples = Vec::new();
/// let report = FlacRecovery::new(&damaged).decode(&mut samples)?;
///
/// assert_eq!(report.samples, 24);
/// assert_eq!(report.skipped.len(), 1);
/// assert_eq!(report.skipped[0].bytes, 0..5);
/// assert_eq!(report.skipped[0].reason, FlacSkipReason::NoSyncPoint);
/// // Junk in front of the first frame costs no audio, and says so.
/// assert_eq!(report.frames_lost(), Some(0));
/// # Ok::<(), decibri_decode::DecodeError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlacRecovery<'a> {
    bytes: &'a [u8],
    info: Option<FlacStreamInfo>,
}

/// A skipped range, before its sample positions can be worked out.
///
/// The conversion from a coded number to a sample position needs the stream's
/// block size, and for a fixed-blocksize stream that is only known once every
/// frame has been seen. So the scan records the coded numbers on each side of
/// a gap and the arithmetic happens at the end.
struct PendingSkip {
    bytes: Range<u64>,
    /// The coded number and block size of the last frame decoded before the
    /// gap, or `None` when nothing had been decoded yet.
    before: Option<(u64, u32)>,
    /// The coded number of the frame decoding resumed at, or `None` when the
    /// input ran out first.
    after: Option<u64>,
    reason: FlacSkipReason,
}

impl<'a> FlacRecovery<'a> {
    /// Recovers from `bytes`, taking the stream's properties from the first
    /// frame it manages to establish sync on.
    pub const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, info: None }
    }

    /// Recovers from `bytes` using a streaminfo block obtained elsewhere,
    /// parsed through [`FlacStreamInfo::from_block`].
    ///
    /// Supplying it resolves the two escape codes, and, when nothing turns
    /// out to have been lost, lets the checksum be verified even though the
    /// input needed recovering.
    pub const fn with_stream_info(bytes: &'a [u8], info: FlacStreamInfo) -> Self {
        Self {
            bytes,
            info: Some(info),
        }
    }

    /// Recovers every frame it can and appends the interleaved samples to
    /// `output`.
    ///
    /// The samples are contiguous; the holes are in the report, not in the
    /// buffer.
    ///
    /// # Errors
    ///
    /// [`DecodeError::Malformed`] when no sync point can be established
    /// anywhere in the input, which is the honest answer for a byte run that
    /// holds no FLAC frame. Damage *after* the first frame is reported rather
    /// than raised: that is what recovery is.
    pub fn decode(&self, output: &mut Vec<f32>) -> Result<FlacRecoveryReport, DecodeError> {
        let before = output.len();
        let end = self.bytes.len() as u64;
        let mut pending: Vec<PendingSkip> = Vec::new();

        // The scan's scratch, held across every candidate sync point rather
        // than built per candidate: see [`FrameDecoder::reset_to`] for what
        // that costs when it is not.
        let mut trial: Option<FrameDecoder> = None;
        let mut scratch: Vec<f32> = Vec::new();

        // The first sync point is established before the loop, so the loop
        // always holds a decoder and never has to ask whether it does. It is
        // also the one failure this path raises rather than reports: a byte
        // run with no frame in it anywhere is not damaged audio, it is not
        // audio.
        let Some((first, header)) = self.find_sync(0, &mut trial, &mut scratch) else {
            return Err(DecodeError::Malformed {
                expected: "at least one frame whose CRC-16 matches and which another frame follows",
                offset: 0,
            });
        };
        let info = match self.info {
            Some(info) => info,
            None => derive_stream_info(&header, first as u64)?,
        };
        let mut decoder = if self.info.is_some() {
            FrameDecoder::new(info)
        } else {
            FrameDecoder::from_first_frame(info)
        };
        // Pushed even when the input opens on a frame, because "no bytes were
        // skipped" is not the same as "no audio is missing": a run cut
        // exactly at a frame boundary loses everything before it and says so
        // through this gap's sample range. A gap that turns out to be empty
        // in both bytes and samples is dropped by `resolve_skips`.
        pending.push(PendingSkip {
            bytes: 0..first as u64,
            before: None,
            after: Some(header.coded_number),
            reason: FlacSkipReason::NoSyncPoint,
        });

        let mut blocking = None;
        let mut widest_block = 0u32;
        let mut last: Option<(u64, u32)> = None;
        let mut first_coded: Option<u64> = None;
        // The frame decoded immediately before the current one with no gap
        // between them, which is the pair `Numbering` reads the coding of the
        // number off, and what it concluded.
        let mut adjacent: Option<(u64, u32)> = None;
        let mut frame_numbered: Option<bool> = None;
        let mut at = first;
        let mut skip_from = first;
        let mut reason = FlacSkipReason::NoSyncPoint;
        let mut locked = true;

        while at < self.bytes.len() {
            if locked {
                let step = parse_frame_header(&self.bytes[at..], at as u64).and_then(|header| {
                    decoder
                        .decode_frame(&self.bytes[at..], at as u64, output)
                        .map(|consumed| (header, consumed))
                });
                match step {
                    Ok((header, consumed)) => {
                        blocking = Some(header.blocking);
                        widest_block = widest_block.max(header.block_size);
                        first_coded.get_or_insert(header.coded_number);
                        // Two frames back to back settle whether the coded
                        // number counts frames or samples, whatever the
                        // header's blocking bit claims.
                        if frame_numbered.is_none() {
                            if let Some((before, block)) = adjacent {
                                if header.coded_number == before + 1 {
                                    frame_numbered = Some(true);
                                } else if header.coded_number == before + u64::from(block) {
                                    frame_numbered = Some(false);
                                }
                            }
                        }
                        adjacent = Some((header.coded_number, header.block_size));
                        last = adjacent;
                        at += consumed;
                        skip_from = at;
                    }
                    Err(_) => {
                        // Sync is lost as of this frame. Everything from here
                        // to the next established sync point is a gap.
                        locked = false;
                        adjacent = None;
                        reason = FlacSkipReason::FrameRejected;
                        at += 1;
                    }
                }
                continue;
            }

            let Some((start, header)) = self.find_sync(at, &mut trial, &mut scratch) else {
                break;
            };
            pending.push(PendingSkip {
                bytes: skip_from as u64..start as u64,
                before: last,
                after: Some(header.coded_number),
                reason,
            });
            reason = FlacSkipReason::NoSyncPoint;
            at = start;
            skip_from = start;
            locked = true;
            adjacent = None;
        }

        if skip_from < self.bytes.len() {
            pending.push(PendingSkip {
                bytes: skip_from as u64..end,
                before: last,
                after: None,
                reason,
            });
        }
        let numbering = Numbering {
            // The measured answer where there was a pair to measure, and the
            // header's own claim where there was only one frame.
            frame_numbered: frame_numbered.unwrap_or(blocking == Some(Blocking::Fixed)),
            known: blocking.is_some(),
            widest: widest_block,
            total: info.total_samples,
            from_zero: self.info.is_some(),
        };
        let mut report = FlacRecoveryReport {
            samples: output.len() - before,
            // Replaced below. Starting from the pessimistic value means a
            // path added here that forgets to set it under-claims rather than
            // over-claims.
            md5: Md5Check::AudioIncomplete,
            skipped: resolve_skips(&pending, &numbering),
            first_frame: first_coded.and_then(|coded| numbering.position(coded)),
            stream_info: info,
        };

        // The checksum describes the whole stream, so it is only worth
        // checking when the whole stream is what was decoded, and `finish`
        // would otherwise reject a recovery for doing exactly what it was
        // asked to do.
        report.md5 = if report.frames_lost() != Some(0) {
            Md5Check::AudioIncomplete
        } else if self.info.is_none() {
            Md5Check::NoStreamInfo
        } else if info.md5.is_none() {
            decoder.finish(end)?;
            Md5Check::ChecksumUnset
        } else {
            decoder.finish(end)?;
            Md5Check::Verified
        };
        Ok(report)
    }

    /// Recovers the whole input, bound to the spec that describes it.
    ///
    /// # Errors
    ///
    /// As [`decode`](Self::decode).
    pub fn decode_to_end(&self) -> Result<(AudioBuffer, FlacRecoveryReport), DecodeError> {
        let mut samples = Vec::new();
        let report = self.decode(&mut samples)?;
        let spec = report.stream_info.spec;
        Ok((AudioBuffer::from_samples(spec, samples), report))
    }

    /// Finds the next position at or after `from` that survives chained
    /// validation, with the header it parsed there.
    ///
    /// The cheap test comes first and rejects all but about one position in
    /// 32,768: a frame begins `0xFF` followed by a byte whose top seven bits
    /// complete the sync code. Only then is a header parsed, and only then is
    /// the frame trial-decoded.
    ///
    /// `trial` and `scratch` are the caller's, reused across every candidate
    /// and across every call, for the reason recorded on
    /// [`FrameDecoder::reset_to`]. Each candidate is still judged in
    /// isolation: the decoder is reset to its just-constructed state before
    /// every trial, so what is shared is the allocation and nothing else.
    fn find_sync(
        &self,
        from: usize,
        trial: &mut Option<FrameDecoder>,
        scratch: &mut Vec<f32>,
    ) -> Option<(usize, FrameHeader)> {
        let last_start = self.bytes.len().checked_sub(MIN_FRAME_HEADER_BYTES)?;
        for start in from..=last_start {
            if self.bytes[start] != 0xFF || self.bytes[start + 1] & 0xFE != 0xF8 {
                continue;
            }
            let Ok(header) = parse_frame_header(&self.bytes[start..], start as u64) else {
                continue;
            };
            let info = match self.info {
                Some(info) => info,
                None => match derive_stream_info(&header, start as u64) {
                    Ok(info) => info,
                    // A frame deferring a field to a streaminfo block that is
                    // not there is not decodable, so it is not a sync point.
                    Err(_) => continue,
                },
            };
            let streaminfo = self.info.is_some();
            let trial = trial.get_or_insert_with(|| FrameDecoder::build(info, streaminfo));
            trial.reset_to(info, streaminfo);
            scratch.clear();
            let Ok(consumed) = trial.decode_frame(&self.bytes[start..], start as u64, scratch)
            else {
                continue;
            };
            let next = start + consumed;
            if next == self.bytes.len()
                || parse_frame_header(&self.bytes[next..], next as u64).is_ok()
            {
                return Some((start, header));
            }
        }
        None
    }
}

/// What the scan of the input established about the stream's numbering.
///
/// # Why the header bit is not taken at its word
///
/// RFC 9639 section 9.1.5 says a fixed-blocksize stream codes a frame number
/// and a variable-blocksize one codes a sample number. The bit that says
/// which was reserved in earlier versions of the format, and encoders of that
/// era wrote variable-blocksize streams with the bit clear and sample numbers
/// in the field. The conformance corpus carries one, written by Flake 0.11,
/// where reading the field as a frame number overstates a gap by a factor of
/// its block size.
///
/// So the two readings are told apart by measurement rather than by the bit:
/// two frames decoded back to back differ by one under frame numbering and by
/// a whole block under sample numbering, and a block is never one sample, so
/// the first consecutive pair settles it. The bit is only the starting guess,
/// used when the input held a single frame and there is no pair to look at.
struct Numbering {
    /// Whether the coded number is a frame number, as measured above.
    frame_numbered: bool,
    /// Whether either reading is available at all: without a decoded frame
    /// there is no blocking strategy to read.
    known: bool,
    /// The largest block size any recovered frame carried, which for a
    /// genuinely fixed-blocksize stream is its constant block size.
    widest: u32,
    /// The declared sample count, when there is one. The only thing that can
    /// close a gap running to the end of the input.
    total: Option<u64>,
    /// Whether the stream is known to be numbered from zero, which is true
    /// exactly when a streaminfo block described it: a stream with a
    /// streaminfo block starts where the stream starts.
    from_zero: bool,
}

impl Numbering {
    /// The interchannel sample position a coded number names.
    fn position(&self, coded: u64) -> Option<u64> {
        match (self.known, self.frame_numbered) {
            (false, _) => None,
            (true, false) => Some(coded),
            (true, true) if self.widest > 0 => Some(coded * u64::from(self.widest)),
            (true, true) => None,
        }
    }
}

/// Turns the coded numbers recorded either side of each gap into sample
/// positions.
fn resolve_skips(pending: &[PendingSkip], numbering: &Numbering) -> Vec<FlacSkip> {
    pending
        .iter()
        .map(|skip| {
            let start = match skip.before {
                Some((coded, block)) => numbering.position(coded).map(|at| at + u64::from(block)),
                // Nothing was decoded before this gap. Where the stream's
                // numbering begins is only knowable if something said so.
                None if numbering.from_zero => Some(0),
                None => match skip.after.and_then(|coded| numbering.position(coded)) {
                    // Unless the frame it resumed at is the stream's own
                    // first, in which case the gap plainly cost nothing.
                    Some(0) => Some(0),
                    _ => None,
                },
            };
            let end = match skip.after {
                Some(coded) => numbering.position(coded),
                None => numbering.total,
            };
            FlacSkip {
                bytes: skip.bytes.clone(),
                frames: match (start, end) {
                    (Some(start), Some(end)) => Some(start..end.max(start)),
                    _ => None,
                },
                reason: skip.reason,
            }
        })
        // A gap that cost no bytes and no frames is not a gap. This drops
        // the placeholder the scan always records at the front, for the
        // ordinary case where the input opens on the stream's first frame.
        .filter(|skip| {
            !(skip.bytes.is_empty() && skip.frames.as_ref().is_some_and(Range::is_empty))
        })
        .collect()
}

// -- The streaming reader -----------------------------------------------------

/// Where the streaming reader is in the stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Waiting for the four-byte `fLaC` signature.
    Signature,
    /// Waiting for a four-byte metadata block header.
    BlockHeader,
    /// Buffering the streaminfo body, the one metadata body that is read.
    StreaminfoBody { last: bool, offset: u64 },
    /// Discarding a metadata body that does not have to be read.
    SkipBlock { last: bool, left: u64 },
    /// Buffering and decoding audio frames.
    Audio,
    /// Past the end of the declared audio. Everything after it is discarded.
    Done,
}

/// Reads a FLAC stream that arrives in pieces.
///
/// The [`StreamSource`] half of this module, with the same shape and the same
/// bounds as [`WavStreamDecoder`](crate::WavStreamDecoder) and
/// [`AiffStreamDecoder`](crate::AiffStreamDecoder). Nothing is buffered in
/// proportion to a size the file declared: the signature and block headers
/// are buffered, the streaminfo body is buffered at its fixed 34 bytes, every
/// other metadata body is discarded as it flows past, and one frame at a time
/// is held while it is assembled, capped at 2,101,216 bytes, which is the
/// largest frame the format can express plus room for its headers, and is a
/// figure stated in this crate rather than read from a file.
///
/// The MD5 is computed as frames are decoded and verified by
/// [`finish`](StreamSource::finish), so a streaming decode has the same
/// end-to-end check a whole-file decode has.
///
/// # Example
///
/// ```
/// use decibri_decode::{FlacStreamDecoder, StreamSource};
///
/// # let file: [u8; 73] = [
/// #     0x66, 0x4c, 0x61, 0x43, 0x80, 0x00, 0x00, 0x22, 0x10, 0x00, 0x10, 0x00,
/// #     0x00, 0x00, 0x1f, 0x00, 0x00, 0x1f, 0x07, 0xd0, 0x00, 0x70, 0x00, 0x00,
/// #     0x00, 0x18, 0xf8, 0xf9, 0xe3, 0x96, 0xf5, 0xcb, 0xcf, 0xc6, 0xdc, 0x80,
/// #     0x7f, 0x99, 0x77, 0x90, 0x6b, 0x32, 0xff, 0xf8, 0x68, 0x02, 0x00, 0x17,
/// #     0xe9, 0x44, 0x00, 0x4f, 0x6f, 0x31, 0x3d, 0x10, 0x47, 0xd2, 0x27, 0xcb,
/// #     0x6d, 0x09, 0x08, 0x31, 0x45, 0x2b, 0xdc, 0x28, 0x22, 0x22, 0x80, 0x57,
/// #     0xa3,
/// # ];
/// // `file` is RFC 9639 appendix D.3's worked example.
/// let mut stream = FlacStreamDecoder::new();
/// let mut samples = Vec::new();
/// for piece in file.chunks(5) {
///     let mut offset = 0;
///     while offset < piece.len() {
///         offset += stream.push(&piece[offset..])?;
///         while stream.pull(&mut samples, usize::MAX)? > 0 {}
///     }
/// }
/// stream.finish(&mut samples)?;
///
/// assert_eq!(stream.spec().map(|spec| spec.sample_rate), Some(32_000));
/// assert_eq!(samples.len(), 24);
/// # Ok::<(), decibri_decode::DecodeError>(())
/// ```
#[derive(Debug)]
pub struct FlacStreamDecoder {
    state: State,
    /// Bytes of a header, the streaminfo body or a frame that have not fully
    /// arrived. Bounded by [`MAX_FRAME_BYTES`].
    pending: Vec<u8>,
    /// Byte offset of the next byte to arrive, for error reporting.
    offset: u64,
    /// Where the frame currently being assembled starts in the input.
    frame_offset: u64,
    /// How many bytes of a frame must be held before decoding is worth
    /// retrying, from the last truncation.
    frame_needs: usize,
    info: Option<FlacStreamInfo>,
    md5_offset: u64,
    decoder: Option<FrameDecoder>,
    /// Whether `info`, once it arrives, is a streaminfo block rather than
    /// properties derived from the first frame. Set by the bare frame
    /// constructors; see [`FrameDecoder::streaminfo`].
    streaminfo: bool,
    /// What [`finish`](StreamSource::finish) did about the checksum, once it
    /// has run.
    md5: Option<Md5Check>,
    /// Decoded samples not yet pulled, and how far into them the caller is.
    ready: Vec<f32>,
    ready_read: usize,
    finished: bool,
    /// Which constructor built this reader, so `reset` returns it to that
    /// starting state rather than to one waiting for a signature it will
    /// never be sent.
    origin: Origin,
}

/// Which of [`FlacStreamDecoder`]'s three constructors built a reader.
#[derive(Debug, Clone, Copy)]
enum Origin {
    /// A whole FLAC stream, signature first.
    Stream,
    /// Bare frames, with the properties to come from the first of them.
    BareFrames,
    /// Bare frames described by a streaminfo block from out of band.
    BareFramesWith(FlacStreamInfo),
}

impl Default for FlacStreamDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl FlacStreamDecoder {
    /// A reader waiting for the first byte of a stream.
    pub fn new() -> Self {
        Self::from_origin(Origin::Stream)
    }

    /// A reader for a **bare frame stream**: no `fLaC` signature and no
    /// metadata, the first byte pushed is the first byte of a frame header.
    ///
    /// The streaming half of [`FlacFrameReader`], with the same properties
    /// and the same two escape-code rejections. [`spec`](StreamSource::spec)
    /// and [`stream_info`](Self::stream_info) answer `None` until the first
    /// frame header has arrived, which is what they already mean on this type
    /// and why a bare stream needs no separate way of saying it.
    pub fn frames() -> Self {
        Self::from_origin(Origin::BareFrames)
    }

    /// A reader for a bare frame stream described by a streaminfo block the
    /// caller obtained elsewhere.
    ///
    /// This is the Ogg and Matroska shape: streaminfo in the container's
    /// codec-private header, bare frames in the packets. The block's bytes
    /// become an `info` through [`FlacStreamInfo::from_block`]. It resolves
    /// the two escape codes and restores the MD5 check to a path that
    /// otherwise has none. See [`md5_check`](Self::md5_check), which is how
    /// a caller finds out whether that check actually ran.
    pub fn frames_with_stream_info(info: FlacStreamInfo) -> Self {
        Self::from_origin(Origin::BareFramesWith(info))
    }

    /// The three constructors' shared body, and what
    /// [`reset`](StreamSource::reset) rebuilds from.
    fn from_origin(origin: Origin) -> Self {
        let mut reader = Self {
            state: State::Signature,
            pending: Vec::new(),
            offset: 0,
            frame_offset: 0,
            frame_needs: MIN_FRAME_HEADER_BYTES,
            info: None,
            md5_offset: 0,
            decoder: None,
            streaminfo: true,
            md5: None,
            ready: Vec::new(),
            ready_read: 0,
            finished: false,
            origin,
        };
        match origin {
            Origin::Stream => {}
            Origin::BareFrames => {
                reader.state = State::Audio;
                reader.streaminfo = false;
            }
            Origin::BareFramesWith(info) => {
                reader.state = State::Audio;
                reader.info = Some(info);
                reader.decoder = Some(FrameDecoder::new(info));
            }
        }
        reader
    }

    /// What the streaminfo metadata block declared, once it has arrived.
    ///
    /// For a bare frame stream this is the description derived from the first
    /// frame, or the one the caller supplied.
    pub const fn stream_info(&self) -> Option<FlacStreamInfo> {
        self.info
    }

    /// What [`finish`](StreamSource::finish) did about the streaminfo MD5, or
    /// `None` before it has run.
    ///
    /// `None` is not a verdict. The check happens at the end of the stream
    /// and this reports what happened, never what is expected to happen, so a
    /// caller cannot read a guarantee off a decode that has not finished.
    pub const fn md5_check(&self) -> Option<Md5Check> {
        self.md5
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

    /// Reads a buffered metadata block header and decides what to do with the
    /// body.
    fn start_block(&mut self) -> Result<(), DecodeError> {
        let at = self.offset - METADATA_HEADER_BYTES as u64;
        let mut header = [0u8; METADATA_HEADER_BYTES];
        header.copy_from_slice(&self.pending);
        self.pending.clear();
        let last = header[0] & 0x80 != 0;
        let kind = header[0] & 0x7F;
        let length =
            (u64::from(header[1]) << 16) | (u64::from(header[2]) << 8) | u64::from(header[3]);

        if kind == BLOCK_FORBIDDEN {
            return Err(DecodeError::Malformed {
                expected: "a metadata block type other than the forbidden 127",
                offset: at,
            });
        }
        if kind == BLOCK_STREAMINFO {
            if self.info.is_some() {
                return Err(DecodeError::Malformed {
                    expected: "exactly one streaminfo metadata block",
                    offset: at,
                });
            }
            if length != STREAMINFO_BYTES as u64 {
                return Err(DecodeError::Malformed {
                    expected: "a streaminfo block of exactly 34 bytes",
                    offset: at,
                });
            }
            self.state = State::StreaminfoBody {
                last,
                offset: self.offset,
            };
            return Ok(());
        }
        if self.info.is_none() {
            return Err(DecodeError::Malformed {
                expected: "a streaminfo block first among the metadata blocks",
                offset: at,
            });
        }
        self.state = if length == 0 {
            self.enter_audio(last)
        } else {
            State::SkipBlock { last, left: length }
        };
        Ok(())
    }

    /// The state to move to once a metadata block has been stepped over.
    fn enter_audio(&mut self, last: bool) -> State {
        if last {
            self.frame_offset = self.offset;
            self.frame_needs = MIN_FRAME_HEADER_BYTES;
            State::Audio
        } else {
            State::BlockHeader
        }
    }

    /// Reads the buffered streaminfo body.
    fn finish_streaminfo(&mut self, last: bool, offset: u64) -> Result<(), DecodeError> {
        let info = parse_streaminfo(&self.pending, offset)?;
        self.pending.clear();
        self.info = Some(info);
        self.md5_offset = offset + 18;
        self.decoder = Some(FrameDecoder::new(info));
        self.state = self.enter_audio(last);
        Ok(())
    }

    /// Decodes as many whole frames out of `pending` as it holds.
    ///
    /// A frame has no length field, so the only way to find its end is to
    /// decode it. A truncation is not a failure here: it says the rest has
    /// not arrived, and the byte count it carries is remembered so the next
    /// attempt waits until it could possibly succeed rather than re-parsing
    /// the same partial frame on every push.
    fn drain_frames(&mut self) -> Result<(), DecodeError> {
        while self.pending.len() >= self.frame_needs {
            // A bare frame stream has no streaminfo block, so the first frame
            // header is where the stream's properties come from. Every other
            // path has a decoder before it ever reaches this state.
            if self.decoder.is_none() {
                match parse_frame_header(&self.pending, self.frame_offset) {
                    Ok(header) => {
                        let info = derive_stream_info(&header, self.frame_offset)?;
                        self.info = Some(info);
                        self.decoder = Some(FrameDecoder::from_first_frame(info));
                    }
                    Err(DecodeError::Truncated { expected, .. }) => {
                        self.frame_needs = (expected as usize).max(self.pending.len() + 1);
                        return Ok(());
                    }
                    Err(other) => return Err(other),
                }
            }
            let Some(decoder) = self.decoder.as_mut() else {
                return Ok(());
            };
            if let Some(info) = self.info {
                if info
                    .total_samples
                    .is_some_and(|total| decoder.samples_done >= total)
                {
                    self.state = State::Done;
                    self.pending.clear();
                    return Ok(());
                }
            }
            match decoder.decode_frame(&self.pending, self.frame_offset, &mut self.ready) {
                Ok(consumed) => {
                    self.pending.drain(..consumed);
                    self.frame_offset += consumed as u64;
                    self.frame_needs = MIN_FRAME_HEADER_BYTES;
                }
                Err(DecodeError::Truncated { expected, .. }) => {
                    // Wait for at least one more byte than is held, and for
                    // at least as many as the parse said it wanted.
                    self.frame_needs = (expected as usize).max(self.pending.len() + 1);
                    if self.frame_needs > MAX_FRAME_BYTES {
                        return Err(DecodeError::Malformed {
                            expected: "a frame no larger than the format's own maximum",
                            offset: self.frame_offset,
                        });
                    }
                    return Ok(());
                }
                Err(other) => return Err(other),
            }
        }
        Ok(())
    }

    /// The body of [`push`](StreamSource::push), split out so a failure can
    /// set the finished flag in one place.
    fn push_inner(&mut self, bytes: &[u8], taken: &mut usize) -> Result<(), DecodeError> {
        while *taken < bytes.len() {
            let rest = &bytes[*taken..];
            match self.state {
                State::Signature => {
                    if !self.accumulate(rest, 4, taken) {
                        break;
                    }
                    let tag = FourCc([
                        self.pending[0],
                        self.pending[1],
                        self.pending[2],
                        self.pending[3],
                    ]);
                    self.pending.clear();
                    if tag != MAGIC {
                        return Err(DecodeError::UnsupportedContainer { tag });
                    }
                    self.state = State::BlockHeader;
                }
                State::BlockHeader => {
                    if !self.accumulate(rest, METADATA_HEADER_BYTES, taken) {
                        break;
                    }
                    self.start_block()?;
                }
                State::StreaminfoBody { last, offset } => {
                    if !self.accumulate(rest, STREAMINFO_BYTES, taken) {
                        break;
                    }
                    self.finish_streaminfo(last, offset)?;
                }
                State::SkipBlock { last, left } => {
                    let step = left.min(rest.len() as u64);
                    *taken += step as usize;
                    self.offset += step;
                    let left = left - step;
                    self.state = if left == 0 {
                        self.enter_audio(last)
                    } else {
                        State::SkipBlock { last, left }
                    };
                }
                State::Audio => {
                    if self.ready.len() - self.ready_read >= READY_LIMIT {
                        // Back-pressure: the caller has to pull before the
                        // reader will take more.
                        break;
                    }
                    let room = MAX_FRAME_BYTES - self.pending.len();
                    if room == 0 {
                        return Err(DecodeError::Malformed {
                            expected: "a frame no larger than the format's own maximum",
                            offset: self.frame_offset,
                        });
                    }
                    let step = room.min(rest.len());
                    self.pending.extend_from_slice(&rest[..step]);
                    *taken += step;
                    self.offset += step as u64;
                    self.drain_frames()?;
                }
                State::Done => {
                    // A trailing tag is not this reader's business; the audio
                    // it was promised is complete.
                    self.offset += (bytes.len() - *taken) as u64;
                    *taken = bytes.len();
                }
            }
        }
        Ok(())
    }
}

impl StreamSource for FlacStreamDecoder {
    fn push(&mut self, bytes: &[u8]) -> Result<usize, DecodeError> {
        if self.finished {
            return Ok(0);
        }
        let mut taken = 0;
        let result = self.push_inner(bytes, &mut taken);
        if result.is_err() {
            // A stream that has failed structurally is over, for the reason
            // recorded on the WAV reader: a caller who keeps pushing must not
            // get a second, different answer from the same stream.
            self.finished = true;
        }
        result.map(|()| taken)
    }

    fn pull(&mut self, output: &mut Vec<f32>, max_frames: usize) -> Result<usize, DecodeError> {
        let Some(info) = self.info else {
            return Ok(0);
        };
        let channels = usize::from(info.spec.channels);
        let available = self.ready.len() - self.ready_read;
        let frames = (available / channels).min(max_frames);
        let count = frames * channels;
        output.extend_from_slice(&self.ready[self.ready_read..self.ready_read + count]);
        self.ready_read += count;
        // Amortised compaction, exactly as the other two readers do it.
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
        self.info.map(|info| info.spec)
    }

    fn buffered_bytes(&self) -> usize {
        self.pending.len()
    }

    fn finish(&mut self, output: &mut Vec<f32>) -> Result<usize, DecodeError> {
        if self.finished {
            return Ok(0);
        }
        self.finished = true;

        match self.state {
            State::Done => {}
            State::Signature => {
                return Err(DecodeError::Truncated {
                    expected: 4,
                    available: self.pending.len() as u64,
                })
            }
            State::BlockHeader => {
                return Err(DecodeError::Truncated {
                    expected: METADATA_HEADER_BYTES as u64,
                    available: self.pending.len() as u64,
                })
            }
            State::StreaminfoBody { .. } => {
                return Err(DecodeError::Truncated {
                    expected: STREAMINFO_BYTES as u64,
                    available: self.pending.len() as u64,
                })
            }
            State::SkipBlock { left, .. } => {
                return Err(DecodeError::Truncated {
                    expected: left,
                    available: 0,
                })
            }
            State::Audio => {
                // A stream can end on a frame boundary; anything left over is
                // a frame that never completed.
                if !self.pending.is_empty() {
                    return Err(DecodeError::Truncated {
                        expected: self.frame_needs as u64,
                        available: self.pending.len() as u64,
                    });
                }
            }
        }

        if let Some(decoder) = self.decoder.take() {
            // A supplied streaminfo block is not in this input, so a checksum
            // rejection names where the decode concluded rather than a byte
            // that does not exist. The metadata walk sets `md5_offset` to a
            // real position, never zero, since a streaminfo body cannot
            // start before byte 26, and the bare frame constructors leave it
            // at zero.
            let at = if self.md5_offset == 0 {
                self.offset
            } else {
                self.md5_offset
            };
            decoder.finish(at)?;
        }
        let Some(info) = self.info else {
            return Ok(0);
        };
        // Set only once `finish` has actually run the check, so `md5_check`
        // reports what happened rather than what was expected to.
        self.md5 = Some(if !self.streaminfo {
            Md5Check::NoStreamInfo
        } else if info.md5.is_none() {
            Md5Check::ChecksumUnset
        } else {
            Md5Check::Verified
        });
        let channels = usize::from(info.spec.channels);
        let frames = (self.ready.len() - self.ready_read) / channels;
        output.extend_from_slice(&self.ready[self.ready_read..self.ready_read + frames * channels]);
        self.ready.clear();
        self.ready_read = 0;
        Ok(frames)
    }

    fn reset(&mut self) {
        *self = Self::from_origin(self.origin);
    }
}

// -- Deterministic mathematics for the writer ---------------------------------
//
// The writer's searches need a cosine (for the analysis window) and a base-2
// logarithm (for size estimation). `f64::cos` and `f64::ln` are libm calls
// whose last bits vary between platforms, which would put the crate's
// byte-identical claim at the mercy of whichever libm linked, so both are
// computed here from arithmetic IEEE 754 fully specifies: add, subtract,
// multiply and divide. Accuracy in the last few bits does not matter for a
// window shape or a search heuristic; producing the same bits everywhere
// does.

/// `cos(pi * x)` for `x` in `[0, 1]`, by Taylor series after folding onto
/// `[0, 0.5]`.
///
/// Ten terms leave the truncation error near 1e-12 at the fold point, far
/// below anything a window shape can express in encoded bytes, and every
/// operation is exactly rounded per IEEE 754, so the result is bit-identical
/// on every target.
fn det_cos_pi(x: f64) -> f64 {
    let (x, sign) = if x > 0.5 { (1.0 - x, -1.0) } else { (x, 1.0) };
    let theta = x * std::f64::consts::PI;
    let z = theta * theta;
    let mut sum = 1.0f64;
    let mut term = 1.0f64;
    for k in 1i32..=10 {
        term *= -z / f64::from((2 * k - 1) * (2 * k));
        sum += term;
    }
    sign * sum
}

/// `log2(x)` for positive finite `x`, from the exponent field and an atanh
/// series over the mantissa.
///
/// The mantissa lands in `[1, 2)`, so the series variable is at most `1/3`
/// and twelve terms leave the error near 1e-12. Used only to rank predictor
/// orders, where being deterministic matters and the last bits do not.
fn det_log2(x: f64) -> f64 {
    debug_assert!(x > 0.0 && x.is_finite());
    let bits = x.to_bits();
    let raw_exponent = (bits >> 52) & 0x7FF;
    if raw_exponent == 0 {
        // Subnormal: scale into the normal range and correct afterwards.
        return det_log2(x * (1u64 << 60) as f64) - 60.0;
    }
    let exponent = raw_exponent as i64 - 1023;
    let mantissa = f64::from_bits((bits & ((1u64 << 52) - 1)) | (1023u64 << 52));
    let z = (mantissa - 1.0) / (mantissa + 1.0);
    let z_squared = z * z;
    let mut term = z;
    let mut sum = 0.0f64;
    for k in 0i32..12 {
        sum += term / f64::from(2 * k + 1);
        term *= z_squared;
    }
    exponent as f64 + 2.0 * sum * std::f64::consts::LOG2_E
}

// -- The bit writer -----------------------------------------------------------

/// A most-significant-bit-first writer, the mirror of [`BitReader`].
///
/// Bits accumulate right-aligned in a 64-bit word and flush to bytes as they
/// fill. Every write is at most [`WINDOW_BITS`] bits, so with at most seven
/// held bits the accumulator cannot overflow.
struct BitWriter {
    out: Vec<u8>,
    /// Bits not yet flushed, right-aligned. At most seven after any write.
    accumulator: u64,
    /// How many bits `accumulator` holds.
    filled: u32,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            out: Vec::new(),
            accumulator: 0,
            filled: 0,
        }
    }

    /// Appends the low `count` bits of `value`, most significant first.
    fn write(&mut self, value: u64, count: u32) {
        debug_assert!((1..=WINDOW_BITS).contains(&count));
        debug_assert!(count == 64 || value >> count == 0);
        self.accumulator = (self.accumulator << count) | value;
        self.filled += count;
        while self.filled >= 8 {
            self.filled -= 8;
            self.out.push((self.accumulator >> self.filled) as u8);
        }
    }

    /// Appends `value` as `count` bits of two's complement.
    fn write_signed(&mut self, value: i64, count: u32) {
        let mask = if count >= 64 {
            u64::MAX
        } else {
            (1u64 << count) - 1
        };
        self.write(value as u64 & mask, count);
    }

    /// Appends `quotient` zero bits and a terminating one bit, the unary
    /// form [`BitReader::read_unary`] reads.
    fn write_unary(&mut self, quotient: u64) {
        let mut left = quotient;
        while left >= 32 {
            self.write(0, 32);
            left -= 32;
        }
        // A one in `left + 1` bits is exactly `left` zeros then the one.
        self.write(1, left as u32 + 1);
    }

    /// Pads with zero bits to the next byte boundary.
    fn align(&mut self) {
        if self.filled > 0 {
            self.write(0, 8 - self.filled);
        }
    }

    /// The bytes written so far. Only meaningful at a byte boundary, which
    /// is where both CRC computations happen.
    fn bytes(&self) -> &[u8] {
        debug_assert_eq!(self.filled, 0);
        &self.out
    }

    /// The finished bytes, aligned.
    fn into_bytes(mut self) -> Vec<u8> {
        self.align();
        self.out
    }
}

/// Writes RFC 9639 section 9.1.5's UTF-8-like coded number, the shortest
/// form that holds `value`.
///
/// The mirror of [`read_coded_number`], for values up to 36 bits.
fn write_coded_number(writer: &mut BitWriter, value: u64) {
    debug_assert!(value < 1u64 << 36);
    if value < 0x80 {
        writer.write(value, 8);
        return;
    }
    // Two through seven octets carry 11, 16, 21, 26, 31 and 36 payload bits.
    let mut octets = 2u32;
    while octets < 7 && value >= 1u64 << (6 * (octets - 1) + (7 - octets)) {
        octets += 1;
    }
    let lead = (0xFFu64 << (8 - octets)) & 0xFF;
    writer.write(lead | (value >> (6 * (octets - 1))), 8);
    for step in (0..octets - 1).rev() {
        writer.write(0x80 | ((value >> (6 * step)) & 0x3F), 8);
    }
}

// -- What a compression level means -------------------------------------------

/// How the writer chooses between the stereo decorrelations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stereo {
    /// Two independent subframes, no search.
    Independent,
    /// The four modes ranked by a cheap estimate over each candidate
    /// channel's second difference, and only the winner's two subframes
    /// actually searched.
    Estimated,
    /// All four candidate channels searched in full and the cheapest pair
    /// taken, at twice the subframe work of [`Stereo::Estimated`].
    Exhaustive,
}

/// The analysis windows a level applies before autocorrelation.
///
/// A rectangular window leaks the block's edges into every lag, so some
/// taper is always worth its cost; which taper wins depends on the material,
/// which is why the upper levels try more than one and keep the cheapest
/// result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WindowShape {
    /// Tukey with a quarter of the block tapered: flat over the middle half.
    Tukey50,
    /// Hann, a Tukey window whose taper is the whole block.
    Hann,
    /// Tukey with an eighth of the block tapered at each end.
    Tukey25,
}

impl WindowShape {
    /// The window as coefficients for a block of `len` samples.
    fn build(self, len: usize) -> Vec<f64> {
        let alpha = match self {
            Self::Tukey50 => 0.5,
            Self::Hann => 1.0,
            Self::Tukey25 => 0.25,
        };
        if len <= 1 {
            return vec![1.0; len];
        }
        let taper = alpha * (len - 1) as f64 / 2.0;
        (0..len)
            .map(|index| {
                let edge = (index.min(len - 1 - index)) as f64;
                if taper <= 0.0 || edge >= taper {
                    1.0
                } else {
                    0.5 * (1.0 + det_cos_pi(1.0 - edge / taper))
                }
            })
            .collect()
    }
}

/// What one compression level searches.
///
/// The levels bound the same search rather than switching algorithms, which
/// is what keeps their outputs mutually decodable and their meaning "spend
/// more time, get smaller files" and nothing else.
struct Level {
    /// Samples per frame, except the last.
    block_size: u32,
    /// How stereo decorrelation is chosen.
    stereo: Stereo,
    /// The highest linear predictor order tried, zero meaning fixed
    /// predictors only.
    max_lpc_order: usize,
    /// The deepest residual partitioning tried.
    max_partition_order: u32,
    /// The analysis windows tried.
    windows: &'static [WindowShape],
    /// How many predictor orders per window are exactly costed rather than
    /// only estimated.
    lpc_candidates: usize,
}

/// The nine levels. Block sizes, predictor bounds and partition depths
/// broadly follow the reference encoder's documented presets: 1152-sample
/// blocks with fixed predictors below level 3, 4096-sample blocks with LPC
/// at 3 and above, order 8 at the middle levels and 12 at the top, and a
/// widening stereo and window search.
const LEVELS: [Level; 9] = [
    Level {
        block_size: 1152,
        stereo: Stereo::Independent,
        max_lpc_order: 0,
        max_partition_order: 3,
        windows: &[],
        lpc_candidates: 0,
    },
    Level {
        block_size: 1152,
        stereo: Stereo::Estimated,
        max_lpc_order: 0,
        max_partition_order: 3,
        windows: &[],
        lpc_candidates: 0,
    },
    Level {
        block_size: 1152,
        stereo: Stereo::Exhaustive,
        max_lpc_order: 0,
        max_partition_order: 3,
        windows: &[],
        lpc_candidates: 0,
    },
    Level {
        block_size: 4096,
        stereo: Stereo::Independent,
        max_lpc_order: 6,
        max_partition_order: 4,
        windows: &[WindowShape::Tukey50],
        lpc_candidates: 1,
    },
    Level {
        block_size: 4096,
        stereo: Stereo::Estimated,
        max_lpc_order: 8,
        max_partition_order: 4,
        windows: &[WindowShape::Tukey50],
        lpc_candidates: 1,
    },
    Level {
        block_size: 4096,
        stereo: Stereo::Exhaustive,
        max_lpc_order: 8,
        max_partition_order: 5,
        windows: &[WindowShape::Tukey50],
        lpc_candidates: 1,
    },
    Level {
        block_size: 4096,
        stereo: Stereo::Exhaustive,
        max_lpc_order: 8,
        max_partition_order: 6,
        windows: &[WindowShape::Tukey50],
        lpc_candidates: 2,
    },
    Level {
        block_size: 4096,
        stereo: Stereo::Exhaustive,
        max_lpc_order: 12,
        max_partition_order: 6,
        windows: &[WindowShape::Tukey50, WindowShape::Hann],
        lpc_candidates: 2,
    },
    Level {
        block_size: 4096,
        stereo: Stereo::Exhaustive,
        max_lpc_order: 12,
        max_partition_order: 6,
        windows: &[
            WindowShape::Tukey50,
            WindowShape::Hann,
            WindowShape::Tukey25,
        ],
        lpc_candidates: 3,
    },
];

/// The quantised coefficient precision written for every LPC subframe.
///
/// Fifteen bits is the most the format's streamable subset permits and what
/// the reference encoder writes at 4096-sample blocks. The header cost is
/// `order * 15` bits per subframe, under half a percent of a typical frame,
/// and precision below this measurably loses prediction accuracy.
const COEFFICIENT_PRECISION: u32 = 15;

/// The largest Rice parameter the 4-bit form can code. 15 is the escape.
const RICE4_MAX: u32 = 14;

/// The largest Rice parameter the 5-bit form can code. 31 is the escape.
const RICE5_MAX: u32 = 30;

// -- The residual search ------------------------------------------------------

/// The best Rice partitioning found for one subframe's residual.
#[derive(Debug, Clone)]
struct ResidualPlan {
    /// The residual section's total cost in bits: method, partition order,
    /// every parameter field and every coded residual.
    bits: u64,
    /// The partition order to write.
    partition_order: u32,
    /// Whether the parameters use the 5-bit form.
    five_bit: bool,
    /// The Rice parameter for each partition, in order.
    parameters: Vec<u32>,
}

/// The zigzag fold RFC 9639 section 9.2.7.4 stores residuals in:
/// non-negative values to even numbers, negative to odd.
fn zigzag(residual: i64) -> u64 {
    (residual.wrapping_shl(1) ^ (residual >> 63)) as u64
}

/// Whether every residual fits the format's 32-bit folded range, which
/// RFC 9639 section 9.2.7.3 requires. A subframe that does not is not
/// Rice-codable and falls back to verbatim.
fn residual_in_range(residual: &[i64]) -> bool {
    residual
        .iter()
        .all(|&value| value.unsigned_abs() <= i32::MAX as u64)
}

/// Finds the cheapest partition order and Rice parameters for `folded`,
/// which is the zigzag residual of a predictor of `order` over a block of
/// `block_size` samples.
///
/// Exact rather than estimated: per-partition sums of `value >> p` are
/// accumulated once at the deepest feasible order and merged pairwise
/// upward, so every (partition order, parameter) cell's true bit cost is
/// known and the minimum is the minimum, not a guess. The per-value work is
/// proportional to the bits the value will actually occupy, so this costs
/// about what writing the residual costs.
fn plan_residual(folded: &[u64], block_size: usize, order: usize, max_po: u32) -> ResidualPlan {
    // The deepest order that divides the block and leaves the first
    // partition at least one residual.
    let mut po = max_po.min(block_size.trailing_zeros());
    while po > 0 && (block_size >> po) <= order {
        po -= 1;
    }

    let mut partitions = 1usize << po;
    let part_len = block_size >> po;
    let mut counts = vec![0u64; partitions];
    // sums[j][p] is the sum of `value >> p` over partition j. Entries past
    // RICE5_MAX are never read: a value needing a larger parameter simply
    // contributes its full shifted magnitude at RICE5_MAX.
    let mut sums = vec![[0u64; (RICE5_MAX + 1) as usize]; partitions];
    for (index, &value) in folded.iter().enumerate() {
        let j = (order + index) / part_len;
        counts[j] += 1;
        let mut shifted = value;
        let mut p = 0usize;
        while shifted > 0 && p <= RICE5_MAX as usize {
            sums[j][p] += shifted;
            shifted >>= 1;
            p += 1;
        }
    }

    let mut best: Option<ResidualPlan> = None;
    loop {
        let mut total4 = 2u64 + 4;
        let mut total5 = 2u64 + 4;
        let mut params4 = Vec::with_capacity(partitions);
        let mut params5 = Vec::with_capacity(partitions);
        for j in 0..partitions {
            let mut best4 = u64::MAX;
            let mut best4_p = 0u32;
            let mut best5 = u64::MAX;
            let mut best5_p = 0u32;
            for p in 0..=RICE5_MAX {
                let cost = counts[j] * u64::from(p + 1) + sums[j][p as usize];
                if p <= RICE4_MAX && cost < best4 {
                    best4 = cost;
                    best4_p = p;
                }
                if cost < best5 {
                    best5 = cost;
                    best5_p = p;
                }
            }
            total4 = total4.saturating_add(4 + best4);
            total5 = total5.saturating_add(5 + best5);
            params4.push(best4_p);
            params5.push(best5_p);
        }
        let (total, five_bit, params) = if total4 <= total5 {
            (total4, false, params4)
        } else {
            (total5, true, params5)
        };
        if best.as_ref().is_none_or(|held| total < held.bits) {
            best = Some(ResidualPlan {
                bits: total,
                partition_order: po,
                five_bit,
                parameters: params,
            });
        }

        if po == 0 {
            break;
        }
        // A partition at the next order up is the union of an adjacent pair
        // at this one, so counts and sums merge pairwise; nothing is
        // rescanned.
        po -= 1;
        partitions /= 2;
        for j in 0..partitions {
            counts[j] = counts[2 * j] + counts[2 * j + 1];
            let left = sums[2 * j];
            let right = sums[2 * j + 1];
            for (slot, (a, b)) in sums[j].iter_mut().zip(left.iter().zip(right.iter())) {
                *slot = a + b;
            }
        }
        counts.truncate(partitions);
        sums.truncate(partitions);
    }
    best.expect("the partition order zero pass always produces a plan")
}

/// Writes a planned residual: the coding method, the partition order, and
/// each partition's parameter and Rice-coded values.
fn write_residual(
    writer: &mut BitWriter,
    residual: &[i64],
    block_size: usize,
    order: usize,
    plan: &ResidualPlan,
) {
    writer.write(u64::from(plan.five_bit), 2);
    writer.write(u64::from(plan.partition_order), 4);
    let parameter_bits = if plan.five_bit { 5 } else { 4 };
    let part_len = block_size >> plan.partition_order;
    let mut at = 0usize;
    for (index, &parameter) in plan.parameters.iter().enumerate() {
        writer.write(u64::from(parameter), parameter_bits);
        let count = if index == 0 {
            part_len - order
        } else {
            part_len
        };
        for &value in &residual[at..at + count] {
            let folded = zigzag(value);
            writer.write_unary(folded >> parameter);
            if parameter > 0 {
                writer.write(folded & ((1u64 << parameter) - 1), parameter);
            }
        }
        at += count;
    }
}

// -- The predictor search -----------------------------------------------------

/// The subframe form the search settled on, with everything needed to write
/// it.
enum SubframeKind {
    /// Every sample is this value.
    Constant(i64),
    /// The samples stored raw.
    Verbatim,
    /// A fixed predictor and its residual.
    Fixed {
        order: usize,
        residual: Vec<i64>,
        partitions: ResidualPlan,
    },
    /// A quantised linear predictor and its residual.
    Lpc {
        order: usize,
        shift: u32,
        /// Exactly `order` coefficients, in bitstream order.
        coefficients: Vec<i64>,
        residual: Vec<i64>,
        partitions: ResidualPlan,
    },
}

/// One subframe as the search decided to write it.
struct SubframePlan {
    kind: SubframeKind,
    /// Wasted bits declared and shifted out of every sample.
    wasted: u32,
    /// The subframe's bit depth before wasted removal, including the side
    /// channel's extra bit where this subframe carries the side.
    depth: u32,
    /// The samples shifted right by `wasted`, which is what warm-up and
    /// verbatim samples are written from.
    shifted: Vec<i64>,
    /// The exact size of the written subframe in bits, which is what the
    /// stereo search compares.
    cost: u64,
}

/// The subframe header's cost: the pad bit, the type, and the wasted-bits
/// field.
fn subframe_header_bits(wasted: u32) -> u64 {
    8 + if wasted > 0 { u64::from(wasted) } else { 0 }
}

/// Levinson-Durbin over the autocorrelation, producing the predictor at
/// every order up to `max_order` and the modelling error at each.
///
/// Returns how many orders were produced; numerical degeneracy (an error
/// that reaches zero, which perfectly predictable input causes) stops the
/// recursion early and the orders before it remain usable.
fn levinson(
    autocorrelation: &[f64],
    max_order: usize,
    coefficients: &mut [[f64; MAX_LPC_ORDER]; MAX_LPC_ORDER],
    errors: &mut [f64; MAX_LPC_ORDER],
) -> usize {
    let mut error = autocorrelation[0];
    let mut lpc = [0f64; MAX_LPC_ORDER];
    for m in 0..max_order {
        if error <= 0.0 || !error.is_finite() {
            return m;
        }
        let mut acc = autocorrelation[m + 1];
        for j in 0..m {
            acc -= lpc[j] * autocorrelation[m - j];
        }
        let reflection = acc / error;
        lpc[m] = reflection;
        for j in 0..m / 2 {
            let held = lpc[j];
            lpc[j] = held - reflection * lpc[m - 1 - j];
            lpc[m - 1 - j] -= reflection * held;
        }
        if m % 2 == 1 {
            let mid = m / 2;
            lpc[mid] -= reflection * lpc[mid];
        }
        error *= 1.0 - reflection * reflection;
        coefficients[m][..=m].copy_from_slice(&lpc[..=m]);
        errors[m] = error;
    }
    max_order
}

/// Quantises floating-point predictor coefficients to integers and a right
/// shift, with error feedback so rounding error accumulates into later
/// coefficients instead of into the prediction.
///
/// The shift puts the largest coefficient against the precision's ceiling,
/// clamped to the 0 through 15 range the format's shift field carries.
/// `None` when the coefficients are degenerate: all zero, or not finite,
/// neither of which can predict anything.
fn quantise_lpc(coefficients: &[f64], precision: u32) -> Option<(Vec<i64>, u32)> {
    let mut largest = 0f64;
    for &coefficient in coefficients {
        let magnitude = coefficient.abs();
        if !magnitude.is_finite() {
            return None;
        }
        if magnitude > largest {
            largest = magnitude;
        }
    }
    if largest <= 0.0 {
        return None;
    }
    // floor(log2(largest)) from the exponent field. A subnormal reads as
    // -1023, which only drives the shift into its upper clamp.
    let exponent = ((largest.to_bits() >> 52) & 0x7FF) as i64 - 1023;
    let shift = (i64::from(precision) - 2 - exponent).clamp(0, 15) as u32;
    let scale = (1u64 << shift) as f64;
    let ceiling = (1i64 << (precision - 1)) - 1;
    let floor = -(1i64 << (precision - 1));

    let mut quantised = Vec::with_capacity(coefficients.len());
    let mut error = 0f64;
    let mut any_nonzero = false;
    for &coefficient in coefficients {
        let target = coefficient * scale + error;
        let rounded = ((target + 0.5).floor() as i64).clamp(floor, ceiling);
        error = target - rounded as f64;
        quantised.push(rounded);
        if rounded != 0 {
            any_nonzero = true;
        }
    }
    if !any_nonzero {
        return None;
    }
    Some((quantised, shift))
}

/// The autocorrelation of the windowed samples at lags 0 through `max_lag`.
fn autocorrelation(windowed: &[f64], max_lag: usize, out: &mut [f64]) {
    for (lag, slot) in out.iter_mut().enumerate().take(max_lag + 1) {
        *slot = windowed[lag..]
            .iter()
            .zip(windowed.iter())
            .map(|(&late, &early)| late * early)
            .sum();
    }
}

/// Searches every subframe form the level allows for `samples` at `depth`
/// bits and returns the cheapest, exactly costed.
fn plan_subframe(samples: &[i64], depth: u32, level: &Level, windows: &[Vec<f64>]) -> SubframePlan {
    let n = samples.len();
    debug_assert!(n > 0);

    // Constant wins outright on constant input: no other form can beat
    // header plus one sample.
    if samples.iter().all(|&sample| sample == samples[0]) {
        return SubframePlan {
            kind: SubframeKind::Constant(samples[0]),
            wasted: 0,
            depth,
            shifted: Vec::new(),
            cost: 8 + u64::from(depth),
        };
    }

    // Wasted bits: low zero bits shared by every sample come off before any
    // prediction, shrinking warm-up samples, verbatim samples and residuals
    // alike. All-zero input cannot reach here, so some sample bounds the
    // count below `depth`.
    let mut wasted = depth - 1;
    for &sample in samples {
        if sample != 0 {
            wasted = wasted.min(sample.trailing_zeros());
            if wasted == 0 {
                break;
            }
        }
    }
    let coded_bits = depth - wasted;
    let shifted: Vec<i64> = if wasted == 0 {
        samples.to_vec()
    } else {
        samples.iter().map(|&sample| sample >> wasted).collect()
    };
    let header = subframe_header_bits(wasted);

    // Verbatim is the floor every other candidate must beat, and the
    // fallback that keeps a conforming frame writable whatever the input.
    let mut best_kind = SubframeKind::Verbatim;
    let mut best_cost = header + n as u64 * u64::from(coded_bits);

    // Fixed predictors: rank the five orders by the standard sum-of-absolute
    // residuals estimate over successive differences, then exactly cost the
    // best two. The differencing identity makes each order's residual one
    // subtraction pass over the previous order's.
    let max_fixed = 4.min(n - 1);
    let mut difference = shifted.clone();
    let mut order_sums = [u64::MAX; 5];
    order_sums[0] = shifted
        .iter()
        .fold(0u64, |acc, &v| acc.saturating_add(v.unsigned_abs()));
    for order in 1..=max_fixed {
        for index in (order..n).rev() {
            difference[index] = difference[index].wrapping_sub(difference[index - 1]);
        }
        order_sums[order] = difference[order..]
            .iter()
            .fold(0u64, |acc, &v| acc.saturating_add(v.unsigned_abs()));
    }
    let mut ranked: Vec<usize> = (0..=max_fixed).collect();
    ranked.sort_by_key(|&order| (order_sums[order], order));
    for &order in ranked.iter().take(2) {
        let residual: Vec<i64> = (order..n)
            .map(|index| shifted[index].wrapping_sub(fixed_prediction(&shifted[..index], order)))
            .collect();
        if !residual_in_range(&residual) {
            continue;
        }
        let folded: Vec<u64> = residual.iter().map(|&value| zigzag(value)).collect();
        let partitions = plan_residual(&folded, n, order, level.max_partition_order);
        let cost = header + order as u64 * u64::from(coded_bits) + partitions.bits;
        if cost < best_cost {
            best_cost = cost;
            best_kind = SubframeKind::Fixed {
                order,
                residual,
                partitions,
            };
        }
    }

    // Linear prediction: autocorrelate under each window, run
    // Levinson-Durbin to every order, estimate each order's size from its
    // modelling error, and exactly cost only the most promising few.
    let max_lpc = level.max_lpc_order.min(n - 1);
    if max_lpc >= 1 {
        for window in windows {
            let windowed: Vec<f64> = shifted
                .iter()
                .zip(window.iter())
                .map(|(&sample, &weight)| sample as f64 * weight)
                .collect();
            let mut autoc = [0f64; MAX_LPC_ORDER + 1];
            autocorrelation(&windowed, max_lpc, &mut autoc);
            if autoc[0] <= 0.0 {
                continue;
            }
            let mut coefficients = [[0f64; MAX_LPC_ORDER]; MAX_LPC_ORDER];
            let mut errors = [0f64; MAX_LPC_ORDER];
            let available = levinson(&autoc, max_lpc, &mut coefficients, &mut errors);

            let mut estimates: Vec<(f64, usize)> = (1..=available)
                .map(|order| {
                    let variance = (errors[order - 1] / n as f64).max(1e-12);
                    let per_residual = (0.5 * det_log2(variance) + 1.6).max(0.0);
                    let estimate = (n - order) as f64 * per_residual
                        + (order as u64 * u64::from(coded_bits + COEFFICIENT_PRECISION)) as f64;
                    (estimate, order)
                })
                .collect();
            estimates.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));

            for &(_, order) in estimates.iter().take(level.lpc_candidates) {
                let Some((quantised, shift)) =
                    quantise_lpc(&coefficients[order - 1][..order], COEFFICIENT_PRECISION)
                else {
                    continue;
                };
                let residual: Vec<i64> = (order..n)
                    .map(|index| {
                        shifted[index].wrapping_sub(lpc_prediction(
                            &quantised,
                            &shifted[..index],
                            shift,
                        ))
                    })
                    .collect();
                if !residual_in_range(&residual) {
                    continue;
                }
                let folded: Vec<u64> = residual.iter().map(|&value| zigzag(value)).collect();
                let partitions = plan_residual(&folded, n, order, level.max_partition_order);
                let cost = header
                    + order as u64 * u64::from(coded_bits)
                    + 4
                    + 5
                    + order as u64 * u64::from(COEFFICIENT_PRECISION)
                    + partitions.bits;
                if cost < best_cost {
                    best_cost = cost;
                    best_kind = SubframeKind::Lpc {
                        order,
                        shift,
                        coefficients: quantised,
                        residual,
                        partitions,
                    };
                }
            }
        }
    }

    SubframePlan {
        kind: best_kind,
        wasted,
        depth,
        shifted,
        cost: best_cost,
    }
}

// -- The stereo search --------------------------------------------------------

/// A cheap size estimate for one candidate channel: the mean absolute
/// second difference converted through the Rice relation that the optimal
/// parameter is about the log of the mean.
///
/// Only the ranking matters, and every mode's estimate shares the same
/// biases, so the shared constants cancel.
fn stereo_estimate(samples: &[i64]) -> f64 {
    let n = samples.len();
    if n < 3 {
        return n as f64 * 8.0;
    }
    let mut sum = 0u64;
    for window in samples.windows(3) {
        let second = window[2]
            .wrapping_sub(window[1].wrapping_mul(2))
            .wrapping_add(window[0]);
        sum = sum.saturating_add(second.unsigned_abs());
    }
    let mean = sum as f64 / (n - 2) as f64;
    n as f64 * (det_log2(mean + 1.0) + 2.0)
}

/// Chooses the frame's channel assignment and plans its two subframes.
///
/// The side channel is one bit deeper than the frame, which is why `side`
/// plans at `bits + 1`; a mid sample halves back into range and stays at
/// `bits`.
fn plan_stereo(
    left: &[i64],
    right: &[i64],
    bits: u32,
    level: &Level,
    windows: &[Vec<f64>],
) -> (u8, Vec<SubframePlan>) {
    let mid: Vec<i64> = left
        .iter()
        .zip(right.iter())
        .map(|(&l, &r)| (l + r) >> 1)
        .collect();
    let side: Vec<i64> = left
        .iter()
        .zip(right.iter())
        .map(|(&l, &r)| l - r)
        .collect();

    match level.stereo {
        Stereo::Independent => (
            1,
            vec![
                plan_subframe(left, bits, level, windows),
                plan_subframe(right, bits, level, windows),
            ],
        ),
        Stereo::Estimated => {
            let estimate_left = stereo_estimate(left);
            let estimate_right = stereo_estimate(right);
            let estimate_mid = stereo_estimate(&mid);
            let estimate_side = stereo_estimate(&side);
            let modes = [
                (estimate_left + estimate_right, 1u8),
                (estimate_left + estimate_side, 8),
                (estimate_side + estimate_right, 9),
                (estimate_mid + estimate_side, 10),
            ];
            let mut chosen = modes[0];
            for &mode in &modes[1..] {
                if mode.0.total_cmp(&chosen.0).is_lt() {
                    chosen = mode;
                }
            }
            let plans = match chosen.1 {
                1 => vec![
                    plan_subframe(left, bits, level, windows),
                    plan_subframe(right, bits, level, windows),
                ],
                8 => vec![
                    plan_subframe(left, bits, level, windows),
                    plan_subframe(&side, bits + 1, level, windows),
                ],
                9 => vec![
                    plan_subframe(&side, bits + 1, level, windows),
                    plan_subframe(right, bits, level, windows),
                ],
                _ => vec![
                    plan_subframe(&mid, bits, level, windows),
                    plan_subframe(&side, bits + 1, level, windows),
                ],
            };
            (chosen.1, plans)
        }
        Stereo::Exhaustive => {
            let plan_left = plan_subframe(left, bits, level, windows);
            let plan_right = plan_subframe(right, bits, level, windows);
            let plan_mid = plan_subframe(&mid, bits, level, windows);
            let plan_side = plan_subframe(&side, bits + 1, level, windows);
            let costs = [
                (plan_left.cost + plan_right.cost, 1u8),
                (plan_left.cost + plan_side.cost, 8),
                (plan_side.cost + plan_right.cost, 9),
                (plan_mid.cost + plan_side.cost, 10),
            ];
            let mut chosen = costs[0];
            for &mode in &costs[1..] {
                if mode.0 < chosen.0 {
                    chosen = mode;
                }
            }
            match chosen.1 {
                1 => (1, vec![plan_left, plan_right]),
                8 => (8, vec![plan_left, plan_side]),
                9 => (9, vec![plan_side, plan_right]),
                _ => (10, vec![plan_mid, plan_side]),
            }
        }
    }
}

// -- Frame assembly -----------------------------------------------------------

/// The frame header's block size field: a common table code, or the
/// explicit 8- or 16-bit form with its stored value.
fn block_size_code(count: usize) -> (u64, Option<(u64, u32)>) {
    match count {
        192 => (1, None),
        576 => (2, None),
        1152 => (3, None),
        2304 => (4, None),
        4608 => (5, None),
        256 => (8, None),
        512 => (9, None),
        1024 => (10, None),
        2048 => (11, None),
        4096 => (12, None),
        8192 => (13, None),
        16384 => (14, None),
        32768 => (15, None),
        _ if count <= 256 => (6, Some((count as u64 - 1, 8))),
        _ => (7, Some((count as u64 - 1, 16))),
    }
}

/// The frame header's sample rate field, from the common table.
///
/// A rate outside the table defers to streaminfo with code zero rather than
/// using the uncommon stored forms: every stream this writer produces has a
/// streaminfo block, deferring is always representable, and it is never
/// larger than the stored forms it replaces.
fn sample_rate_code(rate: u32) -> u64 {
    match rate {
        88_200 => 1,
        176_400 => 2,
        192_000 => 3,
        8_000 => 4,
        16_000 => 5,
        22_050 => 6,
        24_000 => 7,
        32_000 => 8,
        44_100 => 9,
        48_000 => 10,
        96_000 => 11,
        _ => 0,
    }
}

/// The frame header's bit depth field, from the common table, deferring to
/// streaminfo for the depths the table cannot express.
fn bit_depth_code(bits: u32) -> u64 {
    match bits {
        8 => 1,
        12 => 2,
        16 => 4,
        20 => 5,
        24 => 6,
        32 => 7,
        _ => 0,
    }
}

/// Writes one subframe as its plan decided.
fn write_subframe(writer: &mut BitWriter, plan: &SubframePlan, block_size: usize) {
    writer.write(0, 1);
    let type_code = match &plan.kind {
        SubframeKind::Constant(_) => 0u64,
        SubframeKind::Verbatim => 1,
        SubframeKind::Fixed { order, .. } => 8 + *order as u64,
        SubframeKind::Lpc { order, .. } => 31 + *order as u64,
    };
    writer.write(type_code, 6);
    if plan.wasted == 0 {
        writer.write(0, 1);
    } else {
        writer.write(1, 1);
        // The count minus one in unary: wasted - 1 zeros, then the one.
        writer.write(1, plan.wasted);
    }
    let coded_bits = plan.depth - plan.wasted;
    match &plan.kind {
        SubframeKind::Constant(value) => writer.write_signed(*value, coded_bits),
        SubframeKind::Verbatim => {
            for &value in &plan.shifted {
                writer.write_signed(value, coded_bits);
            }
        }
        SubframeKind::Fixed {
            order,
            residual,
            partitions,
        } => {
            for &value in &plan.shifted[..*order] {
                writer.write_signed(value, coded_bits);
            }
            write_residual(writer, residual, block_size, *order, partitions);
        }
        SubframeKind::Lpc {
            order,
            shift,
            coefficients,
            residual,
            partitions,
        } => {
            for &value in &plan.shifted[..*order] {
                writer.write_signed(value, coded_bits);
            }
            writer.write(u64::from(COEFFICIENT_PRECISION - 1), 4);
            writer.write(u64::from(*shift), 5);
            for &coefficient in coefficients {
                writer.write_signed(coefficient, COEFFICIENT_PRECISION);
            }
            write_residual(writer, residual, block_size, *order, partitions);
        }
    }
}

/// Writes one complete frame and returns how many bytes it occupied.
fn write_frame(
    out: &mut Vec<u8>,
    frame_index: u64,
    block_size: usize,
    sample_rate: u32,
    bits: u32,
    assignment: u8,
    plans: &[SubframePlan],
) -> usize {
    let mut writer = BitWriter::new();
    // The sync code and the fixed blocking strategy bit: every stream this
    // writer produces is fixed-blocksize, so the coded number is a frame
    // number.
    writer.write(0b111_1111_1111_1100, 15);
    writer.write(0, 1);
    let (size_code, size_tail) = block_size_code(block_size);
    writer.write(size_code, 4);
    writer.write(sample_rate_code(sample_rate), 4);
    writer.write(u64::from(assignment), 4);
    writer.write(bit_depth_code(bits), 3);
    writer.write(0, 1);
    write_coded_number(&mut writer, frame_index);
    if let Some((value, width)) = size_tail {
        writer.write(value, width);
    }
    let header_crc = crc8(writer.bytes());
    writer.write(u64::from(header_crc), 8);

    for plan in plans {
        write_subframe(&mut writer, plan, block_size);
    }
    writer.align();
    let frame_crc = crc16(writer.bytes());
    writer.write(u64::from(frame_crc), 16);

    let bytes = writer.into_bytes();
    out.extend_from_slice(&bytes);
    bytes.len()
}

// -- The writer ---------------------------------------------------------------

/// Writes native FLAC streams: a `fLaC` signature, one streaminfo block,
/// and Rice-coded audio frames.
///
/// The third writer in the crate, beside [`WavWriter`](crate::WavWriter) and
/// [`AiffWriter`](crate::AiffWriter) and shaped like both: construct with
/// the audio's description, then [`write`](Self::write) or
/// [`to_bytes`](Self::to_bytes). The difference FLAC brings is that writing
/// is a search rather than a transcription, so a [`level`](Self::with_level)
/// chooses how hard to look for a small file, 0 through 8 with 5 the
/// default, in broadly the reference encoder's sense of those numbers.
///
/// # What is written
///
/// Fixed-blocksize frames of every subframe type the decoder reads:
/// constant on runs the search finds constant, verbatim where prediction
/// cannot pay for itself, fixed predictors, and quantised linear predictors
/// found by windowed autocorrelation and Levinson-Durbin recursion. Stereo
/// frames choose per frame between independent, left/side, right/side and
/// mid/side coding. Wasted bits are detected and declared. The streaminfo
/// block states the true minimum and maximum block and frame sizes, the
/// total sample count, and the MD5 of the audio, which this crate's own
/// decoder and any other conforming decoder then verify. No other metadata
/// block is written: no padding, no seek table, no tags.
///
/// # The samples, and what lossless means here
///
/// Input is interleaved `f32`, quantised to `bits_per_sample`-bit integers
/// by the same rule every other writer in the crate uses, stated on
/// [`sample`](crate::sample): clamp to `-1.0..=1.0`, scale by
/// `2^(bits - 1)`, truncate toward zero, clamp in integer space. Encoding
/// is lossless over those integers, exactly; the quantisation is where
/// `f32` input meets an integer format, and it is the identical boundary
/// the decoder's scaling comes back across. At or below 24 bits the round
/// trip through [`FlacReader`] returns the input samples bit for bit.
///
/// # The arithmetic is the decoder's
///
/// Residuals are computed by subtracting the very prediction functions the
/// decoder adds, `fixed_prediction` and `lpc_prediction` in this module, at
/// the same widths in the same order. An encoder with its own arithmetic
/// could disagree with its decoder only on loud material at high depths,
/// which no small test catches; sharing the functions makes the
/// disagreement inexpressible.
///
/// # Determinism
///
/// The same samples at the same level produce the same bytes on every
/// platform. The search uses `f64` only through operations IEEE 754 defines
/// exactly, and the window cosine and estimation logarithm are computed
/// in-crate from arithmetic rather than taken from a libm that varies by
/// the last bit between targets.
///
/// # Example
///
/// ```
/// use decibri_decode::{AudioSpec, FlacReader, FlacWriter};
///
/// // A quiet ramp: every value scales to a whole 16-bit integer, so the
/// // round trip is exact.
/// let samples: Vec<f32> = (0..2000).map(|i| ((i % 128) as f32 - 64.0) / 128.0).collect();
/// let file = FlacWriter::new(AudioSpec::mono(44_100), 16).to_bytes(&samples)?;
///
/// let reader = FlacReader::new(&file)?;
/// assert_eq!(reader.frames(), Some(2000));
/// // decode_to_end verifies the streaminfo MD5 over what it decodes.
/// let decoded = reader.decode_to_end()?;
/// assert_eq!(decoded.samples(), &samples[..]);
/// # Ok::<(), decibri_decode::DecodeError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FlacWriter {
    spec: AudioSpec,
    bits_per_sample: u8,
    level: u8,
}

impl FlacWriter {
    /// A writer for audio at `spec`, quantised to `bits_per_sample` bits,
    /// at the default compression level 5.
    ///
    /// Any depth the format carries, 4 through 32, is accepted; depths the
    /// frame header cannot express directly are deferred to streaminfo,
    /// exactly as the decoder resolves them. Validation happens in
    /// [`write`](Self::write), where there is a `Result` to report it.
    pub const fn new(spec: AudioSpec, bits_per_sample: u8) -> Self {
        Self {
            spec,
            bits_per_sample,
            level: 5,
        }
    }

    /// Sets the compression level, 0 through 8.
    ///
    /// Higher levels search more: larger predictor orders, deeper residual
    /// partitioning, more analysis windows and a fuller stereo search. The
    /// output of every level decodes identically; only the byte count and
    /// the encoding time differ. A level above 8 is rejected by
    /// [`write`](Self::write).
    pub const fn with_level(mut self, level: u8) -> Self {
        self.level = level;
        self
    }

    /// The rate and layout written into streaminfo.
    pub const fn spec(&self) -> AudioSpec {
        self.spec
    }

    /// The bit depth samples are quantised to.
    pub const fn bits_per_sample(&self) -> u8 {
        self.bits_per_sample
    }

    /// The compression level [`write`](Self::write) will search at.
    pub const fn level(&self) -> u8 {
        self.level
    }

    /// Appends a complete FLAC stream to `output`, and returns how many
    /// bytes it appended.
    ///
    /// `output` is appended to, never cleared, the same convention as every
    /// other writer in this crate.
    ///
    /// # Errors
    ///
    /// - [`DecodeError::UnsupportedChannelLayout`] for no channels or more
    ///   than the eight a FLAC frame can carry.
    /// - [`DecodeError::UnsupportedSampleFormat`] for a bit depth outside
    ///   the 4 through 32 the format defines.
    /// - [`DecodeError::Malformed`] for a sample rate of zero or beyond
    ///   streaminfo's 20-bit field, and for a compression level above 8.
    /// - [`DecodeError::Truncated`] when `samples` does not divide into
    ///   whole frames, for the reason recorded on
    ///   [`WavWriter::write`](crate::WavWriter::write): a partial frame
    ///   written would be rejected by every reader, and one dropped would be
    ///   silent truncation.
    pub fn write(&self, samples: &[f32], output: &mut Vec<u8>) -> Result<usize, DecodeError> {
        let channels = usize::from(self.spec.channels);
        if channels == 0 || channels > usize::from(MAX_CHANNELS) {
            return Err(DecodeError::UnsupportedChannelLayout {
                channels: self.spec.channels,
            });
        }
        // Streaminfo's rate field is 20 bits, and zero is not a rate. The
        // offset is where the field sits in the file this writer refused to
        // produce.
        if self.spec.sample_rate == 0 || self.spec.sample_rate > 0xF_FFFF {
            return Err(DecodeError::Malformed {
                expected: "a sample rate between 1 and 1048575, the range streaminfo carries",
                offset: 18,
            });
        }
        if !(4..=32).contains(&self.bits_per_sample) {
            return Err(DecodeError::UnsupportedSampleFormat {
                format: CodecId::Name("FLAC".to_string()),
                bits_per_sample: u16::from(self.bits_per_sample),
            });
        }
        if usize::from(self.level) >= LEVELS.len() {
            return Err(DecodeError::Malformed {
                expected: "a compression level between 0 and 8",
                offset: 0,
            });
        }
        let leftover = samples.len() % channels;
        if leftover != 0 {
            return Err(DecodeError::Truncated {
                expected: channels as u64,
                available: leftover as u64,
            });
        }
        let total_frames = (samples.len() / channels) as u64;
        if total_frames >= 1u64 << 36 {
            return Err(DecodeError::Malformed {
                expected: "no more sample frames than streaminfo's 36-bit total can declare",
                offset: 21,
            });
        }

        let level = &LEVELS[usize::from(self.level)];
        let bits = u32::from(self.bits_per_sample);
        let block = level.block_size as usize;
        let scale = (1u64 << (bits - 1)) as f32;
        let floor = -(1i64 << (bits - 1));
        let ceiling = (1i64 << (bits - 1)) - 1;
        let width = usize::from(self.bits_per_sample).div_ceil(8);

        let mut hasher = Md5::new();
        let mut body: Vec<u8> = Vec::new();
        let mut smallest_frame = u64::MAX;
        let mut largest_frame = 0u64;
        let mut channel_data: Vec<Vec<i64>> = vec![Vec::new(); channels];
        let mut hash_stage: Vec<u8> = Vec::new();
        let mut windows: Vec<Vec<f64>> = Vec::new();
        let mut window_len = 0usize;

        let mut frame_index = 0u64;
        let mut done = 0usize;
        let total = total_frames as usize;
        while done < total {
            let count = block.min(total - done);
            for channel in &mut channel_data {
                channel.clear();
            }
            hash_stage.clear();
            let base = done * channels;
            for frame in 0..count {
                for (channel, data) in channel_data.iter_mut().enumerate() {
                    let value = quantize(
                        samples[base + frame * channels + channel],
                        scale,
                        floor,
                        ceiling,
                    );
                    data.push(value);
                    hash_stage.extend_from_slice(&value.to_le_bytes()[..width]);
                }
            }
            hasher.update(&hash_stage);

            if level.max_lpc_order > 0 && (windows.is_empty() || window_len != count) {
                windows = level
                    .windows
                    .iter()
                    .map(|shape| shape.build(count))
                    .collect();
                window_len = count;
            }

            let (assignment, plans) = if channels == 2 && level.stereo != Stereo::Independent {
                plan_stereo(&channel_data[0], &channel_data[1], bits, level, &windows)
            } else {
                (
                    (channels - 1) as u8,
                    channel_data
                        .iter()
                        .map(|channel| plan_subframe(channel, bits, level, &windows))
                        .collect(),
                )
            };

            let frame_bytes = write_frame(
                &mut body,
                frame_index,
                count,
                self.spec.sample_rate,
                bits,
                assignment,
                &plans,
            ) as u64;
            smallest_frame = smallest_frame.min(frame_bytes);
            largest_frame = largest_frame.max(frame_bytes);
            frame_index += 1;
            done += count;
        }

        let digest = hasher.finish();
        let start = output.len();
        output.extend_from_slice(MAGIC.as_bytes());
        // Streaminfo, the only metadata block, so its header carries the
        // last-block flag.
        output.push(0x80);
        output.extend_from_slice(&[0, 0, STREAMINFO_BYTES as u8]);
        let mut info = BitWriter::new();
        // Every frame but the last holds exactly the nominal block size, so
        // the nominal size is both the minimum and the maximum in RFC 9639
        // section 8.2's sense, which excludes the last block.
        info.write(u64::from(level.block_size), 16);
        info.write(u64::from(level.block_size), 16);
        let (min_frame, max_frame) = if frame_index == 0 {
            // No frames were written, so there are no frame sizes; zero is
            // the field's own spelling of unknown.
            (0, 0)
        } else {
            (smallest_frame, largest_frame)
        };
        info.write(min_frame, 24);
        info.write(max_frame, 24);
        info.write(u64::from(self.spec.sample_rate), 20);
        info.write(channels as u64 - 1, 3);
        info.write(u64::from(bits) - 1, 5);
        // An empty stream writes zero here, which the format reads as
        // unknown: the field has no way to distinguish the two, and this
        // writer does not pretend otherwise.
        info.write(total_frames, 36);
        output.extend_from_slice(&info.into_bytes());
        output.extend_from_slice(&digest);
        output.extend_from_slice(&body);
        Ok(output.len() - start)
    }

    /// A complete FLAC stream as a new `Vec<u8>`.
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
    use crate::sample::{i16_to_f32, i24_to_f32, i32_to_f32, i8_to_f32};

    /// RFC 9639 appendix D.1's worked example: 16-bit stereo, one frame, one
    /// sample per channel, both subframes verbatim with wasted bits.
    const EXAMPLE_1: [u8; 45] = [
        0x66, 0x4c, 0x61, 0x43, 0x80, 0x00, 0x00, 0x22, 0x10, 0x00, 0x10, 0x00, 0x00, 0x00, 0x0f,
        0x00, 0x00, 0x0f, 0x0a, 0xc4, 0x42, 0xf0, 0x00, 0x00, 0x00, 0x01, 0x3e, 0x84, 0xb4, 0x18,
        0x07, 0xdc, 0x69, 0x03, 0x07, 0x58, 0x6a, 0x3d, 0xad, 0x1a, 0x2e, 0x0f, 0xff, 0xf8, 0x69,
    ];

    /// The tail of example 1, which the array above cannot hold in one
    /// literal without becoming unreadable.
    const EXAMPLE_1_TAIL: [u8; 8] = [0x18, 0x00, 0x00, 0xbf, 0x03, 0x58, 0xfd, 0x03];

    /// The frame footer of example 1.
    const EXAMPLE_1_CRC: [u8; 4] = [0x12, 0x8b, 0xaa, 0x9a];

    /// Example 1 assembled.
    fn example_1() -> Vec<u8> {
        let mut file = EXAMPLE_1.to_vec();
        file.extend_from_slice(&EXAMPLE_1_TAIL);
        file.extend_from_slice(&EXAMPLE_1_CRC);
        file
    }

    /// The CRC-8 the format specifies, against values that can be worked out
    /// by hand and against the one example 1 carries.
    ///
    /// `0x01` through the polynomial `x^8 + x^2 + x^1 + x^0` is seven left
    /// shifts to `0x80` and one more that reduces to the polynomial's low
    /// byte, `0x07`.
    #[test]
    fn crc8_matches_the_specification() {
        assert_eq!(crc8(&[0x00]), 0x00);
        assert_eq!(crc8(&[0x01]), 0x07);
        // Example 1's six-byte frame header at 0x2a, whose stored CRC-8 sits
        // at 0x30. The header ends where the CRC begins, sync code included.
        let file = example_1();
        assert_eq!(crc8(&file[0x2a..0x30]), file[0x30]);
    }

    /// The CRC-16 the format specifies, likewise.
    ///
    /// `0x01` through `x^16 + x^15 + x^2 + x^0` reduces to `0x8005`, and
    /// example 1's whole frame, sync code included and the CRC itself
    /// excluded, hashes to the two bytes the file ends with.
    #[test]
    fn crc16_matches_the_specification() {
        assert_eq!(crc16(&[0x00]), 0x0000);
        assert_eq!(crc16(&[0x01]), 0x8005);
        let file = example_1();
        assert_eq!(
            crc16(&file[0x2a..0x37]),
            u16::from_be_bytes([file[0x37], file[0x38]])
        );
    }

    #[test]
    fn the_coded_number_is_not_utf8() {
        // RFC 9639 section 9.1.5's own worked example: 51 billion, in seven
        // octets, which no UTF-8 decoder will read.
        let bytes = [0xFE, 0xAF, 0x9F, 0xB5, 0xA3, 0xB8, 0x80];
        let mut reader = BitReader::new(&bytes, 0);
        assert_eq!(
            read_coded_number(&mut reader).expect("read"),
            51_000_000_000
        );

        // A single octet is itself.
        let mut reader = BitReader::new(&[0x7F], 0);
        assert_eq!(read_coded_number(&mut reader).expect("read"), 127);

        // A continuation octet cannot start a code, and neither can 0xFF.
        for first in [0x80u8, 0xFF] {
            let bytes = [first, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80];
            let mut reader = BitReader::new(&bytes, 0);
            assert!(read_coded_number(&mut reader).is_err());
        }
    }

    #[test]
    fn the_bit_reader_reads_across_byte_boundaries() {
        let bytes = [0b1010_1100, 0b1111_0000, 0b0000_1111];
        let mut reader = BitReader::new(&bytes, 0);
        assert_eq!(reader.read_bits(3).expect("read"), 0b101);
        assert_eq!(reader.read_bits(7).expect("read"), 0b011_0011);
        assert_eq!(reader.read_bits(14).expect("read"), 0b11_0000_0000_1111);
        assert!(reader.read_bits(1).is_err());
    }

    #[test]
    fn the_bit_reader_sign_extends() {
        // 0b111 in three bits is -1, 0b011 is 3, and 0b100 in the last three
        // is -4: the same bits mean different values at different widths.
        let bytes = [0b1110_1110, 0b0000_0000];
        let mut reader = BitReader::new(&bytes, 0);
        assert_eq!(reader.read_signed(3).expect("read"), -1);
        assert_eq!(reader.read_signed(3).expect("read"), 3);
        assert_eq!(reader.read_signed(3).expect("read"), -4);
        // Unsigned, the same three bits are 4 rather than -4.
        let mut reader = BitReader::new(&bytes, 0);
        reader.bit = 6;
        assert_eq!(reader.read_bits(3).expect("read"), 4);
    }

    #[test]
    fn unary_counts_zeros_and_eats_the_one() {
        let bytes = [0b0000_0100, 0b1000_0000];
        let mut reader = BitReader::new(&bytes, 0);
        assert_eq!(reader.read_unary().expect("read"), 5);
        assert_eq!(reader.read_unary().expect("read"), 2);
        // A run that never terminates inside the input is truncation, not a
        // hang.
        let mut reader = BitReader::new(&[0u8; 32], 0);
        assert!(reader.read_unary().is_err());
    }

    #[test]
    fn a_long_unary_run_terminates() {
        // Sixteen zero bytes then a set bit: past the 57-bit window, so this
        // exercises the loop rather than the single-window fast path.
        let mut bytes = vec![0u8; 16];
        bytes.push(0x80);
        let mut reader = BitReader::new(&bytes, 0);
        assert_eq!(reader.read_unary().expect("read"), 128);
    }

    /// The FLAC scaling rule and `sample.rs`'s must agree at every width the
    /// two share, because the same audio decoded through this container and
    /// through WAV has to give the same `f32`.
    ///
    /// This is the tie required by the crate's third coverage lesson: one
    /// rule with two implementations means a control on either one says
    /// nothing about the other. Breaking either divisor turns this red.
    #[test]
    fn flac_scaling_agrees_with_sample_rs() {
        fn flac_scale(sample: i64, bits: u8) -> f32 {
            sample as f32 / (1u64 << (bits - 1)) as f32
        }

        for value in i16::MIN..=i16::MAX {
            assert_eq!(flac_scale(i64::from(value), 16), i16_to_f32(value));
        }
        for value in i8::MIN..=i8::MAX {
            assert_eq!(flac_scale(i64::from(value), 8), i8_to_f32(value));
        }
        for value in [
            crate::sample::I24_MIN,
            -1,
            0,
            1,
            crate::sample::I24_MAX,
            123_456,
        ] {
            assert_eq!(flac_scale(i64::from(value), 24), i24_to_f32(value));
        }
        for value in [i32::MIN, -1, 0, 1, i32::MAX, 1_234_567_890] {
            assert_eq!(flac_scale(i64::from(value), 32), i32_to_f32(value));
        }
    }

    #[test]
    fn mid_side_reconstruction_is_lossless_over_a_swept_domain() {
        // The odd-sample correction is what makes this true. Sweeping both
        // channels over a range that straddles zero catches a sign error in
        // the arithmetic shift as well as a dropped correction.
        for left in -300i64..=300 {
            for right in -300i64..=300 {
                let mid = (left + right) >> 1;
                let side = left - right;
                let mut a = [mid];
                let mut b = [side];
                undo_decorrelation(ChannelAssignment::MidSide, &mut a, &mut b);
                assert_eq!((a[0], b[0]), (left, right), "mid/side lost {left},{right}");
            }
        }
    }

    #[test]
    fn left_side_and_side_right_reconstruction_round_trip() {
        for left in [-32_768i64, -1, 0, 1, 32_767] {
            for right in [-32_768i64, -1, 0, 1, 32_767] {
                let side = left - right;

                let mut a = [left];
                let mut b = [side];
                undo_decorrelation(ChannelAssignment::LeftSide, &mut a, &mut b);
                assert_eq!((a[0], b[0]), (left, right));

                let mut a = [side];
                let mut b = [right];
                undo_decorrelation(ChannelAssignment::SideRight, &mut a, &mut b);
                assert_eq!((a[0], b[0]), (left, right));
            }
        }
    }

    #[test]
    fn the_worked_example_decodes_exactly() {
        let file = example_1();
        let reader = FlacReader::new(&file).expect("open");
        assert_eq!(reader.spec(), AudioSpec::new(44_100, 2));
        assert_eq!(reader.stream_info().bits_per_sample, 16);
        assert_eq!(reader.frames(), Some(1));

        let decoded = reader.decode_to_end().expect("decode");
        // RFC 9639 appendix D.1.4's own numbers, both of which come out of
        // wasted-bits subframes: 25588 and 10416.
        assert_eq!(
            decoded.samples(),
            [25_588.0 / 32_768.0, 10_416.0 / 32_768.0]
        );
    }

    #[test]
    fn a_wrong_magic_is_named_rather_than_categorised() {
        let error = FlacReader::new(b"RIFFxxxxWAVE").expect_err("must reject");
        assert!(
            matches!(error, DecodeError::UnsupportedContainer { tag } if tag.as_bytes() == b"RIFF"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn the_streaming_path_matches_the_whole_file_path() {
        let file = example_1();
        let whole = FlacReader::new(&file)
            .expect("open")
            .decode_to_end()
            .expect("decode");

        for chunk in [1, 2, 3, 7, 16, 64, 4096] {
            let mut stream = FlacStreamDecoder::new();
            let mut samples = Vec::new();
            for piece in file.chunks(chunk) {
                let mut offset = 0;
                while offset < piece.len() {
                    offset += stream.push(&piece[offset..]).expect("push");
                    while stream.pull(&mut samples, usize::MAX).expect("pull") > 0 {}
                }
            }
            stream.finish(&mut samples).expect("finish");
            assert_eq!(
                samples,
                whole.samples(),
                "output changed with a {chunk}-byte feed size"
            );
        }
    }

    #[test]
    fn an_all_zero_md5_is_unset_rather_than_a_failure() {
        let mut file = example_1();
        // The checksum field sits at 0x1a, eighteen bytes into the 34-byte
        // streaminfo body which starts at 8.
        file[26..42].fill(0);
        let reader = FlacReader::new(&file).expect("open");
        assert_eq!(reader.stream_info().md5, None);
        assert!(reader.decode_to_end().is_ok());
    }

    #[test]
    fn a_wrong_md5_is_a_typed_error_rather_than_audio() {
        let mut file = example_1();
        file[26] ^= 0xFF;
        let error = FlacReader::new(&file)
            .expect("open")
            .decode_to_end()
            .expect_err("must reject");
        assert!(
            matches!(
                error,
                DecodeError::Malformed {
                    expected: "decoded audio matching the streaminfo MD5 checksum",
                    ..
                }
            ),
            "unexpected error: {error}"
        );
    }

    // -- The writer's search machinery ----------------------------------------

    /// A deterministic value stream for the searches' unit tests.
    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// The exact bit cost of one partition at one parameter, counted the
    /// slow way: value by value, quotient plus terminator plus parameter
    /// bits.
    fn brute_partition_cost(folded: &[u64], parameter: u32) -> u64 {
        folded
            .iter()
            .map(|&value| (value >> parameter) + 1 + u64::from(parameter))
            .sum()
    }

    /// The partition search against exhaustive enumeration.
    ///
    /// The plan is built by accumulating shifted sums at the deepest order
    /// and merging pairwise; the oracle here recounts every (order, method,
    /// parameter) cell directly from the residuals, so a break in the
    /// accumulation, the merge or the minimum-taking cannot agree with it by
    /// construction. This is the in-tree gate for the Rice search, the one a
    /// negative control on the search must turn red.
    #[test]
    fn the_partition_search_matches_exhaustive_enumeration() {
        let mut state = 0x5EED_CAFE_u64;
        for (block_size, order, max_po) in
            [(64usize, 2usize, 3u32), (96, 0, 4), (256, 4, 8), (33, 1, 6)]
        {
            // A mix of small and occasionally large residuals, so both Rice
            // methods and several parameters are in play.
            let folded: Vec<u64> = (0..block_size - order)
                .map(|_| {
                    let raw = splitmix(&mut state);
                    if raw.is_multiple_of(13) {
                        raw >> 34
                    } else {
                        raw >> 56
                    }
                })
                .collect();
            let plan = plan_residual(&folded, block_size, order, max_po);

            let mut expected = u64::MAX;
            let mut po = max_po.min(block_size.trailing_zeros());
            while block_size >> po <= order {
                po -= 1;
            }
            for po in (0..=po).rev() {
                if !block_size.is_multiple_of(1 << po) {
                    continue;
                }
                let part_len = block_size >> po;
                for (parameter_bits, max_parameter) in [(4u64, RICE4_MAX), (5, RICE5_MAX)] {
                    let mut total = 2 + 4;
                    let mut at = 0usize;
                    for partition in 0..1usize << po {
                        let count = if partition == 0 {
                            part_len - order
                        } else {
                            part_len
                        };
                        let best = (0..=max_parameter)
                            .map(|p| brute_partition_cost(&folded[at..at + count], p))
                            .min()
                            .expect("at least one parameter");
                        total += parameter_bits + best;
                        at += count;
                    }
                    expected = expected.min(total);
                }
            }
            assert_eq!(
                plan.bits, expected,
                "block {block_size} order {order} max_po {max_po}"
            );

            // The written form must occupy exactly the bits the plan
            // claims, or the whole subframe search compared the wrong
            // things. Reconstruct the residuals from the folded values to
            // write them.
            let residual: Vec<i64> = folded
                .iter()
                .map(|&value| ((value >> 1) as i64) ^ -((value & 1) as i64))
                .collect();
            let mut writer = BitWriter::new();
            write_residual(&mut writer, &residual, block_size, order, &plan);
            let written_bits = writer.out.len() as u64 * 8 + u64::from(writer.filled);
            assert_eq!(written_bits, plan.bits, "written size vs planned size");
        }
    }

    /// The bit writer against the bit reader, which is the pairing every
    /// encoded stream lives or dies by.
    #[test]
    fn the_bit_writer_mirrors_the_bit_reader() {
        let mut writer = BitWriter::new();
        writer.write(0b101, 3);
        writer.write_signed(-1, 7);
        writer.write(0x5A5A, 16);
        writer.write_signed(-12345, 33);
        writer.write_unary(0);
        writer.write_unary(5);
        writer.write_unary(100);
        writer.write(1, 1);
        let bytes = writer.into_bytes();

        let mut reader = BitReader::new(&bytes, 0);
        assert_eq!(reader.read_bits(3).expect("read"), 0b101);
        assert_eq!(reader.read_signed(7).expect("read"), -1);
        assert_eq!(reader.read_bits(16).expect("read"), 0x5A5A);
        assert_eq!(reader.read_signed(33).expect("read"), -12345);
        assert_eq!(reader.read_unary().expect("read"), 0);
        assert_eq!(reader.read_unary().expect("read"), 5);
        assert_eq!(reader.read_unary().expect("read"), 100);
        assert_eq!(reader.read_bits(1).expect("read"), 1);
    }

    /// The coded number writer against the coded number reader, at the
    /// length boundaries of the encoding.
    #[test]
    fn the_coded_number_round_trips_at_every_length_boundary() {
        for value in [
            0u64,
            0x7F,
            0x80,
            0x7FF,
            0x800,
            0xFFFF,
            0x1_0000,
            0x1F_FFFF,
            0x20_0000,
            0x3FF_FFFF,
            0x400_0000,
            0x7FFF_FFFF,
            0x8000_0000,
            (1 << 36) - 1,
        ] {
            let mut writer = BitWriter::new();
            write_coded_number(&mut writer, value);
            let bytes = writer.into_bytes();
            let mut reader = BitReader::new(&bytes, 0);
            assert_eq!(
                read_coded_number(&mut reader).expect("read"),
                value,
                "coded number {value:#x}"
            );
        }
    }

    /// The deterministic cosine and logarithm against values checkable by
    /// hand. Accuracy here is loose on purpose: these feed a window shape
    /// and a ranking, and what the crate needs from them is identical bits
    /// everywhere, not the last ulp.
    #[test]
    fn the_deterministic_math_lands_near_the_true_values() {
        assert!((det_cos_pi(0.0) - 1.0).abs() < 1e-12);
        assert!(det_cos_pi(0.5).abs() < 1e-12);
        assert!((det_cos_pi(1.0) + 1.0).abs() < 1e-12);
        assert!((det_cos_pi(1.0 / 3.0) - 0.5).abs() < 1e-9);
        assert!((det_log2(8.0) - 3.0).abs() < 1e-9);
        assert!((det_log2(1.0)).abs() < 1e-9);
        assert!((det_log2(0.125) + 3.0).abs() < 1e-9);
        assert!((det_log2(3.0) - 1.584_962_500_721_156).abs() < 1e-9);
        // Subnormal inputs stay finite and ordered.
        assert!(det_log2(f64::from_bits(1)) < -1000.0);
    }

    /// Quantised coefficients stay inside the fields that carry them.
    #[test]
    fn quantised_coefficients_fit_their_fields() {
        let mut state = 0xC0FF_EE00_u64;
        for _ in 0..200 {
            let order = (splitmix(&mut state) % 32 + 1) as usize;
            let coefficients: Vec<f64> = (0..order)
                .map(|_| {
                    let raw = splitmix(&mut state);
                    // Magnitudes from around 1e-4 to around 16.
                    let magnitude = ((raw >> 40) as f64 + 1.0) / 1_048_576.0;
                    let scaled = magnitude * ((1 << (raw % 18)) as f64 / 8.0);
                    if raw & 1 == 0 {
                        scaled
                    } else {
                        -scaled
                    }
                })
                .collect();
            let Some((quantised, shift)) = quantise_lpc(&coefficients, COEFFICIENT_PRECISION)
            else {
                continue;
            };
            assert!(shift <= 15, "shift {shift} does not fit the 5-bit field");
            assert_eq!(quantised.len(), order);
            let ceiling = (1i64 << (COEFFICIENT_PRECISION - 1)) - 1;
            let floor = -(1i64 << (COEFFICIENT_PRECISION - 1));
            for &q in &quantised {
                assert!(
                    (floor..=ceiling).contains(&q),
                    "coefficient {q} does not fit {COEFFICIENT_PRECISION} bits"
                );
            }
        }
    }

    /// The shared prediction functions on values small enough to check by
    /// hand, so a rewrite of either cannot silently change what "order 2"
    /// means.
    #[test]
    fn the_shared_predictions_match_hand_arithmetic() {
        let history = [3i64, 5, 9, 15, 23];
        assert_eq!(fixed_prediction(&history, 0), 0);
        assert_eq!(fixed_prediction(&history, 1), 23);
        assert_eq!(fixed_prediction(&history, 2), 2 * 23 - 15);
        assert_eq!(fixed_prediction(&history, 3), 3 * 23 - 3 * 15 + 9);
        assert_eq!(fixed_prediction(&history, 4), 4 * 23 - 6 * 15 + 4 * 9 - 5);
        // Coefficients in bitstream order: the first multiplies the newest
        // sample. (2*23 + 1*15) >> 2 = 15 with truncation toward negative
        // infinity, which for a positive sum is plain division.
        assert_eq!(lpc_prediction(&[2, 1], &history, 2), (2 * 23 + 15) >> 2);
    }
}
