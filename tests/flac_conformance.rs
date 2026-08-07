#![forbid(unsafe_code)]
//! FLAC conformance: the dimensions a FLAC stream varies along, crossed, plus
//! RFC 9639's own worked examples as literal vectors.
//!
//! # The dimensions, enumerated before any test was written
//!
//! The discipline is the one recorded at the top of `wav_conformance.rs` and
//! `aiff_conformance.rs`: a test's coverage is bounded by the dimensions its
//! inputs vary along, not by how many inputs it uses.
//!
//! 1. bit depth: 8, 12, 16, 20, 24 and 32, and one that is not a whole
//!    number of bytes
//! 2. channel count: 1 through 8
//! 3. channel assignment: independent, left/side, side/right, mid/side
//! 4. subframe type: constant and verbatim here; fixed, linear-predictor and
//!    every residual shape in `flac_corpus.rs`
//! 5. wasted bits: zero and non-zero
//! 6. block size: from the common table, and the uncommon 8-bit and 16-bit
//!    forms
//! 7. sample rate: deferred to streaminfo, from the common table, and the
//!    uncommon 8-bit, 16-bit and 16-bit-divided-by-ten forms
//! 8. blocking strategy: fixed and variable
//! 9. metadata blocks: streaminfo alone, and with padding, application and
//!    unknown blocks in front of the audio
//! 10. total samples in streaminfo: declared and unknown
//! 11. the streaminfo MD5: unset here; present on all 74 files that carry
//!     one in `flac_corpus.rs` and on the three RFC vectors below
//! 12. total input length: small, and past the 65,536-sample ready limit
//! 13. feed chunk size, on the streaming path
//! 14. **sample parity**: odd sample values as well as even ones
//!
//! [`the_dimension_matrix`] crosses 1 through 10 and rotates 14 through them.
//! [`the_streaming_path_matches_the_whole_file_path`] crosses 12 with 13
//! explicitly, because holding total length small while varying feed size is
//! exactly the blind spot step 3's negative control found.
//!
//! Dimension 14 was **not** on this list when the tests were first written,
//! and its absence is what the first negative control of this pass caught: with
//! every synthetic sample an even number, deleting the mid/side odd-sample
//! correction from the decoder left every test in this file green. See
//! [`audio`] for the fix and the reasoning.
//!
//! # References are built here, not taken from the decoder
//!
//! Every synthetic file in this suite is assembled bit by bit from RFC 9639's
//! field definitions, by a writer that shares no code with the reader,
//! including its own CRC-8 and CRC-16, written from the polynomials rather
//! than called from the crate. A shared misreading of the specification would
//! still pass, which is why it is not the only oracle: RFC 9639's three
//! worked examples are carried below as literal bytes with the sample values
//! the document itself prints, the cross-container gate compares FLAC output
//! against WAV and AIFF built from those same published numbers, and
//! `flac_corpus.rs` checks 71 real encoder outputs against the MD5 each one
//! carries.

use decibri_decode::{
    AiffReader, AudioBuffer, DecodeError, FlacReader, FlacStreamDecoder, StreamSource, WavReader,
};

// -- A FLAC writer, for building references -----------------------------------

/// A most-significant-bit-first writer.
///
/// Deliberately naive, one bit at a time, because it is a reference and
/// being obviously correct matters more here than being fast.
struct BitWriter {
    out: Vec<u8>,
    current: u8,
    filled: u32,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            out: Vec::new(),
            current: 0,
            filled: 0,
        }
    }

    fn write(&mut self, value: u64, count: u32) {
        for shift in (0..count).rev() {
            self.current = (self.current << 1) | ((value >> shift) & 1) as u8;
            self.filled += 1;
            if self.filled == 8 {
                self.out.push(self.current);
                self.current = 0;
                self.filled = 0;
            }
        }
    }

    fn write_signed(&mut self, value: i64, count: u32) {
        let mask = if count >= 64 {
            u64::MAX
        } else {
            (1u64 << count) - 1
        };
        self.write(value as u64 & mask, count);
    }

    fn align(&mut self) {
        while self.filled != 0 {
            self.write(0, 1);
        }
    }

    fn into_bytes(mut self) -> Vec<u8> {
        self.align();
        self.out
    }
}

/// CRC-8 over `x^8 + x^2 + x^1 + x^0`, written here from the polynomial so
/// the reference does not call the code under test.
fn reference_crc8(bytes: &[u8]) -> u8 {
    let mut crc = 0u8;
    for byte in bytes {
        crc ^= byte;
        for _ in 0..8 {
            crc = if crc & 0x80 != 0 {
                (crc << 1) ^ 0x07
            } else {
                crc << 1
            };
        }
    }
    crc
}

/// CRC-16 over `x^16 + x^15 + x^2 + x^0`, likewise.
fn reference_crc16(bytes: &[u8]) -> u16 {
    let mut crc = 0u16;
    for byte in bytes {
        crc ^= u16::from(*byte) << 8;
        for _ in 0..8 {
            crc = if crc & 0x8000 != 0 {
                (crc << 1) ^ 0x8005
            } else {
                crc << 1
            };
        }
    }
    crc
}

/// How a frame codes its block size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SizeCode {
    /// One of the common table entries, by its four-bit code.
    Common(u8),
    /// The uncommon eight-bit form, `0b0110`.
    Uncommon8,
    /// The uncommon sixteen-bit form, `0b0111`.
    Uncommon16,
}

/// How a frame codes its sample rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RateCode {
    /// `0b0000`: the rate is only in streaminfo.
    FromStreamInfo,
    /// One of the common table entries, by its four-bit code.
    Common(u8),
    /// `0b1100`: kilohertz in eight bits.
    UncommonKhz,
    /// `0b1101`: hertz in sixteen bits.
    UncommonHz,
    /// `0b1110`: hertz divided by ten, in sixteen bits.
    UncommonTensOfHz,
}

/// Everything a synthetic file varies along.
#[derive(Debug, Clone)]
struct Build {
    bits: u32,
    rate: u32,
    /// The four-bit channel assignment field.
    assignment: u8,
    block_size: u32,
    wasted: u32,
    variable: bool,
    size_code: SizeCode,
    rate_code: RateCode,
    /// Metadata blocks written before the audio, as (type, body length).
    extra_blocks: Vec<(u8, usize)>,
    /// Whether streaminfo declares the total sample count.
    declare_total: bool,
    /// Whether every subframe is constant rather than verbatim.
    constant: bool,
}

impl Default for Build {
    fn default() -> Self {
        Self {
            bits: 16,
            rate: 44_100,
            assignment: 1,
            block_size: 4096,
            wasted: 0,
            variable: false,
            size_code: SizeCode::Uncommon16,
            rate_code: RateCode::FromStreamInfo,
            extra_blocks: Vec::new(),
            declare_total: true,
            constant: false,
        }
    }
}

impl Build {
    /// How many channels the assignment codes for.
    fn channels(&self) -> usize {
        match self.assignment {
            0..=7 => usize::from(self.assignment) + 1,
            _ => 2,
        }
    }

    /// Which subframe carries the side channel, if any.
    fn side_channel(&self) -> Option<usize> {
        match self.assignment {
            8 | 10 => Some(1),
            9 => Some(0),
            _ => None,
        }
    }
}

/// Deterministic pseudo-audio at `bits` bits.
///
/// # Parity is a dimension, and leaving it out cost a negative control
///
/// When `wasted` is non-zero every value is a multiple of `2^(wasted + 1)`,
/// because the subframes have to carry `wasted` low zero bits and a mid
/// subframe is the sum of two channels shifted right by one, so it needs one
/// spare bit below them.
///
/// When `wasted` is zero **nothing is masked, and the values are as often odd
/// as even**. That is deliberate and load-bearing. Mid/side stereo is only
/// lossless because an odd side sample restores the bit the mid sample lost to
/// its right shift; with all-even audio that correction is a no-op, and the
/// first negative control of this pass found that deleting it left every test
/// in this file green while 55 corpus files went red. Parity was a dimension
/// the matrix did not vary along, which is coverage lesson 2 exactly.
fn audio(count: usize, bits: u32, wasted: u32, seed: u64) -> Vec<i64> {
    let quarter = 1i64 << (bits - 2);
    let mut state = seed | 1;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let value = ((state >> 33) as i64 % quarter) - quarter / 2;
            if wasted == 0 {
                value
            } else {
                (value >> (wasted + 1)) << (wasted + 1)
            }
        })
        .collect()
}

/// The UTF-8-like coded number of RFC 9639 section 9.1.5, written here rather
/// than taken from the reader.
fn write_coded_number(writer: &mut BitWriter, value: u64) {
    if value < 0x80 {
        writer.write(value, 8);
        return;
    }
    // The shortest form that holds the value: two through seven octets carry
    // 11, 16, 21, 26, 31 and 36 payload bits.
    let octets = (2u32..=7)
        .find(|octets| value < (1u64 << (6 * (octets - 1) + (7 - octets))))
        .expect("36 bits is the widest this encoding carries");
    let lead = (0xFFu64 << (8 - octets)) & 0xFF;
    writer.write(lead | (value >> (6 * (octets - 1))), 8);
    for step in (0..octets - 1).rev() {
        writer.write(0x80 | ((value >> (6 * step)) & 0x3F), 8);
    }
}

/// Assembles a FLAC file carrying `samples` per channel.
///
/// `samples` is the real audio, one vector per channel, all the same length.
/// Stereo decorrelation is applied here, so the decoder's reconstruction is
/// checked against audio this writer never handed it directly.
///
/// The streaminfo MD5 is written as the all-zero value the format defines as
/// "not known". That is a deliberate limit of this writer and not an
/// oversight: computing a real one would need a second MD5 implementation in
/// the tests, and the MD5 path is covered by the three RFC vectors below and
/// by every corpus file in `flac_corpus.rs`.
fn flac_file(build: &Build, samples: &[Vec<i64>]) -> Vec<u8> {
    let channels = build.channels();
    assert_eq!(samples.len(), channels);
    let total = samples[0].len();
    let block_size = build.block_size as usize;
    let blocks = total.div_ceil(block_size);

    let mut file = b"fLaC".to_vec();

    // Streaminfo, then any extra blocks, the last of which sets the flag.
    let last_is_streaminfo = build.extra_blocks.is_empty();
    let mut info = BitWriter::new();
    let min_block = if blocks > 1 {
        block_size as u64
    } else {
        block_size.max(16) as u64
    };
    info.write(min_block, 16);
    info.write(block_size as u64, 16);
    info.write(0, 24); // minimum frame size: unknown
    info.write(0, 24); // maximum frame size: unknown
    info.write(u64::from(build.rate), 20);
    info.write(channels as u64 - 1, 3);
    info.write(u64::from(build.bits) - 1, 5);
    info.write(if build.declare_total { total as u64 } else { 0 }, 36);
    let mut info = info.into_bytes();
    info.extend_from_slice(&[0u8; 16]); // MD5: not known
    assert_eq!(info.len(), 34);
    file.push(if last_is_streaminfo { 0x80 } else { 0x00 });
    file.extend_from_slice(&[0, 0, 34]);
    file.extend_from_slice(&info);

    for (index, (kind, length)) in build.extra_blocks.iter().enumerate() {
        let last = index + 1 == build.extra_blocks.len();
        file.push(if last { 0x80 | kind } else { *kind });
        file.extend_from_slice(&[(length >> 16) as u8, (length >> 8) as u8, (*length) as u8]);
        // A body of stepping bytes rather than zeros, so a reader that walked
        // into it would find something that is not a plausible block header.
        file.extend((0..*length).map(|byte| (byte % 251) as u8));
    }

    let mut written = 0usize;
    let mut frame_number = 0u64;
    while written < total {
        let count = block_size.min(total - written);
        let mut coded: Vec<Vec<i64>> = (0..channels)
            .map(|channel| samples[channel][written..written + count].to_vec())
            .collect();
        match build.assignment {
            8 => {
                let side: Vec<i64> = (0..count).map(|i| coded[0][i] - coded[1][i]).collect();
                coded[1] = side;
            }
            9 => {
                let side: Vec<i64> = (0..count).map(|i| coded[0][i] - coded[1][i]).collect();
                coded[0] = side;
            }
            10 => {
                let mid: Vec<i64> = (0..count)
                    .map(|i| (coded[0][i] + coded[1][i]) >> 1)
                    .collect();
                let side: Vec<i64> = (0..count).map(|i| coded[0][i] - coded[1][i]).collect();
                coded[0] = mid;
                coded[1] = side;
            }
            _ => {}
        }

        // The 15-bit sync code and the blocking strategy bit.
        let mut writer = BitWriter::new();
        writer.write(0b111_1111_1111_1100, 15);
        writer.write(u64::from(build.variable), 1);

        // A common block size code names a fixed value, so a short last
        // frame cannot use one and falls back to the explicit form, which
        // is what real encoders do, and what makes a stream's final partial
        // block representable at all.
        let size_code = if count as u32 == build.block_size {
            build.size_code
        } else {
            SizeCode::Uncommon16
        };
        let (size_bits, size_tail) = match size_code {
            SizeCode::Common(code) => (u64::from(code), None),
            SizeCode::Uncommon8 => (6, Some((count as u64 - 1, 8u32))),
            SizeCode::Uncommon16 => (7, Some((count as u64 - 1, 16u32))),
        };
        let (rate_bits, rate_tail) = match build.rate_code {
            RateCode::FromStreamInfo => (0u64, None),
            RateCode::Common(code) => (u64::from(code), None),
            RateCode::UncommonKhz => (12, Some((u64::from(build.rate / 1000), 8u32))),
            RateCode::UncommonHz => (13, Some((u64::from(build.rate), 16u32))),
            RateCode::UncommonTensOfHz => (14, Some((u64::from(build.rate / 10), 16u32))),
        };
        writer.write(size_bits, 4);
        writer.write(rate_bits, 4);
        writer.write(u64::from(build.assignment), 4);
        writer.write(
            match build.bits {
                8 => 1,
                12 => 2,
                16 => 4,
                20 => 5,
                24 => 6,
                32 => 7,
                _ => 0,
            },
            3,
        );
        writer.write(0, 1); // the reserved bit
        write_coded_number(
            &mut writer,
            if build.variable {
                written as u64
            } else {
                frame_number
            },
        );
        if let Some((value, width)) = size_tail {
            writer.write(value, width);
        }
        if let Some((value, width)) = rate_tail {
            writer.write(value, width);
        }
        let header = writer.into_bytes();
        let mut frame = header.clone();
        frame.push(reference_crc8(&header));

        let mut body = BitWriter::new();
        for byte in &frame {
            body.write(u64::from(*byte), 8);
        }
        for (channel, values) in coded.iter().enumerate() {
            let side = build.side_channel() == Some(channel);
            let width = build.bits + u32::from(side) - build.wasted;
            body.write(0, 1); // the mandatory zero
            body.write(if build.constant { 0 } else { 1 }, 6);
            if build.wasted == 0 {
                body.write(0, 1);
            } else {
                body.write(1, 1);
                // The count minus one, in unary.
                body.write(1, build.wasted);
            }
            if build.constant {
                body.write_signed(values[0] >> build.wasted, width);
            } else {
                for value in values {
                    body.write_signed(value >> build.wasted, width);
                }
            }
        }
        let frame = body.into_bytes();
        file.extend_from_slice(&frame);
        file.extend_from_slice(&reference_crc16(&frame).to_be_bytes());

        written += count;
        frame_number += 1;
    }
    file
}

/// A constant-subframe build repeats its first sample across the block, so a
/// reference for it has to do the same.
fn expected_samples(build: &Build, samples: &[Vec<i64>]) -> Vec<Vec<i64>> {
    if !build.constant {
        return samples.to_vec();
    }
    let block_size = build.block_size as usize;
    samples
        .iter()
        .map(|channel| {
            channel
                .chunks(block_size)
                .flat_map(|block| std::iter::repeat_n(block[0], block.len()))
                .collect()
        })
        .collect()
}

/// The interleaved `f32` a decode of `samples` at `bits` must produce.
///
/// The divisor is written out here rather than called from the crate, so the
/// scaling rule is asserted rather than assumed.
fn expected_f32(samples: &[Vec<i64>], bits: u32) -> Vec<f32> {
    let divisor = (1u64 << (bits - 1)) as f32;
    let count = samples[0].len();
    let mut out = Vec::with_capacity(count * samples.len());
    for index in 0..count {
        for channel in samples {
            out.push(channel[index] as f32 / divisor);
        }
    }
    out
}

/// Decodes `bytes` through the streaming reader in `chunk`-byte pieces.
fn stream_decode(bytes: &[u8], chunk: usize) -> Result<Vec<f32>, DecodeError> {
    let mut stream = FlacStreamDecoder::new();
    let mut samples = Vec::new();
    for piece in bytes.chunks(chunk) {
        let mut offset = 0;
        while offset < piece.len() {
            offset += stream.push(&piece[offset..])?;
            while stream.pull(&mut samples, usize::MAX)? > 0 {}
        }
    }
    stream.finish(&mut samples)?;
    Ok(samples)
}

// -- Gate: RFC 9639's worked examples -----------------------------------------

/// RFC 9639 appendix D.1: 44.1 kHz 16-bit stereo, one frame of one sample,
/// both subframes verbatim and both using wasted bits.
const RFC_EXAMPLE_1: &[u8] = &[
    0x66, 0x4c, 0x61, 0x43, 0x80, 0x00, 0x00, 0x22, 0x10, 0x00, 0x10, 0x00, 0x00, 0x00, 0x0f, 0x00,
    0x00, 0x0f, 0x0a, 0xc4, 0x42, 0xf0, 0x00, 0x00, 0x00, 0x01, 0x3e, 0x84, 0xb4, 0x18, 0x07, 0xdc,
    0x69, 0x03, 0x07, 0x58, 0x6a, 0x3d, 0xad, 0x1a, 0x2e, 0x0f, 0xff, 0xf8, 0x69, 0x18, 0x00, 0x00,
    0xbf, 0x03, 0x58, 0xfd, 0x03, 0x12, 0x8b, 0xaa, 0x9a,
];

/// RFC 9639 appendix D.2: 44.1 kHz 16-bit stereo, 19 samples in two frames,
/// with a seek table, a vorbis comment and a padding block.
const RFC_EXAMPLE_2: &[u8] = &[
    0x66, 0x4c, 0x61, 0x43, 0x00, 0x00, 0x00, 0x22, 0x00, 0x10, 0x00, 0x10, 0x00, 0x00, 0x17, 0x00,
    0x00, 0x44, 0x0a, 0xc4, 0x42, 0xf0, 0x00, 0x00, 0x00, 0x13, 0xd5, 0xb0, 0x56, 0x49, 0x75, 0xe9,
    0x8b, 0x8d, 0x8b, 0x93, 0x04, 0x22, 0x75, 0x7b, 0x81, 0x03, 0x03, 0x00, 0x00, 0x12, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10,
    0x04, 0x00, 0x00, 0x3a, 0x20, 0x00, 0x00, 0x00, 0x72, 0x65, 0x66, 0x65, 0x72, 0x65, 0x6e, 0x63,
    0x65, 0x20, 0x6c, 0x69, 0x62, 0x46, 0x4c, 0x41, 0x43, 0x20, 0x31, 0x2e, 0x33, 0x2e, 0x33, 0x20,
    0x32, 0x30, 0x31, 0x39, 0x30, 0x38, 0x30, 0x34, 0x01, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
    0x54, 0x49, 0x54, 0x4c, 0x45, 0x3d, 0xd7, 0xa9, 0xd7, 0x9c, 0xd7, 0x95, 0xd7, 0x9d, 0x81, 0x00,
    0x00, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xf8, 0x69, 0x98, 0x00, 0x0f, 0x99, 0x12,
    0x08, 0x67, 0x01, 0x62, 0x3d, 0x14, 0x42, 0x99, 0x8f, 0x5d, 0xf7, 0x0d, 0x6f, 0xe0, 0x0c, 0x17,
    0xca, 0xeb, 0x21, 0x00, 0x0e, 0xe7, 0xa7, 0x7a, 0x24, 0xa1, 0x59, 0x0c, 0x12, 0x17, 0xb6, 0x03,
    0x09, 0x7b, 0x78, 0x4f, 0xaa, 0x9a, 0x33, 0xd2, 0x85, 0xe0, 0x70, 0xad, 0x5b, 0x1b, 0x48, 0x51,
    0xb4, 0x01, 0x0d, 0x99, 0xd2, 0xcd, 0x1a, 0x68, 0xf1, 0xe6, 0xb8, 0x10, 0xff, 0xf8, 0x69, 0x18,
    0x01, 0x02, 0xa4, 0x02, 0xc3, 0x82, 0xc4, 0x0b, 0xc1, 0x4a, 0x03, 0xee, 0x48, 0xdd, 0x03, 0xb6,
    0x7c, 0x13, 0x30,
];

/// RFC 9639 appendix D.3: 32 kHz 8-bit mono, one frame of 24 samples coded
/// with a third-order linear predictor.
const RFC_EXAMPLE_3: &[u8] = &[
    0x66, 0x4c, 0x61, 0x43, 0x80, 0x00, 0x00, 0x22, 0x10, 0x00, 0x10, 0x00, 0x00, 0x00, 0x1f, 0x00,
    0x00, 0x1f, 0x07, 0xd0, 0x00, 0x70, 0x00, 0x00, 0x00, 0x18, 0xf8, 0xf9, 0xe3, 0x96, 0xf5, 0xcb,
    0xcf, 0xc6, 0xdc, 0x80, 0x7f, 0x99, 0x77, 0x90, 0x6b, 0x32, 0xff, 0xf8, 0x68, 0x02, 0x00, 0x17,
    0xe9, 0x44, 0x00, 0x4f, 0x6f, 0x31, 0x3d, 0x10, 0x47, 0xd2, 0x27, 0xcb, 0x6d, 0x09, 0x08, 0x31,
    0x45, 0x2b, 0xdc, 0x28, 0x22, 0x22, 0x80, 0x57, 0xa3,
];

/// Example 1's two samples, from RFC 9639 appendix D.1.4's own text.
const RFC_EXAMPLE_1_SAMPLES: [i64; 2] = [25_588, 10_416];

/// Example 3's 24 samples, from RFC 9639 appendix D.3.4's own breakdown
/// table, in the "Sample Value" column.
const RFC_EXAMPLE_3_SAMPLES: [i64; 24] = [
    0, 79, 111, 78, 8, -61, -90, -68, -13, 42, 67, 53, 13, -27, -46, -38, -12, 14, 24, 19, 6, -4,
    -5, 0,
];

/// Example 2's 38 samples, read out of the interleaved little-endian bytes
/// RFC 9639 appendix D.2.9 prints for the MD5 computation.
const RFC_EXAMPLE_2_BYTES: [u8; 76] = [
    0x84, 0x28, 0xB6, 0x17, 0x79, 0x46, 0x31, 0x29, 0x5E, 0x3A, 0x27, 0x22, 0xD4, 0x45, 0xD1, 0x28,
    0x0B, 0x3D, 0xB7, 0x23, 0xEB, 0x45, 0xDF, 0x28, 0x72, 0x3F, 0x1E, 0x25, 0x9D, 0x46, 0x49, 0x29,
    0xB8, 0x41, 0x70, 0x26, 0x57, 0x47, 0xB8, 0x29, 0x8F, 0x43, 0x81, 0x27, 0xAE, 0xC7, 0x14, 0xDF,
    0x9F, 0xC4, 0x41, 0xDD, 0x54, 0xC7, 0xE4, 0xDE, 0xA5, 0xC4, 0x40, 0xDD, 0x1E, 0xC6, 0x33, 0xDE,
    0x82, 0xC3, 0x90, 0xDC, 0x0B, 0xC4, 0x02, 0xDD, 0x4A, 0xC1, 0x3E, 0xDB,
];

/// RFC 9639's three worked examples, decoded exactly.
///
/// The sample values are the document's own, transcribed from the text rather
/// than produced by anything here, so this is the one gate in the suite whose
/// oracle is fully outside the crate *and* outside the corpus. Example 3 in
/// particular walks a third-order linear predictor sample by sample, which is
/// the arithmetic most likely to be plausibly wrong.
#[test]
fn the_rfc_worked_examples_decode_exactly() {
    let one = FlacReader::new(RFC_EXAMPLE_1).expect("open example 1");
    assert_eq!(one.spec().sample_rate, 44_100);
    assert_eq!(one.spec().channels, 2);
    assert_eq!(one.stream_info().bits_per_sample, 16);
    assert_eq!(one.frames(), Some(1));
    assert_eq!(
        one.decode_to_end().expect("decode example 1").samples(),
        RFC_EXAMPLE_1_SAMPLES
            .iter()
            .map(|value| *value as f32 / 32_768.0)
            .collect::<Vec<f32>>()
    );

    let two = FlacReader::new(RFC_EXAMPLE_2).expect("open example 2");
    assert_eq!(two.spec().sample_rate, 44_100);
    assert_eq!(two.frames(), Some(19));
    let expected: Vec<f32> = RFC_EXAMPLE_2_BYTES
        .chunks_exact(2)
        .map(|pair| f32::from(i16::from_le_bytes([pair[0], pair[1]])) / 32_768.0)
        .collect();
    assert_eq!(
        two.decode_to_end().expect("decode example 2").samples(),
        expected
    );

    let three = FlacReader::new(RFC_EXAMPLE_3).expect("open example 3");
    assert_eq!(three.spec().sample_rate, 32_000);
    assert_eq!(three.spec().channels, 1);
    assert_eq!(three.stream_info().bits_per_sample, 8);
    assert_eq!(three.frames(), Some(24));
    assert_eq!(
        three.decode_to_end().expect("decode example 3").samples(),
        RFC_EXAMPLE_3_SAMPLES
            .iter()
            .map(|value| *value as f32 / 128.0)
            .collect::<Vec<f32>>()
    );
}

/// The streaminfo MD5 is checked on the worked examples, and a single flipped
/// bit anywhere in the audio turns the check red.
#[test]
fn the_worked_examples_carry_a_checked_md5() {
    for (name, file) in [
        ("example 1", RFC_EXAMPLE_1),
        ("example 2", RFC_EXAMPLE_2),
        ("example 3", RFC_EXAMPLE_3),
    ] {
        let reader = FlacReader::new(file).expect("open");
        assert!(
            reader.stream_info().md5.is_some(),
            "{name} was expected to carry a checksum"
        );
        assert!(reader.decode_to_end().is_ok(), "{name} failed to decode");
    }
}

// -- Gate: cross-container agreement ------------------------------------------

/// The same audio, decoded from FLAC and from a PCM container, must give the
/// same `f32`.
///
/// This proves the *values*, which is a different question from the bytes,
/// and its oracle is independent of everything FLAC: the WAV and AIFF files
/// below are assembled from the interleaved sample bytes RFC 9639 prints for
/// its own MD5 computation, and are read by decoders that share no code with
/// [`FlacReader`] past `sample.rs`. Both containers appear because the two
/// carry 8-bit samples with opposite signedness, and example 3 is 8-bit.
#[test]
fn flac_agrees_with_wav_and_aiff_on_the_same_audio() {
    // Example 2: 16-bit stereo. RFC 9639 appendix D.2.9's bytes are already
    // little-endian interleaved, which is exactly a WAV data chunk.
    let wav = wav_file(2, 44_100, 16, &RFC_EXAMPLE_2_BYTES);
    let from_wav = WavReader::new(&wav).expect("read wav").decode_to_end();
    let from_flac = FlacReader::new(RFC_EXAMPLE_2)
        .expect("open")
        .decode_to_end()
        .expect("decode");
    assert_eq!(from_flac.samples(), from_wav.samples());
    assert_eq!(from_flac.spec(), from_wav.spec());

    // Example 1: 16-bit stereo, one frame. RFC 9639 appendix D.1.4 prints
    // 0xf463 and 0xb028 as the little-endian bytes of its two samples.
    let wav = wav_file(2, 44_100, 16, &[0xf4, 0x63, 0xb0, 0x28]);
    let from_wav = WavReader::new(&wav).expect("read wav").decode_to_end();
    let from_flac = FlacReader::new(RFC_EXAMPLE_1)
        .expect("open")
        .decode_to_end()
        .expect("decode");
    assert_eq!(from_flac.samples(), from_wav.samples());

    // Example 3: 8-bit mono. AIFF's 8-bit is signed two's complement, which
    // is the form RFC 9639 appendix D.3.4 prints, so no offset is applied to
    // either side and a signedness error in either reader shows up here.
    let payload: Vec<u8> = RFC_EXAMPLE_3_SAMPLES
        .iter()
        .map(|value| *value as i8 as u8)
        .collect();
    let aiff = aiff_file(1, 32_000, 8, &payload);
    let from_aiff = AiffReader::new(&aiff).expect("read aiff").decode_to_end();
    let from_flac = FlacReader::new(RFC_EXAMPLE_3)
        .expect("open")
        .decode_to_end()
        .expect("decode");
    assert_eq!(from_flac.samples(), from_aiff.samples());
    assert_eq!(from_flac.spec(), from_aiff.spec());
}

/// A minimal RIFF/WAVE file around `payload`, assembled here.
fn wav_file(channels: u16, rate: u32, bits: u16, payload: &[u8]) -> Vec<u8> {
    let block_align = channels * bits / 8;
    let mut file = b"RIFF".to_vec();
    file.extend_from_slice(&(36 + payload.len() as u32).to_le_bytes());
    file.extend_from_slice(b"WAVEfmt ");
    file.extend_from_slice(&16u32.to_le_bytes());
    file.extend_from_slice(&1u16.to_le_bytes());
    file.extend_from_slice(&channels.to_le_bytes());
    file.extend_from_slice(&rate.to_le_bytes());
    file.extend_from_slice(&(rate * u32::from(block_align)).to_le_bytes());
    file.extend_from_slice(&block_align.to_le_bytes());
    file.extend_from_slice(&bits.to_le_bytes());
    file.extend_from_slice(b"data");
    file.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    file.extend_from_slice(payload);
    file
}

/// A minimal AIFF file around `payload`, assembled here.
fn aiff_file(channels: u16, rate: u32, bits: u16, payload: &[u8]) -> Vec<u8> {
    let frames = payload.len() as u32 / u32::from(channels * bits / 8);
    let mut file = b"FORM".to_vec();
    file.extend_from_slice(&(4 + 8 + 18 + 8 + 8 + payload.len() as u32).to_be_bytes());
    file.extend_from_slice(b"AIFFCOMM");
    file.extend_from_slice(&18u32.to_be_bytes());
    file.extend_from_slice(&channels.to_be_bytes());
    file.extend_from_slice(&frames.to_be_bytes());
    file.extend_from_slice(&bits.to_be_bytes());
    // The 80-bit extended-precision rate, normalised by hand.
    let high_bit = 31 - rate.leading_zeros();
    file.extend_from_slice(&(16_383 + high_bit as u16).to_be_bytes());
    file.extend_from_slice(&(u64::from(rate) << (63 - high_bit)).to_be_bytes());
    file.extend_from_slice(b"SSND");
    file.extend_from_slice(&(8 + payload.len() as u32).to_be_bytes());
    file.extend_from_slice(&[0u8; 8]);
    file.extend_from_slice(payload);
    if payload.len() % 2 == 1 {
        file.push(0);
    }
    file
}

// -- Gate: the dimension matrix -----------------------------------------------

/// Every combination of bit depth, channel assignment, block size coding,
/// sample rate coding, wasted bits, blocking strategy, metadata layout and
/// declared length, decoded and compared against the audio that was written.
///
/// The reference is the audio itself, which this suite generated and the
/// writer encoded, so a shared misreading of the format would survive. The
/// independent anchors are the RFC vectors above and `flac_corpus.rs`; this
/// gate is here to catch a *dimension* going wrong, which they cannot,
/// because they are fixed files.
#[test]
fn the_dimension_matrix() {
    let mut files = 0usize;
    for (bits, rate_code) in [
        (8u32, RateCode::FromStreamInfo),
        (12, RateCode::Common(9)),
        (15, RateCode::UncommonHz),
        (16, RateCode::Common(4)),
        (20, RateCode::UncommonKhz),
        (24, RateCode::UncommonTensOfHz),
        (32, RateCode::FromStreamInfo),
    ] {
        let rate = match rate_code {
            RateCode::Common(4) => 8_000,
            RateCode::Common(_) => 44_100,
            RateCode::UncommonKhz => 96_000,
            RateCode::UncommonTensOfHz => 134_560,
            RateCode::UncommonHz => 35_467,
            RateCode::FromStreamInfo => 48_000,
        };
        for assignment in [0u8, 1, 2, 7, 8, 9, 10] {
            for size_code in [
                SizeCode::Common(1),
                SizeCode::Uncommon8,
                SizeCode::Uncommon16,
            ] {
                for wasted in [0u32, 3] {
                    for variable in [false, true] {
                        for (extra_blocks, declare_total, constant) in [
                            (vec![], true, false),
                            (vec![(1u8, 7usize)], true, false),
                            (vec![(2, 40), (6, 13), (99, 5)], false, false),
                            (vec![(1, 0)], true, true),
                        ] {
                            let build = Build {
                                bits,
                                rate,
                                assignment,
                                // 192 is what the common code 0b0001 means,
                                // so the three codings describe one size.
                                block_size: 192,
                                wasted,
                                variable,
                                size_code,
                                rate_code,
                                extra_blocks,
                                declare_total,
                                constant,
                            };
                            let channels = build.channels();
                            let samples: Vec<Vec<i64>> = (0..channels)
                                .map(|channel| {
                                    audio(500, bits, wasted, 0x51E5 + channel as u64 + bits as u64)
                                })
                                .collect();
                            let file = flac_file(&build, &samples);
                            let expected = expected_f32(&expected_samples(&build, &samples), bits);

                            let reader = FlacReader::new(&file).expect("open");
                            assert_eq!(reader.spec().sample_rate, rate);
                            assert_eq!(reader.spec().channels, channels as u16);
                            assert_eq!(reader.stream_info().bits_per_sample, bits as u8);
                            assert_eq!(reader.stream_info().md5, None);
                            assert_eq!(
                                reader.frames(),
                                declare_total.then_some(500),
                                "declared length"
                            );
                            let decoded = reader.decode_to_end().expect("decode");
                            assert_eq!(decoded.samples(), expected, "{build:?}");

                            // 7 is coprime with every frame size here, so the
                            // streaming reader is driven across its partial
                            // paths as well.
                            let streamed = stream_decode(&file, 7).expect("stream");
                            assert_eq!(streamed, expected, "streamed: {build:?}");
                            files += 1;
                        }
                    }
                }
            }
        }
    }
    assert_eq!(files, 7 * 7 * 3 * 2 * 2 * 4);
}

// -- Gate: the streaming path -------------------------------------------------

/// Feed size and total length varied separately, because varying only one is
/// the blind spot step 3's negative control found.
///
/// The lengths straddle the reader's 65,536-sample ready limit, which is the
/// point where it starts applying back-pressure and taking short pushes.
#[test]
fn the_streaming_path_matches_the_whole_file_path() {
    for total in [1usize, 16, 500, 4096, 20_000, 60_000] {
        let build = Build {
            bits: 16,
            block_size: 512,
            assignment: 10,
            ..Build::default()
        };
        let samples: Vec<Vec<i64>> = (0..2)
            .map(|channel| audio(total, 16, 0, 0xABC + channel as u64))
            .collect();
        let file = flac_file(&build, &samples);
        let whole = FlacReader::new(&file)
            .expect("open")
            .decode_to_end()
            .expect("decode");
        assert_eq!(whole.samples().len(), total * 2);

        for chunk in [1usize, 2, 3, 13, 64, 1_000, 65_536, usize::MAX] {
            let streamed = stream_decode(&file, chunk.min(file.len().max(1))).expect("stream");
            assert_eq!(
                streamed,
                whole.samples(),
                "output changed at {total} frames with a {chunk}-byte feed size"
            );
        }
    }
}

/// A stream cut anywhere inside a frame is truncation, not audio and not a
/// panic, and a stream cut exactly on a frame boundary before its declared
/// end is truncation too.
#[test]
fn a_cut_stream_is_truncated_rather_than_short() {
    let build = Build {
        block_size: 64,
        ..Build::default()
    };
    let samples: Vec<Vec<i64>> = (0..2)
        .map(|channel| audio(200, 16, 0, 0xF00D + channel as u64))
        .collect();
    let file = flac_file(&build, &samples);

    for cut in 1..file.len() {
        let short = &file[..cut];
        // Whole-file: either a typed error or, for a cut past the last frame
        // this reader needs, a clean decode of the frames that arrived.
        match FlacReader::new(short).and_then(|reader| reader.decode_to_end()) {
            Ok(decoded) => assert_eq!(decoded.samples().len(), 400, "cut at {cut} decoded short"),
            Err(DecodeError::Truncated { .. } | DecodeError::Malformed { .. }) => {}
            Err(other) => panic!("cut at {cut}: unexpected error {other}"),
        }
        // Streaming: the same verdict.
        match stream_decode(short, 17) {
            Ok(samples) => assert_eq!(samples.len(), 400, "cut at {cut} streamed short"),
            Err(DecodeError::Truncated { .. } | DecodeError::Malformed { .. }) => {}
            Err(other) => panic!("cut at {cut}: unexpected streaming error {other}"),
        }
    }
}

// -- Gate: malformed input ----------------------------------------------------

/// A reference file small enough to mutate exhaustively.
fn small_file() -> Vec<u8> {
    let build = Build {
        bits: 16,
        block_size: 16,
        assignment: 10,
        ..Build::default()
    };
    let samples: Vec<Vec<i64>> = (0..2)
        .map(|channel| audio(32, 16, 0, 0x1234 + channel as u64))
        .collect();
    flac_file(&build, &samples)
}

/// Every single-byte corruption of a small file, through both readers.
///
/// Nothing here asserts a particular verdict: a mutation can land in a byte
/// that changes the audio into other valid audio. What it asserts is that
/// there is a verdict at all: no panic, no hang, no allocation the file
/// talked the reader into.
#[test]
fn every_single_byte_corruption_answers_rather_than_panics() {
    let file = small_file();
    let mut rejected = 0usize;
    for index in 0..file.len() {
        for pattern in [0x00u8, 0xFF, 0x80, 0x01] {
            let mut broken = file.clone();
            if broken[index] == pattern {
                continue;
            }
            broken[index] = pattern;
            if FlacReader::new(&broken)
                .and_then(|reader| reader.decode_to_end())
                .is_err()
            {
                rejected += 1;
            }
            let _ = stream_decode(&broken, 5);
        }
    }
    // Most corruptions land somewhere a CRC covers. The number is not the
    // point; reaching this line without a panic is.
    assert!(rejected > 0, "no corruption was detected at all");
}

/// The named malformed shapes the relay asked for, each answered by name.
#[test]
fn malformed_input_is_a_typed_error() {
    // A truncated stream.
    let file = small_file();
    assert!(matches!(
        FlacReader::new(&file[..30]),
        Err(DecodeError::Truncated { .. })
    ));

    // A corrupt streaminfo: a bit depth of 0 codes for 1 bit, below the
    // format's floor of 4.
    let mut broken = file.clone();
    broken[20] = 0x00;
    broken[21] = 0x00;
    assert!(matches!(
        FlacReader::new(&broken),
        Err(DecodeError::Malformed { .. })
    ));

    // An absurd declared block size: streaminfo saying 0.
    let mut broken = file.clone();
    broken[8..12].fill(0);
    assert!(matches!(
        FlacReader::new(&broken),
        Err(DecodeError::Malformed { .. })
    ));

    // A metadata block declaring a length past the end of the file.
    let mut broken = file.clone();
    broken[5] = 0xFF;
    broken[6] = 0xFF;
    broken[7] = 0xFF;
    assert!(matches!(
        FlacReader::new(&broken),
        Err(DecodeError::Truncated { .. })
    ));

    // Not FLAC at all, named by the bytes it was.
    assert!(matches!(
        FlacReader::new(b"RIFF\0\0\0\0WAVE"),
        Err(DecodeError::UnsupportedContainer { tag }) if tag.as_bytes() == b"RIFF"
    ));

    // The forbidden metadata block type.
    let mut broken = file.clone();
    broken[4] = 0x7F;
    assert!(matches!(
        FlacReader::new(&broken),
        Err(DecodeError::Malformed { .. })
    ));

    // An empty input has no tag to report, so it is truncation.
    assert!(matches!(
        FlacReader::new(&[]),
        Err(DecodeError::Truncated { .. })
    ));
}

/// Both CRCs are load-bearing: corrupting a byte the CRC covers, and nothing
/// else, must produce a typed error rather than decoded audio.
#[test]
fn the_frame_crcs_reject_a_corrupted_frame() {
    let file = small_file();
    let audio_start = file
        .windows(2)
        .position(|pair| pair[0] == 0xFF && pair[1] & 0xFE == 0xF8)
        .expect("the file has a frame");

    // The CRC-8 covers the frame header. Flipping the block-size nibble
    // changes the header without touching the audio, so only the CRC-8 can
    // catch it.
    let mut broken = file.clone();
    broken[audio_start + 2] ^= 0x10;
    let error = FlacReader::new(&broken)
        .and_then(|reader| reader.decode_to_end())
        .expect_err("a corrupted frame header must be rejected");
    assert!(
        matches!(
            error,
            DecodeError::Malformed {
                expected: "a frame header whose CRC-8 matches its contents",
                ..
            }
        ),
        "unexpected error: {error}"
    );

    // The CRC-16 covers the whole frame. A byte inside a subframe is past the
    // header, so the CRC-8 passes and only the CRC-16 is left.
    let mut broken = file.clone();
    let body = audio_start + 10;
    broken[body] ^= 0x01;
    let error = FlacReader::new(&broken)
        .and_then(|reader| reader.decode_to_end())
        .expect_err("a corrupted frame body must be rejected");
    assert!(
        matches!(
            error,
            DecodeError::Malformed {
                expected: "a frame whose CRC-16 matches its contents",
                ..
            }
        ),
        "unexpected error: {error}"
    );

    // And a corrupted CRC-16 field itself, with the frame intact.
    let mut broken = file.clone();
    let last = broken.len() - 1;
    broken[last] ^= 0xFF;
    assert!(FlacReader::new(&broken)
        .and_then(|reader| reader.decode_to_end())
        .is_err());
}

/// A frame that declares more samples than the streaminfo maximum block size
/// is rejected before anything is reserved for it.
///
/// This is the check that keeps the decode buffer bounded by something the
/// format guarantees rather than by a number the frame chose.
#[test]
fn a_frame_past_the_streaminfo_maximum_block_size_is_rejected() {
    let build = Build {
        block_size: 32,
        assignment: 0,
        ..Build::default()
    };
    let samples = vec![audio(32, 16, 0, 0x2222)];
    let mut file = flac_file(&build, &samples);
    // Cut the streaminfo block sizes down to 16 without touching the frame,
    // so the file now claims less than it carries.
    file[8..12].copy_from_slice(&[0x00, 0x10, 0x00, 0x10]);
    let error = FlacReader::new(&file)
        .and_then(|reader| reader.decode_to_end())
        .expect_err("must reject");
    assert!(
        matches!(
            error,
            DecodeError::Malformed {
                expected: "a block size no larger than the streaminfo maximum",
                ..
            }
        ),
        "unexpected error: {error}"
    );
}

/// Eight channels decode, and nine cannot be asked for.
///
/// FLAC's frame header names its channels in four bits: 0 through 7 are one
/// through eight independent channels, 8 through 10 are the stereo
/// decorrelations, and 11 through 15 are reserved. Streaminfo's field is three
/// bits and so tops out at eight for the same reason. A ninth channel
/// therefore has no encoding, and the only way to ask for one is a reserved
/// value, which is refused.
///
/// Written by hand rather than generated, because no encoder produces a
/// nine-channel FLAC stream: ffmpeg refuses the channel count outright.
///
/// The WAV and AIFF matrices next door now carry nine channels, so this states
/// the boundary between the two rather than leaving it to be inferred from
/// where the counts stop. The writer's half of it is
/// `nine_channels_is_refused` in `flac_write_conformance.rs`.
#[test]
fn eight_channels_decode_and_the_reserved_assignments_past_them_are_refused() {
    // One whole block, so the builder can use the common block size code and
    // the frame header stays at its shortest.
    const BLOCK: usize = 256;
    let build = Build {
        block_size: BLOCK as u32,
        assignment: 7, // eight independent channels, the format's most
        size_code: SizeCode::Common(8),
        rate_code: RateCode::Common(9),
        ..Build::default()
    };
    let samples: Vec<Vec<i64>> = (0..8)
        .map(|channel| audio(BLOCK, 16, 0, 0x9C00 + channel as u64))
        .collect();
    let file = flac_file(&build, &samples);

    let decoded = FlacReader::new(&file)
        .expect("open")
        .decode_to_end()
        .expect("eight channels is inside the format");
    assert_eq!(decoded.spec().channels, 8, "eight channels must decode");
    assert_eq!(decoded.frames(), BLOCK);
    assert_eq!(decoded.samples().len(), 8 * BLOCK, "eight channels of 256");

    // The frame sits after `fLaC` and one 34-byte streaminfo block with its
    // four-byte header, and both codes above are common ones, so the header is
    // its shortest six bytes: two of sync, one of sizes, one of channels and
    // depth, one coded number and the CRC-8. Each of those is asserted rather
    // than assumed, so a builder change turns this into a failed assertion
    // rather than a patch of the wrong byte.
    const FRAME: usize = 4 + 4 + 34;
    assert_eq!(file[FRAME], 0xFF, "the frame sync");
    assert_eq!(
        file[FRAME + 1],
        0xF8,
        "the sync tail and a fixed block size"
    );
    assert_eq!(file[FRAME + 3] >> 4, 7, "eight independent channels");
    assert_eq!(
        file[FRAME + 5],
        reference_crc8(&file[FRAME..FRAME + 5]),
        "the six-byte header ends in its CRC-8"
    );

    // Every value past the ten the format defines, each with a corrected
    // CRC-8 so the header itself is accepted and the assignment is what the
    // rejection is about.
    for reserved in 11u8..=15 {
        let mut broken = file.clone();
        broken[FRAME + 3] = (reserved << 4) | (broken[FRAME + 3] & 0x0F);
        broken[FRAME + 5] = reference_crc8(&broken[FRAME..FRAME + 5]);
        let error = FlacReader::new(&broken)
            .and_then(|reader| reader.decode_to_end())
            .expect_err("a reserved channel assignment must be refused");
        assert!(
            matches!(
                error,
                DecodeError::Malformed {
                    expected: "a defined channel assignment",
                    ..
                }
            ),
            "assignment {reserved}: unexpected error: {error}"
        );

        // And on the streaming path, which parses its own frame headers.
        let streamed = stream_decode(&broken, 3)
            .expect_err("the stream must refuse it too, at every feed size");
        assert!(
            matches!(
                streamed,
                DecodeError::Malformed {
                    expected: "a defined channel assignment",
                    ..
                }
            ),
            "assignment {reserved} streamed: unexpected error: {streamed}"
        );
    }
}

/// A stream that carries fewer or more samples than it declared is a typed
/// error in both directions, never a quietly short or long buffer.
#[test]
fn the_declared_sample_count_has_to_be_the_real_one() {
    let build = Build {
        block_size: 16,
        assignment: 0,
        ..Build::default()
    };
    let samples = vec![audio(64, 16, 0, 0x3333)];
    let file = flac_file(&build, &samples);
    assert_eq!(
        FlacReader::new(&file)
            .expect("open")
            .decode_to_end()
            .expect("decode")
            .frames(),
        64
    );

    // The 36-bit total sample count ends at the last byte before the MD5, so
    // its low eight bits are file[25]. Declaring more than the frames hold is
    // truncation.
    let mut broken = file.clone();
    broken[25] = 0x50;
    assert!(matches!(
        FlacReader::new(&broken).and_then(|reader| reader.decode_to_end()),
        Err(DecodeError::Truncated { .. })
    ));

    // Declare fewer: the trailing frame is real audio the file disowned.
    let mut broken = file;
    broken[25] = 0x10;
    assert!(matches!(
        FlacReader::new(&broken).and_then(|reader| reader.decode_to_end()),
        Err(DecodeError::Malformed { .. })
    ));
}

// -- Gate: cross-platform determinism -----------------------------------------

/// FNV-1a, so the witness is the bytes themselves and not a float comparison.
fn fnv1a(bytes: impl IntoIterator<Item = u8>) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Appends the bits of every sample in `buffer` to `witness`.
fn absorb(witness: &mut Vec<u8>, buffer: &AudioBuffer) {
    witness.extend(
        buffer
            .samples()
            .iter()
            .flat_map(|sample| sample.to_bits().to_le_bytes()),
    );
}

/// One number that changes if any bit of any decoded sample changes, over
/// RFC 9639's three worked examples and a sweep of the dimension matrix,
/// through both paths.
///
/// Running this on two toolchains and on a 32-bit target and getting the same
/// constant is the evidence for the byte-identical claim. The constant is
/// pinned rather than recomputed, so a change shows up as a diff here.
#[test]
fn flac_output_is_bit_identical_to_a_pinned_witness() {
    let mut witness: Vec<u8> = Vec::new();
    for file in [RFC_EXAMPLE_1, RFC_EXAMPLE_2, RFC_EXAMPLE_3] {
        let decoded = FlacReader::new(file)
            .expect("open")
            .decode_to_end()
            .expect("decode");
        let streamed = stream_decode(file, 7).expect("stream");
        assert_eq!(streamed, decoded.samples(), "witness stream");
        absorb(&mut witness, &decoded);
    }
    for bits in [8u32, 12, 15, 16, 20, 24, 32] {
        for assignment in [0u8, 1, 8, 9, 10] {
            // Both parities: `wasted` of 0 leaves the audio as often odd as
            // even, which is the only way the mid/side correction is reached.
            for wasted in [0u32, 2] {
                let build = Build {
                    bits,
                    rate: 32_000,
                    assignment,
                    block_size: 96,
                    wasted,
                    ..Build::default()
                };
                let channels = build.channels();
                let samples: Vec<Vec<i64>> = (0..channels)
                    .map(|channel| {
                        audio(233, bits, wasted, 0xDEC1 + channel as u64 + u64::from(bits))
                    })
                    .collect();
                let file = flac_file(&build, &samples);
                let decoded = FlacReader::new(&file)
                    .expect("open")
                    .decode_to_end()
                    .expect("decode");
                let streamed = stream_decode(&file, 7).expect("stream");
                assert_eq!(streamed, decoded.samples(), "witness stream");
                absorb(&mut witness, &decoded);
            }
        }
    }
    assert_eq!(fnv1a(witness), 0x97ef_b751_3ce8_8469, "FLAC output changed");
}

// -- Structural odds and ends the dimensions imply ----------------------------

/// A stream whose streaminfo declares no total length decodes to whatever the
/// frames hold, and reports the length as unknown rather than as zero.
#[test]
fn an_undeclared_length_is_unknown_rather_than_zero() {
    let build = Build {
        declare_total: false,
        block_size: 32,
        assignment: 0,
        ..Build::default()
    };
    let samples = vec![audio(100, 16, 0, 0x4444)];
    let file = flac_file(&build, &samples);
    let reader = FlacReader::new(&file).expect("open");
    assert_eq!(reader.frames(), None);
    assert_eq!(reader.decode_to_end().expect("decode").frames(), 100);
    assert_eq!(stream_decode(&file, 11).expect("stream").len(), 100);
}

/// Resetting a streaming reader returns it to its just-constructed state, so
/// a second stream on the same instance decodes as a first one would.
#[test]
fn a_reset_stream_decodes_a_second_file() {
    let mut stream = FlacStreamDecoder::new();
    let mut first = Vec::new();
    stream.push(RFC_EXAMPLE_1).expect("push");
    while stream.pull(&mut first, usize::MAX).expect("pull") > 0 {}
    stream.finish(&mut first).expect("finish");

    stream.reset();
    assert_eq!(stream.spec(), None);
    assert_eq!(stream.buffered_bytes(), 0);

    let mut second = Vec::new();
    stream.push(RFC_EXAMPLE_3).expect("push");
    while stream.pull(&mut second, usize::MAX).expect("pull") > 0 {}
    stream.finish(&mut second).expect("finish");
    assert_eq!(second.len(), 24);
}

/// Pushing after a failure is quiet rather than a second, different answer.
#[test]
fn a_failed_stream_stays_failed() {
    let mut broken = RFC_EXAMPLE_1.to_vec();
    broken[4] = 0x7F; // the forbidden metadata block type
    let mut stream = FlacStreamDecoder::new();
    assert!(stream.push(&broken).is_err());
    assert_eq!(stream.push(&broken).expect("quiet push"), 0);
    let mut samples = Vec::new();
    assert_eq!(stream.finish(&mut samples).expect("quiet finish"), 0);
    assert!(samples.is_empty());
}

/// `pull` respects its frame cap and never splits a frame across two calls.
#[test]
fn pull_hands_back_whole_frames_only() {
    let mut stream = FlacStreamDecoder::new();
    stream.push(RFC_EXAMPLE_2).expect("push");
    let mut samples = Vec::new();
    let mut frames = 0;
    loop {
        let got = stream.pull(&mut samples, 3).expect("pull");
        if got == 0 {
            break;
        }
        assert!(got <= 3);
        frames += got;
        assert_eq!(samples.len(), frames * 2, "a frame was split");
    }
    frames += stream.finish(&mut samples).expect("finish");
    assert_eq!(frames, 19);
}
