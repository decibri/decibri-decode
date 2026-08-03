#![forbid(unsafe_code)]
//! WAV conformance: the dimensions a WAV file varies along, crossed.
//!
//! # Why this file is arranged as a cross product
//!
//! Two negative controls in two consecutive steps of this crate found the same
//! class of gap. Step 2 broke the 24-bit little-endian sign extension and the
//! exhaustive 16,777,216-value round trip did not notice, because that gate
//! varies *value* and never touches the byte path. Step 3 broke `feed` so it
//! over-reported consumption and the ten-chunk-size byte-path gate did not
//! notice, because that gate varies *chunk size* while every input was 256
//! bytes and the limit it would have breached was 65,536.
//!
//! The generalisation is that **a test's coverage is bounded by the dimensions
//! its inputs vary along, not by how many inputs it uses**. Both times the
//! most impressive-sounding gate had the blind spot.
//!
//! So the dimensions are enumerated first and the gates are built to cross
//! them, rather than each gate sweeping one dimension with the rest held at a
//! convenient constant:
//!
//! 1. format tag: 1, 3, 6, 7, 0xFFFE
//! 2. bits per sample: 8, 16, 24, 32, 64
//! 3. channel count: 1, 2, and more than 2
//! 4. chunk ordering: `fmt ` before `data`, other chunks between them, chunks
//!    after `data`
//! 5. unknown chunks: before, between and after the known ones
//! 6. chunk length parity: even, and odd with its pad byte
//! 7. total input length: small, and past every internal buffer limit
//! 8. declared sizes against actual: matching, under-declared, over-declared
//! 9. `data` length against frame size: whole frames, and a partial one
//! 10. feed chunk size, on the streaming path
//! 11. RF64
//!
//! [`the_dimension_matrix`] crosses 1 through 8 and 11 in one product and
//! rotates 10 through it; [`a_partial_trailing_frame_is_truncated`] is 9;
//! [`the_streaming_path_matches_the_whole_file_path`] crosses 10 with 7
//! explicitly, because holding total length constant while varying chunk size
//! is exactly what step 3's control found.
//!
//! # References are built here, not taken from the writer
//!
//! Every file in this suite is assembled byte by byte from the format
//! definitions. A reader and a writer that share a misunderstanding agree with
//! each other perfectly, so the writer is gated *against these files* rather
//! than being the thing they are built with.

use decibri_decode::{
    AudioSpec, DecodeError, RiffFlavour, StreamSource, WavCodec, WavHeaderStyle, WavReader,
    WavStreamDecoder, WavWriter,
};

// -- The reference builder ----------------------------------------------------

/// A chunk: identifier, declared size, body, and the RIFF pad byte when the
/// body is odd.
fn chunk(id: &[u8; 4], body: &[u8]) -> Vec<u8> {
    chunk_declaring(id, body.len() as u32, body)
}

/// A chunk whose size field says `declared` whatever the body actually is, for
/// the over- and under-declared cases.
fn chunk_declaring(id: &[u8; 4], declared: u32, body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(body.len() + 9);
    out.extend_from_slice(id);
    out.extend_from_slice(&declared.to_le_bytes());
    out.extend_from_slice(body);
    if body.len() % 2 == 1 {
        out.push(0);
    }
    out
}

/// The same, without the pad byte an odd body is supposed to carry.
fn chunk_unpadded(id: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(body.len() + 8);
    out.extend_from_slice(id);
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(body);
    out
}

/// How truthful the RIFF header's size field is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SizeClaim {
    Truthful,
    Under,
    Over,
}

impl SizeClaim {
    fn field(self, truthful: u32) -> u32 {
        match self {
            Self::Truthful => truthful,
            Self::Under => 4,
            Self::Over => u32::MAX,
        }
    }
}

/// A whole file: magic, size field, form type, then the chunks in order.
fn file(magic: &[u8; 4], claim: SizeClaim, chunks: &[Vec<u8>]) -> Vec<u8> {
    let body: Vec<u8> = chunks.concat();
    let mut out = Vec::with_capacity(body.len() + 12);
    out.extend_from_slice(magic);
    out.extend_from_slice(&claim.field(4 + body.len() as u32).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(&body);
    out
}

/// A sixteen-byte `fmt ` body, written from the `WAVEFORMATEX` field order.
fn fmt_plain(tag: u16, channels: u16, rate: u32, bits: u16) -> Vec<u8> {
    let block_align = channels * bits.div_ceil(8);
    let mut body = Vec::with_capacity(16);
    body.extend_from_slice(&tag.to_le_bytes());
    body.extend_from_slice(&channels.to_le_bytes());
    body.extend_from_slice(&rate.to_le_bytes());
    body.extend_from_slice(&(rate * u32::from(block_align)).to_le_bytes());
    body.extend_from_slice(&block_align.to_le_bytes());
    body.extend_from_slice(&bits.to_le_bytes());
    body
}

/// The fourteen bytes every `KSDATAFORMAT_SUBTYPE_*` GUID ends with, written
/// out from the published `KSDATAFORMAT_SUBTYPE_PCM` value
/// `00000001-0000-0010-8000-00aa00389b71`.
const SUBFORMAT_TAIL: [u8; 14] = [
    0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
];

/// A forty-byte `WAVE_FORMAT_EXTENSIBLE` `fmt ` body naming `sub_tag`, with
/// `cb_size` written as given so an inconsistent one can be built on purpose.
fn fmt_extensible(channels: u16, rate: u32, bits: u16, sub_tag: u16, cb_size: u16) -> Vec<u8> {
    let mut body = fmt_plain(0xFFFE, channels, rate, bits);
    body.extend_from_slice(&cb_size.to_le_bytes());
    body.extend_from_slice(&bits.to_le_bytes()); // wValidBitsPerSample
    body.extend_from_slice(&0u32.to_le_bytes()); // dwChannelMask
    body.extend_from_slice(&sub_tag.to_le_bytes());
    body.extend_from_slice(&SUBFORMAT_TAIL);
    body
}

/// A twenty-eight-byte `ds64` body, per EBU Tech 3306.
fn ds64_body(riff_size: u64, data_size: u64, frames: u64) -> Vec<u8> {
    let mut body = Vec::with_capacity(28);
    body.extend_from_slice(&riff_size.to_le_bytes());
    body.extend_from_slice(&data_size.to_le_bytes());
    body.extend_from_slice(&frames.to_le_bytes());
    body.extend_from_slice(&0u32.to_le_bytes()); // no override table
    body
}

/// The conventional file: `fmt ` then `data`, plain `RIFF`, truthful sizes.
fn wav(tag: u16, channels: u16, rate: u32, bits: u16, payload: &[u8]) -> Vec<u8> {
    file(
        b"RIFF",
        SizeClaim::Truthful,
        &[
            chunk(b"fmt ", &fmt_plain(tag, channels, rate, bits)),
            chunk(b"data", payload),
        ],
    )
}

// -- Deterministic test signals -----------------------------------------------

/// Payload bytes from an LCG, so the same bytes on every target and every run.
///
/// `avoid` is for mu-law's second zero: `0x7F` decodes to silence like its twin
/// `0xFF` and re-encodes to `0xFF`, so it is the one byte in either law that is
/// not a fixed point of a write-read-write cycle.
fn payload_bytes(length: usize, seed: u64, avoid: Option<u8>) -> Vec<u8> {
    let mut state = seed | 1;
    (0..length)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let byte = (state >> 33) as u8;
            match avoid {
                Some(banned) if byte == banned => banned.wrapping_add(1),
                _ => byte,
            }
        })
        .collect()
}

/// A payload of `frames` whole frames for `codec` at `channels`.
fn payload_for(codec: WavCodec, channels: u16, frames: usize, seed: u64) -> Vec<u8> {
    let avoid = (codec == WavCodec::MuLaw).then_some(0x7F);
    payload_bytes(
        frames * usize::from(channels) * codec.bytes_per_sample(),
        seed,
        avoid,
    )
}

/// What the payload decodes to, through the codec layer and not through the
/// container. The container gates below check the container.
fn expected_samples(codec: WavCodec, payload: &[u8]) -> Vec<f32> {
    let mut samples = Vec::new();
    codec.decode(payload, &mut samples);
    samples
}

/// Compares samples by bit pattern rather than by `==`.
///
/// Two reasons, in opposite directions. `f32` equality says a NaN differs from
/// itself, and arbitrary bytes read as `f32` are NaN about once every 256
/// samples: a 32-bit float involves no narrowing, so the crate passes the bit
/// pattern the file carried straight through, and an `==` comparison would
/// report that documented behaviour as a failure. And `+0.0 == -0.0` is true
/// while the two are different bytes, so `==` would miss a sign flip on
/// silence. Comparing bits is the right test for a crate claiming bit-exact
/// output.
///
/// (Signalling NaNs were checked for an ABI hazard before this was written:
/// `i686-pc-windows-msvc` returns `f32` in an SSE register rather than in
/// `ST(0)`, so the quiet bit is not set on the way through and the bit patterns
/// hold on both targets.)
#[track_caller]
fn assert_same_samples(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: sample count");
    for (index, (got, want)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "{label}: sample {index} came back as {got} rather than {want}"
        );
    }
}

/// Every encoding, and the `(tag, bits)` pair a file names it with.
const CODECS: [(WavCodec, u16, u16); 8] = [
    (WavCodec::PcmU8, 1, 8),
    (WavCodec::PcmI16, 1, 16),
    (WavCodec::PcmI24, 1, 24),
    (WavCodec::PcmI32, 1, 32),
    (WavCodec::Float32, 3, 32),
    (WavCodec::Float64, 3, 64),
    (WavCodec::ALaw, 6, 8),
    (WavCodec::MuLaw, 7, 8),
];

// -- An independent reading of ITU-T G.711 ------------------------------------
//
// The crate derives both tables from a bit position, because the segment
// boundaries happen to be powers of two. This carries the same geometry as
// literal data read off the recommendation's tables, so a shift written one
// place out in the crate is not mirrored here.

/// A-law segment geometry in the 13-bit domain: each segment's first magnitude
/// and the width of one of its sixteen intervals. The first two segments share
/// a step of 2, which is the ITU table and not a typo.
const A_LAW_SEGMENTS: [(i32, i32); 8] = [
    (0, 2),
    (32, 2),
    (64, 4),
    (128, 8),
    (256, 16),
    (512, 32),
    (1024, 64),
    (2048, 128),
];

/// mu-law segment geometry in the *biased* 14-bit domain: the recommendation
/// adds 33 before segmenting, which is why the first segment starts at 32.
const MU_LAW_SEGMENTS: [(i32, i32); 8] = [
    (32, 2),
    (64, 4),
    (128, 8),
    (256, 16),
    (512, 32),
    (1024, 64),
    (2048, 128),
    (4096, 256),
];

/// The 16-bit linear sample the ITU table gives for `code`.
fn reference_g711(mu_law: bool, code: u8) -> i16 {
    let (segments, value) = if mu_law {
        // mu-law transmits the complement of the table word.
        (MU_LAW_SEGMENTS, !code)
    } else {
        // A-law transmits the table word with its even bits inverted.
        (A_LAW_SEGMENTS, code ^ 0x55)
    };
    let (start, step) = segments[usize::from((value >> 4) & 0x07)];
    let interval = i32::from(value & 0x0F);
    let midpoint = start + step * interval + step / 2;
    if mu_law {
        // 14-bit, biased, scaled by 4. The sign bit of the complemented word is
        // set for negative.
        let magnitude = (midpoint - 33) * 4;
        if value & 0x80 != 0 {
            -magnitude as i16
        } else {
            magnitude as i16
        }
    } else {
        // 13-bit, unbiased, scaled by 8. The sign bit is set for *positive*,
        // the opposite way round from mu-law.
        let magnitude = midpoint * 8;
        if value & 0x80 == 0 {
            -magnitude as i16
        } else {
            magnitude as i16
        }
    }
}

/// Sign extension written as a comparison and a subtraction rather than as the
/// shift the crate uses, so the two are not the same expression twice.
fn reference_i24(bytes: [u8; 3]) -> i32 {
    let raw = i32::from(bytes[0]) | (i32::from(bytes[1]) << 8) | (i32::from(bytes[2]) << 16);
    if raw >= 0x0080_0000 {
        raw - 0x0100_0000
    } else {
        raw
    }
}

// -- SHA-256, for the round-trip gate -----------------------------------------
//
// The crate takes one dependency and this is not it. Sixty lines of a published
// algorithm with published test vectors is cheaper than a dependency, and the
// vectors below are what make it trustworthy.

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

fn sha256(message: &[u8]) -> String {
    let mut hash: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut padded = message.to_vec();
    let bit_length = (message.len() as u64) * 8;
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_length.to_be_bytes());

    for block in padded.chunks_exact(64) {
        let mut schedule = [0u32; 64];
        for (slot, word) in schedule.iter_mut().zip(block.chunks_exact(4)) {
            *slot = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for index in 16..64 {
            let a = schedule[index - 15];
            let b = schedule[index - 2];
            let s0 = a.rotate_right(7) ^ a.rotate_right(18) ^ (a >> 3);
            let s1 = b.rotate_right(17) ^ b.rotate_right(19) ^ (b >> 10);
            schedule[index] = schedule[index - 16]
                .wrapping_add(s0)
                .wrapping_add(schedule[index - 7])
                .wrapping_add(s1);
        }

        let mut w = hash;
        for (&constant, &word) in SHA256_K.iter().zip(schedule.iter()) {
            let s1 = w[4].rotate_right(6) ^ w[4].rotate_right(11) ^ w[4].rotate_right(25);
            let choose = (w[4] & w[5]) ^ ((!w[4]) & w[6]);
            let temp1 = w[7]
                .wrapping_add(s1)
                .wrapping_add(choose)
                .wrapping_add(constant)
                .wrapping_add(word);
            let s0 = w[0].rotate_right(2) ^ w[0].rotate_right(13) ^ w[0].rotate_right(22);
            let majority = (w[0] & w[1]) ^ (w[0] & w[2]) ^ (w[1] & w[2]);
            let temp2 = s0.wrapping_add(majority);
            w = [
                temp1.wrapping_add(temp2),
                w[0],
                w[1],
                w[2],
                w[3].wrapping_add(temp1),
                w[4],
                w[5],
                w[6],
            ];
        }
        for (slot, added) in hash.iter_mut().zip(w) {
            *slot = slot.wrapping_add(added);
        }
    }

    let mut text = String::with_capacity(64);
    const HEX: [u8; 16] = *b"0123456789abcdef";
    for word in hash {
        for byte in word.to_be_bytes() {
            text.push(HEX[usize::from(byte >> 4)] as char);
            text.push(HEX[usize::from(byte & 0x0F)] as char);
        }
    }
    text
}

/// The hash is only worth what its own gate is worth. These are the published
/// FIPS 180-4 vectors, plus one that crosses the 64-byte block boundary and one
/// that lands exactly on the padding edge.
#[test]
fn the_sha256_used_by_the_round_trip_gate_matches_the_published_vectors() {
    assert_eq!(
        sha256(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        sha256(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
    // One million 'a', the third published vector. 15,625 blocks, so this one
    // is about the block loop rather than about the round constants.
    assert_eq!(
        sha256(&vec![b'a'; 1_000_000]),
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
    );
}

// -- Driving the streaming reader ---------------------------------------------

/// How a caller drives [`WavStreamDecoder`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Drive {
    /// Push a piece, pull everything, repeat. What a network reader does.
    Interleaved,
    /// Push as much as the reader will take before pulling anything, so the
    /// short return is on the path rather than merely permitted.
    ///
    /// This is the mode step 3's Control B says has to exist: a reader that
    /// over-reports consumption loses every byte past its internal limit, and
    /// nothing driven interleaved at a small total ever notices.
    Greedy,
}

fn stream_decode(
    bytes: &[u8],
    piece: usize,
    drive: Drive,
) -> Result<(Option<AudioSpec>, Vec<f32>), DecodeError> {
    let mut stream = WavStreamDecoder::new();
    let mut samples = Vec::new();
    for slice in bytes.chunks(piece.max(1)) {
        let mut offset = 0;
        while offset < slice.len() {
            let taken = stream.push(&slice[offset..])?;
            offset += taken;
            if taken == 0 || drive == Drive::Interleaved {
                while stream.pull(&mut samples, usize::MAX)? > 0 {}
            }
        }
    }
    let spec = stream.spec();
    stream.finish(&mut samples)?;
    Ok((spec, samples))
}

// -- Gate 5: the six cases the step-0 audit named -----------------------------

/// Case 1: mu-law, `wFormatTag` 7, 8 kHz mono.
///
/// The payload is all 256 codes, and the expected samples come from the literal
/// ITU segment tables above rather than from this crate's own tables.
#[test]
fn case_1_mu_law_tag_7_decodes_to_the_itu_table() {
    let payload: Vec<u8> = (0..=u8::MAX).collect();
    let bytes = wav(7, 1, 8_000, 8, &payload);
    let reader = WavReader::new(&bytes).expect("mu-law must be handled");

    assert_eq!(reader.format().codec, WavCodec::MuLaw);
    assert_eq!(reader.spec(), AudioSpec::mono(8_000));
    assert_eq!(reader.frames(), 256);

    let decoded = reader.decode_to_end();
    let expected: Vec<f32> = (0..=u8::MAX)
        .map(|code| f32::from(reference_g711(true, code)) / 32768.0)
        .collect();
    assert_same_samples(decoded.samples(), &expected, "mu-law tag 7");

    // Published anchors, as literals, so a shared misreading between the table
    // and the reference above would still fail here.
    assert_eq!(decoded.samples()[0xFF], 0.0);
    assert_eq!(decoded.samples()[0x00], -32_124.0 / 32768.0);
    assert_eq!(decoded.samples()[0x80], 32_124.0 / 32768.0);
}

/// Case 2: A-law, `wFormatTag` 6, 8 kHz mono.
#[test]
fn case_2_a_law_tag_6_decodes_to_the_itu_table() {
    let payload: Vec<u8> = (0..=u8::MAX).collect();
    let bytes = wav(6, 1, 8_000, 8, &payload);
    let reader = WavReader::new(&bytes).expect("A-law must be handled");

    assert_eq!(reader.format().codec, WavCodec::ALaw);
    assert_eq!(reader.frames(), 256);

    let decoded = reader.decode_to_end();
    let expected: Vec<f32> = (0..=u8::MAX)
        .map(|code| f32::from(reference_g711(false, code)) / 32768.0)
        .collect();
    assert_same_samples(decoded.samples(), &expected, "A-law tag 6");

    assert_eq!(decoded.samples()[0xD5], 8.0 / 32768.0);
    assert_eq!(decoded.samples()[0x55], -8.0 / 32768.0);
    assert_eq!(decoded.samples()[0xAA], 32_256.0 / 32768.0);
    // And the two laws are not the same file read twice: the same byte decodes
    // to different audio under each.
    assert_ne!(decoded.samples()[0x00], -32_124.0 / 32768.0);
}

/// Case 3: 24-bit PCM, `wFormatTag` 1 at 24 bits.
///
/// The case most likely to go silently wrong, because the tag says PCM and only
/// the width differs. Dispatch is on the pair, so it does not.
#[test]
fn case_3_pcm_24_bit_decodes_at_its_own_stride() {
    // Hand-written triples covering the sign-extension boundary: the classic
    // failure of this format is reading 0x800000 as +8388608 rather than as the
    // most negative value.
    let triples: [[u8; 3]; 8] = [
        [0x00, 0x00, 0x00],
        [0x00, 0x00, 0x80],
        [0xFF, 0xFF, 0x7F],
        [0xFF, 0xFF, 0xFF],
        [0x01, 0x00, 0x00],
        [0x56, 0x34, 0x12],
        [0xAA, 0xBB, 0xCC],
        [0x00, 0x00, 0x40],
    ];
    let payload: Vec<u8> = triples.iter().flatten().copied().collect();
    let bytes = wav(1, 1, 16_000, 24, &payload);
    let reader = WavReader::new(&bytes).expect("24-bit PCM must be handled");

    assert_eq!(reader.format().codec, WavCodec::PcmI24);
    assert_eq!(reader.format().bits_per_sample, 24);
    assert_eq!(reader.frames(), 8);

    let decoded = reader.decode_to_end();
    let expected: Vec<f32> = triples
        .iter()
        .map(|triple| reference_i24(*triple) as f32 / 8_388_608.0)
        .collect();
    assert_same_samples(decoded.samples(), &expected, "24-bit PCM");

    // Written out, because these two are the whole point of the case.
    assert_eq!(decoded.samples()[1], -1.0, "0x800000 is the minimum");
    assert_eq!(
        decoded.samples()[2],
        8_388_607.0 / 8_388_608.0,
        "0x7FFFFF is the maximum"
    );
    assert_eq!(decoded.samples()[3], -1.0 / 8_388_608.0, "0xFFFFFF is -1");
}

/// Case 4: 32-bit float, `wFormatTag` 3 at 32 bits.
///
/// The expected values are IEEE 754 binary32 bit patterns written as literals,
/// so the assertion is against the standard rather than against `from_le_bytes`.
#[test]
fn case_4_float_32_decodes_from_ieee_754_bit_patterns() {
    let cases: [([u8; 4], f32); 6] = [
        ([0x00, 0x00, 0x00, 0x00], 0.0),
        ([0x00, 0x00, 0x80, 0x3F], 1.0),
        ([0x00, 0x00, 0x80, 0xBF], -1.0),
        ([0x00, 0x00, 0x00, 0x3F], 0.5),
        ([0x00, 0x00, 0x00, 0xBE], -0.125),
        // 1.5, which is past full scale and must arrive unclamped: a float
        // consumer gets what the chain produced.
        ([0x00, 0x00, 0xC0, 0x3F], 1.5),
    ];
    let payload: Vec<u8> = cases.iter().flat_map(|(bytes, _)| *bytes).collect();
    let bytes = wav(3, 1, 16_000, 32, &payload);
    let reader = WavReader::new(&bytes).expect("32-bit float must be handled");

    assert_eq!(reader.format().codec, WavCodec::Float32);
    assert_eq!(reader.frames(), 6);
    let decoded = reader.decode_to_end();
    let expected: Vec<f32> = cases.iter().map(|(_, value)| *value).collect();
    assert_same_samples(decoded.samples(), &expected, "32-bit float");
}

/// The clamp boundary, stated as bytes: what an integer target and a float
/// target each write for a sample that is not finite.
///
/// This is the statement [`WavWriter`]'s documentation makes, so it is
/// measured rather than asserted. The integer target clamps, taking NaN to
/// silence and the infinities to the extremes; the float targets write the
/// IEEE 754 bit patterns through. Every file below reads back.
#[test]
fn a_non_finite_sample_clamps_into_an_integer_target_and_passes_into_a_float_one() {
    let samples: [f32; 3] = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
    let spec = AudioSpec::mono(48_000);

    // The integer target: silence, then both extremes.
    let integer = WavWriter::new(spec, WavCodec::PcmI16)
        .to_bytes(&samples)
        .expect("write i16");
    let reader = WavReader::new(&integer).expect("read i16");
    assert_eq!(reader.data(), [0x00, 0x00, 0xFF, 0x7F, 0x00, 0x80]);
    assert_same_samples(
        reader.decode_to_end().samples(),
        &[0.0, 32_767.0 / 32_768.0, -1.0],
        "i16 target",
    );

    // The 32-bit float target: the three bit patterns, little-endian, and
    // every one of them survives the round trip exactly.
    let float32 = WavWriter::new(spec, WavCodec::Float32)
        .to_bytes(&samples)
        .expect("write f32");
    let reader = WavReader::new(&float32).expect("read f32");
    assert_eq!(
        reader.data(),
        [0x00, 0x00, 0xC0, 0x7F, 0x00, 0x00, 0x80, 0x7F, 0x00, 0x00, 0x80, 0xFF]
    );
    let back = reader.decode_to_end();
    assert_eq!(back.samples()[0].to_bits(), f32::NAN.to_bits());
    assert_eq!(back.samples()[1], f32::INFINITY);
    assert_eq!(back.samples()[2], f32::NEG_INFINITY);

    // The 64-bit float target: the bit patterns widen and are written whole.
    // The NaN comes back as silence, because narrowing a NaN from `f64` to
    // `f32` normalises it; the infinities come back unchanged.
    let float64 = WavWriter::new(spec, WavCodec::Float64)
        .to_bytes(&samples)
        .expect("write f64");
    let reader = WavReader::new(&float64).expect("read f64");
    assert_eq!(
        reader.data(),
        [
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF8, 0x7F, // NaN
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x7F, // +inf
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0xFF, // -inf
        ]
    );
    let back = reader.decode_to_end();
    assert_eq!(back.samples()[0].to_bits(), 0.0f32.to_bits());
    assert_eq!(back.samples()[1], f32::INFINITY);
    assert_eq!(back.samples()[2], f32::NEG_INFINITY);
}

/// Case 5: `WAVE_FORMAT_EXTENSIBLE` 0xFFFE with a PCM SubFormat GUID.
///
/// The payload is byte-for-byte the same as the plain 16-bit control's, so the
/// assertion is that the GUID resolution changed nothing about the decode.
#[test]
fn case_5_extensible_with_a_pcm_subformat_decodes_as_the_plain_file_does() {
    let payload = payload_bytes(512, 0x5150, None);
    let control = wav(1, 2, 16_000, 16, &payload);
    let extensible = file(
        b"RIFF",
        SizeClaim::Truthful,
        &[
            chunk(b"fmt ", &fmt_extensible(2, 16_000, 16, 1, 22)),
            chunk(b"data", &payload),
        ],
    );

    let plain = WavReader::new(&control).expect("control");
    let reader = WavReader::new(&extensible).expect("extensible must be handled");

    assert_eq!(reader.format().format_tag, 0xFFFE, "the file said 0xFFFE");
    assert!(reader.format().is_extensible());
    assert_eq!(reader.format().codec, WavCodec::PcmI16, "and meant PCM");
    assert_eq!(reader.spec(), plain.spec());
    assert_eq!(reader.frames(), plain.frames());

    // Against an independent reading of the payload, not only against the
    // control: a shared mistake in both readers would still fail here.
    let expected: Vec<f32> = payload
        .chunks_exact(2)
        .map(|pair| f32::from(i16::from_le_bytes([pair[0], pair[1]])) / 32768.0)
        .collect();
    assert_same_samples(
        reader.decode_to_end().samples(),
        &expected,
        "extensible PCM",
    );
    assert_same_samples(
        reader.decode_to_end().samples(),
        plain.decode_to_end().samples(),
        "extensible against the plain control",
    );

    // Every tag the GUID can name resolves the same way.
    for (codec, tag, bits) in CODECS {
        let payload = payload_for(codec, 1, 4, 0x901);
        let bytes = file(
            b"RIFF",
            SizeClaim::Truthful,
            &[
                chunk(b"fmt ", &fmt_extensible(1, 8_000, bits, tag, 22)),
                chunk(b"data", &payload),
            ],
        );
        let reader = WavReader::new(&bytes).expect("every subformat resolves");
        assert_eq!(reader.format().codec, codec);
        assert_same_samples(
            reader.decode_to_end().samples(),
            &expected_samples(codec, &payload),
            &format!("extensible naming tag {tag}"),
        );
    }
}

/// Case 6: an odd-length chunk before `fmt `, with the RIFF pad byte.
///
/// A walk that forgets the pad lands one byte short of the next chunk header
/// and reads four bytes of garbage as an identifier, so this case is decided
/// entirely by one `& 1`.
#[test]
fn case_6_an_odd_length_chunk_is_followed_by_its_pad_byte() {
    let payload = payload_bytes(256, 0x60D, None);
    let control = wav(1, 1, 16_000, 16, &payload);
    let odd = file(
        b"RIFF",
        SizeClaim::Truthful,
        &[
            // The five-byte body the step-0 audit used, which is what makes
            // this chunk odd.
            chunk(b"LIST", b"INFOx"),
            chunk(b"fmt ", &fmt_plain(1, 1, 16_000, 16)),
            chunk(b"data", &payload),
        ],
    );

    let reader = WavReader::new(&odd).expect("the pad byte must be applied");
    let expected: Vec<f32> = payload
        .chunks_exact(2)
        .map(|pair| f32::from(i16::from_le_bytes([pair[0], pair[1]])) / 32768.0)
        .collect();
    assert_same_samples(
        reader.decode_to_end().samples(),
        &expected,
        "padded odd chunk",
    );
    assert_same_samples(
        reader.decode_to_end().samples(),
        WavReader::new(&control)
            .expect("control")
            .decode_to_end()
            .samples(),
        "padded odd chunk against the control",
    );

    // The same file without its pad byte is not the same file: the walk lands
    // inside the `fmt ` header. It must fail rather than decode something.
    let unpadded = file(
        b"RIFF",
        SizeClaim::Truthful,
        &[
            chunk_unpadded(b"LIST", b"INFOx"),
            chunk(b"fmt ", &fmt_plain(1, 1, 16_000, 16)),
            chunk(b"data", &payload),
        ],
    );
    match WavReader::new(&unpadded) {
        Err(_) => {}
        Ok(reader) => assert_ne!(
            reader.decode_to_end().samples().len(),
            expected.len(),
            "a walk that ignored the pad byte agreed with one that applied it"
        ),
    }

    // And the pad byte at the end of the file: an odd `data` chunk. One frame
    // of mono 8-bit is one byte, so three frames is an odd payload.
    let odd_data = wav(1, 1, 8_000, 8, &[0x00, 0x80, 0xFF]);
    let reader = WavReader::new(&odd_data).expect("an odd data chunk");
    assert_eq!(reader.frames(), 3);
    assert_same_samples(
        reader.decode_to_end().samples(),
        &[-1.0, 0.0, 127.0 / 128.0],
        "an odd data chunk",
    );
}

// -- Gate 6: round-trip identity by SHA-256 -----------------------------------

/// Write, read, write: the bytes are a fixed point and the samples are a fixed
/// point.
///
/// The first pass is not always the identity and the gate does not pretend it
/// is. `i32` carries 31 significant bits and `f64` carries 53, and the crate's
/// internal `f32` carries 24, so those two land on the nearest representable
/// value on the way in. Everything at or below 24 bits (`u8`, 16-bit, 24-bit,
/// `f32` and both G.711 laws) survives the first pass exactly as well, and
/// that stronger claim is asserted for them separately below.
#[test]
fn every_written_format_round_trips_to_an_identical_sha256() {
    for (codec, tag, bits) in CODECS {
        for channels in [1u16, 2, 6] {
            for frames in [0usize, 1, 3, 1000] {
                let payload = payload_for(codec, channels, frames, 0x901 + frames as u64);
                let source = wav(tag, channels, 16_000, bits, &payload);
                let label = format!("{codec:?} at {channels} channels, {frames} frames");

                let first = WavReader::new(&source)
                    .unwrap_or_else(|error| panic!("{label}: {error}"))
                    .decode_to_end();
                let writer = WavWriter::new(AudioSpec::new(16_000, channels), codec);
                let written = writer.to_bytes(first.samples()).expect("write");

                let second = WavReader::new(&written).expect("read back").decode_to_end();
                let rewritten = writer.to_bytes(second.samples()).expect("rewrite");

                assert_same_samples(
                    second.samples(),
                    first.samples(),
                    &format!("{label}: samples"),
                );
                assert_eq!(second.spec(), first.spec(), "{label}: spec moved");
                assert_eq!(
                    sha256(&written),
                    sha256(&rewritten),
                    "{label}: the file is not a fixed point"
                );

                // The formats that fit inside an f32 significand are exact on
                // the first pass too, so the payload the writer produced is the
                // payload it was given.
                let lossless = !matches!(codec, WavCodec::PcmI32 | WavCodec::Float64);
                if lossless {
                    let reader = WavReader::new(&written).expect("read back");
                    assert_eq!(reader.data(), payload.as_slice(), "{label}: payload moved");
                }
            }
        }
    }
}

/// The same, for the header styles and RIFF flavours the writer offers. Each
/// combination has to be a fixed point on its own, and each has to read back to
/// the same audio as the others.
#[test]
fn every_header_style_and_flavour_round_trips_and_agrees_with_the_others() {
    for (codec, _, _) in CODECS {
        for channels in [1u16, 3] {
            let payload = payload_for(codec, channels, 101, 0xF1A);
            let samples = expected_samples(codec, &payload);
            let spec = AudioSpec::new(44_100, channels);
            let mut digests = Vec::new();

            for header in [WavHeaderStyle::Plain, WavHeaderStyle::Extensible] {
                for flavour in [RiffFlavour::Automatic, RiffFlavour::Rf64] {
                    let writer = WavWriter::new(spec, codec)
                        .with_header_style(header)
                        .with_flavour(flavour);
                    let written = writer.to_bytes(&samples).expect("write");
                    let label = format!("{codec:?} {channels}ch {header:?} {flavour:?}");

                    let reader =
                        WavReader::new(&written).unwrap_or_else(|e| panic!("{label}: {e}"));
                    assert_eq!(reader.is_rf64(), flavour == RiffFlavour::Rf64, "{label}");
                    assert_eq!(
                        reader.format().is_extensible(),
                        header == WavHeaderStyle::Extensible,
                        "{label}"
                    );
                    assert_eq!(reader.format().codec, codec, "{label}");
                    assert_eq!(reader.spec(), spec, "{label}");
                    assert_eq!(reader.frames(), 101, "{label}");
                    assert_same_samples(reader.decode_to_end().samples(), &samples, &label);

                    let again = writer
                        .to_bytes(reader.decode_to_end().samples())
                        .expect("rewrite");
                    assert_eq!(
                        sha256(&written),
                        sha256(&again),
                        "{label}: not a fixed point"
                    );
                    digests.push((label, sha256(&written)));
                }
            }

            // The four spellings are four different files. If two hashed the
            // same, one of the options was doing nothing.
            for (index, (left, left_hash)) in digests.iter().enumerate() {
                for (right, right_hash) in &digests[index + 1..] {
                    assert_ne!(
                        left_hash, right_hash,
                        "{left} and {right} wrote the same file"
                    );
                }
            }
        }
    }
}

// -- Gate 7: the partial trailing frame ---------------------------------------

/// A `data` chunk whose length is not a whole multiple of the frame size is
/// malformed input, on both paths.
///
/// decibri's reader does not check this: it hands the payload to a
/// `chunks_exact` conversion, so the remainder is dropped and the caller is
/// never told. Silent truncation of audio is the failure class this crate
/// exists to avoid.
#[test]
fn a_partial_trailing_frame_is_truncated() {
    for (codec, tag, bits) in CODECS {
        for channels in [2u16, 3, 6] {
            let frame_bytes = usize::from(channels) * codec.bytes_per_sample();
            for remainder in 1..frame_bytes {
                for whole_frames in [0usize, 1, 17] {
                    let payload =
                        payload_bytes(whole_frames * frame_bytes + remainder, 0x7A11, None);
                    let bytes = wav(tag, channels, 8_000, bits, &payload);
                    let label = format!("{codec:?} {channels}ch, {remainder} bytes over");

                    let error = WavReader::new(&bytes)
                        .err()
                        .unwrap_or_else(|| panic!("{label}: accepted a partial frame"));
                    assert!(
                        matches!(
                            error,
                            DecodeError::Truncated { expected, available }
                                if expected == frame_bytes as u64 && available == remainder as u64
                        ),
                        "{label}: unexpected error: {error}"
                    );

                    // And the streaming path says the same thing at the same
                    // point, rather than discovering it at the end.
                    let error = stream_decode(&bytes, 64, Drive::Interleaved)
                        .err()
                        .unwrap_or_else(|| panic!("{label}: stream accepted a partial frame"));
                    assert!(
                        matches!(
                            error,
                            DecodeError::Truncated { expected, available }
                                if expected == frame_bytes as u64 && available == remainder as u64
                        ),
                        "{label}: stream gave an unexpected error: {error}"
                    );
                }
            }
        }
    }
}

/// The writer will not produce what the reader rejects: a sample count that is
/// not a whole number of frames is refused rather than padded or trimmed.
#[test]
fn the_writer_refuses_to_write_a_partial_trailing_frame() {
    for (codec, _, _) in CODECS {
        for channels in [2u16, 3, 6] {
            let writer = WavWriter::new(AudioSpec::new(8_000, channels), codec);
            for extra in 1..usize::from(channels) {
                let samples = vec![0.25f32; usize::from(channels) * 4 + extra];
                let error = writer
                    .to_bytes(&samples)
                    .expect_err("a partial frame must be refused");
                assert!(
                    matches!(error, DecodeError::Truncated { .. }),
                    "{codec:?} {channels}ch: unexpected error: {error}"
                );
            }
            assert!(writer
                .to_bytes(&vec![0.25; usize::from(channels) * 4])
                .is_ok());
        }
    }
}

// -- Gate 8: robustness -------------------------------------------------------

/// A valid file to mutate, and the samples it decodes to.
fn healthy_file() -> Vec<u8> {
    file(
        b"RIFF",
        SizeClaim::Truthful,
        &[
            chunk(b"LIST", b"INFOx"),
            chunk(b"fmt ", &fmt_plain(1, 2, 44_100, 16)),
            chunk(b"data", &payload_bytes(64, 0x4EA1, None)),
        ],
    )
}

/// Every malformed input the relay names, each with the typed error it must
/// produce, on both the whole-file and the streaming path.
#[test]
fn every_malformed_input_produces_a_typed_error() {
    let payload = payload_bytes(64, 0xBAD, None);

    // Truncated header: every prefix of the twelve-byte header.
    for length in 0..12 {
        let bytes = &wav(1, 1, 8_000, 16, &payload)[..length];
        assert!(
            matches!(
                WavReader::new(bytes),
                Err(DecodeError::Truncated { expected: 12, .. })
            ),
            "{length}-byte header was accepted"
        );
    }

    // Truncated mid-`fmt `: the header is whole, the `fmt ` chunk is not.
    for length in 13..12 + 8 + 16 {
        let bytes = &wav(1, 1, 8_000, 16, &payload)[..length];
        assert!(
            WavReader::new(bytes).is_err(),
            "a file cut at {length} bytes was accepted"
        );
    }

    // Truncated mid-`data`.
    let whole = wav(1, 2, 8_000, 16, &payload);
    for cut in 1..payload.len() {
        let bytes = &whole[..whole.len() - cut];
        let error = WavReader::new(bytes).expect_err("a short data chunk must reject");
        assert!(
            matches!(error, DecodeError::Truncated { .. }),
            "cut {cut}: unexpected error: {error}"
        );
    }

    // Missing `fmt `.
    let no_fmt = file(
        b"RIFF",
        SizeClaim::Truthful,
        &[chunk(b"data", &payload), chunk(b"LIST", b"INFO")],
    );
    assert!(matches!(
        WavReader::new(&no_fmt),
        Err(DecodeError::Malformed {
            expected: "a fmt chunk",
            ..
        })
    ));

    // Missing `data`.
    let no_data = file(
        b"RIFF",
        SizeClaim::Truthful,
        &[chunk(b"fmt ", &fmt_plain(1, 1, 8_000, 16))],
    );
    assert!(matches!(
        WavReader::new(&no_data),
        Err(DecodeError::Malformed {
            expected: "a data chunk",
            ..
        })
    ));

    // `data` declared larger than the file: four gigabytes inside a small one.
    let over = file(
        b"RIFF",
        SizeClaim::Truthful,
        &[
            chunk(b"fmt ", &fmt_plain(1, 1, 8_000, 16)),
            chunk_declaring(b"data", 0xFFFF_FF00, &payload),
        ],
    );
    let error = WavReader::new(&over).expect_err("an over-declared data chunk must reject");
    assert!(
        matches!(
            error,
            DecodeError::Truncated {
                expected: 0xFFFF_FF00,
                available
            } if available == payload.len() as u64
        ),
        "unexpected error: {error}"
    );

    // Zero channels.
    let no_channels = wav(1, 0, 8_000, 16, &payload);
    assert!(matches!(
        WavReader::new(&no_channels),
        Err(DecodeError::UnsupportedChannelLayout { channels: 0 })
    ));

    // Zero sample rate.
    let no_rate = wav(1, 1, 0, 16, &payload);
    assert!(matches!(
        WavReader::new(&no_rate),
        Err(DecodeError::Malformed {
            expected: "a non-zero sample rate",
            ..
        })
    ));

    // `cbSize` inconsistent with the extensible layout.
    for cb_size in [0u16, 1, 21] {
        let bytes = file(
            b"RIFF",
            SizeClaim::Truthful,
            &[
                chunk(b"fmt ", &fmt_extensible(1, 8_000, 16, 1, cb_size)),
                chunk(b"data", &payload),
            ],
        );
        assert!(
            matches!(WavReader::new(&bytes), Err(DecodeError::Malformed { .. })),
            "cbSize {cb_size} was accepted"
        );
    }
    // And an extensible header whose `fmt ` chunk is too short to hold a GUID.
    let short_extensible = file(
        b"RIFF",
        SizeClaim::Truthful,
        &[
            chunk(b"fmt ", &fmt_plain(0xFFFE, 1, 8_000, 16)),
            chunk(b"data", &payload),
        ],
    );
    assert!(matches!(
        WavReader::new(&short_extensible),
        Err(DecodeError::Malformed { .. })
    ));

    // A chunk size chosen to wrap `offset + size` in 32-bit arithmetic.
    for declared in [u32::MAX, u32::MAX - 1, 0xFFFF_FFF0, 0x8000_0000] {
        for id in [b"data", b"LIST", b"fmt "] {
            let bytes = file(
                b"RIFF",
                SizeClaim::Truthful,
                &[
                    chunk_declaring(id, declared, &payload),
                    chunk(b"fmt ", &fmt_plain(1, 1, 8_000, 16)),
                    chunk(b"data", &payload),
                ],
            );
            let error = WavReader::new(&bytes)
                .err()
                .unwrap_or_else(|| panic!("{declared:#x} in {id:?} was accepted"));
            assert!(
                matches!(error, DecodeError::Truncated { .. }),
                "{declared:#x} in {id:?}: unexpected error: {error}"
            );
        }
    }

    // Not a RIFF file at all, and a RIFF file that is not WAVE.
    assert!(matches!(
        WavReader::new(b"FORM\x00\x00\x00\x00AIFF"),
        Err(DecodeError::UnsupportedContainer { .. })
    ));
    let not_wave = file(b"RIFF", SizeClaim::Truthful, &[chunk(b"data", &payload)]);
    let mut avi = not_wave.clone();
    avi[8..12].copy_from_slice(b"AVI ");
    assert!(matches!(
        WavReader::new(&avi),
        Err(DecodeError::UnsupportedContainer { .. })
    ));

    // An unsupported codec, and a carried codec at an unsupported width.
    assert!(matches!(
        WavReader::new(&wav(0x11, 1, 8_000, 4, &payload)),
        Err(DecodeError::UnsupportedCodec { .. })
    ));
    assert!(matches!(
        WavReader::new(&wav(1, 1, 8_000, 20, &payload)),
        Err(DecodeError::UnsupportedSampleFormat { .. })
    ));
}

/// Every prefix of a healthy file, and every prefix of a healthy RF64 file.
///
/// Any of them may be rejected and none of them may panic, hang or decode to
/// something longer than the bytes that were there. This is the gate that
/// varies *input length* exhaustively rather than at a few chosen points.
#[test]
fn no_prefix_of_a_valid_file_panics_or_over_reads() {
    let payload = payload_bytes(200, 0xCAFE, None);
    let plain = healthy_file();
    let rf64 = file(
        b"RF64",
        SizeClaim::Over,
        &[
            chunk(b"ds64", &ds64_body(0, payload.len() as u64, 50)),
            chunk(b"fmt ", &fmt_plain(1, 2, 48_000, 16)),
            chunk_declaring(b"data", u32::MAX, &payload),
        ],
    );

    for whole in [&plain, &rf64] {
        for length in 0..=whole.len() {
            let bytes = &whole[..length];
            match WavReader::new(bytes) {
                Err(_) => {}
                Ok(reader) => {
                    let decoded = reader.decode_to_end();
                    assert!(
                        decoded.samples().len() * reader.format().codec.bytes_per_sample()
                            <= bytes.len(),
                        "{length} bytes decoded to more audio than the input held"
                    );
                    assert_eq!(decoded.frames() as u64, reader.frames());
                }
            }
            // The streaming path over the same prefix, at two feed sizes.
            for piece in [1, 13] {
                let _ = stream_decode(bytes, piece, Drive::Interleaved);
            }
        }
    }
}

/// Single-byte mutations of a healthy file. None may panic; the ones that are
/// accepted must still decode within the bytes present.
#[test]
fn no_single_byte_mutation_of_a_valid_file_panics() {
    let healthy = healthy_file();
    for offset in 0..healthy.len() {
        for value in [0x00u8, 0x01, 0x7F, 0x80, 0xFE, 0xFF] {
            let mut bytes = healthy.clone();
            bytes[offset] = value;
            if let Ok(reader) = WavReader::new(&bytes) {
                let decoded = reader.decode_to_end();
                assert!(
                    decoded.samples().len() * reader.format().codec.bytes_per_sample()
                        <= bytes.len(),
                    "byte {offset} set to {value:#04x} decoded past the input"
                );
            }
            let _ = stream_decode(&bytes, 7, Drive::Greedy);
        }
    }
}

/// An under-declared `data` size is a shorter file, not a broken one: the
/// declared length is what is decoded and the trailing bytes are not audio.
#[test]
fn an_under_declared_data_size_yields_exactly_what_it_declared() {
    let payload = payload_bytes(64, 0x0DEC, None);
    for declared in [0u32, 2, 32, 62] {
        let bytes = file(
            b"RIFF",
            SizeClaim::Truthful,
            &[
                chunk(b"fmt ", &fmt_plain(1, 1, 8_000, 16)),
                chunk_declaring(b"data", declared, &payload),
            ],
        );
        let reader = WavReader::new(&bytes).expect("an under-declared chunk still parses");
        assert_eq!(
            reader.frames(),
            u64::from(declared) / 2,
            "declared {declared}"
        );
        assert_eq!(reader.data(), &payload[..declared as usize]);
        let (_, streamed) = stream_decode(&bytes, 5, Drive::Interleaved).expect("stream");
        assert_same_samples(
            &streamed,
            reader.decode_to_end().samples(),
            &format!("declared {declared}"),
        );
    }
}

// -- Gate 9: the streaming path -----------------------------------------------

/// Feed chunk size **and** total length, crossed.
///
/// Step 3's Control B is why the two are named separately: that pass drove ten
/// chunk sizes over a 256-byte input and the limit it needed to breach was
/// 65,536, so the short-return path never ran and a `feed` that over-reported
/// consumption went unnoticed. The largest total here is past this crate's
/// 65,536-sample ready limit in every layout, and the greedy drive mode is the
/// one that reaches it.
#[test]
fn the_streaming_path_matches_the_whole_file_path() {
    // 70,000 mono frames is past the 65,536-sample ready limit; the multichannel
    // rows pass it several times over.
    const TOTALS: [usize; 5] = [0, 1, 3, 1_000, 70_000];
    const PIECES: [usize; 5] = [1, 7, 512, 4_096, 1 << 20];

    for (codec, tag, bits) in CODECS {
        for channels in [1u16, 2] {
            for frames in TOTALS {
                let payload = payload_for(codec, channels, frames, 0x57EA);
                let bytes = wav(tag, channels, 32_000, bits, &payload);
                let whole = WavReader::new(&bytes).expect("whole-file").decode_to_end();
                let expected = expected_samples(codec, &payload);
                assert_same_samples(whole.samples(), &expected, "whole-file");

                for piece in PIECES {
                    for drive in [Drive::Interleaved, Drive::Greedy] {
                        let label = format!(
                            "{codec:?} {channels}ch {frames} frames, {piece}-byte {drive:?}"
                        );
                        let (spec, samples) = stream_decode(&bytes, piece, drive)
                            .unwrap_or_else(|error| panic!("{label}: {error}"));
                        assert_eq!(spec, Some(whole.spec()), "{label}: spec");
                        assert_same_samples(&samples, &expected, &label);
                    }
                }
            }
        }
    }
}

/// Pulling a bounded number of frames at a time gives the same audio as pulling
/// everything, and the reader does not grow while it is being drained slowly.
#[test]
fn a_caller_pulling_a_frame_at_a_time_gets_the_same_audio() {
    let payload = payload_for(WavCodec::PcmI16, 2, 5_000, 0x1F1F);
    let bytes = wav(1, 2, 48_000, 16, &payload);
    let expected = expected_samples(WavCodec::PcmI16, &payload);

    for max_frames in [1usize, 2, 97] {
        let mut stream = WavStreamDecoder::new();
        let mut samples = Vec::new();
        for slice in bytes.chunks(333) {
            let mut offset = 0;
            while offset < slice.len() {
                let taken = stream.push(&slice[offset..]).expect("push");
                offset += taken;
                loop {
                    let pulled = stream.pull(&mut samples, max_frames).expect("pull");
                    assert!(pulled <= max_frames, "pull ignored its frame limit");
                    if pulled == 0 {
                        break;
                    }
                }
            }
        }
        stream.finish(&mut samples).expect("finish");
        assert_same_samples(
            &samples,
            &expected,
            &format!("pulling {max_frames} frames at a time"),
        );
    }
}

/// A stream that ends early is `Truncated`, wherever it ends; a stream that
/// ends on a chunk boundary without ever naming its audio is `Malformed`.
#[test]
fn a_stream_that_ends_early_reports_where_it_ended() {
    let payload = payload_bytes(128, 0xE0F, None);
    let whole = wav(1, 2, 16_000, 16, &payload);
    for length in 0..whole.len() {
        let error = stream_decode(&whole[..length], 9, Drive::Interleaved)
            .err()
            .unwrap_or_else(|| panic!("a stream cut at {length} bytes was accepted"));
        assert!(
            matches!(
                error,
                DecodeError::Truncated { .. } | DecodeError::Malformed { .. }
            ),
            "cut at {length}: unexpected error: {error}"
        );
    }
    // The whole thing is fine.
    assert!(stream_decode(&whole, 9, Drive::Interleaved).is_ok());
}

/// The pad byte, on the streaming path, after each of the three chunk kinds the
/// reader treats differently.
///
/// **This gate exists because of what a negative control measured.** The two
/// readers implement the pad byte in two different places. The whole-file walk
/// applies `size & 1` to an offset and the streaming reader has a `Pad` state,
/// so a control that breaks one says nothing about the other. Breaking the
/// streaming half was caught by exactly one test, `the_dimension_matrix`, and
/// one gate between a caller and a mis-parsed file is one too few.
///
/// The three kinds are: a chunk whose body is skipped, a `fmt ` chunk whose
/// body is buffered, and a `ds64` chunk, which is buffered on a different code
/// path again. All three are given odd-length bodies, which is unusual for the
/// last two and perfectly legal.
#[test]
fn the_streaming_reader_applies_the_pad_byte_after_every_kind_of_chunk() {
    let payload = payload_bytes(64, 0x9AD, None);
    let expected = expected_samples(WavCodec::PcmI16, &payload);

    // After a skipped chunk: five bytes of `LIST`, then the pad.
    let after_skip = file(
        b"RIFF",
        SizeClaim::Truthful,
        &[
            chunk(b"LIST", b"INFOx"),
            chunk(b"fmt ", &fmt_plain(1, 1, 8_000, 16)),
            chunk(b"data", &payload),
        ],
    );

    // After a buffered `fmt `: a seventeen-byte body, which is `WAVEFORMATEX`
    // plus one trailing byte and still at least the sixteen the format needs.
    let mut odd_fmt = fmt_plain(1, 1, 8_000, 16);
    odd_fmt.push(0);
    let after_fmt = file(
        b"RIFF",
        SizeClaim::Truthful,
        &[chunk(b"fmt ", &odd_fmt), chunk(b"data", &payload)],
    );

    // After a buffered `ds64`: a twenty-nine-byte body.
    let mut odd_ds64 = ds64_body(0, payload.len() as u64, 32);
    odd_ds64.push(0);
    let after_ds64 = file(
        b"RF64",
        SizeClaim::Truthful,
        &[
            chunk(b"ds64", &odd_ds64),
            chunk(b"fmt ", &fmt_plain(1, 1, 8_000, 16)),
            chunk_declaring(b"data", u32::MAX, &payload),
        ],
    );

    for (label, bytes) in [
        ("after a skipped chunk", &after_skip),
        ("after a buffered fmt chunk", &after_fmt),
        ("after a buffered ds64 chunk", &after_ds64),
    ] {
        // The whole-file walk first, so a failure names which half is wrong.
        let reader = WavReader::new(bytes).unwrap_or_else(|e| panic!("{label}, whole file: {e}"));
        assert_same_samples(reader.decode_to_end().samples(), &expected, label);

        // Then the stream, at feed sizes that land on and around the pad byte.
        for piece in [1usize, 2, 3, 5, 8, 4_096] {
            let (spec, samples) = stream_decode(bytes, piece, Drive::Interleaved)
                .unwrap_or_else(|e| panic!("{label}, streamed in {piece}-byte pieces: {e}"));
            assert_eq!(spec, Some(AudioSpec::mono(8_000)), "{label}");
            assert_same_samples(&samples, &expected, &format!("{label} at {piece} bytes"));
        }
    }
}

/// The one thing the streaming path cannot do that the whole-file path can, and
/// it says so rather than mis-decoding.
#[test]
fn the_stream_requires_fmt_before_data_and_the_whole_file_reader_does_not() {
    let payload = payload_bytes(64, 0xDA7A, None);
    let bytes = file(
        b"RIFF",
        SizeClaim::Truthful,
        &[
            chunk(b"data", &payload),
            chunk(b"fmt ", &fmt_plain(1, 1, 8_000, 16)),
        ],
    );

    let reader = WavReader::new(&bytes).expect("the whole-file reader accepts either order");
    assert_eq!(reader.frames(), 32);

    let error = stream_decode(&bytes, 16, Drive::Interleaved).expect_err("the stream cannot");
    assert!(
        matches!(
            error,
            DecodeError::Malformed {
                expected: "a fmt chunk before the data chunk",
                ..
            }
        ),
        "unexpected error: {error}"
    );
}

/// `reset` returns the reader to its just-constructed state, so one instance
/// decodes a second file without inheriting anything from the first.
#[test]
fn a_reset_stream_decodes_a_second_file_from_scratch() {
    let first = wav(1, 1, 8_000, 16, &payload_bytes(32, 1, None));
    let second = wav(7, 2, 48_000, 8, &payload_bytes(64, 2, Some(0x7F)));

    let mut stream = WavStreamDecoder::new();
    let mut samples = Vec::new();
    stream.push(&first).expect("push");
    while stream.pull(&mut samples, usize::MAX).expect("pull") > 0 {}
    stream.finish(&mut samples).expect("finish");
    assert_eq!(stream.spec(), Some(AudioSpec::mono(8_000)));

    stream.reset();
    assert_eq!(stream.spec(), None, "reset kept the first file's spec");
    let mut again = Vec::new();
    stream.push(&second).expect("push");
    while stream.pull(&mut again, usize::MAX).expect("pull") > 0 {}
    stream.finish(&mut again).expect("finish");
    assert_eq!(stream.spec(), Some(AudioSpec::new(48_000, 2)));
    assert_same_samples(
        &again,
        WavReader::new(&second)
            .expect("read")
            .decode_to_end()
            .samples(),
        "the second file after a reset",
    );
}

// -- The dimension matrix -----------------------------------------------------

/// Where the unknown chunks sit and which order the known ones come in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Layout {
    /// `fmt ` then `data`, nothing else.
    Plain,
    /// `data` then `fmt `: legal RIFF, and the whole-file reader's business.
    DataFirst,
    /// An even-length unknown chunk before `fmt `.
    UnknownBefore,
    /// An odd-length unknown chunk before `fmt `, so a pad byte decides where
    /// `fmt ` starts.
    OddUnknownBefore,
    /// An unknown chunk between `fmt ` and `data`.
    UnknownBetween,
    /// Unknown chunks after `data`, which the walk never reaches.
    UnknownAfter,
    /// One of each, in every position at once.
    UnknownEverywhere,
}

const LAYOUTS: [Layout; 7] = [
    Layout::Plain,
    Layout::DataFirst,
    Layout::UnknownBefore,
    Layout::OddUnknownBefore,
    Layout::UnknownBetween,
    Layout::UnknownAfter,
    Layout::UnknownEverywhere,
];

impl Layout {
    /// `true` when the streaming reader can read this layout. Only
    /// `data`-before-`fmt ` is out, and for a stated reason.
    fn is_streamable(self) -> bool {
        self != Layout::DataFirst
    }

    fn chunks(self, fmt: Vec<u8>, data: Vec<u8>) -> Vec<Vec<u8>> {
        let even = || chunk(b"JUNK", b"padding!");
        let odd = || chunk(b"LIST", b"INFOx");
        match self {
            Self::Plain => vec![fmt, data],
            Self::DataFirst => vec![data, fmt],
            Self::UnknownBefore => vec![even(), fmt, data],
            Self::OddUnknownBefore => vec![odd(), fmt, data],
            Self::UnknownBetween => vec![fmt, odd(), data],
            Self::UnknownAfter => vec![fmt, data, odd(), even()],
            Self::UnknownEverywhere => vec![odd(), fmt, even(), data, odd(), even()],
        }
    }
}

/// The cross product: every encoding, at three channel counts, at four lengths,
/// in seven chunk layouts, under three RIFF size claims, as RIFF and as RF64,
/// with the streaming feed size rotated through the product so that it is
/// crossed with all of them rather than swept on its own.
///
/// 8 * 3 * 4 * 7 * 3 * 2 = 4,032 files. Each is decoded through both paths and
/// checked against the payload decoded directly, which is a statement about the
/// container and nothing else: the codec layer has its own gates.
#[test]
fn the_dimension_matrix() {
    /// Rotated through the product rather than nested inside it, so every feed
    /// size meets every other dimension without multiplying the file count.
    const PIECES: [usize; 8] = [1, 2, 3, 5, 13, 64, 997, 65_536];
    let mut index = 0usize;
    let mut files = 0usize;

    for (codec, tag, bits) in CODECS {
        for channels in [1u16, 2, 6] {
            for frames in [0usize, 1, 3, 129] {
                for layout in LAYOUTS {
                    for claim in [SizeClaim::Truthful, SizeClaim::Under, SizeClaim::Over] {
                        for rf64 in [false, true] {
                            index += 1;
                            files += 1;
                            let payload = payload_for(codec, channels, frames, index as u64);
                            let expected = expected_samples(codec, &payload);
                            let spec = AudioSpec::new(22_050, channels);
                            let label = format!(
                                "{codec:?} {channels}ch {frames}f {layout:?} {claim:?} rf64={rf64}"
                            );

                            let fmt = chunk(b"fmt ", &fmt_plain(tag, channels, 22_050, bits));
                            let chunks = if rf64 {
                                // RF64 leaves the `data` size at the sentinel
                                // and states it in `ds64`, which has to be the
                                // first chunk whatever the rest of the layout
                                // is.
                                let data = chunk_declaring(b"data", u32::MAX, &payload);
                                let mut chunks = layout.chunks(fmt, data);
                                chunks.insert(
                                    0,
                                    chunk(
                                        b"ds64",
                                        &ds64_body(0, payload.len() as u64, frames as u64),
                                    ),
                                );
                                chunks
                            } else {
                                layout.chunks(fmt, chunk(b"data", &payload))
                            };
                            let bytes = file(if rf64 { b"RF64" } else { b"RIFF" }, claim, &chunks);

                            let reader =
                                WavReader::new(&bytes).unwrap_or_else(|e| panic!("{label}: {e}"));
                            assert_eq!(reader.format().codec, codec, "{label}");
                            assert_eq!(reader.spec(), spec, "{label}");
                            assert_eq!(reader.frames(), frames as u64, "{label}");
                            assert_eq!(reader.is_rf64(), rf64, "{label}");
                            assert_same_samples(
                                reader.decode_to_end().samples(),
                                &expected,
                                &label,
                            );

                            if layout.is_streamable() {
                                let piece = PIECES[index % PIECES.len()];
                                let drive = if index.is_multiple_of(2) {
                                    Drive::Interleaved
                                } else {
                                    Drive::Greedy
                                };
                                let (streamed_spec, streamed) = stream_decode(&bytes, piece, drive)
                                    .unwrap_or_else(|e| {
                                        panic!("{label} at {piece}-byte {drive:?}: {e}")
                                    });
                                assert_eq!(streamed_spec, Some(spec), "{label} streamed spec");
                                assert_same_samples(
                                    &streamed,
                                    &expected,
                                    &format!("{label} streamed at {piece} bytes, {drive:?}"),
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!(files, 8 * 3 * 4 * 7 * 3 * 2);
}

// -- Gate 10: cross-platform determinism --------------------------------------

/// FNV-1a, so the witness is the bytes themselves and not a float comparison.
fn fnv1a(bytes: impl IntoIterator<Item = u8>) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// One number that changes if any bit of any decoded sample or any written byte
/// changes, over every encoding, both header styles and both RIFF flavours.
///
/// Running this on two toolchains and on a 32-bit target and getting the same
/// constant is the evidence for the byte-identical claim; a tolerance-based
/// test would pass on all three while the outputs differed. The constant is
/// pinned rather than recomputed so that a change shows up as a diff here.
#[test]
fn wav_output_is_bit_identical_to_a_pinned_witness() {
    let mut witness: Vec<u8> = Vec::new();
    for (codec, tag, bits) in CODECS {
        for channels in [1u16, 3] {
            let payload = payload_for(codec, channels, 97, 0x1CE);
            let bytes = wav(tag, channels, 24_000, bits, &payload);

            // The decode, through both paths. 7 is coprime with every sample
            // width here, so the streaming reader is driven across its
            // partial-sample path as well as its whole one.
            let decoded = WavReader::new(&bytes).expect("read").decode_to_end();
            let (_, streamed) = stream_decode(&bytes, 7, Drive::Greedy).expect("stream");
            assert_same_samples(&streamed, decoded.samples(), "witness stream");
            witness.extend(
                decoded
                    .samples()
                    .iter()
                    .flat_map(|s| s.to_bits().to_le_bytes()),
            );

            // And every spelling the writer offers.
            for header in [WavHeaderStyle::Plain, WavHeaderStyle::Extensible] {
                for flavour in [RiffFlavour::Automatic, RiffFlavour::Rf64] {
                    let written = WavWriter::new(AudioSpec::new(24_000, channels), codec)
                        .with_header_style(header)
                        .with_flavour(flavour)
                        .to_bytes(decoded.samples())
                        .expect("write");
                    witness.extend_from_slice(&written);
                }
            }
        }
    }
    assert_eq!(fnv1a(witness), 0x51fe_2597_ebf5_2432, "WAV output changed");
}
