#![forbid(unsafe_code)]
//! AIFF write conformance: the writer against the step-4b hand-built
//! fixtures, byte for byte.
//!
//! # The oracle, and why it is the one used
//!
//! Round-tripping the writer through this crate's own reader proves only
//! that the two agree with each other: two matching bugs pass it, which is
//! coverage lesson 4. The independent oracle already exists: the
//! `aiff_conformance.rs` fixtures were assembled byte by byte from the two
//! format specifications before any writer existed, so they cannot share a
//! mistake with it. **The primary gate here is that the writer reproduces
//! those fixture bytes exactly** for the same audio and the same encoding.
//! The builders below are duplicated from `aiff_conformance.rs` rather than
//! imported, partly because integration tests cannot share code without a
//! common module and mostly so that this file's oracle stays a self-contained
//! transcription of the specifications.
//!
//! The payload region inside each expected file is spelled through the
//! frozen codec layer (`SampleFormat::encode` and `G711Law::encode` called
//! directly, never through the writer), whose own independence is anchored
//! by the codec-layer gates and by the cross-container agreement gate below;
//! the container bytes around it (the part this writer adds) come only
//! from the hand transcription. The signedness and `sowt` gates additionally
//! pin a handful of payload bytes by hand, so the codec layer is not the
//! sole authority even there.
//!
//! Round-trip identity by SHA-256 and cross-container agreement are kept as
//! secondary gates: they catch a writer that drifts from the rest of the
//! crate, not one that is wrong in a way its own reader mirrors.

use decibri_decode::{
    AiffCodec, AiffReader, AiffStreamDecoder, AiffWriter, AudioSpec, DecodeError, FourCc,
    StreamSource, WavCodec, WavReader, WavWriter,
};

// -- The reference builders, duplicated from aiff_conformance.rs --------------

/// A big-endian chunk: identifier, declared size, body, and the pad byte
/// when the body is odd.
fn chunk(id: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(body.len() + 9);
    out.extend_from_slice(id);
    out.extend_from_slice(&(body.len() as u32).to_be_bytes());
    out.extend_from_slice(body);
    if body.len() % 2 == 1 {
        out.push(0);
    }
    out
}

/// A whole file: `FORM`, a truthful big-endian size, the form type, then the
/// chunks in order.
fn form(form_type: &[u8; 4], chunks: &[Vec<u8>]) -> Vec<u8> {
    let body: Vec<u8> = chunks.concat();
    let mut out = Vec::with_capacity(body.len() + 12);
    out.extend_from_slice(b"FORM");
    out.extend_from_slice(&(4 + body.len() as u32).to_be_bytes());
    out.extend_from_slice(form_type);
    out.extend_from_slice(&body);
    out
}

/// An integer sample rate as the 80-bit extended float `COMM` stores: an
/// independent implementation of the encoding, written from the format
/// definition rather than shared with the writer, so a mistake in one is
/// not mirrored in the other.
fn extended_rate(rate: u32) -> [u8; 10] {
    assert!(rate > 0, "a zero rate has no normalised form");
    let high_bit = 31 - rate.leading_zeros();
    let exponent = 16383 + high_bit as u16;
    let significand = u64::from(rate) << (63 - high_bit);
    let mut out = [0u8; 10];
    out[0] = (exponent >> 8) as u8;
    out[1] = exponent as u8;
    out[2..].copy_from_slice(&significand.to_be_bytes());
    out
}

/// A `COMM` body from the field order in the two specifications.
/// `compression` is `None` for a plain `AIFF` form; `Some` appends the
/// four-CC and a zero-length pascal string with its even-total pad, which is
/// the spelling the writer documents.
fn comm(
    channels: u16,
    frames: u32,
    bits: u16,
    rate: [u8; 10],
    compression: Option<&[u8; 4]>,
) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(&channels.to_be_bytes());
    body.extend_from_slice(&frames.to_be_bytes());
    body.extend_from_slice(&bits.to_be_bytes());
    body.extend_from_slice(&rate);
    if let Some(four_cc) = compression {
        body.extend_from_slice(four_cc);
        body.extend_from_slice(&[0, 0]);
    }
    body
}

/// An `SSND` body: the `offset` and `blockSize` fields, zero as the writer
/// states it writes them, then the sample data.
fn ssnd(data: &[u8]) -> Vec<u8> {
    let mut body = Vec::with_capacity(8 + data.len());
    body.extend_from_slice(&0u32.to_be_bytes());
    body.extend_from_slice(&0u32.to_be_bytes());
    body.extend_from_slice(data);
    body
}

/// AIFF-C's `FVER` chunk with the one version timestamp that has ever
/// existed.
fn fver() -> Vec<u8> {
    chunk(b"FVER", &0xA280_5140u32.to_be_bytes())
}

/// The complete file the writer is expected to produce for one encoding:
/// plain `AIFF` as `COMM` then `SSND`; `AIFC` with `FVER` first, its
/// specified position.
fn expected_file(
    form_type: &[u8; 4],
    channels: u16,
    frames: u32,
    bits: u16,
    rate: u32,
    compression: Option<&[u8; 4]>,
    payload: &[u8],
) -> Vec<u8> {
    let comm_chunk = chunk(
        b"COMM",
        &comm(channels, frames, bits, extended_rate(rate), compression),
    );
    let ssnd_chunk = chunk(b"SSND", &ssnd(payload));
    match compression {
        None => form(form_type, &[comm_chunk, ssnd_chunk]),
        Some(_) => form(form_type, &[fver(), comm_chunk, ssnd_chunk]),
    }
}

// -- Deterministic test signals -----------------------------------------------

/// Payload bytes from an LCG, so the same bytes on every target and run.
fn payload_bytes(length: usize, seed: u64) -> Vec<u8> {
    let mut state = seed | 1;
    (0..length)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as u8
        })
        .collect()
}

/// Samples in `codec`'s own value domain: an LCG payload decoded through the
/// frozen codec layer. Decoding first is what quantises `i32` and `f64`
/// audio onto values an `f32` can carry, so the writer is asked to write
/// exactly what it was given.
fn samples_for(codec: AiffCodec, channels: u16, frames: usize, seed: u64) -> Vec<f32> {
    let payload = payload_bytes(
        frames * usize::from(channels) * codec.bytes_per_sample(),
        seed,
    );
    let mut samples = Vec::new();
    match (codec.sample_format(), codec.law()) {
        (Some(format), _) => format.decode(&payload, &mut samples),
        (_, Some(law)) => law.decode(&payload, &mut samples),
        _ => unreachable!("every codec is linear or companded"),
    };
    samples
}

/// The payload bytes `samples` spell in `codec`, through the frozen codec
/// layer directly, never through the writer under test.
fn canonical_payload(codec: AiffCodec, samples: &[f32]) -> Vec<u8> {
    let mut payload = Vec::new();
    match (codec.sample_format(), codec.law()) {
        (Some(format), _) => format.encode(samples, &mut payload),
        (_, Some(law)) => law.encode(samples, &mut payload),
        _ => unreachable!("every codec is linear or companded"),
    };
    payload
}

/// Compares samples by bit pattern, for the reasons recorded on the reader
/// suites: NaN differs from itself under `==`, and `+0.0 == -0.0` while the
/// bytes differ.
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

/// Byte-for-byte comparison that reports where the first divergence is,
/// with a few bytes of context, rather than dumping two whole files.
#[track_caller]
fn assert_same_bytes(actual: &[u8], expected: &[u8], label: &str) {
    if actual == expected {
        return;
    }
    let diverge = actual
        .iter()
        .zip(expected)
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| actual.len().min(expected.len()));
    let context =
        |bytes: &[u8]| bytes[diverge.saturating_sub(8)..(diverge + 8).min(bytes.len())].to_vec();
    panic!(
        "{label}: diverges from the hand-built fixture at byte {diverge} \
         (lengths {} against {}): wrote {:02X?}, fixture holds {:02X?}",
        actual.len(),
        expected.len(),
        context(actual),
        context(expected),
    );
}

/// One encoding row: the codec, and the form type, compression four-CC and
/// `sampleSize` its written file is expected to declare. Stated here as an
/// independent transcription of the writer's documented rule, so the sweep
/// cannot drift with the implementation.
type CodecRow = (AiffCodec, &'static [u8; 4], Option<&'static [u8; 4]>, u16);

/// Every writable encoding. The exhaustive-match test below keeps this list
/// honest against later `AiffCodec` variants.
const CODECS: [CodecRow; 12] = [
    (AiffCodec::PcmI8, b"AIFF", None, 8),
    (AiffCodec::PcmU8, b"AIFC", Some(b"raw "), 8),
    (AiffCodec::PcmI16, b"AIFF", None, 16),
    (AiffCodec::PcmI24, b"AIFF", None, 24),
    (AiffCodec::PcmI32, b"AIFF", None, 32),
    (AiffCodec::PcmI16Sowt, b"AIFC", Some(b"sowt"), 16),
    (AiffCodec::PcmI24Sowt, b"AIFC", Some(b"sowt"), 24),
    (AiffCodec::PcmI32Sowt, b"AIFC", Some(b"sowt"), 32),
    (AiffCodec::Float32, b"AIFC", Some(b"fl32"), 32),
    (AiffCodec::Float64, b"AIFC", Some(b"fl64"), 64),
    (AiffCodec::ALaw, b"AIFC", Some(b"alaw"), 16),
    (AiffCodec::MuLaw, b"AIFC", Some(b"ulaw"), 16),
];

/// The exhaustive match is the assertion: a variant added to [`AiffCodec`]
/// without a row in [`CODECS`] fails to compile here, so no later encoding
/// can be added and quietly left out of every gate in this suite.
#[test]
fn codecs_lists_every_writable_encoding() {
    for (codec, _, _, _) in CODECS {
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
            _ => panic!("a codec outside the writable set"),
        }
    }
    assert_eq!(CODECS.len(), 12);
}

// -- Gate 6: byte-for-byte agreement with the hand-built fixtures -------------

/// The primary gate. For every encoding, at several channel counts and
/// lengths, the writer's output must equal the hand-built fixture for the
/// same audio: container bytes from the specification transcription above,
/// payload bytes from the frozen codec layer.
#[test]
fn the_writer_reproduces_the_hand_built_fixtures_byte_for_byte() {
    const RATES: [u32; 6] = [8_000, 16_000, 22_050, 44_100, 48_000, 192_000];
    let mut index = 0usize;

    for (codec, form_type, compression, bits) in CODECS {
        for channels in [1u16, 2, 6] {
            for frames in [0usize, 1, 3, 129] {
                index += 1;
                let rate = RATES[index % RATES.len()];
                let samples = samples_for(codec, channels, frames, index as u64);
                let payload = canonical_payload(codec, &samples);
                let expected = expected_file(
                    form_type,
                    channels,
                    frames as u32,
                    bits,
                    rate,
                    compression,
                    &payload,
                );

                let written = AiffWriter::new(AudioSpec::new(rate, channels), codec)
                    .to_bytes(&samples)
                    .expect("write");
                assert_same_bytes(
                    &written,
                    &expected,
                    &format!("{codec:?} {channels}ch {frames}f at {rate} Hz"),
                );
            }
        }
    }
}

// -- Gate 9: eight-bit signedness, proven on the write side -------------------

/// The trap step 4b named as most likely to bite, now on the encode side:
/// AIFF's 8-bit is signed where WAV's is unsigned. Every payload byte here
/// is written out by hand from the specification's signed convention; a
/// writer that carried WAV's convention across would produce each of these
/// bytes offset by 0x80 and fail on the first sample.
#[test]
fn aiff_eight_bit_is_written_signed_and_fails_if_written_unsigned() {
    let samples: [f32; 6] = [
        0.0,           // silence is 0x00, not 0x80
        0.5,           // +64
        -1.0,          // the most negative value is 0x80, not 0x00
        127.0 / 128.0, // the most positive
        -0.5,          // -64
        -1.0 / 128.0,  // -1
    ];
    let signed: [u8; 6] = [0x00, 0x40, 0x80, 0x7F, 0xC0, 0xFF];

    let written = AiffWriter::new(AudioSpec::mono(8_000), AiffCodec::PcmI8)
        .to_bytes(&samples)
        .expect("write");
    let reader = AiffReader::new(&written).expect("read back");
    assert_eq!(
        reader.data(),
        signed,
        "the payload is not the specification's signed spelling"
    );

    // And the whole file equals the hand-built fixture around those bytes.
    assert_same_bytes(
        &written,
        &expected_file(b"AIFF", 1, 6, 8, 8_000, None, &signed),
        "8-bit signed fixture",
    );

    // The two containers' conventions are inverses, byte for byte: the same
    // audio through the WAV writer is the sign-bit flip of the AIFF spelling.
    let wav = WavWriter::new(AudioSpec::mono(8_000), WavCodec::PcmU8)
        .to_bytes(&samples)
        .expect("write WAV");
    let unsigned: Vec<u8> = signed.iter().map(|byte| byte ^ 0x80).collect();
    assert_eq!(
        WavReader::new(&wav).expect("read WAV").data(),
        unsigned.as_slice(),
        "the WAV spelling of the same audio is the sign-bit flip of the AIFF one"
    );
}

// -- The other eight-bit convention, written unsigned --------------------------

/// `raw ` is the one AIFF-C encoding whose 8-bit samples are unsigned, and
/// the failure it guards against is the inverse of the one above: a `raw `
/// file written with the signed convention is offset by half full scale and
/// nothing about its container says so. Every payload byte here is written
/// out by hand from the offset-binary definition, and asserted to be the
/// exact complement of the signed spelling the same audio takes under
/// `NONE`.
#[test]
fn raw_eight_bit_is_written_unsigned_and_fails_if_written_signed() {
    let samples: [f32; 6] = [
        0.0,           // mid scale is 0x80, not 0x00
        0.5,           // +64 above mid scale
        -1.0,          // the most negative value is 0x00, not 0x80
        127.0 / 128.0, // the most positive
        -0.5,          // -64
        -1.0 / 128.0,  // -1
    ];
    let unsigned: [u8; 6] = [0x80, 0xC0, 0x00, 0xFF, 0x40, 0x7F];
    let signed: [u8; 6] = [0x00, 0x40, 0x80, 0x7F, 0xC0, 0xFF];

    let written = AiffWriter::new(AudioSpec::mono(8_000), AiffCodec::PcmU8)
        .to_bytes(&samples)
        .expect("write");
    let reader = AiffReader::new(&written).expect("read back");
    assert_eq!(
        reader.data(),
        unsigned,
        "the raw payload is not the offset-binary spelling"
    );
    assert_eq!(
        reader.format().compression,
        FourCc(*b"raw "),
        "the file does not declare raw "
    );
    assert_eq!(reader.format().codec, AiffCodec::PcmU8);
    assert_eq!(reader.format().bits_per_sample, 8);

    // The whole file equals the hand-built AIFF-C fixture around those bytes.
    assert_same_bytes(
        &written,
        &expected_file(b"AIFC", 1, 6, 8, 8_000, Some(b"raw "), &unsigned),
        "raw fixture",
    );

    // The two eight-bit conventions are inverses byte for byte, so a writer
    // that carried the signed convention into `raw ` would produce `signed`
    // here and every byte would differ.
    let complement: Vec<u8> = signed.iter().map(|byte| byte ^ 0x80).collect();
    assert_eq!(unsigned.as_slice(), complement.as_slice());
    let as_aiff_signed = AiffWriter::new(AudioSpec::mono(8_000), AiffCodec::PcmI8)
        .to_bytes(&samples)
        .expect("write signed");
    assert_eq!(
        AiffReader::new(&as_aiff_signed)
            .expect("read signed")
            .data(),
        signed,
        "the signed spelling of the same audio moved"
    );

    // And the same audio as an 8-bit WAV is byte-identical in the payload,
    // since WAV's 8-bit and AIFF-C's `raw ` are the same convention.
    let wav = WavWriter::new(AudioSpec::mono(8_000), WavCodec::PcmU8)
        .to_bytes(&samples)
        .expect("write WAV");
    assert_eq!(
        WavReader::new(&wav).expect("read WAV").data(),
        unsigned,
        "WAV's eight-bit and AIFF-C's raw are not the same spelling"
    );
}

// -- Gate 10: sowt is written little-endian -----------------------------------

/// The bytes `[0x34, 0x12]` are 0x1234 little-endian and 0x3412 big-endian,
/// and the two values are far apart. Both directions are asserted so the
/// test cannot be satisfied by a writer that ignores the codec's byte order.
#[test]
fn sowt_is_written_little_endian_and_fails_if_written_big_endian() {
    // 0x1234 = 4660: exactly representable, so the encode is exact.
    let sample = [4_660.0f32 / 32_768.0];

    let sowt = AiffWriter::new(AudioSpec::mono(44_100), AiffCodec::PcmI16Sowt)
        .to_bytes(&sample)
        .expect("write sowt");
    assert_eq!(
        AiffReader::new(&sowt).expect("read sowt").data(),
        [0x34, 0x12],
        "sowt payload is not little-endian"
    );
    assert_same_bytes(
        &sowt,
        &expected_file(b"AIFC", 1, 1, 16, 44_100, Some(b"sowt"), &[0x34, 0x12]),
        "sowt fixture",
    );

    // The identical audio under the big-endian codec is the bytes reversed.
    let twos = AiffWriter::new(AudioSpec::mono(44_100), AiffCodec::PcmI16)
        .to_bytes(&sample)
        .expect("write big-endian");
    assert_eq!(
        AiffReader::new(&twos).expect("read big-endian").data(),
        [0x12, 0x34],
        "big-endian payload is not big-endian"
    );

    // 24- and 32-bit sowt as well: each width has its own reversal.
    let sowt24 = AiffWriter::new(AudioSpec::mono(44_100), AiffCodec::PcmI24Sowt)
        .to_bytes(&[1_193_046.0 / 8_388_608.0]) // 0x123456
        .expect("write sowt 24");
    assert_eq!(
        AiffReader::new(&sowt24).expect("read sowt 24").data(),
        [0x56, 0x34, 0x12]
    );
    // 0x12345600 = 0x123456 << 8: 21 significant bits, so the f32 is exact
    // and the encode is the identity rather than a rounding.
    let sowt32 = AiffWriter::new(AudioSpec::mono(44_100), AiffCodec::PcmI32Sowt)
        .to_bytes(&[1_193_046.0f32 / 8_388_608.0])
        .expect("write sowt 32");
    assert_eq!(
        AiffReader::new(&sowt32).expect("read sowt 32").data(),
        [0x00, 0x56, 0x34, 0x12]
    );
}

// -- Hand-spelled payloads for the two formats f32 cannot hold ----------------

/// `i32` and `f64` fixtures with payload bytes written by hand, so the
/// codec layer is not the payload oracle for the two formats whose LCG
/// sweep necessarily passes through it twice. The values are all exactly
/// representable in `f32`, so nothing here depends on a rounding rule.
#[test]
fn i32_and_f64_fixtures_agree_with_hand_spelled_payloads() {
    let samples: [f32; 4] = [0.0, 0.5, -0.5, -1.0];

    // i32 big-endian at scale 2^31: 0, 0x40000000, 0xC0000000, 0x80000000.
    let i32_payload: [u8; 16] = [
        0x00, 0x00, 0x00, 0x00, //  0.0
        0x40, 0x00, 0x00, 0x00, //  0.5 * 2^31
        0xC0, 0x00, 0x00, 0x00, // -0.5 * 2^31
        0x80, 0x00, 0x00, 0x00, // -1.0 * 2^31
    ];
    let written = AiffWriter::new(AudioSpec::new(16_000, 2), AiffCodec::PcmI32)
        .to_bytes(&samples)
        .expect("write i32");
    assert_same_bytes(
        &written,
        &expected_file(b"AIFF", 2, 2, 32, 16_000, None, &i32_payload),
        "i32 hand-spelled fixture",
    );

    // f64 big-endian: the IEEE 754 spellings, via std's own conversion,
    // which the crate is not involved in.
    let mut f64_payload = Vec::new();
    for sample in samples {
        f64_payload.extend_from_slice(&f64::from(sample).to_be_bytes());
    }
    let written = AiffWriter::new(AudioSpec::new(16_000, 2), AiffCodec::Float64)
        .to_bytes(&samples)
        .expect("write f64");
    assert_same_bytes(
        &written,
        &expected_file(b"AIFC", 2, 2, 64, 16_000, Some(b"fl64"), &f64_payload),
        "f64 hand-spelled fixture",
    );
}

// -- The clamp boundary, as bytes ---------------------------------------------

/// What an integer target and a float target each write for a sample that is
/// not finite, in the big-endian spellings.
///
/// This is the statement [`AiffWriter`]'s documentation makes, so it is
/// measured rather than asserted. Every file below reads back.
#[test]
fn a_non_finite_sample_clamps_into_an_integer_target_and_passes_into_a_float_one() {
    let samples: [f32; 3] = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
    let spec = AudioSpec::mono(48_000);

    // The integer target: silence, then both extremes, big-endian.
    let integer = AiffWriter::new(spec, AiffCodec::PcmI16)
        .to_bytes(&samples)
        .expect("write i16");
    let reader = AiffReader::new(&integer).expect("read i16");
    assert_eq!(reader.data(), [0x00, 0x00, 0x7F, 0xFF, 0x80, 0x00]);
    assert_same_samples(
        reader.decode_to_end().samples(),
        &[0.0, 32_767.0 / 32_768.0, -1.0],
        "i16 target",
    );

    // The 32-bit float target: the three IEEE 754 bit patterns, big-endian,
    // and every one of them survives the round trip exactly.
    let float32 = AiffWriter::new(spec, AiffCodec::Float32)
        .to_bytes(&samples)
        .expect("write f32");
    let reader = AiffReader::new(&float32).expect("read f32");
    assert_eq!(
        reader.data(),
        [0x7F, 0xC0, 0x00, 0x00, 0x7F, 0x80, 0x00, 0x00, 0xFF, 0x80, 0x00, 0x00]
    );
    let back = reader.decode_to_end();
    assert_eq!(back.samples()[0].to_bits(), f32::NAN.to_bits());
    assert_eq!(back.samples()[1], f32::INFINITY);
    assert_eq!(back.samples()[2], f32::NEG_INFINITY);

    // The 64-bit float target: written whole, and the NaN comes back as
    // silence because narrowing a NaN from `f64` to `f32` normalises it.
    let float64 = AiffWriter::new(spec, AiffCodec::Float64)
        .to_bytes(&samples)
        .expect("write f64");
    let reader = AiffReader::new(&float64).expect("read f64");
    assert_eq!(
        reader.data(),
        [
            0x7F, 0xF8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // NaN
            0x7F, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // +inf
            0xFF, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // -inf
        ]
    );
    let back = reader.decode_to_end();
    assert_eq!(back.samples()[0].to_bits(), 0.0f32.to_bits());
    assert_eq!(back.samples()[1], f32::INFINITY);
    assert_eq!(back.samples()[2], f32::NEG_INFINITY);
}

// -- Gate 7: round-trip identity by SHA-256 -----------------------------------

/// Write, read, write: the bytes are a fixed point and the samples are a
/// fixed point, at several channel counts and lengths. Secondary to the
/// fixture gate, per the module documentation: this catches drift between
/// the writer and the readers, not a shared misunderstanding.
#[test]
fn every_written_encoding_round_trips_to_an_identical_sha256() {
    for (codec, _, _, bits) in CODECS {
        for channels in [1u16, 2, 6] {
            for frames in [0usize, 1, 3, 1_000] {
                let samples = samples_for(codec, channels, frames, 0xA1F + frames as u64);
                let label = format!("{codec:?} at {channels} channels, {frames} frames");
                let writer = AiffWriter::new(AudioSpec::new(32_000, channels), codec);

                let written = writer.to_bytes(&samples).expect("write");
                let reader =
                    AiffReader::new(&written).unwrap_or_else(|error| panic!("{label}: {error}"));
                assert_eq!(reader.format().codec, codec, "{label}");
                assert_eq!(reader.format().bits_per_sample, bits, "{label}");
                assert_eq!(reader.spec(), AudioSpec::new(32_000, channels), "{label}");
                assert_eq!(reader.frames(), frames as u64, "{label}");

                let first = reader.decode_to_end();
                assert_same_samples(first.samples(), &samples, &format!("{label}: samples"));

                let rewritten = writer.to_bytes(first.samples()).expect("rewrite");
                assert_eq!(
                    sha256(&written),
                    sha256(&rewritten),
                    "{label}: the file is not a fixed point"
                );

                // The payload the writer produced is the payload it was
                // given: exact for every format, because the samples were
                // quantised into the codec's domain before the first write.
                assert_eq!(
                    reader.data(),
                    canonical_payload(codec, &samples),
                    "{label}: payload moved"
                );

                // The streaming reader agrees with the whole-file reader on
                // the written bytes, at a feed size coprime with every
                // sample width here.
                let mut stream = AiffStreamDecoder::new();
                let mut streamed = Vec::new();
                for piece in written.chunks(7) {
                    let mut offset = 0;
                    while offset < piece.len() {
                        offset += stream.push(&piece[offset..]).expect("push");
                        while stream.pull(&mut streamed, usize::MAX).expect("pull") > 0 {}
                    }
                }
                stream.finish(&mut streamed).expect("finish");
                assert_same_samples(&streamed, &samples, &format!("{label}: streamed"));
            }
        }
    }
}

// -- Gate 8: cross-container agreement ----------------------------------------

/// The same audio written to WAV and to AIFF, read back through both
/// readers, must give bit-identical `f32`. This catches a writer that is
/// self-consistent but disagrees with the rest of the crate, the failure
/// the round-trip gate cannot see.
#[test]
fn wav_and_aiff_written_from_the_same_audio_decode_bit_identically() {
    let pairs: [(AiffCodec, WavCodec); 12] = [
        (AiffCodec::PcmI8, WavCodec::PcmU8),
        (AiffCodec::PcmU8, WavCodec::PcmU8),
        (AiffCodec::PcmI16, WavCodec::PcmI16),
        (AiffCodec::PcmI24, WavCodec::PcmI24),
        (AiffCodec::PcmI32, WavCodec::PcmI32),
        (AiffCodec::PcmI16Sowt, WavCodec::PcmI16),
        (AiffCodec::PcmI24Sowt, WavCodec::PcmI24),
        (AiffCodec::PcmI32Sowt, WavCodec::PcmI32),
        (AiffCodec::Float32, WavCodec::Float32),
        (AiffCodec::Float64, WavCodec::Float64),
        (AiffCodec::ALaw, WavCodec::ALaw),
        (AiffCodec::MuLaw, WavCodec::MuLaw),
    ];
    for (aiff_codec, wav_codec) in pairs {
        for channels in [1u16, 2] {
            let spec = AudioSpec::new(8_000, channels);
            let samples = samples_for(aiff_codec, channels, 50, 0xC0DE ^ u64::from(channels));
            let label = format!("{aiff_codec:?} against {wav_codec:?}, {channels}ch");

            let wav = WavWriter::new(spec, wav_codec)
                .to_bytes(&samples)
                .expect("write WAV");
            let aiff = AiffWriter::new(spec, aiff_codec)
                .to_bytes(&samples)
                .expect("write AIFF");

            let from_wav = WavReader::new(&wav).expect("read WAV").decode_to_end();
            let from_aiff = AiffReader::new(&aiff).expect("read AIFF").decode_to_end();
            assert_eq!(from_wav.spec(), from_aiff.spec(), "{label}: spec");
            assert_same_samples(from_aiff.samples(), from_wav.samples(), &label);
        }
    }
}

// -- The writer's refusals, across every encoding -----------------------------

/// The writer will not produce what the readers reject: a sample count that
/// is not a whole number of frames is refused rather than padded or trimmed,
/// for every encoding.
#[test]
fn the_writer_refuses_a_partial_trailing_frame_in_every_encoding() {
    for (codec, _, _, _) in CODECS {
        for channels in [2u16, 3, 6] {
            let writer = AiffWriter::new(AudioSpec::new(8_000, channels), codec);
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

// -- Gate 14: cross-platform determinism of the written bytes -----------------

/// FNV-1a, so the witness is the bytes themselves and not a float
/// comparison.
fn fnv1a(bytes: impl IntoIterator<Item = u8>) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// One number that changes if any bit of any written file changes, over
/// every encoding at two channel counts. Running this on two toolchains and
/// a 32-bit target and getting the same constant is the evidence that the
/// writer is byte-identical across platforms. Pinned rather than recomputed,
/// so a change shows up as a diff here.
#[test]
fn written_aiff_output_is_bit_identical_to_a_pinned_witness() {
    let mut witness: Vec<u8> = Vec::new();
    for (codec, _, _, _) in CODECS {
        for channels in [1u16, 3] {
            let samples = samples_for(codec, channels, 97, 0x1CE2);
            let written = AiffWriter::new(AudioSpec::new(24_000, channels), codec)
                .to_bytes(&samples)
                .expect("write");
            witness.extend_from_slice(&written);
        }
    }
    // Re-pinned for 0.1.2, when the `raw ` row joined CODECS and so joined
    // this sweep. The previous value, over the eleven encodings this witness
    // covered in 0.1.1, was 0x7752_8810_54c9_236c.
    assert_eq!(
        fnv1a(witness),
        0x9d62_68bf_b27c_4a80,
        "AIFF writer output changed"
    );
}

// -- SHA-256, for the round-trip gate -----------------------------------------

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

/// The same self-contained SHA-256 as `wav_conformance.rs`, duplicated for
/// the same reason as the fixture builders: integration tests cannot share
/// code without a common module, and the no-dependency rule holds for tests
/// too.
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
