#![forbid(unsafe_code)]
//! AIFF conformance: the dimensions an AIFF or AIFF-C file varies along,
//! crossed.
//!
//! # The dimensions, enumerated before any test was written
//!
//! The discipline is the one recorded at the top of `wav_conformance.rs`: a
//! test's coverage is bounded by the dimensions its inputs vary along, not by
//! how many inputs it uses, so the dimensions come first and the gates are
//! built to cross them.
//!
//! 1. form type: `AIFF` and `AIFC`
//! 2. compression type: `NONE`, `twos`, `sowt`, `fl32`, `fl64`, `alaw`,
//!    `ulaw`, and unsupported ones
//! 3. bits per sample: 8, 16, 24, 32, 64
//! 4. channel count: 1, 2, more than 2, and more than eight
//! 5. `SSND` `offset`: zero and non-zero
//! 6. `SSND` `blockSize`: zero and non-zero
//! 7. chunk ordering: `COMM` before and after `SSND`, `FVER` first and
//!    between the known chunks
//! 8. unknown chunks: before, between and after the known ones
//! 9. chunk length parity: even, and odd with its pad byte
//! 10. `numSampleFrames` against actual `SSND` length: matching, under, over
//! 11. `compressionName` length: absent, zero, odd and even
//! 12. total input length: small, and past the 65,536-sample ready limit
//! 13. feed chunk size, on the streaming path
//! 14. sample rate values: the common integers, and one with a non-zero
//!     fractional part
//!
//! [`the_dimension_matrix`] crosses 1 through 9 and 11 in one product,
//! rotating 13 and 14 through it; [`the_streaming_path_matches_the_whole_file_path`]
//! crosses 12 with 13 explicitly, because holding total length small while
//! varying feed size is exactly the blind spot step 3's control found;
//! [`comm_and_ssnd_must_agree_on_the_frame_count`] is 10 in all three cases.
//!
//! # References are built here, not taken from the writer
//!
//! Every file in this suite is assembled byte by byte from the format
//! definitions, and the 80-bit sample rate is *encoded* here by an
//! implementation independent of the crate's parser, so a shared misreading
//! cannot cancel out. These hand-built files predate `AiffWriter` and are
//! deliberately never rebuilt through it: they are the independent oracle
//! the writer itself is gated against, byte for byte, in
//! `aiff_write_conformance.rs`.
//!
//! The strongest anchor is cross-container agreement: the same audio carried
//! by a WAV file (written and read by machinery two steps proven) and by a
//! hand-built AIFF must decode to bit-identical `f32`. Round-trip identity
//! through `AiffWriter` lives in `aiff_write_conformance.rs` as a secondary
//! gate; the agreement gate here is the one whose reference never touches
//! AIFF code.

use decibri_decode::{
    AiffCodec, AiffForm, AiffReader, AiffStreamDecoder, AudioSpec, DecodeError, FourCc,
    StreamSource, WavCodec, WavReader, WavWriter,
};

// -- The reference builder ----------------------------------------------------

/// A big-endian chunk: identifier, declared size, body, and the pad byte when
/// the body is odd.
fn chunk(id: &[u8; 4], body: &[u8]) -> Vec<u8> {
    chunk_declaring(id, body.len() as u32, body)
}

/// A chunk whose size field says `declared` whatever the body actually is.
fn chunk_declaring(id: &[u8; 4], declared: u32, body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(body.len() + 9);
    out.extend_from_slice(id);
    out.extend_from_slice(&declared.to_be_bytes());
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
    out.extend_from_slice(&(body.len() as u32).to_be_bytes());
    out.extend_from_slice(body);
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

/// An integer sample rate as the 80-bit extended float `COMM` stores.
///
/// An independent implementation of the encoding, normalising to the explicit
/// integer bit and biasing the exponent by 16383, written from the format
/// definition rather than shared with the crate's parser, so a mistake in one
/// is not mirrored in the other.
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

/// How the AIFF-C `compressionName` pascal string is spelled, when at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pstring {
    /// No pascal string at all: a 22-byte `COMM`, which nonconforming writers
    /// produce and the reader tolerates.
    Absent,
    /// A zero-length name with its even-total pad: two bytes.
    Zero,
    /// A zero-length name without the pad: one byte, making the chunk odd.
    ZeroUnpadded,
    /// A three-character name, whose 1 + 3 bytes are already even.
    OddName,
    /// A four-character name plus its pad: six bytes.
    EvenName,
}

const PSTRINGS: [Pstring; 5] = [
    Pstring::Absent,
    Pstring::Zero,
    Pstring::ZeroUnpadded,
    Pstring::OddName,
    Pstring::EvenName,
];

impl Pstring {
    fn bytes(self) -> Vec<u8> {
        match self {
            Self::Absent => Vec::new(),
            Self::Zero => vec![0, 0],
            Self::ZeroUnpadded => vec![0],
            Self::OddName => vec![3, b'o', b'd', b'd'],
            Self::EvenName => vec![4, b'n', b'a', b'm', b'e', 0],
        }
    }
}

/// A `COMM` body, written from the field order in the two specifications.
///
/// `compression` is `None` for a plain `AIFF` form, which has no such field,
/// and `Some` with a pascal-string spelling for `AIFC`.
fn comm(
    channels: u16,
    frames: u32,
    bits: u16,
    rate: [u8; 10],
    compression: Option<(&[u8; 4], Pstring)>,
) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(&channels.to_be_bytes());
    body.extend_from_slice(&frames.to_be_bytes());
    body.extend_from_slice(&bits.to_be_bytes());
    body.extend_from_slice(&rate);
    if let Some((four_cc, pstring)) = compression {
        body.extend_from_slice(four_cc);
        body.extend_from_slice(&pstring.bytes());
    }
    body
}

/// An `SSND` body: the `offset` and `blockSize` fields, `offset` bytes of
/// alignment fill that must never be decoded as audio, then the sample data.
fn ssnd(offset: u32, block_size: u32, data: &[u8]) -> Vec<u8> {
    let mut body = Vec::with_capacity(8 + offset as usize + data.len());
    body.extend_from_slice(&offset.to_be_bytes());
    body.extend_from_slice(&block_size.to_be_bytes());
    // 0xEE fill: loud, recognisable garbage if it ever leaks into the audio.
    body.extend(std::iter::repeat_n(0xEE, offset as usize));
    body.extend_from_slice(data);
    body
}

/// AIFF-C's `FVER` chunk with the one version timestamp that has ever
/// existed.
fn fver() -> Vec<u8> {
    chunk(b"FVER", &0xA280_5140u32.to_be_bytes())
}

/// The conventional file: `COMM` then `SSND`, truthful everywhere.
fn aiff_file(
    form_type: &[u8; 4],
    channels: u16,
    frames: u32,
    bits: u16,
    rate: u32,
    compression: Option<(&[u8; 4], Pstring)>,
    payload: &[u8],
) -> Vec<u8> {
    form(
        form_type,
        &[
            chunk(
                b"COMM",
                &comm(channels, frames, bits, extended_rate(rate), compression),
            ),
            chunk(b"SSND", &ssnd(0, 0, payload)),
        ],
    )
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

/// A payload of `frames` whole frames for `codec` at `channels`.
fn payload_for(codec: AiffCodec, channels: u16, frames: usize, seed: u64) -> Vec<u8> {
    payload_bytes(
        frames * usize::from(channels) * codec.bytes_per_sample(),
        seed,
    )
}

/// What the payload decodes to, through the codec layer and not through the
/// container: the container gates check the container and nothing else. The
/// codec layer is anchored separately, by its own exhaustive gates and by
/// cross-container agreement against WAV's proven chain.
fn expected_samples(codec: AiffCodec, payload: &[u8]) -> Vec<f32> {
    let mut samples = Vec::new();
    codec.decode(payload, &mut samples);
    samples
}

/// Compares samples by bit pattern, for the reasons recorded on the WAV
/// suite: NaN differs from itself under `==`, and `+0.0 == -0.0` while the
/// bytes differ. Bit equality is the claim this crate makes, so it is the
/// claim the tests state.
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

/// One encoding row of the matrix: form type, compression (`None` for a
/// plain AIFF form, which has no field), declared bits, and what it must
/// resolve to.
type CodecRow = (&'static [u8; 4], Option<&'static [u8; 4]>, u16, AiffCodec);

/// Every encoding row the matrix runs.
const CODECS: [CodecRow; 14] = [
    (b"AIFF", None, 8, AiffCodec::PcmI8),
    (b"AIFC", Some(b"raw "), 8, AiffCodec::PcmU8),
    (b"AIFF", None, 16, AiffCodec::PcmI16),
    (b"AIFF", None, 24, AiffCodec::PcmI24),
    (b"AIFF", None, 32, AiffCodec::PcmI32),
    (b"AIFC", Some(b"NONE"), 16, AiffCodec::PcmI16),
    (b"AIFC", Some(b"twos"), 16, AiffCodec::PcmI16),
    (b"AIFC", Some(b"sowt"), 8, AiffCodec::PcmI8),
    (b"AIFC", Some(b"sowt"), 16, AiffCodec::PcmI16Sowt),
    (b"AIFC", Some(b"sowt"), 24, AiffCodec::PcmI24Sowt),
    (b"AIFC", Some(b"sowt"), 32, AiffCodec::PcmI32Sowt),
    (b"AIFC", Some(b"fl32"), 32, AiffCodec::Float32),
    (b"AIFC", Some(b"FL64"), 64, AiffCodec::Float64),
    (b"AIFC", Some(b"ulaw"), 8, AiffCodec::MuLaw),
];

// -- Driving the streaming reader ---------------------------------------------

/// How a caller drives [`AiffStreamDecoder`]: the same two modes as the WAV
/// suite, for the same reason: the greedy drive is the one that reaches the
/// short-return path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Drive {
    Interleaved,
    Greedy,
}

fn stream_decode(
    bytes: &[u8],
    piece: usize,
    drive: Drive,
) -> Result<(Option<AudioSpec>, Vec<f32>), DecodeError> {
    let mut stream = AiffStreamDecoder::new();
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

// -- Gate 5: cross-container agreement ----------------------------------------

/// How a WAV payload's bytes respell into an AIFF payload carrying the same
/// audio.
#[derive(Debug, Clone, Copy)]
enum Respell {
    /// Identical bytes: `sowt` and G.711.
    Same,
    /// The sign bit flips: WAV's unsigned 8-bit against AIFF's signed.
    Xor80,
    /// Each sample's bytes reverse: little-endian to big-endian.
    Swap(usize),
}

fn respell(bytes: &[u8], how: Respell) -> Vec<u8> {
    match how {
        Respell::Same => bytes.to_vec(),
        Respell::Xor80 => bytes.iter().map(|byte| byte ^ 0x80).collect(),
        Respell::Swap(width) => bytes
            .chunks_exact(width)
            .flat_map(|sample| sample.iter().rev().copied())
            .collect(),
    }
}

/// The gate that replaces round-trip identity: known samples written to WAV
/// by the proven writer and read back, against a hand-built AIFF carrying the
/// same audio, must decode to **bit-identical** `f32`, for every bit depth,
/// for both byte orders where AIFF has both, and for both G.711 laws.
///
/// A systematic error in AIFF's byte order, signedness, scaling or dispatch
/// cannot pass this, because the WAV half of the comparison never went
/// through any AIFF code.
#[test]
fn wav_and_aiff_carrying_the_same_audio_decode_bit_identically() {
    let cases: [(WavCodec, &[u8; 4], u16, Respell); 13] = [
        (WavCodec::PcmU8, b"NONE", 8, Respell::Xor80),
        (WavCodec::PcmU8, b"sowt", 8, Respell::Xor80),
        // `raw ` is WAV's own unsigned convention, so the bytes do not move.
        (WavCodec::PcmU8, b"raw ", 8, Respell::Same),
        (WavCodec::PcmI16, b"NONE", 16, Respell::Swap(2)),
        (WavCodec::PcmI16, b"twos", 16, Respell::Swap(2)),
        (WavCodec::PcmI16, b"sowt", 16, Respell::Same),
        (WavCodec::PcmI24, b"NONE", 24, Respell::Swap(3)),
        (WavCodec::PcmI24, b"sowt", 24, Respell::Same),
        (WavCodec::PcmI32, b"NONE", 32, Respell::Swap(4)),
        (WavCodec::PcmI32, b"sowt", 32, Respell::Same),
        (WavCodec::Float32, b"fl32", 32, Respell::Swap(4)),
        (WavCodec::Float64, b"fl64", 64, Respell::Swap(8)),
        (WavCodec::ALaw, b"alaw", 16, Respell::Same),
    ];
    for (wav_codec, compression, bits, how) in cases {
        for channels in [1u16, 2] {
            cross_check(wav_codec, compression, bits, how, channels);
        }
    }
    // Both laws, not just one: mu-law separately, and at the wire width some
    // writers declare as well as the decoded width above.
    for bits in [8u16, 16] {
        cross_check(WavCodec::MuLaw, b"ulaw", bits, Respell::Same, 1);
    }
    // And the plain AIFF form for the four integer widths, which carries the
    // same audio with no compression field at all.
    for (wav_codec, bits, how) in [
        (WavCodec::PcmU8, 8u16, Respell::Xor80),
        (WavCodec::PcmI16, 16, Respell::Swap(2)),
        (WavCodec::PcmI24, 24, Respell::Swap(3)),
        (WavCodec::PcmI32, 32, Respell::Swap(4)),
    ] {
        let seed = 0xC0DE + u64::from(bits);
        let source = payload_bytes(60 * wav_codec.bytes_per_sample(), seed);
        let mut samples = Vec::new();
        wav_codec.decode(&source, &mut samples);

        let written = WavWriter::new(AudioSpec::new(16_000, 2), wav_codec)
            .to_bytes(&samples)
            .expect("write");
        let wav = WavReader::new(&written).expect("read back");
        let decoded_wav = wav.decode_to_end();

        let aiff_bytes = aiff_file(
            b"AIFF",
            2,
            30,
            bits,
            16_000,
            None,
            &respell(wav.data(), how),
        );
        let aiff = AiffReader::new(&aiff_bytes).expect("AIFF form");
        assert_eq!(aiff.format().form, AiffForm::Aiff);
        assert_same_samples(
            aiff.decode_to_end().samples(),
            decoded_wav.samples(),
            &format!("AIFF form at {bits} bits"),
        );
    }
}

fn cross_check(wav_codec: WavCodec, compression: &[u8; 4], bits: u16, how: Respell, channels: u16) {
    let frames = 50usize;
    let seed = 0xA1FF ^ (u64::from(bits) << 8) ^ u64::from(channels);
    let source = payload_bytes(
        frames * usize::from(channels) * wav_codec.bytes_per_sample(),
        seed,
    );
    let mut samples = Vec::new();
    wav_codec.decode(&source, &mut samples);

    // The WAV half: written by the proven writer, read back by the proven
    // reader. Its payload bytes are the reference the AIFF file is built
    // from, so both containers carry the same audio by construction.
    let written = WavWriter::new(AudioSpec::new(8_000, channels), wav_codec)
        .to_bytes(&samples)
        .expect("write");
    let wav = WavReader::new(&written).expect("read back");
    let decoded_wav = wav.decode_to_end();

    let aiff_bytes = aiff_file(
        b"AIFC",
        channels,
        frames as u32,
        bits,
        8_000,
        Some((compression, Pstring::Zero)),
        &respell(wav.data(), how),
    );
    let label = format!(
        "{wav_codec:?} against {} at {bits} bits, {channels}ch",
        FourCc(*compression)
    );
    let aiff = AiffReader::new(&aiff_bytes).unwrap_or_else(|error| panic!("{label}: {error}"));
    assert_same_samples(
        aiff.decode_to_end().samples(),
        decoded_wav.samples(),
        &label,
    );

    // The streaming path agrees too, so the gate anchors both readers.
    let (spec, streamed) = stream_decode(&aiff_bytes, 11, Drive::Interleaved)
        .unwrap_or_else(|error| panic!("{label} streamed: {error}"));
    assert_eq!(spec, Some(AudioSpec::new(8_000, channels)), "{label}");
    assert_same_samples(
        &streamed,
        decoded_wav.samples(),
        &format!("{label} streamed"),
    );
}

// -- `raw `: the one AIFF-C encoding whose eight-bit samples are unsigned ------

/// A complete AIFF-C `raw ` file spelled out one byte at a time from the
/// AIFF-C specification, with every field labelled.
///
/// The literal is the point. The builders above are an independent
/// transcription of the format, but they are still a program, and a
/// misunderstanding in `comm` would be shared by every fixture they produce.
/// This array shares nothing with them: it is what the specification says a
/// six-frame mono `raw ` file at 8000 Hz looks like, written out.
const HAND_BUILT_RAW_FILE: [u8; 79] = [
    // FORM header: the magic, the size of everything after it, the form type.
    b'F', b'O', b'R', b'M', //
    0x00, 0x00, 0x00, 0x47, // 71 bytes follow
    b'A', b'I', b'F', b'C', //
    // FVER: the one version timestamp AIFF-C has ever had.
    b'F', b'V', b'E', b'R', //
    0x00, 0x00, 0x00, 0x04, //
    0xA2, 0x80, 0x51, 0x40, //
    // COMM: 24 bytes of body.
    b'C', b'O', b'M', b'M', //
    0x00, 0x00, 0x00, 0x18, //
    0x00, 0x01, // numChannels = 1
    0x00, 0x00, 0x00, 0x06, // numSampleFrames = 6
    0x00, 0x08, // sampleSize = 8
    // sampleRate, 80-bit extended: exponent 0x400B is 16383 + 12, and the
    // significand 0xFA00... is 8000 << 51, so the value is exactly 8000.
    0x40, 0x0B, 0xFA, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, //
    b'r', b'a', b'w', b' ', // compressionType
    0x00, 0x00, // a zero-length compressionName, and its pad
    // SSND: 14 bytes of body, so a pad byte follows the chunk.
    b'S', b'S', b'N', b'D', //
    0x00, 0x00, 0x00, 0x0E, //
    0x00, 0x00, 0x00, 0x00, // offset
    0x00, 0x00, 0x00, 0x00, // blockSize
    // The audio: offset binary, so 0x80 is silence and 0x00 is full negative.
    0x80, 0xC0, 0x00, 0xFF, 0x40, 0x7F, //
    0x00, // the pad byte for the odd SSND body
];

/// What [`HAND_BUILT_RAW_FILE`] holds, read off the offset-binary definition:
/// subtract 128, divide by 128.
const HAND_BUILT_RAW_SAMPLES: [f32; 6] = [
    0.0,           // 0x80 is silence, not -1.0
    0.5,           // 0xC0 is +64
    -1.0,          // 0x00 is the most negative value, not silence
    127.0 / 128.0, // 0xFF is the most positive
    -0.5,          // 0x40 is -64
    -1.0 / 128.0,  // 0x7F is -1
];

/// The byte oracle for `raw `: a file this crate had no part in building
/// decodes to the samples the specification says it holds.
///
/// A reader that carried AIFF's signed convention into `raw ` decodes 0x80
/// to -1.0 rather than 0.0 and every sample lands half full scale away, which
/// is the failure this gate exists to catch and the one the round trip
/// through this crate's own writer cannot see.
#[test]
fn a_hand_built_raw_file_decodes_to_the_samples_the_specification_states() {
    let reader = AiffReader::new(&HAND_BUILT_RAW_FILE).expect("hand-built raw file");
    assert_eq!(reader.format().form, AiffForm::Aifc);
    assert_eq!(reader.format().compression, FourCc(*b"raw "));
    assert_eq!(reader.format().codec, AiffCodec::PcmU8);
    assert_eq!(reader.format().bits_per_sample, 8);
    assert_eq!(reader.spec(), AudioSpec::mono(8_000));
    assert_eq!(reader.frames(), 6);
    assert_eq!(reader.data(), [0x80, 0xC0, 0x00, 0xFF, 0x40, 0x7F]);
    assert_same_samples(
        reader.decode_to_end().samples(),
        &HAND_BUILT_RAW_SAMPLES,
        "hand-built raw file",
    );

    // The streaming reader reaches the same audio from the same bytes.
    let (spec, streamed) = stream_decode(&HAND_BUILT_RAW_FILE, 3, Drive::Greedy).expect("stream");
    assert_eq!(spec, Some(AudioSpec::mono(8_000)));
    assert_same_samples(&streamed, &HAND_BUILT_RAW_SAMPLES, "hand-built raw stream");

    // The same payload under `NONE` is the signed reading of the same bytes,
    // half full scale away from every value above. Both spellings are
    // asserted, so a reader that resolved `raw ` to the signed format would
    // make these two agree and fail here.
    let as_signed = aiff_file(
        b"AIFF",
        1,
        6,
        8,
        8_000,
        None,
        &[0x80, 0xC0, 0x00, 0xFF, 0x40, 0x7F],
    );
    let signed = AiffReader::new(&as_signed).expect("signed").decode_to_end();
    let expected_signed: [f32; 6] = [-1.0, -0.5, 0.0, -1.0 / 128.0, 0.5, 127.0 / 128.0];
    assert_same_samples(signed.samples(), &expected_signed, "the signed reading");
}

/// Every pascal-string spelling of a `raw ` `COMM`, since the compression
/// field is the only place the encoding is named.
#[test]
fn raw_resolves_under_every_compression_name_spelling() {
    let payload: [u8; 6] = [0x80, 0xC0, 0x00, 0xFF, 0x40, 0x7F];
    for pstring in PSTRINGS {
        let bytes = aiff_file(b"AIFC", 1, 6, 8, 8_000, Some((b"raw ", pstring)), &payload);
        let reader = AiffReader::new(&bytes).unwrap_or_else(|e| panic!("{pstring:?}: {e}"));
        assert_eq!(reader.format().codec, AiffCodec::PcmU8, "{pstring:?}");
        assert_same_samples(
            reader.decode_to_end().samples(),
            &HAND_BUILT_RAW_SAMPLES,
            &format!("{pstring:?}"),
        );
    }
}

/// `raw ` names one width. At any other `sampleSize` the file is a width
/// rejection naming the four-CC and the width, never a silent accept at some
/// other stride.
#[test]
fn raw_at_a_width_other_than_eight_is_rejected_on_the_width() {
    let payload = payload_bytes(48, 0x5A17);
    for bits in [1u16, 4, 12, 16, 20, 24, 32, 64] {
        let bytes = aiff_file(
            b"AIFC",
            1,
            6,
            bits,
            8_000,
            Some((b"raw ", Pstring::Zero)),
            &payload[..6 * (usize::from(bits).div_ceil(8))],
        );
        let error = AiffReader::new(&bytes).expect_err("must reject");
        assert!(
            matches!(
                &error,
                DecodeError::UnsupportedSampleFormat { format, bits_per_sample }
                    if *bits_per_sample == bits && format.to_string().contains("raw")
            ),
            "raw at {bits} bits: unexpected error: {error}"
        );
    }
}

/// A plain `AIFF` form has no compression field, so the four bytes that would
/// hold one in an `AIFF-C` file are not read as one.
///
/// This is what confines `raw ` to `AIFF-C`. A file declaring `FORM`/`AIFF`
/// with `raw ` sitting where `AIFC` keeps its compressionType resolves as the
/// form's implicit `NONE`, signed, and never as unsigned: a reader that
/// consulted those bytes regardless of the form would accept an encoding the
/// format cannot express there, and would decode the same file two different
/// ways depending on a field its form does not have.
#[test]
fn raw_bytes_in_a_plain_aiff_form_are_not_a_compression_field() {
    let payload: [u8; 6] = [0x80, 0xC0, 0x00, 0xFF, 0x40, 0x7F];
    // A plain AIFF whose COMM body is the four fixed fields followed by the
    // bytes `raw ` and a zero-length pascal string: the exact byte layout an
    // AIFC COMM has, under the form that does not define it.
    let mut body = comm(1, 6, 8, extended_rate(8_000), None);
    body.extend_from_slice(b"raw ");
    body.extend_from_slice(&[0, 0]);
    let bytes = form(
        b"AIFF",
        &[chunk(b"COMM", &body), chunk(b"SSND", &ssnd(0, 0, &payload))],
    );

    let reader = AiffReader::new(&bytes).expect("a plain AIFF with a long COMM still reads");
    assert_eq!(reader.format().form, AiffForm::Aiff);
    assert_eq!(
        reader.format().compression,
        FourCc(*b"NONE"),
        "a plain AIFF form reported a compression type it cannot carry"
    );
    assert_eq!(
        reader.format().codec,
        AiffCodec::PcmI8,
        "`raw ` was honoured in a form that has no compression field"
    );
    let expected: [f32; 6] = [-1.0, -0.5, 0.0, -1.0 / 128.0, 0.5, 127.0 / 128.0];
    assert_same_samples(
        reader.decode_to_end().samples(),
        &expected,
        "plain AIFF with raw bytes in the COMM",
    );
}

// -- Gate 6: eight-bit signedness, named and explicit -------------------------

/// The single most likely defect in the step, so it gets its own test rather
/// than being folded into the agreement gate. Every expected value is written
/// from the AIFF specification's signed convention; a reader that treated
/// AIFF's 8-bit as unsigned decodes `0x00` to `-1.0` instead of `0.0` and
/// fails on the first sample.
#[test]
fn aiff_eight_bit_is_signed_and_fails_if_read_as_unsigned() {
    let payload: [u8; 6] = [0x00, 0x40, 0x80, 0x7F, 0xC0, 0xFF];
    let expected: [f32; 6] = [
        0.0,           // 0x00 is silence, not -1.0
        0.5,           // +64
        -1.0,          // 0x80 is the most negative value, not the midpoint
        127.0 / 128.0, // the most positive
        -0.5,          // -64
        -1.0 / 128.0,  // -1
    ];

    // Through both form types and all three compression spellings that carry
    // 8-bit linear PCM.
    let files: [Vec<u8>; 4] = [
        aiff_file(b"AIFF", 1, 6, 8, 8_000, None, &payload),
        aiff_file(
            b"AIFC",
            1,
            6,
            8,
            8_000,
            Some((b"NONE", Pstring::Zero)),
            &payload,
        ),
        aiff_file(
            b"AIFC",
            1,
            6,
            8,
            8_000,
            Some((b"twos", Pstring::Zero)),
            &payload,
        ),
        aiff_file(
            b"AIFC",
            1,
            6,
            8,
            8_000,
            Some((b"sowt", Pstring::Zero)),
            &payload,
        ),
    ];
    for (index, file) in files.iter().enumerate() {
        let reader = AiffReader::new(file).expect("8-bit AIFF");
        assert_same_samples(
            reader.decode_to_end().samples(),
            &expected,
            &format!("8-bit spelling {index}"),
        );
        let (_, streamed) = stream_decode(file, 3, Drive::Interleaved).expect("stream");
        assert_same_samples(
            &streamed,
            &expected,
            &format!("8-bit spelling {index}, streamed"),
        );
    }

    // And the two containers' 8-bit conventions are inverses, byte for byte:
    // the same payload through WAV decodes half full scale away.
    let wav_file = WavWriter::new(AudioSpec::mono(8_000), WavCodec::PcmU8)
        .to_bytes(&expected)
        .expect("write");
    let wav = WavReader::new(&wav_file).expect("read");
    assert_eq!(
        wav.data(),
        payload
            .iter()
            .map(|byte| byte ^ 0x80)
            .collect::<Vec<_>>()
            .as_slice(),
        "the WAV spelling of the same audio is the sign-bit flip of the AIFF one"
    );
}

// -- Gate 8: sowt is little-endian, proven against hand-written bytes ---------

/// The bytes `[0x34, 0x12]` are 0x1234 little-endian and 0x3412 big-endian,
/// and the two values are far apart, so a reader that took the container's
/// byte order as the data's fails loudly here.
#[test]
fn sowt_is_little_endian_and_fails_if_read_big_endian() {
    // 16-bit: 0x1234 = 4660.
    let sowt16 = aiff_file(
        b"AIFC",
        1,
        1,
        16,
        44_100,
        Some((b"sowt", Pstring::Zero)),
        &[0x34, 0x12],
    );
    let reader = AiffReader::new(&sowt16).expect("sowt 16");
    assert_eq!(reader.format().codec, AiffCodec::PcmI16Sowt);
    assert_same_samples(
        reader.decode_to_end().samples(),
        &[4660.0 / 32768.0],
        "sowt 16-bit",
    );
    // The identical bytes under `twos` are the other number: the compression
    // four-CC alone decides, and both directions are asserted so the test
    // cannot be satisfied by ignoring the field.
    let twos16 = aiff_file(
        b"AIFC",
        1,
        1,
        16,
        44_100,
        Some((b"twos", Pstring::Zero)),
        &[0x34, 0x12],
    );
    assert_same_samples(
        AiffReader::new(&twos16)
            .expect("twos 16")
            .decode_to_end()
            .samples(),
        &[0x3412 as f32 / 32768.0],
        "the same bytes under twos",
    );

    // 24-bit: 0x123456 = 1193046.
    let sowt24 = aiff_file(
        b"AIFC",
        1,
        1,
        24,
        44_100,
        Some((b"sowt", Pstring::Zero)),
        &[0x56, 0x34, 0x12],
    );
    assert_same_samples(
        AiffReader::new(&sowt24)
            .expect("sowt 24")
            .decode_to_end()
            .samples(),
        &[1_193_046.0 / 8_388_608.0],
        "sowt 24-bit",
    );

    // 32-bit: 0x12345678, which rounds in f32 exactly as the reference does.
    let sowt32 = aiff_file(
        b"AIFC",
        1,
        1,
        32,
        44_100,
        Some((b"sowt", Pstring::Zero)),
        &[0x78, 0x56, 0x34, 0x12],
    );
    assert_same_samples(
        AiffReader::new(&sowt32)
            .expect("sowt 32")
            .decode_to_end()
            .samples(),
        &[305_419_896_f32 / 2_147_483_648.0],
        "sowt 32-bit",
    );

    // The streaming path reads sowt the same way.
    let (_, streamed) = stream_decode(&sowt16, 1, Drive::Interleaved).expect("stream");
    assert_same_samples(&streamed, &[4660.0 / 32768.0], "sowt 16-bit streamed");
}

// -- Gate 9: the SSND offset and blockSize fields -----------------------------

/// Sample data does not begin at the start of the `SSND` body. Files whose
/// `offset` is non-zero must decode to the same audio as their zero-offset
/// twins, and the 0xEE alignment fill must never leak into the samples.
#[test]
fn a_non_zero_ssnd_offset_moves_the_data_and_nothing_else() {
    let payload = payload_bytes(64, 0x0FF5);
    let expected = expected_samples(AiffCodec::PcmI16, &payload);

    let reference = form(
        b"AIFF",
        &[
            chunk(b"COMM", &comm(2, 16, 16, extended_rate(22_050), None)),
            chunk(b"SSND", &ssnd(0, 0, &payload)),
        ],
    );
    assert_same_samples(
        AiffReader::new(&reference)
            .expect("offset 0")
            .decode_to_end()
            .samples(),
        &expected,
        "offset 0",
    );

    for (offset, block_size) in [(6u32, 0u32), (0, 4), (10, 4), (1, 0), (513, 16)] {
        let bytes = form(
            b"AIFF",
            &[
                chunk(b"COMM", &comm(2, 16, 16, extended_rate(22_050), None)),
                chunk(b"SSND", &ssnd(offset, block_size, &payload)),
            ],
        );
        let label = format!("offset {offset}, blockSize {block_size}");
        let reader = AiffReader::new(&bytes).unwrap_or_else(|error| panic!("{label}: {error}"));
        assert_eq!(reader.frames(), 16, "{label}");
        assert_same_samples(reader.decode_to_end().samples(), &expected, &label);

        // The streaming path skips the same gap, at feed sizes that land
        // inside it.
        for piece in [1usize, 7, 4_096] {
            let (_, streamed) = stream_decode(&bytes, piece, Drive::Greedy)
                .unwrap_or_else(|error| panic!("{label} at {piece}: {error}"));
            assert_same_samples(
                &streamed,
                &expected,
                &format!("{label} streamed at {piece}"),
            );
        }
    }

    // An offset pointing past its own chunk is malformed, not an over-read,
    // including one chosen to wrap 32-bit arithmetic.
    for offset in [57u32, 1_000, u32::MAX, u32::MAX - 7] {
        let bytes = form(
            b"AIFF",
            &[
                chunk(b"COMM", &comm(2, 16, 16, extended_rate(22_050), None)),
                chunk(b"SSND", &ssnd_declaring_offset(offset, &payload)),
            ],
        );
        let error = AiffReader::new(&bytes)
            .err()
            .unwrap_or_else(|| panic!("offset {offset} was accepted"));
        assert!(
            matches!(
                error,
                DecodeError::Malformed { .. } | DecodeError::Truncated { .. }
            ),
            "offset {offset}: unexpected error: {error}"
        );
        assert!(stream_decode(&bytes, 9, Drive::Interleaved).is_err());
    }
}

/// An `SSND` body whose offset field *claims* `offset` without carrying the
/// fill bytes, for the rejection cases.
fn ssnd_declaring_offset(offset: u32, data: &[u8]) -> Vec<u8> {
    let mut body = Vec::with_capacity(8 + data.len());
    body.extend_from_slice(&offset.to_be_bytes());
    body.extend_from_slice(&0u32.to_be_bytes());
    body.extend_from_slice(data);
    body
}

// -- The two sources of truth for length --------------------------------------

/// Dimension 10, in all three cases. `COMM` declares a frame count and `SSND`
/// has a length; matching decodes, and either direction of disagreement is a
/// typed error rather than a silent preference: under is `Truncated` naming
/// both numbers, over is `Malformed`.
#[test]
fn comm_and_ssnd_must_agree_on_the_frame_count() {
    let payload = payload_bytes(40, 0x2A97); // 10 stereo 16-bit frames
    let build = |frames: u32| {
        form(
            b"AIFF",
            &[
                chunk(b"COMM", &comm(2, frames, 16, extended_rate(8_000), None)),
                chunk(b"SSND", &ssnd(0, 0, &payload)),
            ],
        )
    };

    // Matching: decodes on both paths.
    let matching = build(10);
    let reader = AiffReader::new(&matching).expect("matching");
    assert_eq!(reader.frames(), 10);
    let (_, streamed) = stream_decode(&matching, 13, Drive::Interleaved).expect("stream");
    assert_eq!(streamed.len(), 20);

    // COMM claims more than SSND holds: Truncated, with both numbers.
    let under = build(12);
    let error = AiffReader::new(&under).expect_err("under must reject");
    assert!(
        matches!(
            error,
            DecodeError::Truncated {
                expected: 48,
                available: 40
            }
        ),
        "under: unexpected error: {error}"
    );
    let error = stream_decode(&under, 13, Drive::Interleaved).expect_err("stream under");
    assert!(
        matches!(
            error,
            DecodeError::Truncated {
                expected: 48,
                available: 40
            }
        ),
        "streamed under: unexpected error: {error}"
    );

    // COMM claims fewer than SSND holds: Malformed, because the extra bytes
    // are either audio the count disowns or garbage the count conceals, and
    // the reader cannot know which.
    let over = build(8);
    let error = AiffReader::new(&over).expect_err("over must reject");
    assert!(
        matches!(
            error,
            DecodeError::Malformed {
                expected: "an SSND chunk holding exactly numSampleFrames frames",
                ..
            }
        ),
        "over: unexpected error: {error}"
    );
    assert!(stream_decode(&over, 13, Drive::Interleaved).is_err());

    // A partial trailing frame is one spelling of disagreement: 41 bytes is
    // not a whole number of 4-byte frames whatever the count says.
    let ragged = form(
        b"AIFF",
        &[
            chunk(b"COMM", &comm(2, 10, 16, extended_rate(8_000), None)),
            chunk(b"SSND", &ssnd(0, 0, &payload_bytes(41, 0x2A98))),
        ],
    );
    assert!(AiffReader::new(&ragged).is_err());
}

// -- The sample rate, through real files --------------------------------------

/// Rates through the whole reader, not only the parser's unit tests: common
/// integers land exactly, and the field being a *float* is honoured: a rate
/// with a fractional part is parsed correctly and then rejected with a typed
/// error, never rounded into a lie.
#[test]
fn sample_rates_land_exactly_and_a_fractional_rate_is_rejected() {
    let payload = payload_bytes(8, 0x9A7E);
    for rate in [
        1u32, 8_000, 11_025, 16_000, 22_050, 44_100, 48_000, 96_000, 192_000,
    ] {
        let bytes = aiff_file(b"AIFF", 1, 8, 8, rate, None, &payload);
        let reader = AiffReader::new(&bytes).unwrap_or_else(|error| panic!("{rate}: {error}"));
        assert_eq!(reader.spec(), AudioSpec::mono(rate), "{rate}");
        let (spec, _) = stream_decode(&bytes, 5, Drive::Interleaved).expect("stream");
        assert_eq!(spec, Some(AudioSpec::mono(rate)), "{rate} streamed");
    }

    // 11025.5: hand-encoded, since the integer encoder above cannot spell it.
    // 22051 = 0x5623 normalises to 0xAC46... at exponent 16383 + 13 - 1.
    let fractional: [u8; 10] = [0x40, 0x0C, 0xAC, 0x46, 0, 0, 0, 0, 0, 0];
    let bytes = form(
        b"AIFF",
        &[
            chunk(b"COMM", &comm(1, 8, 8, fractional, None)),
            chunk(b"SSND", &ssnd(0, 0, &payload)),
        ],
    );
    for error in [
        AiffReader::new(&bytes).expect_err("whole-file must reject"),
        stream_decode(&bytes, 7, Drive::Interleaved).expect_err("stream must reject"),
    ] {
        assert!(
            matches!(
                error,
                DecodeError::Malformed {
                    expected: "a positive integral sample rate that fits in 32 bits",
                    ..
                }
            ),
            "unexpected error: {error}"
        );
    }
}

// -- Gate 10: malformed input, every named case -------------------------------

/// A representative healthy file with every optional structure present: FVER
/// first, an odd unknown chunk, a compressionName, and a non-zero SSND
/// offset.
fn healthy_file() -> Vec<u8> {
    form(
        b"AIFC",
        &[
            fver(),
            chunk(b"ANNO", b"junk!"),
            chunk(
                b"COMM",
                &comm(
                    2,
                    32,
                    16,
                    extended_rate(48_000),
                    Some((b"sowt", Pstring::OddName)),
                ),
            ),
            chunk(b"SSND", &ssnd(6, 0, &payload_bytes(128, 0x4EA2))),
        ],
    )
}

/// Every malformed input the relay names, each producing a typed error on
/// both paths, never a panic, never an over-read.
#[test]
fn every_malformed_input_produces_a_typed_error() {
    let payload = payload_bytes(64, 0xBAD2);

    // Truncated FORM header: every prefix under twelve bytes.
    let whole = aiff_file(b"AIFF", 1, 64, 8, 8_000, None, &payload);
    for length in 0..12 {
        assert!(
            matches!(
                AiffReader::new(&whole[..length]),
                Err(DecodeError::Truncated { expected: 12, .. })
            ),
            "{length}-byte header was accepted"
        );
    }

    // Truncated COMM: the header is whole, the COMM chunk is not.
    for length in 13..12 + 8 + 18 {
        assert!(
            AiffReader::new(&whole[..length]).is_err(),
            "a file cut at {length} bytes was accepted"
        );
    }

    // Truncated SSND: every cut inside the sample data.
    for cut in 1..payload.len() {
        let bytes = &whole[..whole.len() - cut];
        let error = AiffReader::new(bytes).expect_err("a short SSND must reject");
        assert!(
            matches!(error, DecodeError::Truncated { .. }),
            "cut {cut}: unexpected error: {error}"
        );
    }

    // Missing COMM.
    let no_comm = form(b"AIFF", &[chunk(b"SSND", &ssnd(0, 0, &payload))]);
    assert!(matches!(
        AiffReader::new(&no_comm),
        Err(DecodeError::Malformed {
            expected: "a COMM chunk",
            ..
        })
    ));

    // Missing SSND, with frames declared.
    let no_ssnd = form(
        b"AIFF",
        &[chunk(b"COMM", &comm(1, 64, 8, extended_rate(8_000), None))],
    );
    assert!(matches!(
        AiffReader::new(&no_ssnd),
        Err(DecodeError::Malformed {
            expected: "an SSND chunk",
            ..
        })
    ));

    // SSND declared larger than the file holds: four gigabytes inside a
    // two-hundred-byte file.
    let over = form(
        b"AIFF",
        &[
            chunk(b"COMM", &comm(1, 64, 8, extended_rate(8_000), None)),
            chunk_declaring(b"SSND", 0xFFFF_FF00, &ssnd(0, 0, &payload)),
        ],
    );
    let error = AiffReader::new(&over).expect_err("an over-declared SSND must reject");
    assert!(
        matches!(
            error,
            DecodeError::Truncated {
                expected: 0xFFFF_FF00,
                available
            } if available == 8 + payload.len() as u64
        ),
        "unexpected error: {error}"
    );

    // Zero channels.
    let no_channels = form(
        b"AIFF",
        &[
            chunk(b"COMM", &comm(0, 64, 8, extended_rate(8_000), None)),
            chunk(b"SSND", &ssnd(0, 0, &payload)),
        ],
    );
    assert!(matches!(
        AiffReader::new(&no_channels),
        Err(DecodeError::UnsupportedChannelLayout { channels: 0 })
    ));

    // A sample rate of zero: ten zero bytes in the extended field.
    let no_rate = form(
        b"AIFF",
        &[
            chunk(b"COMM", &comm(1, 64, 8, [0; 10], None)),
            chunk(b"SSND", &ssnd(0, 0, &payload)),
        ],
    );
    assert!(matches!(
        AiffReader::new(&no_rate),
        Err(DecodeError::Malformed {
            expected: "a positive integral sample rate that fits in 32 bits",
            ..
        })
    ));

    // A chunk size chosen to wrap `offset + size` in 32-bit arithmetic. The
    // u64 comparison rejects it on every target; the i686 gate is where the
    // wrapping form would have passed.
    for declared in [u32::MAX, u32::MAX - 1, 0xFFFF_FFF0, 0x8000_0000] {
        for id in [b"SSND", b"COMM", b"ANNO"] {
            let bytes = form(
                b"AIFF",
                &[
                    chunk_declaring(id, declared, b"tiny"),
                    chunk(b"COMM", &comm(1, 64, 8, extended_rate(8_000), None)),
                    chunk(b"SSND", &ssnd(0, 0, &payload)),
                ],
            );
            let error = AiffReader::new(&bytes)
                .err()
                .unwrap_or_else(|| panic!("{declared:#x} in {id:?} was accepted"));
            assert!(
                matches!(error, DecodeError::Truncated { .. }),
                "{declared:#x} in {id:?}: unexpected error: {error}"
            );
        }
    }

    // Not an IFF file, and an IFF file that is neither AIFF nor AIFC. RIFF
    // arriving here is the mirror of FORM arriving at the WAV reader.
    assert!(matches!(
        AiffReader::new(b"RIFF\x00\x00\x00\x04WAVE"),
        Err(DecodeError::UnsupportedContainer { tag }) if tag == FourCc(*b"RIFF")
    ));
    let mut amiga = whole.clone();
    amiga[8..12].copy_from_slice(b"8SVX");
    assert!(matches!(
        AiffReader::new(&amiga),
        Err(DecodeError::UnsupportedContainer { tag }) if tag == FourCc(*b"8SVX")
    ));

    // An unsupported compression type, and a carried one at an unsupported
    // width.
    let ima4 = aiff_file(
        b"AIFC",
        1,
        64,
        16,
        8_000,
        Some((b"ima4", Pstring::Zero)),
        &payload,
    );
    assert!(matches!(
        AiffReader::new(&ima4),
        Err(DecodeError::UnsupportedCodec { codec })
            if codec == decibri_decode::CodecId::FourCc(FourCc(*b"ima4"))
    ));
    let twenty = aiff_file(b"AIFF", 1, 64, 20, 8_000, None, &payload);
    assert!(matches!(
        AiffReader::new(&twenty),
        Err(DecodeError::UnsupportedSampleFormat {
            bits_per_sample: 20,
            ..
        })
    ));

    // Every one of the rejections above holds on the streaming path too.
    for (label, bytes) in [
        ("missing COMM", &no_comm),
        ("missing SSND", &no_ssnd),
        ("over-declared SSND", &over),
        ("zero channels", &no_channels),
        ("zero rate", &no_rate),
        ("ima4", &ima4),
        ("20-bit", &twenty),
    ] {
        assert!(
            stream_decode(bytes, 7, Drive::Interleaved).is_err(),
            "{label} was accepted by the stream"
        );
    }
}

/// Every prefix of a healthy file, on both paths: any may be rejected, none
/// may panic, hang or decode past the bytes present.
#[test]
fn no_prefix_of_a_valid_file_panics_or_over_reads() {
    let whole = healthy_file();
    for length in 0..=whole.len() {
        let bytes = &whole[..length];
        match AiffReader::new(bytes) {
            Err(_) => {}
            Ok(reader) => {
                let decoded = reader.decode_to_end();
                assert!(
                    decoded.samples().len() * reader.format().codec.bytes_per_sample()
                        <= bytes.len(),
                    "{length} bytes decoded to more audio than the input held"
                );
            }
        }
        for piece in [1, 13] {
            let _ = stream_decode(bytes, piece, Drive::Interleaved);
        }
    }
}

/// Single-byte mutations of a healthy file: none may panic, and the accepted
/// ones must still decode within the bytes present.
#[test]
fn no_single_byte_mutation_of_a_valid_file_panics() {
    let healthy = healthy_file();
    for offset in 0..healthy.len() {
        for value in [0x00u8, 0x01, 0x7F, 0x80, 0xFE, 0xFF] {
            let mut bytes = healthy.clone();
            bytes[offset] = value;
            if let Ok(reader) = AiffReader::new(&bytes) {
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

/// A stream cut anywhere is `Truncated` or `Malformed`, never accepted and
/// never a panic; the whole thing is fine.
#[test]
fn a_stream_that_ends_early_reports_where_it_ended() {
    let whole = healthy_file();
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
    assert!(stream_decode(&whole, 9, Drive::Interleaved).is_ok());
}

// -- The zero-frame file ------------------------------------------------------

/// The specification lets a zero-frame file omit `SSND`; with or without one,
/// it decodes to nothing on both paths, and a zero-frame claim over a
/// non-empty `SSND` is still a disagreement.
#[test]
fn a_zero_frame_file_decodes_to_nothing_with_or_without_ssnd() {
    let without = form(
        b"AIFF",
        &[chunk(b"COMM", &comm(2, 0, 16, extended_rate(48_000), None))],
    );
    let with = form(
        b"AIFF",
        &[
            chunk(b"COMM", &comm(2, 0, 16, extended_rate(48_000), None)),
            chunk(b"SSND", &ssnd(0, 0, &[])),
        ],
    );
    for (label, bytes) in [("without SSND", &without), ("with an empty SSND", &with)] {
        let reader = AiffReader::new(bytes).unwrap_or_else(|error| panic!("{label}: {error}"));
        assert_eq!(reader.frames(), 0, "{label}");
        assert!(reader.decode_to_end().is_empty(), "{label}");

        let (spec, samples) = stream_decode(bytes, 5, Drive::Interleaved)
            .unwrap_or_else(|error| panic!("{label} streamed: {error}"));
        assert_eq!(spec, Some(AudioSpec::new(48_000, 2)), "{label}");
        assert!(samples.is_empty(), "{label} streamed");
    }

    // Zero frames over a non-empty SSND disagrees like any other mismatch.
    let lying = form(
        b"AIFF",
        &[
            chunk(b"COMM", &comm(2, 0, 16, extended_rate(48_000), None)),
            chunk(b"SSND", &ssnd(0, 0, &payload_bytes(16, 1))),
        ],
    );
    assert!(AiffReader::new(&lying).is_err());
    assert!(stream_decode(&lying, 5, Drive::Interleaved).is_err());
}

// -- The chunk-order difference between the two paths -------------------------

/// The one thing the streaming path cannot do that the whole-file path can,
/// and it says so rather than mis-decoding, the same difference, for the
/// same reason, as WAV's `fmt `-before-`data` rule.
#[test]
fn the_stream_requires_comm_before_ssnd_and_the_whole_file_reader_does_not() {
    let payload = payload_bytes(32, 0xDA7A);
    let bytes = form(
        b"AIFF",
        &[
            chunk(b"SSND", &ssnd(0, 0, &payload)),
            chunk(b"COMM", &comm(1, 32, 8, extended_rate(8_000), None)),
        ],
    );

    let reader = AiffReader::new(&bytes).expect("the whole-file reader accepts either order");
    assert_eq!(reader.frames(), 32);

    let error = stream_decode(&bytes, 16, Drive::Interleaved).expect_err("the stream cannot");
    assert!(
        matches!(
            error,
            DecodeError::Malformed {
                expected: "a COMM chunk before the SSND chunk",
                ..
            }
        ),
        "unexpected error: {error}"
    );
}

/// `reset` returns the stream to its just-constructed state.
#[test]
fn a_reset_stream_decodes_a_second_file_from_scratch() {
    let first = aiff_file(b"AIFF", 1, 16, 8, 8_000, None, &payload_bytes(16, 1));
    let second = aiff_file(
        b"AIFC",
        2,
        16,
        16,
        48_000,
        Some((b"sowt", Pstring::Zero)),
        &payload_bytes(64, 2),
    );

    let mut stream = AiffStreamDecoder::new();
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
        AiffReader::new(&second)
            .expect("read")
            .decode_to_end()
            .samples(),
        "the second file after a reset",
    );
}

/// Pulling a bounded number of frames at a time gives the same audio as
/// pulling everything.
#[test]
fn a_caller_pulling_a_frame_at_a_time_gets_the_same_audio() {
    let payload = payload_for(AiffCodec::PcmI16Sowt, 2, 3_000, 0x1F2F);
    let bytes = aiff_file(
        b"AIFC",
        2,
        3_000,
        16,
        48_000,
        Some((b"sowt", Pstring::Zero)),
        &payload,
    );
    let expected = expected_samples(AiffCodec::PcmI16Sowt, &payload);

    for max_frames in [1usize, 2, 97] {
        let mut stream = AiffStreamDecoder::new();
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

// -- Gate 11: feed chunk size and total length, crossed -----------------------

/// Total length and feed size are separate dimensions, named separately
/// because step 3's Control B was missed when every input was small while the
/// limit was large. The largest total here is past the 65,536-sample ready
/// limit in every layout, and the greedy drive is the one that reaches the
/// short-return path.
#[test]
fn the_streaming_path_matches_the_whole_file_path() {
    const TOTALS: [usize; 5] = [0, 1, 3, 1_000, 70_000];
    const PIECES: [usize; 5] = [1, 7, 512, 4_096, 1 << 20];
    type StreamRow = (
        &'static [u8; 4],
        Option<(&'static [u8; 4], Pstring)>,
        u16,
        AiffCodec,
        u16,
    );
    let rows: [StreamRow; 4] = [
        (b"AIFF", None, 8, AiffCodec::PcmI8, 1),
        (
            b"AIFC",
            Some((b"sowt", Pstring::Zero)),
            16,
            AiffCodec::PcmI16Sowt,
            2,
        ),
        (
            b"AIFC",
            Some((b"ulaw", Pstring::Zero)),
            8,
            AiffCodec::MuLaw,
            1,
        ),
        (
            b"AIFC",
            Some((b"FL64", Pstring::Zero)),
            64,
            AiffCodec::Float64,
            2,
        ),
    ];

    for (form_type, compression, bits, codec, channels) in rows {
        for frames in TOTALS {
            let payload = payload_for(codec, channels, frames, 0x57EB);
            let bytes = aiff_file(
                form_type,
                channels,
                frames as u32,
                bits,
                32_000,
                compression,
                &payload,
            );
            let whole = AiffReader::new(&bytes).expect("whole-file").decode_to_end();
            let expected = expected_samples(codec, &payload);
            assert_same_samples(whole.samples(), &expected, "whole-file");

            for piece in PIECES {
                for drive in [Drive::Interleaved, Drive::Greedy] {
                    let label =
                        format!("{codec:?} {channels}ch {frames} frames, {piece}-byte {drive:?}");
                    let (spec, samples) = stream_decode(&bytes, piece, drive)
                        .unwrap_or_else(|error| panic!("{label}: {error}"));
                    assert_eq!(
                        spec,
                        Some(AudioSpec::new(32_000, channels)),
                        "{label}: spec"
                    );
                    assert_same_samples(&samples, &expected, &label);
                }
            }
        }
    }
}

// -- The dimension matrix -----------------------------------------------------

/// Where the unknown chunks sit and which order the known ones come in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Layout {
    /// `COMM` then `SSND`, nothing else.
    Plain,
    /// `SSND` then `COMM`: legal IFF, and the whole-file reader's business.
    SsndFirst,
    /// `FVER` first, its specified position in AIFF-C.
    FverFirst,
    /// `FVER` between the known chunks, where lenient writers put it.
    FverBetween,
    /// An odd-length unknown chunk before `COMM`, so a pad byte decides where
    /// `COMM` starts.
    OddUnknownBefore,
    /// An unknown chunk between `COMM` and `SSND`.
    UnknownBetween,
    /// Unknown chunks after `SSND`, which the walk never reaches.
    UnknownAfter,
    /// One of everything, everywhere at once.
    UnknownEverywhere,
}

const LAYOUTS: [Layout; 8] = [
    Layout::Plain,
    Layout::SsndFirst,
    Layout::FverFirst,
    Layout::FverBetween,
    Layout::OddUnknownBefore,
    Layout::UnknownBetween,
    Layout::UnknownAfter,
    Layout::UnknownEverywhere,
];

impl Layout {
    /// `true` when the streaming reader can read this layout. Only
    /// `SSND`-before-`COMM` is out, for the stated reason.
    fn is_streamable(self) -> bool {
        self != Layout::SsndFirst
    }

    fn chunks(self, comm: Vec<u8>, ssnd: Vec<u8>) -> Vec<Vec<u8>> {
        let even = || chunk(b"APPL", b"padding!");
        let odd = || chunk(b"ANNO", b"junk!");
        match self {
            Self::Plain => vec![comm, ssnd],
            Self::SsndFirst => vec![ssnd, comm],
            Self::FverFirst => vec![fver(), comm, ssnd],
            Self::FverBetween => vec![comm, fver(), ssnd],
            Self::OddUnknownBefore => vec![odd(), comm, ssnd],
            Self::UnknownBetween => vec![comm, odd(), ssnd],
            Self::UnknownAfter => vec![comm, ssnd, odd(), even()],
            Self::UnknownEverywhere => vec![fver(), odd(), comm, even(), ssnd, odd()],
        }
    }
}

/// The cross product: fourteen encoding rows, four channel counts, four
/// lengths, eight chunk layouts and four `SSND` shapes (7,168 files) with
/// the feed size, the drive mode, the sample rate and the `compressionName`
/// spelling rotated through the product so each meets every other dimension
/// without multiplying the file count. Every file is decoded through both
/// paths and checked against the payload decoded directly, which is a
/// statement about the container and nothing else.
///
/// Nine is in the channel counts because AIFF's `numChannels` is a signed
/// `short` and this crate applies no ceiling below it. Eight is the largest
/// count any format this crate carries restricts itself to, so nine is the
/// first value a mistakenly global limit would refuse.
#[test]
fn the_dimension_matrix() {
    const PIECES: [usize; 8] = [1, 2, 3, 5, 13, 64, 997, 65_536];
    const RATES: [u32; 6] = [8_000, 16_000, 22_050, 44_100, 48_000, 192_000];
    const SSND_SHAPES: [(u32, u32); 4] = [(0, 0), (6, 0), (0, 4), (10, 4)];
    let mut index = 0usize;
    let mut files = 0usize;

    for (form_type, compression, bits, codec) in CODECS {
        for channels in [1u16, 2, 6, 9] {
            for frames in [0usize, 1, 3, 129] {
                for layout in LAYOUTS {
                    for (offset, block_size) in SSND_SHAPES {
                        index += 1;
                        files += 1;
                        let rate = RATES[index % RATES.len()];
                        let payload = payload_for(codec, channels, frames, index as u64);
                        let expected = expected_samples(codec, &payload);
                        let spec = AudioSpec::new(rate, channels);
                        let pstring = PSTRINGS[index % PSTRINGS.len()];
                        let label = format!(
                            "{codec:?} {channels}ch {frames}f {layout:?} offset={offset} block={block_size} {pstring:?}"
                        );

                        let comm_chunk = chunk(
                            b"COMM",
                            &comm(
                                channels,
                                frames as u32,
                                bits,
                                extended_rate(rate),
                                compression.map(|four_cc| (four_cc, pstring)),
                            ),
                        );
                        let ssnd_chunk = chunk(b"SSND", &ssnd(offset, block_size, &payload));
                        let bytes = form(form_type, &layout.chunks(comm_chunk, ssnd_chunk));

                        let reader =
                            AiffReader::new(&bytes).unwrap_or_else(|e| panic!("{label}: {e}"));
                        assert_eq!(reader.format().codec, codec, "{label}");
                        assert_eq!(reader.spec(), spec, "{label}");
                        assert_eq!(reader.frames(), frames as u64, "{label}");
                        assert_eq!(reader.format().sample_frames, frames as u32, "{label}");
                        assert_eq!(reader.format().bits_per_sample, bits, "{label}");
                        let expected_form = if form_type == b"AIFF" {
                            AiffForm::Aiff
                        } else {
                            AiffForm::Aifc
                        };
                        assert_eq!(reader.format().form, expected_form, "{label}");
                        let expected_compression = compression.copied().unwrap_or(*b"NONE");
                        assert_eq!(
                            reader.format().compression,
                            FourCc(expected_compression),
                            "{label}"
                        );
                        assert_same_samples(reader.decode_to_end().samples(), &expected, &label);

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
    assert_eq!(files, 14 * 4 * 4 * 8 * 4);
}

// -- A nine-channel file from an outside encoder ------------------------------

/// A nine-channel AIFF as ffmpeg 8.1.2 writes one: 22,050 Hz in the 80-bit
/// field, 16-bit big-endian PCM, twelve frames. Carried byte for byte rather
/// than rebuilt, so the layout under test is an encoder's and not this file's
/// builder's.
///
/// The first frame is the ends of the range in every channel, `-32768`,
/// `32767`, `0`, `-1`, `1`, `-32767`, `32766`, `256`, `-4096`, so a sign slip
/// or a channel transposition cannot come back looking right.
const FFMPEG_NINE_CHANNEL_AIFF: [u8; 270] = [
    0x46, 0x4f, 0x52, 0x4d, 0x00, 0x00, 0x01, 0x06, 0x41, 0x49, 0x46, 0x46, 0x43, 0x4f, 0x4d, 0x4d,
    0x00, 0x00, 0x00, 0x12, 0x00, 0x09, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x10, 0x40, 0x0d, 0xac, 0x44,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x53, 0x53, 0x4e, 0x44, 0x00, 0x00, 0x00, 0xe0, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x00, 0x7f, 0xff, 0x00, 0x00, 0xff, 0xff, 0x00, 0x01,
    0x80, 0x01, 0x7f, 0xfe, 0x01, 0x00, 0xf0, 0x00, 0x8a, 0xab, 0x9a, 0xae, 0xaa, 0xb1, 0xba, 0xb4,
    0xca, 0xb7, 0xda, 0xba, 0xea, 0xbd, 0xfa, 0xc0, 0x0a, 0xc3, 0x95, 0x56, 0xa5, 0x59, 0xb5, 0x5c,
    0xc5, 0x5f, 0xd5, 0x62, 0xe5, 0x65, 0xf5, 0x68, 0x05, 0x6b, 0x15, 0x6e, 0xa0, 0x01, 0xb0, 0x04,
    0xc0, 0x07, 0xd0, 0x0a, 0xe0, 0x0d, 0xf0, 0x10, 0x00, 0x13, 0x10, 0x16, 0x20, 0x19, 0xaa, 0xac,
    0xba, 0xaf, 0xca, 0xb2, 0xda, 0xb5, 0xea, 0xb8, 0xfa, 0xbb, 0x0a, 0xbe, 0x1a, 0xc1, 0x2a, 0xc4,
    0xb5, 0x57, 0xc5, 0x5a, 0xd5, 0x5d, 0xe5, 0x60, 0xf5, 0x63, 0x05, 0x66, 0x15, 0x69, 0x25, 0x6c,
    0x35, 0x6f, 0xc0, 0x02, 0xd0, 0x05, 0xe0, 0x08, 0xf0, 0x0b, 0x00, 0x0e, 0x10, 0x11, 0x20, 0x14,
    0x30, 0x17, 0x40, 0x1a, 0xca, 0xad, 0xda, 0xb0, 0xea, 0xb3, 0xfa, 0xb6, 0x0a, 0xb9, 0x1a, 0xbc,
    0x2a, 0xbf, 0x3a, 0xc2, 0x4a, 0xc5, 0xd5, 0x58, 0xe5, 0x5b, 0xf5, 0x5e, 0x05, 0x61, 0x15, 0x64,
    0x25, 0x67, 0x35, 0x6a, 0x45, 0x6d, 0x55, 0x70, 0xe0, 0x03, 0xf0, 0x06, 0x00, 0x09, 0x10, 0x0c,
    0x20, 0x0f, 0x30, 0x12, 0x40, 0x15, 0x50, 0x18, 0x60, 0x1b, 0xea, 0xae, 0xfa, 0xb1, 0x0a, 0xb4,
    0x1a, 0xb7, 0x2a, 0xba, 0x3a, 0xbd, 0x4a, 0xc0, 0x5a, 0xc3, 0x6a, 0xc6, 0xf5, 0x59, 0x05, 0x5c,
    0x15, 0x5f, 0x25, 0x62, 0x35, 0x65, 0x45, 0x68, 0x55, 0x6b, 0x65, 0x6e, 0x75, 0x71,
];

/// What ffmpeg 8.1.2 itself decodes that file to: the bytes of
/// `ffmpeg -i nine.aiff -f f32le -c:a pcm_f32le`, 108 interleaved samples in
/// twelve interchannel frames, little-endian as that command wrote them.
///
/// The expected values come from the reference tool rather than from this
/// crate's writer or its codec layer, so agreement here is agreement with an
/// outside implementation and not with itself.
const FFMPEG_NINE_CHANNEL_SAMPLES: [u8; 432] = [
    0x00, 0x00, 0x80, 0xbf, 0x00, 0xfe, 0x7f, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xb8,
    0x00, 0x00, 0x00, 0x38, 0x00, 0xfe, 0x7f, 0xbf, 0x00, 0xfc, 0x7f, 0x3f, 0x00, 0x00, 0x00, 0x3c,
    0x00, 0x00, 0x00, 0xbe, 0x00, 0xaa, 0x6a, 0xbf, 0x00, 0xa4, 0x4a, 0xbf, 0x00, 0x9e, 0x2a, 0xbf,
    0x00, 0x98, 0x0a, 0xbf, 0x00, 0x24, 0xd5, 0xbe, 0x00, 0x18, 0x95, 0xbe, 0x00, 0x18, 0x2a, 0xbe,
    0x00, 0x00, 0x28, 0xbd, 0x00, 0x30, 0xac, 0x3d, 0x00, 0x54, 0x55, 0xbf, 0x00, 0x4e, 0x35, 0xbf,
    0x00, 0x48, 0x15, 0xbf, 0x00, 0x84, 0xea, 0xbe, 0x00, 0x78, 0xaa, 0xbe, 0x00, 0xd8, 0x54, 0xbe,
    0x00, 0x80, 0xa9, 0xbd, 0x00, 0x60, 0x2d, 0x3d, 0x00, 0x70, 0x2b, 0x3e, 0x00, 0xfe, 0x3f, 0xbf,
    0x00, 0xf8, 0x1f, 0xbf, 0x00, 0xe4, 0xff, 0xbe, 0x00, 0xd8, 0xbf, 0xbe, 0x00, 0x98, 0x7f, 0xbe,
    0x00, 0x00, 0xff, 0xbd, 0x00, 0x00, 0x18, 0x3a, 0x00, 0xb0, 0x00, 0x3e, 0x00, 0x64, 0x80, 0x3e,
    0x00, 0xa8, 0x2a, 0xbf, 0x00, 0xa2, 0x0a, 0xbf, 0x00, 0x38, 0xd5, 0xbe, 0x00, 0x2c, 0x95, 0xbe,
    0x00, 0x40, 0x2a, 0xbe, 0x00, 0xa0, 0x28, 0xbd, 0x00, 0xe0, 0xab, 0x3d, 0x00, 0x08, 0x56, 0x3e,
    0x00, 0x10, 0xab, 0x3e, 0x00, 0x52, 0x15, 0xbf, 0x00, 0x98, 0xea, 0xbe, 0x00, 0x8c, 0xaa, 0xbe,
    0x00, 0x00, 0x55, 0xbe, 0x00, 0xd0, 0xa9, 0xbd, 0x00, 0xc0, 0x2c, 0x3d, 0x00, 0x48, 0x2b, 0x3e,
    0x00, 0xb0, 0x95, 0x3e, 0x00, 0xbc, 0xd5, 0x3e, 0x00, 0xf8, 0xff, 0xbe, 0x00, 0xec, 0xbf, 0xbe,
    0x00, 0xc0, 0x7f, 0xbe, 0x00, 0x50, 0xff, 0xbd, 0x00, 0x00, 0xe0, 0x39, 0x00, 0x88, 0x00, 0x3e,
    0x00, 0x50, 0x80, 0x3e, 0x00, 0x5c, 0xc0, 0x3e, 0x00, 0x34, 0x00, 0x3f, 0x00, 0x4c, 0xd5, 0xbe,
    0x00, 0x40, 0x95, 0xbe, 0x00, 0x68, 0x2a, 0xbe, 0x00, 0x40, 0x29, 0xbd, 0x00, 0x90, 0xab, 0x3d,
    0x00, 0xe0, 0x55, 0x3e, 0x00, 0xfc, 0xaa, 0x3e, 0x00, 0x08, 0xeb, 0x3e, 0x00, 0x8a, 0x15, 0x3f,
    0x00, 0xa0, 0xaa, 0xbe, 0x00, 0x28, 0x55, 0xbe, 0x00, 0x20, 0xaa, 0xbd, 0x00, 0x20, 0x2c, 0x3d,
    0x00, 0x20, 0x2b, 0x3e, 0x00, 0x9c, 0x95, 0x3e, 0x00, 0xa8, 0xd5, 0x3e, 0x00, 0xda, 0x0a, 0x3f,
    0x00, 0xe0, 0x2a, 0x3f, 0x00, 0xe8, 0x7f, 0xbe, 0x00, 0xa0, 0xff, 0xbd, 0x00, 0x00, 0x90, 0x39,
    0x00, 0x60, 0x00, 0x3e, 0x00, 0x3c, 0x80, 0x3e, 0x00, 0x48, 0xc0, 0x3e, 0x00, 0x2a, 0x00, 0x3f,
    0x00, 0x30, 0x20, 0x3f, 0x00, 0x36, 0x40, 0x3f, 0x00, 0x90, 0x2a, 0xbe, 0x00, 0xe0, 0x29, 0xbd,
    0x00, 0x40, 0xab, 0x3d, 0x00, 0xb8, 0x55, 0x3e, 0x00, 0xe8, 0xaa, 0x3e, 0x00, 0xf4, 0xea, 0x3e,
    0x00, 0x80, 0x15, 0x3f, 0x00, 0x86, 0x35, 0x3f, 0x00, 0x8c, 0x55, 0x3f, 0x00, 0x70, 0xaa, 0xbd,
    0x00, 0x80, 0x2b, 0x3d, 0x00, 0xf8, 0x2a, 0x3e, 0x00, 0x88, 0x95, 0x3e, 0x00, 0x94, 0xd5, 0x3e,
    0x00, 0xd0, 0x0a, 0x3f, 0x00, 0xd6, 0x2a, 0x3f, 0x00, 0xdc, 0x4a, 0x3f, 0x00, 0xe2, 0x6a, 0x3f,
];

/// Nine channels, through both paths, against ffmpeg's own decode of the same
/// bytes.
///
/// The matrix above covers nine channels across every other dimension, but its
/// files are built here and its expected samples come from this crate's codec
/// layer. This one is the outside anchor: an encoder that is not this crate
/// wrote the file, and a decoder that is not this crate said what is in it.
#[test]
fn a_real_nine_channel_aiff_agrees_with_ffmpeg() {
    let expected: Vec<f32> = FFMPEG_NINE_CHANNEL_SAMPLES
        .chunks_exact(4)
        .map(|four| f32::from_le_bytes([four[0], four[1], four[2], four[3]]))
        .collect();
    let bytes = &FFMPEG_NINE_CHANNEL_AIFF;

    let reader = AiffReader::new(bytes).expect("ffmpeg's nine-channel AIFF");
    assert_eq!(reader.spec(), AudioSpec::new(22_050, 9));
    assert_eq!(reader.frames(), 12, "twelve interchannel frames");
    assert_eq!(reader.format().form, AiffForm::Aiff);
    assert_eq!(reader.format().codec, AiffCodec::PcmI16);
    let decoded = reader.decode_to_end();
    assert_eq!(decoded.samples().len(), 108, "nine channels of twelve");
    assert_same_samples(decoded.samples(), &expected, "ffmpeg nine-channel AIFF");

    // The streaming path, at feed sizes coprime with the eighteen-byte frame
    // as well as at one that swallows the file whole.
    for piece in [1usize, 7, 13, 64, 997, 65_536] {
        for drive in [Drive::Greedy, Drive::Interleaved] {
            let (spec, streamed) = stream_decode(bytes, piece, drive)
                .unwrap_or_else(|e| panic!("streamed at {piece} bytes, {drive:?}: {e}"));
            assert_eq!(spec, Some(AudioSpec::new(22_050, 9)));
            assert_same_samples(
                &streamed,
                &expected,
                &format!("ffmpeg nine-channel AIFF streamed at {piece} bytes, {drive:?}"),
            );
        }
    }
}

// -- Gate 12: cross-platform determinism --------------------------------------

/// FNV-1a, so the witness is the bytes themselves and not a float comparison.
fn fnv1a(bytes: impl IntoIterator<Item = u8>) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// One number that changes if any bit of any decoded sample changes, over
/// every encoding row at two channel counts, through both paths.
///
/// Running this on two toolchains and on a 32-bit target and getting the same
/// constant is the evidence for the byte-identical claim. The constant is
/// pinned rather than recomputed so that a change shows up as a diff here.
#[test]
fn aiff_output_is_bit_identical_to_a_pinned_witness() {
    let mut witness: Vec<u8> = Vec::new();
    for (form_type, compression, bits, codec) in CODECS {
        for channels in [1u16, 3] {
            let payload = payload_for(codec, channels, 97, 0x1CE2);
            let bytes = aiff_file(
                form_type,
                channels,
                97,
                bits,
                24_000,
                compression.map(|four_cc| (four_cc, Pstring::OddName)),
                &payload,
            );
            let decoded = AiffReader::new(&bytes).expect("read").decode_to_end();
            // 7 is coprime with every sample width here, so the streaming
            // reader is driven across its partial-sample path as well.
            let (_, streamed) = stream_decode(&bytes, 7, Drive::Greedy).expect("stream");
            assert_same_samples(&streamed, decoded.samples(), "witness stream");
            witness.extend(
                decoded
                    .samples()
                    .iter()
                    .flat_map(|s| s.to_bits().to_le_bytes()),
            );
        }
    }
    // Re-pinned for 0.1.2, when the `raw ` row joined CODECS and so joined
    // this sweep. The previous value, over the thirteen rows this witness
    // covered in 0.1.1, was 0x428d_f9aa_ce0c_4172.
    assert_eq!(fnv1a(witness), 0x34d1_5ab8_d25f_1822, "AIFF output changed");
}

// -- Structural odds and ends the dimensions imply ----------------------------

/// The pad byte is load-bearing on the whole-file path: removing an odd
/// chunk's pad must not let the reader find the chunks that follow it.
#[test]
fn an_unpadded_odd_chunk_desynchronises_the_walk_rather_than_being_forgiven() {
    let payload = payload_bytes(32, 0x9AD2);
    let expected = expected_samples(AiffCodec::PcmI8, &payload);
    let padded = form(
        b"AIFF",
        &[
            chunk(b"ANNO", b"junk!"),
            chunk(b"COMM", &comm(1, 32, 8, extended_rate(8_000), None)),
            chunk(b"SSND", &ssnd(0, 0, &payload)),
        ],
    );
    assert_same_samples(
        AiffReader::new(&padded)
            .expect("padded")
            .decode_to_end()
            .samples(),
        &expected,
        "padded odd chunk",
    );

    let unpadded = form(
        b"AIFF",
        &[
            chunk_unpadded(b"ANNO", b"junk!"),
            chunk(b"COMM", &comm(1, 32, 8, extended_rate(8_000), None)),
            chunk(b"SSND", &ssnd(0, 0, &payload)),
        ],
    );
    match AiffReader::new(&unpadded) {
        Err(_) => {}
        Ok(reader) => assert_ne!(
            reader.decode_to_end().samples().len(),
            expected.len(),
            "a walk that ignored the pad byte agreed with one that applied it"
        ),
    }
}

/// A compressionName whose length byte counts past its chunk is a miscount,
/// not something to shrug at, on both paths.
#[test]
fn a_compression_name_overrunning_its_chunk_is_malformed() {
    let mut body = comm(1, 8, 16, extended_rate(8_000), None);
    body.extend_from_slice(b"sowt");
    body.push(200); // claims 200 characters; none follow
    let bytes = form(
        b"AIFC",
        &[
            chunk(b"COMM", &body),
            chunk(b"SSND", &ssnd(0, 0, &payload_bytes(16, 3))),
        ],
    );
    for error in [
        AiffReader::new(&bytes).expect_err("whole-file must reject"),
        stream_decode(&bytes, 5, Drive::Interleaved).expect_err("stream must reject"),
    ] {
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
    }
}
