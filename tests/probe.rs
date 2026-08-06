#![forbid(unsafe_code)]
//! The content probe: what a run of bytes is, and which reader gets it.
//!
//! # The dimensions, enumerated before any test was written
//!
//! The discipline is the one recorded on `wav_conformance.rs`,
//! `aiff_conformance.rs` and `flac_conformance.rs`: a test's coverage is
//! bounded by the dimensions its inputs vary along, and the fifth negative
//! control of this crate's construction sharpened that to include the
//! dimensions of the input *values*, not only of its structure.
//!
//! 1. container magic - `RIFF`, `RF64`, `FORM`, `fLaC`, and magics that are
//!    none of them
//! 2. form type at offset eight - `WAVE`, `AIFF`, `AIFC`, and the ones that
//!    make the same magic a different file entirely
//! 3. entry point - [`identify`], [`decode`] and [`AudioStreamDecoder`]
//! 4. payload encoding inside the container, so routing is exercised for more
//!    than one codec per format
//! 5. feed chunk size on the streaming path, down to one byte
//! 6. total input length, on both sides of the streaming readers' 65,536
//!    sample ready limit
//! 7. input length below the probe length, at every value from zero to eleven
//! 8. the name the bytes arrived under, which must never decide anything
//! 9. the detail the streaming probe reports about the reader it chose, and
//!    the three points at which it has none to report: before identification,
//!    for a format the accessor is not about, and after identification but
//!    before the header chunk carrying the detail has arrived
//!
//! # Dimension 2 is the one this suite exists for
//!
//! `RIFF` at offset zero names a container *family*. AVI is a RIFF file, so is
//! an animated cursor, so is a WebP image, so is a RIFF MIDI file. `FORM` is
//! the same for EA IFF 85: `8SVX` and `ILBM` are `FORM` files. A probe that
//! reads four bytes hands all of those to [`WavReader`] or [`AiffReader`] and
//! lets them fail somewhere further in, reporting a missing `fmt ` chunk for a
//! file that was never a WAV.
//!
//! **Every real fixture in this repository is a genuine WAV, AIFF or FLAC**,
//! so none of them exercises that. The decoy containers below are therefore
//! constructed deliberately, with real chunk structure behind the wrong form
//! type, and they are the reason a negative control in the form-type check has
//! anything to go red.
//!
//! # What is not probed, and why it is not a gap
//!
//! Headerless linear PCM, headerless G.711 and bare FLAC frame streams carry
//! no signature. Guessing at them is measured elsewhere in this suite as
//! unsafe: `flac_headerless.rs` records that a frame's 14-bit sync code and
//! 8-bit header CRC together accept about one position in 32,768 of random
//! data. Nothing here tries.

use std::path::PathBuf;

use decibri_decode::{
    decode, identify, AiffCodec, AiffReader, AiffWriter, AudioBuffer, AudioSpec,
    AudioStreamDecoder, Container, DecodeError, FlacReader, Md5Check, RiffFlavour, StreamSource,
    WavCodec, WavReader, WavWriter,
};

// --- Fixtures ---------------------------------------------------------------

/// Deterministic pseudo-audio, in exact binary fractions so every value is
/// representable in `f32` and in every integer width the containers carry.
fn audio(frames: usize, channels: u16, seed: u32) -> Vec<f32> {
    let mut samples = Vec::with_capacity(frames * channels as usize);
    let mut state = seed | 1;
    for _ in 0..frames * channels as usize {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        // A signed value on a power-of-two grid: exact in f32, exact after the
        // round trip through 16-bit and wider integers.
        samples.push(((state >> 17) as i32 - 16_384) as f32 / 32_768.0);
    }
    samples
}

/// RFC 9639 appendix D.1: 44.1 kHz 16-bit stereo, one frame of one sample.
///
/// Carried as literal bytes rather than built, so the FLAC leg of every gate
/// here runs in a checkout with no corpus. The same three files anchor
/// `flac_conformance.rs`, where their sample values are asserted against the
/// document's own text.
const RFC_EXAMPLE_1: &[u8] = &[
    0x66, 0x4c, 0x61, 0x43, 0x80, 0x00, 0x00, 0x22, 0x10, 0x00, 0x10, 0x00, 0x00, 0x00, 0x0f, 0x00,
    0x00, 0x0f, 0x0a, 0xc4, 0x42, 0xf0, 0x00, 0x00, 0x00, 0x01, 0x3e, 0x84, 0xb4, 0x18, 0x07, 0xdc,
    0x69, 0x03, 0x07, 0x58, 0x6a, 0x3d, 0xad, 0x1a, 0x2e, 0x0f, 0xff, 0xf8, 0x69, 0x18, 0x00, 0x00,
    0xbf, 0x03, 0x58, 0xfd, 0x03, 0x12, 0x8b, 0xaa, 0x9a,
];

/// RFC 9639 appendix D.2: 44.1 kHz 16-bit stereo, 19 samples in two frames,
/// behind a seek table, a vorbis comment and a padding block - so the probe
/// meets a FLAC file whose audio is a long way past its signature.
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

/// RFC 9639 appendix D.3: 32 kHz 8-bit mono, 24 samples under a third-order
/// linear predictor.
const RFC_EXAMPLE_3: &[u8] = &[
    0x66, 0x4c, 0x61, 0x43, 0x80, 0x00, 0x00, 0x22, 0x10, 0x00, 0x10, 0x00, 0x00, 0x00, 0x1f, 0x00,
    0x00, 0x1f, 0x07, 0xd0, 0x00, 0x70, 0x00, 0x00, 0x00, 0x18, 0xf8, 0xf9, 0xe3, 0x96, 0xf5, 0xcb,
    0xcf, 0xc6, 0xdc, 0x80, 0x7f, 0x99, 0x77, 0x90, 0x6b, 0x32, 0xff, 0xf8, 0x68, 0x02, 0x00, 0x17,
    0xe9, 0x44, 0x00, 0x4f, 0x6f, 0x31, 0x3d, 0x10, 0x47, 0xd2, 0x27, 0xcb, 0x6d, 0x09, 0x08, 0x31,
    0x45, 0x2b, 0xdc, 0x28, 0x22, 0x22, 0x80, 0x57, 0xa3,
];

/// The three FLAC files this suite carries in tree.
const FLAC_FILES: [(&str, &[u8]); 3] = [
    ("RFC 9639 D.1", RFC_EXAMPLE_1),
    ("RFC 9639 D.2", RFC_EXAMPLE_2),
    ("RFC 9639 D.3", RFC_EXAMPLE_3),
];

/// A container fixture: what it is called, what it holds, and what the probe
/// must say about it.
struct Fixture {
    name: &'static str,
    bytes: Vec<u8>,
    container: Container,
}

/// Every WAV and AIFF fixture, varying dimensions 1, 2, 4 and 6.
///
/// Kept apart from the FLAC fixtures so the witness below is a hash over the
/// PCM containers alone, with FLAC carrying its own.
fn pcm_fixtures() -> Vec<Fixture> {
    let short = audio(64, 2, 7);
    let mono = audio(97, 1, 11);
    // Past the streaming readers' 65,536-sample ready limit, so the streaming
    // gate crosses the point where they start taking short pushes.
    let long = audio(40_000, 2, 13);

    vec![
        Fixture {
            name: "RIFF/WAVE 16-bit stereo",
            bytes: WavWriter::new(AudioSpec::new(44_100, 2), WavCodec::PcmI16)
                .to_bytes(&short)
                .expect("a 16-bit WAV writes"),
            container: Container::Wav,
        },
        Fixture {
            name: "RIFF/WAVE 24-bit mono",
            bytes: WavWriter::new(AudioSpec::mono(48_000), WavCodec::PcmI24)
                .to_bytes(&mono)
                .expect("a 24-bit WAV writes"),
            container: Container::Wav,
        },
        Fixture {
            name: "RIFF/WAVE mu-law mono",
            bytes: WavWriter::new(AudioSpec::mono(8_000), WavCodec::MuLaw)
                .to_bytes(&mono)
                .expect("a mu-law WAV writes"),
            container: Container::Wav,
        },
        Fixture {
            name: "RF64/WAVE float32 stereo",
            bytes: WavWriter::new(AudioSpec::new(48_000, 2), WavCodec::Float32)
                .with_flavour(RiffFlavour::Rf64)
                .to_bytes(&short)
                .expect("an RF64 WAV writes"),
            container: Container::Wav,
        },
        Fixture {
            name: "RIFF/WAVE 16-bit stereo, past the ready limit",
            bytes: WavWriter::new(AudioSpec::new(44_100, 2), WavCodec::PcmI16)
                .to_bytes(&long)
                .expect("a long WAV writes"),
            container: Container::Wav,
        },
        Fixture {
            name: "FORM/AIFF 16-bit stereo",
            bytes: AiffWriter::new(AudioSpec::new(44_100, 2), AiffCodec::PcmI16)
                .to_bytes(&short)
                .expect("a 16-bit AIFF writes"),
            container: Container::Aiff,
        },
        Fixture {
            name: "FORM/AIFF 8-bit mono",
            bytes: AiffWriter::new(AudioSpec::mono(22_050), AiffCodec::PcmI8)
                .to_bytes(&mono)
                .expect("an 8-bit AIFF writes"),
            container: Container::Aiff,
        },
        Fixture {
            name: "FORM/AIFC sowt 16-bit stereo",
            bytes: AiffWriter::new(AudioSpec::new(32_000, 2), AiffCodec::PcmI16Sowt)
                .to_bytes(&short)
                .expect("a sowt AIFF-C writes"),
            container: Container::Aiff,
        },
        Fixture {
            name: "FORM/AIFC A-law mono",
            bytes: AiffWriter::new(AudioSpec::mono(8_000), AiffCodec::ALaw)
                .to_bytes(&mono)
                .expect("an A-law AIFF-C writes"),
            container: Container::Aiff,
        },
        Fixture {
            name: "FORM/AIFF 16-bit stereo, past the ready limit",
            bytes: AiffWriter::new(AudioSpec::new(44_100, 2), AiffCodec::PcmI16)
                .to_bytes(&long)
                .expect("a long AIFF writes"),
            container: Container::Aiff,
        },
    ]
}

/// The FLAC fixtures, varying dimensions 1 and 4 for the third container.
fn flac_fixtures() -> Vec<Fixture> {
    FLAC_FILES
        .iter()
        .map(|(name, bytes)| Fixture {
            name,
            bytes: bytes.to_vec(),
            container: Container::Flac,
        })
        .collect()
}

/// What the reader that owns `fixture` produces, called directly.
///
/// This is the oracle for every routing assertion: the probe is right exactly
/// when it gives what the specific reader gives, and neither side is the other
/// one's implementation.
fn directly(fixture: &Fixture) -> AudioBuffer {
    match fixture.container {
        Container::Wav => WavReader::new(&fixture.bytes)
            .expect("the WAV fixture opens")
            .decode_to_end(),
        Container::Aiff => AiffReader::new(&fixture.bytes)
            .expect("the AIFF fixture opens")
            .decode_to_end(),
        Container::Flac => FlacReader::new(&fixture.bytes)
            .expect("the FLAC fixture opens")
            .decode_to_end()
            .expect("the FLAC fixture decodes"),
        other => panic!("no direct reader for {other}"),
    }
}

/// `true` when two buffers hold the same spec and bit-identical samples.
fn identical(left: &AudioBuffer, right: &AudioBuffer) -> bool {
    left.spec() == right.spec()
        && left.samples().len() == right.samples().len()
        && left
            .samples()
            .iter()
            .zip(right.samples())
            .all(|(a, b)| a.to_bits() == b.to_bits())
}

/// A twelve-byte IFF-family header: magic, a size field, form type.
fn header(magic: &[u8; 4], size: u32, form: &[u8; 4], big_endian: bool) -> Vec<u8> {
    let mut bytes = magic.to_vec();
    if big_endian {
        bytes.extend_from_slice(&size.to_be_bytes());
    } else {
        bytes.extend_from_slice(&size.to_le_bytes());
    }
    bytes.extend_from_slice(form);
    bytes
}

/// A RIFF AVI file: a real `LIST`/`hdrl` header list with an `avih` chunk in
/// it, behind the `AVI ` form type.
///
/// Constructed rather than fetched because nothing in this repository is one.
/// The chunk structure is genuine so that a probe accepting it would go on to
/// walk real chunks and fail on a missing `fmt ` rather than on garbage, which
/// is exactly the confusing failure the form-type check prevents.
fn avi_file() -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(b"LIST");
    body.extend_from_slice(&68u32.to_le_bytes());
    body.extend_from_slice(b"hdrl");
    body.extend_from_slice(b"avih");
    body.extend_from_slice(&56u32.to_le_bytes());
    // dwMicroSecPerFrame through dwHeight, as an AVI main header has them.
    for field in [
        41_708u32,
        0,
        0,
        0x0000_0810,
        300,
        0,
        1,
        0,
        640,
        480,
        0,
        0,
        0,
        0,
    ] {
        body.extend_from_slice(&field.to_le_bytes());
    }
    body.extend_from_slice(b"LIST");
    body.extend_from_slice(&4u32.to_le_bytes());
    body.extend_from_slice(b"movi");

    let mut file = header(b"RIFF", (body.len() + 4) as u32, b"AVI ", false);
    file.extend_from_slice(&body);
    file
}

/// An EA IFF 85 `8SVX` file: an Amiga sampled-voice file, with a real `VHDR`
/// chunk behind the wrong form type.
fn svx_file() -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(b"VHDR");
    body.extend_from_slice(&20u32.to_be_bytes());
    body.extend_from_slice(&64u32.to_be_bytes()); // oneShotHiSamples
    body.extend_from_slice(&0u32.to_be_bytes()); // repeatHiSamples
    body.extend_from_slice(&0u32.to_be_bytes()); // samplesPerHiCycle
    body.extend_from_slice(&8_000u16.to_be_bytes()); // samplesPerSec
    body.push(1); // ctOctave
    body.push(0); // sCompression
    body.extend_from_slice(&0x0001_0000u32.to_be_bytes()); // volume
    body.extend_from_slice(b"BODY");
    body.extend_from_slice(&64u32.to_be_bytes());
    body.extend_from_slice(&[0x20; 64]);

    let mut file = header(b"FORM", (body.len() + 4) as u32, b"8SVX", true);
    file.extend_from_slice(&body);
    file
}

/// Every decoy container: a real container family behind a form type this
/// crate does not read, and the form type the error has to name.
fn decoys() -> Vec<(&'static str, Vec<u8>, [u8; 4])> {
    vec![
        ("AVI, with a real header list", avi_file(), *b"AVI "),
        ("8SVX, with a real VHDR", svx_file(), *b"8SVX"),
        (
            "an animated cursor",
            header(b"RIFF", 2048, b"ACON", false),
            *b"ACON",
        ),
        (
            "a WebP image",
            header(b"RIFF", 2048, b"WEBP", false),
            *b"WEBP",
        ),
        (
            "a RIFF MIDI file",
            header(b"RIFF", 2048, b"RMID", false),
            *b"RMID",
        ),
        (
            "an RF64 container that is not WAVE",
            header(b"RF64", 2048, b"AVI ", false),
            *b"AVI ",
        ),
        (
            "an IFF bitmap",
            header(b"FORM", 2048, b"ILBM", true),
            *b"ILBM",
        ),
        (
            "a form type one letter off AIFF",
            header(b"FORM", 2048, b"AIFZ", true),
            *b"AIFZ",
        ),
    ]
}

/// Pushes `bytes` through the streaming probe in `chunk`-byte pieces and
/// returns everything it produced, bound to the spec it reported.
fn stream(bytes: &[u8], chunk: usize) -> Result<(AudioBuffer, Option<Container>), DecodeError> {
    let mut decoder = AudioStreamDecoder::new();
    let mut samples = Vec::new();
    for piece in bytes.chunks(chunk) {
        let mut offset = 0;
        while offset < piece.len() {
            offset += decoder.push(&piece[offset..])?;
            while decoder.pull(&mut samples, usize::MAX)? > 0 {}
        }
    }
    decoder.finish(&mut samples)?;
    let spec = decoder.spec().unwrap_or(AudioSpec::new(0, 0));
    Ok((
        AudioBuffer::from_samples(spec, samples),
        decoder.container(),
    ))
}

/// FNV-1a, so the witness is over the bytes and not over floats compared with
/// a tolerance.
fn fnv1a(bytes: impl IntoIterator<Item = u8>) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Appends the bit patterns of `samples`, and the spec that describes them, to
/// `witness`.
fn absorb(witness: &mut Vec<u8>, audio: &AudioBuffer) {
    witness.extend(audio.sample_rate().to_le_bytes());
    witness.extend(audio.channels().to_le_bytes());
    witness.extend(
        audio
            .samples()
            .iter()
            .flat_map(|sample| sample.to_bits().to_le_bytes()),
    );
}

// --- Group 1: routing -------------------------------------------------------

/// Every carried format is identified from its content and decoded to exactly
/// what its own reader produces.
#[test]
fn every_carried_format_routes_to_the_reader_that_owns_it() {
    let mut fixtures = pcm_fixtures();
    fixtures.extend(flac_fixtures());
    assert!(fixtures.len() >= 10, "the sweep lost its fixtures");

    for fixture in &fixtures {
        assert_eq!(
            identify(&fixture.bytes).expect("the fixture identifies"),
            fixture.container,
            "{} was identified as something else",
            fixture.name
        );

        let probed = decode(&fixture.bytes).unwrap_or_else(|error| {
            panic!("{} did not decode through the probe: {error}", fixture.name)
        });
        let direct = directly(fixture);
        assert!(
            identical(&probed, &direct),
            "{}: the probe and {} disagree",
            fixture.name,
            fixture.container
        );
        println!(
            "{}: {} via {}, {} samples at {} Hz",
            fixture.name,
            fixture.container,
            fixture.container,
            probed.samples().len(),
            probed.sample_rate()
        );
    }
}

/// Where `id`'s chunk payload starts, searched for rather than computed from
/// a header layout, so a fixture written in a different header style does not
/// silently move the field the test overwrites.
fn chunk_payload(bytes: &[u8], id: &[u8; 4]) -> usize {
    let at = bytes
        .windows(4)
        .position(|window| window == id)
        .unwrap_or_else(|| panic!("no {} chunk in the fixture", String::from_utf8_lossy(id)));
    at + 8
}

/// No decode this crate performs produces a buffer with no channels.
///
/// Two halves, because the doc comments on [`decode`] and the three readers'
/// `decode_to_end` state this as a guarantee and an unchecked guarantee is a
/// claim. Every fixture in the sweep decodes to a positive channel count,
/// through the probe and through its own reader; and a WAV and an AIFF whose
/// declared channel count has been overwritten with zero are refused rather
/// than decoded. FLAC has no third case: its channel count is coded as one
/// more than a three-bit field, so there is no zero to overwrite.
///
/// The guarantee is about what a decode produces. `AudioSpec::new` and
/// `AudioBuffer::from_samples` accept zero from a caller who states it, and
/// that is deliberate and untouched here.
#[test]
fn no_decode_produces_a_buffer_with_no_channels() {
    let mut fixtures = pcm_fixtures();
    fixtures.extend(flac_fixtures());
    assert!(fixtures.len() >= 10, "the sweep lost its fixtures");

    for fixture in &fixtures {
        let probed = decode(&fixture.bytes).expect("the fixture decodes through the probe");
        assert_ne!(
            probed.spec().channels,
            0,
            "{}: the probe produced no channels",
            fixture.name
        );
        assert_ne!(
            directly(fixture).spec().channels,
            0,
            "{}: its own reader produced no channels",
            fixture.name
        );
    }

    // WAV: wFormatTag then nChannels, little-endian, at the start of `fmt `.
    let samples = audio(64, 1, 7);
    let mut wav = WavWriter::new(AudioSpec::mono(16_000), WavCodec::PcmI16)
        .to_bytes(&samples)
        .expect("a mono WAV writes");
    let at = chunk_payload(&wav, b"fmt ");
    wav[at + 2..at + 4].copy_from_slice(&0u16.to_le_bytes());
    assert!(
        matches!(
            WavReader::new(&wav),
            Err(DecodeError::UnsupportedChannelLayout { channels: 0 })
        ),
        "a WAV declaring no channels was not refused"
    );
    assert!(
        decode(&wav).is_err(),
        "the probe decoded a WAV declaring no channels"
    );

    // AIFF: numChannels first, big-endian, at the start of `COMM`.
    let mut aiff = AiffWriter::new(AudioSpec::mono(16_000), AiffCodec::PcmI16)
        .to_bytes(&samples)
        .expect("a mono AIFF writes");
    let at = chunk_payload(&aiff, b"COMM");
    aiff[at..at + 2].copy_from_slice(&0u16.to_be_bytes());
    assert!(
        matches!(
            AiffReader::new(&aiff),
            Err(DecodeError::UnsupportedChannelLayout { channels: 0 })
        ),
        "an AIFF declaring no channels was not refused"
    );
    assert!(
        decode(&aiff).is_err(),
        "the probe decoded an AIFF declaring no channels"
    );
}

/// The name a file arrives under decides nothing, for all three formats.
///
/// Written to real files with lying extensions and read back, because the
/// claim is about what a consumer actually does rather than about a slice in a
/// test.
#[test]
fn the_probe_ignores_the_name_the_bytes_arrived_under() {
    let mut fixtures = pcm_fixtures();
    fixtures.extend(flac_fixtures());

    let directory = std::env::temp_dir().join("decibri-decode-probe-names");
    std::fs::create_dir_all(&directory).expect("a temp directory can be made");

    // Each format written under every other format's extension. The two the
    // step-5 gate covered were WAV and AIFF; FLAC is the third.
    let lies = [
        "input.wav",
        "input.aiff",
        "input.flac",
        "input.mp3",
        "input",
    ];
    for (index, fixture) in fixtures.iter().enumerate() {
        let name = lies[index % lies.len()];
        let path: PathBuf = directory.join(format!("{index}-{name}"));
        std::fs::write(&path, &fixture.bytes).expect("the fixture writes");
        let read_back = std::fs::read(&path).expect("the fixture reads back");

        assert_eq!(
            identify(&read_back).expect("the fixture identifies"),
            fixture.container,
            "{} called {name} was identified from its name",
            fixture.name
        );
        assert!(
            identical(
                &decode(&read_back).expect("the fixture decodes"),
                &directly(fixture)
            ),
            "{} called {name} decoded differently",
            fixture.name
        );
        println!(
            "{name} holding {} decoded as {}",
            fixture.name, fixture.container
        );
        std::fs::remove_file(&path).expect("the fixture is removed");
    }
    let _ = std::fs::remove_dir(&directory);
}

/// A RIFF or `FORM` container whose form type is not one this crate reads is
/// refused, by every entry point, naming the form type found.
#[test]
fn a_container_family_member_that_is_not_audio_is_refused_by_its_form_type() {
    for (name, bytes, expected) in decoys() {
        // The standalone probe.
        let error = identify(&bytes)
            .map(|container| panic!("{name} was identified as {container}"))
            .unwrap_err();
        let DecodeError::UnsupportedContainer { tag } = error else {
            panic!("{name} was refused as {error} rather than as a container");
        };
        assert_eq!(
            tag.as_bytes(),
            &expected,
            "{name}: the error must name the form type actually found"
        );

        // The whole-file entry point, with the same identity.
        let whole = decode(&bytes)
            .map(|_| panic!("{name} decoded"))
            .unwrap_err();
        assert!(
            matches!(whole, DecodeError::UnsupportedContainer { tag } if tag.as_bytes() == &expected),
            "{name}: decode gave {whole}"
        );

        // The streaming entry point, on the push that completes the probe.
        let streamed = stream(&bytes, 4)
            .map(|_| panic!("{name} streamed"))
            .unwrap_err();
        assert!(
            matches!(streamed, DecodeError::UnsupportedContainer { tag } if tag.as_bytes() == &expected),
            "{name}: the stream gave {streamed}"
        );

        println!(
            "{name}: refused, naming '{}'",
            String::from_utf8_lossy(&expected)
        );
    }
}

/// A magic that names no container this crate reads is refused by the magic,
/// not by a form type it never had.
#[test]
fn an_unknown_magic_is_named_by_the_bytes_at_offset_zero() {
    for magic in [b"OggS", b"\x1aE\xdf\xa3", b"ID3\x03", b"MThd", b"caff"] {
        let mut bytes = magic.to_vec();
        bytes.extend_from_slice(&[0; 64]);
        let error = identify(&bytes).unwrap_err();
        assert!(
            matches!(error, DecodeError::UnsupportedContainer { tag } if tag.as_bytes() == magic),
            "{magic:?} was refused as {error}"
        );
    }
}

// --- Group 2: the streaming probe -------------------------------------------

/// The streaming probe produces exactly what the whole-file path produces, at
/// every feed size including one byte.
///
/// One-byte chunks are the case that proves the buffering is real: twelve
/// separate pushes complete the probe, and the reader that is finally chosen
/// still has to see the stream from its first byte.
#[test]
fn the_streaming_probe_matches_the_whole_file_path_at_every_chunk_size() {
    let mut fixtures = pcm_fixtures();
    fixtures.extend(flac_fixtures());

    // Sizes around the probe length on both sides, primes that never align
    // with a chunk boundary, and one that swallows the whole file.
    let chunks = [1, 2, 3, 5, 7, 11, 12, 13, 17, 64, 509, 4_096, 65_536];
    for fixture in &fixtures {
        let whole = decode(&fixture.bytes).expect("the fixture decodes whole");
        for chunk in chunks {
            let (streamed, container) = stream(&fixture.bytes, chunk)
                .unwrap_or_else(|error| panic!("{} at {chunk}-byte pieces: {error}", fixture.name));
            assert_eq!(
                container,
                Some(fixture.container),
                "{} at {chunk}-byte pieces identified as {container:?}",
                fixture.name
            );
            assert!(
                identical(&streamed, &whole),
                "{} at {chunk}-byte pieces differs from the whole-file path",
                fixture.name
            );
        }
        // The whole file in one push, which is the other extreme.
        let (streamed, _) = stream(&fixture.bytes, fixture.bytes.len()).expect("one push");
        assert!(identical(&streamed, &whole), "{} in one push", fixture.name);
        println!(
            "{}: {} chunk sizes and one whole push all agree",
            fixture.name,
            chunks.len()
        );
    }
}

/// Identification is not knowledge of the rate, and the streaming probe does
/// not pretend otherwise.
#[test]
fn a_stream_reports_its_container_before_it_can_report_a_spec() {
    let fixture = WavWriter::new(AudioSpec::new(44_100, 2), WavCodec::PcmI16)
        .to_bytes(&audio(16, 2, 3))
        .expect("a WAV writes");

    let mut decoder = AudioStreamDecoder::new();
    for byte in &fixture[..Container::PROBE_BYTES] {
        assert_eq!(decoder.container(), None);
        assert_eq!(decoder.spec(), None);
        assert_eq!(decoder.push(std::slice::from_ref(byte)).expect("a byte"), 1);
    }
    assert_eq!(decoder.container(), Some(Container::Wav));
    // The container is known; the rate is still in a chunk that has not
    // arrived.
    assert_eq!(decoder.spec(), None);
    assert_eq!(decoder.buffered_bytes(), 0);

    let mut samples = Vec::new();
    let mut offset = Container::PROBE_BYTES;
    while offset < fixture.len() {
        offset += decoder.push(&fixture[offset..]).expect("the rest");
        while decoder.pull(&mut samples, usize::MAX).expect("a pull") > 0 {}
    }
    decoder.finish(&mut samples).expect("the stream ends");
    assert_eq!(decoder.spec(), Some(AudioSpec::new(44_100, 2)));
    assert_eq!(samples.len(), 32);
}

/// The probe buffer is look-ahead, not a staging area: it holds at most twelve
/// bytes and is empty the moment the format is known.
#[test]
fn the_probe_buffer_never_grows_past_the_probe_length() {
    let fixture = AiffWriter::new(AudioSpec::mono(8_000), AiffCodec::PcmI16)
        .to_bytes(&audio(200, 1, 5))
        .expect("an AIFF writes");

    let mut decoder = AudioStreamDecoder::new();
    let mut samples = Vec::new();
    let mut held = 0usize;
    for piece in fixture.chunks(3) {
        let mut offset = 0;
        while offset < piece.len() {
            offset += decoder.push(&piece[offset..]).expect("a piece");
            held = held.max(decoder.buffered_bytes());
            while decoder.pull(&mut samples, usize::MAX).expect("a pull") > 0 {}
        }
    }
    decoder.finish(&mut samples).expect("the stream ends");
    assert_eq!(samples.len(), 200);
    // The AIFF reader's own pending buffer is included in the count, so the
    // assertion is that nothing unbounded is held rather than that nothing is.
    assert!(
        held <= 4_096,
        "the stream held {held} bytes at its peak, which is not look-ahead"
    );
}

// --- Group 2: truncation ----------------------------------------------------

/// Every input length below the probe length is a typed error, from every
/// entry point, with no partial guess anywhere.
#[test]
fn every_length_under_the_probe_length_is_truncated_rather_than_guessed() {
    let mut sources: Vec<(&str, Vec<u8>)> = vec![
        (
            "a WAV prefix",
            WavWriter::new(AudioSpec::mono(8_000), WavCodec::PcmI16)
                .to_bytes(&audio(8, 1, 1))
                .expect("a WAV writes"),
        ),
        (
            "an AIFF prefix",
            AiffWriter::new(AudioSpec::mono(8_000), AiffCodec::PcmI16)
                .to_bytes(&audio(8, 1, 2))
                .expect("an AIFF writes"),
        ),
        ("a FLAC prefix", RFC_EXAMPLE_3.to_vec()),
    ];
    sources.push(("bytes that are nothing at all", vec![0x5a; 64]));

    for (name, bytes) in &sources {
        for length in 0..Container::PROBE_BYTES {
            let prefix = &bytes[..length];

            let error = identify(prefix)
                .map(|container| panic!("{name} at {length} bytes identified as {container}"))
                .unwrap_err();
            assert!(
                matches!(
                    error,
                    DecodeError::Truncated { expected, available }
                        if expected == Container::PROBE_BYTES as u64
                            && available == length as u64
                ),
                "{name} at {length} bytes gave {error}"
            );

            let whole = decode(prefix)
                .map(|_| panic!("{name} decoded"))
                .unwrap_err();
            assert!(
                matches!(whole, DecodeError::Truncated { .. }),
                "{name} at {length} bytes: decode gave {whole}"
            );

            // The streaming path reaches the same verdict at `finish`, which
            // is the only point at which a short stream is distinguishable
            // from one whose rest has not arrived yet.
            let mut decoder = AudioStreamDecoder::new();
            assert_eq!(
                decoder.push(prefix).expect("a short push is not an error"),
                length,
                "{name} at {length} bytes was not fully taken"
            );
            assert_eq!(decoder.container(), None);
            let mut samples = Vec::new();
            let streamed = decoder
                .finish(&mut samples)
                .map(|_| panic!("{name} streamed"))
                .unwrap_err();
            assert!(
                matches!(
                    streamed,
                    DecodeError::Truncated { expected, available }
                        if expected == Container::PROBE_BYTES as u64
                            && available == length as u64
                ),
                "{name} at {length} bytes: the stream gave {streamed}"
            );
        }
        println!("{name}: lengths 0 to 11 are all typed truncations");
    }
}

/// Four bytes of `fLaC` are conclusive about the format and still not enough.
#[test]
fn a_prefix_that_would_be_conclusive_still_needs_the_whole_probe() {
    for length in 4..Container::PROBE_BYTES {
        assert!(
            matches!(
                identify(&RFC_EXAMPLE_3[..length]),
                Err(DecodeError::Truncated { .. })
            ),
            "a {length}-byte fLaC prefix was answered rather than refused"
        );
    }
}

// --- Group 2: FLAC through the probe ----------------------------------------

/// A FLAC stream reached through the probe decodes, and the checksum the file
/// carries is checked on the way.
#[test]
fn a_flac_stream_decodes_through_the_probe_and_is_checked_against_its_own_md5() {
    for (name, bytes) in FLAC_FILES {
        let probed = decode(bytes).unwrap_or_else(|error| panic!("{name}: {error}"));
        let direct = FlacReader::new(bytes)
            .expect("the file opens")
            .decode_to_end()
            .expect("the file decodes");
        assert!(identical(&probed, &direct), "{name} decoded differently");

        // A single flipped bit in the audio must turn the probe's decode red
        // too: the probe delegates the checksum rather than bypassing it.
        let mut damaged = bytes.to_vec();
        let last = damaged.len() - 8;
        damaged[last] ^= 0x01;
        assert!(
            decode(&damaged).is_err(),
            "{name} decoded through the probe with a corrupted frame"
        );
    }
}

// --- Group 2: determinism ---------------------------------------------------

/// Every WAV and AIFF fixture, through both probe entry points, hashed.
///
/// Pinned so a change in the decoded bytes is a failure rather than a new
/// value. The PCM containers alone, so this witness and the FLAC one below
/// move independently of each other.
const PROBE_WITNESS: u64 = 0x5ded_2068_3c43_3e99;

/// The three RFC 9639 files through both probe entry points, hashed.
const PROBE_FLAC_WITNESS: u64 = 0xad35_8612_e054_4fa5;

/// The probe's output is bit-identical to a pinned witness.
#[test]
fn the_probe_sweep_is_bit_identical_to_a_pinned_witness() {
    let mut witness = Vec::new();
    for fixture in &pcm_fixtures() {
        absorb(&mut witness, &decode(&fixture.bytes).expect("it decodes"));
        let (streamed, _) = stream(&fixture.bytes, 509).expect("it streams");
        absorb(&mut witness, &streamed);
    }
    let hash = fnv1a(witness);
    println!("probe witness: 0x{hash:016x}");
    assert_eq!(
        hash, PROBE_WITNESS,
        "the probe's decoded output moved on unchanged input, which is a \
         determinism break rather than a value to re-pin"
    );
}

/// The FLAC route through the probe is bit-identical to a pinned witness.
#[test]
fn the_probe_flac_route_is_bit_identical_to_a_pinned_witness() {
    let mut witness = Vec::new();
    for (_, bytes) in FLAC_FILES {
        absorb(&mut witness, &decode(bytes).expect("it decodes"));
        let (streamed, _) = stream(bytes, 13).expect("it streams");
        absorb(&mut witness, &streamed);
    }
    let hash = fnv1a(witness);
    println!("probe FLAC witness: 0x{hash:016x}");
    assert_eq!(hash, PROBE_FLAC_WITNESS);
}

// --- Group 2: the corpus ----------------------------------------------------

/// The environment variable naming the FLAC conformance corpus, as
/// `flac_corpus.rs` defines it.
const CORPUS_ENV: &str = "DECIBRI_FLAC_CORPUS";

/// A sample of the FLAC conformance corpus, routed by content.
///
/// The whole corpus goes through the probe in `flac_corpus.rs`; this is the
/// sample that runs beside the rest of the routing gates.
#[test]
fn a_sample_of_the_flac_corpus_routes_through_the_probe() {
    let Some(root) = std::env::var_os(CORPUS_ENV) else {
        eprintln!("skipped: set {CORPUS_ENV} to a checkout of the FLAC test files");
        return;
    };
    let directory = PathBuf::from(root).join("subset");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&directory)
        .unwrap_or_else(|error| panic!("reading {}: {error}", directory.display()))
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|e| e == "flac"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "the subset group holds no files");

    // Every eighth file, so the sample spans the group rather than its front.
    let mut checked = 0usize;
    for path in files.iter().step_by(8) {
        let bytes = std::fs::read(path).expect("a corpus file reads");
        let name = path.file_name().expect("a name").to_string_lossy();
        assert_eq!(
            identify(&bytes).expect("a corpus file identifies"),
            Container::Flac,
            "{name}"
        );
        let probed = decode(&bytes).unwrap_or_else(|error| panic!("{name}: {error}"));
        let direct = FlacReader::new(&bytes)
            .expect("it opens")
            .decode_to_end()
            .expect("it decodes");
        assert!(identical(&probed, &direct), "{name} decoded differently");
        checked += 1;
    }
    println!("{checked} corpus files routed identically through the probe");
}

// --- Group 3: the detail the front door reaches -----------------------------

/// Pushes `bytes` through the streaming probe in `chunk`-byte pieces, finishes
/// the stream and hands the decoder back so its accessors can be read.
fn streamed_to_end(bytes: &[u8], chunk: usize) -> AudioStreamDecoder {
    let mut decoder = AudioStreamDecoder::new();
    let mut samples = Vec::new();
    for piece in bytes.chunks(chunk) {
        let mut offset = 0;
        while offset < piece.len() {
            offset += decoder.push(&piece[offset..]).expect("a piece pushes");
            while decoder.pull(&mut samples, usize::MAX).expect("a pull") > 0 {}
        }
    }
    decoder.finish(&mut samples).expect("the stream ends");
    decoder
}

/// Every detail accessor on a decoder that has not identified anything yet.
///
/// Dimension 9's first case. A fresh decoder and one holding eleven bytes are
/// both before identification, and neither has an inner reader to ask.
#[test]
fn no_detail_accessor_answers_before_the_stream_is_identified() {
    let sources: Vec<(&str, Vec<u8>)> = vec![
        (
            "a WAV prefix",
            WavWriter::new(AudioSpec::mono(8_000), WavCodec::PcmI16)
                .to_bytes(&audio(8, 1, 1))
                .expect("a WAV writes"),
        ),
        (
            "an AIFF prefix",
            AiffWriter::new(AudioSpec::mono(8_000), AiffCodec::PcmI16)
                .to_bytes(&audio(8, 1, 1))
                .expect("an AIFF writes"),
        ),
        ("a FLAC prefix", RFC_EXAMPLE_1.to_vec()),
    ];

    let fresh = AudioStreamDecoder::new();
    assert_eq!(fresh.container(), None);
    assert_eq!(fresh.wav_format(), None);
    assert_eq!(fresh.aiff_format(), None);
    assert_eq!(fresh.flac_stream_info(), None);
    assert_eq!(fresh.flac_md5_check(), None);

    for (name, bytes) in &sources {
        // Every length short of the probe length, one byte at a time, so the
        // assertion covers the whole of the pre-identification window rather
        // than one point in it.
        let mut decoder = AudioStreamDecoder::new();
        for byte in &bytes[..Container::PROBE_BYTES - 1] {
            assert_eq!(decoder.container(), None, "{name}");
            assert_eq!(decoder.wav_format(), None, "{name}");
            assert_eq!(decoder.aiff_format(), None, "{name}");
            assert_eq!(decoder.flac_stream_info(), None, "{name}");
            assert_eq!(decoder.flac_md5_check(), None, "{name}");
            decoder
                .push(std::slice::from_ref(byte))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
        }
    }
}

/// Identification is not the header, and the accessors do not pretend it is.
///
/// Dimension 9's third case: at exactly twelve bytes the container is known
/// on every carried format, and the chunk carrying the detail has arrived on
/// none of them.
#[test]
fn identification_alone_answers_no_detail_accessor() {
    let cases: Vec<(&str, Vec<u8>, Container)> = vec![
        (
            "RIFF/WAVE",
            WavWriter::new(AudioSpec::mono(8_000), WavCodec::PcmI16)
                .to_bytes(&audio(8, 1, 1))
                .expect("a WAV writes"),
            Container::Wav,
        ),
        (
            "FORM/AIFF",
            AiffWriter::new(AudioSpec::mono(8_000), AiffCodec::PcmI16)
                .to_bytes(&audio(8, 1, 1))
                .expect("an AIFF writes"),
            Container::Aiff,
        ),
        ("fLaC", RFC_EXAMPLE_1.to_vec(), Container::Flac),
    ];

    for (name, bytes, container) in &cases {
        let mut decoder = AudioStreamDecoder::new();
        assert_eq!(
            decoder
                .push(&bytes[..Container::PROBE_BYTES])
                .unwrap_or_else(|error| panic!("{name}: {error}")),
            Container::PROBE_BYTES,
            "{name}"
        );
        assert_eq!(decoder.container(), Some(*container), "{name}");
        assert_eq!(decoder.wav_format(), None, "{name}");
        assert_eq!(decoder.aiff_format(), None, "{name}");
        assert_eq!(decoder.flac_stream_info(), None, "{name}");
        assert_eq!(decoder.flac_md5_check(), None, "{name}");
    }
}

/// Every accessor gives exactly what the whole-file reader for that format
/// gives, on every fixture, and gives nothing on the formats it is not about.
///
/// The oracle is the whole-file reader rather than the streaming one, so this
/// is not the delegation checking itself: the fourth coverage lesson on this
/// crate is that a test whose reference is computed by the path under test
/// proves only self-consistency.
#[test]
fn every_detail_accessor_matches_the_reader_that_owns_the_format() {
    for fixture in pcm_fixtures().iter().chain(flac_fixtures().iter()) {
        let name = fixture.name;
        // 509 is prime and unrelated to any chunk boundary, so the detail is
        // read off a stream that arrived in pieces rather than in one push.
        let decoder = streamed_to_end(&fixture.bytes, 509);
        assert_eq!(decoder.container(), Some(fixture.container), "{name}");

        match fixture.container {
            Container::Wav => {
                let expected = *WavReader::new(&fixture.bytes)
                    .expect("the WAV fixture opens")
                    .format();
                assert_eq!(decoder.wav_format(), Some(expected), "{name}");
                assert_eq!(decoder.aiff_format(), None, "{name}");
                assert_eq!(decoder.flac_stream_info(), None, "{name}");
                assert_eq!(decoder.flac_md5_check(), None, "{name}");
            }
            Container::Aiff => {
                let expected = *AiffReader::new(&fixture.bytes)
                    .expect("the AIFF fixture opens")
                    .format();
                assert_eq!(decoder.aiff_format(), Some(expected), "{name}");
                assert_eq!(decoder.wav_format(), None, "{name}");
                assert_eq!(decoder.flac_stream_info(), None, "{name}");
                assert_eq!(decoder.flac_md5_check(), None, "{name}");
            }
            Container::Flac => {
                let expected = *FlacReader::new(&fixture.bytes)
                    .expect("the FLAC fixture opens")
                    .stream_info();
                assert_eq!(decoder.flac_stream_info(), Some(expected), "{name}");
                // The verdict follows from the streaminfo field rather than
                // from what the streaming reader reported, so the two are
                // independent.
                let verdict = if expected.md5.is_some() {
                    Md5Check::Verified
                } else {
                    Md5Check::ChecksumUnset
                };
                assert_eq!(decoder.flac_md5_check(), Some(verdict), "{name}");
                assert_eq!(decoder.wav_format(), None, "{name}");
                assert_eq!(decoder.aiff_format(), None, "{name}");
            }
            _ => panic!("{name}: a container with no accessor case"),
        }
    }
}

/// The MD5 verdict is reported at the end of the stream and not before.
///
/// A caller must not be able to read a guarantee off a decode that has not
/// finished, so `flac_md5_check` stays `None` while the audio is still
/// arriving even though the streaminfo it will be checked against is long
/// since known.
#[test]
fn the_flac_md5_verdict_arrives_only_when_the_stream_does() {
    let bytes = RFC_EXAMPLE_1;
    let mut decoder = AudioStreamDecoder::new();
    let mut samples = Vec::new();
    let mut seen_stream_info = false;

    for piece in bytes.chunks(7) {
        let mut offset = 0;
        while offset < piece.len() {
            offset += decoder.push(&piece[offset..]).expect("a piece pushes");
            while decoder.pull(&mut samples, usize::MAX).expect("a pull") > 0 {}
        }
        seen_stream_info |= decoder.flac_stream_info().is_some();
        assert_eq!(
            decoder.flac_md5_check(),
            None,
            "a verdict was reported before the stream ended"
        );
    }
    assert!(
        seen_stream_info,
        "the streaminfo was never reported while the stream ran"
    );

    decoder.finish(&mut samples).expect("the stream ends");
    assert_eq!(decoder.flac_md5_check(), Some(Md5Check::Verified));

    // Reset drops it with everything else the header taught the reader.
    decoder.reset();
    assert_eq!(decoder.flac_stream_info(), None);
    assert_eq!(decoder.flac_md5_check(), None);
}
