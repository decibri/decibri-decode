//! Coverage-guided fuzzing of the three writers, [`WavWriter`],
//! [`AiffWriter`] and [`FlacWriter`].
//!
//! A writer's input is samples, not a file, so the bytes are read as `f32`
//! rather than parsed. **The sample values are the point.** Reading four
//! bytes at a time as an IEEE 754 binary32 produces NaN, both infinities,
//! subnormals and values far outside minus one to plus one naturally and in
//! quantity, which is what a generator of plausible audio never does. The
//! crate documents that the float targets pass non-finite values through, so
//! those values reach an encoder rather than a rejection.
//!
//! # How the input is read
//!
//! A nine-byte prefix, then the samples:
//!
//! | Bytes | Meaning |
//! |---|---|
//! | 0 | which writer, taken modulo three |
//! | 1 | the codec, taken modulo the writer's table, or the FLAC bit depth |
//! | 2 | the FLAC compression level, or the WAV header style and RIFF flavour |
//! | 3..5 | the channel count, little-endian |
//! | 5..9 | the sample rate, little-endian |
//! | 9.. | the samples, four bytes each, little-endian, the last group padded |
//!
//! The bit depth, the level, the channel count and the sample rate are taken
//! whole rather than folded into a legal range, so the rejections each writer
//! documents are reachable rather than designed out.
//!
//! A typed error is correct behaviour here and is the common case, so nothing
//! is asserted about the result. A panic, a hang or a sanitizer report is the
//! finding.

#![no_main]

use decibri_decode::{
    AiffCodec, AiffWriter, AudioSpec, FlacWriter, RiffFlavour, WavCodec, WavHeaderStyle, WavWriter,
};
use libfuzzer_sys::fuzz_target;

/// Every encoding [`WavWriter`] writes.
const WAV_CODECS: [WavCodec; 8] = [
    WavCodec::PcmU8,
    WavCodec::PcmI16,
    WavCodec::PcmI24,
    WavCodec::PcmI32,
    WavCodec::Float32,
    WavCodec::Float64,
    WavCodec::ALaw,
    WavCodec::MuLaw,
];

/// Every encoding [`AiffWriter`] writes, across both of its forms.
const AIFF_CODECS: [AiffCodec; 12] = [
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

/// How many leading bytes describe the file rather than the audio.
const PREFIX: usize = 9;

fuzz_target!(|data: &[u8]| {
    if data.len() < PREFIX {
        return;
    }
    let (prefix, body) = data.split_at(PREFIX);
    let select = prefix[0];
    let codec = usize::from(prefix[1]);
    let setting = prefix[2];
    let channels = u16::from_le_bytes([prefix[3], prefix[4]]);
    let sample_rate = u32::from_le_bytes([prefix[5], prefix[6], prefix[7], prefix[8]]);
    let spec = AudioSpec::new(sample_rate, channels);

    // Four bytes at a time, with the last group zero-padded rather than
    // dropped, so no byte of the input is unreachable.
    let samples: Vec<f32> = body
        .chunks(4)
        .map(|piece| {
            let mut bytes = [0u8; 4];
            bytes[..piece.len()].copy_from_slice(piece);
            f32::from_le_bytes(bytes)
        })
        .collect();

    let mut out = Vec::new();
    match select % 3 {
        0 => {
            let header = if setting & 1 == 0 {
                WavHeaderStyle::Plain
            } else {
                WavHeaderStyle::Extensible
            };
            let flavour = if setting & 2 == 0 {
                RiffFlavour::Automatic
            } else {
                RiffFlavour::Rf64
            };
            let _ = WavWriter::new(spec, WAV_CODECS[codec % WAV_CODECS.len()])
                .with_header_style(header)
                .with_flavour(flavour)
                .write(&samples, &mut out);
        }
        1 => {
            let _ = AiffWriter::new(spec, AIFF_CODECS[codec % AIFF_CODECS.len()])
                .write(&samples, &mut out);
        }
        // The bit depth and the level go in whole, so the depths outside 4
        // through 32 and the levels above 8 reach their rejections.
        _ => {
            let _ = FlacWriter::new(spec, prefix[1])
                .with_level(setting)
                .write(&samples, &mut out);
        }
    }
});
