//! The whole-file integration path: encoded bytes of any supported container
//! in, mono `f32` at a target rate the caller declares, out.
//!
//! This is the shape a consumer holding a complete file uses. The stages after
//! the decode are deliberately visible rather than wrapped in a convenience
//! function, because each one is a decision the consumer owns: whether to
//! downmix, and what rate to resample to. This crate makes neither of those
//! decisions, has no rate policy and no default sample rate, and nothing in it
//! branches on a file name. Choosing the container reader is *not* one of
//! those decisions, which is why it is one call.
//!
//! Run against a real WAV, AIFF or FLAC file:
//!
//! ```text
//! cargo run --example decode_whole_file -- path/to/audio.wav
//! ```
//!
//! or with no argument to run against a WAV built in memory.

use decibri_decode::decibri_resampler::{PolyphaseResampler, Resampler};
use decibri_decode::{
    decode, downmix_to_mono, identify, AiffReader, AudioBuffer, AudioSpec, Container, DecodeError,
    FlacReader, WavCodec, WavReader, WavWriter,
};

/// The target rate the caller declares. Nothing in this crate supplies one.
const TARGET_RATE: u32 = 16_000;

/// Decodes a complete file of any supported format to mono `f32` at
/// `target_rate`.
///
/// The container comes out of the content, never out of a file name or an
/// extension, and the crate does that part: [`decode`] reads the leading twelve
/// bytes, picks the reader and rejects anything it does not carry with a typed
/// error that names what it found.
fn decode_to_mono_at(bytes: &[u8], target_rate: u32) -> Result<AudioBuffer, DecodeError> {
    // 1. Container to interleaved samples, at the file's own rate and layout.
    //    The rate travels with the samples from here on; nothing downstream
    //    infers it.
    let decoded = decode(bytes)?;

    // 2. Interleaved to mono. The channel count comes from the buffer itself,
    //    and the downmix is the arithmetic mean of the channels, the same
    //    formula decibri uses, read from its source rather than chosen.
    let mut mono = Vec::new();
    downmix_to_mono(decoded.samples(), decoded.channels(), &mut mono);

    // 3. The file's rate to the declared target rate, through the one
    //    resampler this crate defers all rate conversion to. When the rates
    //    already match, the resampler is an identity passthrough and the
    //    samples come back bit-identical with nothing appended at the flush.
    let mut resampler = PolyphaseResampler::new(decoded.sample_rate(), target_rate)?;
    let mut samples = Vec::new();
    resampler.process(&mono, &mut samples)?;
    resampler.flush(&mut samples);

    Ok(AudioBuffer::from_samples(
        AudioSpec::mono(target_rate),
        samples,
    ))
}

/// What a probe of the input establishes before any sample is decoded: the
/// file's own rate and channel count, and how many frames it holds.
///
/// [`identify`] gives the container and nothing else, which is what makes the
/// reader-specific part of this short. The rate and the count are the
/// container's own to state, so they come from the reader that understands it;
/// FLAC is the one that may honestly not know its length, and says `None`
/// rather than nought.
fn probe(bytes: &[u8]) -> Result<(AudioSpec, Option<u64>), DecodeError> {
    match identify(bytes)? {
        Container::Aiff => {
            let reader = AiffReader::new(bytes)?;
            Ok((reader.spec(), Some(reader.frames())))
        }
        Container::Flac => {
            let reader = FlacReader::new(bytes)?;
            Ok((reader.spec(), reader.frames()))
        }
        // `Container` is non-exhaustive, so this arm is required. It covers
        // WAV, and anything a later version of the crate adds, which the WAV
        // reader refuses by naming the bytes it saw.
        _ => {
            let reader = WavReader::new(bytes)?;
            Ok((reader.spec(), Some(reader.frames())))
        }
    }
}

/// A two-channel 16-bit PCM WAV at 44.1 kHz, built in memory.
///
/// The signal is integer arithmetic scaled by powers of two, not a
/// transcendental, so every sample is exactly representable and the file is
/// byte-identical on every platform.
fn demo_wav() -> Result<Vec<u8>, DecodeError> {
    const FRAMES: usize = 4_410;
    let mut samples = Vec::with_capacity(FRAMES * 2);
    for frame in 0..FRAMES {
        let value = ((frame * 37) % 2_000) as f32 / 2_048.0 - 0.488_281_25;
        samples.push(value); // left
        samples.push(-value * 0.5); // right
    }
    WavWriter::new(AudioSpec::new(44_100, 2), WavCodec::PcmI16).to_bytes(&samples)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bytes = match std::env::args_os().nth(1) {
        Some(path) => std::fs::read(path)?,
        None => demo_wav()?,
    };

    let (spec, frames) = probe(&bytes)?;
    let counted = match frames {
        Some(frames) => frames.to_string(),
        None => "an unstated number of".to_string(),
    };
    println!(
        "input: {} channel(s) at {} Hz, {counted} frames",
        spec.channels, spec.sample_rate
    );

    let audio = decode_to_mono_at(&bytes, TARGET_RATE)?;
    println!(
        "output: {} samples of mono f32 at {} Hz",
        audio.samples().len(),
        audio.sample_rate()
    );

    // The crate's count guarantee, checked on the identity path: with the
    // target equal to the file's own rate the resampler is a passthrough, so
    // the output holds exactly the frame count the container declared. The
    // resampled count above is the resampler's statement, not this crate's,
    // which is why this check runs at the identity rate. A container that did
    // not state a count has nothing to check against and says so.
    if let Some(frames) = frames {
        let identity = decode_to_mono_at(&bytes, spec.sample_rate)?;
        assert_eq!(
            identity.samples().len() as u64,
            frames,
            "identity-rate decode must hold exactly the container's frame count"
        );
        println!("identity-rate decode holds exactly {frames} frames");
    }

    Ok(())
}
