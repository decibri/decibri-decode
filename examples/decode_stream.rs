//! The streaming integration path: bytes in as they arrive, mono `f32` at a
//! target rate declared up front, out as soon as it is ready.
//!
//! The target rate is stated before the first byte of the file exists, which
//! is the real shape of a live consumer: it knows what rate its pipeline runs
//! at, and it does not know what rate the file will turn out to be. The file's
//! own rate becomes known part-way through, when the header has arrived,
//! and that is the moment the resampler is constructed. Nothing here defaults
//! a rate, and nothing waits for the whole file.
//!
//! The example ends by decoding the same bytes through the whole-file path and
//! asserting the two outputs are bit-identical. That agreement is guaranteed,
//! not incidental: piece boundaries are invisible to the streaming decoder,
//! and chunk boundaries are invisible to the resampler, so the two paths are
//! the same arithmetic in a different delivery order.
//!
//! Neither path names a container. `AudioStreamDecoder` holds the leading
//! twelve bytes until it can tell what they are, then delegates to the reader
//! for that format; `decode` does the same for the whole file. The demo below
//! happens to build a WAV, and nothing in the pipeline knows that.

use decibri_decode::decibri_resampler::{PolyphaseResampler, Resampler};
use decibri_decode::{
    decode, downmix_to_mono, AudioBuffer, AudioSpec, AudioStreamDecoder, DecodeError, StreamSource,
    WavCodec, WavWriter,
};

/// The target rate the caller declares up front. Nothing in this crate
/// supplies one.
const TARGET_RATE: u32 = 16_000;

/// Everything the streaming path holds between arriving pieces.
struct StreamPipeline {
    target_rate: u32,
    decoder: AudioStreamDecoder,
    /// Built the moment the header arrives and the file's own rate is known.
    /// `None` until then: a stream cannot be resampled from a rate it has
    /// not been told.
    resampler: Option<PolyphaseResampler>,
    /// Interleaved frames pulled from the decoder, reused between pieces.
    batch: Vec<f32>,
    /// The downmix of one batch, reused between pieces.
    mono: Vec<f32>,
    /// The mono output at `target_rate`, growing as pieces arrive.
    samples: Vec<f32>,
}

impl StreamPipeline {
    fn new(target_rate: u32) -> Self {
        Self {
            target_rate,
            decoder: AudioStreamDecoder::new(),
            resampler: None,
            batch: Vec::new(),
            mono: Vec::new(),
            samples: Vec::new(),
        }
    }

    /// Hands one arriving piece to the pipeline, in whatever size it turned
    /// up. A short return from the decoder is back-pressure, answered by
    /// draining and offering the remainder again.
    fn push(&mut self, mut piece: &[u8]) -> Result<(), DecodeError> {
        while !piece.is_empty() {
            let taken = self.decoder.push(piece)?;
            piece = &piece[taken..];
            self.drain()?;
        }
        Ok(())
    }

    /// Pulls every frame the decoder has ready and sends it through downmix
    /// and resample. Before the header has arrived there is no spec and
    /// nothing to pull, and this is a quiet no-op.
    fn drain(&mut self) -> Result<(), DecodeError> {
        let Some(spec) = self.decoder.spec() else {
            return Ok(());
        };
        loop {
            self.batch.clear();
            if self.decoder.pull(&mut self.batch, usize::MAX)? == 0 {
                break;
            }
            self.convert(spec)?;
        }
        Ok(())
    }

    /// Downmixes the current batch and resamples it onto the output. `pull`
    /// and `finish` deliver whole frames only, so the downmix never sees a
    /// partial frame.
    fn convert(&mut self, spec: AudioSpec) -> Result<(), DecodeError> {
        self.mono.clear();
        downmix_to_mono(&self.batch, spec.channels, &mut self.mono);
        if self.resampler.is_none() {
            self.resampler = Some(PolyphaseResampler::new(spec.sample_rate, self.target_rate)?);
        }
        let resampler = self
            .resampler
            .as_mut()
            .expect("constructed on the line above");
        resampler.process(&self.mono, &mut self.samples)?;
        Ok(())
    }

    /// Declares end of stream, drains the decoder's tail and the resampler's
    /// tail, and returns the output bound to the spec that describes it.
    fn finish(mut self) -> Result<AudioBuffer, DecodeError> {
        self.batch.clear();
        self.decoder.finish(&mut self.batch)?;
        if let Some(spec) = self.decoder.spec() {
            if !self.batch.is_empty() {
                self.convert(spec)?;
            }
        }
        if let Some(resampler) = self.resampler.as_mut() {
            resampler.flush(&mut self.samples);
        }
        Ok(AudioBuffer::from_samples(
            AudioSpec::mono(self.target_rate),
            self.samples,
        ))
    }
}

/// A two-channel 32-bit float WAV at 48 kHz, built in memory.
///
/// The signal is integer arithmetic scaled by powers of two, not a
/// transcendental, so every sample is exactly representable and the file is
/// byte-identical on every platform.
fn demo_wav() -> Result<Vec<u8>, DecodeError> {
    const FRAMES: usize = 4_800;
    let mut samples = Vec::with_capacity(FRAMES * 2);
    for frame in 0..FRAMES {
        let value = ((frame * 41) % 1_024) as f32 / 1_024.0 - 0.5;
        samples.push(value); // left
        samples.push(value * 0.25); // right
    }
    WavWriter::new(AudioSpec::new(48_000, 2), WavCodec::Float32).to_bytes(&samples)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file = demo_wav()?;

    // The target rate is declared here, before the first byte is pushed.
    let mut pipeline = StreamPipeline::new(TARGET_RATE);

    // 509 is deliberately awkward: prime, so it splits the RIFF header, the
    // fmt chunk and every frame across piece boundaries somewhere in the file.
    for piece in file.chunks(509) {
        pipeline.push(piece)?;
    }
    let streamed = pipeline.finish()?;
    println!(
        "streamed: {} samples of mono f32 at {} Hz",
        streamed.samples().len(),
        streamed.sample_rate()
    );

    // The same bytes through the whole-file path, for the agreement check.
    let decoded = decode(&file)?;
    let mut mono = Vec::new();
    downmix_to_mono(decoded.samples(), decoded.channels(), &mut mono);
    let mut resampler = PolyphaseResampler::new(decoded.sample_rate(), TARGET_RATE)?;
    let mut whole = Vec::new();
    resampler.process(&mono, &mut whole)?;
    resampler.flush(&mut whole);

    assert_eq!(
        streamed.samples().len(),
        whole.len(),
        "the two paths must produce the same sample count"
    );
    assert!(
        streamed
            .samples()
            .iter()
            .zip(&whole)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "the two paths must agree bit-for-bit"
    );
    println!("streamed output is bit-identical to the whole-file path");

    Ok(())
}
