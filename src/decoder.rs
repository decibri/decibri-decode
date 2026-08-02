//! The decoder boundary: bytes in, samples out.

use crate::audio::AudioSpec;
use crate::error::DecodeError;

/// Turns encoded frames into `f32` samples.
///
/// # Why this shape
///
/// The trait is a *pull* model, feed bytes and then pull whatever samples are
/// ready, rather than a chunk-in-chunk-out function. Nothing in the pure-PCM
/// codecs needs that: they are stateless and a byte slice maps to a sample
/// slice. MP3 does. Its bit reservoir lets a frame borrow bits from earlier
/// frames, so a decoder must be able to accept bytes that produce nothing yet
/// and produce samples from bytes it accepted earlier. Establishing that here
/// costs almost nothing; establishing it once four codecs are written above a
/// chunk-in-chunk-out API means rewriting all of them.
///
/// The trait is [object safe](https://doc.rust-lang.org/reference/items/traits.html#object-safety),
/// because the decoder for a given input is chosen at run time from what the
/// container turned out to declare, and because platform decoders, AudioToolbox
/// on iOS, MediaCodec on Android, both hardware-accelerated and already licensed
/// by the platform vendor, must be able to sit behind the same `dyn Decoder`
/// as the crate's own. That is the single most expensive property here to
/// retrofit.
///
/// Implementations are [`Send`], so an instance can be constructed on one
/// thread and driven on another.
///
/// # Contract
///
/// - [`output_spec`](Decoder::output_spec) is fixed for the life of the
///   instance and known at construction, because a decoder is built by whatever
///   parsed the header. A stream whose header has not arrived has no decoder
///   yet; see [`StreamSource::spec`](crate::StreamSource::spec).
/// - [`feed`](Decoder::feed) may consume less than it is offered and may
///   produce nothing. [`decode`](Decoder::decode) may produce samples from
///   bytes fed in earlier calls. Neither is one-in-one-out and neither assumes
///   the caller holds the whole input.
/// - Where a byte lands in the feed sequence makes no difference to the output.
///   Feeding a file in one slice, in 4096-byte slices, or one byte at a time
///   produces the same samples in the same order.
/// - After [`flush`](Decoder::flush), the decoder is at end of stream:
///   `feed` consumes nothing and `decode` produces nothing until
///   [`reset`](Decoder::reset). This mirrors
///   `decibri_resampler::Resampler`, except that it is a quiet no-op rather
///   than an error, because [`DecodeError`] has no variant for driving a
///   decoder wrongly and is not going to grow one.
///
/// # Sample-count guarantee
///
/// For an input the container describes as `n` frames, the totals across a
/// full `feed`/`decode`/`flush` sequence come to exactly `n * channels`
/// samples. Not approximately, and not "plus whatever the last partial frame
/// yields". A codec that cannot state its output count for a given input does
/// not belong behind this trait.
pub trait Decoder: Send {
    /// The rate and layout of the samples this decoder produces.
    ///
    /// Fixed for the life of the instance. This is the authority for the rate
    /// of everything [`decode`](Decoder::decode) and [`flush`](Decoder::flush)
    /// append; pair the two with
    /// [`AudioBuffer::from_samples`](crate::AudioBuffer::from_samples) at the
    /// point where the samples travel any further.
    fn output_spec(&self) -> AudioSpec;

    /// Hands `input` to the decoder and returns how many of its bytes were
    /// taken.
    ///
    /// A short return means the decoder's internal buffer is full: drain it
    /// with [`decode`](Decoder::decode) and offer the remainder again. A return
    /// of `input.len()` does not mean any samples are ready, and a return of
    /// `0` does not mean the input is bad.
    ///
    /// A partial frame at the end of `input` is held, not rejected. It is only
    /// an error if the stream then ends; see [`flush`](Decoder::flush).
    ///
    /// # Errors
    ///
    /// Returns [`DecodeError::Malformed`] when the accepted bytes cannot be a
    /// valid continuation of the stream at all. Running out of bytes is not an
    /// error here.
    fn feed(&mut self, input: &[u8]) -> Result<usize, DecodeError>;

    /// Appends every sample the decoder can produce from what it has been fed,
    /// and returns how many it appended.
    ///
    /// `output` is appended to, never cleared, and the caller owns it, the
    /// same convention as `decibri_resampler::Resampler::process`, so a
    /// decoder's output vector feeds a resampler with no copy in between.
    ///
    /// A return of `0` means "starved, feed me more", not "end of stream".
    /// Callers loop `decode` until it returns `0`, then `feed` again.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeError::Malformed`] when a buffered frame turns out to be
    /// invalid once enough of it has arrived to judge.
    fn decode(&mut self, output: &mut Vec<f32>) -> Result<usize, DecodeError>;

    /// Declares end of input, appends any remaining samples and returns how
    /// many were appended.
    ///
    /// Call it once the last byte has been fed. Codecs with a decode tail emit
    /// it here; codecs without one append nothing and return `0`. It is
    /// idempotent.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeError::Truncated`] when bytes of an incomplete frame are
    /// still held. That is the one place a partial frame becomes an error:
    /// while the stream is open it is simply data that has not arrived yet.
    fn flush(&mut self, output: &mut Vec<f32>) -> Result<usize, DecodeError>;

    /// Returns the decoder to its just-constructed state.
    ///
    /// Buffered bytes, undelivered samples and any end-of-stream condition set
    /// by [`flush`](Decoder::flush) are dropped. The output spec is unchanged.
    /// Use it to decode a second stream on the same instance, or to resume
    /// after a seek.
    fn reset(&mut self);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::AudioBuffer;

    /// A decoder whose "codec" is the identity: four input bytes are one
    /// output sample, read back as the `f32` they already were.
    ///
    /// It exists to prove the trait can actually be implemented and used
    /// through `dyn Decoder`. A trait that compiles but cannot be implemented
    /// is the failure this fixture catches. It is `#[cfg(test)]` and never
    /// ships in the crate proper, because this crate contains no codec.
    struct PassThrough {
        spec: AudioSpec,
        /// Bytes fed but not yet forming a whole sample. Never more than three.
        pending: Vec<u8>,
        /// Samples decoded but not yet handed back.
        ready: Vec<f32>,
        finished: bool,
    }

    impl PassThrough {
        fn new(spec: AudioSpec) -> Self {
            Self {
                spec,
                pending: Vec::new(),
                ready: Vec::new(),
                finished: false,
            }
        }
    }

    impl Decoder for PassThrough {
        fn output_spec(&self) -> AudioSpec {
            self.spec
        }

        fn feed(&mut self, input: &[u8]) -> Result<usize, DecodeError> {
            if self.finished {
                return Ok(0);
            }
            self.pending.extend_from_slice(input);
            let whole = self.pending.len() / 4 * 4;
            for chunk in self.pending[..whole].chunks_exact(4) {
                let bytes: [u8; 4] = chunk.try_into().expect("chunks_exact(4) yields 4 bytes");
                self.ready.push(f32::from_ne_bytes(bytes));
            }
            self.pending.drain(..whole);
            Ok(input.len())
        }

        fn decode(&mut self, output: &mut Vec<f32>) -> Result<usize, DecodeError> {
            if self.finished {
                return Ok(0);
            }
            let produced = self.ready.len();
            output.append(&mut self.ready);
            Ok(produced)
        }

        fn flush(&mut self, output: &mut Vec<f32>) -> Result<usize, DecodeError> {
            if self.finished {
                return Ok(0);
            }
            self.finished = true;
            if !self.pending.is_empty() {
                return Err(DecodeError::Truncated {
                    expected: 4,
                    available: self.pending.len() as u64,
                });
            }
            let produced = self.ready.len();
            output.append(&mut self.ready);
            Ok(produced)
        }

        fn reset(&mut self) {
            self.pending.clear();
            self.ready.clear();
            self.finished = false;
        }
    }

    fn encode(samples: &[f32]) -> Vec<u8> {
        samples.iter().flat_map(|s| s.to_ne_bytes()).collect()
    }

    /// Drives a decoder to completion over `input` split into `chunk` byte
    /// pieces, the way a caller is expected to.
    fn drive(
        decoder: &mut dyn Decoder,
        input: &[u8],
        chunk: usize,
    ) -> Result<AudioBuffer, DecodeError> {
        let mut samples = Vec::new();
        for piece in input.chunks(chunk) {
            let mut offset = 0;
            while offset < piece.len() {
                offset += decoder.feed(&piece[offset..])?;
                while decoder.decode(&mut samples)? > 0 {}
            }
        }
        decoder.flush(&mut samples)?;
        Ok(AudioBuffer::from_samples(decoder.output_spec(), samples))
    }

    #[test]
    fn the_trait_is_object_safe() {
        // Decoders are selected at run time, so this has to hold.
        let boxed: Box<dyn Decoder> = Box::new(PassThrough::new(AudioSpec::mono(16_000)));
        assert_eq!(boxed.output_spec(), AudioSpec::mono(16_000));

        let mut decoders: Vec<Box<dyn Decoder>> =
            vec![Box::new(PassThrough::new(AudioSpec::new(48_000, 2)))];
        let by_ref: &mut dyn Decoder = decoders[0].as_mut();
        assert_eq!(by_ref.output_spec().channels, 2);
    }

    #[test]
    fn a_decoder_can_be_moved_between_threads() {
        fn assert_send<T: Send>() {}
        assert_send::<PassThrough>();
        assert_send::<Box<dyn Decoder>>();
    }

    #[test]
    fn samples_survive_the_round_trip() {
        let expected = [0.0_f32, 1.0, -1.0, 0.5, -0.25];
        let mut decoder = PassThrough::new(AudioSpec::mono(16_000));
        let decoded = drive(&mut decoder, &encode(&expected), 1024).expect("decode");
        assert_eq!(decoded.samples(), expected);
        assert_eq!(decoded.sample_rate(), 16_000);
        assert_eq!(decoded.frames(), 5);
    }

    #[test]
    fn the_chunk_boundary_is_invisible() {
        let expected: Vec<f32> = (0..64).map(|i| i as f32 / 64.0).collect();
        let encoded = encode(&expected);
        // 1, 3 and 7 all split samples mid-way; 4 lands on the boundary.
        for chunk in [1, 3, 4, 7, 256] {
            let mut decoder = PassThrough::new(AudioSpec::mono(8_000));
            let decoded = drive(&mut decoder, &encoded, chunk).expect("decode");
            assert_eq!(
                decoded.samples(),
                expected.as_slice(),
                "output changed with a {chunk}-byte feed size"
            );
        }
    }

    #[test]
    fn the_sample_count_is_exact() {
        let frames = 100;
        let spec = AudioSpec::new(44_100, 2);
        let encoded = encode(&vec![0.125_f32; frames * spec.channels as usize]);
        let mut decoder = PassThrough::new(spec);
        let decoded = drive(&mut decoder, &encoded, 13).expect("decode");
        assert_eq!(decoded.samples().len(), frames * spec.channels as usize);
        assert_eq!(decoded.frames(), frames);
    }

    #[test]
    fn a_starved_decoder_reports_zero_rather_than_end_of_stream() {
        let mut decoder = PassThrough::new(AudioSpec::mono(16_000));
        let mut out = Vec::new();
        assert_eq!(decoder.decode(&mut out).expect("decode"), 0);

        // Three bytes is not yet a sample. Still not an error.
        assert_eq!(decoder.feed(&[1, 2, 3]).expect("feed"), 3);
        assert_eq!(decoder.decode(&mut out).expect("decode"), 0);
        assert!(out.is_empty());

        // The fourth byte completes it.
        assert_eq!(decoder.feed(&[4]).expect("feed"), 1);
        assert_eq!(decoder.decode(&mut out).expect("decode"), 1);
    }

    #[test]
    fn a_partial_frame_at_end_of_stream_is_truncated() {
        let mut decoder = PassThrough::new(AudioSpec::mono(16_000));
        decoder.feed(&[0, 0, 0]).expect("feed");
        let mut out = Vec::new();
        let error = decoder.flush(&mut out).expect_err("flush must reject");
        assert!(
            matches!(
                error,
                DecodeError::Truncated {
                    expected: 4,
                    available: 3
                }
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn feeding_after_flush_is_quiet_until_reset() {
        let mut decoder = PassThrough::new(AudioSpec::mono(16_000));
        let mut out = Vec::new();
        assert_eq!(decoder.flush(&mut out).expect("flush"), 0);
        // Idempotent.
        assert_eq!(decoder.flush(&mut out).expect("second flush"), 0);
        assert_eq!(decoder.feed(&encode(&[1.0])).expect("feed"), 0);
        assert_eq!(decoder.decode(&mut out).expect("decode"), 0);
        assert!(out.is_empty());

        decoder.reset();
        assert_eq!(decoder.feed(&encode(&[1.0])).expect("feed"), 4);
        assert_eq!(decoder.decode(&mut out).expect("decode"), 1);
        assert_eq!(out, [1.0]);
    }

    #[test]
    fn reset_drops_a_held_partial_frame() {
        let mut decoder = PassThrough::new(AudioSpec::mono(16_000));
        decoder.feed(&[9, 9, 9]).expect("feed");
        decoder.reset();
        let mut out = Vec::new();
        assert_eq!(decoder.flush(&mut out).expect("flush after reset"), 0);
    }
}
