//! The decoder for headerless linear PCM.

use crate::audio::AudioSpec;
use crate::decoder::Decoder;
use crate::error::DecodeError;
use crate::sample::{SampleFormat, MAX_BYTES_PER_SAMPLE};

/// How many decoded samples accumulate before [`feed`](Decoder::feed) stops
/// taking bytes.
///
/// 65,536 `f32` is 256 KiB. A caller feeding ordinary 4 KiB blocks and draining
/// between them never reaches it; a caller handing over a whole file in one call
/// gets back-pressure and a bounded decoder rather than a buffer the size of the
/// file. This is the short-return path the [`Decoder`] contract describes, and
/// the reason it is implemented rather than assumed is that this is the first
/// real implementation of that contract. A clause nothing exercises is a clause
/// nobody has checked.
const READY_LIMIT: usize = 65_536;

/// Decodes a headerless linear PCM stream: bytes in one of the
/// [`SampleFormat`]s, samples out at a [`AudioSpec`] the caller states.
///
/// Headerless PCM is what a microphone produces and what several streaming
/// transports carry, so nothing declares its rate, layout or sample format:
/// they arrive out of band and are given at construction. Both are then fixed
/// for the life of the instance, as [`Decoder::output_spec`] requires.
///
/// # Partial samples and partial frames
///
/// A `feed` that ends part-way through a sample holds the leftover bytes until
/// the rest arrive. Three bytes of a four-byte `f32` are not an error while the
/// stream is open, and the boundary a caller happens to split its buffers on is
/// not something a decoder gets to have an opinion about: feeding a stream in
/// one slice, in 4096-byte slices or one byte at a time produces the same
/// samples in the same order.
///
/// [`flush`](Decoder::flush) is where an incomplete item becomes
/// [`DecodeError::Truncated`], and it counts *frames*, not samples. A stereo
/// stream that ends after an odd number of samples ended mid-frame, and
/// reporting that as a clean end would hand the caller a buffer whose length is
/// not a whole number of frames.
///
/// # Example
///
/// ```
/// use decibri_decode::{AudioSpec, Decoder, PcmDecoder, SampleFormat};
///
/// let mut decoder = PcmDecoder::new(SampleFormat::I16Le, AudioSpec::mono(16_000));
/// let mut samples = Vec::new();
///
/// // 0x8000 is -1.0; 0x4000 is 0.5. The second sample arrives split in two.
/// decoder.feed(&[0x00, 0x80, 0x00])?;
/// decoder.decode(&mut samples)?;
/// assert_eq!(samples, [-1.0]);
///
/// decoder.feed(&[0x40])?;
/// decoder.decode(&mut samples)?;
/// decoder.flush(&mut samples)?;
/// assert_eq!(samples, [-1.0, 0.5]);
/// # Ok::<(), decibri_decode::DecodeError>(())
/// ```
#[derive(Debug)]
pub struct PcmDecoder {
    /// The format of the bytes fed in.
    format: SampleFormat,
    /// The rate and layout of the samples produced.
    spec: AudioSpec,
    /// Bytes of a sample that has not fully arrived. Never more than one
    /// sample's worth, which is why this is an array and not a `Vec`.
    pending: [u8; MAX_BYTES_PER_SAMPLE],
    /// How many of `pending` are live.
    pending_len: usize,
    /// Samples decoded but not yet handed to a caller.
    ready: Vec<f32>,
    /// Samples produced since construction or the last `reset`, which is what
    /// says whether the stream is on a frame boundary.
    produced: u64,
    /// Set by `flush`: end of stream until `reset`.
    finished: bool,
}

impl PcmDecoder {
    /// A decoder for `format` bytes producing samples at `spec`.
    ///
    /// Nothing is validated because nothing can be: a headerless stream carries
    /// no claim to check the arguments against. `spec` is the assertion.
    pub const fn new(format: SampleFormat, spec: AudioSpec) -> Self {
        Self {
            format,
            spec,
            pending: [0; MAX_BYTES_PER_SAMPLE],
            pending_len: 0,
            ready: Vec::new(),
            produced: 0,
            finished: false,
        }
    }

    /// The format this decoder reads.
    pub const fn format(&self) -> SampleFormat {
        self.format
    }

    /// How many bytes are held awaiting the rest of their frame.
    ///
    /// `0` means the stream is on a frame boundary and could be cut here
    /// without loss, the same meaning the figure carries on
    /// [`StreamSource::buffered_bytes`](crate::StreamSource::buffered_bytes),
    /// and the same figure [`flush`](Decoder::flush) reports as `available`
    /// when it rejects a truncated stream. It counts the bytes of an incomplete
    /// sample *and* any whole samples sitting past the last frame boundary.
    pub fn buffered_bytes(&self) -> usize {
        let channels = u64::from(self.spec.channels).max(1);
        let inside_frame = (self.produced % channels) as usize;
        inside_frame * self.format.bytes_per_sample() + self.pending_len
    }

    /// How many bytes one whole frame occupies.
    ///
    /// A spec with no channels has no frame at all, so one sample stands in for
    /// one: it keeps the truncation report meaningful instead of dividing by
    /// zero, and matches [`AudioSpec::frames`] answering `0` rather than
    /// panicking.
    fn frame_bytes(&self) -> usize {
        usize::from(self.spec.channels).max(1) * self.format.bytes_per_sample()
    }

    /// Decodes whole samples out of `bytes` into `ready`, keeping the running
    /// count that `buffered_bytes` reads.
    fn push_samples(&mut self, bytes: &[u8]) {
        self.produced += self.format.decode(bytes, &mut self.ready) as u64;
    }
}

impl Decoder for PcmDecoder {
    fn output_spec(&self) -> AudioSpec {
        self.spec
    }

    fn feed(&mut self, input: &[u8]) -> Result<usize, DecodeError> {
        if self.finished {
            return Ok(0);
        }
        // Room is counted in samples, so a format's width decides how many
        // bytes that is. At zero, the caller has to drain before feeding again.
        let mut room = READY_LIMIT.saturating_sub(self.ready.len());
        if room == 0 {
            return Ok(0);
        }

        let width = self.format.bytes_per_sample();
        let mut taken = 0;

        // A sample left half-arrived by an earlier feed is completed from the
        // head of this one, before anything else is looked at.
        if self.pending_len > 0 {
            let take = (width - self.pending_len).min(input.len());
            self.pending[self.pending_len..self.pending_len + take].copy_from_slice(&input[..take]);
            self.pending_len += take;
            taken = take;
            if self.pending_len < width {
                // Still short. Hold it; the rest is not late, it is unsent.
                return Ok(taken);
            }
            let completed = self.pending;
            self.push_samples(&completed[..width]);
            self.pending_len = 0;
            room -= 1;
        }

        let rest = &input[taken..];
        let whole = (rest.len() / width).min(room) * width;
        self.push_samples(&rest[..whole]);
        taken += whole;

        // Whatever is left is either a partial sample, which is held, or bytes
        // that did not fit in `room`, which are left for the caller to offer
        // again once they have drained.
        let tail = &input[taken..];
        if tail.len() < width {
            self.pending[..tail.len()].copy_from_slice(tail);
            self.pending_len = tail.len();
            taken += tail.len();
        }
        Ok(taken)
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
        let held = self.buffered_bytes();
        if held > 0 {
            // The one place a partial frame is an error. A caller following the
            // documented loop has already drained everything whole through
            // `decode`, so nothing complete is lost with the rejection.
            return Err(DecodeError::Truncated {
                expected: self.frame_bytes() as u64,
                available: held as u64,
            });
        }
        let produced = self.ready.len();
        output.append(&mut self.ready);
        Ok(produced)
    }

    fn reset(&mut self) {
        self.pending_len = 0;
        self.ready.clear();
        self.produced = 0;
        self.finished = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::AudioBuffer;
    use crate::sample::{I24_MAX, I24_MIN};

    const ALL: [SampleFormat; 12] = [
        SampleFormat::U8,
        SampleFormat::I8,
        SampleFormat::I16Le,
        SampleFormat::I16Be,
        SampleFormat::I24Le,
        SampleFormat::I24Be,
        SampleFormat::I32Le,
        SampleFormat::I32Be,
        SampleFormat::F32Le,
        SampleFormat::F32Be,
        SampleFormat::F64Le,
        SampleFormat::F64Be,
    ];

    /// Drives a decoder to completion over `input` split into `chunk`-byte
    /// pieces, the way the trait documents a caller should.
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

    fn encode(format: SampleFormat, samples: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::new();
        format.encode(samples, &mut bytes);
        bytes
    }

    #[test]
    fn a_headerless_stream_decodes_at_the_spec_it_was_given() {
        let mut decoder = PcmDecoder::new(SampleFormat::I16Le, AudioSpec::mono(8_000));
        assert_eq!(decoder.format(), SampleFormat::I16Le);
        assert_eq!(decoder.output_spec(), AudioSpec::mono(8_000));

        // Hand-written: 0x0000, 0x4000, 0x8000, 0xC000 little-endian.
        let bytes = [0x00, 0x00, 0x00, 0x40, 0x00, 0x80, 0x00, 0xc0];
        let decoded = drive(&mut decoder, &bytes, 64).expect("decode");
        assert_eq!(decoded.samples(), [0.0, 0.5, -1.0, -0.5]);
        assert_eq!(decoded.sample_rate(), 8_000);
        assert_eq!(decoded.frames(), 4);
    }

    /// The property the whole partial-sample machinery exists for: where a byte
    /// lands in the feed sequence makes no difference to the output. One byte
    /// at a time splits every format mid-sample; the primes split the wide ones
    /// at every possible offset.
    #[test]
    fn the_feed_boundary_is_invisible_for_every_format() {
        let samples: Vec<f32> = (0..64).map(|i| (i as f32 / 32.0) - 1.0).collect();
        for format in ALL {
            let bytes = encode(format, &samples);
            let reference = {
                let mut decoder = PcmDecoder::new(format, AudioSpec::mono(16_000));
                drive(&mut decoder, &bytes, bytes.len()).expect("whole-input decode")
            };
            for chunk in [1, 2, 3, 5, 7, 8, 13, 64] {
                let mut decoder = PcmDecoder::new(format, AudioSpec::mono(16_000));
                let decoded = drive(&mut decoder, &bytes, chunk).expect("chunked decode");
                assert_eq!(
                    decoded.samples(),
                    reference.samples(),
                    "{format:?} changed with a {chunk}-byte feed size"
                );
            }
        }
    }

    /// A partial sample is held rather than rejected, across as many feeds as
    /// it takes. Eight bytes of `f64` delivered one at a time produce nothing
    /// until the eighth.
    #[test]
    fn a_partial_sample_is_held_until_the_rest_arrives() {
        let mut decoder = PcmDecoder::new(SampleFormat::F64Be, AudioSpec::mono(48_000));
        let bytes = [0x3f, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]; // 1.0
        let mut out = Vec::new();

        for (index, byte) in bytes.iter().enumerate() {
            assert_eq!(decoder.feed(&[*byte]).expect("feed"), 1);
            let produced = decoder.decode(&mut out).expect("decode");
            if index < 7 {
                assert_eq!(produced, 0, "byte {index} completed a sample too early");
                assert_eq!(decoder.buffered_bytes(), index + 1);
            } else {
                assert_eq!(produced, 1, "the last byte must complete the sample");
                assert_eq!(decoder.buffered_bytes(), 0);
            }
        }
        assert_eq!(out, [1.0]);
        assert_eq!(decoder.flush(&mut out).expect("flush"), 0);
    }

    /// A sample split across a feed boundary is reassembled from both halves,
    /// not dropped and not decoded twice.
    #[test]
    fn a_sample_split_across_two_feeds_is_reassembled() {
        let mut decoder = PcmDecoder::new(SampleFormat::I24Le, AudioSpec::mono(44_100));
        let mut out = Vec::new();
        // 0x7FFFFF, the largest positive 24-bit value, split 2 + 1.
        assert_eq!(decoder.feed(&[0xff, 0xff]).expect("feed"), 2);
        assert_eq!(decoder.decode(&mut out).expect("decode"), 0);
        assert_eq!(decoder.buffered_bytes(), 2);
        assert_eq!(decoder.feed(&[0x7f]).expect("feed"), 1);
        assert_eq!(decoder.decode(&mut out).expect("decode"), 1);
        assert_eq!(out, [8_388_607.0 / 8_388_608.0]);
    }

    /// A feed that spans the end of one pending sample and the start of the
    /// next partial one leaves exactly the right tail behind.
    #[test]
    fn a_feed_completing_one_sample_and_starting_another_holds_only_the_tail() {
        let mut decoder = PcmDecoder::new(SampleFormat::I32Le, AudioSpec::mono(16_000));
        let mut out = Vec::new();
        assert_eq!(decoder.feed(&[0x00, 0x00]).expect("feed"), 2);
        // Completes the first sample (four bytes) and leaves three of the next.
        assert_eq!(
            decoder.feed(&[0x00, 0x40, 0x11, 0x22, 0x33]).expect("feed"),
            5
        );
        assert_eq!(decoder.decode(&mut out).expect("decode"), 1);
        assert_eq!(out, [0.5]);
        assert_eq!(decoder.buffered_bytes(), 3);
    }

    /// End of stream with part of a sample held is the one place a partial item
    /// is an error, and the error names the frame it was short of.
    #[test]
    fn a_partial_sample_at_end_of_stream_is_truncated() {
        let mut decoder = PcmDecoder::new(SampleFormat::I32Be, AudioSpec::mono(16_000));
        decoder.feed(&[0x11, 0x22, 0x33]).expect("feed");
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
        // Idempotent: the second flush is a quiet no-op, not a second error.
        assert_eq!(decoder.flush(&mut out).expect("second flush"), 0);
    }

    /// A stereo stream that ends after an odd number of samples ended
    /// mid-frame. Reporting that as a clean end would hand back a buffer whose
    /// length is not a whole number of frames.
    #[test]
    fn a_partial_frame_at_end_of_stream_is_truncated() {
        let mut decoder = PcmDecoder::new(SampleFormat::I16Le, AudioSpec::new(44_100, 2));
        let mut out = Vec::new();
        // One whole sample: half a stereo frame.
        decoder.feed(&[0x00, 0x40]).expect("feed");
        assert_eq!(decoder.decode(&mut out).expect("decode"), 1);
        assert_eq!(
            decoder.buffered_bytes(),
            2,
            "one sample into a 4-byte frame"
        );

        let error = decoder.flush(&mut out).expect_err("flush must reject");
        assert!(
            matches!(
                error,
                DecodeError::Truncated {
                    expected: 4,
                    available: 2
                }
            ),
            "unexpected error: {error}"
        );

        // And the same stream one sample longer ends cleanly.
        let mut decoder = PcmDecoder::new(SampleFormat::I16Le, AudioSpec::new(44_100, 2));
        let decoded = drive(&mut decoder, &[0x00, 0x40, 0x00, 0xc0], 3).expect("decode");
        assert_eq!(decoded.frames(), 1);
        assert_eq!(decoded.samples(), [0.5, -0.5]);
    }

    /// Three of a stereo frame's four bytes: a partial sample and a partial
    /// frame at once, reported against the frame.
    #[test]
    fn a_partial_sample_inside_a_partial_frame_reports_the_frame() {
        let mut decoder = PcmDecoder::new(SampleFormat::I16Be, AudioSpec::new(48_000, 2));
        decoder.feed(&[0x01, 0x02, 0x03]).expect("feed");
        let mut out = Vec::new();
        while decoder.decode(&mut out).expect("decode") > 0 {}
        assert_eq!(decoder.buffered_bytes(), 3);
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

    /// The sample-count guarantee, stated for this decoder: `n` frames in,
    /// exactly `n * channels` samples out, whatever the feed size.
    #[test]
    fn the_sample_count_is_exact_for_every_format_and_layout() {
        for format in ALL {
            for channels in [1_u16, 2, 6] {
                let frames = 50;
                let spec = AudioSpec::new(48_000, channels);
                let count = frames * usize::from(channels);
                let bytes = encode(format, &vec![0.125; count]);
                let mut decoder = PcmDecoder::new(format, spec);
                let decoded = drive(&mut decoder, &bytes, 13).expect("decode");
                assert_eq!(
                    decoded.samples().len(),
                    count,
                    "{format:?} at {channels} channels produced the wrong count"
                );
                assert_eq!(decoded.frames(), frames);
            }
        }
    }

    /// A starved decoder says `0`, which means "feed me", not "end of stream".
    #[test]
    fn a_starved_decoder_reports_zero_rather_than_end_of_stream() {
        let mut decoder = PcmDecoder::new(SampleFormat::I16Le, AudioSpec::mono(16_000));
        let mut out = Vec::new();
        assert_eq!(decoder.decode(&mut out).expect("decode"), 0);
        assert_eq!(decoder.feed(&[0x01]).expect("feed"), 1);
        assert_eq!(decoder.decode(&mut out).expect("decode"), 0);
        assert!(out.is_empty());
        assert_eq!(decoder.feed(&[0x02]).expect("feed"), 1);
        assert_eq!(decoder.decode(&mut out).expect("decode"), 1);
    }

    #[test]
    fn feeding_after_flush_is_quiet_until_reset() {
        let mut decoder = PcmDecoder::new(SampleFormat::U8, AudioSpec::mono(8_000));
        let mut out = Vec::new();
        assert_eq!(decoder.flush(&mut out).expect("flush"), 0);
        assert_eq!(decoder.feed(&[0x80]).expect("feed"), 0);
        assert_eq!(decoder.decode(&mut out).expect("decode"), 0);
        assert!(out.is_empty());

        decoder.reset();
        assert_eq!(decoder.feed(&[0xff]).expect("feed"), 1);
        assert_eq!(decoder.decode(&mut out).expect("decode"), 1);
        assert_eq!(out, [127.0 / 128.0]);
    }

    /// `reset` drops a held partial sample and the frame position with it, so
    /// the next stream starts on a boundary rather than inheriting one.
    #[test]
    fn reset_drops_a_held_partial_frame() {
        let mut decoder = PcmDecoder::new(SampleFormat::I16Le, AudioSpec::new(16_000, 2));
        decoder.feed(&[0x01, 0x02, 0x03]).expect("feed");
        assert_eq!(decoder.buffered_bytes(), 3);
        decoder.reset();
        assert_eq!(decoder.buffered_bytes(), 0);
        let mut out = Vec::new();
        assert_eq!(decoder.flush(&mut out).expect("flush after reset"), 0);
    }

    /// Back-pressure: a caller that hands over more than the decoder will hold
    /// gets a short return and has to drain. Nothing is lost and the loop still
    /// terminates, the property that makes the short return usable rather than
    /// merely permitted.
    #[test]
    fn a_full_decoder_applies_back_pressure_instead_of_growing() {
        let spec = AudioSpec::mono(16_000);
        let mut decoder = PcmDecoder::new(SampleFormat::U8, spec);
        // Three times what the decoder will hold, so the short return is
        // unavoidable rather than incidental.
        let bytes: Vec<u8> = (0..READY_LIMIT * 3).map(|i| (i % 256) as u8).collect();

        let first = decoder.feed(&bytes).expect("feed");
        assert!(first < bytes.len(), "a full decoder must return short");
        assert_eq!(first, READY_LIMIT);
        assert_eq!(decoder.feed(&bytes[first..]).expect("feed"), 0);

        let decoded = drive(&mut decoder, &bytes, bytes.len()).expect("decode");
        // The first feed's samples are still in there, ahead of the rest.
        assert_eq!(decoded.samples().len(), bytes.len() + first);
    }

    /// Every format decodes what it encoded through the full decoder, at the
    /// values where each format's edges live.
    #[test]
    fn every_format_survives_the_decoder_round_trip() {
        let edges = [0.0_f32, 0.5, -0.5, 1.0, -1.0, 1.0 / 128.0, -0.75];
        for format in ALL {
            let bytes = encode(format, &edges);
            let mut decoder = PcmDecoder::new(format, AudioSpec::mono(16_000));
            let decoded = drive(&mut decoder, &bytes, 5).expect("decode");
            assert_eq!(decoded.samples().len(), edges.len());
            for (original, recovered) in edges.iter().zip(decoded.samples()) {
                // 1/128 is the 8-bit step; every wider format is exact here.
                assert!(
                    (original - recovered).abs() <= 1.0 / 128.0,
                    "{format:?}: {original} came back as {recovered}"
                );
            }
        }
    }

    /// The extreme raw values of each integer format, fed as bytes rather than
    /// produced by this crate's encoder.
    #[test]
    fn the_integer_extremes_decode_to_the_expected_floats() {
        let cases: [(SampleFormat, &[u8], f32); 6] = [
            (SampleFormat::U8, &[0x00], -1.0),
            (SampleFormat::I16Be, &[0x80, 0x00], -1.0),
            (SampleFormat::I16Be, &[0x7f, 0xff], 32767.0 / 32768.0),
            (SampleFormat::I24Be, &[0x80, 0x00, 0x00], -1.0),
            (SampleFormat::I32Be, &[0x80, 0x00, 0x00, 0x00], -1.0),
            (SampleFormat::I32Be, &[0x7f, 0xff, 0xff, 0xff], 1.0),
        ];
        for (format, bytes, expected) in cases {
            let mut decoder = PcmDecoder::new(format, AudioSpec::mono(8_000));
            let decoded = drive(&mut decoder, bytes, 1).expect("decode");
            assert_eq!(decoded.samples(), [expected], "{format:?} on {bytes:02x?}");
        }
        // And the packed 24-bit bounds agree with the constants.
        assert_eq!(I24_MIN, -8_388_608);
        assert_eq!(I24_MAX, 8_388_607);
    }

    #[test]
    fn the_decoder_is_object_safe_and_sendable() {
        fn assert_send<T: Send>() {}
        assert_send::<PcmDecoder>();

        let boxed: Box<dyn Decoder> = Box::new(PcmDecoder::new(
            SampleFormat::F32Le,
            AudioSpec::new(44_100, 2),
        ));
        assert_eq!(boxed.output_spec().channels, 2);
    }

    /// A spec with no channels is degenerate but must not divide by zero.
    #[test]
    fn a_zero_channel_spec_does_not_divide_by_zero() {
        let mut decoder = PcmDecoder::new(SampleFormat::I16Le, AudioSpec::new(16_000, 0));
        let mut out = Vec::new();
        decoder.feed(&[0x00, 0x40]).expect("feed");
        assert_eq!(decoder.decode(&mut out).expect("decode"), 1);
        assert_eq!(decoder.buffered_bytes(), 0);
        assert_eq!(decoder.flush(&mut out).expect("flush"), 0);
    }

    // -- Gate 10: cross-platform determinism ------------------------------

    /// FNV-1a over the little-endian bit patterns of the samples, so the
    /// witness is the bytes themselves and not a float comparison.
    fn fnv1a(samples: &[f32]) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        for sample in samples {
            for byte in sample.to_bits().to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        hash
    }

    /// A fixed signal, built from integer arithmetic and an exact division by a
    /// power of two so it is the same on every target.
    ///
    /// The last two values are past full scale, so the witness covers the
    /// clamping path as well as the ordinary one.
    fn witness_signal() -> Vec<f32> {
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        let mut signal: Vec<f32> = (0..254)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                crate::sample::i32_to_f32((state >> 32) as u32 as i32)
            })
            .collect();
        signal.push(1.5);
        signal.push(-3.0);
        signal
    }

    /// One number that changes if any bit of any decoded sample changes, for a
    /// fixed signal put through every format.
    ///
    /// The claim on this crate is bit-exact, cross-platform, byte-identical
    /// output. Running this on two toolchains and getting the same constant is
    /// the evidence for it; a tolerance-based test would pass on both while the
    /// outputs differed. The constant is pinned rather than recomputed at run
    /// time so that a change shows up as a diff in this file.
    ///
    /// The signal is encoded and decoded rather than fed as arbitrary bytes,
    /// which keeps it a statement about real audio. That used to be load-bearing
    /// as well: arbitrary bytes reach `f64` as a NaN roughly once in a thousand
    /// samples, and the bit pattern a NaN keeps through the `f64` to `f32`
    /// narrowing is not something IEEE 754 pins down. It is no longer a
    /// constraint: that narrowing now normalises NaN to silence, so the witness
    /// would hold over arbitrary bytes too.
    #[test]
    fn decoded_output_is_bit_identical_to_a_pinned_witness() {
        let signal = witness_signal();
        let mut samples = Vec::new();
        for format in ALL {
            let bytes = encode(format, &signal);
            let mut decoder = PcmDecoder::new(format, AudioSpec::mono(16_000));
            // 7 is coprime with every sample width here, so each format is
            // driven across its partial-sample path as well as its whole one.
            let decoded = drive(&mut decoder, &bytes, 7).expect("decode");
            samples.extend_from_slice(decoded.samples());
        }
        assert_eq!(samples.len(), ALL.len() * signal.len());
        // Re-pinned when `I8` joined `ALL`: the witness now covers twelve
        // formats rather than eleven, so its input, and therefore its value,
        // changed. The eleven original formats are unchanged, which the WAV
        // conformance witness attests independently: it hashes the same
        // decode paths and did not move.
        assert_eq!(
            fnv1a(&samples),
            0x57ac_66d6_2a28_b665,
            "decoded output changed"
        );
    }
}
