//! Decoded audio, and the description that travels with it.

/// The sample rate and channel count of a block of audio.
///
/// Rate and layout are one value, not two, so they cannot be passed separately
/// and cannot drift apart. decibri has been bitten repeatedly by a rate
/// travelling independently of the samples it describes; the most expensive
/// instance was the AEC reference path, where a reference at the wrong rate is
/// accepted without complaint and cancels nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AudioSpec {
    /// Samples per second, per channel.
    pub sample_rate: u32,
    /// Number of interleaved channels.
    pub channels: u16,
}

impl AudioSpec {
    /// Describes audio at `sample_rate` with `channels` interleaved channels.
    pub const fn new(sample_rate: u32, channels: u16) -> Self {
        Self {
            sample_rate,
            channels,
        }
    }

    /// Describes single-channel audio at `sample_rate`.
    ///
    /// The shape the crate contract promises at its output: mono `f32` at a
    /// declared rate.
    pub const fn mono(sample_rate: u32) -> Self {
        Self::new(sample_rate, 1)
    }

    /// `true` when this describes single-channel audio.
    pub const fn is_mono(&self) -> bool {
        self.channels == 1
    }

    /// How many whole frames `sample_count` interleaved samples make up.
    ///
    /// Returns `0` for a spec with no channels rather than dividing by zero.
    pub const fn frames(&self, sample_count: usize) -> usize {
        if self.channels == 0 {
            0
        } else {
            sample_count / self.channels as usize
        }
    }
}

/// Decoded audio together with the [`AudioSpec`] that describes it.
///
/// Samples are `f32`, interleaved when there is more than one channel, and are
/// nominally in `-1.0..=1.0`. Formats whose integer range maps outside that
/// interval are not clamped by the decoder; clamping is a policy decision for
/// whoever consumes the audio.
///
/// This type is what leaves the crate. A [`Decoder`](crate::Decoder) appends
/// into a plain `Vec<f32>` because it is the caller's own decoder and its
/// [`output_spec`](crate::Decoder::output_spec) is authoritative, but at every
/// boundary where audio travels further than that, it travels as an
/// `AudioBuffer` so no downstream stage has to infer a rate.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioBuffer {
    spec: AudioSpec,
    samples: Vec<f32>,
}

impl AudioBuffer {
    /// An empty buffer described by `spec`.
    pub const fn new(spec: AudioSpec) -> Self {
        Self {
            spec,
            samples: Vec::new(),
        }
    }

    /// An empty buffer described by `spec`, with room for `capacity` samples.
    pub fn with_capacity(spec: AudioSpec, capacity: usize) -> Self {
        Self {
            spec,
            samples: Vec::with_capacity(capacity),
        }
    }

    /// Binds `samples` to the `spec` that describes them.
    ///
    /// The pairing is not checked, because there is nothing to check it
    /// against: `spec` is the assertion. Use it at the one point where the
    /// rate is known, immediately after pulling samples from the decoder that
    /// reported it.
    pub const fn from_samples(spec: AudioSpec, samples: Vec<f32>) -> Self {
        Self { spec, samples }
    }

    /// The rate and layout of the samples held here.
    pub const fn spec(&self) -> AudioSpec {
        self.spec
    }

    /// Samples per second, per channel.
    pub const fn sample_rate(&self) -> u32 {
        self.spec.sample_rate
    }

    /// Number of interleaved channels.
    pub const fn channels(&self) -> u16 {
        self.spec.channels
    }

    /// The interleaved samples.
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    /// The interleaved samples, mutably.
    pub fn samples_mut(&mut self) -> &mut [f32] {
        &mut self.samples
    }

    /// The number of whole frames held.
    pub fn frames(&self) -> usize {
        self.spec.frames(self.samples.len())
    }

    /// `true` when no samples are held.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Appends interleaved samples, which must already be at this buffer's
    /// rate and layout.
    pub fn extend_from_slice(&mut self, samples: &[f32]) {
        self.samples.extend_from_slice(samples);
    }

    /// Drops the samples, keeping the spec and the allocation.
    pub fn clear(&mut self) {
        self.samples.clear();
    }

    /// Takes the samples out, discarding the description.
    ///
    /// Named to be conspicuous at a call site: past this point the rate no
    /// longer travels with the audio, which is the situation this type exists
    /// to prevent.
    pub fn into_samples(self) -> Vec<f32> {
        self.samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_spec_carries_rate_and_layout_together() {
        let spec = AudioSpec::new(48_000, 2);
        assert_eq!(spec.sample_rate, 48_000);
        assert_eq!(spec.channels, 2);
        assert!(!spec.is_mono());
        assert!(AudioSpec::mono(16_000).is_mono());
    }

    #[test]
    fn frames_divide_by_the_channel_count() {
        assert_eq!(AudioSpec::new(48_000, 2).frames(1000), 500);
        // A trailing partial frame is not a frame.
        assert_eq!(AudioSpec::new(48_000, 2).frames(1001), 500);
        assert_eq!(AudioSpec::mono(48_000).frames(1000), 1000);
    }

    #[test]
    fn a_zero_channel_spec_reports_no_frames_rather_than_dividing_by_zero() {
        assert_eq!(AudioSpec::new(48_000, 0).frames(1000), 0);
        let buffer = AudioBuffer::from_samples(AudioSpec::new(48_000, 0), vec![0.0; 8]);
        assert_eq!(buffer.frames(), 0);
    }

    #[test]
    fn a_buffer_keeps_its_spec_through_every_mutation() {
        let spec = AudioSpec::mono(8_000);
        let mut buffer = AudioBuffer::with_capacity(spec, 4);
        assert!(buffer.is_empty());

        buffer.extend_from_slice(&[0.25, -0.25]);
        assert_eq!(buffer.spec(), spec);
        assert_eq!(buffer.sample_rate(), 8_000);
        assert_eq!(buffer.channels(), 1);
        assert_eq!(buffer.frames(), 2);
        assert_eq!(buffer.samples(), &[0.25, -0.25]);

        buffer.samples_mut()[0] = 0.5;
        assert_eq!(buffer.samples(), &[0.5, -0.25]);

        buffer.clear();
        assert!(buffer.is_empty());
        assert_eq!(buffer.spec(), spec);
    }

    #[test]
    fn samples_can_be_taken_out_and_bound_back_to_a_spec() {
        let spec = AudioSpec::mono(16_000);
        let buffer = AudioBuffer::from_samples(spec, vec![1.0, 2.0, 3.0]);
        let samples = buffer.into_samples();
        let rebound = AudioBuffer::from_samples(spec, samples);
        assert_eq!(rebound.samples(), &[1.0, 2.0, 3.0]);
        assert_eq!(rebound.spec(), spec);
    }

    #[test]
    fn an_empty_buffer_still_declares_a_rate() {
        let buffer = AudioBuffer::new(AudioSpec::mono(44_100));
        assert!(buffer.is_empty());
        assert_eq!(buffer.sample_rate(), 44_100);
    }
}
