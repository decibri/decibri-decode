//! The container-to-codec bridge the streaming readers hold by value.
//!
//! A container's streaming reader owns the decoder for its payload, and it
//! needs the decoder's buffered-byte count to stay reachable, which
//! `Box<dyn Decoder>` would hide. Both linear PCM and G.711 payloads exist in
//! both containers this crate reads, so the enum lives here rather than in
//! either of them: WAV and AIFF construct it from their own resolved codecs
//! and drive it identically.

use crate::audio::AudioSpec;
use crate::decoder::Decoder;
use crate::error::DecodeError;
use crate::g711::{G711Decoder, G711Law};
use crate::pcm::PcmDecoder;
use crate::sample::SampleFormat;

/// The payload decoder, held by value so its buffered-byte count stays
/// reachable.
#[derive(Debug)]
pub(crate) enum Payload {
    Pcm(PcmDecoder),
    G711(G711Decoder),
}

impl Payload {
    /// A decoder for a payload that is either linear PCM or companded, at
    /// `spec`.
    ///
    /// Every codec either container resolves is exactly one of the two, so
    /// `(None, None)` is unreachable from real callers; it falls back to `U8`
    /// rather than panicking because a panic on untrusted input is the failure
    /// class this crate refuses.
    pub(crate) fn from_parts(
        format: Option<SampleFormat>,
        law: Option<G711Law>,
        spec: AudioSpec,
    ) -> Self {
        match (format, law) {
            (Some(format), _) => Self::Pcm(PcmDecoder::new(format, spec)),
            (_, Some(law)) => Self::G711(G711Decoder::new(law, spec)),
            _ => Self::Pcm(PcmDecoder::new(SampleFormat::U8, spec)),
        }
    }

    pub(crate) fn feed(&mut self, input: &[u8]) -> Result<usize, DecodeError> {
        match self {
            Self::Pcm(decoder) => decoder.feed(input),
            Self::G711(decoder) => decoder.feed(input),
        }
    }

    pub(crate) fn decode(&mut self, output: &mut Vec<f32>) -> Result<usize, DecodeError> {
        match self {
            Self::Pcm(decoder) => decoder.decode(output),
            Self::G711(decoder) => decoder.decode(output),
        }
    }

    pub(crate) fn flush(&mut self, output: &mut Vec<f32>) -> Result<usize, DecodeError> {
        match self {
            Self::Pcm(decoder) => decoder.flush(output),
            Self::G711(decoder) => decoder.flush(output),
        }
    }

    pub(crate) fn buffered_bytes(&self) -> usize {
        match self {
            Self::Pcm(decoder) => decoder.buffered_bytes(),
            Self::G711(decoder) => decoder.buffered_bytes(),
        }
    }
}
