//! The one error type this crate returns.

use std::fmt;

use decibri_resampler::ResamplerError;

use crate::codec::{CodecId, FourCc};

/// Every way decoding can fail.
///
/// The variant list is closed by design rather than grown as codecs land. Each
/// variant decibri surfaces has to be registered by name, code and sample
/// instance in the `error_identity!` table in decibri's `error.rs`, and that
/// macro has no catch-all arm, so a variant added later is a change in two
/// crates and a new stable code on decibri's frozen error surface. Settling the
/// list while the crate is empty costs nothing; growing it later costs a
/// version bump in both crates.
///
/// The four `Unsupported*` variants are split rather than folded into one for
/// the reason recorded on [`CodecId`]: "unsupported" alone does not tell a
/// caller which encoding they hit or what to do about it.
///
/// This enum is `#[non_exhaustive]`: consumers pattern-matching on it must
/// include a `_ =>` catch-all arm so future variant additions are not
/// source-breaking.
#[derive(Debug)]
#[non_exhaustive]
pub enum DecodeError {
    /// The container format was not recognised.
    ///
    /// Carries the identifying bytes that were seen at the start of the input,
    /// so a caller can tell "this is not audio at all" from "this is audio in a
    /// container this build does not carry".
    UnsupportedContainer {
        /// The leading bytes read where a container magic was expected.
        tag: FourCc,
    },

    /// The container parsed and named a codec this build does not support.
    ///
    /// Distinct from [`UnsupportedContainer`](Self::UnsupportedContainer): the
    /// file is well-formed and its structure was understood. Only the payload
    /// encoding is out of reach.
    UnsupportedCodec {
        /// The codec identity exactly as the container stated it.
        codec: CodecId,
    },

    /// A recognised codec in a sample format this build does not support.
    ///
    /// Distinct from [`UnsupportedCodec`](Self::UnsupportedCodec): the codec
    /// itself is carried, but not at this width or in this representation,
    /// 24-bit PCM where only 16- and 32-bit are implemented, for instance.
    UnsupportedSampleFormat {
        /// The format tag, in the form the container stated it.
        format: CodecId,
        /// Bits per sample as declared by the container.
        bits_per_sample: u16,
    },

    /// The channel count or layout is not one this build can decode.
    UnsupportedChannelLayout {
        /// The channel count the container declared.
        channels: u16,
    },

    /// A parse failure: the bytes at a known position were not what the format
    /// requires there.
    ///
    /// Distinct from [`Truncated`](Self::Truncated), which is about running out
    /// of input rather than finding the wrong input.
    Malformed {
        /// What the parser required at that position, as a short literal.
        expected: &'static str,
        /// Byte offset from the start of the input where the parse failed.
        offset: u64,
    },

    /// The input ended part-way through a frame, a header or a declared
    /// length.
    ///
    /// From a file this means the file is short. From a stream it means the
    /// stream was finished while a frame was still incomplete: a stream that
    /// simply has not received the rest yet is not an error, and the stream
    /// adapter holds the partial frame instead of reporting one.
    Truncated {
        /// How much of the incomplete item was needed. Bytes on every path
        /// but one: a FLAC stream holding fewer samples than its streaminfo
        /// declares reports the two counts in interchannel samples, because
        /// samples are the unit the declaration is in.
        expected: u64,
        /// How much was actually available, in the same unit as `expected`.
        available: u64,
    },

    /// The container declares one codec and the payload is another.
    ///
    /// Reached when both identities parse and disagree, a WAVE header claiming
    /// integer PCM over a payload whose frames are mu-law, say. Reported
    /// separately from the `Unsupported*` variants because nothing here is
    /// unsupported: the file is internally inconsistent and no single decoder
    /// choice is correct.
    ContainerCodecMismatch {
        /// What the container's header said the payload is.
        declared: CodecId,
        /// What the payload actually turned out to be.
        found: CodecId,
    },

    /// Rate conversion failed.
    ///
    /// The only variant that chains a source. This crate contains no resampling
    /// logic; it calls `decibri-resampler` and passes its failure through
    /// unaltered.
    Resample(ResamplerError),
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedContainer { tag } => {
                write!(f, "unrecognised container, leading bytes were '{tag}'")
            }
            Self::UnsupportedCodec { codec } => {
                write!(f, "container names {codec}, which this build cannot decode")
            }
            Self::UnsupportedSampleFormat {
                format,
                bits_per_sample,
            } => write!(
                f,
                "{format} at {bits_per_sample} bits per sample is not a supported sample format"
            ),
            Self::UnsupportedChannelLayout { channels } => {
                write!(f, "{channels}-channel audio is not a supported layout")
            }
            Self::Malformed { expected, offset } => {
                write!(f, "malformed input at byte {offset}: expected {expected}")
            }
            Self::Truncated {
                expected,
                available,
            } => write!(
                f,
                "input ended early: {expected} needed, {available} available"
            ),
            Self::ContainerCodecMismatch { declared, found } => write!(
                f,
                "container declares {declared} but the payload is {found}"
            ),
            Self::Resample(source) => write!(f, "resampling failed: {source}"),
        }
    }
}

impl std::error::Error for DecodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Resample(source) => Some(source),
            _ => None,
        }
    }
}

impl From<ResamplerError> for DecodeError {
    fn from(source: ResamplerError) -> Self {
        Self::Resample(source)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One instance of every variant. If a variant is added without a line
    /// here the match below stops compiling, which is the point: no variant
    /// gets to be unreachable, and every one of them has a `Display` string
    /// somebody has looked at.
    fn one_of_each() -> Vec<DecodeError> {
        vec![
            DecodeError::UnsupportedContainer {
                tag: FourCc(*b"\x00\x01\x02\x03"),
            },
            DecodeError::UnsupportedCodec {
                codec: CodecId::WaveFormatTag(0x0011),
            },
            DecodeError::UnsupportedSampleFormat {
                format: CodecId::WaveFormatTag(1),
                bits_per_sample: 24,
            },
            DecodeError::UnsupportedChannelLayout { channels: 8 },
            DecodeError::Malformed {
                expected: "a 'data' chunk header",
                offset: 36,
            },
            DecodeError::Truncated {
                expected: 4096,
                available: 1200,
            },
            DecodeError::ContainerCodecMismatch {
                declared: CodecId::WaveFormatTag(1),
                found: CodecId::WaveFormatTag(7),
            },
            DecodeError::Resample(ResamplerError::ProcessAfterFlush),
        ]
    }

    #[test]
    fn every_variant_is_constructible() {
        // The exhaustive match is the assertion: adding a variant to the enum
        // without adding it to `one_of_each` fails to compile here.
        for error in one_of_each() {
            match error {
                DecodeError::UnsupportedContainer { .. }
                | DecodeError::UnsupportedCodec { .. }
                | DecodeError::UnsupportedSampleFormat { .. }
                | DecodeError::UnsupportedChannelLayout { .. }
                | DecodeError::Malformed { .. }
                | DecodeError::Truncated { .. }
                | DecodeError::ContainerCodecMismatch { .. }
                | DecodeError::Resample(_) => {}
            }
        }
    }

    #[test]
    fn display_names_the_specific_thing() {
        assert_eq!(
            DecodeError::UnsupportedSampleFormat {
                format: CodecId::WaveFormatTag(1),
                bits_per_sample: 24,
            }
            .to_string(),
            "WAVE format tag 0x0001 at 24 bits per sample is not a supported sample format"
        );
        assert_eq!(
            DecodeError::UnsupportedCodec {
                codec: CodecId::WaveFormatTag(7),
            }
            .to_string(),
            "container names WAVE format tag 0x0007, which this build cannot decode"
        );
        // Three encodings decibri's current WAV reader rejects with one shared
        // reason string. Here they are three distinguishable messages.
        let mulaw = DecodeError::UnsupportedCodec {
            codec: CodecId::WaveFormatTag(7),
        }
        .to_string();
        let alaw = DecodeError::UnsupportedCodec {
            codec: CodecId::WaveFormatTag(6),
        }
        .to_string();
        let pcm24 = DecodeError::UnsupportedSampleFormat {
            format: CodecId::WaveFormatTag(1),
            bits_per_sample: 24,
        }
        .to_string();
        assert_ne!(mulaw, alaw);
        assert_ne!(alaw, pcm24);
        assert_ne!(mulaw, pcm24);
    }

    #[test]
    fn no_display_string_is_empty_or_generic() {
        for error in one_of_each() {
            let text = error.to_string();
            assert!(!text.is_empty());
            assert!(
                !text.eq_ignore_ascii_case("unsupported"),
                "variant reported a bare category: {text}"
            );
        }
    }

    #[test]
    fn only_the_resample_variant_chains_a_source() {
        use std::error::Error as _;

        for error in one_of_each() {
            let chains = error.source().is_some();
            assert_eq!(
                chains,
                matches!(error, DecodeError::Resample(_)),
                "source chaining is for the Resample variant only: {error}"
            );
        }
    }

    #[test]
    fn resampler_errors_convert() {
        let error: DecodeError = ResamplerError::ZeroSampleRate.into();
        assert!(matches!(error, DecodeError::Resample(_)));
        assert_eq!(
            error.to_string(),
            "resampling failed: sample rate must be greater than 0"
        );
    }

    #[test]
    fn the_error_crosses_threads() {
        // Decoders run off the audio thread, so their failures have to be
        // sendable. Asserted at compile time.
        fn assert_send_sync<T: Send + Sync + 'static>() {}
        assert_send_sync::<DecodeError>();
    }
}
