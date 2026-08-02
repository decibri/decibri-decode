//! Identities for containers, codecs and sample formats.
//!
//! These types exist so that [`DecodeError`](crate::DecodeError) can name the
//! specific thing it rejected rather than a category. The step-0 audit found
//! decibri's WAV reader rejecting mu-law, A-law and 24-bit PCM with the same
//! reason string, leaving a caller unable to tell which unsupported encoding
//! they hit or what to do about it. Every rejection this crate emits carries
//! enough detail to act on.

use std::fmt;

/// A four-character code, as used by RIFF, AIFF, ISO base media and friends to
/// name a chunk, a container or a codec.
///
/// Stored as the four bytes actually seen, never as an interpretation of them.
/// [`Display`](fmt::Display) prints them as ASCII when all four are printable
/// and as `0x` hex otherwise, so a garbage tag is reported as the bytes it was
/// rather than as mojibake.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FourCc(pub [u8; 4]);

impl FourCc {
    /// The four bytes as seen on the wire.
    pub const fn as_bytes(&self) -> &[u8; 4] {
        &self.0
    }
}

impl From<[u8; 4]> for FourCc {
    fn from(bytes: [u8; 4]) -> Self {
        Self(bytes)
    }
}

impl fmt::Display for FourCc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.iter().all(|b| b.is_ascii_graphic() || *b == b' ') {
            // Every byte is printable ASCII, so the tag is readable as text.
            for b in &self.0 {
                write!(f, "{}", *b as char)?;
            }
            Ok(())
        } else {
            write!(
                f,
                "0x{:02x}{:02x}{:02x}{:02x}",
                self.0[0], self.0[1], self.0[2], self.0[3]
            )
        }
    }
}

/// How a container named the codec or sample format of its payload.
///
/// Containers identify codecs in incompatible ways, so this carries the
/// identity in the form it was read in rather than mapping it to a normalised
/// enum that would have to grow a variant for every codec the crate does not
/// support. A caller reading a [`DecodeError`](crate::DecodeError) sees the
/// same value the file contained.
///
/// In WAV the codec identity and the sample-format identity are the same field
/// (`wFormatTag`: 1 is integer PCM, 3 is IEEE float, 6 is A-law, 7 is mu-law),
/// which is why one type serves both.
///
/// This enum is `#[non_exhaustive]`: consumers matching on it must include a
/// `_ =>` catch-all arm so a future container's identity scheme is not
/// source-breaking to add.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CodecId {
    /// A RIFF/WAVE `wFormatTag`, as read from the `fmt ` chunk.
    WaveFormatTag(u16),
    /// A four-character code, as used by AIFF-C `COMM` compression types and
    /// ISO base media sample entries.
    FourCc(FourCc),
    /// A textual codec name, for containers that name codecs as strings.
    ///
    /// Owned rather than borrowed because the whole point of reaching this
    /// variant is that the name came out of the file and is not one of a set
    /// the build knows.
    Name(String),
}

impl fmt::Display for CodecId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WaveFormatTag(tag) => write!(f, "WAVE format tag 0x{tag:04x}"),
            Self::FourCc(code) => write!(f, "four-CC '{code}'"),
            Self::Name(name) => write!(f, "codec '{name}'"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn printable_four_cc_displays_as_text() {
        assert_eq!(FourCc(*b"RIFF").to_string(), "RIFF");
        // A trailing space is part of many real tags and must survive.
        assert_eq!(FourCc(*b"fmt ").to_string(), "fmt ");
    }

    #[test]
    fn unprintable_four_cc_displays_as_hex() {
        assert_eq!(FourCc([0x00, 0x01, 0xff, 0x7f]).to_string(), "0x0001ff7f");
    }

    #[test]
    fn codec_id_names_the_specific_identity() {
        assert_eq!(
            CodecId::WaveFormatTag(7).to_string(),
            "WAVE format tag 0x0007"
        );
        assert_eq!(
            CodecId::FourCc(FourCc(*b"alaw")).to_string(),
            "four-CC 'alaw'"
        );
        assert_eq!(
            CodecId::Name("opus".to_string()).to_string(),
            "codec 'opus'"
        );
    }

    #[test]
    fn four_cc_exposes_the_bytes_it_was_built_from() {
        let tag = FourCc::from(*b"fLaC");
        assert_eq!(tag.as_bytes(), b"fLaC");
    }
}
