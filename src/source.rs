//! How encoded audio arrives when it turns up in pieces: bytes are pushed
//! in, samples are pulled out, and an incomplete frame waits inside for the
//! rest of itself.
//!
//! The trait is declared here and implemented by every streaming reader in
//! the crate. Whole-input decoding needs no trait: a caller holding all the
//! bytes hands them to a reader directly.

use crate::audio::AudioSpec;
use crate::error::DecodeError;

/// A stream that arrives in pieces: bytes are pushed in, samples are pulled
/// out, and an incomplete frame waits inside for the rest of itself.
///
/// This is a pull model even though every codec in this crate's first four
/// steps is stateless and would be served perfectly well by a chunk-in,
/// chunk-out function. MP3's bit reservoir lets a frame borrow bits from
/// earlier frames, which makes chunk-in-chunk-out wrong for MP3 before a line
/// of it is written. The shape costs almost nothing now and costs everything
/// above it later.
pub trait StreamSource {
    /// Hands `bytes` to the stream and returns how many were taken.
    ///
    /// A short return is back-pressure: pull samples out with
    /// [`pull`](StreamSource::pull) and offer the remainder again. Bytes that
    /// complete no frame are held, not rejected.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeError::UnsupportedContainer`] or
    /// [`DecodeError::UnsupportedCodec`] on the push that completes the
    /// header, and [`DecodeError::Malformed`] when the accepted bytes cannot
    /// continue the stream.
    ///
    /// **Not** [`DecodeError::ContainerCodecMismatch`]. Every container this
    /// crate reads names an encoding whose payload carries no second identity
    /// to disagree with, so no implementation here can reach that variant. It
    /// stays on the error type because a container carrying self-identifying
    /// frames, MP3 in RIFF being the obvious one, makes it reachable, and
    /// removing a variant from a `#[non_exhaustive]` enum is a breaking
    /// change for a consumer that names it.
    fn push(&mut self, bytes: &[u8]) -> Result<usize, DecodeError>;

    /// Appends up to `max_frames` frames of decoded samples to `output`, and
    /// returns how many frames it appended.
    ///
    /// A return of `0` means "nothing ready yet", never "end of stream": a
    /// stream has no end until the caller declares one with
    /// [`finish`](StreamSource::finish).
    ///
    /// # Errors
    ///
    /// Returns [`DecodeError::Malformed`] when a buffered frame turns out to be
    /// invalid once enough of it has arrived to judge.
    fn pull(&mut self, output: &mut Vec<f32>, max_frames: usize) -> Result<usize, DecodeError>;

    /// The rate and layout of what [`pull`](StreamSource::pull) produces, once
    /// enough of the header has arrived to know.
    ///
    /// `None` before then, and deliberately not a placeholder: a stream
    /// cannot report a rate it has not been told, and returning a plausible
    /// default instead is how a wrong rate gets accepted downstream without
    /// complaint.
    fn spec(&self) -> Option<AudioSpec>;

    /// How many bytes are held awaiting the rest of their frame.
    ///
    /// `0` means the stream is on a frame boundary and could be cut here
    /// without loss.
    fn buffered_bytes(&self) -> usize;

    /// Declares that no more bytes are coming, appends whatever remains and
    /// returns how many frames it appended.
    ///
    /// Idempotent. After it, [`push`](StreamSource::push) takes nothing and
    /// [`pull`](StreamSource::pull) produces nothing.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeError::Truncated`] when the stream ended with an
    /// incomplete frame held, and [`DecodeError::Truncated`] when it ended
    /// before the header was complete. Those bytes are only an error here:
    /// while the stream is open they are data that has not arrived yet.
    fn finish(&mut self, output: &mut Vec<f32>) -> Result<usize, DecodeError>;

    /// Returns the stream to its just-constructed state, dropping buffered
    /// bytes, undelivered samples and anything learned from the header.
    fn reset(&mut self);
}
