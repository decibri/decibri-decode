//! One entry point that picks the reader out of the bytes.
//!
//! # Why this exists
//!
//! Before it, every caller of this crate wrote the same magic-byte check. That
//! is one rule with as many implementations as there are users, and the moment
//! FLAC landed the crate's own README was wrong: its first example dispatched
//! between two containers and there were three. A rule that has to be restated
//! at every call site is a rule that will be restated wrongly somewhere, which
//! is the same reasoning that put the IFF pad byte in a single function.
//!
//! So identification lives here, once, and three entry points are built on it:
//! [`identify`] says what the bytes are, [`decode`] reads a whole input, and
//! [`AudioStreamDecoder`] reads one that arrives in pieces. The streaming one
//! matters as much as the whole-file one: without it a streaming caller still
//! hand-rolls the dispatch and the defect is only half fixed.
//!
//! # Twelve bytes, not four
//!
//! `RIFF` at offset zero does not mean WAV. It means a RIFF container, and AVI
//! is one too. The form type sits at offset eight and it is the field that
//! decides. `FORM` is the same story: an EA IFF 85 container whose form type
//! may be `AIFF`, `AIFC`, `8SVX`, `ILBM` or anything else that family carries.
//!
//! A four-byte probe would therefore hand an AVI file to [`WavReader`] and let
//! it fail somewhere deeper in, reporting a missing `fmt ` chunk for a file
//! that was never a WAV. This one reads twelve bytes and rejects a RIFF that
//! is not `WAVE`, or a `FORM` that is neither `AIFF` nor `AIFC`, with
//! [`DecodeError::UnsupportedContainer`] carrying the form type it actually
//! found.
//!
//! `fLaC` needs no form type, but identification still requires twelve bytes
//! before it will answer, so that "too short to identify" is one rule rather
//! than one per format. Nothing is lost by it: the shortest legal FLAC stream
//! is a signature, a metadata block header and a 34-byte streaminfo body.
//!
//! # What the probe does not cover, and will not
//!
//! Headerless linear PCM, headerless G.711 and bare FLAC frame streams have no
//! signature to probe. They stay explicit, with the caller supplying what it
//! knows, exactly as [`PcmDecoder`], [`G711Decoder`] and the bare frame reader
//! already require.
//!
//! Bare FLAC frames are the one that looks detectable and is not. A frame
//! begins with a 14-bit sync code and its header ends in a CRC-8, and the
//! measurement recorded on the recovery reader is that the pair together let
//! about one position in 32,768 of random data parse as a frame header: some
//! twenty accepting positions in ten million random runs. Sniffing for that in
//! bytes of unknown provenance would misfire on input that is not FLAC at all.
//! Headerless means told, not guessed, and there is no argument that makes a
//! guess safe here.
//!
//! # Nothing here looks at a name
//!
//! No entry point in this module takes a path, a file name or an extension,
//! and none of them is consulted anywhere in the crate. A file called
//! `input.wav` that holds AIFF bytes decodes as AIFF.

use std::fmt;

use crate::aiff::{AiffReader, AiffStreamDecoder};
use crate::audio::{AudioBuffer, AudioSpec};
use crate::codec::FourCc;
use crate::error::DecodeError;
use crate::flac::{FlacReader, FlacStreamDecoder};
use crate::source::StreamSource;
use crate::wav::{WavReader, WavStreamDecoder};
use crate::{aiff, riff};

/// The four bytes every FLAC stream starts with.
///
/// Stated here rather than in the FLAC module because this is where a FLAC
/// stream is recognised. The FLAC module takes its own signature from this
/// one, so the bytes are written down once.
pub(crate) const FLAC_MAGIC: FourCc = FourCc(*b"fLaC");

/// What the leading bytes of an input turned out to be.
///
/// One variant per whole-file reader, not one per file format: `AIFF` and
/// `AIFF-C` are both [`Aiff`](Self::Aiff) because [`AiffReader`] reads both,
/// and `RIFF` and `RF64` are both [`Wav`](Self::Wav) for the same reason. The
/// question this answers is which reader the bytes belong to.
///
/// This enum is `#[non_exhaustive]`: consumers matching on it must include a
/// `_ =>` catch-all arm, so the format after next is not source-breaking to
/// add.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Container {
    /// A RIFF or RF64 file whose form type is `WAVE`, read by [`WavReader`]
    /// and [`WavStreamDecoder`].
    Wav,
    /// An EA IFF 85 file whose form type is `AIFF` or `AIFC`, read by
    /// [`AiffReader`] and [`AiffStreamDecoder`].
    Aiff,
    /// A FLAC stream, opening with the `fLaC` signature, read by
    /// [`FlacReader`] and [`FlacStreamDecoder`].
    Flac,
}

impl Container {
    /// How many leading bytes identification reads.
    ///
    /// Twelve, because a four-CC at offset zero names a container *family* for
    /// two of the three formats here and the form type at offset eight is what
    /// separates a WAV from an AVI. Applied to `fLaC` as well, which needs only
    /// four, so that an input too short to identify is one rule rather than
    /// three.
    pub const PROBE_BYTES: usize = 12;
}

impl fmt::Display for Container {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Wav => "WAV",
            Self::Aiff => "AIFF",
            Self::Flac => "FLAC",
        };
        f.write_str(name)
    }
}

/// Says what `bytes` are, without decoding any of them.
///
/// Reads [`Container::PROBE_BYTES`] bytes and nothing else. Useful on its own,
/// and it is what [`decode`] and [`AudioStreamDecoder`] are built on, so all
/// three agree by construction rather than by review.
///
/// # Errors
///
/// - [`DecodeError::Truncated`] when the input is shorter than
///   [`Container::PROBE_BYTES`]. A four-byte file that happens to start with
///   `RIFF` is not a WAV, and guessing from a prefix is how a truncated
///   download becomes a confusing parse failure much later.
/// - [`DecodeError::UnsupportedContainer`] carrying the **form type** for a
///   RIFF or RF64 file that is not `WAVE` and a `FORM` file that is neither
///   `AIFF` nor `AIFC`, and carrying the **magic** for leading bytes that name
///   no container this crate reads.
///
/// # Example
///
/// ```
/// use decibri_decode::{identify, Container, DecodeError};
///
/// // A RIFF container that is not a WAV. Rejected by what is at offset
/// // eight, naming what was found there.
/// let mut avi = Vec::new();
/// avi.extend_from_slice(b"RIFF");
/// avi.extend_from_slice(&2048u32.to_le_bytes());
/// avi.extend_from_slice(b"AVI ");
/// assert!(matches!(
///     identify(&avi),
///     Err(DecodeError::UnsupportedContainer { tag }) if tag.as_bytes() == b"AVI "
/// ));
///
/// // A FLAC stream needs no form type, but still needs twelve bytes.
/// let mut flac = b"fLaC".to_vec();
/// flac.extend_from_slice(&[0x80, 0x00, 0x00, 0x22, 0x10, 0x00, 0x10, 0x00]);
/// assert_eq!(identify(&flac)?, Container::Flac);
/// assert!(matches!(
///     identify(&flac[..11]),
///     Err(DecodeError::Truncated { expected: 12, available: 11 })
/// ));
/// # Ok::<(), DecodeError>(())
/// ```
pub fn identify(bytes: &[u8]) -> Result<Container, DecodeError> {
    // Both reads are required, so any input under twelve bytes lands here
    // whatever its first four bytes say.
    let (Some(magic), Some(form)) = (riff::four_cc_at(bytes, 0), riff::four_cc_at(bytes, 8)) else {
        return Err(DecodeError::Truncated {
            expected: Container::PROBE_BYTES as u64,
            available: bytes.len() as u64,
        });
    };

    if magic == FLAC_MAGIC {
        return Ok(Container::Flac);
    }
    if magic == riff::RIFF || magic == riff::RF64 {
        return if form == riff::WAVE {
            Ok(Container::Wav)
        } else {
            Err(DecodeError::UnsupportedContainer { tag: form })
        };
    }
    if magic == aiff::FORM {
        return if form == aiff::AIFF || form == aiff::AIFC {
            Ok(Container::Aiff)
        } else {
            Err(DecodeError::UnsupportedContainer { tag: form })
        };
    }
    Err(DecodeError::UnsupportedContainer { tag: magic })
}

/// Decodes a whole input, whatever of the carried formats it is.
///
/// The reader comes from the content by way of [`identify`]. The result is
/// identical to constructing that reader by hand and calling its own
/// `decode_to_end`, which is asserted rather than assumed.
///
/// The channel count of the returned buffer is never zero, on every carried
/// container. [`AudioSpec`](crate::AudioSpec) and
/// [`AudioBuffer::from_samples`](crate::AudioBuffer::from_samples) accept zero
/// from a caller who states it, and this is a guarantee about what a decode
/// produces rather than about what the types can hold.
///
/// # Errors
///
/// Everything [`identify`] returns, plus everything the chosen reader returns,
/// unaltered.
///
/// # Example
///
/// ```
/// use decibri_decode::{decode, AudioSpec, WavCodec, WavWriter};
///
/// let file = WavWriter::new(AudioSpec::mono(16_000), WavCodec::PcmI16)
///     .to_bytes(&[0.0, 0.25, -0.25])?;
///
/// // Nothing here names WAV. The bytes do.
/// let audio = decode(&file)?;
/// assert_eq!(audio.sample_rate(), 16_000);
/// assert_eq!(audio.samples(), [0.0, 0.25, -0.25]);
/// # Ok::<(), decibri_decode::DecodeError>(())
/// ```
pub fn decode(bytes: &[u8]) -> Result<AudioBuffer, DecodeError> {
    match identify(bytes)? {
        Container::Wav => Ok(WavReader::new(bytes)?.decode_to_end()),
        Container::Aiff => Ok(AiffReader::new(bytes)?.decode_to_end()),
        Container::Flac => FlacReader::new(bytes)?.decode_to_end(),
    }
}

/// The streaming reader the probe chose, held by value so its buffered-byte
/// count stays reachable.
///
/// The FLAC arm is boxed and the other two are not. `FlacStreamDecoder` holds
/// the predictor and residual working buffers a frame is reconstructed in and
/// is some twenty times the size of the other two, so an unboxed enum would
/// make every WAV stream carry FLAC's footprint. One allocation per FLAC
/// stream is the cheaper side of that trade by a wide margin.
#[derive(Debug)]
enum Inner {
    Wav(WavStreamDecoder),
    Aiff(AiffStreamDecoder),
    Flac(Box<FlacStreamDecoder>),
}

impl Inner {
    /// The reader for `container`.
    fn new(container: Container) -> Self {
        match container {
            Container::Wav => Self::Wav(WavStreamDecoder::new()),
            Container::Aiff => Self::Aiff(AiffStreamDecoder::new()),
            // `Box::default` rather than `Box::new(FlacStreamDecoder::new())`:
            // the reader's `Default` is its `new`, and clippy is right that the
            // longer form allocates a default and then overwrites it.
            Container::Flac => Self::Flac(Box::default()),
        }
    }

    /// The chosen reader as the trait every arm implements.
    ///
    /// [`StreamSource`] is object safe, so delegation is one line per method
    /// here rather than one line per method per format.
    fn source(&mut self) -> &mut dyn StreamSource {
        match self {
            Self::Wav(reader) => reader,
            Self::Aiff(reader) => reader,
            Self::Flac(reader) => reader.as_mut(),
        }
    }

    /// The chosen reader, for the two methods that only read.
    fn source_ref(&self) -> &dyn StreamSource {
        match self {
            Self::Wav(reader) => reader,
            Self::Aiff(reader) => reader,
            Self::Flac(reader) => reader.as_ref(),
        }
    }
}

/// Reads a stream that arrives in pieces, in whatever of the carried formats
/// its leading bytes turn out to name.
///
/// # How it buffers
///
/// The first [`Container::PROBE_BYTES`] bytes are held rather than delegated,
/// because there is no reader to delegate them to until they have all arrived.
/// On the push that completes them the format is identified, the matching
/// streaming reader is constructed, and the held bytes go into it **in front
/// of** everything that follows, so the inner reader sees the stream from its
/// first byte and not from byte thirteen. Whatever is left of that same push
/// goes straight on to the inner reader, so a caller that hands over the whole
/// file at once pays nothing for the indirection.
///
/// The buffer is twelve bytes and never grows: it is look-ahead, not a
/// staging area, and every byte past identification is the inner reader's own
/// business.
///
/// [`spec`](StreamSource::spec) is `None` until the inner reader knows it,
/// which is strictly later than identification: knowing a stream is a WAV is
/// not knowing its rate.
///
/// # Example
///
/// One byte at a time, which is the case that proves the buffering is real:
/// identification spans twelve separate pushes and the audio still arrives.
///
/// ```
/// use decibri_decode::{
///     AudioSpec, AudioStreamDecoder, Container, StreamSource, WavCodec, WavWriter,
/// };
///
/// let file = WavWriter::new(AudioSpec::mono(8_000), WavCodec::PcmI16)
///     .to_bytes(&[0.0, 0.5, -0.5])?;
///
/// let mut stream = AudioStreamDecoder::new();
/// let mut samples = Vec::new();
/// for byte in &file {
///     let mut offset = 0;
///     while offset < 1 {
///         offset += stream.push(&[*byte][offset..])?;
///         while stream.pull(&mut samples, usize::MAX)? > 0 {}
///     }
/// }
/// stream.finish(&mut samples)?;
///
/// assert_eq!(stream.container(), Some(Container::Wav));
/// assert_eq!(samples, [0.0, 0.5, -0.5]);
/// # Ok::<(), decibri_decode::DecodeError>(())
/// ```
#[derive(Debug)]
pub struct AudioStreamDecoder {
    /// The reader chosen from the leading bytes, once there have been enough
    /// of them.
    inner: Option<Inner>,
    /// The leading bytes, held until there are [`Container::PROBE_BYTES`] of
    /// them.
    header: Vec<u8>,
    /// How much of `header` the inner reader has taken. Everything it has not
    /// taken is offered again before any later byte is.
    fed: usize,
    /// What the probe decided, kept because a caller usually wants to know and
    /// cannot recover it from the samples.
    container: Option<Container>,
    finished: bool,
}

impl Default for AudioStreamDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioStreamDecoder {
    /// A reader waiting for the first byte of a stream.
    pub fn new() -> Self {
        Self {
            inner: None,
            header: Vec::with_capacity(Container::PROBE_BYTES),
            fed: 0,
            container: None,
            finished: false,
        }
    }

    /// What the leading bytes turned out to be, once enough of them have
    /// arrived.
    ///
    /// `None` before [`Container::PROBE_BYTES`] bytes have been pushed.
    pub const fn container(&self) -> Option<Container> {
        self.container
    }

    /// Offers the held leading bytes to the inner reader, front first.
    ///
    /// A no-op before identification, and after it a loop rather than a single
    /// push because [`StreamSource::push`] may take fewer bytes than it is
    /// offered. In practice it never does here: the inner reader was
    /// constructed a moment ago and has no decoded samples to apply
    /// back-pressure with. The loop is written for the contract, not for the
    /// case.
    fn feed_header(&mut self) -> Result<(), DecodeError> {
        while self.fed < self.header.len() {
            let Some(inner) = self.inner.as_mut() else {
                return Ok(());
            };
            let taken = inner.source().push(&self.header[self.fed..])?;
            if taken == 0 {
                break;
            }
            self.fed += taken;
        }
        Ok(())
    }

    /// The body of [`push`](StreamSource::push), split out so a failure sets
    /// the finished flag in one place.
    fn push_inner(&mut self, bytes: &[u8]) -> Result<usize, DecodeError> {
        if self.inner.is_none() {
            let want = Container::PROBE_BYTES - self.header.len();
            let take = want.min(bytes.len());
            self.header.extend_from_slice(&bytes[..take]);
            if self.header.len() < Container::PROBE_BYTES {
                return Ok(take);
            }
            let container = identify(&self.header)?;
            self.container = Some(container);
            self.inner = Some(Inner::new(container));
            self.feed_header()?;
            if self.fed < self.header.len() {
                return Ok(take);
            }
            // Whatever is left of this push is ordinary stream data now.
            let Some(inner) = self.inner.as_mut() else {
                return Ok(take);
            };
            return Ok(take + inner.source().push(&bytes[take..])?);
        }

        self.feed_header()?;
        if self.fed < self.header.len() {
            return Ok(0);
        }
        let Some(inner) = self.inner.as_mut() else {
            return Ok(0);
        };
        inner.source().push(bytes)
    }
}

impl StreamSource for AudioStreamDecoder {
    fn push(&mut self, bytes: &[u8]) -> Result<usize, DecodeError> {
        if self.finished {
            return Ok(0);
        }
        let result = self.push_inner(bytes);
        if result.is_err() {
            // A stream that has failed structurally is over, for the reason
            // recorded on the WAV reader: a caller who keeps pushing must not
            // get a second, different answer from the same stream.
            self.finished = true;
        }
        result
    }

    fn pull(&mut self, output: &mut Vec<f32>, max_frames: usize) -> Result<usize, DecodeError> {
        match self.inner.as_mut() {
            Some(inner) => inner.source().pull(output, max_frames),
            // Nothing has been identified yet, so nothing can be ready.
            None => Ok(0),
        }
    }

    fn spec(&self) -> Option<AudioSpec> {
        self.inner
            .as_ref()
            .and_then(|inner| inner.source_ref().spec())
    }

    fn buffered_bytes(&self) -> usize {
        (self.header.len() - self.fed)
            + self
                .inner
                .as_ref()
                .map_or(0, |inner| inner.source_ref().buffered_bytes())
    }

    fn finish(&mut self, output: &mut Vec<f32>) -> Result<usize, DecodeError> {
        if self.finished {
            return Ok(0);
        }
        self.finished = true;
        // Anything still held belongs to the inner reader, which has to see it
        // before it is asked whether the stream ended cleanly.
        self.feed_header()?;
        let Some(inner) = self.inner.as_mut() else {
            // The stream ended before it could be identified. That is the same
            // shortfall a whole-file caller gets from `identify`, reported the
            // same way.
            return Err(DecodeError::Truncated {
                expected: Container::PROBE_BYTES as u64,
                available: self.header.len() as u64,
            });
        };
        inner.source().finish(output)
    }

    fn reset(&mut self) {
        *self = Self::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wav::{WavCodec, WavWriter};

    /// A twelve-byte header with `magic` at zero and `form` at eight.
    fn header(magic: &[u8; 4], form: &[u8; 4]) -> Vec<u8> {
        let mut bytes = magic.to_vec();
        bytes.extend_from_slice(&64u32.to_le_bytes());
        bytes.extend_from_slice(form);
        bytes
    }

    #[test]
    fn the_three_carried_formats_are_identified_by_content() {
        assert_eq!(identify(&header(b"RIFF", b"WAVE")).unwrap(), Container::Wav);
        assert_eq!(identify(&header(b"RF64", b"WAVE")).unwrap(), Container::Wav);
        assert_eq!(
            identify(&header(b"FORM", b"AIFF")).unwrap(),
            Container::Aiff
        );
        assert_eq!(
            identify(&header(b"FORM", b"AIFC")).unwrap(),
            Container::Aiff
        );
        assert_eq!(
            identify(&header(b"fLaC", b"\0\0\0\0")).unwrap(),
            Container::Flac
        );
    }

    #[test]
    fn a_riff_that_is_not_wave_is_rejected_by_its_form_type() {
        // The trap a four-byte probe walks into. Every one of these is a real
        // RIFF or IFF form type that is not audio this crate reads.
        for form in [b"AVI ", b"ACON", b"RMID", b"CDXA"] {
            let error = identify(&header(b"RIFF", form)).unwrap_err();
            let DecodeError::UnsupportedContainer { tag } = error else {
                panic!("a RIFF of form {form:?} was not rejected as a container: {error}");
            };
            assert_eq!(tag.as_bytes(), form, "the error names the form type found");
        }
    }

    #[test]
    fn a_form_that_is_not_aiff_or_aifc_is_rejected_by_its_form_type() {
        for form in [b"8SVX", b"ILBM", b"ANBM", b"AIFZ"] {
            let error = identify(&header(b"FORM", form)).unwrap_err();
            let DecodeError::UnsupportedContainer { tag } = error else {
                panic!("a FORM of type {form:?} was not rejected as a container: {error}");
            };
            assert_eq!(tag.as_bytes(), form);
        }
    }

    #[test]
    fn an_unknown_magic_is_named_by_its_leading_bytes() {
        let error = identify(&header(b"OggS", b"\0\0\0\0")).unwrap_err();
        let DecodeError::UnsupportedContainer { tag } = error else {
            panic!("an Ogg page was not rejected as a container: {error}");
        };
        assert_eq!(tag.as_bytes(), b"OggS");
    }

    #[test]
    fn every_length_under_the_probe_length_is_truncated_not_a_guess() {
        let full = header(b"RIFF", b"WAVE");
        for length in 0..Container::PROBE_BYTES {
            let error = identify(&full[..length]).unwrap_err();
            assert!(
                matches!(
                    error,
                    DecodeError::Truncated { expected, available }
                        if expected == Container::PROBE_BYTES as u64
                            && available == length as u64
                ),
                "{length} bytes gave {error} rather than a truncation"
            );
        }
    }

    #[test]
    fn a_prefix_that_would_identify_still_needs_the_whole_probe() {
        // Four bytes of `fLaC` need no form type to be conclusive, and are
        // still refused: "too short to identify" is one rule, not three.
        assert!(matches!(
            identify(b"fLaC"),
            Err(DecodeError::Truncated { .. })
        ));
    }

    #[test]
    fn the_container_names_itself_for_a_report() {
        assert_eq!(Container::Wav.to_string(), "WAV");
        assert_eq!(Container::Aiff.to_string(), "AIFF");
        assert_eq!(Container::Flac.to_string(), "FLAC");
    }

    #[test]
    fn the_whole_file_path_matches_the_reader_it_chose() {
        let samples = [0.0, 0.25, -0.25, 0.5];
        let file = WavWriter::new(AudioSpec::new(16_000, 2), WavCodec::PcmI16)
            .to_bytes(&samples)
            .expect("a two-channel 16-bit WAV writes");
        let direct = WavReader::new(&file)
            .expect("the file opens")
            .decode_to_end();
        let probed = decode(&file).expect("the file opens through the probe");
        assert_eq!(probed, direct);
    }

    #[test]
    fn the_streaming_path_reports_what_it_identified_and_nothing_before() {
        let file = WavWriter::new(AudioSpec::mono(8_000), WavCodec::PcmU8)
            .to_bytes(&[0.0, 0.5])
            .expect("an 8-bit WAV writes");

        let mut stream = AudioStreamDecoder::new();
        assert_eq!(stream.container(), None);
        assert_eq!(stream.spec(), None);

        // Eleven bytes is one short of identification.
        assert_eq!(stream.push(&file[..11]).expect("eleven bytes"), 11);
        assert_eq!(stream.container(), None);
        assert_eq!(stream.buffered_bytes(), 11);

        assert_eq!(stream.push(&file[11..12]).expect("the twelfth"), 1);
        assert_eq!(stream.container(), Some(Container::Wav));
        // Identified, but the rate is the inner reader's to know and it has
        // not read `fmt ` yet.
        assert_eq!(stream.spec(), None);
        assert_eq!(stream.buffered_bytes(), 0);
    }

    #[test]
    fn a_stream_that_ends_before_identification_is_truncated() {
        let mut stream = AudioStreamDecoder::new();
        assert_eq!(stream.push(b"RIFF\x40\x00").expect("six bytes"), 6);
        let mut samples = Vec::new();
        let error = stream.finish(&mut samples).unwrap_err();
        assert!(
            matches!(
                error,
                DecodeError::Truncated {
                    expected: 12,
                    available: 6
                }
            ),
            "a stream cut inside the probe reported {error}"
        );
        // Idempotent, exactly as the trait requires.
        assert_eq!(stream.finish(&mut samples).expect("second finish"), 0);
    }

    #[test]
    fn a_failed_stream_stays_failed() {
        let mut stream = AudioStreamDecoder::new();
        let mut avi = b"RIFF".to_vec();
        avi.extend_from_slice(&2048u32.to_le_bytes());
        avi.extend_from_slice(b"AVI ");
        assert!(stream.push(&avi).is_err());
        // A caller who keeps pushing must not get a second answer.
        assert_eq!(
            stream.push(&avi).expect("a finished stream takes nothing"),
            0
        );
    }

    #[test]
    fn reset_returns_the_stream_to_its_just_constructed_state() {
        let file = WavWriter::new(AudioSpec::mono(8_000), WavCodec::PcmI16)
            .to_bytes(&[0.0, 0.5])
            .expect("a 16-bit WAV writes");
        let mut stream = AudioStreamDecoder::new();
        stream.push(&file).expect("the whole file at once");
        assert_eq!(stream.container(), Some(Container::Wav));

        stream.reset();
        assert_eq!(stream.container(), None);
        assert_eq!(stream.spec(), None);
        assert_eq!(stream.buffered_bytes(), 0);
    }
}
