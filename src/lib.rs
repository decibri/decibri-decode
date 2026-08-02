#![forbid(unsafe_code)]
#![warn(missing_docs)]
//! Arbitrary encoded bytes in, mono `f32` at a declared sample rate out, with a
//! stated and tested sample-count guarantee.
//!
//! That sentence is the crate. Codecs fill the contract in; they are not the
//! contract. The trait boundaries, the error type and the buffer types are
//! settled before any codec exists, because changing them afterwards is a
//! breaking change and changing them now is free.
//!
//! # The contract
//!
//! - **In:** encoded bytes, held whole in memory and handed to a reader, or
//!   arriving in pieces via [`StreamSource`].
//! - **Out:** `f32` samples that always travel with the [`AudioSpec`] that
//!   describes them, so no downstream stage ever infers or assumes a rate. That
//!   is the whole reason [`AudioBuffer`] exists rather than a bare `Vec<f32>`:
//!   decibri has been bitten repeatedly by rate and layout travelling
//!   separately from the samples they describe, most expensively on the AEC
//!   reference path, where a wrong-rate reference is accepted silently and
//!   cancels nothing.
//! - **Guarantee:** the number of samples produced for a given input is stated
//!   and tested, not incidental. Decoding a source of `n` frames without rate
//!   conversion yields exactly `n * channels` samples. Decoding with rate
//!   conversion yields exactly the count `decibri-resampler` states for that
//!   rate pair and input length. A codec that cannot state its output count
//!   does not go behind [`Decoder`].
//! - **Failure:** every rejection is a [`DecodeError`] that names the specific
//!   thing it rejected. "Unsupported" on its own tells a caller nothing they
//!   can act on.
//!
//! # Choosing a reader
//!
//! [`decode`] takes bytes in any carried container and gives back an
//! [`AudioBuffer`], [`AudioStreamDecoder`] does the same for a stream that
//! arrives in pieces, and [`identify`] answers what an input is without
//! decoding it. All three read the same twelve leading bytes, so a caller
//! never writes a magic-byte check and there is one statement of what a WAV
//! looks like rather than one per call site.
//!
//! Twelve, not four, because `RIFF` names a container *family*: the form type
//! at offset eight is what separates a WAV from an AVI, and `FORM` from an
//! `8SVX`. A container whose form type is not one this crate reads is
//! [`DecodeError::UnsupportedContainer`] naming the form type found, never a
//! reader handed a file that was never its.
//!
//! Headerless linear PCM, headerless G.711 and bare FLAC frame streams have no
//! signature and are outside this. They stay explicit, because the alternative
//! is guessing: a FLAC frame's 14-bit sync code and 8-bit header CRC together
//! let about one position in 32,768 of random data parse as a frame header, so
//! sniffing for one in bytes of unknown provenance would misfire on input that
//! is not FLAC at all.
//!
//! # Determinism
//!
//! For every format this crate carries (PCM, G.711, WAV, AIFF and FLAC),
//! decoding is **bit-exact and cross-platform byte-identical**. The same input
//! produces the same bytes on every target: no floating-point reassociation, no
//! platform-specific fast paths, no reliance on FMA contraction.
//!
//! If MP3 is ever added, its decode is compliant within the ISO/IEC 11172-4
//! accuracy bound and is explicitly **not** claimed as byte-identical. MPEG
//! never specified bit-exact decoding, and two conforming decoders legitimately
//! differ. That exception will be stated on the MP3 decoder itself and does not
//! weaken the guarantee above for any other format.
//!
//! This is written now, while it is a true statement about an empty crate,
//! rather than retrofitted once somebody has assumed otherwise.
//!
//! **Determinism is not losslessness.** Determinism holds for everything here.
//! Losslessness, a value surviving the round trip through `f32` with its
//! bits intact, holds for every format at or below 24 significant bits, all of
//! them except `i32` and `f64`. An `f32` significand is 24 bits, so 31-bit and
//! 53-bit values land on the nearest representable `f32` instead of on
//! themselves. That is inherent to an `f32` internal representation and is the
//! right trade for a crate feeding decibri's `f32` chain, but it is stated here
//! rather than left to be discovered. The error is bounded by half an ulp at the
//! value's magnitude; nothing wraps and no sign flips.
//!
//! # Rate conversion
//!
//! There is none here, and there never will be. Rate conversion is
//! [`decibri_resampler`], the crate's only dependency, called and never
//! reimplemented. Its failures arrive as [`DecodeError::Resample`]. Additional
//! *instances* of a resampler are expected and fine; a second *implementation*
//! is a correctness hazard, because resampler choice was measured during the
//! echo-cancellation work as moving AECMOS by up to 1.03, larger than the
//! effect being measured.
//!
//! # Sample conversion
//!
//! [`SampleFormat`] carries every linear PCM format the crate reads: `u8`
//! and `i8` (WAV's 8-bit is unsigned and AIFF's is signed, and those are the
//! two specifications' rules rather than a choice here), and 16-, 24- and
//! 32-bit integer and 32- and 64-bit float in both byte orders. It converts
//! each to and from `f32`. The numerical conventions are not
//! this crate's to choose: the scale factor, the rounding and clamping rule and
//! the downmix formula were all read out of decibri so that the same input
//! decoded through either crate gives the same samples. They are recorded, with
//! the file and line each came from, on the [`sample`] module.
//!
//! There is no dither, no feature flag for one and no plan for one. It needs a
//! random source, which would either break the byte-identical claim above or
//! make a generator seed part of the public contract.
//!
//! # G.711
//!
//! [`G711Law`] carries both ITU-T companding laws, mu-law and A-law, in both
//! directions, and [`G711Decoder`] decodes a headerless stream in either. The
//! tables are derived from the recommendation's own segment geometry rather than
//! transcribed from a third-party implementation, because the crate is published
//! under Apache-2.0 and reference data with an unestablished licence is a real
//! constraint.
//!
//! G.711 is a *sample format, not a rate*. It is overwhelmingly carried at
//! 8 kHz in telephony and nothing in the recommendation says so, so the rate
//! comes from the [`AudioSpec`] the caller states and nothing here defaults to
//! 8000.
//!
//! Decoding is a code to `i16` through the table, then `i16` to `f32` through
//! [`sample`]; encoding is the reverse. There is no direct code-to-`f32` path,
//! so the crate has one scaling rule rather than two.
//!
//! # WAV
//!
//! [`WavReader`] reads a RIFF/WAVE or RF64 file held whole in memory and
//! [`WavStreamDecoder`] reads one that arrives in pieces; [`WavWriter`] writes
//! every encoding either of them reads. [`WavCodec`] is the set of encodings a
//! WAV file can carry and this crate can decode: four widths of linear PCM, two
//! of IEEE float, and both G.711 laws.
//!
//! Dispatch is on the **`(wFormatTag, wBitsPerSample)` pair**, never on the tag
//! alone. Tag 1 covers four widths, and a reader that picked one would decode
//! the others at the wrong stride and produce noise rather than an error. It is
//! never, at any point, on a file name. `WAVE_FORMAT_EXTENSIBLE` resolves
//! through its SubFormat GUID.
//!
//! Malformed input is a typed [`DecodeError`] rather than a panic, and no
//! allocation anywhere is proportional to a size the file declared: a `data`
//! chunk announcing four gigabytes inside a two-kilobyte file is rejected
//! before a byte is reserved for it. All chunk arithmetic is 64-bit and
//! checked, which is what makes that true on a 32-bit target as well.
//!
//! # AIFF and AIFF-C
//!
//! [`AiffReader`] reads an AIFF or AIFF-C file held whole in memory,
//! [`AiffStreamDecoder`] reads one that arrives in pieces, and [`AiffWriter`]
//! writes every encoding either of them reads: plain `AIFF` where big-endian
//! PCM makes that form legal, `AIFC` with its `FVER` chunk and compression
//! field where the encoding requires it. [`AiffCodec`] is the set of encodings
//! this crate carries in the container: linear PCM at four widths (big-endian
//! under `NONE`/`twos`, little-endian under `sowt`, and **signed** at 8 bits
//! where WAV is unsigned), IEEE float under `fl32`/`fl64`, and both G.711 laws
//! under `alaw`/`ulaw`. Every other compression type is
//! [`DecodeError::UnsupportedCodec`].
//!
//! Dispatch is on the **form type and the compression four-CC** (paired with
//! `sampleSize` for the linear widths), never on a file name. The 80-bit
//! extended-precision sample rate is parsed by hand, and `COMM`'s
//! `numSampleFrames` is required to agree exactly with the `SSND` chunk's
//! length. Disagreement is a typed error, never a silent preference for
//! either source.
//!
//! # FLAC
//!
//! `FlacReader` reads a FLAC stream held whole in memory and
//! `FlacStreamDecoder` reads one that arrives in pieces; `FlacStreamInfo` is
//! what the streaminfo metadata block declared. Both readers are always
//! built: this crate has no Cargo features, so a FLAC stream decodes in every
//! build of it.
//!
//! Decoding is written from RFC 9639 and covers the whole format this crate
//! claims: every bit depth from 4 to 32, every channel count from 1 to 8, the
//! three stereo decorrelations, constant, verbatim, fixed and linear
//! predictor subframes, both Rice parameter widths and the escaped
//! partitions, fixed and variable block size streams, and the uncommon block
//! sizes and sample rates that live outside the streamable subset.
//!
//! **The integrity check is a runtime feature, not a test.** Every FLAC file
//! carries an MD5 of its own unencoded audio, so unlike every other format
//! here the decoder can check its own output on *every file it ever sees*. It
//! does: a whole-file decode hashes as it goes, a streaming decode hashes
//! incrementally, and both verify at the end. A mismatch is a typed error
//! rather than audio. An all-zero checksum means "not known", which the
//! format permits and which is not a failure. The frame header CRC-8 and the
//! whole-frame CRC-16 are verified as well, before any sample from that frame
//! is delivered.
//!
//! **Properties may not change mid-stream.** A FLAC frame header restates the
//! sample rate, channel count and bit depth, and the format allows them to
//! differ between frames. This crate rejects that with a typed error, because
//! [`AudioBuffer`] carries one [`AudioSpec`] for the samples it holds and
//! there is no honest way to describe a buffer whose rate changed half way
//! through.
//!
//! [`FlacWriter`] writes native FLAC streams at compression levels 0
//! through 8, computing residuals through the same integer prediction
//! functions the decoder reconstructs with, so the two cannot disagree about
//! the arithmetic. The Ogg and Matroska mappings are not here.
//!
//! ## Bare frames, out-of-band streaminfo, and recovery
//!
//! A FLAC frame is self-describing, so a run of frames with no `fLaC`
//! signature and no metadata in front of it is still decodable.
//! `FlacFrameReader` reads one held whole in memory and
//! `FlacStreamDecoder::frames` reads one that arrives in pieces, the third
//! headerless decoder in the crate, beside [`PcmDecoder`] and
//! [`G711Decoder`]. The two frame header codes that mean "take this field
//! from streaminfo", sample rate `0b0000` and bit depth `0b000`, have nothing
//! to be taken from on that path and are a typed error naming which field was
//! unresolved, never a default.
//!
//! Both readers accept a streaminfo block from **out of band**, which is what
//! Ogg and Matroska do: the block travels in the container's codec-private
//! header and the packets carry bare frames. The block's 34-byte body parses
//! through `FlacStreamInfo::from_block`, and supplying the result resolves
//! those two codes and restores the MD5 check to a path that otherwise has
//! none.
//! Whether that check actually ran is reported, as `Md5Check`, because a
//! caller cannot tell from the samples and a decode that quietly verified
//! nothing is how a guarantee gets claimed that was never provided.
//!
//! `FlacRecovery` finds frame boundaries in a byte run that may begin
//! mid-frame, hold garbage or be damaged. It is a separate type rather than a
//! flag, because everywhere else here a damaged frame is an error rather than
//! quietly shortened audio: recovery has to be asked for, and every decode
//! through it reports the byte ranges that produced nothing and the sample
//! positions lost. A sync point is accepted only when the frame at it passes
//! its CRC-16 *and* another frame header parses where that frame ends. The
//! 14-bit sync code and 8-bit header CRC together let about one position in
//! 32,768 of random data look like a frame, so a single validating header is
//! not evidence.
//!
//! **Recovery is whole-buffer only.** `FlacRecovery` takes a slice, and there
//! is no streaming form of it: a live stream that meets corruption part-way
//! through is a typed error from `FlacStreamDecoder`, not a resync. Accepting
//! a sync point requires the bytes *after* the candidate frame, so a
//! streaming resync would have to buffer forward without a bound or accept
//! weaker evidence than the standard above. Recovering a damaged live stream
//! means collecting its bytes and handing them to `FlacRecovery`.
//!
//! # Layout
//!
//! - [`decode`], [`AudioStreamDecoder`] and [`identify`] are the front door:
//!   the reader comes from the content, not from the caller.
//! - [`Decoder`] is the boundary: frames in, samples out. It is object safe, so
//!   platform decoders (AudioToolbox, MediaCodec) can sit behind it alongside
//!   the crate's own.
//! - [`PcmDecoder`] is the implementation of it for headerless PCM, and
//!   [`G711Decoder`] for headerless G.711.
//! - [`StreamSource`] is how bytes arrive when they arrive in pieces.
//! - [`WavReader`], [`WavStreamDecoder`] and [`WavWriter`] are the RIFF/WAVE
//!   and RF64 container.
//! - [`AiffReader`], [`AiffStreamDecoder`] and [`AiffWriter`] are the AIFF
//!   and AIFF-C container.
//! - `FlacReader`, `FlacStreamDecoder` and [`FlacWriter`] are FLAC;
//!   `FlacFrameReader` is a bare frame stream and `FlacRecovery` is a
//!   damaged one.
//! - [`AudioSpec`] and [`AudioBuffer`] keep rate and layout attached to
//!   samples.
//! - [`SampleFormat`] and the functions beside it convert samples;
//!   [`downmix_to_mono`], [`interleave`] and [`deinterleave`] rearrange them.
//! - [`G711Law`] is the mu-law and A-law table, in both directions.
//! - [`CodecId`] and [`FourCc`] let [`DecodeError`] name what it saw, and
//!   [`Container`] is what [`identify`] answers with.
//!
//! # Status
//!
//! Linear PCM, G.711, the RIFF/WAVE and RF64 container, the AIFF and AIFF-C
//! container, and FLAC. [`PcmDecoder`] decodes headerless streams in any
//! [`SampleFormat`] and [`G711Decoder`] decodes headerless mu-law and A-law;
//! [`WavReader`] and [`WavStreamDecoder`] read WAV whole or in pieces and
//! [`WavWriter`] writes it; [`AiffReader`], [`AiffStreamDecoder`] and
//! [`AiffWriter`] do the same three things for AIFF; `FlacReader` and
//! `FlacStreamDecoder` read FLAC whole or in pieces, [`FlacWriter`] writes
//! it at compression levels 0 through 8, `FlacFrameReader` reads a bare
//! frame stream and `FlacRecovery` recovers what a damaged byte run holds.
//! The Ogg and Matroska mappings of FLAC are not here; the codec they would
//! build on is. [`decode`] and [`AudioStreamDecoder`] pick between the three
//! containers by content. [`WavStreamDecoder`], [`AiffStreamDecoder`],
//! `FlacStreamDecoder` and [`AudioStreamDecoder`] are the [`StreamSource`]s.

mod aiff;
mod audio;
mod codec;
mod decoder;
mod error;
mod flac;
mod g711;
mod md5;
mod payload;
mod pcm;
mod probe;
mod riff;
pub mod sample;
mod source;
mod wav;

pub use aiff::{AiffCodec, AiffForm, AiffFormat, AiffReader, AiffStreamDecoder, AiffWriter};
pub use audio::{AudioBuffer, AudioSpec};
pub use codec::{CodecId, FourCc};
pub use decoder::Decoder;
pub use error::DecodeError;
pub use flac::{
    FlacFrameReader, FlacFrameReport, FlacReader, FlacRecovery, FlacRecoveryReport, FlacSkip,
    FlacSkipReason, FlacStreamDecoder, FlacStreamInfo, FlacWriter, Md5Check,
};
pub use g711::{G711Decoder, G711Law};
pub use pcm::PcmDecoder;
pub use probe::{decode, identify, AudioStreamDecoder, Container};
pub use sample::{
    deinterleave, downmix_to_mono, f32_to_i16, f32_to_i24, f32_to_i32, f32_to_i8, f32_to_u8,
    i16_to_f32, i24_to_f32, i32_to_f32, i8_to_f32, interleave, u8_to_f32, SampleFormat, I24_MAX,
    I24_MIN, MAX_BYTES_PER_SAMPLE,
};
pub use source::StreamSource;
pub use wav::{
    RiffFlavour, WavCodec, WavFormat, WavHeaderStyle, WavReader, WavStreamDecoder, WavWriter,
};

/// The resampler this crate defers all rate conversion to.
///
/// Re-exported so a consumer can name [`ResamplerError`](decibri_resampler::ResamplerError)
/// when matching [`DecodeError::Resample`] without taking its own dependency,
/// and, more to the point, without taking a *different version* of it.
pub use decibri_resampler;

/// The README's Rust blocks, compiled and run as doctests so the examples on
/// the crate's front page cannot rot.
///
/// `cfg(doctest)` is set only while rustdoc collects doctests, so this item
/// exists to `cargo test` and to nothing else: it is not part of the public
/// API, does not appear in the built documentation, and the crate-level
/// documentation above stays the curated text rather than the README.
///
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;
