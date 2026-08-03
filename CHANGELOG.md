# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-08-03

Documentation and package metadata only. The crate's behaviour is unchanged
from 0.1.0. No public item was added, removed or altered, no decoded or
encoded byte differs, and every pinned witness hash holds the value 0.1.0
pinned. Upgrading requires no change to any caller.

### Changed

- The README, the crate-level documentation, the `Cargo.toml` description and
  the doc comment on `AudioSpec::mono` now state that decoding returns
  interleaved samples at the source's own sample rate, with the rate and
  channel count attached, and that downmixing and rate conversion are the
  caller's.
- A README section, "Mixing and rate conversion", naming `downmix_to_mono` for
  mixing and `decibri-resampler` for rate conversion.
- `Cargo.toml` description, keywords and categories. Keywords are `audio`,
  `wav`, `flac`, `aiff` and `g711`. Categories are `multimedia::audio` and
  `multimedia::encoding`.

### Fixed

- The 0.1.0 changelog heading, which read `Unreleased` after 0.1.0 published
  to crates.io on 2 August 2026.

## [0.1.0] - 2026-08-02

### Added

- Linear PCM sample conversion for `u8`, `i8`, 16-, 24- and 32-bit integer,
  and 32- and 64-bit float, in both byte orders. Scale factor `2^(n-1)`, clamp
  then truncate toward zero on encode, arithmetic-mean downmix, and NaN
  converted to silence in both directions at every width.
- G.711 mu-law and A-law, both directions, as `G711Law` for the tables and
  `G711Decoder` for headerless streams. No default sample rate.
- WAV reading through `WavReader` for whole files, accepting chunks in any
  order, and `WavStreamDecoder` for streaming, requiring `fmt ` first with
  bounded buffering.
- WAV writing through `WavWriter` for every encoding the readers decode. Four
  integer PCM widths, two float widths, both G.711 laws, plain or
  `WAVE_FORMAT_EXTENSIBLE`, as RIFF or RF64.
- AIFF and AIFF-C reading through `AiffReader` for whole files and
  `AiffStreamDecoder` for streaming. Signed 8-bit samples, 80-bit
  extended-precision sample rates, `sowt` little-endian data, and `SSND`
  `offset` and `blockSize`.
- AIFF writing through `AiffWriter` for every encoding the readers decode,
  emitting plain `AIFF` for big-endian PCM and `AIFC` with an `FVER` chunk
  where the compression field is required.
- FLAC reading through `FlacReader` for whole streams and `FlacStreamDecoder`
  for streaming, implemented to RFC 9639. Bit depths 4 through 32, 1 through 8
  channels, left/side, side/right and mid/side decorrelation, constant,
  verbatim, fixed-predictor and linear-predictor subframes, 4-bit and 5-bit
  Rice parameters with escaped partitions, fixed and variable block size
  streams, and block sizes and sample rates outside the streamable subset. All
  prediction and decorrelation arithmetic is 64-bit.
- Streaminfo MD5 verification at run time. Whole-file and streaming decodes
  hash the decoded audio as they go and compare against the checksum the
  stream carries, returning `Malformed` on a mismatch. An all-zero checksum
  means unset and is not a failure. MD5 is implemented in-crate to RFC 1321
  and is not public API.
- FLAC frame CRC verification. The frame header CRC-8 and the whole-frame
  CRC-16 are both checked before any sample from that frame is delivered.
- Bare FLAC frame streams through `FlacFrameReader` for whole inputs and
  `FlacStreamDecoder::frames` for streaming, accepting input that begins at a
  frame boundary with no `fLaC` signature and no metadata, with the stream's
  properties read from the first frame header.
- Out-of-band streaminfo through `FlacFrameReader::with_stream_info` and
  `FlacStreamDecoder::frames_with_stream_info`, resolving the two frame header
  codes that defer to streaminfo, sample rate `0b0000` and bit depth `0b000`,
  and restoring MD5 verification to that path. Without it those codes return
  `Malformed` naming the unresolved field.
- `Md5Check`, `#[non_exhaustive]`, reporting whether a decode verified its
  output against a streaminfo checksum and why not when it did not.
- Recovery from arbitrary bytes through `FlacRecovery`, for input that may
  begin part way through a frame, hold garbage or be damaged in the middle. A
  sync point is accepted only when the frame at it passes its CRC-16 and
  another frame header parses where that frame ends. Every decode returns a
  `FlacRecoveryReport` naming the byte ranges that produced no audio
  (`FlacSkip`, `FlacSkipReason`), the frames lost, and where the output starts
  in the stream's own numbering. Recovery is whole-buffer only; a streaming
  decode that meets a corrupted frame returns a typed error.
- FLAC writing through `FlacWriter`, producing the `fLaC` signature, one
  STREAMINFO block and fixed-blocksize audio frames, with no padding, seek
  table or other metadata. Compression levels 0 through 8, default 5: blocks
  of 1152 interchannel samples with fixed predictors below level 3 and 4096
  with linear prediction at 3 and above, predictor orders to 8 at the middle
  levels and 12 at the top, residual partition orders to between 3 and 6, and
  one to three analysis windows. Bit depths 4 through 32, channel counts 1
  through 8, sample rates 1 through 1048575. Per frame stereo decorrelation
  choosing between independent, left/side, right/side and mid/side; constant,
  verbatim, fixed and linear predictor subframes; wasted bits detected and
  declared; 4-bit and 5-bit Rice parameters chosen by exact cost over merged
  partition sums. Residuals are computed through the same integer prediction
  functions the decoder reconstructs with. The STREAMINFO MD5, block size
  bounds, frame size bounds and total sample count are computed and stored.
  Output is byte-identical across platforms and toolchains, held to a pinned
  witness hash. All 71 decodable conformance corpus files re-encode at level 5
  and round-trip to identical samples, and every re-encoded file passes
  `flac -t` under flac 1.5.0. On a fifteen file selection spanning bit depths,
  channel counts and content types, level 5 output measured 0.04 percent
  smaller in aggregate than `flac -5 --no-padding --no-seektable`, per file
  between 0.52 percent smaller and 0.06 percent larger.
- A content probe in three entry points built on one twelve-byte rule.
  `identify` returns a `Container` naming what a run of bytes is without
  decoding it, `decode` returns an `AudioBuffer` from a whole input in any
  carried container, and `AudioStreamDecoder` is the `StreamSource` form,
  holding the leading twelve bytes until the format is known and then feeding
  them to the chosen reader ahead of everything that follows. A RIFF whose
  form type is not `WAVE`, or a `FORM` whose form type is neither `AIFF` nor
  `AIFC`, is `UnsupportedContainer` carrying the form type found. Input
  shorter than twelve bytes is `Truncated` at every length from zero to
  eleven. Headerless PCM, headerless G.711 and bare FLAC frame streams are
  outside the probe and stay explicit.
- A seeded fuzz gate, `tests/fuzz_seeded.rs`: every parser reachable from the
  public API driven over mutations of a seed corpus built from the crate's own
  writers, asserting no panic and no stalled caller loop. Deterministic under
  a hand-rolled PRNG with no dependency.
- A bound on the recovery scan's cumulative allocation, measured in
  `tests/allocation_ceiling.rs` as a total rather than a peak. A scan of 1,024
  maximal frame headers allocates 4,194,240 bytes in total.
- A source encoding gate, `tests/source_encoding.rs`: no byte-order mark,
  valid UTF-8, and every byte ASCII, across every tracked text file.
- No Cargo features. Every format the crate reads is in every build of it.
- `FlacStreamInfo`, `#[non_exhaustive]`, reporting the streaminfo fields.
  Unknown frame sizes, an unknown total sample count and an unset MD5 are
  `None` rather than the zero the format stores. `FlacStreamInfo::from_block`
  parses the 34-byte block body a container carries out of band, such as Ogg's
  identification header or Matroska's `CodecPrivate`.
- Format dispatch on content. The probe dispatches on the container magic and
  the form type at offset eight. WAV dispatches on the
  `(wFormatTag, wBitsPerSample)` pair. AIFF dispatches on the form type and
  the compression four-CC. Nothing dispatches on a file name or extension.
- `DecodeError`, `#[non_exhaustive]`, in eight variants, each naming the
  specific thing rejected. A partial trailing frame returns `Truncated`.
  `numSampleFrames` disagreeing with the `SSND` length returns `Truncated`
  when under and `Malformed` when over. Fractional AIFF sample rates and audio
  past AIFF's 32-bit size fields return `Malformed`. A FLAC frame whose sample
  rate, channel count or bit depth differs from streaminfo returns
  `Malformed`, as does a stream carrying more audio than it declares; one
  carrying less returns `Truncated`.
- The `Decoder` trait, object-safe, with `feed`, `decode`, `flush` and
  `reset`, plus `PcmDecoder` for headerless PCM and the `StreamSource` trait.
- `AudioBuffer` and `AudioSpec`, carrying sample rate and channel count with
  the samples.
- Bit-exact cross-platform decoding for every carried format and for both
  probe entry points, held to pinned witness hashes across toolchains, targets
  and optimisation levels, and `AiffWriter` output held byte for byte against
  hand-built conformance fixtures.
- Bounded behaviour on malformed input. No panic and no hang under an
  exhaustive prefix and single-byte-mutation sweep over all three containers.
  Measured with a `GlobalAlloc` counter, whole-file rejection paths peak at 16
  bytes or allocate nothing, and streaming rejection paths peak at 13,324
  bytes. FLAC metadata block lengths, streaminfo frame sizes and the
  streaminfo total sample count reach no allocation at all; a FLAC frame's
  declared block size reaches one, bounded by the streaminfo maximum block
  size and by the format's ceiling of 4,194,240 bytes. All chunk arithmetic is
  64-bit and checked, verified on `i686-pc-windows-msvc` with sizes chosen to
  wrap 32-bit arithmetic.
- `#![forbid(unsafe_code)]` on the library. The packaged tests carry one named
  exception, a `GlobalAlloc` allocation counter.
- One dependency, `decibri-resampler` 0.4.
