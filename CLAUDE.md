# decibri-decode

Encoded audio bytes in, `f32` samples out, with the sample rate and channel
count travelling with them and a stated and tested sample-count guarantee. The
README is the contract; this file is the working conventions.

## Gates

Every change passes all of these, run from the repository root, before it is
considered done:

```text
cargo build --all-targets
cargo clippy --all-targets -- -D warnings
cargo test                                  # builds and exercises the examples too
cargo fmt --check
cargo doc --no-deps                         # with RUSTDOCFLAGS=-D warnings
cargo publish --dry-run                     # publishing itself is always manual
cargo tree -i decibri-resampler             # exactly one entry
```

Before a release, additionally:

```text
rustup run 1.88.0 cargo test                # the MSRV floor, verified not declared
cargo test --target i686-pc-windows-msvc    # 32-bit: where usize arithmetic differs
cargo test --release                        # witnesses under release codegen
DECIBRI_FLAC_CORPUS=<path> cargo test --release --test flac_corpus
```

The last one is the FLAC conformance corpus, which is **not in this
repository**: it is `github.com/ietf-wg-cellar/flac-test-files`, CC0-1.0 and
therefore carriable, but 310 MB and therefore not carried. Without the
variable set the gate prints where to get the corpus and passes, so a
checkout without it is not a failing suite. `--release` because an
unoptimised decode of 310 MB is minutes rather than seconds.

Clippy runs `--all-targets` deliberately, because tests and examples ship in
the published tarball and so are published source. Doc builds fail on warnings
deliberately, because a broken intra-doc link is a defect rather than a log
line. Both are divergences from decibri's CI and are commented in
`.github/workflows/ci.yml`; do not "fix" them back.

Tracked source is **ASCII, every byte**, enforced by `tests/source_encoding.rs`
and named as its own CI step. Spell an em dash as a comma or a full stop, a
rule as hyphens, and the arithmetic signs as `*`, `-` and `+/-`.

## Hard rules

- **One dependency**: `decibri-resampler`, with a version requirement kept
  identical to the one decibri declares, so cargo unifies the two edges into a
  single tree node. Two resampler implementations in one binary is a
  correctness hazard. No other dependency, dev or otherwise.
- **No `unsafe` in `src/`**: `#![forbid(unsafe_code)]` on the library crate.
  Every test file carries its own `forbid` except `tests/allocation_ceiling.rs`,
  whose `GlobalAlloc` counter is `unsafe` to implement by definition; that
  exception stays exactly one named file.
- **No `#[allow]`** to silence a clippy warning. Restructure the code instead.
- **`#![warn(missing_docs)]` stays clean**: every public item has a doc
  comment, and the comment says why, not just what.
- **`samples` means interleaved, `frames` means interchannel.** One value per
  channel is a frame; frames times channels is samples. Every public count,
  length or position states which of the two it is, in those words. A name
  that says one while the value is the other is a defect rather than a
  documentation gap.
- **No claim that is not measured.** README, CHANGELOG and rustdoc state what
  was run and observed, never what ought to be true.
- **Nothing public-facing carries design rationale.** The README, the
  CHANGELOG, `Cargo.toml` comments and every doc comment state what the crate
  does. They do not explain why a choice was made, what was considered, or
  what was rejected.

## The claims and their exact boundaries

- **Determinism**: decoding is bit-exact and cross-platform byte-identical for
  every carried format, and FLAC encoding is too. Enforced by pinned FNV-1a
  witness hashes over the decoded output of every format, over both probe
  entry points and over the FLAC writer's encoded bytes, run across operating
  systems, architectures and toolchains in CI. The encoder's searches use an
  in-crate cosine and logarithm rather than a libm call, which is what makes
  its hash pinnable. A witness is re-pinned only when its input changes, such
  as a format added to the sweep, never to make a failing hash pass; a hash
  that moved on unchanged input is a determinism break. The FLAC write witness
  also moves when the search changes, and that re-pin is legitimate only with
  the size benchmark re-run and reported.
- **The FLAC encoder shares the decoder's arithmetic.** Residuals are computed
  by subtracting `fixed_prediction` and `lpc_prediction`, the same functions
  decoding adds, at the same widths in the same order. The encoder does not
  get its own prediction path. The independent gates are the reference tool
  runs and the corpus, not the round trip.
- **Losslessness** holds at or below 24 significant bits (`u8`, `i8`, 16-bit,
  24-bit, `f32`, and FLAC at bit depths up to 24), and deliberately not for
  `i32`, `f64` or FLAC above 24 bits, whose values land on the nearest
  representable `f32`. FLAC is a lossless codec and this crate reproduces its
  integers exactly; the boundary is the `f32` significand it then scales into.
  Do not restate losslessness without the boundary.
- **The FLAC checksums are a runtime feature, not a test.** Every FLAC stream
  carries an MD5 of its own unencoded audio, so the decoder verifies its
  output on every file rather than only on test files, on both the whole-file
  and streaming paths. An all-zero checksum means unset and is not a failure.
  The frame CRC-8 and CRC-16 are verified before any sample from that frame is
  delivered.
- **The crate has no Cargo features**, and adding one needs a measured case.
  Adding a feature later is not a breaking change and removing one is.
- **The probe reads twelve bytes, not four.** `RIFF` and `FORM` name container
  families, so the form type at offset eight decides. A RIFF that is not
  `WAVE`, or a `FORM` that is neither `AIFF` nor `AIFC`, is
  `UnsupportedContainer` naming the form type found. Headerless PCM, headerless
  G.711 and bare FLAC frames have no signature and stay explicit; the probe
  does not sniff for a bare frame.
- **The unsafe claim is about the library, not the repository.** The packaged
  `tests/` contain the one `GlobalAlloc` exception above.
- **No rate policy**: no default sample rate anywhere, and G.711 does not
  default to 8000. No branching on file name or extension, and no resampling
  logic. Rate conversion is `decibri-resampler`, called, never reimplemented.
  