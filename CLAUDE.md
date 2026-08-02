# decibri-decode

Encoded audio bytes in, mono `f32` at a declared sample rate out, with a
stated and tested sample-count guarantee. The README is the contract; this
file is the working conventions.

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
and named as its own CI step. That is prevention rather than detection: a
re-encode of an ASCII file has no multi-byte sequence to double-encode, so the
corruption that once passed every other gate cannot be constructed. Spell an
em dash as a comma or a full stop, a rule as hyphens, and the arithmetic signs
as `*`, `-` and `+/-`.

## Hard rules

- **One dependency**: `decibri-resampler`, with a version requirement kept
  identical to the one `crates/decibri` declares, so cargo unifies the two
  edges into a single tree node. Two resampler implementations in one binary
  is a correctness hazard (resampler choice moved AECMOS by up to 1.03 in
  the AEC benchmarking work). No other dependency, dev or otherwise.
- **No `unsafe` in `src/`**: `#![forbid(unsafe_code)]` on the library crate.
  Every test file carries its own `forbid` except `tests/allocation_ceiling.rs`,
  whose `GlobalAlloc` counter is `unsafe` to implement by definition; that
  exception stays exactly one named file.
- **No `#[allow]`** to silence a clippy warning. Restructure the code instead.
- **`#![warn(missing_docs)]` stays clean**: every public item has a doc
  comment, and the comment says why, not just what.
- **`samples` means interleaved, `frames` means interchannel.** One value
  per channel is a frame; frames times channels is samples. Every public
  count, length or position states which of the two it is, in those words.
  A name that says one while the value is the other is a defect and not a
  documentation gap, which is why `FlacSkip::frames` and
  `FlacRecoveryReport::frames_lost` carry the names they do rather than the
  `samples` spellings they were written with: a caller subtracting two
  things whose names say they are the same quantity is wrong by the channel
  count and gets no warning.
- **No claim that is not measured.** README, CHANGELOG, rustdoc and reports
  state what was run and observed, never what ought to be true.

## The claims and their exact boundaries

- **Determinism**: decoding is bit-exact and cross-platform byte-identical
  for every carried format, and FLAC *encoding* is too. Enforced by pinned
  FNV-1a witness hashes over the
  decoded output, PCM `0x57ac_66d6_2a28_b665`, WAV `0x51fe_2597_ebf5_2432`,
  AIFF `0x428d_f9aa_ce0c_4172`, FLAC `0x97ef_b751_3ce8_8469`, the probe sweep
  `0x5ded_2068_3c43_3e99` and its FLAC route `0xad35_8612_e054_4fa5`, plus the
  G.711 witness, and over the FLAC writer's encoded bytes,
  `0x7d49_2b0a_1148_06b4`, all run across OSes, architectures and toolchains
  in CI. The encoder's searches use no libm call (in-crate cosine and
  logarithm), which is what makes its hash pinnable. A
  witness is re-pinned only when its *input* changes (a format added to the
  sweep), never to make a failing hash pass; a hash that moved on unchanged
  input is a determinism break. The FLAC write witness also moves when the
  *search* is deliberately improved; that re-pin is legitimate only with the
  size benchmark re-run and reported.
- **The FLAC encoder shares the decoder's arithmetic.** Residuals are
  computed by subtracting `fixed_prediction` and `lpc_prediction`, the same
  functions decoding adds, at the same widths in the same order. Do not give
  the encoder its own prediction path: the failure mode is a stream that
  round-trips through this crate perfectly and decodes differently
  everywhere else, on loud material at high depths that no small test
  catches. The independent gates are the reference tool runs and the
  corpus, not the round trip.
- **Losslessness** holds at or below 24 significant bits (`u8`, `i8`, 16-bit,
  24-bit, `f32`, and FLAC at bit depths up to 24), and deliberately not for
  `i32`, `f64` or FLAC above 24 bits, whose values land on the nearest
  representable `f32`. FLAC is a lossless *codec* and this crate reproduces
  its integers exactly; the boundary is the `f32` significand it then scales
  into, and it is the same boundary as everywhere else. Do not restate
  losslessness without the boundary.
- **The FLAC checksums are a runtime feature, not a test.** Every FLAC stream
  carries an MD5 of its own unencoded audio, so the decoder verifies its
  output on every file rather than only on test files, on both the whole-file
  and streaming paths. An all-zero checksum means unset and is not a failure.
  The frame CRC-8 and CRC-16 are verified before any sample from that frame is
  delivered. This is what makes the FLAC gate stronger than the others: the
  oracle ships inside the data and cannot share a bug with this decoder.
- **The crate has no Cargo features**, and adding one needs a measured case.
  A feature would mean identical bytes behaving differently depending on how
  the crate was built, which contradicts what the crate promises. Adding a
  feature later is not a breaking change and removing one is, so the cheap
  position is none.
- **The probe reads twelve bytes, not four.** `RIFF` and `FORM` name container
  families, so the form type at offset eight decides. A RIFF that is not
  `WAVE`, or a `FORM` that is neither `AIFF` nor `AIFC`, is
  `UnsupportedContainer` naming the form type found, never a reader handed a
  file that was never its. Headerless PCM, headerless G.711 and bare FLAC
  frames have no signature and stay explicit; sniffing for a bare frame is
  measured as unsafe and is not done.
- **The unsafe claim is about the library, not the repository.** The packaged
  `tests/` contain the one `GlobalAlloc` exception above.
- **No rate policy**: no default sample rate anywhere (G.711 does not default
  to 8000), no branching on file name or extension, no resampling logic.
  Rate conversion is `decibri-resampler`, called, never reimplemented.

## Coverage lessons, learned the expensive way

Five negative controls across the crate's construction each exposed a way a
green suite can say less than it appears to. They are recorded here because
the next one would otherwise be rediscovered through a sixth control:

1. **An exhaustive value-domain test proves nothing about the byte path.**
   Breaking i24 sign extension survived a 16,777,216-value round trip,
   because the round trip never went through bytes.
2. **A test's coverage is bounded by the dimensions its inputs vary along.**
   A byte-path gate at ten chunk sizes missed a size-dependent branch because
   every input was 256 bytes and the branch triggered at 65,536.
3. **One rule with two implementations means a control exercises only one.**
   The RIFF pad rule lived in two places; a control on the first said nothing
   about the second. (The rule is now one function, `riff::pad_len`, and a
   later control confirmed a single break turns both containers' suites red.)
4. **A test whose reference is computed by the path under test proves only
   self-consistency.** The AIFF dimension matrix cannot catch codec-layer
   breaks because its expected values share the codec path; the independent
   anchoring lives in the cross-container agreement gate, which is
   load-bearing, not redundant.
5. **A dimension can hide in the *values* of an input rather than in its
   shape.** The FLAC dimension matrix varied bit depth, channel count,
   channel assignment, block size, sample rate, wasted bits, blocking
   strategy and metadata layout, and every one of its samples was an even
   number, because the generator masked low bits unconditionally. Mid/side
   stereo is only lossless because an odd side sample restores the bit the
   mid sample lost to its right shift, so deleting that correction left the
   whole in-tree suite green while 55 corpus files went red. Sample parity is
   now dimension 14 in `flac_conformance.rs`, and lesson 2 has a sharper
   form: enumerate the dimensions of the *values*, not only of the structure.

The general rule the five share: **enumerate the input dimensions before
writing the tests, and check that the oracle is independent of the path being
tested.**

## Writing conventions for reports

- Lead with what changed in behaviour, then how it is verified.
- Every gate appears with its actual command output, not a summary of it.
- Claims are measured; anything not measured is listed under "what this pass
  could not establish".
- Where the specified shape did not survive contact with the code, say so
  explicitly. An empty "nothing diverged" is itself a required statement.
- Defects noticed outside the crate are recorded in one line each and not
  acted on.
