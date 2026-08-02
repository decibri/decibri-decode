# Welcome to the decibri-decode Contribution Guide

Thank you for investing your time in contributing to decibri-decode. We welcome all sorts of different contributions.

Before making any type of contribution, please read our [Code of Conduct](CODE_OF_CONDUCT.md) to keep our community approachable and respectable.

This guide walks through the contribution workflow, from opening an issue to submitting a pull request.

## New contributor resources

For a good overview of the project, please first read the [README](README.md). General resources for getting started with open-source contributions:

- [Finding ways to contribute to open source on GitHub](https://docs.github.com/en/get-started/exploring-projects-on-github/finding-ways-to-contribute-to-open-source-on-github)
- [Collaborating with pull requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests)

## Ways to contribute

There are multiple ways you can contribute to this project:

- Reporting a bug
- Submitting a fix
- Suggesting new features or improvements
- Adding or updating documentation
- Improving test coverage
- Reporting a file that decodes incorrectly, with the file attached where you are able to share it
- Anything else we may have forgotten

## Getting started

### Prerequisites

To build from source you will need:

- **Rust 1.88 or later** via [rustup](https://rustup.rs/)

That is the whole list. This crate has one dependency, no C toolchain, no build script and no platform-specific requirements. It builds and tests identically on Windows, macOS and Linux.

### Setting up the development environment

1. Fork this repository to your own account and clone it to your local machine:

   ```bash
   git clone https://github.com/YOUR_USERNAME/decibri-decode.git
   cd decibri-decode
   ```

2. Build the crate and run its tests:

   ```bash
   cargo build
   cargo test
   ```

The full suite takes about a minute and needs no audio hardware, no network access and no external tools.

### The optional FLAC conformance corpus

Six tests exercise the [FLAC decoder testbench](https://github.com/ietf-wg-cellar/flac-test-files), which is roughly 310 MB and is not carried in this repository. Those tests skip cleanly when the corpus is absent, so `cargo test` passes without it.

To run them, clone the testbench somewhere outside this repository and point at it:

```bash
DECIBRI_FLAC_CORPUS=/path/to/flac-test-files cargo test
```

If you are changing anything in the FLAC paths, please run with the corpus before opening a pull request.

### Repository layout

- `src/`: the library
- `tests/`: integration tests and conformance suites
- `examples/`: the examples used in the README, compiled as doctests so they cannot drift

### Project conventions

These are stricter than most Rust projects and they are deliberate. A pull request that breaks one of them will be asked to change, so it is worth reading before you start.

**No new dependencies.** The crate depends on `decibri-resampler` and nothing else. If a change appears to need a dependency, please open an issue first so we can talk about it rather than declining a finished pull request.

**No unsafe code in the library.** `src/` carries `#![forbid(unsafe_code)]`. One integration test implements `GlobalAlloc` to measure what the parsers allocate on malformed input, and that is the only exception in the repository.

**No `#[allow]` to silence a lint.** `cargo clippy --all-targets -- -D warnings` must pass without suppressions. If a lint is genuinely wrong for a piece of code, say so in the pull request and we will discuss it.

**ASCII only in tracked source.** A test enforces this. It exists because a text substitution tool once mangled non-ASCII characters in a source file and every other gate passed with the corrupted file.

**Every claim is measured.** The README, the changelog and the documentation state numbers that came from a test rather than from an estimate. Please do not add a claim that nothing verifies.

**Pinned witnesses must not move.** Several tests hash decoded output against a pinned value, which is how the crate proves its output is identical across platforms and toolchains. If your change moves a witness, that is a behaviour change and the pull request needs to say what changed and why. A witness is re-pinned only when its input changed, never to make a failing hash pass.

**Codecs are written from their specifications.** G.711 comes from the ITU-T recommendation, FLAC from RFC 9639, MD5 from RFC 1321. No codec in this crate is a port of another implementation. If you contribute codec work, please write it from the specification rather than from someone else's source.

### Reporting a bug

We use GitHub Issues to track bugs. All open, pending and closed cases are at [decibri-decode Issue Tracking](https://github.com/decibri/decibri-decode/issues).

Before opening a new issue, please search [existing issues](https://github.com/decibri/decibri-decode/issues) to see if the bug has already been reported. You may be able to add more information or your own experience to an existing issue.

If no related issue exists, you can open a new one using the [issues form](https://github.com/decibri/decibri-decode/issues/new).

To help us reproduce and fix bugs quickly, please include the following where applicable:

- A quick summary and background
- Your operating system and architecture (for example Windows 11 x64, macOS arm64, Ubuntu 22.04 x64)
- Rust version (`rustc --version`)
- The crate version
- Steps to reproduce the bug
- Code samples that trigger the issue
- What you expected to happen against what actually happened
- Exact error messages

**If a file decodes incorrectly or is rejected when it should not be, the file itself is the most useful thing you can send.** Please attach it if you are able to share it. If you cannot, tell us what produced it and what its format and sample rate are, and we will try to reproduce one.

**If the issue is that a malformed file causes a panic, a hang or excessive memory use, please report it privately instead.** See [SECURITY.md](SECURITY.md).

### Proposing codebase changes

We welcome contributions from everyone interested in making decibri-decode better. To propose a change:

1. Fork this repository and clone it to your local machine.
2. Create a new branch from `main` with a descriptive name that reflects your changes.
3. Make your changes.
4. Test your changes thoroughly:

   ```bash
   cargo test
   cargo test --release
   ```

   With the corpus, if you touched the FLAC paths:

   ```bash
   DECIBRI_FLAC_CORPUS=/path/to/flac-test-files cargo test
   ```

5. Run the linters:

   ```bash
   cargo clippy --all-targets -- -D warnings
   cargo fmt --all -- --check
   RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
   ```

   Note the `--all-targets` on clippy, which lints test code as well. Test code ships in the published package, so it is published source.

6. Commit your changes with a clear and descriptive commit message.
7. Push your branch to your fork.
8. Open a pull request against the `main` branch of this repository. Include a description of your changes, the reasons for them, and the benefits they provide.

Our team will review your pull request and provide feedback. We may ask for additional changes, so please be prepared to iterate before merging.

### Adding a format

New format support is welcome, and there are two things to settle before writing code, so please open an issue first.

**Licensing.** The crate carries no format whose specification or patents impose obligations on users. Formats have been excluded on those grounds, so a proposal needs to establish the position rather than assume it.

**The oracle.** Every format in the crate is verified against something that did not come from this crate. G.711 has the published ITU tables, FLAC has a conformance corpus and a checksum inside every file, WAV and AIFF have hand-built fixtures. A format with no independent reference to check against is hard to accept, because a decoder and its tests agreeing with each other proves very little.

### CI pipeline

Every pull request runs an automated pipeline covering build, lint, format, documentation, dependency audit and the test suite across Linux, macOS and Windows, on the minimum supported Rust version as well as stable, and on a 32-bit target.

It also runs a determinism matrix that decodes the same input across every operating system and toolchain and requires byte-identical output. That job is the one that enforces the crate's central claim, so a change that fails it is a real finding rather than a flaky test.

Your pull request must pass CI before it can be merged. Details of the pipeline live in `.github/workflows/ci.yml`.

We appreciate your contributions and thank you for your time in submitting a pull request.

## Contributor License Agreement

Before your first contribution can be merged, we ask you to agree to the decibri Contributor License Agreement. It is a one-time step that lets the project include your work under its current and future licenses, with clear provenance, and it does not take away your copyright in what you contribute. You are welcome to read the full agreements first: the [Individual CLA](https://github.com/decibri/decibri-cla-action/blob/main/agreements/Individual-CLA-v1.md) and, for contributions made on behalf of a company, the [Corporate CLA](https://github.com/decibri/decibri-cla-action/blob/main/agreements/Corporate-CLA-v1.md).

When you open a pull request, an automated check looks at whether you are already covered. If you are not, it leaves a comment with a short sentence to agree to. Reply with that exact sentence as a comment on your own pull request, and the check turns green. That is the whole process, and once you have done it you are covered for your future contributions too. Until the check passes, the pull request cannot be merged.

If you are contributing as part of your work, your employer may need a Corporate CLA on file instead of an individual one. If that applies to you, or the check asks about it, contact the maintainers and we will sort it out.

The record we keep is deliberately minimal: your GitHub username and account ID, which version of the agreement you agreed to, and the date. How we handle that information, and how to request its removal, is set out in our [Privacy Policy](https://decibri.com/privacy).

The CLA covers your contributions across the decibri organisation's repositories, so you only need to agree once.

## License

The decibri-decode source is released under the [Apache License 2.0](LICENSE).

Contributions are governed by the Contributor License Agreement described above. Under the CLA you keep your copyright in what you contribute and grant the project the rights it needs to include and license your work, including under future licenses. Contributed code or content must be your own work, and you confirm that you have the right to grant those rights.
