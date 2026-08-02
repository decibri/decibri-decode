# Security

decibri-decode parses audio files and byte streams that may come from anywhere, so the security surface is narrow and well defined. **The crate reads untrusted input.** Everything below follows from that.

If you believe you have found a security vulnerability in this repository, please report it as described below.

## Responsible Disclosure

We are strongly committed to the responsible disclosure of security vulnerabilities. Please follow these guidelines when reporting security issues:

- Email [hello@decibri.com](mailto:hello@decibri.com) with "SECURITY - decibri-decode" in the subject line.
- Alternatively, use [GitHub's private vulnerability reporting](https://github.com/decibri/decibri-decode/security/advisories/new) to report directly through GitHub.
- Please do not report security vulnerabilities through public GitHub issues.

When reporting, please include the following details where applicable:

- A description of the issue and how it can be triggered
- **The input that triggers it**, attached if you are able to share it. For this crate the file or byte sequence is usually the whole report
- The affected version of the crate
- The platform and architecture
- Rust version
- Steps to reproduce
- Any other relevant information

We review reports as quickly as possible and work with reporters to coordinate remediation and disclosure.

## What counts as a vulnerability here

The crate has no network access, no filesystem access beyond what a caller hands it, no process execution and no unsafe code in the library. So memory unsafety is not reachable. What is reachable, and what we treat as a security issue, is any input that causes:

- A panic, which takes down the caller
- A hang or an unbounded loop
- Memory use disproportionate to the size of the input
- Processing time disproportionate to the size of the input

**Please report all four privately** rather than through a public issue.

A file that decodes to the wrong samples is a correctness bug rather than a security issue, and a normal issue is the right place for it.

## How the crate is protected

### Memory safety

The library carries `#![forbid(unsafe_code)]`, so no unsafe block can compile into it. One integration test implements `GlobalAlloc` in order to measure allocation, and that is the only exception in the repository. A test enforces that every other file forbids unsafe as well, so the exception is one named file rather than an absence of policy.

### Sizes in a file are treated as claims rather than facts

A malformed file can declare any length it likes. Nothing in the crate allocates against a declared size. Buffers are bounded by what the format itself permits or by what has actually arrived, whichever is smaller. Those ceilings are measured by a test using an allocator counter, not asserted.

All chunk and offset arithmetic is 64-bit and checked, verified on a 32-bit target where the conventional `usize` form would wrap.

### Every failure is typed

Malformed input returns a typed error naming what was rejected. Panics, hangs and silent truncation are treated as defects rather than acceptable behaviour on bad input. A file that ends part way through a frame is an error rather than quietly shortened audio.

### Fuzzing

The parsers are fuzzed with both coverage-guided and seeded deterministic campaigns, under overflow checks and debug assertions. Before the first release the crate reached over 700 million executions across seven targets with no panics. Every crash ever found is kept as a committed regression test carrying the literal bytes, so it can never recur unnoticed.

Exhaustive prefix and single-byte-mutation sweeps run over the container formats as ordinary tests.

### Determinism

Decoded output is byte-identical across operating systems, toolchains and optimisation levels, and CI enforces that with pinned hashes on every pull request. This is a correctness property rather than a security one, but it means a report reproduces the same way on our machine as on yours.

## Dependencies

The crate has **one** direct dependency, `decibri-resampler`, which brings `thiserror` and its proc-macro chain. There is no C toolchain, no build script and no vendored code.

- Dependencies are monitored by Dependabot for security advisories and version updates.
- `cargo-audit` runs on every pull request against the RustSec advisory database.
- `Cargo.lock` is committed so CI builds from a fixed dependency resolution.

## Supply chain

The crate contains no third-party code. Every codec was written from its published specification rather than ported from another implementation, so there is no upstream to inherit a vulnerability from. The specifications used are recorded in [ATTRIBUTION.md](ATTRIBUTION.md).

The first release to crates.io was published manually. Subsequent releases use keyless [Trusted Publishing via OIDC](https://crates.io/docs/trusted-publishing), where a short-lived, crate-specific publish token is issued per run by exchanging a GitHub OIDC token. No long-lived crates.io token is stored in this repository or in CI.

The full build and release configuration is open source and auditable in `.github/workflows/`.

## Supported Versions

| Version | Supported              |
|:-------:|:----------------------:|
| 0.1.x   | :white_check_mark: Yes |

Security fixes are applied to the latest release. While the crate is in its `0.x` series, a fix may arrive in a new minor version rather than a patch, since `0.x` minor versions are the semantic versioning boundary for breaking changes.

## CVE Policy

For confirmed vulnerabilities we will request a CVE identifier where appropriate and publish a GitHub Security Advisory with details of the issue, affected versions and remediation steps. Security advisories are visible at the [decibri-decode security advisories page](https://github.com/decibri/decibri-decode/security/advisories).

## Security Best Practices for Users

- Keep your dependencies up to date, and use `cargo audit` to check for known vulnerabilities in your tree
- Treat any audio arriving from a user or a network as untrusted, and handle the returned error rather than unwrapping it
- Set your own limit on input size before handing bytes to this crate. It bounds its own allocation, but a caller who reads an arbitrarily large file into memory first has already spent that memory
- Prefer the streaming path for input of unknown size, since it decodes as bytes arrive rather than requiring the whole input in memory
- Pin dependencies to specific versions in production environments

## Reporting Concerns About This Policy

If you have questions about this security policy itself, or suggestions for improvement, please open a regular issue on the repository. These are not security vulnerability reports and do not require private disclosure.

## Acknowledgments

Thank you to the researchers and community members who help keep decibri users secure. If you report a valid issue and would like public acknowledgment, we will credit you in the security advisory and release notes.

## Contact

For security questions, email [hello@decibri.com](mailto:hello@decibri.com) with "SECURITY - decibri-decode" in the subject line.
