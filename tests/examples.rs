#![forbid(unsafe_code)]
//! Runs both examples as real processes, so they cannot rot.
//!
//! The examples are integration collateral: they are the two shapes decibri
//! will drive this crate in, and each one carries its own assertions: the
//! identity-rate count guarantee in `decode_whole_file`, and bit-identical
//! agreement between the streamed and whole-file paths in `decode_stream`. A
//! failed assertion is a non-zero exit, which these tests turn red.
//!
//! Plain `cargo test` builds the examples before any test runs, so the
//! binaries exist by the time this file executes. A bare
//! `cargo test --test examples` does not build them, which is what the missing
//! -binary message below is about.

use std::path::PathBuf;
use std::process::Command;

/// The compiled example beside this test binary: the test runs from
/// `target/<profile>/deps/`, the examples live in `target/<profile>/examples/`,
/// and the layout is the same under a `--target` directory.
fn example(name: &str) -> PathBuf {
    let mut path = std::env::current_exe().expect("the test binary has a path");
    path.pop(); // deps/
    path.pop(); // the profile directory
    path.push("examples");
    path.push(format!("{name}{}", std::env::consts::EXE_SUFFIX));
    assert!(
        path.is_file(),
        "example binary not found at {}, run plain `cargo test`, which builds the examples",
        path.display()
    );
    path
}

/// Runs an example with no arguments and returns its stdout, failing the test
/// on a non-zero exit.
fn run(name: &str) -> String {
    let path = example(name);
    let output = Command::new(&path)
        .output()
        .unwrap_or_else(|error| panic!("could not run {}: {error}", path.display()));
    assert!(
        output.status.success(),
        "{name} exited with {:?}\nstdout:\n{}stderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("example output is UTF-8")
}

#[test]
fn the_whole_file_example_runs_and_holds_its_count_guarantee() {
    let stdout = run("decode_whole_file");
    assert!(
        stdout.contains("input: 2 channel(s) at 44100 Hz, 4410 frames"),
        "probe line missing from:\n{stdout}"
    );
    assert!(
        stdout.contains("mono f32 at 16000 Hz"),
        "output line missing from:\n{stdout}"
    );
    assert!(
        stdout.contains("identity-rate decode holds exactly 4410 frames"),
        "identity count line missing from:\n{stdout}"
    );
}

#[test]
fn the_streaming_example_runs_and_matches_the_whole_file_path() {
    let stdout = run("decode_stream");
    assert!(
        stdout.contains("mono f32 at 16000 Hz"),
        "output line missing from:\n{stdout}"
    );
    assert!(
        stdout.contains("streamed output is bit-identical to the whole-file path"),
        "agreement line missing from:\n{stdout}"
    );
}
