#![forbid(unsafe_code)]
//! The encoding of this repository's own source, checked at the byte level.
//!
//! # Why this gate exists
//!
//! While `src/flac.rs` was being written, a PowerShell 5.1 text substitution
//! re-encoded it: every non-ASCII character was double-encoded from UTF-8
//! into UTF-8 and a byte-order mark was prepended. The crate then **built,
//! formatted, linted and passed 262 tests** with corrupted source. Nothing in
//! the toolchain objects: rustc accepts a leading BOM, an em dash whose three
//! UTF-8 bytes have each been re-encoded as a separate character is still
//! valid UTF-8 inside a comment, and no test reads its own source. The
//! corruption was found by hashing the file's bytes against a copy, which is
//! not a thing anybody does twice.
//!
//! So this is a silent corruption path that every other gate in the crate is
//! blind to, and it is closed here rather than remembered.
//!
//! # Prevention, not detection
//!
//! This gate used to carry an allowlist of the six typographic characters the
//! repository used, on the reasoning that every mangling of those six
//! produces characters outside the list, so a corrupted file fails on its
//! first character. That is *detection*, and detection is the weaker of the
//! two available rules: it holds only for as long as the allowlist is exactly
//! right, and it leaves the corruption possible.
//!
//! The rule now is **plain ASCII**, everywhere in the tracked source, and the
//! six characters were removed to make it true. A re-encode of an ASCII file
//! has nothing to work on: there is no multi-byte sequence to double-encode,
//! so the failure mode that produced the incident cannot be constructed. The
//! byte-order mark check stays as its own case, because a BOM is the one
//! non-ASCII sequence a tool adds to a file that had none.
//!
//! Nothing is lost by it. An em dash was doing a comma's or a full stop's
//! work in prose, box-drawing rules were hyphens with a nicer shape, and the
//! arithmetic signs had ASCII spellings already. The published documentation
//! reads the same.
//!
//! # What is checked
//!
//! Three things, in increasing strength:
//!
//! 1. **No byte-order mark.** A BOM is invisible in an editor, survives every
//!    build, and is the signature an accidental re-encode leaves behind.
//! 2. **Valid UTF-8.** Rust source must be, and the two data formats here are
//!    read as text by tooling that assumes it.
//! 3. **Every byte is ASCII.** The rule above, stated as the byte test it is.
//!    A file that fails this either gained a character somebody meant to add,
//!    in which case the decision is to spell it in ASCII, or gained one
//!    nobody meant to add, which is the incident.

use std::path::{Path, PathBuf};

/// The file extensions this gate treats as text.
///
/// Everything the repository tracks is one of these; a binary fixture would
/// be neither.
const TEXT: &[&str] = &["rs", "md", "toml", "yml", "yaml", "lock", "txt"];

/// Directory names never descended into: build output and git's own store.
const SKIP_DIRS: &[&str] = &["target", ".git"];

/// Files not part of the tracked source.
///
/// `CLAUDE.local.md` is gitignored by rule, since it carries machine-local
/// conventions and must never be published, so it is not tracked source and
/// is not held to the tracked source's rule.
const SKIP_FILES: &[&str] = &["CLAUDE.local.md"];

/// The repository root, which is where this test binary's crate lives.
fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Every text file under `directory`, recursively, in a stable order.
fn text_files(directory: &Path, into: &mut Vec<PathBuf>) {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("reading {}: {error}", directory.display()))
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .collect();
    entries.sort();
    for path in entries {
        let name = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        if path.is_dir() {
            if !SKIP_DIRS.contains(&name.as_str()) {
                text_files(&path, into);
            }
        } else if !SKIP_FILES.contains(&name.as_str())
            && path
                .extension()
                .is_some_and(|extension| TEXT.iter().any(|text| extension == *text))
        {
            into.push(path);
        }
    }
}

#[test]
fn no_tracked_source_file_carries_a_byte_order_mark_or_a_non_ascii_byte() {
    let root = root();
    let mut files = Vec::new();
    text_files(&root, &mut files);
    assert!(
        files.len() > 10,
        "only {} text files were found under {}; is the walk working?",
        files.len(),
        root.display()
    );

    let mut failures: Vec<String> = Vec::new();
    for path in &files {
        let name = path
            .strip_prefix(&root)
            .unwrap_or(path)
            .display()
            .to_string();
        let bytes = std::fs::read(path).unwrap_or_else(|error| panic!("reading {name}: {error}"));

        if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
            failures.push(format!("{name}: starts with a UTF-8 byte-order mark"));
            continue;
        }
        if bytes.starts_with(&[0xFF, 0xFE]) || bytes.starts_with(&[0xFE, 0xFF]) {
            failures.push(format!("{name}: starts with a UTF-16 byte-order mark"));
            continue;
        }
        let text = match std::str::from_utf8(&bytes) {
            Ok(text) => text,
            Err(error) => {
                failures.push(format!("{name}: is not valid UTF-8 ({error})"));
                continue;
            }
        };
        for (index, character) in text.char_indices() {
            if character.is_ascii() {
                continue;
            }
            let line = text[..index].matches('\n').count() + 1;
            failures.push(format!(
                "{name}:{line}: U+{:04X} {character:?} is not ASCII, and this repository's \
                 tracked source is ASCII so that a re-encode has nothing to corrupt",
                character as u32
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "{} encoding problem(s) across {} text files:\n{}",
        failures.len(),
        files.len(),
        failures.join("\n")
    );
    println!(
        "{} text files checked: no byte-order mark, valid UTF-8, every byte ASCII",
        files.len()
    );
}

#[test]
fn the_library_source_is_ascii_only_and_is_the_reason_the_rule_can_be_ascii() {
    // Reported as a measurement rather than folded into the walk above, so a
    // reader of the log sees `src/` named specifically. The rule applies to
    // the whole tracked tree; `src/` is the part the incident happened in.
    let source = root().join("src");
    let mut files = Vec::new();
    text_files(&source, &mut files);
    assert!(!files.is_empty(), "src/ holds no source files");

    let mut offenders: Vec<String> = Vec::new();
    let mut bytes_seen = 0usize;
    for path in &files {
        let raw = std::fs::read(path).expect("a source file reads");
        bytes_seen += raw.len();
        let name = path
            .file_name()
            .expect("a file has a name")
            .to_string_lossy();
        for (offset, byte) in raw.iter().enumerate() {
            if !byte.is_ascii() {
                offenders.push(format!("{name}: byte 0x{byte:02x} at offset {offset}"));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "src/ is not ASCII-only:\n{}",
        offenders.join("\n")
    );
    println!(
        "src/: {} files, {bytes_seen} bytes, every one of them ASCII",
        files.len()
    );
}
