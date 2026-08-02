#![forbid(unsafe_code)]
//! The FLAC decoder testbench, run against this crate.
//!
//! # Why this corpus is not in the repository
//!
//! The testbench is Martijn van Beurden's, at
//! `github.com/ietf-wg-cellar/flac-test-files`, released under **CC0-1.0**,
//! a public domain dedication, which is compatible with this crate's
//! Apache-2.0 and would raise no licensing objection to carrying it here.
//!
//! It is out of tree for a different reason: it is **310 MB**, of which the
//! subset group alone is 294 MB. `cargo package` ships `tests/`, crates.io
//! caps a published crate at a small fraction of that, and a git repository
//! carrying it would be unusable. So the files stay outside and this gate
//! points at them.
//!
//! # Running it
//!
//! ```text
//! git clone --depth 1 https://github.com/ietf-wg-cellar/flac-test-files
//! DECIBRI_FLAC_CORPUS=<that directory> cargo test --release --test flac_corpus
//! ```
//!
//! `--release` because the corpus is 310 MB of audio and an unoptimised
//! decode of it is minutes rather than seconds. Without the variable set the
//! test prints where to get the corpus and passes, so a checkout with no
//! corpus is not a failing suite: the in-tree gates in `flac_conformance.rs`
//! are the ones that run everywhere.
//!
//! # What is asserted
//!
//! Every file is decoded and its output hashed, and the hash is compared
//! against the MD5 the file carries in its own streaminfo block. That is an
//! oracle shipped inside the data: it cannot share a bug with this decoder,
//! because it was computed by an encoder that never saw this code.
//!
//! Files this crate deliberately rejects are listed in [`EXPECTED`] with the
//! reason, and are asserted to fail with a typed error rather than to decode.
//! A file that is expected to decode and does not, or is expected to fail and
//! decodes, fails this test either way: the table is a claim about behaviour,
//! not a mute list.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use decibri_decode::{
    DecodeError, FlacFrameReader, FlacReader, FlacRecovery, FlacSkipReason, FlacStreamDecoder,
    FlacStreamInfo, Md5Check, StreamSource,
};

/// The environment variable naming the corpus checkout.
const CORPUS_ENV: &str = "DECIBRI_FLAC_CORPUS";

/// What this crate is expected to do with a file it does not decode.
///
/// Every entry is a deliberate decision recorded with its reason, not a
/// suppression. The file name is the corpus's own, relative to the group
/// directory.
struct Expected {
    group: &'static str,
    file: &'static str,
    reason: &'static str,
}

/// The files this crate rejects, and why each one.
///
/// Everything not listed here must decode with a matching MD5, including
/// every one of the `faulty` group whose fault this decoder is entitled to
/// ignore.
const EXPECTED: &[Expected] = &[
    Expected {
        group: "uncommon",
        file: "01 - changing samplerate.flac",
        reason: "the sample rate changes mid-stream, and one AudioBuffer carries one rate",
    },
    Expected {
        group: "uncommon",
        file: "02 - increasing number of channels.flac",
        reason: "the channel count changes mid-stream, and one AudioBuffer carries one layout",
    },
    Expected {
        group: "uncommon",
        file: "03 - decreasing number of channels.flac",
        reason: "the channel count changes mid-stream, and one AudioBuffer carries one layout",
    },
    Expected {
        group: "uncommon",
        file: "04 - changing bitdepth.flac",
        reason: "the bit depth changes mid-stream, so no single scaling is correct for the file",
    },
    Expected {
        group: "uncommon",
        file: "10 - file starting at frame header.flac",
        reason: "a bare frame stream with no fLaC signature, which FlacFrameReader reads and the \
                 container reader is right to refuse",
    },
    Expected {
        group: "uncommon",
        file: "11 - file starting with unparsable data.flac",
        reason: "a bare frame stream behind unparsable bytes, which FlacRecovery reads and the \
                 container reader is right to refuse",
    },
    Expected {
        group: "faulty",
        file: "01 - wrong max blocksize.flac",
        reason: "a frame exceeds the streaminfo maximum block size, which bounds the decode buffer",
    },
    Expected {
        group: "faulty",
        file: "03 - wrong bit depth.flac",
        reason: "the frame bit depth disagrees with streaminfo",
    },
    Expected {
        group: "faulty",
        file: "04 - wrong number of channels.flac",
        reason: "the frame channel count disagrees with streaminfo",
    },
    Expected {
        group: "faulty",
        file: "05 - wrong total number of samples.flac",
        reason: "the stream carries more audio than streaminfo declares",
    },
    Expected {
        group: "faulty",
        file: "06 - missing streaminfo metadata block.flac",
        reason: "no streaminfo block at all",
    },
    Expected {
        group: "faulty",
        file: "07 - other metadata blocks preceding streaminfo metadata block.flac",
        reason: "streaminfo is not the first metadata block",
    },
    Expected {
        group: "faulty",
        file: "08 - blocksize 65536.flac",
        reason: "streaminfo carries a block size outside the 16-65535 range the format allows",
    },
    Expected {
        group: "faulty",
        file: "09 - blocksize 1.flac",
        reason: "streaminfo carries a block size outside the 16-65535 range the format allows",
    },
    Expected {
        group: "faulty",
        file: "11 - incorrect metadata block length.flac",
        reason: "a metadata block length that lands the walk in the middle of a block",
    },
];

/// The rejection this crate is expected to give for `file`, if any.
fn expected_rejection(group: &str, file: &str) -> Option<&'static str> {
    EXPECTED
        .iter()
        .find(|entry| entry.group == group && entry.file == file)
        .map(|entry| entry.reason)
}

/// Every `.flac` file in `directory`, sorted by name so the run is
/// reproducible.
fn flac_files(directory: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("reading {}: {error}", directory.display()))
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "flac")
        })
        .collect();
    files.sort();
    files
}

/// What decoding one file produced.
enum Outcome {
    /// Decoded, with the MD5 checked or absent, and this many samples.
    Decoded(usize),
    /// Rejected with a typed error.
    Rejected(String),
}

/// Decodes `bytes` whole and reports what happened.
fn decode_whole(bytes: &[u8]) -> Outcome {
    match FlacReader::new(bytes).and_then(|reader| reader.decode_to_end()) {
        Ok(decoded) => Outcome::Decoded(decoded.samples().len()),
        Err(error) => Outcome::Rejected(error.to_string()),
    }
}

/// Decodes `bytes` through the streaming reader in `chunk`-byte pieces and
/// reports what happened.
fn decode_streaming(bytes: &[u8], chunk: usize) -> Result<Vec<f32>, DecodeError> {
    let mut stream = FlacStreamDecoder::new();
    let mut samples = Vec::new();
    for piece in bytes.chunks(chunk) {
        let mut offset = 0;
        while offset < piece.len() {
            offset += stream.push(&piece[offset..])?;
            while stream.pull(&mut samples, usize::MAX)? > 0 {}
        }
    }
    stream.finish(&mut samples)?;
    Ok(samples)
}

#[test]
fn the_conformance_corpus_decodes_with_a_matching_md5() {
    let Some(root) = std::env::var_os(CORPUS_ENV) else {
        eprintln!(
            "skipped: set {CORPUS_ENV} to a checkout of \
             github.com/ietf-wg-cellar/flac-test-files and rerun with --release"
        );
        return;
    };
    let root = PathBuf::from(root);

    let mut decoded = 0usize;
    let mut rejected = 0usize;
    let mut failures: Vec<String> = Vec::new();
    let mut per_group: BTreeMap<&str, (usize, usize)> = BTreeMap::new();

    for group in ["subset", "uncommon", "faulty"] {
        let directory = root.join(group);
        assert!(
            directory.is_dir(),
            "{} is not a directory; is {CORPUS_ENV} pointing at the corpus root?",
            directory.display()
        );
        for path in flac_files(&directory) {
            let name = path
                .file_name()
                .expect("a file has a name")
                .to_string_lossy()
                .into_owned();
            let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("reading {name}: {e}"));
            let expectation = expected_rejection(group, &name);

            match (decode_whole(&bytes), expectation) {
                (Outcome::Decoded(count), None) => {
                    decoded += 1;
                    per_group.entry(group).or_default().0 += 1;
                    println!("{group}/{name}: {count} samples, MD5 matched");
                }
                (Outcome::Decoded(_), Some(reason)) => failures.push(format!(
                    "{group}/{name}: decoded, but was expected to be rejected ({reason})"
                )),
                (Outcome::Rejected(error), None) => {
                    failures.push(format!("{group}/{name}: rejected with {error}"));
                }
                (Outcome::Rejected(error), Some(reason)) => {
                    rejected += 1;
                    per_group.entry(group).or_default().1 += 1;
                    println!("{group}/{name}: rejected as expected ({reason}): {error}");
                }
            }

            // The streaming reader must reach the same verdict on the same
            // bytes. Two chunk sizes rather than one because a chunk that
            // happens to align with a frame boundary exercises a different
            // path from one that does not.
            for chunk in [1_021, 65_536] {
                let streamed = decode_streaming(&bytes, chunk);
                match (streamed, expectation) {
                    (Ok(samples), None) => {
                        let whole = FlacReader::new(&bytes)
                            .and_then(|reader| reader.decode_to_end())
                            .expect("the whole-file path already decoded this");
                        if samples != whole.samples() {
                            failures.push(format!(
                                "{group}/{name}: streaming at {chunk} bytes disagreed with the \
                                 whole-file path"
                            ));
                        }
                    }
                    (Ok(_), Some(_)) => failures.push(format!(
                        "{group}/{name}: streaming at {chunk} bytes decoded a file the \
                         whole-file path rejects"
                    )),
                    (Err(_), Some(_)) => {}
                    (Err(error), None) => failures.push(format!(
                        "{group}/{name}: streaming at {chunk} bytes rejected with {error}"
                    )),
                }
            }
        }
    }

    for (group, (ok, no)) in &per_group {
        println!("{group}: {ok} decoded, {no} rejected as expected");
    }
    println!("total: {decoded} decoded with a matching MD5, {rejected} rejected as expected");

    assert!(
        failures.is_empty(),
        "{} corpus file(s) behaved unexpectedly:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert!(
        decoded + rejected > 80,
        "only {} files were seen; is the corpus complete?",
        decoded + rejected
    );
}

// -- The content probe over the whole corpus ----------------------------------

/// The corpus files that carry no `fLaC` signature.
///
/// Both are headerless by construction, and both are already in [`EXPECTED`]
/// as files the container reader is right to refuse. They are named again
/// here because the probe's claim about them is a different one: not "this
/// does not decode" but "this is not identifiable, and guessing is refused".
/// `FlacFrameReader` and `FlacRecovery` read them, when a caller says so.
const HEADERLESS: &[(&str, &str)] = &[
    ("uncommon", "10 - file starting at frame header.flac"),
    ("uncommon", "11 - file starting with unparsable data.flac"),
];

/// Every corpus file through `decode`, reaching the same verdict as
/// `FlacReader` reaches on the same bytes.
///
/// The probe is only a dispatch, so the interesting claim is not that these
/// files decode: `the_conformance_corpus_decodes_with_a_matching_md5` above
/// already establishes that against the MD5 each file carries. The claim here
/// is that routing changes **nothing**, on every file, including the twelve
/// this crate deliberately rejects, whose errors must arrive unaltered rather
/// than reworded by the layer in front of them.
///
/// Both entry points run: the whole-file one against `FlacReader`, and the
/// streaming one against the whole-file one, because a probe that buffered
/// its twelve leading bytes and then failed to hand them on would produce
/// audio that starts thirteen bytes in and is otherwise plausible.
///
/// Two corpus files carry no signature at all, and the probe is right to
/// refuse them: [`HEADERLESS`] names them, and the refusal is asserted rather
/// than excused.
#[test]
fn the_whole_corpus_reaches_the_same_verdict_through_the_probe() {
    use decibri_decode::{decode, identify, AudioStreamDecoder, Container};

    let Some(root) = corpus_root() else { return };

    let mut seen = 0usize;
    let mut rejected = 0usize;
    let mut headerless = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for group in ["subset", "uncommon", "faulty"] {
        let directory = root.join(group);
        assert!(directory.is_dir(), "{} is missing", directory.display());
        for path in flac_files(&directory) {
            let name = path
                .file_name()
                .expect("a file has a name")
                .to_string_lossy()
                .into_owned();
            let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("reading {name}: {e}"));
            seen += 1;

            // Identification succeeds on every corpus file that opens with
            // the signature, which is all of them except the two headerless
            // ones. Those must be refused, naming the bytes actually found:
            // the probe does not guess at a bare frame stream, and this is
            // the only place in the suite where a real file proves it.
            let signed = !HEADERLESS.contains(&(group, name.as_str()));
            match (identify(&bytes), signed) {
                (Ok(Container::Flac), true) => {}
                (Ok(other), true) => {
                    failures.push(format!("{group}/{name}: identified as {other}"))
                }
                (Err(error), true) => {
                    failures.push(format!("{group}/{name}: did not identify: {error}"));
                }
                (Err(DecodeError::UnsupportedContainer { tag }), false) => {
                    headerless += 1;
                    if tag.as_bytes() != &bytes[..4] {
                        failures.push(format!(
                            "{group}/{name}: refused naming '{tag}', not the bytes it opens with"
                        ));
                    }
                }
                (Err(error), false) => failures.push(format!(
                    "{group}/{name}: headerless, but refused as {error} rather than as a container"
                )),
                (Ok(container), false) => failures.push(format!(
                    "{group}/{name}: headerless, but the probe guessed {container}"
                )),
            }

            let direct = FlacReader::new(&bytes).and_then(|reader| reader.decode_to_end());
            let probed = decode(&bytes);
            match (&direct, &probed) {
                (Ok(direct), Ok(probed)) => {
                    if direct.spec() != probed.spec() || direct.samples() != probed.samples() {
                        failures.push(format!("{group}/{name}: the probe decoded differently"));
                    }
                }
                (Err(direct), Err(probed)) => {
                    rejected += 1;
                    // The same rejection, not merely a rejection. A probe
                    // that swallowed the reader's error and reported its own
                    // would pass a weaker test than this.
                    if direct.to_string() != probed.to_string() {
                        failures.push(format!(
                            "{group}/{name}: reader said \"{direct}\", the probe said \"{probed}\""
                        ));
                    }
                }
                (Ok(_), Err(error)) => {
                    failures.push(format!("{group}/{name}: the probe refused it: {error}"));
                }
                (Err(error), Ok(_)) => {
                    failures.push(format!(
                        "{group}/{name}: the probe decoded a file the reader refuses: {error}"
                    ));
                }
            }

            // The streaming probe, at a chunk size that never aligns with the
            // twelve-byte probe length or with a frame boundary.
            let mut stream = AudioStreamDecoder::new();
            let mut samples = Vec::new();
            let streamed = (|| -> Result<(), DecodeError> {
                for piece in bytes.chunks(1_021) {
                    let mut offset = 0;
                    while offset < piece.len() {
                        offset += stream.push(&piece[offset..])?;
                        while stream.pull(&mut samples, usize::MAX)? > 0 {}
                    }
                }
                stream.finish(&mut samples)?;
                Ok(())
            })();
            match (&direct, streamed) {
                (Ok(direct), Ok(())) => {
                    if direct.samples() != samples {
                        failures.push(format!(
                            "{group}/{name}: the streaming probe disagreed with the whole-file path"
                        ));
                    }
                }
                (Err(_), Err(_)) => {}
                (Ok(_), Err(error)) => failures.push(format!(
                    "{group}/{name}: the streaming probe refused it: {error}"
                )),
                (Err(_), Ok(())) => failures.push(format!(
                    "{group}/{name}: the streaming probe decoded a file the reader refuses"
                )),
            }
        }
    }

    assert!(
        failures.is_empty(),
        "{} corpus file(s) routed differently through the probe:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert!(seen > 80, "only {seen} files were seen through the probe");
    assert_eq!(
        headerless,
        HEADERLESS.len(),
        "the corpus's headerless files were not all seen"
    );
    println!(
        "{seen} corpus files routed through the probe, all reaching the reader's own verdict \
         ({rejected} of them a rejection, reported with the reader's own message; {headerless} \
         of them headerless and refused by the probe, naming the bytes they open with)"
    );
}

// -- Bare frame streams, out-of-band streaminfo and recovery ------------------

/// The corpus root, or `None` when the variable is unset and the gate is to
/// print where to get it and pass.
fn corpus_root() -> Option<PathBuf> {
    match std::env::var_os(CORPUS_ENV) {
        Some(root) => Some(PathBuf::from(root)),
        None => {
            eprintln!(
                "skipped: set {CORPUS_ENV} to a checkout of \
                 github.com/ietf-wg-cellar/flac-test-files and rerun with --release"
            );
            None
        }
    }
}

/// SplitMix64, so the garbage prepended below is the same garbage on every
/// machine.
struct SplitMix64(u64);

impl SplitMix64 {
    fn bytes(&mut self, len: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(len + 8);
        while out.len() < len {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            out.extend_from_slice(&(z ^ (z >> 31)).to_le_bytes());
        }
        out.truncate(len);
        out
    }
}

/// `info` with the two fields a partial stream cannot satisfy removed.
///
/// Used only to confirm that a byte offset recovery resynced at really is a
/// frame boundary, by decoding from it through a different reader. That
/// decode holds a suffix of the stream, so the declared total and the
/// whole-stream checksum are both wrong for it by construction.
fn info_for_a_suffix(info: FlacStreamInfo) -> FlacStreamInfo {
    let mut info = info;
    info.total_samples = None;
    info.md5 = None;
    info
}

#[test]
fn the_two_bare_frame_corpus_files_decode() {
    let Some(root) = corpus_root() else { return };
    let directory = root.join("uncommon");

    // The file that is nothing but frames.
    let path = directory.join("10 - file starting at frame header.flac");
    let bytes = std::fs::read(&path).expect("reading uncommon/10");
    let reader = FlacFrameReader::new(&bytes).expect("uncommon/10 opens at a frame boundary");
    let info = *reader.stream_info();
    let (decoded, report) = reader.decode_to_end().expect("uncommon/10 decodes");
    println!(
        "uncommon/10: {} bytes, {} Hz, {} channel(s), {} bits, {} samples, md5 {:?}",
        bytes.len(),
        info.spec.sample_rate,
        info.spec.channels,
        info.bits_per_sample,
        decoded.samples().len(),
        report.md5
    );
    assert_eq!(report.samples, decoded.samples().len());
    // Nothing came with the frames, so nothing checked them, and it says so.
    assert_eq!(report.md5, Md5Check::NoStreamInfo);
    assert!(!decoded.samples().is_empty());

    // The same audio behind bytes that are not frames.
    let path = directory.join("11 - file starting with unparsable data.flac");
    let bytes = std::fs::read(&path).expect("reading uncommon/11");
    let mut samples = Vec::new();
    let report = FlacRecovery::new(&bytes)
        .decode(&mut samples)
        .expect("uncommon/11 recovers");
    println!(
        "uncommon/11: {} bytes, {} Hz, {} channel(s), {} bits, {} samples, {} skip(s), \
         first skip {:?}, lost {:?}",
        bytes.len(),
        report.stream_info.spec.sample_rate,
        report.stream_info.spec.channels,
        report.stream_info.bits_per_sample,
        report.samples,
        report.skipped.len(),
        report.skipped.first().map(|skip| skip.bytes.clone()),
        report.frames_lost()
    );
    assert!(!samples.is_empty());
    let leading = report
        .skipped
        .first()
        .expect("the unparsable head is a skip");
    assert_eq!(leading.bytes.start, 0);
    assert_eq!(leading.reason, FlacSkipReason::NoSyncPoint);

    // Once the junk is behind it, the recovered stream is exactly what the
    // bare frame reader produces from the same offset. That is an oracle for
    // the recovering path that does not come from the recovering path.
    let resync = leading.bytes.end as usize;
    let (direct, _) = FlacFrameReader::new(&bytes[resync..])
        .expect("the resync offset is a frame boundary")
        .decode_to_end()
        .expect("and decodes on its own");
    assert_eq!(direct.samples(), samples);
}

#[test]
fn every_decodable_corpus_file_decodes_the_same_with_its_header_stripped() {
    let Some(root) = corpus_root() else { return };

    let mut checked = 0usize;
    let mut derived_ok = 0usize;
    let mut failures: Vec<String> = Vec::new();
    let mut derived_refusals: Vec<(String, String)> = Vec::new();

    for group in ["subset", "uncommon", "faulty"] {
        for path in flac_files(&root.join(group)) {
            let name = path
                .file_name()
                .expect("a file has a name")
                .to_string_lossy()
                .into_owned();
            if expected_rejection(group, &name).is_some() {
                continue;
            }
            let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("reading {name}: {e}"));
            let reader = FlacReader::new(&bytes).expect("this file already decoded");
            let info = *reader.stream_info();
            let audio = reader.frame_data();
            let full = reader.decode_to_end().expect("this file already decoded");
            checked += 1;

            // With the streaminfo block supplied out of band, which is the
            // Ogg and Matroska shape: identical output, and the checksum is
            // still checked.
            match FlacFrameReader::with_stream_info(audio, info).decode_to_end() {
                Ok((decoded, report)) => {
                    if decoded.samples() != full.samples() {
                        failures.push(format!(
                            "{group}/{name}: supplied-streaminfo output differs"
                        ));
                    }
                    let want = if info.md5.is_some() {
                        Md5Check::Verified
                    } else {
                        Md5Check::ChecksumUnset
                    };
                    if report.md5 != want {
                        failures.push(format!(
                            "{group}/{name}: reported {:?}, expected {want:?}",
                            report.md5
                        ));
                    }
                }
                Err(error) => failures.push(format!(
                    "{group}/{name}: supplied streaminfo rejected: {error}"
                )),
            }

            // With nothing supplied, so every property comes out of the first
            // frame header. A file whose frames defer a field to streaminfo
            // cannot be read this way and must say which field.
            match FlacFrameReader::new(audio).and_then(|reader| reader.decode_to_end()) {
                Ok((decoded, report)) => {
                    derived_ok += 1;
                    if decoded.samples() != full.samples() {
                        failures.push(format!("{group}/{name}: derived-property output differs"));
                    }
                    if report.md5 != Md5Check::NoStreamInfo {
                        failures.push(format!(
                            "{group}/{name}: a derived stream claimed {:?}",
                            report.md5
                        ));
                    }
                }
                Err(error) => derived_refusals.push((format!("{group}/{name}"), error.to_string())),
            }
        }
    }

    println!("{checked} decodable file(s) decoded identically with the header stripped");
    println!("{derived_ok} of them also decoded with every property derived from the first frame");
    for (name, error) in &derived_refusals {
        println!("derived properties refused: {name}: {error}");
    }
    assert!(
        failures.is_empty(),
        "{} file(s) behaved unexpectedly:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert!(checked > 60, "only {checked} files were seen");

    // The two escape codes, on real files rather than on constructed ones.
    // These are the only two places in the corpus where a frame defers a
    // field to streaminfo, and they are exactly the two fields the format
    // allows to be deferred: 768 kHz is outside the header's sample rate
    // codes and 15 bits is outside its bit depth codes, so each encoder had
    // no choice. Both decode when the block is supplied, which the loop above
    // already asserted, and neither is guessed at when it is not.
    assert_eq!(
        derived_refusals.len(),
        ESCAPE_CODED.len(),
        "the files refusing derived properties are not the ones expected: {derived_refusals:?}"
    );
    for (group, file, field) in ESCAPE_CODED {
        let name = format!("{group}/{file}");
        let (_, error) = derived_refusals
            .iter()
            .find(|(seen, _)| *seen == name)
            .unwrap_or_else(|| panic!("{name} was expected to refuse derived properties"));
        assert!(
            error.contains(field),
            "{name}: the rejection must name {field}, and said: {error}"
        );
    }
}

/// The corpus files whose frames defer a field to streaminfo, and which field.
///
/// A short list because the escape codes are only reachable for values the
/// frame header cannot express itself.
const ESCAPE_CODED: &[(&str, &str, &str)] = &[
    (
        "uncommon",
        "06 - samplerate 768kHz.flac",
        "sample rate code 0b0000",
    ),
    (
        "uncommon",
        "07 - 15 bit per sample.flac",
        "bit depth code 0b000",
    ),
];

// -- The recovery sweep -------------------------------------------------------

/// One file in the recovery sweep, and what it is in it for.
struct Selected {
    group: &'static str,
    file: &'static str,
    why: &'static str,
}

/// The sweep's selection.
///
/// Fifteen files rather than all eighty-six, chosen to span the format's
/// features rather than sampled at random, a representative selection whose
/// representativeness is not stated is just a selection. What it covers is
/// asserted at the end of the test from what was actually observed, not from
/// this table: bit depths 8 through 32, one to eight channels, all four
/// channel assignments, both blocking strategies, the Rice escape codes, and
/// the uncommon group.
const SWEEP: &[Selected] = &[
    Selected {
        group: "subset",
        file: "03 - blocksize 16.flac",
        why: "the smallest block size the format allows, so the most frame boundaries per byte",
    },
    Selected {
        group: "subset",
        file: "14 - wasted bits.flac",
        why: "subframes with wasted bits, the first of the five silent-wrongness paths",
    },
    Selected {
        group: "subset",
        file: "15 - only verbatim subframes.flac",
        why: "verbatim subframes, where a frame is nearly its own unencoded size",
    },
    Selected {
        group: "subset",
        file: "16 - partition order 8 containing escaped partitions.flac",
        why: "the Rice escape code, at a high partition order",
    },
    Selected {
        group: "subset",
        file: "22 - 12 bit per sample.flac",
        why: "12-bit audio, a depth that is not a whole number of bytes",
    },
    Selected {
        group: "subset",
        file: "23 - 8 bit per sample.flac",
        why: "8-bit audio, the shallowest depth in the subset",
    },
    Selected {
        group: "subset",
        file: "26 - variable blocksize file created with CUETools.Flake 2.1.6.flac",
        why: "a variable-blocksize stream, where the coded number is a sample number",
    },
    Selected {
        group: "subset",
        file: "27 - old format variable blocksize file created with Flake 0.11.flac",
        why: "a second variable-blocksize encoder, written before the format settled",
    },
    Selected {
        group: "subset",
        file: "38 - 3 channels (3.0).flac",
        why: "three channels, which can only be coded independently",
    },
    Selected {
        group: "subset",
        file: "43 - 8 channels (7.1).flac",
        why: "eight channels, the most the format allows and the widest decode buffer",
    },
    Selected {
        group: "subset",
        file: "60 - mono audio.flac",
        why: "one channel, where no decorrelation happens at all",
    },
    Selected {
        group: "subset",
        file: "63 - predictor overflow check, 24-bit.flac",
        why: "24-bit audio at the top of the losslessness boundary, with extreme predictors",
    },
    Selected {
        group: "subset",
        file: "64 - rice partitions with escape code zero.flac",
        why: "the Rice escape code at width zero, which codes a partition of nothing but zeros",
    },
    Selected {
        group: "uncommon",
        file: "05 - 32bps audio.flac",
        why: "32-bit audio, the deepest the format allows, from the uncommon group",
    },
    Selected {
        group: "uncommon",
        file: "09 - Rice partition order 15.flac",
        why: "the highest Rice partition order, from the uncommon group",
    },
];

/// The seed the sweep's garbage is drawn from.
const SWEEP_SEED: u64 = 0x5EED_F1AC_2ECD_0E11;

/// CRC-8 over RFC 9639 section 9.1.8's frame header polynomial.
fn crc8(bytes: &[u8]) -> u8 {
    let mut crc = 0u8;
    for &byte in bytes {
        crc ^= byte;
        for _ in 0..8 {
            crc = if crc & 0x80 != 0 {
                (crc << 1) ^ 0x07
            } else {
                crc << 1
            };
        }
    }
    crc
}

/// Counts the channel assignments that appear in `audio`, as observed rather
/// than as assumed.
///
/// This is a *scan*, not a decode: it does not know where frames really
/// start, so a position inside a frame body can validate by chance. The
/// counts are used only to say which assignments a file contains, and a
/// chance match needs the 14-bit sync code, valid field values and a matching
/// CRC-8 all at once, so it is roughly one position in 3 x 10^7. Indices are
/// independent, left/side, side/right, mid/side.
fn assignments_seen(audio: &[u8]) -> [usize; 4] {
    let mut counts = [0usize; 4];
    for at in 0..audio.len().saturating_sub(6) {
        if audio[at] != 0xFF || audio[at + 1] & 0xFE != 0xF8 {
            continue;
        }
        let block = audio[at + 2] >> 4;
        let rate = audio[at + 2] & 0x0F;
        let channels = audio[at + 3] >> 4;
        let depth = (audio[at + 3] >> 1) & 0x07;
        if audio[at + 3] & 1 != 0 || block == 0 || rate == 15 || channels > 10 || depth == 3 {
            continue;
        }
        let extra = match audio[at + 4].leading_ones() {
            0 => 0,
            leading @ 2..=7 => leading as usize - 1,
            _ => continue,
        };
        let mut length = 5 + extra;
        length += match block {
            6 => 1,
            7 => 2,
            _ => 0,
        };
        length += match rate {
            12 => 1,
            13 | 14 => 2,
            _ => 0,
        };
        if at + length >= audio.len() || crc8(&audio[at..at + length]) != audio[at + length] {
            continue;
        }
        counts[match channels {
            8 => 1,
            9 => 2,
            10 => 3,
            _ => 0,
        }] += 1;
    }
    counts
}

/// How many times an assignment has to appear before the sweep counts the
/// file as containing it, which is far above what chance produces.
const ASSIGNMENT_FLOOR: usize = 8;

#[test]
fn recovery_across_a_selection_spanning_the_formats_features() {
    let Some(root) = corpus_root() else { return };
    let mut random = SplitMix64(SWEEP_SEED);

    let mut depths: Vec<u8> = Vec::new();
    let mut channel_counts: Vec<u16> = Vec::new();
    let mut assignments = [0usize; 4];
    let mut variable_blocksize = 0usize;

    for entry in SWEEP {
        let path = root.join(entry.group).join(entry.file);
        let bytes = std::fs::read(&path)
            .unwrap_or_else(|error| panic!("reading {}/{}: {error}", entry.group, entry.file));
        let name = format!("{}/{}", entry.group, entry.file);
        println!("selected {name}: {}", entry.why);

        let reader = FlacReader::new(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        let info = *reader.stream_info();
        let audio = reader.frame_data();
        // The ground truth: a decode this crate already checked against the
        // MD5 the file carries, which no part of this crate computed.
        let full = reader
            .decode_to_end()
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(
            info.md5.is_some(),
            "{name}: the sweep needs a file whose own checksum anchors it"
        );
        let channels = usize::from(info.spec.channels);
        let bare = info_for_a_suffix(info);

        if !depths.contains(&info.bits_per_sample) {
            depths.push(info.bits_per_sample);
        }
        if !channel_counts.contains(&info.spec.channels) {
            channel_counts.push(info.spec.channels);
        }
        let seen = assignments_seen(audio);
        for (total, count) in assignments.iter_mut().zip(seen) {
            if count >= ASSIGNMENT_FLOOR {
                *total += 1;
            }
        }
        if info.min_block_size != info.max_block_size {
            variable_blocksize += 1;
        }

        // -- Case 1: the header stripped ----------------------------------
        let (stripped, report) = FlacFrameReader::with_stream_info(audio, info)
            .decode_to_end()
            .unwrap_or_else(|e| panic!("{name}: header stripped: {e}"));
        assert_eq!(
            stripped.samples(),
            full.samples(),
            "{name}: header stripped"
        );
        assert_eq!(report.md5, Md5Check::Verified, "{name}: header stripped");

        // -- Case 2: garbage prepended ------------------------------------
        // The junk opens with this file's own first sixteen bytes, so it
        // begins with a frame header that parses and whose CRC-8 matches,
        // followed by bytes that are not that frame. A recogniser that
        // accepted a lone validating header would lock on at offset zero and
        // report the gap differently; chained validation walks past it.
        //
        // This shape was added because a negative control on the chained
        // validation left this sweep green while four in-tree tests went red:
        // real files do not naturally put a decoy header in front of their
        // audio, so the sweep was blind to the check it most depends on.
        let mut junk = audio[..16].to_vec();
        junk.extend_from_slice(&random.bytes(984));
        let mut damaged = junk.clone();
        damaged.extend_from_slice(audio);
        let (recovered, report) = FlacRecovery::with_stream_info(&damaged, info)
            .decode_to_end()
            .unwrap_or_else(|e| panic!("{name}: garbage prepended: {e}"));
        assert_eq!(recovered.samples(), full.samples(), "{name}: prepended");
        assert_eq!(report.skipped.len(), 1, "{name}: prepended");
        assert_eq!(report.skipped[0].bytes, 0..1_000, "{name}: prepended");
        assert_eq!(report.skipped[0].frames, Some(0..0), "{name}: prepended");
        assert_eq!(report.frames_lost(), Some(0), "{name}: prepended");
        assert_eq!(report.first_frame, Some(0), "{name}: prepended");
        // Nothing was lost, so the checksum still covers what came out, and
        // a recovered decode is allowed to be as strong as an ordinary one.
        assert_eq!(report.md5, Md5Check::Verified, "{name}: prepended");

        // -- Case 3: cut part-way through a frame -------------------------
        let cut = audio.len() / 3;
        let tail = &audio[cut..];
        let (recovered, report) = FlacRecovery::with_stream_info(tail, info)
            .decode_to_end()
            .unwrap_or_else(|e| panic!("{name}: cut at {cut}: {e}"));
        // Measured against the ground truth, not against the report: the
        // output has to be exactly the end of the undamaged decode.
        let kept = recovered.samples().len();
        assert_eq!(
            recovered.samples(),
            &full.samples()[full.samples().len() - kept..],
            "{name}: cut is not a suffix of the whole decode"
        );
        let lost = (full.samples().len() - kept) / channels;
        assert_eq!(report.skipped.len(), 1, "{name}: cut");
        let skip = &report.skipped[0];
        assert_eq!(skip.bytes.start, 0, "{name}: cut");
        assert_eq!(skip.reason, FlacSkipReason::NoSyncPoint, "{name}: cut");
        assert_eq!(
            skip.frames,
            Some(0..lost as u64),
            "{name}: the reported loss must be the measured loss"
        );
        assert_eq!(report.frames_lost(), Some(lost as u64), "{name}: cut");
        assert_eq!(report.first_frame, Some(lost as u64), "{name}: cut");
        assert_eq!(report.md5, Md5Check::AudioIncomplete, "{name}: cut");
        // The offset it resumed at is a real frame boundary, confirmed by a
        // reader that does no searching at all.
        let resync = skip.bytes.end as usize;
        if let Some(largest) = info.max_frame_size {
            assert!(
                resync <= largest as usize,
                "{name}: resync skipped {resync} bytes, more than the largest frame {largest}"
            );
        }
        let (direct, _) = FlacFrameReader::with_stream_info(&tail[resync..], bare)
            .decode_to_end()
            .unwrap_or_else(|e| panic!("{name}: resync offset {resync} is not a boundary: {e}"));
        assert_eq!(direct.samples(), recovered.samples(), "{name}: cut resync");

        // -- Case 4: a frame body corrupted -------------------------------
        let mut damaged = audio.to_vec();
        let flipped = damaged.len() / 2;
        damaged[flipped] ^= 0x5A;
        let (recovered, report) = FlacRecovery::with_stream_info(&damaged, info)
            .decode_to_end()
            .unwrap_or_else(|e| panic!("{name}: corrupted at {flipped}: {e}"));
        let missing = full.samples().len() - recovered.samples().len();
        assert!(missing > 0, "{name}: a flipped bit produced no loss at all");
        assert_eq!(report.skipped.len(), 1, "{name}: corrupted");
        let skip = &report.skipped[0];
        assert_eq!(
            skip.reason,
            FlacSkipReason::FrameRejected,
            "{name}: corrupted"
        );
        assert!(
            skip.bytes.contains(&(flipped as u64)),
            "{name}: the flipped byte at {flipped} is outside the reported skip {:?}",
            skip.bytes
        );

        // Where the gap sits in the output is established by decoding from
        // the offset recovery resumed at, through a reader that does no
        // searching at all, not by looking for where the samples stop
        // matching. Several of these files hold digital silence, so a common
        // prefix runs past the real boundary and would put the gap in the
        // wrong place.
        let resync = skip.bytes.end as usize;
        let (direct, _) = FlacFrameReader::with_stream_info(&damaged[resync..], bare)
            .decode_to_end()
            .unwrap_or_else(|e| panic!("{name}: resync offset {resync} is not a boundary: {e}"));
        let tail = direct.samples().len();
        assert!(
            tail <= recovered.samples().len(),
            "{name}: corrupted resync"
        );
        let head = recovered.samples().len() - tail;
        assert_eq!(
            &recovered.samples()[head..],
            direct.samples(),
            "{name}: corrupted resync"
        );
        // Both halves are the undamaged decode's own samples, in place.
        assert_eq!(
            &recovered.samples()[..head],
            &full.samples()[..head],
            "{name}: the recovered head is not the undamaged head"
        );
        assert_eq!(
            &recovered.samples()[head..],
            &full.samples()[full.samples().len() - tail..],
            "{name}: the recovered tail is not the undamaged tail"
        );
        assert_eq!(
            skip.frames,
            Some((head / channels) as u64..((head + missing) / channels) as u64),
            "{name}: the reported gap must be the measured gap"
        );
        assert_eq!(
            report.frames_lost(),
            Some((missing / channels) as u64),
            "{name}: corrupted"
        );
        assert_eq!(report.md5, Md5Check::AudioIncomplete, "{name}: corrupted");

        println!(
            "{name}: {} Hz, {} channel(s), {} bits, blocksize {}-{}, {} samples; \
             cut at {cut} lost {lost} sample(s) and resynced {} byte(s) in; \
             flip at {flipped} lost {} sample(s) from position {}",
            info.spec.sample_rate,
            info.spec.channels,
            info.bits_per_sample,
            info.min_block_size,
            info.max_block_size,
            full.samples().len() / channels,
            resync,
            missing / channels,
            head / channels,
        );
    }

    depths.sort_unstable();
    channel_counts.sort_unstable();
    println!("bit depths covered: {depths:?}");
    println!("channel counts covered: {channel_counts:?}");
    println!(
        "files containing each channel assignment (independent, left/side, side/right, \
         mid/side): {assignments:?}"
    );
    println!("variable-blocksize files in the selection: {variable_blocksize}");

    // The coverage claim, asserted from what was observed rather than from
    // the table above.
    for depth in [8u8, 12, 16, 24, 32] {
        assert!(depths.contains(&depth), "no {depth}-bit file in the sweep");
    }
    for count in [1u16, 2, 3, 8] {
        assert!(
            channel_counts.contains(&count),
            "no {count}-channel file in the sweep"
        );
    }
    for (index, files) in assignments.iter().enumerate() {
        assert!(*files > 0, "no file in the sweep uses assignment {index}");
    }
    assert!(
        variable_blocksize > 0,
        "no variable-blocksize file in the sweep"
    );
    assert!(
        SWEEP.len() >= 12,
        "the sweep must span at least twelve files"
    );
}

// -- The writer over the whole corpus ------------------------------------------

/// Every decodable corpus file decoded, re-encoded by [`FlacWriter`] at the
/// default level, and decoded again to identical samples.
///
/// The second decode is not a formality: it verifies the streaminfo MD5 the
/// writer computed over its own quantised integers, so a writer that
/// scrambled the audio and a writer that mis-hashed it both fail here. What
/// this gate deliberately cannot establish is that the *bytes* are right
/// rather than merely self-consistent: encoder and decoder share the
/// prediction arithmetic on purpose, so a shared break round-trips green.
/// The independent gates for that are the reference tool runs recorded in
/// the pass report, and the in-tree witness hashes in
/// `flac_write_conformance.rs`, which pin the exact bytes.
///
/// Set `DECIBRI_FLAC_REENCODE_DIR` to also write each re-encoded stream to
/// that directory, which is how the reference tool verification gets its
/// input.
#[test]
fn every_decodable_corpus_file_reencodes_and_round_trips() {
    use decibri_decode::FlacWriter;

    let Some(root) = corpus_root() else { return };
    let dump = std::env::var_os("DECIBRI_FLAC_REENCODE_DIR").map(PathBuf::from);
    if let Some(directory) = &dump {
        std::fs::create_dir_all(directory).expect("creating the re-encode dump directory");
    }

    let mut checked = 0usize;
    let mut original_bytes = 0u64;
    let mut reencoded_bytes = 0u64;
    let mut failures: Vec<String> = Vec::new();

    for group in ["subset", "uncommon", "faulty"] {
        for path in flac_files(&root.join(group)) {
            let name = path
                .file_name()
                .expect("a file has a name")
                .to_string_lossy()
                .into_owned();
            if expected_rejection(group, &name).is_some() {
                continue;
            }
            let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("reading {name}: {e}"));
            let reader = FlacReader::new(&bytes).expect("this file already decoded");
            let info = *reader.stream_info();
            let full = reader.decode_to_end().expect("this file already decoded");

            let writer = FlacWriter::new(info.spec, info.bits_per_sample);
            let encoded = match writer.to_bytes(full.samples()) {
                Ok(encoded) => encoded,
                Err(error) => {
                    failures.push(format!("{group}/{name}: encoding failed: {error}"));
                    continue;
                }
            };
            match FlacReader::new(&encoded).and_then(|again| again.decode_to_end()) {
                Ok(decoded) if decoded.samples() == full.samples() => {
                    checked += 1;
                    original_bytes += bytes.len() as u64;
                    reencoded_bytes += encoded.len() as u64;
                }
                Ok(_) => failures.push(format!("{group}/{name}: re-encoded samples differ")),
                Err(error) => {
                    failures.push(format!("{group}/{name}: re-encoded decode failed: {error}"))
                }
            }
            if let Some(directory) = &dump {
                std::fs::write(directory.join(format!("{group} - {name}")), &encoded)
                    .unwrap_or_else(|e| panic!("dumping {name}: {e}"));
            }
        }
    }

    println!(
        "{checked} file(s) re-encoded and round-tripped; {original_bytes} bytes in, \
         {reencoded_bytes} bytes out"
    );
    assert!(
        failures.is_empty(),
        "{} file(s) failed the re-encode round trip:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert_eq!(checked, 71, "every decodable corpus file must round-trip");
}
