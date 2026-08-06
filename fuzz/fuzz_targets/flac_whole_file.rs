//! Coverage-guided fuzzing of [`FlacReader`], the whole-file FLAC reader.
//!
//! The input is a file, so the bytes go straight in. Everything the reader
//! learns about the stream comes from the metadata blocks in front of it, and
//! a wrapper that reserved bytes for a selector would put the `fLaC`
//! signature at an offset no real file has.
//!
//! A typed error is correct behaviour here and is the common case, so nothing
//! is asserted about the result. A panic, a hang or a sanitizer report is the
//! finding.

#![no_main]

use decibri_decode::FlacReader;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(reader) = FlacReader::new(data) else {
        return;
    };
    // The accessors first: they are what a caller reads before deciding
    // whether to decode, and a stream whose metadata parsed can still hold
    // fields no frame agrees with.
    let _ = reader.stream_info();
    let _ = reader.spec();
    let _ = reader.frames();
    let _ = reader.frame_data();

    // Both decode entry points, because they differ in more than their
    // return type: one appends into a caller's buffer and the other sizes
    // and fills its own.
    let mut out = Vec::new();
    let _ = reader.decode(&mut out);
    let _ = reader.decode_to_end();
});
