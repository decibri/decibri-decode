//! Coverage-guided fuzzing of [`decode`] and [`identify`], the front door
//! that reads twelve bytes and hands the input to a reader.
//!
//! The input is a file, so the bytes go straight in with no structure imposed
//! on them: a container is decided by its signature and its form type, and a
//! wrapper that reserved bytes for a selector would make some of those
//! signatures unreachable.
//!
//! A typed error is correct behaviour here and is the common case, so nothing
//! is asserted about the result. A panic, a hang or a sanitizer report is the
//! finding.

#![no_main]

use decibri_decode::{decode, identify};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Both, because they read the same twelve bytes and then diverge:
    // `identify` stops there and `decode` carries on into the reader the
    // twelve bytes chose.
    let _ = identify(data);
    let _ = decode(data);
});
