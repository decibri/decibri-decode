//! Coverage-guided fuzzing of the bare FLAC frame readers,
//! [`FlacFrameReader`] and [`FlacStreamDecoder::frames`].
//!
//! A frame stream has no signature and no metadata, so every byte of it is
//! either a frame header or frame content. The bytes go straight in: this is
//! the entry point with the least external structure in the crate, and a
//! wrapper that reserved bytes for a selector would make some header
//! encodings unreachable.
//!
//! The chunk schedule the streaming half is driven on is fixed rather than
//! taken from the input, for the same reason: a schedule prefix would be
//! bytes the frame parser never sees.
//!
//! A typed error is correct behaviour here and is the common case, so nothing
//! is asserted about the result. A panic, a hang or a sanitizer report is the
//! finding.

#![no_main]

use decibri_decode::{FlacFrameReader, FlacStreamDecoder};
use libfuzzer_sys::fuzz_target;

mod stream_driver;

/// The chunk sizes the streaming half cycles through: the smallest possible,
/// two sizes that divide no frame header field, and the whole remainder.
const CHUNKS: [usize; 4] = [1, 7, 64, usize::MAX];

fuzz_target!(|data: &[u8]| {
    let mut out = Vec::new();
    if let Ok(reader) = FlacFrameReader::new(data) {
        let _ = reader.stream_info();
        let _ = reader.spec();
        let _ = reader.frames();
        let _ = reader.decode(&mut out);
        out.clear();
        let _ = reader.decode_to_end();
    }
    stream_driver::drive(&mut FlacStreamDecoder::frames(), data, &CHUNKS);
});
