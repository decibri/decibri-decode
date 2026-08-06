//! Coverage-guided fuzzing of [`WavStreamDecoder`], the RIFF/WAVE and RF64
//! reader for a stream that arrives in pieces.
//!
//! A streaming decoder has two input dimensions, the bytes and the boundaries
//! they arrive on, and both are under the mutator's control here. Chunk
//! boundaries matter more in this container than in most: a `fmt ` chunk cut
//! across two pushes has to be held and resumed rather than reparsed.
//!
//! # How the input is read
//!
//! The first byte is how many chunk sizes follow, one through eight. Each of
//! those bytes is one chunk size, with zero meaning "everything remaining".
//! Everything after them is the stream, and the sizes cycle over it.
//!
//! A typed error is correct behaviour here and is the common case, so nothing
//! is asserted about the result. A panic, a hang or a sanitizer report is the
//! finding.

#![no_main]

use decibri_decode::WavStreamDecoder;
use libfuzzer_sys::fuzz_target;

mod stream_driver;

fuzz_target!(|data: &[u8]| {
    let Some((count, rest)) = data.split_first() else {
        return;
    };
    let taken = usize::min(1 + usize::from(count % 8), rest.len());
    let (schedule, body) = rest.split_at(taken);
    let chunks: Vec<usize> = schedule
        .iter()
        .map(|&size| {
            if size == 0 {
                usize::MAX
            } else {
                usize::from(size)
            }
        })
        .collect();

    let mut decoder = WavStreamDecoder::new();
    stream_driver::drive(&mut decoder, body, &chunks);
    let _ = decoder.format();
    let _ = decoder.ready_samples();
});
