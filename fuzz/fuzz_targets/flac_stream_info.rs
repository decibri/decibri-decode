//! Coverage-guided fuzzing of [`FlacStreamInfo::from_block`], the
//! out-of-band streaminfo constructor, and of the readers that take what it
//! returns.
//!
//! The input is read as a streaminfo body followed by bare frames, which is
//! the shape a container hands over: Matroska's `CodecPrivate` then its
//! packets, Ogg's header packet then its packets. The split is at 34 bytes
//! because the format fixes a streaminfo body at 34 bytes, so it is the file
//! layout rather than a wrapper, and every byte still reaches a parser.
//!
//! This is the one path where a fuzzer-chosen streaminfo steers the frame
//! decoder: the bit depth, channel count, sample rate and maximum block size
//! a frame is checked against all come from these 34 bytes rather than from
//! the frame itself.
//!
//! A typed error is correct behaviour here and is the common case, so nothing
//! is asserted about the result. A panic, a hang or a sanitizer report is the
//! finding.

#![no_main]

use decibri_decode::{FlacFrameReader, FlacStreamDecoder, FlacStreamInfo};
use libfuzzer_sys::fuzz_target;

mod stream_driver;

/// RFC 9639 section 8.2's streaminfo block body length.
const STREAMINFO_BYTES: usize = 34;

/// The chunk sizes the streaming half cycles through: the smallest possible,
/// two sizes that divide no frame header field, and the whole remainder.
const CHUNKS: [usize; 4] = [1, 7, 64, usize::MAX];

fuzz_target!(|data: &[u8]| {
    let (block, frames) = data.split_at(data.len().min(STREAMINFO_BYTES));
    let Ok(info) = FlacStreamInfo::from_block(block) else {
        return;
    };
    let mut out = Vec::new();
    let _ = FlacFrameReader::with_stream_info(frames, info).decode(&mut out);
    stream_driver::drive(
        &mut FlacStreamDecoder::frames_with_stream_info(info),
        frames,
        &CHUNKS,
    );
});
