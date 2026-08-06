//! The loop a caller of [`StreamSource`] writes, shared by every target that
//! drives one.
//!
//! One file rather than one copy per target, so the boundary handling, the
//! stall guard and the terminal-state handling are identical across the four
//! targets that use it. Four copies would let a target look green because its
//! own copy gave up a push earlier than the others.
//!
//! This is a module, not a fuzz target: it carries no `fuzz_target!` and has
//! no entry in `Cargo.toml`'s `[[bin]]` table.

use decibri_decode::StreamSource;

/// Drives `source` over `data`, cut into pieces whose sizes cycle through
/// `chunks`, the way [`StreamSource`] documents.
///
/// A zero-length `chunks` and a zero entry in it both mean "everything
/// remaining", so no schedule a mutator can produce is unusable.
///
/// Errors are the expected currency here and are discarded: a typed error on
/// malformed input is correct behaviour. The one thing this asserts is
/// progress. A live stream that takes no bytes and produces no samples leaves
/// a caller following the documented loop with nothing to do but offer the
/// same bytes again, which is a hang rather than a rejection, and a hang is a
/// finding.
pub fn drive(source: &mut dyn StreamSource, data: &[u8], chunks: &[usize]) {
    let mut out = Vec::new();
    let mut offset = 0;
    let mut step = 0usize;
    'stream: while offset < data.len() {
        let chunk = chunks
            .get(step % chunks.len().max(1))
            .copied()
            .unwrap_or(usize::MAX)
            .max(1);
        step = step.wrapping_add(1);
        let end = offset.saturating_add(chunk).min(data.len());
        let piece = &data[offset..end];
        let mut taken_total = 0;
        let mut stalls = 0u32;
        while taken_total < piece.len() {
            match source.push(&piece[taken_total..]) {
                Ok(0) => match source.pull(&mut out, usize::MAX) {
                    // Nothing taken and nothing ready is the documented end
                    // state of a failed stream; anything else pushing on
                    // would loop forever.
                    Ok(0) => {
                        stalls += 1;
                        assert!(
                            stalls <= 2,
                            "a live stream took no bytes and produced no samples"
                        );
                    }
                    Ok(_) => stalls = 0,
                    Err(_) => break 'stream,
                },
                Ok(taken) => {
                    taken_total += taken;
                    stalls = 0;
                    if source.pull(&mut out, usize::MAX).is_err() {
                        break 'stream;
                    }
                }
                Err(_) => break 'stream,
            }
            // Delivered samples are dropped as they arrive, so peak memory
            // tracks one pull rather than the whole decode and an output
            // disproportionate to the input still shows up as one.
            out.clear();
        }
        offset = end;
    }
    // Reached after an error too, because `finish` is callable at any point
    // and the state it meets after a rejection is the state least often
    // reached by a test.
    let _ = source.finish(&mut out);
    let _ = source.pull(&mut out, usize::MAX);
    let _ = source.spec();
    let _ = source.buffered_bytes();
    source.reset();
}
