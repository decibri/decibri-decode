//! Coverage-guided fuzzing of [`FlacRecovery`], the crate's damaged-stream
//! reader.
//!
//! Recovery scans for frame syncs and trial-decodes every plausible one, so it
//! is the entry point that does the most work on the least trustworthy bytes.
//!
//! A typed error is correct behaviour here and is the common case, so nothing
//! is asserted about the result. A panic, a hang or a sanitizer report is the
//! finding.

#![no_main]

use std::sync::OnceLock;

use decibri_decode::{AudioSpec, FlacReader, FlacRecovery, FlacStreamInfo, FlacWriter};
use libfuzzer_sys::fuzz_target;

/// A streaminfo block from a writer-produced file, so the out-of-band
/// constructor is exercised on the same inputs as the self-describing one.
fn info() -> &'static FlacStreamInfo {
    static INFO: OnceLock<FlacStreamInfo> = OnceLock::new();
    INFO.get_or_init(|| {
        let ramp: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 32.0).collect();
        let file = FlacWriter::new(AudioSpec::mono(32_000), 8)
            .to_bytes(&ramp)
            .expect("a one-block mono FLAC writes");
        *FlacReader::new(&file)
            .expect("the writer's own output reads")
            .stream_info()
    })
}

fuzz_target!(|data: &[u8]| {
    // The first byte selects the constructor; everything after it is the
    // stream. Selecting inside the input rather than from a separate source
    // keeps the choice under the mutator's control.
    let Some((select, body)) = data.split_first() else {
        return;
    };
    let mut out = Vec::new();
    if select % 3 == 2 {
        let _ = FlacRecovery::with_stream_info(body, *info()).decode(&mut out);
    } else {
        let _ = FlacRecovery::new(body).decode(&mut out);
    }
});
