#![forbid(unsafe_code)]
//! Bare FLAC frame streams, out-of-band streaminfo, and recovery.
//!
//! # The oracle
//!
//! Everything here is anchored on **RFC 9639 appendix D.3's worked example**,
//! a 73-byte FLAC file the specification prints in full together with the 24
//! sample values it decodes to. It is real published reference data rather
//! than something this crate generated, so a break in the decoder cannot move
//! it. The corpus gates in `flac_corpus.rs` carry the same properties across
//! 86 real files with their own MD5 checksums; this binary is what runs on a
//! checkout with no corpus.
//!
//! The multi-frame fixtures below are that one frame repeated with its coded
//! number rewritten and both CRCs recomputed. That is synthetic in exactly
//! one respect, recorded here rather than left to be discovered: **every
//! frame carries identical audio**, which real audio never does. It is used
//! only where the property under test is about frame *boundaries* (where a
//! gap starts, which bytes were skipped, how many samples were lost) and
//! never to establish that a sample value is right. Where a value has to be
//! right, the reference is D.3's own numbers or the corpus's MD5.
//!
//! # What is gated here
//!
//! - A bare frame stream decodes to the same samples as the file it came out
//!   of, whole and in pieces.
//! - The two frame header codes that defer a field to streaminfo are a typed
//!   error naming the field when there is no streaminfo, and are resolved
//!   when the caller supplies one.
//! - Whether the MD5 was checked is reported, and reported honestly.
//! - Ten million runs of random bytes through the recovery entry point emit
//!   nothing. This is the gate that says chained sync validation works, and
//!   it is the cheapest decisive test in the crate.
//! - Recovery reports the byte ranges it skipped and the sample positions it
//!   lost, checked against a decode of the undamaged stream.
//! - Malformed input is a typed error, never a panic and never a hang.

use decibri_decode::{
    AudioBuffer, AudioSpec, DecodeError, FlacFrameReader, FlacReader, FlacRecovery, FlacSkipReason,
    FlacStreamDecoder, FlacStreamInfo, Md5Check, StreamSource,
};

// -- RFC 9639 appendix D.3 ----------------------------------------------------

/// The specification's worked example in full: 8-bit mono at 32 kHz, 24
/// samples, one frame.
const D3_FILE: [u8; 73] = [
    0x66, 0x4c, 0x61, 0x43, 0x80, 0x00, 0x00, 0x22, 0x10, 0x00, 0x10, 0x00, 0x00, 0x00, 0x1f, 0x00,
    0x00, 0x1f, 0x07, 0xd0, 0x00, 0x70, 0x00, 0x00, 0x00, 0x18, 0xf8, 0xf9, 0xe3, 0x96, 0xf5, 0xcb,
    0xcf, 0xc6, 0xdc, 0x80, 0x7f, 0x99, 0x77, 0x90, 0x6b, 0x32, 0xff, 0xf8, 0x68, 0x02, 0x00, 0x17,
    0xe9, 0x44, 0x00, 0x4f, 0x6f, 0x31, 0x3d, 0x10, 0x47, 0xd2, 0x27, 0xcb, 0x6d, 0x09, 0x08, 0x31,
    0x45, 0x2b, 0xdc, 0x28, 0x22, 0x22, 0x80, 0x57, 0xa3,
];

/// Where the single audio frame starts: four signature bytes, a four-byte
/// metadata block header and a 34-byte streaminfo body.
const D3_AUDIO: usize = 42;

/// How many bytes of that frame are header, including its CRC-8.
///
/// Sync, blocking strategy, the four coded fields, a one-byte coded number, a
/// one-byte block size (the header codes 0b0110, "the real value follows"),
/// then the CRC-8.
const D3_HEADER_BYTES: usize = 7;

/// The frame on its own, which is what a container handing over bare frames
/// would deliver.
fn d3_frame() -> Vec<u8> {
    D3_FILE[D3_AUDIO..].to_vec()
}

/// What the whole file decodes to, through the ordinary reader.
fn d3_reference() -> AudioBuffer {
    FlacReader::new(&D3_FILE)
        .expect("D.3 parses")
        .decode_to_end()
        .expect("D.3 decodes with a matching MD5")
}

/// The streaminfo block D.3 carries, as a caller supplying it out of band
/// would hold it.
fn d3_stream_info() -> FlacStreamInfo {
    *FlacReader::new(&D3_FILE).expect("D.3 parses").stream_info()
}

// -- Constructing frames the corpus does not contain --------------------------

/// CRC-8 over `x^8 + x^2 + x^1 + x^0`, RFC 9639 section 9.1.8's frame header
/// polynomial.
///
/// Written out here rather than reached for inside the crate, so that a
/// rewritten header is only accepted because the decoder agreed with an
/// independently computed check value.
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

/// CRC-16 over `x^16 + x^15 + x^2 + x^0`, RFC 9639 section 9.3's whole-frame
/// polynomial.
fn crc16(bytes: &[u8]) -> u16 {
    let mut crc = 0u16;
    for &byte in bytes {
        crc ^= u16::from(byte) << 8;
        for _ in 0..8 {
            crc = if crc & 0x8000 != 0 {
                (crc << 1) ^ 0x8005
            } else {
                crc << 1
            };
        }
    }
    crc
}

/// Restores both CRCs of a frame whose header bytes have been rewritten.
///
/// If either of these were wrong the decoder would reject the frame and every
/// test using it would fail, so the rewrite cannot quietly produce something
/// that is not a FLAC frame.
fn reseal(frame: &mut [u8]) {
    frame[D3_HEADER_BYTES - 1] = crc8(&frame[..D3_HEADER_BYTES - 1]);
    let body = frame.len() - 2;
    let sum = crc16(&frame[..body]).to_be_bytes();
    frame[body..].copy_from_slice(&sum);
}

/// D.3's frame with its sample rate and bit depth codes replaced by the two
/// escape codes that mean "take this field from streaminfo".
///
/// `rate` and `depth` say which of the two to escape. Both fields sit in the
/// third and fourth header bytes: the rate is the low nibble of byte 2 and
/// the depth is bits 3-1 of byte 3. D.3 codes rate 0b1000 (32 kHz) and depth
/// 0b001 (8 bits), and both escape codes are all-zero, so neither rewrite
/// changes the header's length.
fn d3_frame_escaping(rate: bool, depth: bool) -> Vec<u8> {
    let mut frame = d3_frame();
    if rate {
        frame[2] &= 0xF0;
    }
    if depth {
        frame[3] &= 0xF1;
    }
    reseal(&mut frame);
    frame
}

/// D.3's frame carrying frame number `number` instead of zero.
///
/// The coded number is the fifth header byte and stays one octet for every
/// value below 128, so the header's length is unchanged.
fn d3_frame_numbered(number: u8) -> Vec<u8> {
    assert!(number < 0x80, "a longer coded number would move the header");
    let mut frame = d3_frame();
    frame[4] = number;
    reseal(&mut frame);
    frame
}

/// `count` correctly numbered frames, which is a bare stream that a decoder
/// can tell one frame of from another.
fn d3_stream(count: u8) -> Vec<u8> {
    (0..count).flat_map(d3_frame_numbered).collect()
}

/// What `d3_stream(count)` decodes to.
fn d3_stream_reference(count: u8) -> Vec<f32> {
    let one = d3_reference();
    let mut all = Vec::new();
    for _ in 0..count {
        all.extend_from_slice(one.samples());
    }
    all
}

// -- Bare frame streams -------------------------------------------------------

#[test]
fn a_bare_frame_stream_decodes_to_the_same_samples_as_its_file() {
    let reference = d3_reference();
    let frames = d3_frame();

    let reader = FlacFrameReader::new(&frames).expect("the first frame header parses");
    assert_eq!(reader.spec(), reference.spec());
    assert_eq!(reader.stream_info().bits_per_sample, 8);
    // Nothing declared a length, so there is none to report. `None` is not
    // zero: an unknown length and an empty stream are different things.
    assert_eq!(reader.frames(), None);

    let mut samples = Vec::new();
    let report = reader.decode(&mut samples).expect("bare frames decode");
    assert_eq!(samples, reference.samples());
    assert_eq!(report.samples, 24);
}

#[test]
fn a_bare_stream_says_nothing_verified_it() {
    let frames = d3_frame();
    let report = FlacFrameReader::new(&frames)
        .expect("parses")
        .decode(&mut Vec::new())
        .expect("decodes");
    assert_eq!(report.md5, Md5Check::NoStreamInfo);
    assert!(!report.md5.is_verified());
}

#[test]
fn a_bare_stream_derives_the_formats_own_ceiling_rather_than_a_promise() {
    let frames = d3_frame();
    let info = *FlacFrameReader::new(&frames).expect("parses").stream_info();
    // D.3's real streaminfo says 4096. A derived description must not borrow
    // that number from the first frame it happened to see, because a later
    // frame may legitimately be larger: the only honest bound is the
    // format's.
    assert_eq!(info.max_block_size, 65_535);
    assert_eq!(info.total_samples, None);
    assert_eq!(info.md5, None);
    assert_eq!(info.min_frame_size, None);
    assert_eq!(info.max_frame_size, None);
}

/// The streaming reader driven to exhaustion over `bytes` in `chunk`-byte
/// pieces.
fn stream_all(mut reader: FlacStreamDecoder, bytes: &[u8], chunk: usize) -> Vec<f32> {
    let mut samples = Vec::new();
    for piece in bytes.chunks(chunk) {
        let mut offset = 0;
        while offset < piece.len() {
            offset += reader.push(&piece[offset..]).expect("push");
            while reader.pull(&mut samples, usize::MAX).expect("pull") > 0 {}
        }
    }
    reader.finish(&mut samples).expect("finish");
    samples
}

#[test]
fn a_bare_stream_arriving_in_pieces_matches_the_whole_one() {
    let frames = d3_stream(4);
    let reference = d3_stream_reference(4);
    // 1 splits every header; 7 lands on D.3's header boundary; 31 is exactly
    // one frame; 4096 is the whole thing in one piece.
    for chunk in [1, 2, 3, 7, 13, 31, 64, 4096] {
        let samples = stream_all(FlacStreamDecoder::frames(), &frames, chunk);
        assert_eq!(samples, reference, "output changed at a {chunk}-byte feed");
    }
}

#[test]
fn a_bare_streaming_reader_reports_its_derived_properties() {
    let frames = d3_stream(2);
    let mut reader = FlacStreamDecoder::frames();
    // Nothing is known before the first header arrives, which is what `None`
    // already means on this type.
    assert_eq!(reader.spec(), None);
    assert_eq!(reader.stream_info(), None);
    assert_eq!(reader.md5_check(), None);

    let mut samples = Vec::new();
    reader.push(&frames).expect("push");
    while reader.pull(&mut samples, usize::MAX).expect("pull") > 0 {}
    assert_eq!(reader.spec().map(|spec| spec.sample_rate), Some(32_000));

    reader.finish(&mut samples).expect("finish");
    assert_eq!(samples, d3_stream_reference(2));
    assert_eq!(reader.md5_check(), Some(Md5Check::NoStreamInfo));
}

#[test]
fn resetting_a_bare_reader_returns_it_to_a_bare_reader() {
    // `reset` used to rebuild a reader waiting for a `fLaC` signature. On a
    // bare stream that signature never arrives, so a reset reader would
    // reject the very next byte it was given.
    let frames = d3_stream(2);
    let mut reader = FlacStreamDecoder::frames();
    let mut samples = Vec::new();
    reader.push(&frames[..20]).expect("push");
    reader.reset();

    reader.push(&frames).expect("push after reset");
    while reader.pull(&mut samples, usize::MAX).expect("pull") > 0 {}
    reader.finish(&mut samples).expect("finish");
    assert_eq!(samples, d3_stream_reference(2));
}

#[test]
fn a_partial_trailing_frame_is_an_error_rather_than_short_audio() {
    let frames = d3_stream(3);
    let cut = &frames[..frames.len() - 4];
    let error = FlacFrameReader::new(cut)
        .expect("the first header is intact")
        .decode(&mut Vec::new())
        .expect_err("a partial frame must not be quietly dropped");
    assert!(
        matches!(error, DecodeError::Truncated { .. }),
        "unexpected error: {error}"
    );
}

// -- The two escape codes -----------------------------------------------------

#[test]
fn an_unresolved_sample_rate_names_the_field() {
    let frames = d3_frame_escaping(true, false);
    let error = FlacFrameReader::new(&frames).expect_err("code 0b0000 is unresolvable here");
    match error {
        DecodeError::Malformed { expected, offset } => {
            assert!(
                expected.contains("sample rate code 0b0000"),
                "the error must name the field: {expected}"
            );
            assert_eq!(offset, 0);
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn an_unresolved_bit_depth_names_the_field() {
    let frames = d3_frame_escaping(false, true);
    let error = FlacFrameReader::new(&frames).expect_err("code 0b000 is unresolvable here");
    match error {
        DecodeError::Malformed { expected, .. } => assert!(
            expected.contains("bit depth code 0b000"),
            "the error must name the field: {expected}"
        ),
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn an_escape_code_in_a_later_frame_is_rejected_too() {
    // The first frame states both fields, so the stream's properties are
    // derivable, and a decoder that then answered a later frame's escape
    // code from them would be guessing at a streaminfo block that does not
    // exist. Both frames here are byte-identical apart from the codes.
    let mut frames = d3_frame_numbered(0);
    let mut second = d3_frame_escaping(true, false);
    second[4] = 1;
    reseal(&mut second);
    frames.extend_from_slice(&second);

    let error = FlacFrameReader::new(&frames)
        .expect("the first header is complete")
        .decode(&mut Vec::new())
        .expect_err("the second frame defers to nothing");
    match error {
        DecodeError::Malformed { expected, .. } => assert!(
            expected.contains("sample rate code 0b0000"),
            "the error must name the field: {expected}"
        ),
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn out_of_band_streaminfo_resolves_both_escape_codes() {
    let reference = d3_reference();
    for (rate, depth) in [(true, false), (false, true), (true, true)] {
        let frames = d3_frame_escaping(rate, depth);
        let reader = FlacFrameReader::with_stream_info(&frames, d3_stream_info());
        let mut samples = Vec::new();
        let report = reader
            .decode(&mut samples)
            .unwrap_or_else(|error| panic!("rate={rate} depth={depth}: {error}"));
        assert_eq!(samples, reference.samples(), "rate={rate} depth={depth}");
        // And the checksum, which a bare stream has no other way to get, was
        // actually used.
        assert_eq!(report.md5, Md5Check::Verified, "rate={rate} depth={depth}");
    }
}

#[test]
fn out_of_band_streaminfo_restores_the_checksum_on_the_streaming_path() {
    let frames = d3_frame_escaping(true, true);
    let mut reader = FlacStreamDecoder::frames_with_stream_info(d3_stream_info());
    let mut samples = Vec::new();
    reader.push(&frames).expect("push");
    while reader.pull(&mut samples, usize::MAX).expect("pull") > 0 {}
    // Before `finish` there is no verdict, and the reader does not pretend
    // there is one.
    assert_eq!(reader.md5_check(), None);

    reader.finish(&mut samples).expect("finish");
    assert_eq!(samples, d3_reference().samples());
    assert_eq!(reader.md5_check(), Some(Md5Check::Verified));
}

#[test]
fn a_raw_streaminfo_block_builds_the_info_that_decodes_bare_frames() {
    // The documented Ogg and Matroska use case end to end: the container
    // hands over the 34-byte streaminfo body and a run of bare frames, and
    // nothing else reaches the caller. In D.3's file the body sits at bytes
    // 8..42, which is exactly where Matroska's `CodecPrivate` carries it,
    // after the `fLaC` signature and the four-byte block header.
    let body = &D3_FILE[8..D3_AUDIO];
    let info = FlacStreamInfo::from_block(body).expect("the worked example's block parses");
    // The block parsed from raw bytes is the block the whole-file reader
    // read, field for field.
    assert_eq!(info, d3_stream_info());

    // The whole input at once.
    let mut samples = Vec::new();
    let report = FlacFrameReader::with_stream_info(&D3_FILE[D3_AUDIO..], info)
        .decode(&mut samples)
        .expect("bare frames decode under the supplied block");
    assert_eq!(samples, d3_reference().samples());
    // The block carries the MD5, so the audio was verified against it.
    assert_eq!(report.md5, Md5Check::Verified);

    // The same frames arriving in pieces.
    let mut reader = FlacStreamDecoder::frames_with_stream_info(info);
    let mut streamed = Vec::new();
    for piece in D3_FILE[D3_AUDIO..].chunks(7) {
        let mut offset = 0;
        while offset < piece.len() {
            offset += reader.push(&piece[offset..]).expect("push");
            while reader.pull(&mut streamed, usize::MAX).expect("pull") > 0 {}
        }
    }
    reader.finish(&mut streamed).expect("finish");
    assert_eq!(streamed, d3_reference().samples());
    assert_eq!(reader.md5_check(), Some(Md5Check::Verified));

    // A slice that is not exactly the body is a typed rejection, not a
    // guess, and the rejection names the length rule.
    let error = FlacStreamInfo::from_block(&D3_FILE[8..D3_AUDIO - 1])
        .expect_err("33 bytes is not a streaminfo body");
    match error {
        DecodeError::Malformed { expected, .. } => assert!(
            expected.contains("34"),
            "the error must name the length: {expected}"
        ),
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn a_streaminfo_with_no_checksum_says_so_rather_than_claiming_a_check() {
    let mut info = d3_stream_info();
    info.md5 = None;
    info.total_samples = None;
    let frames = d3_frame();
    let report = FlacFrameReader::with_stream_info(&frames, info)
        .decode(&mut Vec::new())
        .expect("decodes");
    assert_eq!(report.md5, Md5Check::ChecksumUnset);
    assert!(!report.md5.is_verified());
}

#[test]
fn a_supplied_checksum_that_does_not_match_is_an_error_not_audio() {
    let mut info = d3_stream_info();
    info.md5 = Some([0xAB; 16]);
    let frames = d3_frame();
    let error = FlacFrameReader::with_stream_info(&frames, info)
        .decode(&mut Vec::new())
        .expect_err("a wrong checksum must be rejected");
    match error {
        DecodeError::Malformed { expected, .. } => assert!(
            expected.contains("MD5"),
            "the error must name the checksum: {expected}"
        ),
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn a_supplied_streaminfo_that_disagrees_with_the_frames_is_rejected() {
    let mut info = d3_stream_info();
    info.spec = AudioSpec::mono(44_100);
    let frames = d3_frame();
    let error = FlacFrameReader::with_stream_info(&frames, info)
        .decode(&mut Vec::new())
        .expect_err("32000 in the frame against 44100 supplied");
    match error {
        DecodeError::Malformed { expected, .. } => assert!(
            expected.contains("sample rate"),
            "the error must name the field: {expected}"
        ),
        other => panic!("unexpected error: {other}"),
    }
}

// -- False sync ---------------------------------------------------------------

/// SplitMix64, so the runs below are the same runs on every machine and the
/// seed reproduces them exactly.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn fill(&mut self, buffer: &mut Vec<u8>, len: usize) {
        buffer.clear();
        while buffer.len() < len {
            buffer.extend_from_slice(&self.next_u64().to_le_bytes());
        }
        buffer.truncate(len);
    }
}

/// The seed the false-sync gate runs from. Stated so the run is reproducible
/// rather than "some random data".
const FALSE_SYNC_SEED: u64 = 0x0DEC_1B21_F1AC_5EED;

/// How many independent runs of random bytes go through the recovery entry
/// point.
const FALSE_SYNC_TRIALS: usize = 10_000_000;

#[test]
fn random_bytes_never_produce_audio() {
    // The sync code is 14 bits and the header CRC-8 is 8, so about one
    // position in 32,768 of random data opens something that could be a frame
    // header and about one in 3.5 x 10^7 carries one that also passes its
    // CRC-8. A recogniser that stopped there would emit audio out of noise
    // several times in a run this size. Chained validation (the frame's own
    // CRC-16, and a further header where that frame ends) is what makes the
    // count zero, and this is the gate that says so.
    let mut random = SplitMix64(FALSE_SYNC_SEED);
    let mut buffer = Vec::with_capacity(160);
    let mut emitted = 0usize;
    let mut accepted = 0usize;

    for trial in 0..FALSE_SYNC_TRIALS {
        // 16 to 143 bytes: long enough to hold several of the smallest legal
        // frames, short enough that ten million of them stay cheap.
        let len = 16 + (trial % 128);
        random.fill(&mut buffer, len);

        let mut samples = Vec::new();
        if let Ok(report) = FlacRecovery::new(&buffer).decode(&mut samples) {
            accepted += 1;
            emitted += samples.len();
            assert_eq!(
                samples.len(),
                report.samples,
                "trial {trial} of seed {FALSE_SYNC_SEED:#x} disagreed with its own report"
            );
        }
    }

    assert_eq!(
        (accepted, emitted),
        (0, 0),
        "{FALSE_SYNC_TRIALS} runs from seed {FALSE_SYNC_SEED:#x} produced \
         {accepted} accepted stream(s) and {emitted} sample(s) out of noise"
    );
}

// -- Recovery -----------------------------------------------------------------

#[test]
fn recovery_finds_a_frame_behind_garbage_and_loses_nothing() {
    let reference = d3_stream_reference(3);
    let mut random = SplitMix64(1);
    let mut junk = Vec::new();
    random.fill(&mut junk, 97);
    let mut damaged = junk.clone();
    damaged.extend_from_slice(&d3_stream(3));

    let mut samples = Vec::new();
    let report = FlacRecovery::new(&damaged)
        .decode(&mut samples)
        .expect("the frames behind the junk are findable");

    assert_eq!(samples, reference);
    assert_eq!(report.skipped.len(), 1);
    assert_eq!(report.skipped[0].bytes, 0..97);
    assert_eq!(report.skipped[0].reason, FlacSkipReason::NoSyncPoint);
    // Junk in front of the first frame costs no audio, and the report says
    // that rather than saying nothing.
    assert_eq!(report.skipped[0].frames, Some(0..0));
    assert_eq!(report.frames_lost(), Some(0));
}

#[test]
fn recovery_from_a_cut_mid_frame_reports_what_the_cut_cost() {
    let reference = d3_stream_reference(3);
    let frame_bytes = d3_frame().len();
    let per_frame = reference.len() / 3;

    // Ten bytes into the first frame, which is inside its header.
    let cut = 10;
    let damaged = &d3_stream(3)[cut..];
    let mut samples = Vec::new();
    let report = FlacRecovery::new(damaged)
        .decode(&mut samples)
        .expect("the two whole frames after the cut are recoverable");

    // The output is exactly the tail of the undamaged decode. Measured
    // against the reference, not against the report.
    assert_eq!(samples, reference[per_frame..]);
    assert_eq!(report.skipped.len(), 1);
    // The skip runs from the first byte to the start of the frame that
    // survived, which is one whole frame minus the bytes the cut removed.
    assert_eq!(report.skipped[0].bytes, 0..(frame_bytes - cut) as u64);
    // Where the output starts in the stream is always knowable: the frame it
    // resumed at says so.
    assert_eq!(report.first_frame, Some(per_frame as u64));
    // How much the gap cost is *not*, on this path. Nothing here says the
    // input began where the stream began, so a run handed over from the
    // middle of a longer stream is indistinguishable from a truncated one,
    // and claiming a loss would be an invention. The corpus has a bare
    // stream whose first surviving frame is number 12,927; answering "12,927
    // frames were lost" for a 404 KB file is exactly what this refuses.
    assert_eq!(
        report.skipped[0].frames, None,
        "an unanchored leading gap must not be given a width"
    );
    assert_eq!(report.frames_lost(), None);
    // Nothing verified this, and it says so: audio may be missing.
    assert_eq!(report.md5, Md5Check::AudioIncomplete);
}

/// D.3's frame with the blocking strategy bit set, so its coded number reads
/// as a sample number instead of a frame number.
///
/// The bit is the low bit of the second header byte: the 14-bit sync code
/// fills byte 0 and the top six bits of byte 1, then one reserved bit, then
/// this one. Both checksums are resealed, so a decoder that rejected the
/// result would fail the test rather than pass it quietly.
fn d3_frame_variable(number: u8) -> Vec<u8> {
    let mut frame = d3_frame_numbered(number);
    frame[1] |= 0x01;
    reseal(&mut frame);
    frame
}

/// A stream holding exactly one frame has no consecutive pair, so the coded
/// number's meaning falls back to the blocking strategy bit the header
/// declares.
///
/// # Why this is worth a test of its own
///
/// [`FlacRecovery`] deliberately does not trust that bit where it has
/// evidence: two frames decoded back to back differ by one under frame
/// numbering and by a whole block under sample numbering, and the corpus
/// carries a Flake 0.11 file whose bit is wrong. A single frame provides no
/// pair, and the question this answers is what happens then. The measured
/// answer is that the declared bit is used, which is the only evidence
/// available, and that neither reading is an error.
#[test]
fn a_single_frame_falls_back_to_the_blocking_strategy_it_declares() {
    let per_frame = d3_reference().samples().len() as u64;
    assert_eq!(per_frame, 24, "D.3 is a 24-sample block");

    // Fixed blocking: the coded number counts frames, so frame 3 starts at
    // sample 3 * 24.
    let mut samples = Vec::new();
    let report = FlacRecovery::new(&d3_frame_numbered(3))
        .decode(&mut samples)
        .expect("a lone fixed-blocksize frame recovers");
    assert_eq!(samples.len() as u64, per_frame);
    assert_eq!(
        report.first_frame,
        Some(3 * per_frame),
        "with no pair to measure, a fixed-blocksize frame number is multiplied \
         by the block size"
    );

    // Variable blocking: the coded number *is* the sample number, so the same
    // value 3 means sample 3.
    let mut samples = Vec::new();
    let report = FlacRecovery::new(&d3_frame_variable(3))
        .decode(&mut samples)
        .expect("a lone variable-blocksize frame recovers");
    assert_eq!(samples.len() as u64, per_frame);
    assert_eq!(
        report.first_frame,
        Some(3),
        "with no pair to measure, a variable-blocksize coded number is the \
         sample position itself"
    );

    // The two readings differ, which is what makes the fallback observable:
    // a test that could not tell them apart would pass whatever the code did.
    assert_ne!(3 * per_frame, 3);
}

#[test]
fn a_supplied_streaminfo_anchors_what_a_cut_mid_frame_cost() {
    // The same cut, with a streaminfo block supplied. A stream with a
    // streaminfo block starts where the stream starts, so the leading gap now
    // has both ends and the loss is a number rather than a shrug.
    let reference = d3_stream_reference(3);
    let per_frame = reference.len() / 3;
    let damaged = &d3_stream(3)[10..];

    let mut samples = Vec::new();
    let report = FlacRecovery::with_stream_info(damaged, d3_stream_info())
        .decode(&mut samples)
        .expect("the two whole frames after the cut are recoverable");

    assert_eq!(samples, reference[per_frame..]);
    assert_eq!(report.skipped.len(), 1);
    assert_eq!(
        report.skipped[0].frames,
        Some(0..per_frame as u64),
        "the lost sample positions must be the stream's own"
    );
    assert_eq!(report.frames_lost(), Some(per_frame as u64));
    assert_eq!(report.first_frame, Some(per_frame as u64));
    // Anchored, but still not verified: the checksum describes audio this
    // decode does not hold.
    assert_eq!(report.md5, Md5Check::AudioIncomplete);
}

#[test]
fn a_run_cut_exactly_on_a_frame_boundary_still_reports_the_missing_audio() {
    // No bytes are skipped at all (the input opens on a frame) and audio is
    // missing all the same. A report that counted only skipped *bytes* would
    // say nothing was lost here, which is the quiet shortening the rest of
    // the crate refuses.
    let reference = d3_stream_reference(3);
    let per_frame = reference.len() / 3;
    let damaged = &d3_stream(3)[d3_frame().len()..];

    let mut samples = Vec::new();
    let report = FlacRecovery::with_stream_info(damaged, d3_stream_info())
        .decode(&mut samples)
        .expect("two whole frames");

    assert_eq!(samples, reference[per_frame..]);
    assert_eq!(report.skipped.len(), 1);
    assert!(report.skipped[0].bytes.is_empty());
    assert_eq!(report.skipped[0].frames, Some(0..per_frame as u64));
    assert_eq!(report.frames_lost(), Some(per_frame as u64));
    assert_eq!(report.md5, Md5Check::AudioIncomplete);
}

#[test]
fn recovery_resyncs_after_a_corrupted_frame_and_names_what_it_lost() {
    let reference = d3_stream_reference(3);
    let frame_bytes = d3_frame().len();
    let per_frame = reference.len() / 3;

    // A bit flipped in the middle of the second frame's body. The CRC-16 must
    // reject that frame, and only that frame.
    let mut damaged = d3_stream(3);
    damaged[frame_bytes + 15] ^= 0x40;

    let mut samples = Vec::new();
    let report = FlacRecovery::new(&damaged)
        .decode(&mut samples)
        .expect("the first and third frames are intact");

    // The first and third frames, and nothing in between.
    let mut expected = reference[..per_frame].to_vec();
    expected.extend_from_slice(&reference[2 * per_frame..]);
    assert_eq!(samples, expected);

    assert_eq!(report.skipped.len(), 1);
    assert_eq!(
        report.skipped[0].bytes,
        frame_bytes as u64..2 * frame_bytes as u64
    );
    assert_eq!(report.skipped[0].reason, FlacSkipReason::FrameRejected);
    assert_eq!(
        report.skipped[0].frames,
        Some(per_frame as u64..2 * per_frame as u64)
    );
    assert_eq!(report.frames_lost(), Some(per_frame as u64));
    assert_eq!(report.md5, Md5Check::AudioIncomplete);
}

#[test]
fn recovery_on_an_undamaged_bare_stream_skips_nothing() {
    let frames = d3_stream(3);
    let mut samples = Vec::new();
    let report = FlacRecovery::new(&frames)
        .decode(&mut samples)
        .expect("clean");
    assert_eq!(samples, d3_stream_reference(3));
    assert!(report.skipped.is_empty());
    assert_eq!(report.frames_lost(), Some(0));
    // Clean, but still nothing checked it: there was no streaminfo.
    assert_eq!(report.md5, Md5Check::NoStreamInfo);
}

#[test]
fn recovery_with_streaminfo_verifies_when_nothing_was_lost() {
    let mut junk = Vec::new();
    SplitMix64(7).fill(&mut junk, 40);
    let mut damaged = junk;
    damaged.extend_from_slice(&d3_frame());

    let (decoded, report) = FlacRecovery::with_stream_info(&damaged, d3_stream_info())
        .decode_to_end()
        .expect("recovers");
    assert_eq!(decoded.samples(), d3_reference().samples());
    assert_eq!(report.frames_lost(), Some(0));
    // Bytes were skipped, none of them were audio, and the checksum still
    // covers everything that came out. That is the one case where a recovered
    // decode is as strong as an ordinary one, and it is reported as such.
    assert_eq!(report.md5, Md5Check::Verified);
}

#[test]
fn recovery_reports_the_stream_it_found() {
    let frames = d3_stream(2);
    let report = FlacRecovery::new(&frames)
        .decode(&mut Vec::new())
        .expect("clean");
    assert_eq!(report.stream_info.spec.sample_rate, 32_000);
    assert_eq!(report.stream_info.bits_per_sample, 8);
    assert_eq!(report.stream_info.max_block_size, 65_535);
}

// -- Malformed input ----------------------------------------------------------

#[test]
fn a_byte_run_holding_no_frame_at_all_is_rejected() {
    let mut random = SplitMix64(0x5AFE);
    let mut buffer = Vec::new();
    for len in [0, 1, 5, 6, 100, 4096] {
        random.fill(&mut buffer, len);
        let error = FlacRecovery::new(&buffer)
            .decode(&mut Vec::new())
            .expect_err("{len} bytes of noise are not audio");
        assert!(
            matches!(error, DecodeError::Malformed { .. }),
            "at {len} bytes: {error}"
        );
    }
}

#[test]
fn a_single_validating_header_is_not_a_sync_point() {
    // A whole frame header, CRC-8 and all, with nothing behind it. One header
    // is what a naive recogniser accepts and is exactly what must not be
    // enough: there is no frame to check a CRC-16 against and no successor to
    // chain to.
    let header = &d3_frame()[..D3_HEADER_BYTES];
    assert_eq!(crc8(&header[..D3_HEADER_BYTES - 1]), header[6]);

    let error = FlacRecovery::new(header)
        .decode(&mut Vec::new())
        .expect_err("a header on its own is not evidence");
    assert!(
        matches!(error, DecodeError::Malformed { .. }),
        "unexpected error: {error}"
    );
}

#[test]
fn a_run_whose_every_frame_fails_its_crc_is_rejected() {
    // Three well-formed frames, each with one byte of its body flipped. Every
    // header still validates; no frame does.
    let frame_bytes = d3_frame().len();
    let mut damaged = d3_stream(3);
    for frame in 0..3 {
        damaged[frame * frame_bytes + 12] ^= 0x80;
    }
    let error = FlacRecovery::new(&damaged)
        .decode(&mut Vec::new())
        .expect_err("no frame survives its CRC-16");
    assert!(
        matches!(error, DecodeError::Malformed { .. }),
        "unexpected error: {error}"
    );
}

#[test]
fn every_truncation_and_every_flipped_byte_is_an_error_rather_than_a_panic() {
    let stream = d3_stream(3);
    let info = d3_stream_info();

    for cut in 0..=stream.len() {
        let piece = &stream[..cut];
        // Four entry points, none of which may panic or hang on any prefix.
        let _ = FlacFrameReader::new(piece).and_then(|r| r.decode(&mut Vec::new()));
        let _ = FlacFrameReader::with_stream_info(piece, info).decode(&mut Vec::new());
        let _ = FlacRecovery::new(piece).decode(&mut Vec::new());
        let _ = FlacRecovery::with_stream_info(piece, info).decode(&mut Vec::new());
    }

    for byte in 0..stream.len() {
        let mut damaged = stream.clone();
        damaged[byte] ^= 0xFF;
        let _ = FlacFrameReader::new(&damaged).and_then(|r| r.decode(&mut Vec::new()));
        let _ = FlacRecovery::new(&damaged).decode(&mut Vec::new());
        let _ = FlacRecovery::with_stream_info(&damaged, info).decode(&mut Vec::new());
    }
}

// -- The pinned witness -------------------------------------------------------

/// FNV-1a, so the witness is the bytes themselves and not a float comparison.
fn fnv1a(bytes: impl IntoIterator<Item = u8>) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Appends the bit patterns of `samples` to `witness`.
fn absorb(witness: &mut Vec<u8>, samples: &[f32]) {
    witness.extend(
        samples
            .iter()
            .flat_map(|sample| sample.to_bits().to_le_bytes()),
    );
}

#[test]
fn bare_frame_output_is_bit_identical_to_a_pinned_witness() {
    // The same claim the crate makes for every other format, extended to the
    // three paths this pass added. The constant is re-pinned only when the
    // sweep below changes; a hash that moved on an unchanged sweep is a
    // determinism break, not a number to update.
    let mut witness: Vec<u8> = Vec::new();
    let info = d3_stream_info();

    // Whole-buffer, derived properties.
    for count in 1..=4u8 {
        let frames = d3_stream(count);
        let mut samples = Vec::new();
        FlacFrameReader::new(&frames)
            .expect("parses")
            .decode(&mut samples)
            .expect("decodes");
        absorb(&mut witness, &samples);
    }

    // Whole-buffer, supplied streaminfo, both escape codes exercised.
    for (rate, depth) in [(false, false), (true, false), (false, true), (true, true)] {
        let frames = d3_frame_escaping(rate, depth);
        let mut samples = Vec::new();
        FlacFrameReader::with_stream_info(&frames, info)
            .decode(&mut samples)
            .expect("decodes");
        absorb(&mut witness, &samples);
    }

    // Streaming, at feed sizes that split headers and frames differently.
    for chunk in [1, 3, 7, 31, 4096] {
        let samples = stream_all(FlacStreamDecoder::frames(), &d3_stream(3), chunk);
        absorb(&mut witness, &samples);
    }

    // Recovery, over each of the four damage shapes.
    let frame_bytes = d3_frame().len();
    let mut junk = Vec::new();
    SplitMix64(11).fill(&mut junk, 53);
    let mut prepended = junk;
    prepended.extend_from_slice(&d3_stream(3));
    let mut corrupted = d3_stream(3);
    corrupted[frame_bytes + 15] ^= 0x40;
    let cut = d3_stream(3)[10..].to_vec();
    for input in [d3_stream(3), prepended, cut, corrupted] {
        let mut samples = Vec::new();
        FlacRecovery::new(&input)
            .decode(&mut samples)
            .expect("recovers");
        absorb(&mut witness, &samples);
    }

    assert_eq!(
        fnv1a(witness),
        0xb927_8d5a_d076_c4e3,
        "bare frame, out-of-band streaminfo or recovery output changed"
    );
}
