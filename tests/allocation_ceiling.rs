//! What the reader allocates, measured rather than reasoned about.
//!
//! "A size declared in a file is a claim, not a fact" is easy to write in a doc
//! comment and easy to stop being true. A `data` chunk announcing four
//! gigabytes inside a two-kilobyte file must not cause a four-gigabyte
//! allocation, and the only way to know that it does not is to count the bytes
//! the allocator hands out while the file is being parsed.
//!
//! # Why this is its own test binary
//!
//! The counter below is a `#[global_allocator]`, so it sees every allocation
//! the process makes. `cargo test` runs a binary's tests on several threads at
//! once, so a peak measured while another test is running would be that other
//! test's peak. This file therefore holds **exactly one** `#[test]`, and every
//! measurement is inside it.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use decibri_decode::{
    AiffReader, AiffStreamDecoder, DecodeError, FlacReader, FlacRecovery, FlacStreamDecoder,
    StreamSource, WavReader, WavStreamDecoder,
};

/// Bytes currently allocated, and the high-water mark since the last reset.
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

/// Every byte the allocator has ever handed out, never decremented.
///
/// A second measurement rather than a refinement of [`PEAK`], because the two
/// catch different failures. A peak bounds how much is held at once and says
/// nothing about how often; a scan that allocates and frees four megabytes
/// once per byte of input has a flat peak and unbounded *work*. That is the
/// shape the pre-publish audit's fuzzing found in the recovery scanner, and
/// it is invisible to a high-water mark.
static TOTAL: AtomicUsize = AtomicUsize::new(0);

/// The system allocator with a running total in front of it.
struct Counting;

// SAFETY-free by construction: every method forwards to `System` and the only
// addition is arithmetic on two atomics. The crate under test is
// `#![forbid(unsafe_code)]`; a `GlobalAlloc` implementation cannot be, because
// the trait itself is unsafe to implement.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            record(layout.size() as isize);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        record(-(layout.size() as isize));
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let moved = unsafe { System.realloc(pointer, layout, new_size) };
        if !moved.is_null() {
            record(new_size as isize - layout.size() as isize);
        }
        moved
    }
}

fn record(delta: isize) {
    let live = if delta >= 0 {
        TOTAL.fetch_add(delta as usize, Ordering::Relaxed);
        LIVE.fetch_add(delta as usize, Ordering::Relaxed) + delta as usize
    } else {
        LIVE.fetch_sub(delta.unsigned_abs(), Ordering::Relaxed) - delta.unsigned_abs()
    };
    PEAK.fetch_max(live, Ordering::Relaxed);
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Runs `work` and returns the largest number of bytes live at any moment
/// during it, over and above what was live when it started.
fn peak_bytes<T>(work: impl FnOnce() -> T) -> (T, usize) {
    let base = LIVE.load(Ordering::Relaxed);
    PEAK.store(base, Ordering::Relaxed);
    let value = work();
    let peak = PEAK.load(Ordering::Relaxed);
    (value, peak.saturating_sub(base))
}

/// Runs `work` and returns every byte the allocator handed out during it,
/// whether or not it was still held at the end.
fn total_bytes<T>(work: impl FnOnce() -> T) -> (T, usize) {
    let before = TOTAL.load(Ordering::Relaxed);
    let value = work();
    (value, TOTAL.load(Ordering::Relaxed).saturating_sub(before))
}

// -- The files ----------------------------------------------------------------

fn chunk(id: &[u8; 4], declared: u32, body: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(id);
    out.extend_from_slice(&declared.to_le_bytes());
    out.extend_from_slice(body);
    if body.len() % 2 == 1 {
        out.push(0);
    }
    out
}

fn fmt_body(channels: u16, rate: u32, bits: u16) -> Vec<u8> {
    let block_align = channels * bits / 8;
    let mut body = Vec::new();
    body.extend_from_slice(&1u16.to_le_bytes());
    body.extend_from_slice(&channels.to_le_bytes());
    body.extend_from_slice(&rate.to_le_bytes());
    body.extend_from_slice(&(rate * u32::from(block_align)).to_le_bytes());
    body.extend_from_slice(&block_align.to_le_bytes());
    body.extend_from_slice(&bits.to_le_bytes());
    body
}

fn file(magic: &[u8; 4], chunks: &[Vec<u8>]) -> Vec<u8> {
    let body: Vec<u8> = chunks.concat();
    let mut out = Vec::new();
    out.extend_from_slice(magic);
    out.extend_from_slice(&(4 + body.len() as u32).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(&body);
    out
}

/// Two kilobytes of file, with a `data` chunk declaring four gigabytes.
fn four_gigabytes_in_two_kilobytes() -> Vec<u8> {
    file(
        b"RIFF",
        &[
            chunk(b"fmt ", 16, &fmt_body(2, 48_000, 16)),
            chunk(b"data", 0xFFFF_FF00, &vec![0x5A; 2_000]),
        ],
    )
}

/// The same claim made through RF64's 64-bit path, where the declared size is
/// not merely large but larger than a `u32` can hold.
fn sixteen_gigabytes_in_two_kilobytes() -> Vec<u8> {
    let mut ds64 = Vec::new();
    ds64.extend_from_slice(&0u64.to_le_bytes()); // riffSize
    ds64.extend_from_slice(&17_179_869_184u64.to_le_bytes()); // dataSize: 16 GiB
    ds64.extend_from_slice(&4_294_967_296u64.to_le_bytes()); // sampleCount
    ds64.extend_from_slice(&0u32.to_le_bytes()); // no override table
    file(
        b"RF64",
        &[
            chunk(b"ds64", 28, &ds64),
            chunk(b"fmt ", 16, &fmt_body(2, 48_000, 16)),
            chunk(b"data", u32::MAX, &vec![0x5A; 2_000]),
        ],
    )
}

/// A `ds64` chunk declaring an override table of four billion entries inside a
/// twenty-eight-byte body. Reserving for the count before checking it would be
/// a 51 GiB allocation.
fn a_ds64_table_of_four_billion_entries() -> Vec<u8> {
    let mut ds64 = Vec::new();
    ds64.extend_from_slice(&0u64.to_le_bytes());
    ds64.extend_from_slice(&256u64.to_le_bytes());
    ds64.extend_from_slice(&0u64.to_le_bytes());
    ds64.extend_from_slice(&u32::MAX.to_le_bytes()); // tableLength
    file(
        b"RF64",
        &[
            chunk(b"ds64", 28, &ds64),
            chunk(b"fmt ", 16, &fmt_body(2, 48_000, 16)),
            chunk(b"data", u32::MAX, &vec![0x5A; 256]),
        ],
    )
}

/// A `fmt ` chunk declaring sixty-four kilobytes inside a small file, which is
/// the one chunk body the streaming reader buffers rather than skips.
fn an_oversized_fmt_chunk() -> Vec<u8> {
    file(
        b"RIFF",
        &[
            chunk(b"fmt ", 60_000, &fmt_body(2, 48_000, 16)),
            chunk(b"data", 8, &[0x5A; 8]),
        ],
    )
}

/// The stream driven to exhaustion, so the measurement covers `push`, `pull`
/// and `finish` rather than only the parse.
fn stream_all(bytes: &[u8], piece: usize) -> Result<usize, DecodeError> {
    let mut stream = WavStreamDecoder::new();
    let mut samples = Vec::new();
    for slice in bytes.chunks(piece) {
        let mut offset = 0;
        while offset < slice.len() {
            let taken = stream.push(&slice[offset..])?;
            offset += taken;
            if taken == 0 {
                while stream.pull(&mut samples, usize::MAX)? > 0 {}
            }
        }
    }
    while stream.pull(&mut samples, usize::MAX)? > 0 {}
    stream.finish(&mut samples)?;
    Ok(samples.len())
}

// -- The AIFF files -----------------------------------------------------------

/// A big-endian chunk with an untruthful declared size.
fn be_chunk(id: &[u8; 4], declared: u32, body: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(id);
    out.extend_from_slice(&declared.to_be_bytes());
    out.extend_from_slice(body);
    if body.len() % 2 == 1 {
        out.push(0);
    }
    out
}

/// A `COMM` body: channels, frames, bits, and an 8 kHz 80-bit rate.
fn comm_body(channels: u16, frames: u32, bits: u16) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(&channels.to_be_bytes());
    body.extend_from_slice(&frames.to_be_bytes());
    body.extend_from_slice(&bits.to_be_bytes());
    body.extend_from_slice(&[0x40, 0x0B, 0xFA, 0, 0, 0, 0, 0, 0, 0]);
    body
}

fn aiff_file(chunks: &[Vec<u8>]) -> Vec<u8> {
    let body: Vec<u8> = chunks.concat();
    let mut out = Vec::new();
    out.extend_from_slice(b"FORM");
    out.extend_from_slice(&(4 + body.len() as u32).to_be_bytes());
    out.extend_from_slice(b"AIFF");
    out.extend_from_slice(&body);
    out
}

/// An `SSND` declaring four gigabytes inside a two-kilobyte file, with a
/// `numSampleFrames` to match the lie.
fn aiff_four_gigabytes_in_two_kilobytes() -> Vec<u8> {
    let mut ssnd = vec![0u8; 8]; // offset, blockSize
    ssnd.extend_from_slice(&[0x5A; 2_000]);
    aiff_file(&[
        be_chunk(b"COMM", 18, &comm_body(2, 0xFFFF_FF00, 16)),
        be_chunk(b"SSND", 0xFFFF_FF00, &ssnd),
    ])
}

/// A truthful `SSND` whose `COMM` claims four billion frames: the mismatch
/// must be rejected without reserving for the claim.
fn aiff_four_billion_declared_frames() -> Vec<u8> {
    let mut ssnd = vec![0u8; 8];
    ssnd.extend_from_slice(&[0x5A; 2_000]);
    aiff_file(&[
        be_chunk(b"COMM", 18, &comm_body(2, u32::MAX, 16)),
        be_chunk(b"SSND", 2_008, &ssnd),
    ])
}

/// A `COMM` declaring sixty kilobytes, which is the one chunk body the AIFF
/// streaming reader buffers rather than skips.
fn an_oversized_comm_chunk() -> Vec<u8> {
    let mut ssnd = vec![0u8; 8];
    ssnd.extend_from_slice(&[0x5A; 8]);
    aiff_file(&[
        be_chunk(b"COMM", 60_000, &comm_body(2, 2, 16)),
        be_chunk(b"SSND", 16, &ssnd),
    ])
}

/// The AIFF stream driven to exhaustion, mirroring [`stream_all`].
fn aiff_stream_all(bytes: &[u8], piece: usize) -> Result<usize, DecodeError> {
    let mut stream = AiffStreamDecoder::new();
    let mut samples = Vec::new();
    for slice in bytes.chunks(piece) {
        let mut offset = 0;
        while offset < slice.len() {
            let taken = stream.push(&slice[offset..])?;
            offset += taken;
            if taken == 0 {
                while stream.pull(&mut samples, usize::MAX)? > 0 {}
            }
        }
    }
    while stream.pull(&mut samples, usize::MAX)? > 0 {}
    stream.finish(&mut samples)?;
    Ok(samples.len())
}

// -- The FLAC streams ---------------------------------------------------------

/// CRC-8 over `x^8 + x^2 + x^1 + x^0`, so these files can carry frame headers
/// that get past the header check and are rejected for the reason under test.
fn flac_crc8(bytes: &[u8]) -> u8 {
    let mut crc = 0u8;
    for byte in bytes {
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

/// A FLAC stream: the signature, a streaminfo block, any extra metadata
/// blocks, then `audio` verbatim.
fn flac_file(
    max_block: u16,
    channels: u8,
    bits: u8,
    total: u64,
    extra: &[(u8, u32, usize)],
    audio: &[u8],
) -> Vec<u8> {
    let mut info = Vec::new();
    info.extend_from_slice(&16u16.to_be_bytes()); // minimum block size
    info.extend_from_slice(&max_block.to_be_bytes());
    // frame sizes: not known
    info.extend_from_slice(&[0, 0, 0, 0, 0, 0]);
    // 20 bits of rate, 3 of channels-1, 5 of bits-1, 36 of total samples.
    let packed = (48_000u64 << 44)
        | (u64::from(channels - 1) << 41)
        | (u64::from(bits - 1) << 36)
        | (total & 0xF_FFFF_FFFF);
    info.extend_from_slice(&packed.to_be_bytes());
    info.extend_from_slice(&[0u8; 16]); // MD5: not known
    assert_eq!(info.len(), 34);

    let mut out = b"fLaC".to_vec();
    out.push(if extra.is_empty() { 0x80 } else { 0x00 });
    out.extend_from_slice(&[0, 0, 34]);
    out.extend_from_slice(&info);
    for (index, (kind, declared, present)) in extra.iter().enumerate() {
        let last = index + 1 == extra.len();
        out.push(if last { 0x80 | kind } else { *kind });
        out.extend_from_slice(&declared.to_be_bytes()[1..]);
        out.extend(std::iter::repeat_n(0x5Au8, *present));
    }
    out.extend_from_slice(audio);
    out
}

/// An eight-byte frame header declaring `block_size` samples in `channels`
/// channels, with a correct CRC-8 so the header itself is accepted.
fn flac_frame_header(block_size: u32, channels: u8) -> Vec<u8> {
    let mut header = vec![
        0xFF,
        0xF8,
        0x70, // uncommon 16-bit block size, rate from streaminfo
        (channels - 1) << 4,
        0x00, // coded number: frame 0
    ];
    header.extend_from_slice(&((block_size - 1) as u16).to_be_bytes());
    header.push(flac_crc8(&header));
    header
}

/// The same header, but stating its own sample rate and bit depth rather than
/// deferring them to streaminfo.
///
/// [`FlacRecovery`] has no streaminfo to defer to, so a header that defers is
/// rejected before anything is reserved for it, which would make a
/// measurement over such headers measure nothing. This one is self-contained
/// and reaches the decode buffer: 48 kHz, 16 bits, `channels` channels.
fn flac_self_describing_frame_header(block_size: u32, channels: u8) -> Vec<u8> {
    let mut header = vec![
        0xFF,
        0xF8,
        0x7A, // uncommon 16-bit block size, sample rate code 10 (48 kHz)
        ((channels - 1) << 4) | (4 << 1), // bit depth code 4 (16 bits)
        0x00, // coded number: frame 0
    ];
    header.extend_from_slice(&((block_size - 1) as u16).to_be_bytes());
    header.push(flac_crc8(&header));
    header
}

/// The FLAC stream driven to exhaustion, mirroring [`stream_all`].
fn flac_stream_all(bytes: &[u8], piece: usize) -> Result<usize, DecodeError> {
    let mut stream = FlacStreamDecoder::new();
    let mut samples = Vec::new();
    for slice in bytes.chunks(piece) {
        let mut offset = 0;
        while offset < slice.len() {
            let taken = stream.push(&slice[offset..])?;
            offset += taken;
            if taken == 0 {
                while stream.pull(&mut samples, usize::MAX)? > 0 {}
            }
        }
    }
    while stream.pull(&mut samples, usize::MAX)? > 0 {}
    stream.finish(&mut samples)?;
    Ok(samples.len())
}

/// What the FLAC reader is allowed to allocate while rejecting a frame that
/// declares the largest block the format permits.
///
/// **This is the one place in the crate where an allocation follows a number
/// from the file**, and it is recorded here rather than explained away. A
/// FLAC frame declares its own sample count and a decoder must hold that many
/// samples to reconstruct them; there is no equivalent of WAV's "the payload
/// is as long as the bytes that arrived", because a constant subframe codes
/// 65535 samples in about forty bits. So the bound is the format's own:
/// 65535 samples in 8 channels at 8 bytes each, or 4,194,240 bytes, and the
/// streaminfo maximum block size (which the reader enforces on every frame)
/// brings it down to whatever a real file declares.
///
/// Every *other* FLAC size is bounded the way WAV's and AIFF's are: the
/// metadata block lengths, the streaminfo frame sizes and the total sample
/// count reach no allocation at all, which is what the cases below measure.
const FLAC_FRAME_CEILING: usize = 65_535 * 8 * std::mem::size_of::<i64>();

/// Every over-declared case, measured. One test, because the counter is
/// process-wide and `cargo test` is not single-threaded.
#[test]
fn no_declared_size_reaches_the_allocator() {
    /// What a rejected parse is allowed to allocate. Generous by two orders of
    /// magnitude against the sizes being declared (four gigabytes is
    /// 4,294,967,296) and tight enough that reserving for any of them fails
    /// it. The reader is expected to allocate nothing at all on these paths;
    /// the room is for the error value and the test harness's own churn.
    const CEILING: usize = 64 * 1024;

    let mut report = Vec::new();

    // -- data declaring four gigabytes inside two kilobytes --------------
    let bytes = four_gigabytes_in_two_kilobytes();
    let (result, peak) = peak_bytes(|| WavReader::new(&bytes).err());
    assert!(result.is_some(), "an over-declared data chunk was accepted");
    report.push(("RIFF data declaring 0xFFFFFF00 in a 2,020-byte file", peak));
    assert!(peak <= CEILING, "whole-file parse allocated {peak} bytes");

    let (streamed, stream_peak) = peak_bytes(|| stream_all(&bytes, 64).err());
    assert!(streamed.is_some(), "the stream accepted a short data chunk");
    report.push(("  the same file, streamed in 64-byte pieces", stream_peak));
    assert!(
        stream_peak <= CEILING,
        "stream allocated {stream_peak} bytes"
    );

    // -- RF64 ds64 declaring sixteen gigabytes ---------------------------
    let bytes = sixteen_gigabytes_in_two_kilobytes();
    let (result, peak) = peak_bytes(|| WavReader::new(&bytes).err());
    assert!(result.is_some(), "a 16 GiB ds64 data size was accepted");
    report.push(("RF64 ds64 declaring 16 GiB in a 2,048-byte file", peak));
    assert!(peak <= CEILING, "RF64 parse allocated {peak} bytes");

    let (streamed, stream_peak) = peak_bytes(|| stream_all(&bytes, 512).err());
    assert!(streamed.is_some(), "the stream accepted a 16 GiB data size");
    report.push(("  the same file, streamed in 512-byte pieces", stream_peak));
    assert!(
        stream_peak <= CEILING,
        "RF64 stream allocated {stream_peak}"
    );

    // -- a ds64 override table of four billion entries -------------------
    let bytes = a_ds64_table_of_four_billion_entries();
    let (result, peak) = peak_bytes(|| WavReader::new(&bytes).err());
    assert!(
        result.is_some(),
        "a 4-billion-entry ds64 table was accepted"
    );
    report.push(("ds64 declaring 4,294,967,295 table entries", peak));
    assert!(peak <= CEILING, "ds64 table parse allocated {peak} bytes");

    let (streamed, stream_peak) = peak_bytes(|| stream_all(&bytes, 7).err());
    assert!(streamed.is_some(), "the stream accepted the ds64 table");
    report.push(("  the same file, streamed in 7-byte pieces", stream_peak));
    assert!(
        stream_peak <= CEILING,
        "ds64 stream allocated {stream_peak}"
    );

    // -- an oversized fmt chunk, the one body the stream buffers ---------
    let bytes = an_oversized_fmt_chunk();
    let (result, peak) = peak_bytes(|| WavReader::new(&bytes).err());
    assert!(result.is_some(), "an over-declared fmt chunk was accepted");
    report.push(("fmt declaring 60,000 bytes in a 60-byte file", peak));
    assert!(peak <= CEILING, "fmt parse allocated {peak} bytes");

    let (streamed, stream_peak) = peak_bytes(|| stream_all(&bytes, 3).err());
    assert!(streamed.is_some(), "the stream accepted the fmt chunk");
    report.push(("  the same file, streamed in 3-byte pieces", stream_peak));
    assert!(stream_peak <= CEILING, "fmt stream allocated {stream_peak}");

    // -- the control: a file that is what it says it is ------------------
    //
    // Without this the gate would pass just as well on a reader that allocated
    // nothing because it decoded nothing.
    let healthy = file(
        b"RIFF",
        &[
            chunk(b"fmt ", 16, &fmt_body(2, 48_000, 16)),
            chunk(b"data", 2_000, &vec![0x5A; 2_000]),
        ],
    );
    let (frames, peak) = peak_bytes(|| {
        let reader = WavReader::new(&healthy).expect("a healthy file");
        reader.decode_to_end().samples().len()
    });
    assert_eq!(frames, 1_000, "the control decoded nothing");
    report.push((
        "control: a truthful 2 KiB file, decoded to 1,000 samples",
        peak,
    ));
    // 1,000 f32 is 4,000 bytes; the reservation is from the payload's real
    // length, so the peak is that and not the declared anything.
    assert!(
        (4_000..=CEILING).contains(&peak),
        "the control allocated {peak} bytes, which is not one f32 per sample"
    );

    // -- AIFF: SSND declaring four gigabytes inside two kilobytes --------
    let bytes = aiff_four_gigabytes_in_two_kilobytes();
    let (result, peak) = peak_bytes(|| AiffReader::new(&bytes).err());
    assert!(result.is_some(), "an over-declared SSND chunk was accepted");
    report.push(("AIFF SSND declaring 0xFFFFFF00 in a 2,038-byte file", peak));
    assert!(
        peak <= CEILING,
        "AIFF whole-file parse allocated {peak} bytes"
    );

    let (streamed, stream_peak) = peak_bytes(|| aiff_stream_all(&bytes, 64).err());
    assert!(streamed.is_some(), "the AIFF stream accepted a short SSND");
    report.push(("  the same file, streamed in 64-byte pieces", stream_peak));
    assert!(
        stream_peak <= CEILING,
        "AIFF stream allocated {stream_peak}"
    );

    // -- AIFF: COMM claiming four billion frames over a truthful SSND ----
    let bytes = aiff_four_billion_declared_frames();
    let (result, peak) = peak_bytes(|| AiffReader::new(&bytes).err());
    assert!(
        result.is_some(),
        "a 4-billion-frame COMM claim was accepted"
    );
    report.push((
        "AIFF COMM declaring 4,294,967,295 frames over 2,000 bytes",
        peak,
    ));
    assert!(peak <= CEILING, "the frame-count mismatch allocated {peak}");

    let (streamed, stream_peak) = peak_bytes(|| aiff_stream_all(&bytes, 512).err());
    assert!(
        streamed.is_some(),
        "the AIFF stream accepted the frame claim"
    );
    report.push(("  the same file, streamed in 512-byte pieces", stream_peak));
    assert!(
        stream_peak <= CEILING,
        "AIFF stream allocated {stream_peak}"
    );

    // -- AIFF: an oversized COMM, the one body the stream buffers --------
    let bytes = an_oversized_comm_chunk();
    let (result, peak) = peak_bytes(|| AiffReader::new(&bytes).err());
    assert!(result.is_some(), "an over-declared COMM chunk was accepted");
    report.push(("AIFF COMM declaring 60,000 bytes in a 62-byte file", peak));
    assert!(peak <= CEILING, "AIFF COMM parse allocated {peak} bytes");

    let (streamed, stream_peak) = peak_bytes(|| aiff_stream_all(&bytes, 3).err());
    assert!(
        streamed.is_some(),
        "the AIFF stream accepted the COMM chunk"
    );
    report.push(("  the same file, streamed in 3-byte pieces", stream_peak));
    assert!(
        stream_peak <= CEILING,
        "AIFF COMM stream allocated {stream_peak}"
    );

    // -- the AIFF control: a truthful file, decoded in full --------------
    let mut ssnd = vec![0u8; 8];
    ssnd.extend_from_slice(&[0x5A; 2_000]);
    let healthy_aiff = aiff_file(&[
        be_chunk(b"COMM", 18, &comm_body(2, 500, 16)),
        be_chunk(b"SSND", 2_008, &ssnd),
    ]);
    let (samples, peak) = peak_bytes(|| {
        let reader = AiffReader::new(&healthy_aiff).expect("a healthy AIFF");
        reader.decode_to_end().samples().len()
    });
    assert_eq!(samples, 1_000, "the AIFF control decoded nothing");
    report.push((
        "control: a truthful 2 KiB AIFF, decoded to 1,000 samples",
        peak,
    ));
    assert!(
        (4_000..=CEILING).contains(&peak),
        "the AIFF control allocated {peak} bytes, which is not one f32 per sample"
    );

    // -- FLAC ------------------------------------------------------------
    {
        // A metadata block declaring sixteen megabytes inside a 60-byte file.
        // This is the FLAC equivalent of the `data` chunk claim above.
        let bytes = flac_file(4_096, 2, 16, 1_000, &[(1, 0x00FF_FFFF, 8)], &[]);
        let (result, peak) = peak_bytes(|| FlacReader::new(&bytes).err());
        assert!(
            result.is_some(),
            "a metadata block declaring 16 MiB was accepted"
        );
        report.push((
            "FLAC metadata block declaring 16,777,215 bytes in a 54-byte file",
            peak,
        ));
        assert!(peak <= CEILING, "FLAC metadata parse allocated {peak}");

        let (streamed, stream_peak) = peak_bytes(|| flac_stream_all(&bytes, 7).err());
        assert!(streamed.is_some(), "the FLAC stream accepted the block");
        report.push(("  the same stream, pushed in 7-byte pieces", stream_peak));
        assert!(
            stream_peak <= CEILING,
            "FLAC metadata stream allocated {stream_peak}"
        );

        // Streaminfo declaring sixty-eight billion samples over no audio: the
        // total sample count reaches no allocation at all.
        let bytes = flac_file(4_096, 8, 32, 0xF_FFFF_FFFF, &[], &[]);
        let (result, peak) = peak_bytes(|| {
            FlacReader::new(&bytes)
                .and_then(|reader| reader.decode_to_end())
                .err()
        });
        assert!(result.is_some(), "a 68-billion-sample claim was accepted");
        report.push((
            "FLAC streaminfo declaring 68,719,476,735 samples over no audio",
            peak,
        ));
        assert!(peak <= CEILING, "the sample-count claim allocated {peak}");

        // A frame header declaring the largest block the format permits,
        // followed by nothing. This is the one number a file gets to set.
        let bytes = flac_file(65_535, 8, 32, 0, &[], &flac_frame_header(65_535, 8));
        let (result, peak) = peak_bytes(|| {
            FlacReader::new(&bytes)
                .and_then(|reader| reader.decode_to_end())
                .err()
        });
        assert!(
            result.is_some(),
            "an empty 65,535-sample frame was accepted"
        );
        report.push((
            "FLAC frame declaring 65,535 samples in 8 channels with no body",
            peak,
        ));
        assert!(
            peak <= FLAC_FRAME_CEILING + CEILING,
            "the frame buffer allocated {peak}, past the format's own bound"
        );

        // The same frame behind a streaminfo that permits far less: the
        // maximum block size check keeps the buffer to what the file could
        // legitimately need, so nothing is reserved at all.
        let bytes = flac_file(4_096, 8, 32, 0, &[], &flac_frame_header(65_535, 8));
        let (result, peak) = peak_bytes(|| {
            FlacReader::new(&bytes)
                .and_then(|reader| reader.decode_to_end())
                .err()
        });
        assert!(
            result.is_some(),
            "a frame past the declared maximum decoded"
        );
        report.push((
            "FLAC frame declaring 65,535 samples where streaminfo permits 4,096",
            peak,
        ));
        assert!(
            peak <= CEILING,
            "a frame past the streaminfo maximum allocated {peak}"
        );

        // The control: a truthful stream, decoded in full. RFC 9639 appendix
        // D.2's worked example, 19 frames of stereo.
        let healthy: &[u8] = &[
            0x66, 0x4c, 0x61, 0x43, 0x80, 0x00, 0x00, 0x22, 0x10, 0x00, 0x10, 0x00, 0x00, 0x00,
            0x1f, 0x00, 0x00, 0x1f, 0x07, 0xd0, 0x00, 0x70, 0x00, 0x00, 0x00, 0x18, 0xf8, 0xf9,
            0xe3, 0x96, 0xf5, 0xcb, 0xcf, 0xc6, 0xdc, 0x80, 0x7f, 0x99, 0x77, 0x90, 0x6b, 0x32,
            0xff, 0xf8, 0x68, 0x02, 0x00, 0x17, 0xe9, 0x44, 0x00, 0x4f, 0x6f, 0x31, 0x3d, 0x10,
            0x47, 0xd2, 0x27, 0xcb, 0x6d, 0x09, 0x08, 0x31, 0x45, 0x2b, 0xdc, 0x28, 0x22, 0x22,
            0x80, 0x57, 0xa3,
        ];
        let (samples, peak) = peak_bytes(|| {
            FlacReader::new(healthy)
                .expect("a healthy FLAC")
                .decode_to_end()
                .expect("decode")
                .samples()
                .len()
        });
        assert_eq!(samples, 24, "the FLAC control decoded nothing");
        report.push(("control: RFC 9639's example 3, decoded to 24 samples", peak));
        assert!(peak <= CEILING, "the FLAC control allocated {peak} bytes");

        // The recovering reader, whose ceiling is the same figure and for the
        // same reason. It has no streaminfo to narrow the bound with, so a
        // frame header it finds may legitimately declare the format's
        // largest block, and it trial-decodes every candidate sync point it
        // meets, so a byte run engineered to look like a run of maximal
        // frame headers is the shape that would blow it up if anything did.
        let mut headers = Vec::new();
        for _ in 0..64 {
            headers.extend_from_slice(&flac_self_describing_frame_header(65_535, 8));
        }
        let (result, peak) =
            peak_bytes(|| FlacRecovery::new(&headers).decode(&mut Vec::new()).err());
        assert!(
            result.is_some(),
            "sixty-four maximal frame headers with no bodies were accepted as audio"
        );
        report.push((
            "FLAC recovery over 64 headers each declaring 65,535 samples in 8 channels",
            peak,
        ));
        assert!(
            peak >= FLAC_FRAME_CEILING,
            "the recovery scan allocated only {peak}, so it never reached the decode buffer \
             and this case is measuring nothing"
        );
        assert!(
            peak <= FLAC_FRAME_CEILING + CEILING,
            "the recovery scan allocated {peak}, past the format's own bound"
        );

        // The same run behind a streaminfo permitting 4,096, supplied out of
        // band. The narrowing the whole-file reader gets from the block it
        // read, this reader gets from the block it was given.
        let narrow = flac_file(4_096, 8, 16, 0, &[], &[]);
        let info = *FlacReader::new(&narrow)
            .expect("a streaminfo permitting 4,096")
            .stream_info();
        let (result, peak) = peak_bytes(|| {
            FlacRecovery::with_stream_info(&headers, info)
                .decode(&mut Vec::new())
                .err()
        });
        assert!(
            result.is_some(),
            "maximal headers past a 4,096 maximum decoded"
        );
        report.push((
            "  the same run, with a streaminfo permitting 4,096 supplied",
            peak,
        ));
        assert!(
            peak <= CEILING,
            "a supplied maximum did not narrow the recovery bound: {peak}"
        );

        // -- The scan's cumulative cost, which a peak cannot see ----------
        //
        // Found by the pre-publish audit's fuzzing and kept here as its
        // regression. `FlacRecovery` trial-decodes a candidate frame at every
        // position whose two leading bytes could open one, and a maximal
        // header declares a four-megabyte buffer. Building a decoder per
        // candidate meant the scan allocated and zeroed four megabytes *per
        // candidate*: the peak stayed flat at one buffer, so every assertion
        // above passed, while the work grew with the input times the
        // format's largest frame. A megabyte of such headers took 330
        // seconds. The buffer is now reused across candidates, so the whole
        // scan allocates about one buffer however many candidates it meets.
        //
        // The bound is stated as a multiple of the ceiling rather than a byte
        // count, because it is a statement about *shape*: cumulative
        // allocation must not scale with the number of candidate positions.
        // Restoring a per-candidate decoder turns this red at roughly 300
        // times the bound while leaving every peak measurement above
        // untouched.
        let mut many = Vec::new();
        for _ in 0..1_024 {
            many.extend_from_slice(&flac_self_describing_frame_header(65_535, 8));
        }
        let candidates = 1_024;
        let (result, total) =
            total_bytes(|| FlacRecovery::new(&many).decode(&mut Vec::new()).err());
        assert!(
            result.is_some(),
            "1,024 maximal frame headers with no bodies were accepted as audio"
        );
        println!(
            "{total:>8} bytes TOTAL :: FLAC recovery over {candidates} maximal headers \
             (cumulative, not peak)"
        );
        assert!(
            total >= FLAC_FRAME_CEILING,
            "the scan allocated only {total} in total, so it never reached the decode buffer \
             and this case is measuring nothing"
        );
        assert!(
            total <= 4 * FLAC_FRAME_CEILING,
            "the recovery scan allocated {total} bytes in total over {candidates} candidate \
             sync points, which is per-candidate rather than per-scan"
        );
    }

    for (case, peak) in report {
        println!("{peak:>8} bytes peak :: {case}");
    }
}
