//! IFF-family container mechanics: the twelve-byte header, the chunk walk,
//! the pad byte and RF64's 64-bit size overrides.
//!
//! RIFF and AIFF's EA IFF 85 share one chunk structure, a four-CC, a 32-bit
//! size, a body, and a pad byte after an odd body, and differ only in the
//! byte order of the size field: little-endian in RIFF, big-endian in IFF.
//! [`ChunkWalker`] is therefore parameterised on [`ByteOrder`] rather than
//! duplicated per container, because the pad rule is exactly the kind of rule
//! that must not exist twice: two implementations of `size & 1` is two places
//! for one of them to be wrong, and a walk that forgets the pad reads garbage
//! as a chunk identifier. The one statement of the rule is [`pad_len`], and
//! every path in the crate that skips a pad byte, this walk and both
//! streaming state machines, calls it rather than restating it.
//!
//! Everything here is about *structure*. Nothing in this module knows what a
//! `fmt ` chunk means or that `WAVE` is the form it is looking at; that is
//! [`wav`](crate::wav)'s and [`aiff`](crate::aiff)'s job. Keeping them apart
//! is what lets the size arithmetic be reasoned about on its own, and this is
//! the first module in the crate that parses bytes it did not produce.
//!
//! # Every size in a file is a claim
//!
//! A chunk header declares its body length and a caller has no way to check it
//! except against the bytes that actually arrived. So:
//!
//! - **Nothing is allocated in proportion to a declared size.** The only
//!   allocation this module makes is the `ds64` override table, and its length
//!   is bounded by the `ds64` chunk body that was actually read, not by the
//!   count the body declares.
//! - **Every offset computation is in `u64` and checked.** `usize` is 32 bits
//!   on a 32-bit target, so `body_offset + declared_size` in `usize` wraps for a
//!   crafted size and turns a range check into a pass. decibri's own
//!   `parse_wav` computes `body + size > bytes.len()` in `usize`
//!   (`crates/decibri/src/file.rs:880`) and is reachable that way. Here the
//!   comparison is `size > available`, both `u64`, which cannot wrap whatever
//!   the file says.
//!
//! # The pad byte
//!
//! RIFF pads an odd-length chunk body to an even boundary. The pad byte is not
//! counted in the chunk's declared size and *is* counted in the enclosing
//! form's. A walk that forgets it lands one byte short of the next chunk header
//! and reads four bytes of garbage as a chunk identifier, which is why this is
//! one of the six cases the step-0 audit named.
//!
//! A file whose *last* chunk is odd-length and carries no pad byte is accepted:
//! the chunk itself is complete, and rejecting the file would be rejecting
//! audio that is all there.

use crate::codec::FourCc;
use crate::error::DecodeError;

/// The magic of a plain RIFF file.
pub(crate) const RIFF: FourCc = FourCc(*b"RIFF");

/// The magic of an RF64 file: RIFF's structure with 64-bit sizes beside it.
pub(crate) const RF64: FourCc = FourCc(*b"RF64");

/// The form type this crate reads.
pub(crate) const WAVE: FourCc = FourCc(*b"WAVE");

/// The RF64 chunk carrying the 64-bit sizes.
pub(crate) const DS64: FourCc = FourCc(*b"ds64");

/// The chunk describing the payload's encoding.
pub(crate) const FMT: FourCc = FourCc(*b"fmt ");

/// The chunk carrying the payload.
pub(crate) const DATA: FourCc = FourCc(*b"data");

/// The 32-bit size that means "the real size is in the `ds64` chunk".
///
/// RF64 leaves every oversized field at `0xFFFFFFFF` so that a plain RIFF
/// reader meeting the file fails on the size rather than reading four
/// gigabytes of nothing.
pub(crate) const SIZE_SENTINEL: u32 = u32::MAX;

/// How large a `ds64` chunk body this crate will read.
///
/// The override table is twelve bytes an entry, so this is 5,461 overridden
/// chunks: four orders of magnitude past the two a real RF64 file carries,
/// `data` and, rarely, an oversized `LIST`. The limit exists because
/// `ds64` is the one chunk whose body is buffered rather than skipped on the
/// streaming path, so its declared size governs a buffer.
pub(crate) const MAX_DS64_BYTES: u64 = 65_536;

/// The byte order a container stores its multi-byte fields in.
///
/// RIFF is little-endian; AIFF's EA IFF 85 is big-endian. This is the *only*
/// difference between their chunk structures, which is why the walk is
/// parameterised on it rather than written twice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ByteOrder {
    /// Least significant byte first: RIFF, RF64.
    Little,
    /// Most significant byte first: IFF, so AIFF and AIFF-C.
    Big,
}

/// How many pad bytes follow a chunk body of `size` bytes.
///
/// The one statement of the IFF-family pad rule: an odd body is followed by a
/// single pad byte that the declared size does not count. Every place in the
/// crate that steps over a pad calls this rather than restating `size & 1`.
pub(crate) const fn pad_len(size: u64) -> u64 {
    size & 1
}

/// A little-endian `u16` at `at`, or `None` when the slice is too short.
///
/// Returning an `Option` rather than indexing is not defensiveness for its own
/// sake: every one of these offsets is derived from a number the file chose.
pub(crate) fn u16_at(bytes: &[u8], at: usize) -> Option<u16> {
    let field = bytes.get(at..at.checked_add(2)?)?;
    Some(u16::from_le_bytes([field[0], field[1]]))
}

/// A big-endian `u16` at `at`, or `None` when the slice is too short.
pub(crate) fn u16_be_at(bytes: &[u8], at: usize) -> Option<u16> {
    let field = bytes.get(at..at.checked_add(2)?)?;
    Some(u16::from_be_bytes([field[0], field[1]]))
}

/// A little-endian `u32` at `at`, or `None` when the slice is too short.
pub(crate) fn u32_at(bytes: &[u8], at: usize) -> Option<u32> {
    let field = bytes.get(at..at.checked_add(4)?)?;
    Some(u32::from_le_bytes([field[0], field[1], field[2], field[3]]))
}

/// A big-endian `u32` at `at`, or `None` when the slice is too short.
pub(crate) fn u32_be_at(bytes: &[u8], at: usize) -> Option<u32> {
    let field = bytes.get(at..at.checked_add(4)?)?;
    Some(u32::from_be_bytes([field[0], field[1], field[2], field[3]]))
}

/// A `u32` at `at` in `order`, or `None` when the slice is too short.
pub(crate) fn u32_at_in(bytes: &[u8], at: usize, order: ByteOrder) -> Option<u32> {
    match order {
        ByteOrder::Little => u32_at(bytes, at),
        ByteOrder::Big => u32_be_at(bytes, at),
    }
}

/// A four-character code at `at`, or `None` when the slice is too short.
pub(crate) fn four_cc_at(bytes: &[u8], at: usize) -> Option<FourCc> {
    let field = bytes.get(at..at.checked_add(4)?)?;
    Some(FourCc([field[0], field[1], field[2], field[3]]))
}

/// An RF64 64-bit size, stored as a low `u32` followed by a high `u32`.
fn u64_at(bytes: &[u8], at: usize) -> Option<u64> {
    let low = u32_at(bytes, at)?;
    let high = u32_at(bytes, at.checked_add(4)?)?;
    Some(u64::from(low) | (u64::from(high) << 32))
}

/// What the twelve bytes at the start of a RIFF file declared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RiffHeader {
    /// `RIFF` or `RF64`.
    pub(crate) magic: FourCc,
    /// The form type: `WAVE` for the files this crate reads.
    pub(crate) form: FourCc,
}

/// How many bytes a RIFF header occupies.
pub(crate) const HEADER_BYTES: u64 = 12;

/// How many bytes a chunk header occupies.
pub(crate) const CHUNK_HEADER_BYTES: u64 = 8;

/// Reads the twelve-byte header at the start of `bytes`.
///
/// The size field at bytes 4..8 is read past and deliberately not returned. A
/// RIFF size that is too small would silently cut the chunk walk short and drop
/// audio that is present in the file, which is the failure class this crate
/// exists to avoid; a RIFF size that is too large cannot make the walk read
/// past the input, because the input length bounds it. Either way there is
/// nothing the walk can correctly do with the field, so it does not carry it
/// around and invite a later use.
///
/// # Errors
///
/// [`DecodeError::Truncated`] for an input shorter than twelve bytes, where
/// there is no tag to report and reporting one would mean inventing it, and
/// [`DecodeError::UnsupportedContainer`] for a magic that is neither `RIFF` nor
/// `RF64`.
pub(crate) fn read_riff_header(bytes: &[u8]) -> Result<RiffHeader, DecodeError> {
    let (Some(magic), Some(form)) = (four_cc_at(bytes, 0), four_cc_at(bytes, 8)) else {
        return Err(DecodeError::Truncated {
            expected: HEADER_BYTES,
            available: bytes.len() as u64,
        });
    };
    if magic != RIFF && magic != RF64 {
        return Err(DecodeError::UnsupportedContainer { tag: magic });
    }
    Ok(RiffHeader { magic, form })
}

/// Reads a `ds64` chunk body into the size overrides it declares.
///
/// The `data` size the chunk carries in its own field leads the returned table,
/// followed by whatever its override table names.
///
/// The `riffSize` and `sampleCount` fields are read past rather than kept.
/// `riffSize` is the 64-bit form of a field [`read_riff_header`] already
/// declines to act on, and `sampleCount` is a second statement of a frame count
/// the `data` chunk's own length gives exactly. Keeping either would mean
/// deciding what to do when it disagrees with the bytes that are actually
/// present, and the bytes win in every case, so there is nothing to decide.
///
/// # Errors
///
/// [`DecodeError::Malformed`] for a body under 28 bytes, and for a table length
/// that does not fit the body it was read from. That second check is the one
/// that matters: the count is a `u32` from the file, and reserving for it
/// before checking would be a 51 GiB allocation on a 28-byte chunk.
pub(crate) fn parse_ds64(body: &[u8], offset: u64) -> Result<Vec<(FourCc, u64)>, DecodeError> {
    /// Sixteen bytes of sizes, eight of sample count, four of table length.
    const FIXED_BYTES: usize = 28;
    /// Four bytes of identifier and eight of size, per override.
    const ENTRY_BYTES: u64 = 12;

    let (Some(data_size), Some(table_length)) = (u64_at(body, 8), u32_at(body, 24)) else {
        return Err(DecodeError::Malformed {
            expected: "a ds64 chunk of at least 28 bytes",
            offset,
        });
    };

    // The declared entry count against the body that actually arrived, in u64,
    // before a single byte is reserved for it.
    let declared = u64::from(table_length)
        .checked_mul(ENTRY_BYTES)
        .and_then(|table| table.checked_add(FIXED_BYTES as u64));
    if declared.is_none_or(|needed| needed > body.len() as u64) {
        return Err(DecodeError::Malformed {
            expected: "a ds64 table length that fits the ds64 chunk",
            offset,
        });
    }

    let count = table_length as usize;
    let mut overrides = Vec::with_capacity(count + 1);
    overrides.push((DATA, data_size));
    for entry in 0..count {
        let at = FIXED_BYTES + entry * ENTRY_BYTES as usize;
        let (Some(id), Some(size)) = (four_cc_at(body, at), u64_at(body, at + 4)) else {
            // Unreachable given the length check above, and cheaper to answer
            // than to argue about.
            return Err(DecodeError::Malformed {
                expected: "a ds64 table length that fits the ds64 chunk",
                offset,
            });
        };
        overrides.push((id, size));
    }
    Ok(overrides)
}

/// One chunk of a RIFF file, with its body already bounds-checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Chunk<'a> {
    /// The chunk's four-character identifier.
    pub(crate) id: FourCc,
    /// Byte offset of the chunk *header* from the start of the input.
    pub(crate) offset: u64,
    /// The chunk's body. Its length is the declared size, and it is a real
    /// subslice of the input rather than a promise about one.
    pub(crate) body: &'a [u8],
}

/// Walks the chunks of an IFF-family file (RIFF, RF64 or AIFF), applying the
/// pad byte and any `ds64` size overrides.
///
/// The walk is identical for every container; only the byte order of the size
/// field differs, and it is a constructor argument rather than a second
/// implementation. Yields `Err` once and then stops: a chunk that does not fit
/// the input ends the walk, because everything after it is at an offset the
/// file was wrong about.
#[derive(Debug)]
pub(crate) struct ChunkWalker<'a> {
    bytes: &'a [u8],
    offset: usize,
    order: ByteOrder,
    overrides: Vec<(FourCc, u64)>,
    finished: bool,
}

impl<'a> ChunkWalker<'a> {
    /// A walk over `bytes` starting immediately after the twelve-byte header,
    /// reading each chunk's size field in `order`.
    pub(crate) fn new(bytes: &'a [u8], order: ByteOrder) -> Self {
        Self {
            bytes,
            offset: HEADER_BYTES as usize,
            order,
            overrides: Vec::new(),
            finished: bytes.len() < HEADER_BYTES as usize,
        }
    }

    /// Installs the size overrides read from a `ds64` chunk.
    pub(crate) fn set_overrides(&mut self, overrides: Vec<(FourCc, u64)>) {
        self.overrides = overrides;
    }
}

/// The size a chunk header's declared size resolves to.
///
/// An override applies only where RF64 says it does: to a field left at
/// `0xFFFFFFFF`. A file that states a real 32-bit size *and* an override has
/// stated the size twice, and the one in the chunk header is the one a plain
/// RIFF reader would use, so it is the one used here.
pub(crate) fn resolve_size(overrides: &[(FourCc, u64)], id: FourCc, declared: u32) -> u64 {
    if declared == SIZE_SENTINEL {
        if let Some((_, size)) = overrides.iter().find(|(name, _)| *name == id) {
            return *size;
        }
    }
    u64::from(declared)
}

impl<'a> Iterator for ChunkWalker<'a> {
    type Item = Result<Chunk<'a>, DecodeError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        let total = self.bytes.len() as u64;
        let offset = self.offset as u64;

        // Fewer than eight bytes left is the end of the walk and not an error.
        // A trailing pad byte, or a stray byte from a writer that rounded
        // something up, is not a chunk that failed to parse.
        if total - offset < CHUNK_HEADER_BYTES {
            self.finished = true;
            return None;
        }

        let (Some(id), Some(declared)) = (
            four_cc_at(self.bytes, self.offset),
            u32_at_in(self.bytes, self.offset + 4, self.order),
        ) else {
            self.finished = true;
            return None;
        };

        let size = resolve_size(&self.overrides, id, declared);
        let body_start = self.offset + CHUNK_HEADER_BYTES as usize;
        let available = total - body_start as u64;

        // The whole point of the module, in one comparison. Both sides are
        // u64, so no declared size can wrap this into a pass, including on a
        // 32-bit target, where the `body + size` form decibri uses would.
        if size > available {
            self.finished = true;
            return Some(Err(DecodeError::Truncated {
                expected: size,
                available,
            }));
        }
        let Ok(size) = usize::try_from(size) else {
            // Unreachable: `size <= available <= bytes.len()`, and that is a
            // `usize`. Answered rather than asserted, because an assertion here
            // would be a panic on untrusted input.
            self.finished = true;
            return Some(Err(DecodeError::Truncated {
                expected: available,
                available,
            }));
        };

        let body = &self.bytes[body_start..body_start + size];
        // The pad byte, and a file that ends without one. `min` is what makes
        // the missing final pad acceptable rather than a walk past the end.
        self.offset = (body_start + size)
            .saturating_add(pad_len(size as u64) as usize)
            .min(self.bytes.len());
        Some(Ok(Chunk { id, offset, body }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Assembles a chunk: identifier, declared size, body, and the pad byte
    /// when the body is odd. Written here rather than reached for from the
    /// writer, so the walk is tested against bytes the walk did not produce.
    fn chunk(id: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(id);
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(body);
        if body.len() % 2 == 1 {
            out.push(0);
        }
        out
    }

    /// A RIFF header with a truthful size field over `chunks`.
    fn riff(magic: &[u8; 4], chunks: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(magic);
        out.extend_from_slice(&(4 + chunks.len() as u32).to_le_bytes());
        out.extend_from_slice(b"WAVE");
        out.extend_from_slice(chunks);
        out
    }

    fn ids(bytes: &[u8]) -> Vec<FourCc> {
        ChunkWalker::new(bytes, ByteOrder::Little)
            .map(|chunk| chunk.expect("walk").id)
            .collect()
    }

    #[test]
    fn a_short_input_is_truncated_rather_than_an_invented_tag() {
        for length in 0..12 {
            let error = read_riff_header(&vec![b'R'; length]).expect_err("must reject");
            assert!(
                matches!(error, DecodeError::Truncated { expected: 12, available } if available == length as u64),
                "{length} bytes: unexpected error: {error}"
            );
        }
    }

    #[test]
    fn an_unknown_magic_names_the_bytes_it_saw() {
        let bytes = riff(b"FORM", b"");
        let error = read_riff_header(&bytes).expect_err("must reject");
        assert!(
            matches!(error, DecodeError::UnsupportedContainer { tag } if tag == FourCc(*b"FORM")),
            "unexpected error: {error}"
        );
        // And both accepted magics are accepted.
        assert_eq!(
            read_riff_header(&riff(b"RIFF", b"")).expect("RIFF").magic,
            RIFF
        );
        assert_eq!(
            read_riff_header(&riff(b"RF64", b"")).expect("RF64").magic,
            RF64
        );
    }

    #[test]
    fn the_declared_riff_size_is_reported_and_never_acted_on() {
        let body = [chunk(b"one ", &[1, 2, 3, 4]), chunk(b"two ", &[5, 6])].concat();
        let truthful = riff(b"RIFF", &body);

        // Under-declared: a walk that trusted the field would drop `two `.
        let mut under = truthful.clone();
        under[4..8].copy_from_slice(&4u32.to_le_bytes());
        // Over-declared: a walk that trusted the field would read past the end.
        let mut over = truthful.clone();
        over[4..8].copy_from_slice(&u32::MAX.to_le_bytes());

        let expected = vec![FourCc(*b"one "), FourCc(*b"two ")];
        assert_eq!(ids(&truthful), expected, "matching");
        assert_eq!(ids(&under), expected, "under-declared");
        assert_eq!(ids(&over), expected, "over-declared");
        // And the header still parses in all three: the field is read past,
        // not validated.
        for bytes in [&truthful, &under, &over] {
            assert_eq!(read_riff_header(bytes).expect("header").magic, RIFF);
        }
    }

    #[test]
    fn an_odd_chunk_is_followed_by_a_pad_byte() {
        // Five-byte body, so the pad byte decides where `two ` starts.
        let body = [chunk(b"one ", b"INFOx"), chunk(b"two ", &[5, 6])].concat();
        let bytes = riff(b"RIFF", &body);
        assert_eq!(ids(&bytes), vec![FourCc(*b"one "), FourCc(*b"two ")]);

        // Without the pad the walk lands one byte early and reads a chunk
        // identifier out of the middle of the next header. This is the
        // assertion that the pad is load-bearing rather than cosmetic: the
        // walk either names a chunk that is not there or fails outright, and
        // either way it does not find `two `.
        let mut unpadded = bytes.clone();
        unpadded.remove(12 + 8 + 5);
        let walked: Vec<_> = ChunkWalker::new(&unpadded, ByteOrder::Little).collect();
        let found: Vec<FourCc> = walked
            .iter()
            .filter_map(|chunk| chunk.as_ref().ok().map(|chunk| chunk.id))
            .collect();
        assert!(
            !found.contains(&FourCc(*b"two ")),
            "the walk found `two ` without the pad byte: {found:?}"
        );
    }

    #[test]
    fn a_final_odd_chunk_without_its_pad_byte_is_still_read() {
        let mut bytes = riff(b"RIFF", &chunk(b"one ", b"odd"));
        bytes.pop(); // the pad byte
        let walked: Vec<_> = ChunkWalker::new(&bytes, ByteOrder::Little).collect();
        assert_eq!(walked.len(), 1);
        let chunk = walked[0].as_ref().expect("the chunk itself is complete");
        assert_eq!(chunk.body, b"odd");
    }

    #[test]
    fn a_chunk_declaring_more_than_the_file_holds_is_truncated() {
        let mut bytes = riff(b"RIFF", &chunk(b"data", &[0; 16]));
        // Four gigabytes declared inside a forty-byte file.
        bytes[16..20].copy_from_slice(&0xFFFF_FFF0u32.to_le_bytes());
        let mut walker = ChunkWalker::new(&bytes, ByteOrder::Little);
        let error = walker
            .next()
            .expect("a chunk header is present")
            .expect_err("must reject");
        assert!(
            matches!(
                error,
                DecodeError::Truncated {
                    expected: 0xFFFF_FFF0,
                    available: 16
                }
            ),
            "unexpected error: {error}"
        );
        // And the walk stops rather than resuming at a wrong offset.
        assert!(walker.next().is_none());
    }

    /// The arithmetic the whole module is arranged around. A `usize` sum wraps
    /// on a 32-bit target for these sizes; the `size > available` form does
    /// not, on any target.
    #[test]
    fn a_chunk_size_chosen_to_wrap_the_offset_arithmetic_is_rejected() {
        for declared in [u32::MAX, u32::MAX - 7, 0xFFFF_FFF8, 0x8000_0000] {
            let mut bytes = riff(b"RIFF", &chunk(b"data", &[0; 8]));
            bytes[16..20].copy_from_slice(&declared.to_le_bytes());
            let outcome: Vec<_> = ChunkWalker::new(&bytes, ByteOrder::Little).collect();
            assert_eq!(outcome.len(), 1, "declared {declared:#x}");
            assert!(
                matches!(outcome[0], Err(DecodeError::Truncated { .. })),
                "declared {declared:#x} did not reject: {:?}",
                outcome[0]
            );
        }
    }

    #[test]
    fn a_ds64_table_length_is_checked_against_the_body_before_it_is_reserved_for() {
        let mut body = Vec::new();
        body.extend_from_slice(&[0; 8]); // riffSize
        body.extend_from_slice(&256u64.to_le_bytes()); // dataSize
        body.extend_from_slice(&[0; 8]); // sampleCount
        body.extend_from_slice(&u32::MAX.to_le_bytes()); // tableLength

        let error = parse_ds64(&body, 12).expect_err("must reject");
        assert!(
            matches!(
                error,
                DecodeError::Malformed {
                    expected: "a ds64 table length that fits the ds64 chunk",
                    offset: 12
                }
            ),
            "unexpected error: {error}"
        );

        // A count that does fit is read, and the data size leads the table.
        body[24..28].copy_from_slice(&1u32.to_le_bytes());
        body.extend_from_slice(b"big ");
        body.extend_from_slice(&5_000_000_000u64.to_le_bytes());
        assert_eq!(
            parse_ds64(&body, 12).expect("a fitting table"),
            vec![(DATA, 256), (FourCc(*b"big "), 5_000_000_000)]
        );
    }

    #[test]
    fn a_short_ds64_body_is_malformed() {
        for length in 0..28 {
            let error = parse_ds64(&vec![0; length], 12).expect_err("must reject");
            assert!(
                matches!(
                    error,
                    DecodeError::Malformed {
                        expected: "a ds64 chunk of at least 28 bytes",
                        ..
                    }
                ),
                "{length} bytes: unexpected error: {error}"
            );
        }
    }

    #[test]
    fn an_override_applies_only_where_the_size_field_is_the_sentinel() {
        let overrides = [(DATA, 5_000_000_000)];
        assert_eq!(resolve_size(&overrides, DATA, SIZE_SENTINEL), 5_000_000_000);
        // A real size stated in the header is the one a plain RIFF reader would
        // use, so it is the one used here.
        assert_eq!(resolve_size(&overrides, DATA, 128), 128);
        // And an override for another chunk does not leak across.
        assert_eq!(
            resolve_size(&overrides, FMT, SIZE_SENTINEL),
            u64::from(SIZE_SENTINEL)
        );
    }

    #[test]
    fn the_walk_reports_where_each_chunk_started() {
        let body = [chunk(b"one ", &[1, 2]), chunk(b"two ", &[3, 4, 5, 6])].concat();
        let bytes = riff(b"RIFF", &body);
        let offsets: Vec<u64> = ChunkWalker::new(&bytes, ByteOrder::Little)
            .map(|chunk| chunk.expect("walk").offset)
            .collect();
        assert_eq!(offsets, vec![12, 22]);
    }

    #[test]
    fn a_walk_over_an_input_with_no_room_for_a_chunk_yields_nothing() {
        for length in 0..20 {
            let bytes = vec![0u8; length];
            let walked: Vec<_> = ChunkWalker::new(&bytes, ByteOrder::Little).collect();
            assert!(walked.is_empty(), "{length} bytes yielded a chunk");
        }
    }

    /// A big-endian chunk: identifier, big-endian declared size, body, pad.
    fn be_chunk(id: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(id);
        out.extend_from_slice(&(body.len() as u32).to_be_bytes());
        out.extend_from_slice(body);
        if body.len() % 2 == 1 {
            out.push(0);
        }
        out
    }

    /// The same walk over big-endian sizes: the IFF half of the shared walker.
    /// The bodies are chosen so that a walker reading the size field in the
    /// wrong byte order sees a wildly different size and cannot stumble onto
    /// the right answer.
    #[test]
    fn the_walk_reads_big_endian_sizes_when_told_to() {
        let body = [
            be_chunk(b"COMM", &[1, 2, 3, 4, 5]),
            be_chunk(b"SSND", &[6, 7]),
        ]
        .concat();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"FORM");
        bytes.extend_from_slice(&(4 + body.len() as u32).to_be_bytes());
        bytes.extend_from_slice(b"AIFF");
        bytes.extend_from_slice(&body);

        let walked: Vec<_> = ChunkWalker::new(&bytes, ByteOrder::Big)
            .map(|chunk| chunk.expect("walk"))
            .collect();
        assert_eq!(walked.len(), 2);
        assert_eq!(walked[0].id, FourCc(*b"COMM"));
        assert_eq!(walked[0].body, &[1, 2, 3, 4, 5]);
        assert_eq!(walked[1].id, FourCc(*b"SSND"));
        assert_eq!(walked[1].body, &[6, 7]);

        // The same bytes walked little-endian read the five-byte size as
        // 0x05000000 and reject it, which is the assertion that the parameter
        // is load-bearing.
        let wrong: Vec<_> = ChunkWalker::new(&bytes, ByteOrder::Little).collect();
        assert!(
            matches!(wrong[0], Err(DecodeError::Truncated { .. })),
            "a little-endian read of a big-endian size was accepted: {:?}",
            wrong[0]
        );
    }
}
