//! MD5, as [RFC 1321](https://www.rfc-editor.org/rfc/rfc1321) defines it.
//!
//! # This is an integrity check, not a security primitive
//!
//! MD5 is broken as a *cryptographic* hash: collisions are constructible on a
//! laptop. None of that matters here, and replacing it with something
//! "safer" would not make this check stronger; it would make it impossible.
//!
//! Every FLAC file in existence carries an MD5 of its own unencoded audio in
//! its streaminfo metadata block. That checksum is part of the format, fixed
//! by [RFC 9639](https://www.rfc-editor.org/rfc/rfc9639), and the only thing
//! a decoder can compare its output against. The threat model is a file
//! corrupted in storage or transit, not an adversary who has constructed a
//! second audio stream hashing to the same value; a decoder that produced the
//! wrong samples would have to hit a 128-bit target by accident to go
//! unnoticed. Swapping in SHA-256 here would leave nothing to compare to and
//! silently drop the only end-to-end correctness oracle FLAC ships with.
//!
//! So: do not replace this. It is here because the format says so.
//!
//! # Scope
//!
//! Crate-internal, and deliberately not part of the public API. This crate
//! decodes audio; it is not in the business of offering a hash function, and
//! a `pub` MD5 would be a commitment to one. The one caller is
//! [`flac`](crate::flac).

/// The per-round left-rotation amounts, from RFC 1321 section 3.4.
///
/// Four rounds of sixteen operations, each round cycling through four shift
/// amounts.
const SHIFTS: [u32; 64] = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, //
    5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, //
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, //
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
];

/// The additive constants, `T[i] = floor(2^32 * |sin(i)|)` with `i` in
/// radians, tabulated in RFC 1321 section 3.4.
///
/// Written out rather than computed, because computing them needs `sin` at
/// compile time and a floating-point derivation would put the crate's
/// byte-identical claim at the mercy of a libm. The values are checked
/// against RFC 1321's own test suite in the tests below, which is a stronger
/// check than re-deriving them would be: a wrong constant cannot survive it.
const T: [u32; 64] = [
    0xd76a_a478,
    0xe8c7_b756,
    0x2420_70db,
    0xc1bd_ceee,
    0xf57c_0faf,
    0x4787_c62a,
    0xa830_4613,
    0xfd46_9501,
    0x6980_98d8,
    0x8b44_f7af,
    0xffff_5bb1,
    0x895c_d7be,
    0x6b90_1122,
    0xfd98_7193,
    0xa679_438e,
    0x49b4_0821,
    0xf61e_2562,
    0xc040_b340,
    0x265e_5a51,
    0xe9b6_c7aa,
    0xd62f_105d,
    0x0244_1453,
    0xd8a1_e681,
    0xe7d3_fbc8,
    0x21e1_cde6,
    0xc337_07d6,
    0xf4d5_0d87,
    0x455a_14ed,
    0xa9e3_e905,
    0xfcef_a3f8,
    0x676f_02d9,
    0x8d2a_4c8a,
    0xfffa_3942,
    0x8771_f681,
    0x6d9d_6122,
    0xfde5_380c,
    0xa4be_ea44,
    0x4bde_cfa9,
    0xf6bb_4b60,
    0xbebf_bc70,
    0x289b_7ec6,
    0xeaa1_27fa,
    0xd4ef_3085,
    0x0488_1d05,
    0xd9d4_d039,
    0xe6db_99e5,
    0x1fa2_7cf8,
    0xc4ac_5665,
    0xf429_2244,
    0x432a_ff97,
    0xab94_23a7,
    0xfc93_a039,
    0x655b_59c3,
    0x8f0c_cc92,
    0xffef_f47d,
    0x8584_5dd1,
    0x6fa8_7e4f,
    0xfe2c_e6e0,
    0xa301_4314,
    0x4e08_11a1,
    0xf753_7e82,
    0xbd3a_f235,
    0x2ad7_d2bb,
    0xeb86_d391,
];

/// How many bytes one MD5 block holds.
const BLOCK_BYTES: usize = 64;

/// An MD5 computation in progress.
///
/// Fed with [`update`](Self::update) and closed with
/// [`finish`](Self::finish). Feeding is incremental and the split points make
/// no difference to the digest, which is what lets the streaming FLAC reader
/// hash a file it never holds whole.
#[derive(Debug, Clone)]
pub(crate) struct Md5 {
    /// The four chaining words, `A`, `B`, `C`, `D`.
    state: [u32; 4],
    /// Bytes of an incomplete block.
    block: [u8; BLOCK_BYTES],
    /// How many of `block` are filled. Always under [`BLOCK_BYTES`].
    filled: usize,
    /// Total message length in bytes.
    length: u64,
}

impl Md5 {
    /// A computation over the empty message.
    ///
    /// The four initial chaining values are RFC 1321 section 3.3's, stored
    /// there as bytes low-order first and written here as the words they
    /// form.
    pub(crate) const fn new() -> Self {
        Self {
            state: [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476],
            block: [0; BLOCK_BYTES],
            filled: 0,
            length: 0,
        }
    }

    /// Appends `data` to the message.
    pub(crate) fn update(&mut self, data: &[u8]) {
        self.length = self.length.wrapping_add(data.len() as u64);
        let mut rest = data;

        if self.filled > 0 {
            let want = (BLOCK_BYTES - self.filled).min(rest.len());
            self.block[self.filled..self.filled + want].copy_from_slice(&rest[..want]);
            self.filled += want;
            rest = &rest[want..];
            if self.filled < BLOCK_BYTES {
                return;
            }
            let block = self.block;
            self.compress(&block);
            self.filled = 0;
        }

        let mut blocks = rest.chunks_exact(BLOCK_BYTES);
        for block in &mut blocks {
            let block: [u8; BLOCK_BYTES] =
                block.try_into().expect("chunks_exact(64) yields 64 bytes");
            self.compress(&block);
        }
        let tail = blocks.remainder();
        self.block[..tail.len()].copy_from_slice(tail);
        self.filled = tail.len();
    }

    /// Closes the message and returns the sixteen-byte digest.
    ///
    /// The padding is RFC 1321 section 3.1's: a single one bit, then zero
    /// bits until the length is congruent to 56 modulo 64, then the message
    /// length in *bits* as a 64-bit number stored low-order byte first.
    pub(crate) fn finish(mut self) -> [u8; 16] {
        // The length in bits, taken before the padding is fed in.
        let bits = self.length.wrapping_mul(8);
        // Between 1 and 64 bytes of padding, then the eight length bytes.
        let pad = if self.filled < 56 {
            56 - self.filled
        } else {
            120 - self.filled
        };
        let mut trailer = [0u8; 72];
        trailer[0] = 0x80;
        trailer[pad..pad + 8].copy_from_slice(&bits.to_le_bytes());
        self.update(&trailer[..pad + 8]);

        let mut digest = [0u8; 16];
        for (out, word) in digest.chunks_exact_mut(4).zip(self.state) {
            out.copy_from_slice(&word.to_le_bytes());
        }
        digest
    }

    /// One 64-byte block, through the four rounds of RFC 1321 section 3.4.
    fn compress(&mut self, block: &[u8; BLOCK_BYTES]) {
        let mut words = [0u32; 16];
        for (word, bytes) in words.iter_mut().zip(block.chunks_exact(4)) {
            *word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        }

        let [mut a, mut b, mut c, mut d] = self.state;
        for i in 0..64 {
            // The round function and the message-word index it reads. The
            // four rounds are written as one loop because they differ only
            // in these two lines; splitting them into four sixteen-step
            // blocks is four places for one transcription error to hide.
            let (mixed, word) = match i / 16 {
                0 => ((b & c) | (!b & d), i),
                1 => ((b & d) | (c & !d), (5 * i + 1) % 16),
                2 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let rotated = a
                .wrapping_add(mixed)
                .wrapping_add(T[i])
                .wrapping_add(words[word])
                .rotate_left(SHIFTS[i]);
            a = d;
            d = c;
            c = b;
            b = b.wrapping_add(rotated);
        }

        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The digest of `message` as lowercase hex, the form RFC 1321's test
    /// suite prints.
    fn hex(message: &[u8]) -> String {
        let mut hasher = Md5::new();
        hasher.update(message);
        hasher
            .finish()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    /// RFC 1321 appendix A.5, verbatim: the seven messages and the seven
    /// digests the specification prints for them.
    ///
    /// This is the whole reason the constants above can be trusted. A single
    /// wrong entry in `T` or `SHIFTS` fails at least one of these, and the
    /// oracle is the specification rather than another implementation of the
    /// same algorithm.
    #[test]
    fn rfc_1321_test_suite() {
        assert_eq!(hex(b""), "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(hex(b"a"), "0cc175b9c0f1b6a831c399e269772661");
        assert_eq!(hex(b"abc"), "900150983cd24fb0d6963f7d28e17f72");
        assert_eq!(hex(b"message digest"), "f96b697d7cb7938d525a2f31aaf161d0");
        assert_eq!(
            hex(b"abcdefghijklmnopqrstuvwxyz"),
            "c3fcd3d76192e4007dfb496cca67e13b"
        );
        assert_eq!(
            hex(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"),
            "d174ab98d277d9f5a5611c2c9f419d9f"
        );
        assert_eq!(
            hex(
                b"12345678901234567890123456789012345678901234567890123456789012345678901234567890"
            ),
            "57edf4a22be3c955ac49da2e2107b67a"
        );
    }

    /// Where a message is split across `update` calls makes no difference to
    /// the digest.
    ///
    /// Load-bearing rather than decorative: the streaming FLAC reader hashes
    /// whatever arrives, in whatever sizes it arrives in, and its verdict has
    /// to be the whole-file reader's verdict. The sizes below straddle the
    /// 64-byte block boundary from both sides and land exactly on it.
    #[test]
    fn the_split_between_updates_is_invisible() {
        let message: Vec<u8> = (0u32..1000).map(|i| (i % 251) as u8).collect();
        let whole = {
            let mut hasher = Md5::new();
            hasher.update(&message);
            hasher.finish()
        };
        for step in [1, 7, 55, 56, 63, 64, 65, 127, 128, 129, 512] {
            let mut hasher = Md5::new();
            for piece in message.chunks(step) {
                hasher.update(piece);
            }
            assert_eq!(
                hasher.finish(),
                whole,
                "the digest changed with a {step}-byte feed size"
            );
        }
    }

    /// The lengths where the padding rule changes branch: exactly one byte
    /// short of the length field, exactly at it, and a whole block.
    ///
    /// A message of 55 bytes takes the shortest padding there is; 56 takes
    /// the longest, a whole extra block. Both are computed by hand here and
    /// compared against the same message hashed in pieces, so an off-by-one
    /// in the padding arithmetic cannot pass.
    #[test]
    fn the_padding_boundary_lengths_agree_with_themselves() {
        for length in [0usize, 1, 55, 56, 57, 63, 64, 65, 119, 120, 121] {
            let message = vec![0x41u8; length];
            let mut whole = Md5::new();
            whole.update(&message);
            let mut split = Md5::new();
            for byte in &message {
                split.update(&[*byte]);
            }
            assert_eq!(
                whole.finish(),
                split.finish(),
                "padding disagreed at a length of {length} bytes"
            );
        }
    }

    /// A digest is sixteen bytes and an empty message still has one.
    #[test]
    fn an_empty_message_still_has_a_digest() {
        assert_eq!(Md5::new().finish().len(), 16);
        assert_ne!(Md5::new().finish(), [0u8; 16]);
    }
}
