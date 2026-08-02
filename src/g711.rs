//! G.711 mu-law and A-law: the two ITU-T companding laws, both directions, and
//! a [`Decoder`] for headerless G.711 streams.
//!
//! # Why this is the first codec
//!
//! G.711's correct answer is published and finite. There are exactly 256 codes
//! per law, so the decode gate is not a sample of the input domain. It *is*
//! the input domain, and so is the 65,536-value encode gate. Nothing else this
//! crate will carry has that property, which makes this the cheapest place to
//! establish that the codec layer is sound.
//!
//! # A sample format, not a rate
//!
//! G.711 is overwhelmingly carried at 8 kHz in telephony and nothing in the
//! recommendation says so. The rate comes from the [`AudioSpec`] the caller
//! hands [`G711Decoder::new`]; nothing here defaults to 8000 or checks for it.
//!
//! # Where the tables come from
//!
//! The ITU-T G.711 recommendation defines both laws as explicit tables: eight
//! segments per law, sixteen intervals per segment, and a decoder output at each
//! interval's midpoint. The code below derives that geometry from the
//! recommendation's own structure (segment boundaries are powers of two, so a
//! segment index is a bit position and an interval index is a shift) rather
//! than transcribing a third-party implementation's tables. The crate is
//! Apache-2.0 and published, so reference data with an unestablished licence is
//! a real constraint rather than a formality.
//!
//! # Layering
//!
//! Decode is a G.711 code to `i16` through the table, then `i16` to `f32`
//! through [`i16_to_f32`]. Encode is the reverse: `f32` to `i16` through
//! [`f32_to_i16`], then `i16` to a code. There is deliberately **no** direct
//! code-to-`f32` path. One scaling rule in the crate is one opportunity to
//! disagree with decibri by a fraction of a bit; two would be two, and the rule
//! in [`sample`](crate::sample) has already been matched against decibri and
//! exhaustively tested.
//!
//! # Lossy, and idempotent anyway
//!
//! G.711 discards information: a round trip through it is not the identity, and
//! the crate does not pretend otherwise. What *does* hold is that a second pass
//! costs nothing: encoding, decoding and encoding again returns the code the
//! first encode produced. A codec failing that loses information on every pass
//! rather than only on the first.
//!
//! The one wrinkle is mu-law's second zero. The recommendation gives mu-law two
//! codes for silence, `0xFF` and `0x7F`, and the encoder can only ever emit
//! `0xFF` because a two's-complement zero has no sign to carry. `0x7F` decodes
//! to `0` like its twin and re-encodes to `0xFF`; nothing is lost, and it is the
//! one code in either law that is not its own fixed point. See
//! [`G711Law::code_to_linear`].

use crate::audio::AudioSpec;
use crate::decoder::Decoder;
use crate::error::DecodeError;
use crate::sample::{f32_to_i16, i16_to_f32};

/// How many decoded samples accumulate before [`feed`](Decoder::feed) stops
/// taking bytes.
///
/// The same figure [`PcmDecoder`](crate::PcmDecoder) holds, for the same
/// reason: a caller handing over a whole file in one call gets back-pressure and
/// a bounded decoder rather than a buffer the size of the file. G.711 is one
/// byte per sample, so this is also the largest number of bytes a single `feed`
/// will ever take.
const READY_LIMIT: usize = 65_536;

/// The bias G.711 adds to a mu-law magnitude before segmenting it, and takes
/// off again on the way out.
///
/// 33 in the 14-bit domain the recommendation writes mu-law in. It is what
/// makes the first mu-law segment start at 32 rather than at 0, and it is the
/// single constant that distinguishes the two laws' otherwise identical
/// segment geometry.
const MU_BIAS: i32 = 33;

/// One of the two G.711 companding laws.
///
/// Not `#[non_exhaustive]`, unlike [`SampleFormat`](crate::SampleFormat): G.711
/// defines exactly two laws and has since 1972. There is no third one to leave
/// room for, so a consumer gets an exhaustive match instead of a `_` arm they
/// can never reach.
///
/// mu-law is the North American and Japanese law; A-law is the European and
/// international one. They are not interchangeable, because decoding one as
/// the other produces audible, wrong audio rather than an error, so the law is
/// a constructor argument to [`G711Decoder`] and never inferred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum G711Law {
    /// mu-law, ITU-T G.711 clause 2: 14-bit magnitude, biased by 33.
    MuLaw,
    /// A-law, ITU-T G.711 clause 3: 13-bit magnitude, unbiased.
    ALaw,
}

impl G711Law {
    /// One G.711 code to the linear 16-bit sample the ITU table gives for it.
    ///
    /// The recommendation writes mu-law over a 14-bit magnitude and A-law over a
    /// 13-bit one; both are scaled here to the 16-bit linear domain the rest of
    /// the crate converts from, by 4 and by 8 respectively. That puts mu-law's
    /// range at +/-32124 and A-law's at +/-32256. Neither reaches [`i16::MAX`],
    /// which is a property of the laws and not of this implementation.
    ///
    /// Every one of the 256 codes is valid input. There is no such thing as a
    /// malformed G.711 code, which is why this returns a sample rather than a
    /// `Result`.
    ///
    /// ```
    /// use decibri_decode::G711Law;
    ///
    /// // Silence: 0xFF in mu-law, 0xD5 in A-law. A file of 0x00 bytes is loud
    /// // in both, which is the mistake this asymmetry exists to make visible.
    /// assert_eq!(G711Law::MuLaw.code_to_linear(0xFF), 0);
    /// assert_eq!(G711Law::ALaw.code_to_linear(0xD5), 8);
    /// assert_eq!(G711Law::MuLaw.code_to_linear(0x00), -32124);
    ///
    /// // mu-law's second zero decodes to silence like its twin.
    /// assert_eq!(G711Law::MuLaw.code_to_linear(0x7F), 0);
    /// ```
    pub const fn code_to_linear(self, code: u8) -> i16 {
        match self {
            Self::MuLaw => mu_law_to_linear(code),
            Self::ALaw => a_law_to_linear(code),
        }
    }

    /// One linear 16-bit sample to the G.711 code whose quantisation interval
    /// contains it.
    ///
    /// Total over the whole `i16` domain: every input has a code, and a
    /// magnitude past the top of the last segment clamps to the extreme code
    /// rather than wrapping.
    ///
    /// ```
    /// use decibri_decode::G711Law;
    ///
    /// assert_eq!(G711Law::MuLaw.linear_to_code(0), 0xFF);
    /// assert_eq!(G711Law::ALaw.linear_to_code(0), 0xD5);
    /// // Past the top of the last segment, and still the extreme code.
    /// assert_eq!(G711Law::MuLaw.linear_to_code(i16::MAX), 0x80);
    /// ```
    pub const fn linear_to_code(self, sample: i16) -> u8 {
        match self {
            Self::MuLaw => linear_to_mu_law(sample),
            Self::ALaw => linear_to_a_law(sample),
        }
    }

    /// Appends one `f32` per code in `codes` to `output`, and returns how many
    /// it appended.
    ///
    /// One code is one byte and one sample, so the return is `codes.len()`.
    /// There is no partial sample to hold and no code this can reject.
    ///
    /// `output` is appended to, never cleared, the same convention as
    /// [`SampleFormat::decode`](crate::SampleFormat::decode) and
    /// [`Decoder::decode`], so a decoder's output vector feeds a resampler with
    /// no copy in between.
    ///
    /// ```
    /// use decibri_decode::G711Law;
    ///
    /// let mut samples = Vec::new();
    /// assert_eq!(G711Law::ALaw.decode(&[0xD5, 0x55], &mut samples), 2);
    /// assert_eq!(samples, [8.0 / 32768.0, -8.0 / 32768.0]);
    /// ```
    pub fn decode(self, codes: &[u8], output: &mut Vec<f32>) -> usize {
        output.reserve(codes.len());
        for &code in codes {
            // Through the i16 table and then through the crate's one scaling
            // rule. Never straight from a code to an f32.
            output.push(i16_to_f32(self.code_to_linear(code)));
        }
        codes.len()
    }

    /// Appends one code per sample in `samples` to `output`, and returns how
    /// many bytes it appended.
    ///
    /// One sample is one byte, so the return is `samples.len()`. The `f32`
    /// reaches the table through [`f32_to_i16`], so a sample past full scale
    /// clamps exactly as it does for linear PCM.
    ///
    /// ```
    /// use decibri_decode::G711Law;
    ///
    /// let mut codes = Vec::new();
    /// assert_eq!(G711Law::MuLaw.encode(&[0.0, 1.0], &mut codes), 2);
    /// assert_eq!(codes, [0xFF, 0x80]);
    /// ```
    pub fn encode(self, samples: &[f32], output: &mut Vec<u8>) -> usize {
        output.reserve(samples.len());
        for &sample in samples {
            output.push(self.linear_to_code(f32_to_i16(sample)));
        }
        samples.len()
    }
}

/// A mu-law code to its 16-bit linear sample.
///
/// The code travels with every bit inverted, so the table is read off the
/// complement: bit 7 is the sign, bits 6-4 the segment, bits 3-0 the interval.
/// The magnitude is the interval's midpoint in the biased 14-bit domain, less
/// the bias, scaled to 16 bits.
const fn mu_law_to_linear(code: u8) -> i16 {
    let value = !code;
    let negative = value & 0x80 != 0;
    let segment = ((value >> 4) & 0x07) as i32;
    let interval = (value & 0x0F) as i32;

    // Segment `n` of mu-law covers biased magnitudes `2^(n+5)` upwards in steps
    // of `2^(n+1)`; sixteen such steps reach the start of segment `n+1`.
    let start = 1_i32 << (segment + 5);
    let step = 1_i32 << (segment + 1);
    let magnitude = start + step * interval + step / 2 - MU_BIAS;

    // 14-bit table to the 16-bit linear domain.
    let linear = magnitude * 4;
    if negative {
        (-linear) as i16
    } else {
        linear as i16
    }
}

/// A 16-bit linear sample to its mu-law code.
const fn linear_to_mu_law(sample: i16) -> u8 {
    let sample = sample as i32;
    let negative = sample < 0;
    // The 14-bit magnitude. The sign is taken first, so `>> 2` is the two bits
    // the law discards rather than an arithmetic shift of a negative value.
    let magnitude = if negative {
        -(sample >> 2)
    } else {
        sample >> 2
    };

    // Bias, then clamp to the top of the last segment. i16::MIN reduces to a
    // magnitude of 8192, which is past the 8158 the table reaches; the
    // recommendation sends everything above the last segment to the extreme
    // code.
    let biased = magnitude + MU_BIAS;
    let biased = if biased > 0x1FFF { 0x1FFF } else { biased };

    // The segment is the bit position, because the segment boundaries are
    // powers of two: biased magnitudes run 33..=8191, i.e. 2^5..=2^13 - 1.
    let segment = biased.ilog2() as i32 - 5;
    let start = 1_i32 << (segment + 5);
    let step = 1_i32 << (segment + 1);
    let interval = (biased - start) / step;

    let word = if negative { 0x80_u8 } else { 0 } | ((segment as u8) << 4) | interval as u8;
    // mu-law inverts every bit for transmission, which is what puts silence at
    // 0xFF rather than at 0x00.
    !word
}

/// An A-law code to its 16-bit linear sample.
///
/// A-law inverts the even bits for transmission, so the table is read off
/// `code ^ 0x55`: bit 7 is the sign (set for *positive*, the opposite way
/// round from mu-law's complemented word), bits 6-4 the segment, bits 3-0 the
/// interval.
const fn a_law_to_linear(code: u8) -> i16 {
    let value = code ^ 0x55;
    let positive = value & 0x80 != 0;
    let segment = ((value >> 4) & 0x07) as i32;
    let interval = (value & 0x0F) as i32;

    // A-law's first two segments share a step of 2: segment 0 covers 13-bit
    // magnitudes 0..=31 and segment 1 covers 32..=63, which is why the start of
    // segment `n` is `2^(n+4)` for every segment but the first, where it is 0.
    let start = if segment == 0 {
        0
    } else {
        1_i32 << (segment + 4)
    };
    let step = if segment == 0 { 2 } else { 1_i32 << segment };
    let magnitude = start + step * interval + step / 2;

    // 13-bit table to the 16-bit linear domain.
    let linear = magnitude * 8;
    if positive {
        linear as i16
    } else {
        (-linear) as i16
    }
}

/// A 16-bit linear sample to its A-law code.
const fn linear_to_a_law(sample: i16) -> u8 {
    let sample = sample as i32;
    let positive = sample >= 0;
    // The 13-bit magnitude. The negative half folds about -1/2 rather than about
    // 0, because the magnitude of a negative sample is `(-1 - x) >> 3`, which
    // makes A-law's two halves symmetric and every one of its 256 codes its own
    // fixed point under a second encode.
    let magnitude = if positive {
        sample >> 3
    } else {
        (-1 - sample) >> 3
    };

    // 13 bits from a 16-bit input never exceeds 4095, which is inside the last
    // segment, so A-law needs no clamp: unlike mu-law, its table covers the
    // whole reduced domain.
    let segment = if magnitude < 32 {
        0
    } else {
        magnitude.ilog2() as i32 - 4
    };
    let start = if segment == 0 {
        0
    } else {
        1_i32 << (segment + 4)
    };
    let step = if segment == 0 { 2 } else { 1_i32 << segment };
    let interval = (magnitude - start) / step;

    let word = if positive { 0x80_u8 } else { 0 } | ((segment as u8) << 4) | interval as u8;
    // A-law inverts the even bits for transmission, which puts silence at 0xD5.
    word ^ 0x55
}

/// Decodes a headerless G.711 stream: one byte per sample in the law given at
/// construction, samples out at an [`AudioSpec`] the caller states.
///
/// Headerless G.711 is what an RTP payload and a raw telephony capture carry, so
/// nothing in the stream declares its rate, layout or law: they arrive out of
/// band and are given at construction. Both are then fixed for the life of the
/// instance, as [`Decoder::output_spec`] requires.
///
/// # Partial frames
///
/// One byte is one whole sample, so unlike [`PcmDecoder`](crate::PcmDecoder)
/// there is never a partial *sample* to hold. There is still a partial *frame*:
/// a stereo stream that ends after an odd number of bytes ended mid-frame, and
/// reporting that as a clean end would hand the caller a buffer whose length is
/// not a whole number of frames. [`flush`](Decoder::flush) is where that becomes
/// [`DecodeError::Truncated`].
///
/// # Example
///
/// ```
/// use decibri_decode::{AudioSpec, Decoder, G711Decoder, G711Law};
///
/// // The rate is the caller's to state. 8 kHz is the usual one for telephony
/// // and nothing here assumes it.
/// let mut decoder = G711Decoder::new(G711Law::MuLaw, AudioSpec::mono(16_000));
/// let mut samples = Vec::new();
///
/// decoder.feed(&[0xFF, 0x80])?;
/// decoder.decode(&mut samples)?;
/// decoder.flush(&mut samples)?;
/// assert_eq!(samples, [0.0, 32124.0 / 32768.0]);
/// # Ok::<(), decibri_decode::DecodeError>(())
/// ```
#[derive(Debug)]
pub struct G711Decoder {
    /// Which companding law the incoming bytes are in.
    law: G711Law,
    /// The rate and layout of the samples produced.
    spec: AudioSpec,
    /// Samples decoded but not yet handed to a caller.
    ready: Vec<f32>,
    /// Samples produced since construction or the last `reset`, which is what
    /// says whether the stream is on a frame boundary.
    produced: u64,
    /// Set by `flush`: end of stream until `reset`.
    finished: bool,
}

impl G711Decoder {
    /// A decoder for `law` codes producing samples at `spec`.
    ///
    /// Nothing is validated because nothing can be: a headerless stream carries
    /// no claim to check the arguments against. `spec` is the assertion, and so
    /// is `law`: decoding mu-law bytes as A-law is wrong but not detectable.
    pub const fn new(law: G711Law, spec: AudioSpec) -> Self {
        Self {
            law,
            spec,
            ready: Vec::new(),
            produced: 0,
            finished: false,
        }
    }

    /// The companding law this decoder reads.
    pub const fn law(&self) -> G711Law {
        self.law
    }

    /// How many bytes are held awaiting the rest of their frame.
    ///
    /// `0` means the stream is on a frame boundary and could be cut here without
    /// loss, the same meaning the figure carries on
    /// [`PcmDecoder::buffered_bytes`](crate::PcmDecoder::buffered_bytes), and
    /// the same figure [`flush`](Decoder::flush) reports as `available` when it
    /// rejects a truncated stream.
    ///
    /// There is never a partial sample here, so this counts only whole samples
    /// sitting past the last frame boundary, one byte each.
    pub fn buffered_bytes(&self) -> usize {
        let channels = u64::from(self.spec.channels).max(1);
        (self.produced % channels) as usize
    }

    /// How many bytes one whole frame occupies.
    ///
    /// A spec with no channels has no frame at all, so one sample stands in for
    /// one: it keeps the truncation report meaningful instead of dividing by
    /// zero, matching [`PcmDecoder`](crate::PcmDecoder).
    fn frame_bytes(&self) -> usize {
        usize::from(self.spec.channels).max(1)
    }
}

impl Decoder for G711Decoder {
    fn output_spec(&self) -> AudioSpec {
        self.spec
    }

    fn feed(&mut self, input: &[u8]) -> Result<usize, DecodeError> {
        if self.finished {
            return Ok(0);
        }
        // One byte is one sample, so room in samples is room in bytes. At zero,
        // the caller has to drain before feeding again.
        let room = READY_LIMIT.saturating_sub(self.ready.len());
        let taken = input.len().min(room);
        self.produced += self.law.decode(&input[..taken], &mut self.ready) as u64;
        Ok(taken)
    }

    fn decode(&mut self, output: &mut Vec<f32>) -> Result<usize, DecodeError> {
        if self.finished {
            return Ok(0);
        }
        let produced = self.ready.len();
        output.append(&mut self.ready);
        Ok(produced)
    }

    fn flush(&mut self, output: &mut Vec<f32>) -> Result<usize, DecodeError> {
        if self.finished {
            return Ok(0);
        }
        self.finished = true;
        let held = self.buffered_bytes();
        if held > 0 {
            // The one place a partial frame is an error. A caller following the
            // documented loop has already drained everything whole through
            // `decode`, so nothing complete is lost with the rejection.
            return Err(DecodeError::Truncated {
                expected: self.frame_bytes() as u64,
                available: held as u64,
            });
        }
        let produced = self.ready.len();
        output.append(&mut self.ready);
        Ok(produced)
    }

    fn reset(&mut self) {
        self.ready.clear();
        self.produced = 0;
        self.finished = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::AudioBuffer;

    const BOTH: [G711Law; 2] = [G711Law::MuLaw, G711Law::ALaw];

    // -- The reference, built from the recommendation's tables ------------
    //
    // Everything below this line is an independent reading of ITU-T G.711.
    // The implementation above derives each segment's start and step from a bit
    // position, because the boundaries happen to be powers of two; the reference
    // carries the same geometry as literal data, read off the recommendation's
    // tables. A shift written one place out above is therefore not mirrored
    // here, which is the whole point of having a reference at all.

    /// A-law segment geometry in the 13-bit domain the recommendation writes it
    /// in: the first magnitude each of the eight segments covers, and the width
    /// of one of its sixteen intervals.
    ///
    /// The first two segments share a step of 2. That is the ITU table, not a
    /// typo, and it is why A-law resolves small signals as finely as it does.
    const A_LAW_SEGMENTS: [(i32, i32); 8] = [
        (0, 2),
        (32, 2),
        (64, 4),
        (128, 8),
        (256, 16),
        (512, 32),
        (1024, 64),
        (2048, 128),
    ];

    /// mu-law segment geometry in the *biased* 14-bit domain: the recommendation
    /// adds 33 before segmenting, which is why the first segment starts at 32
    /// rather than at 0 and why mu-law's steps are one power of two larger than
    /// A-law's throughout.
    const MU_LAW_SEGMENTS: [(i32, i32); 8] = [
        (32, 2),
        (64, 4),
        (128, 8),
        (256, 16),
        (512, 32),
        (1024, 64),
        (2048, 128),
        (4096, 256),
    ];

    /// The decoder output the ITU table gives for `code`, in the 16-bit linear
    /// domain.
    fn reference_decode(law: G711Law, code: u8) -> i16 {
        let (segments, value) = match law {
            // mu-law transmits the complement of the table word.
            G711Law::MuLaw => (MU_LAW_SEGMENTS, !code),
            // A-law transmits the table word with its even bits inverted.
            G711Law::ALaw => (A_LAW_SEGMENTS, code ^ 0x55),
        };
        let (start, step) = segments[usize::from((value >> 4) & 0x07)];
        let interval = i32::from(value & 0x0F);
        let midpoint = start + step * interval + step / 2;
        let (magnitude, negative) = match law {
            // 14-bit, biased, scaled by 4. The sign bit of the complemented word
            // is set for negative.
            G711Law::MuLaw => ((midpoint - 33) * 4, value & 0x80 != 0),
            // 13-bit, unbiased, scaled by 8. The sign bit is set for *positive*,
            // the opposite way round.
            G711Law::ALaw => (midpoint * 8, value & 0x80 == 0),
        };
        if negative {
            -magnitude as i16
        } else {
            magnitude as i16
        }
    }

    /// The range of 16-bit linear inputs the recommendation assigns to `code`,
    /// or `None` for a code the encoder can never emit.
    ///
    /// Derived from the segment table and from the domain reduction each law
    /// applies (mu-law drops two bits and takes the magnitude of a negative
    /// sample as `-(x >> 2)`, A-law drops three and folds about `-1/2`), not
    /// from the encoder under test.
    fn reference_input_range(law: G711Law, code: u8) -> Option<(i16, i16)> {
        let (segments, value) = match law {
            G711Law::MuLaw => (MU_LAW_SEGMENTS, !code),
            G711Law::ALaw => (A_LAW_SEGMENTS, code ^ 0x55),
        };
        let segment = usize::from((value >> 4) & 0x07);
        let interval = i32::from(value & 0x0F);
        let (start, step) = segments[segment];

        let positive = match law {
            G711Law::MuLaw => value & 0x80 == 0,
            G711Law::ALaw => value & 0x80 != 0,
        };
        // The reduced-domain magnitudes this code stands for.
        let low = match law {
            G711Law::MuLaw => start + step * interval - 33,
            G711Law::ALaw => start + step * interval,
        };
        let high = low + step - 1;
        // Back out to 16 bits, through each law's own sign convention.
        let (mut lo, mut hi) = match (law, positive) {
            (G711Law::MuLaw, true) => (low * 4, high * 4 + 3),
            (G711Law::MuLaw, false) => (-(high * 4), -(low * 4) + 3),
            (G711Law::ALaw, true) => (low * 8, high * 8 + 7),
            (G711Law::ALaw, false) => (-(high * 8) - 8, -(low * 8) - 1),
        };
        // The topmost interval absorbs everything past the last segment: a
        // magnitude the table does not reach clamps to the extreme code.
        if segment == 7 && interval == 15 {
            if positive {
                hi = i32::from(i16::MAX);
            } else {
                lo = i32::from(i16::MIN);
            }
        }
        // A code's range only covers its own half of the domain. mu-law's second
        // zero is the code this rules out entirely: its interval lies wholly on
        // the positive side while its sign bit says negative.
        let (floor, ceiling) = if positive {
            (0, i32::from(i16::MAX))
        } else {
            (i32::from(i16::MIN), -1)
        };
        let (lo, hi) = (lo.max(floor), hi.min(ceiling));
        if lo > hi {
            None
        } else {
            Some((lo as i16, hi as i16))
        }
    }

    /// Every 16-bit input mapped to its code, painted one code at a time from
    /// [`reference_input_range`].
    ///
    /// Building it this way proves the intervals tile the domain: a value
    /// claimed twice or not at all fails here rather than showing up as a
    /// mysterious mismatch later.
    fn reference_encode_table(law: G711Law) -> Vec<u8> {
        let mut table: Vec<Option<u8>> = vec![None; 65_536];
        for code in 0..=u8::MAX {
            let Some((lo, hi)) = reference_input_range(law, code) else {
                continue;
            };
            for sample in lo..=hi {
                let slot =
                    &mut table[usize::try_from(i32::from(sample) + 32_768).expect("in range")];
                assert!(
                    slot.is_none(),
                    "{law:?}: {sample} is claimed by two codes, 0x{:02x} and 0x{code:02x}",
                    slot.unwrap_or_default()
                );
                *slot = Some(code);
            }
        }
        table
            .into_iter()
            .enumerate()
            .map(|(index, code)| {
                code.unwrap_or_else(|| {
                    panic!("{law:?}: no code covers {}", index as i32 - 32_768);
                })
            })
            .collect()
    }

    fn reference_encode(table: &[u8], sample: i16) -> u8 {
        table[usize::try_from(i32::from(sample) + 32_768).expect("in range")]
    }

    /// Drives a decoder to completion over `input` split into `chunk`-byte
    /// pieces, the way the trait documents a caller should.
    fn drive(
        decoder: &mut dyn Decoder,
        input: &[u8],
        chunk: usize,
    ) -> Result<AudioBuffer, DecodeError> {
        let mut samples = Vec::new();
        for piece in input.chunks(chunk) {
            let mut offset = 0;
            while offset < piece.len() {
                offset += decoder.feed(&piece[offset..])?;
                while decoder.decode(&mut samples)? > 0 {}
            }
        }
        decoder.flush(&mut samples)?;
        Ok(AudioBuffer::from_samples(decoder.output_spec(), samples))
    }

    /// Every code, in order, as a byte slice.
    fn all_codes() -> Vec<u8> {
        (0..=u8::MAX).collect()
    }

    // -- Gate 5: exhaustive decode against the ITU table -------------------
    //
    // Path exercised: `code_to_linear`, i.e. the table itself. This is the
    // complete input domain, all 256 codes per law, and it says nothing at
    // all about the byte path, which gate 8 covers separately.

    /// All 256 mu-law codes against the ITU table, exact.
    #[test]
    fn every_mu_law_code_decodes_to_the_itu_table() {
        for code in 0..=u8::MAX {
            assert_eq!(
                G711Law::MuLaw.code_to_linear(code),
                reference_decode(G711Law::MuLaw, code),
                "mu-law 0x{code:02x}"
            );
        }
    }

    /// All 256 A-law codes against the ITU table, exact.
    #[test]
    fn every_a_law_code_decodes_to_the_itu_table() {
        for code in 0..=u8::MAX {
            assert_eq!(
                G711Law::ALaw.code_to_linear(code),
                reference_decode(G711Law::ALaw, code),
                "A-law 0x{code:02x}"
            );
        }
    }

    /// The published values, written as literals rather than derived from
    /// anything.
    ///
    /// A reference that shares a misreading with the implementation agrees with
    /// it perfectly, so these anchors are the outside check on both: silence,
    /// the extremes and the two laws' well-known ranges.
    #[test]
    fn the_published_anchor_values_hold() {
        // Silence. A G.711 file of 0x00 bytes is full-scale noise in both laws,
        // which is the mistake this asymmetry exists to make visible.
        assert_eq!(G711Law::MuLaw.code_to_linear(0xFF), 0);
        assert_eq!(G711Law::ALaw.code_to_linear(0xD5), 8);
        assert_eq!(G711Law::ALaw.code_to_linear(0x55), -8);

        // The ranges: mu-law reaches +/-32124, A-law +/-32256. Neither reaches
        // i16::MAX, and they do not reach the same place as each other.
        assert_eq!(G711Law::MuLaw.code_to_linear(0x80), 32_124);
        assert_eq!(G711Law::MuLaw.code_to_linear(0x00), -32_124);
        assert_eq!(G711Law::ALaw.code_to_linear(0xAA), 32_256);
        assert_eq!(G711Law::ALaw.code_to_linear(0x2A), -32_256);

        // The step at the bottom of each law: A-law's first segment steps by 16
        // in the 16-bit domain, mu-law's by 8.
        assert_eq!(G711Law::ALaw.code_to_linear(0xD4), 24);
        assert_eq!(G711Law::MuLaw.code_to_linear(0xFE), 8);

        // And the two laws genuinely differ: decoding one as the other is
        // wrong, not merely differently scaled.
        assert_ne!(
            G711Law::MuLaw.code_to_linear(0x00),
            G711Law::ALaw.code_to_linear(0x00)
        );
    }

    // -- Gate 6: exhaustive encode against the ITU table -------------------
    //
    // Path exercised: `linear_to_code`. Also the complete input domain: all
    // 65,536 i16 values per law.

    /// The reference's own gate. If the 256 quantisation intervals did not tile
    /// the 16-bit domain exactly once, the reference would be wrong and every
    /// encode assertion built on it would be worthless.
    #[test]
    fn the_reference_intervals_tile_the_whole_16_bit_domain() {
        // `reference_encode_table` panics on an overlap or a gap; reaching the
        // end is the assertion.
        for law in BOTH {
            let table = reference_encode_table(law);
            assert_eq!(table.len(), 65_536);
        }

        // A-law can emit all 256 codes. mu-law can emit 255: its second zero is
        // a negative code whose interval lies wholly on the positive side, so
        // no input reaches it.
        let emitted = |law| {
            (0..=u8::MAX)
                .filter(|&code| reference_input_range(law, code).is_some())
                .count()
        };
        assert_eq!(emitted(G711Law::ALaw), 256);
        assert_eq!(emitted(G711Law::MuLaw), 255);
        assert!(reference_input_range(G711Law::MuLaw, 0x7F).is_none());
    }

    /// All 65,536 `i16` values per law against the reference, exact.
    #[test]
    fn every_i16_encodes_to_the_itu_code_in_both_laws() {
        for law in BOTH {
            let table = reference_encode_table(law);
            for sample in i16::MIN..=i16::MAX {
                assert_eq!(
                    law.linear_to_code(sample),
                    reference_encode(&table, sample),
                    "{law:?} on {sample}"
                );
            }
        }
    }

    /// The extremes and the clamp, stated separately from the sweep so the
    /// failure names the case rather than the first of 65,536.
    #[test]
    fn the_encoder_is_total_and_clamps_rather_than_wrapping() {
        for law in BOTH {
            // The extremes land on the extreme codes, not somewhere in the
            // middle of the table.
            assert_eq!(
                law.code_to_linear(law.linear_to_code(i16::MAX)).signum(),
                1,
                "{law:?} sent i16::MAX negative"
            );
            assert_eq!(
                law.code_to_linear(law.linear_to_code(i16::MIN)).signum(),
                -1,
                "{law:?} sent i16::MIN positive"
            );
            // Silence encodes to silence.
            assert!(
                law.code_to_linear(law.linear_to_code(0)).abs() <= 8,
                "{law:?} did not put silence at the bottom of the table"
            );
        }
        assert_eq!(G711Law::MuLaw.linear_to_code(0), 0xFF);
        assert_eq!(G711Law::ALaw.linear_to_code(0), 0xD5);
        assert_eq!(G711Law::MuLaw.linear_to_code(i16::MAX), 0x80);
        assert_eq!(G711Law::MuLaw.linear_to_code(i16::MIN), 0x00);
        assert_eq!(G711Law::ALaw.linear_to_code(i16::MAX), 0xAA);
        assert_eq!(G711Law::ALaw.linear_to_code(i16::MIN), 0x2A);
    }

    // -- Gate 7: idempotence -----------------------------------------------
    //
    // Path exercised: `code_to_linear` and `linear_to_code` together.

    /// G.711 is lossy, so a round trip is not the identity. A *second* pass
    /// must cost nothing: encoding, decoding and encoding again returns the
    /// code the first encode produced. A codec failing this loses information
    /// on every pass rather than only on the first.
    ///
    /// The one exception is mu-law's second zero, `0x7F`. The recommendation
    /// gives mu-law two codes for silence and the encoder can only emit one of
    /// them, because a two's-complement zero has no sign to carry. `0x7F` is
    /// therefore not in the encoder's image, and re-encoding it yields its twin
    /// `0xFF`. Nothing is lost, since both decode to `0`, but it is a real
    /// departure from the property, stated here rather than smoothed over.
    #[test]
    fn re_encoding_a_decoded_code_returns_it_except_for_mu_laws_second_zero() {
        for law in BOTH {
            for code in 0..=u8::MAX {
                let linear = law.code_to_linear(code);
                let again = law.linear_to_code(linear);

                // The information-preserving form, which holds for all 256
                // codes of both laws with no exception at all.
                assert_eq!(
                    law.code_to_linear(again),
                    linear,
                    "{law:?} 0x{code:02x} lost information on a second pass"
                );

                if law == G711Law::MuLaw && code == 0x7F {
                    assert_eq!(again, 0xFF, "mu-law's second zero must fold onto 0xFF");
                    assert_eq!(linear, 0, "mu-law's second zero must be silence");
                } else {
                    assert_eq!(
                        again, code,
                        "{law:?} 0x{code:02x} is not its own fixed point"
                    );
                }
            }
        }
    }

    /// Idempotence from the other end: for every `i16`, encoding twice with a
    /// decode in between gives the same code. This is the property a repeated
    /// transcode actually depends on.
    #[test]
    fn a_second_encode_of_any_sample_returns_the_first_code() {
        for law in BOTH {
            for sample in i16::MIN..=i16::MAX {
                let first = law.linear_to_code(sample);
                let again = law.linear_to_code(law.code_to_linear(first));
                assert_eq!(again, first, "{law:?} on {sample}");
            }
        }
    }

    // -- Gate 8: the byte path, exercised separately from the table --------
    //
    // Paths exercised: `G711Law::decode`, `G711Law::encode` and
    // `G711Decoder::{feed, decode, flush}`. These gates would pass with the
    // table completely broken and the table gates would pass with these
    // completely broken, which is exactly why both exist. Step 2's negative
    // control lived in a byte path that its strongest table gate never touched.

    /// The slice decode API against the scalar table, code by code, at several
    /// chunk sizes and offsets. Any reordering, duplication or skipped byte in
    /// the loop shows up here and nowhere in gate 5.
    #[test]
    fn the_slice_decode_api_matches_the_scalar_path() {
        let codes = all_codes();
        for law in BOTH {
            let expected: Vec<f32> = codes
                .iter()
                .map(|&code| i16_to_f32(law.code_to_linear(code)))
                .collect();

            // In one call.
            let mut whole = Vec::new();
            assert_eq!(law.decode(&codes, &mut whole), codes.len());
            assert_eq!(whole, expected, "{law:?} in one call");

            // And split at every size that lands on a different set of
            // boundaries, appending into the same vector each time.
            for chunk in [1, 2, 3, 5, 7, 17, 128, 255] {
                let mut pieced = Vec::new();
                for piece in codes.chunks(chunk) {
                    assert_eq!(law.decode(piece, &mut pieced), piece.len());
                }
                assert_eq!(pieced, expected, "{law:?} in {chunk}-byte pieces");
            }
        }
    }

    /// The slice encode API against the scalar path, sample by sample.
    #[test]
    fn the_slice_encode_api_matches_the_scalar_path() {
        // A signal built from integer arithmetic, so it is the same everywhere,
        // and reaching past full scale so the clamp is on the path too.
        let samples: Vec<f32> = (-600..=600)
            .map(|i| i as f32 / 512.0)
            .chain([f32::NAN, f32::INFINITY, f32::NEG_INFINITY])
            .collect();
        for law in BOTH {
            let expected: Vec<u8> = samples
                .iter()
                .map(|&sample| law.linear_to_code(f32_to_i16(sample)))
                .collect();

            let mut whole = Vec::new();
            assert_eq!(law.encode(&samples, &mut whole), samples.len());
            assert_eq!(whole, expected, "{law:?} in one call");

            for chunk in [1, 3, 7, 64] {
                let mut pieced = Vec::new();
                for piece in samples.chunks(chunk) {
                    assert_eq!(law.encode(piece, &mut pieced), piece.len());
                }
                assert_eq!(pieced, expected, "{law:?} in {chunk}-sample pieces");
            }
        }
    }

    /// The decoder's byte path, `feed` and `decode` on a slice, against the
    /// scalar path, at several feed sizes.
    ///
    /// This is the gate that would have caught step 2's defect: the exhaustive
    /// table gate above never touches `feed`, and `feed` could drop, duplicate
    /// or reorder bytes with the table perfectly correct.
    #[test]
    fn the_decoder_byte_path_matches_the_scalar_path_at_every_chunk_size() {
        let codes = all_codes();
        for law in BOTH {
            let expected: Vec<f32> = codes
                .iter()
                .map(|&code| i16_to_f32(law.code_to_linear(code)))
                .collect();
            for chunk in [1, 2, 3, 5, 7, 17, 128, 255, 256, 1024] {
                let mut decoder = G711Decoder::new(law, AudioSpec::mono(16_000));
                let decoded = drive(&mut decoder, &codes, chunk).expect("decode");
                assert_eq!(
                    decoded.samples(),
                    expected.as_slice(),
                    "{law:?} changed with a {chunk}-byte feed size"
                );
            }
        }
    }

    /// The bytes arrive in the order they were fed. A path that decoded each
    /// chunk correctly but assembled the chunks wrongly passes every assertion
    /// about individual codes and fails this one.
    #[test]
    fn the_decoder_preserves_byte_order_across_feeds() {
        let mut decoder = G711Decoder::new(G711Law::ALaw, AudioSpec::mono(8_000));
        let mut out = Vec::new();
        for code in [0xD5_u8, 0x2A, 0xAA, 0x55] {
            assert_eq!(decoder.feed(&[code]).expect("feed"), 1);
            assert_eq!(decoder.decode(&mut out).expect("decode"), 1);
        }
        decoder.flush(&mut out).expect("flush");
        assert_eq!(
            out,
            [
                8.0 / 32768.0,
                -32_256.0 / 32768.0,
                32_256.0 / 32768.0,
                -8.0 / 32768.0
            ]
        );
    }

    /// Back-pressure: a caller that hands over more than the decoder will hold
    /// gets a short return and has to drain. Nothing is lost and the loop still
    /// terminates.
    ///
    /// This is the only gate that exercises `feed`'s short-return path, which
    /// makes it the only gate standing between a caller and silently dropped
    /// audio on a whole-file feed.
    #[test]
    fn a_full_decoder_applies_back_pressure_instead_of_growing() {
        let law = G711Law::MuLaw;
        let mut decoder = G711Decoder::new(law, AudioSpec::mono(16_000));
        // Three times what the decoder will hold, so the short return is
        // unavoidable rather than incidental.
        let bytes: Vec<u8> = (0..READY_LIMIT * 3).map(|i| (i % 256) as u8).collect();

        let first = decoder.feed(&bytes).expect("feed");
        assert!(first < bytes.len(), "a full decoder must return short");
        assert_eq!(first, READY_LIMIT);
        assert_eq!(decoder.feed(&bytes[first..]).expect("feed"), 0);

        let decoded = drive(&mut decoder, &bytes, bytes.len()).expect("decode");
        // The first feed's samples are still in there, ahead of the rest, and
        // every byte of the second pass is accounted for.
        assert_eq!(decoded.samples().len(), bytes.len() + first);
        let expected: Vec<f32> = bytes
            .iter()
            .map(|&code| i16_to_f32(law.code_to_linear(code)))
            .collect();
        assert_eq!(&decoded.samples()[..first], &expected[..first]);
        assert_eq!(&decoded.samples()[first..], expected.as_slice());
    }

    /// The documented caller loop over an input larger than the decoder will
    /// hold, end to end: every byte arrives, in order, and none is lost to the
    /// short return.
    ///
    /// This exists because of what the byte-path negative control measured. A
    /// `feed` that decodes the right prefix but reports the whole input as
    /// consumed loses every byte past the limit, and *nothing else in this file
    /// notices*: the 256-code table gates, the 65,536-value encode gate and the
    /// chunk-size gate all use inputs far below `READY_LIMIT`, so the
    /// short-return path never runs. One gate standing between a caller and
    /// silently dropped audio is one too few, so this is the second.
    #[test]
    fn a_whole_file_feed_loses_nothing_to_the_short_return() {
        // Deliberately not a multiple of READY_LIMIT: the last partial pass
        // through the limit is where an off-by-one would sit.
        let bytes: Vec<u8> = (0..READY_LIMIT * 2 + 7).map(|i| (i % 251) as u8).collect();
        for law in BOTH {
            let expected: Vec<f32> = bytes
                .iter()
                .map(|&code| i16_to_f32(law.code_to_linear(code)))
                .collect();
            // Fed in one call, so every byte past the first limit has to come
            // back through a short return.
            let mut decoder = G711Decoder::new(law, AudioSpec::mono(8_000));
            let decoded = drive(&mut decoder, &bytes, bytes.len()).expect("decode");
            assert_eq!(
                decoded.samples(),
                expected.as_slice(),
                "{law:?} lost or reordered bytes on a whole-file feed"
            );
        }
    }

    // -- Gate 9: frame straddling ------------------------------------------
    //
    // Path exercised: `G711Decoder::{flush, buffered_bytes}`.

    /// A stereo stream that ends after an odd number of bytes ended mid-frame.
    /// One byte per sample means there is no straddling *sample* to hold, but
    /// there is still a straddling *frame*, and reporting it clean would hand
    /// back a buffer whose length is not a whole number of frames.
    #[test]
    fn a_stereo_stream_ending_mid_frame_is_truncated() {
        let mut decoder = G711Decoder::new(G711Law::MuLaw, AudioSpec::new(8_000, 2));
        let mut out = Vec::new();
        decoder.feed(&[0xFF]).expect("feed");
        assert_eq!(decoder.decode(&mut out).expect("decode"), 1);
        assert_eq!(
            decoder.buffered_bytes(),
            1,
            "one sample into a 2-byte frame"
        );

        let error = decoder.flush(&mut out).expect_err("flush must reject");
        assert!(
            matches!(
                error,
                DecodeError::Truncated {
                    expected: 2,
                    available: 1
                }
            ),
            "unexpected error: {error}"
        );
        // Idempotent: the second flush is a quiet no-op, not a second error.
        assert_eq!(decoder.flush(&mut out).expect("second flush"), 0);

        // And the same stream one byte longer ends cleanly.
        let mut decoder = G711Decoder::new(G711Law::MuLaw, AudioSpec::new(8_000, 2));
        let decoded = drive(&mut decoder, &[0xFF, 0x80], 1).expect("decode");
        assert_eq!(decoded.frames(), 1);
        assert_eq!(decoded.samples(), [0.0, 32_124.0 / 32768.0]);
    }

    /// The same at wider layouts, at every position inside a frame, so the
    /// report is against the frame and not against whatever the last chunk
    /// happened to be.
    #[test]
    fn every_position_inside_a_frame_is_truncated_and_only_the_boundary_is_clean() {
        for channels in [2_u16, 3, 6] {
            for bytes in 1..=(channels * 2) {
                let mut decoder = G711Decoder::new(G711Law::ALaw, AudioSpec::new(48_000, channels));
                let input = vec![0xD5_u8; usize::from(bytes)];
                let result = drive(&mut decoder, &input, 4);
                if bytes % channels == 0 {
                    let decoded = result.expect("a whole number of frames must decode");
                    assert_eq!(decoded.frames(), usize::from(bytes / channels));
                } else {
                    let error = result.expect_err("a partial frame must reject");
                    assert!(
                        matches!(
                            error,
                            DecodeError::Truncated { expected, available }
                                if expected == u64::from(channels)
                                    && available == u64::from(bytes % channels)
                        ),
                        "{channels} channels, {bytes} bytes: unexpected error: {error}"
                    );
                }
            }
        }
    }

    // -- The decoder's remaining contract ----------------------------------

    #[test]
    fn a_headerless_g711_stream_decodes_at_the_spec_it_was_given() {
        let mut decoder = G711Decoder::new(G711Law::ALaw, AudioSpec::mono(8_000));
        assert_eq!(decoder.law(), G711Law::ALaw);
        assert_eq!(decoder.output_spec(), AudioSpec::mono(8_000));

        let decoded = drive(&mut decoder, &[0xD5, 0x55, 0xAA, 0x2A], 64).expect("decode");
        assert_eq!(
            decoded.samples(),
            [
                8.0 / 32768.0,
                -8.0 / 32768.0,
                32_256.0 / 32768.0,
                -32_256.0 / 32768.0
            ]
        );
        assert_eq!(decoded.sample_rate(), 8_000);
        assert_eq!(decoded.frames(), 4);
    }

    /// The rate is the caller's and nothing here defaults to telephony's 8 kHz.
    /// G.711 is a sample format, not a rate.
    #[test]
    fn the_rate_is_whatever_the_spec_says_and_never_assumed() {
        for rate in [8_000, 16_000, 44_100, 48_000, 192_000, 1] {
            let mut decoder = G711Decoder::new(G711Law::MuLaw, AudioSpec::mono(rate));
            let decoded = drive(&mut decoder, &[0xFF, 0xFF], 1).expect("decode");
            assert_eq!(decoded.sample_rate(), rate);
            // The samples do not depend on the rate either.
            assert_eq!(decoded.samples(), [0.0, 0.0]);
        }
    }

    /// The sample-count guarantee, stated for this decoder: `n` frames in,
    /// exactly `n * channels` samples out, whatever the feed size.
    #[test]
    fn the_sample_count_is_exact_for_every_layout() {
        for law in BOTH {
            for channels in [1_u16, 2, 6] {
                let frames = 50;
                let spec = AudioSpec::new(48_000, channels);
                let count = frames * usize::from(channels);
                let input: Vec<u8> = (0..count).map(|i| (i % 256) as u8).collect();
                let mut decoder = G711Decoder::new(law, spec);
                let decoded = drive(&mut decoder, &input, 13).expect("decode");
                assert_eq!(
                    decoded.samples().len(),
                    count,
                    "{law:?} at {channels} channels produced the wrong count"
                );
                assert_eq!(decoded.frames(), frames);
            }
        }
    }

    /// A starved decoder says `0`, which means "feed me", not "end of stream".
    #[test]
    fn a_starved_decoder_reports_zero_rather_than_end_of_stream() {
        let mut decoder = G711Decoder::new(G711Law::ALaw, AudioSpec::mono(16_000));
        let mut out = Vec::new();
        assert_eq!(decoder.decode(&mut out).expect("decode"), 0);
        assert!(out.is_empty());
        assert_eq!(decoder.feed(&[]).expect("empty feed"), 0);
        assert_eq!(decoder.decode(&mut out).expect("decode"), 0);
        assert_eq!(decoder.feed(&[0xD5]).expect("feed"), 1);
        assert_eq!(decoder.decode(&mut out).expect("decode"), 1);
    }

    #[test]
    fn feeding_after_flush_is_quiet_until_reset() {
        let mut decoder = G711Decoder::new(G711Law::MuLaw, AudioSpec::mono(8_000));
        let mut out = Vec::new();
        assert_eq!(decoder.flush(&mut out).expect("flush"), 0);
        assert_eq!(decoder.flush(&mut out).expect("second flush"), 0);
        assert_eq!(decoder.feed(&[0xFF]).expect("feed"), 0);
        assert_eq!(decoder.decode(&mut out).expect("decode"), 0);
        assert!(out.is_empty());

        decoder.reset();
        assert_eq!(decoder.feed(&[0x80]).expect("feed"), 1);
        assert_eq!(decoder.decode(&mut out).expect("decode"), 1);
        assert_eq!(out, [32_124.0 / 32768.0]);
    }

    /// `reset` drops the frame position, so the next stream starts on a
    /// boundary rather than inheriting one.
    #[test]
    fn reset_drops_the_frame_position() {
        let mut decoder = G711Decoder::new(G711Law::MuLaw, AudioSpec::new(16_000, 2));
        decoder.feed(&[0xFF]).expect("feed");
        assert_eq!(decoder.buffered_bytes(), 1);
        decoder.reset();
        assert_eq!(decoder.buffered_bytes(), 0);
        let mut out = Vec::new();
        assert_eq!(decoder.flush(&mut out).expect("flush after reset"), 0);
    }

    #[test]
    fn the_decoder_is_object_safe_and_sendable() {
        fn assert_send<T: Send>() {}
        assert_send::<G711Decoder>();

        let boxed: Box<dyn Decoder> =
            Box::new(G711Decoder::new(G711Law::ALaw, AudioSpec::new(44_100, 2)));
        assert_eq!(boxed.output_spec().channels, 2);
    }

    /// A spec with no channels is degenerate but must not divide by zero.
    #[test]
    fn a_zero_channel_spec_does_not_divide_by_zero() {
        let mut decoder = G711Decoder::new(G711Law::MuLaw, AudioSpec::new(16_000, 0));
        let mut out = Vec::new();
        decoder.feed(&[0xFF]).expect("feed");
        assert_eq!(decoder.decode(&mut out).expect("decode"), 1);
        assert_eq!(decoder.buffered_bytes(), 0);
        assert_eq!(decoder.flush(&mut out).expect("flush"), 0);
    }

    // -- Gate 10: cross-platform determinism -------------------------------

    /// FNV-1a over the little-endian bit patterns, so the witness is the bytes
    /// themselves and not a float comparison.
    fn fnv1a(bytes: impl IntoIterator<Item = u8>) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        for byte in bytes {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash
    }

    /// One number that changes if any bit of any decoded sample or encoded code
    /// changes, over the complete code domain and a fixed signal.
    ///
    /// Running this on two toolchains and getting the same constant is the
    /// evidence for the byte-identical claim; a tolerance-based test would pass
    /// on both while the outputs differed. The constant is pinned rather than
    /// recomputed at run time so that a change shows up as a diff in this file.
    ///
    /// Both directions and both paths are in the hash: decode goes through the
    /// decoder's `feed`, so a byte-path change moves the constant too.
    #[test]
    fn g711_output_is_bit_identical_to_a_pinned_witness() {
        let codes = all_codes();
        // Integer arithmetic and an exact division by a power of two, so the
        // signal is the same on every target. The tail is past full scale, so
        // the clamp is inside the witness.
        let signal: Vec<f32> = (-2048..=2048)
            .map(|i| i as f32 / 2048.0)
            .chain([1.5, -3.0])
            .collect();

        let mut bytes: Vec<u8> = Vec::new();
        for law in BOTH {
            let mut decoder = G711Decoder::new(law, AudioSpec::mono(16_000));
            // 7 is coprime with every chunk size that matters here, so the
            // decoder is driven across its short-feed path as well.
            let decoded = drive(&mut decoder, &codes, 7).expect("decode");
            bytes.extend(
                decoded
                    .samples()
                    .iter()
                    .flat_map(|s| s.to_bits().to_le_bytes()),
            );
            law.encode(&signal, &mut bytes);
        }
        assert_eq!(bytes.len(), 2 * (256 * 4 + signal.len()));
        assert_eq!(fnv1a(bytes), 0x81b6_c667_9170_0ce0, "G.711 output changed");
    }
}
