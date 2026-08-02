//! Linear PCM sample formats, and the conversions between them and the `f32`
//! this crate works in.
//!
//! This is the least interesting code in the crate and the foundation of all of
//! it: mu-law decodes to linear PCM, WAV dispatches to it, and every decoded
//! sample in every format passes through one of the functions here.
//!
//! # The numerical conventions, and where they come from
//!
//! These three were read out of decibri rather than chosen. A convention that
//! differs from decibri's does not fail a test here. It produces a silent
//! disagreement between this crate's output and decibri's on the same input,
//! which is the expensive kind of difference.
//!
//! **Scale.** An integer sample of `n` bits is divided by `2^(n-1)`, not by
//! `2^(n-1) - 1`. For `i16` that is `32768.0`, from
//! `crates/decibri/src/sample.rs:42` and `:62` in decibri, and from the same
//! divisor on decibri's numpy path in `bindings/python/src/lib.rs:874`. So
//! [`i16::MIN`] maps to exactly `-1.0` and [`i16::MAX`] to `32767/32768`,
//! slightly short of `+1.0`. The asymmetry is the price of the round trip being
//! exact, which is what [`f32_to_i16`] and [`i16_to_f32`] give in return.
//!
//! **Rounding and clamping.** Float to integer clamps the float into
//! `-1.0..=1.0`, scales, truncates toward zero, then clamps again in integer
//! space, `crates/decibri/src/sample.rs:8-14`. A value exactly halfway between
//! two integers goes toward zero, because truncation is what decibri's cast
//! does. Both clamps stay: decoded and processed audio does exceed full scale
//! in practice: decibri measured 1.41 out of its echo canceller against a
//! capture peaking at 0.81, and a conversion that wrapped it would turn a loud
//! passage into a loud passage of the opposite sign. decibri guards that with a
//! regression test at `crates/decibri/src/sample.rs:188`; this crate guards it
//! in `a_float_beyond_full_scale_clamps_and_never_wraps`.
//!
//! Float *output* formats are not clamped, matching
//! `f32_to_f32_le_bytes` at `crates/decibri/src/sample.rs:23-29`: an `f32`
//! consumer is delivered what the chain produced, overshoot included.
//!
//! **Downmix.** The arithmetic mean of a frame's channels, summed in `f32` in
//! channel order, `crates/decibri/src/sample.rs:94-103`, which is what
//! decibri's `File::open` reaches through `build_capture_stage`'s `Downmix`
//! stage at `crates/decibri/src/stage.rs:104`. Not a sum, not a weighted
//! fold. See [`downmix_to_mono`].
//!
//! # Determinism, and the two things it is not
//!
//! Every conversion here is bit-exact and byte-identical across targets. Two
//! properties are deliberately *not* claimed alongside it.
//!
//! **Losslessness.** A round trip through `f32` is exact for every format at or
//! below 24 significant bits (`u8`, `i8`, `i16`, the packed 24-bit formats and
//! `f32` itself), and not for `i32` or `f64`, whose 31 and 53 significant
//! bits do not fit in a 24-bit significand. Those land on the nearest
//! representable `f32`, deterministically, with the error bounded by half an ulp
//! at that magnitude. Nothing wraps and no sign flips. It is inherent to an
//! `f32` internal representation, and the right trade for a crate feeding
//! decibri's `f32` chain, but it is not losslessness and is not described as it.
//!
//! **A NaN's identity.** A NaN converted to an integer becomes silence, and a
//! NaN narrowed from `f64` to `f32` becomes silence as well. The second was the
//! one place in this crate where the output bits were architecture-dependent,
//! because IEEE 754 does not specify the payload a NaN keeps through it, so
//! normalising it removes the exception rather than documenting it.
//!
//! # No dither
//!
//! Float to integer clamps and rounds, deterministically, with nothing added.
//! Dither trades correlated distortion for uncorrelated noise when reducing bit
//! depth for a listener, and it is the wrong tool twice over here: this crate
//! claims bit-exact cross-platform byte-identical output, and a random source
//! would either break that claim or make a generator seed part of the public
//! contract. There is no listener on a decode front end feeding a speech
//! pipeline. If dither is ever wanted it belongs in an output stage.

/// The widest sample any [`SampleFormat`] occupies, in bytes.
///
/// `f64` sets it. A decoder holding a partial sample needs a buffer this big
/// and never bigger, which is why [`PcmDecoder`](crate::PcmDecoder) can hold
/// its straddling bytes in a fixed array rather than a `Vec`.
pub const MAX_BYTES_PER_SAMPLE: usize = 8;

/// The most negative value a packed 24-bit sample can hold.
pub const I24_MIN: i32 = -8_388_608;

/// The most positive value a packed 24-bit sample can hold.
pub const I24_MAX: i32 = 8_388_607;

/// A linear PCM sample format: a width, a signedness and, above one byte, a
/// byte order.
///
/// Little-endian covers WAV, big-endian covers AIFF, and both are needed.
///
/// **Eight-bit signedness is the container's, not a convention.** WAV's 8-bit
/// PCM is unsigned, offset by 128; AIFF's is signed two's complement. Those
/// are what the two specifications say, and they are exact inverses of each
/// other, which is why [`U8`](Self::U8) and [`I8`](Self::I8) are two variants
/// rather than one variant and a flag somebody forgets to set. A converter
/// that reads either one as the other produces audio offset by half full
/// scale, which sounds like severe distortion rather than like a subtle bug;
/// see [`u8_to_f32`] and [`i8_to_f32`].
///
/// This enum is `#[non_exhaustive]`: a consumer matching on it needs a `_`
/// arm, so a later container's format can be added without breaking source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SampleFormat {
    /// Unsigned 8-bit, offset by 128, as WAV carries. One byte, so no byte
    /// order.
    U8,
    /// Signed 8-bit two's complement, as AIFF carries. One byte, so no byte
    /// order.
    I8,
    /// Signed 16-bit, little-endian.
    I16Le,
    /// Signed 16-bit, big-endian.
    I16Be,
    /// Signed 24-bit packed into three bytes, little-endian.
    I24Le,
    /// Signed 24-bit packed into three bytes, big-endian.
    I24Be,
    /// Signed 32-bit, little-endian.
    I32Le,
    /// Signed 32-bit, big-endian.
    I32Be,
    /// IEEE 754 binary32, little-endian.
    F32Le,
    /// IEEE 754 binary32, big-endian.
    F32Be,
    /// IEEE 754 binary64, little-endian.
    F64Le,
    /// IEEE 754 binary64, big-endian.
    F64Be,
}

impl SampleFormat {
    /// How many bytes one sample occupies.
    ///
    /// Three for the 24-bit formats: they are packed, not padded into four.
    pub const fn bytes_per_sample(self) -> usize {
        match self {
            Self::U8 | Self::I8 => 1,
            Self::I16Le | Self::I16Be => 2,
            Self::I24Le | Self::I24Be => 3,
            Self::I32Le | Self::I32Be | Self::F32Le | Self::F32Be => 4,
            Self::F64Le | Self::F64Be => 8,
        }
    }

    /// How many bits of resolution one sample carries.
    ///
    /// The form a container declares, and the form
    /// [`DecodeError::UnsupportedSampleFormat`](crate::DecodeError::UnsupportedSampleFormat)
    /// reports.
    pub const fn bits_per_sample(self) -> u16 {
        (self.bytes_per_sample() * 8) as u16
    }

    /// `true` for the IEEE 754 formats.
    ///
    /// The distinction is not cosmetic: the float formats pass values through
    /// unclamped, and the integer formats clamp.
    pub const fn is_float(self) -> bool {
        matches!(self, Self::F32Le | Self::F32Be | Self::F64Le | Self::F64Be)
    }

    /// Appends every whole sample in `bytes` to `output` as `f32`, and returns
    /// how many it appended.
    ///
    /// `output` is appended to, never cleared, the same convention as
    /// [`Decoder::decode`](crate::Decoder::decode), so a decoder's output vector
    /// feeds a resampler with no copy in between. A trailing partial sample is
    /// ignored rather than rejected; the caller owns the decision about whether
    /// running out of bytes mid-sample is an error, because on an open stream it
    /// is not. [`PcmDecoder`](crate::PcmDecoder) makes that decision.
    ///
    /// ```
    /// use decibri_decode::SampleFormat;
    ///
    /// let mut samples = Vec::new();
    /// // 0x8000 is i16::MIN little-endian: exactly -1.0.
    /// let appended = SampleFormat::I16Le.decode(&[0x00, 0x80, 0x00, 0x40], &mut samples);
    /// assert_eq!(appended, 2);
    /// assert_eq!(samples, [-1.0, 0.5]);
    /// ```
    pub fn decode(self, bytes: &[u8], output: &mut Vec<f32>) -> usize {
        let width = self.bytes_per_sample();
        let whole = bytes.len() / width;
        output.reserve(whole);
        for sample in bytes.chunks_exact(width) {
            output.push(self.decode_sample(sample));
        }
        whole
    }

    /// Appends `samples` to `output` in this format, and returns how many bytes
    /// it appended.
    ///
    /// Integer formats clamp; float formats pass the value through as it is.
    ///
    /// ```
    /// use decibri_decode::SampleFormat;
    ///
    /// let mut bytes = Vec::new();
    /// // Beyond full scale clamps to the extreme rather than wrapping.
    /// let appended = SampleFormat::I16Be.encode(&[1.5], &mut bytes);
    /// assert_eq!(appended, 2);
    /// assert_eq!(bytes, [0x7f, 0xff]);
    /// ```
    pub fn encode(self, samples: &[f32], output: &mut Vec<u8>) -> usize {
        let written = samples.len() * self.bytes_per_sample();
        output.reserve(written);
        for &sample in samples {
            self.encode_sample(sample, output);
        }
        written
    }

    /// One sample's worth of bytes to `f32`. `bytes` is exactly
    /// [`bytes_per_sample`](Self::bytes_per_sample) long.
    fn decode_sample(self, bytes: &[u8]) -> f32 {
        debug_assert_eq!(bytes.len(), self.bytes_per_sample());
        match self {
            Self::U8 => u8_to_f32(bytes[0]),
            Self::I8 => i8_to_f32(bytes[0] as i8),
            Self::I16Le => i16_to_f32(i16::from_le_bytes([bytes[0], bytes[1]])),
            Self::I16Be => i16_to_f32(i16::from_be_bytes([bytes[0], bytes[1]])),
            // Sign extension from 24 bits to 32 is the classic bug in this
            // format. Landing the three bytes in the *top* of an i32 and
            // shifting right arithmetically does it without a branch and
            // without a mask that has to be got right twice.
            Self::I24Le => i24_to_f32(i32::from_le_bytes([0, bytes[0], bytes[1], bytes[2]]) >> 8),
            Self::I24Be => i24_to_f32(i32::from_be_bytes([bytes[0], bytes[1], bytes[2], 0]) >> 8),
            Self::I32Le => i32_to_f32(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])),
            Self::I32Be => i32_to_f32(i32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])),
            Self::F32Le => f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            Self::F32Be => f32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            Self::F64Le => narrow(f64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ])),
            Self::F64Be => narrow(f64::from_be_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ])),
        }
    }

    /// One `f32` to this format's bytes, appended to `output`.
    fn encode_sample(self, sample: f32, output: &mut Vec<u8>) {
        match self {
            Self::U8 => output.push(f32_to_u8(sample)),
            Self::I8 => output.push(f32_to_i8(sample) as u8),
            Self::I16Le => output.extend_from_slice(&f32_to_i16(sample).to_le_bytes()),
            Self::I16Be => output.extend_from_slice(&f32_to_i16(sample).to_be_bytes()),
            // The fourth byte of the i32 is sign extension, which is exactly
            // what a packed 24-bit sample leaves out: drop the high byte on the
            // little-endian side and the low byte on the big-endian one.
            Self::I24Le => output.extend_from_slice(&f32_to_i24(sample).to_le_bytes()[..3]),
            Self::I24Be => output.extend_from_slice(&f32_to_i24(sample).to_be_bytes()[1..]),
            Self::I32Le => output.extend_from_slice(&f32_to_i32(sample).to_le_bytes()),
            Self::I32Be => output.extend_from_slice(&f32_to_i32(sample).to_be_bytes()),
            Self::F32Le => output.extend_from_slice(&sample.to_le_bytes()),
            Self::F32Be => output.extend_from_slice(&sample.to_be_bytes()),
            Self::F64Le => output.extend_from_slice(&f64::from(sample).to_le_bytes()),
            Self::F64Be => output.extend_from_slice(&f64::from(sample).to_be_bytes()),
        }
    }
}

/// `f64` to `f32`, with NaN normalised to silence.
///
/// The narrowing itself is round to nearest, ties to even, which IEEE 754
/// specifies exactly, for every value except NaN. The bit pattern a NaN keeps
/// through a narrowing conversion is *not* specified, so a decoded `f64` NaN was
/// the one value in this crate whose `f32` representation could legitimately
/// differ between architectures.
///
/// Sending it to `0.0` removes that exception rather than documenting it, and it
/// is the answer the integer path already gives: `quantize` maps NaN to
/// silence too, because silence is the only value a NaN can safely become. The
/// determinism claim on this crate then stands without a carve-out, which is
/// worth more than an accurate footnote: a contract with one exception invites
/// the question of what else has one.
///
/// The 32-bit float formats are untouched by this: they involve no narrowing, so
/// a NaN in the input arrives in the output as exactly the bit pattern the file
/// carried, on every target.
fn narrow(sample: f64) -> f32 {
    if sample.is_nan() {
        0.0
    } else {
        sample as f32
    }
}

/// The float-to-integer rule, in one place because every width shares it.
///
/// Clamp the float, scale, truncate toward zero, clamp in integer space. The
/// second clamp is not redundant: `1.0 * 32768.0` is `32768`, one past
/// [`i16::MAX`]. The order matters: a cast before the clamp is the wrap this
/// exists to prevent, and is the regression decibri records at
/// `crates/decibri/src/sample.rs:186`.
///
/// A NaN sample clamps to NaN, and NaN cast to an integer is `0` in Rust,
/// silence. That is the same answer decibri's path gives, and it is a
/// deliberate one: silence is the only value a NaN can safely become.
pub(crate) fn quantize(sample: f32, scale: f32, min: i64, max: i64) -> i64 {
    let clamped = sample.clamp(-1.0, 1.0);
    ((clamped * scale) as i64).clamp(min, max)
}

/// Unsigned 8-bit PCM to `f32`, offset by 128.
///
/// `0` is `-1.0`, `128` is `0.0`, `255` is `127/128`. The offset is what makes
/// 8-bit WAV different from every wider format, and reading it as signed shifts
/// the whole signal by half full scale.
pub fn u8_to_f32(sample: u8) -> f32 {
    (i16::from(sample) - 128) as f32 / 128.0
}

/// `f32` to unsigned 8-bit PCM, offset by 128.
pub fn f32_to_u8(sample: f32) -> u8 {
    (quantize(sample, 128.0, -128, 127) + 128) as u8
}

/// Signed 8-bit PCM to `f32`, divided by `128`.
///
/// The 8-bit format AIFF carries, and the exact inverse of WAV's unsigned
/// convention: `-128` is `-1.0`, `0` is `0.0`, `127` is `127/128`. The same
/// byte read through [`u8_to_f32`] instead lands half full scale away, so
/// which of the two a container uses is the container's most audible single
/// bit of documentation.
pub fn i8_to_f32(sample: i8) -> f32 {
    f32::from(sample) / 128.0
}

/// `f32` to signed 8-bit PCM, scaled by `128` and clamped.
pub fn f32_to_i8(sample: f32) -> i8 {
    quantize(sample, 128.0, -128, 127) as i8
}

/// Signed 16-bit PCM to `f32`, divided by `32768`.
///
/// [`i16::MIN`] is exactly `-1.0`; [`i16::MAX`] is `32767/32768`.
pub fn i16_to_f32(sample: i16) -> f32 {
    f32::from(sample) / 32768.0
}

/// `f32` to signed 16-bit PCM, scaled by `32768` and clamped.
///
/// The exact inverse of [`i16_to_f32`] across the whole 16-bit domain: every
/// one of the 65,536 values survives the round trip bit-identically.
pub fn f32_to_i16(sample: f32) -> i16 {
    quantize(sample, 32768.0, i16::MIN as i64, i16::MAX as i64) as i16
}

/// Sign-extended 24-bit PCM to `f32`, divided by `8388608`.
///
/// `sample` is the sign-extended value, in `I24_MIN..=I24_MAX`; a value outside
/// that range converts to a magnitude past `1.0` rather than being clamped,
/// because a 24-bit sample cannot be out of range and a caller passing one that
/// is has a bug upstream worth seeing.
pub fn i24_to_f32(sample: i32) -> f32 {
    sample as f32 / 8_388_608.0
}

/// `f32` to a sign-extended 24-bit value in `I24_MIN..=I24_MAX`.
pub fn f32_to_i24(sample: f32) -> i32 {
    quantize(sample, 8_388_608.0, I24_MIN as i64, I24_MAX as i64) as i32
}

/// Signed 32-bit PCM to `f32`, divided by `2147483648`.
///
/// `f32` carries 24 bits of significand, so this is lossy above `2^24`: the
/// value lands on the nearest representable `f32`, which is what the format
/// costs and not something this crate can avoid while its internal
/// representation is `f32`. Every 32-bit sample the ear can tell apart survives;
/// the bits that do not are 48 dB below the noise floor of any real recording.
pub fn i32_to_f32(sample: i32) -> f32 {
    sample as f32 / 2_147_483_648.0
}

/// `f32` to signed 32-bit PCM, scaled by `2147483648` and clamped.
pub fn f32_to_i32(sample: f32) -> i32 {
    quantize(sample, 2_147_483_648.0, i32::MIN as i64, i32::MAX as i64) as i32
}

/// Collapses interleaved multichannel audio to mono by averaging each frame's
/// channels, appends the result to `output` and returns how many samples it
/// appended.
///
/// The formula is decibri's, read from `crates/decibri/src/sample.rs:94-103`
/// and reproduced rather than reinvented: the arithmetic mean, summed in `f32`
/// in channel order, divided by the channel count. Not a sum, and not weighted.
/// If this crate averaged differently, the same stereo file would give two
/// different mono results depending on which path read it.
///
/// `channels <= 1` copies the input through unchanged. A trailing partial frame
/// is dropped, as decibri drops it.
///
/// ```
/// use decibri_decode::downmix_to_mono;
///
/// let mut mono = Vec::new();
/// // L R L R -> two frames, each the mean of its two channels.
/// downmix_to_mono(&[0.5, 0.3, 0.4, 0.6], 2, &mut mono);
/// assert_eq!(mono.len(), 2);
/// ```
pub fn downmix_to_mono(samples: &[f32], channels: u16, output: &mut Vec<f32>) -> usize {
    if channels <= 1 {
        output.extend_from_slice(samples);
        return samples.len();
    }
    let channels = usize::from(channels);
    let frames = samples.len() / channels;
    output.reserve(frames);
    for frame in samples.chunks_exact(channels) {
        // decibri's exact expression. `Iterator::sum` folds left to right from
        // 0.0, and f32 addition does not associate, so the fold order is part
        // of the answer and not an implementation detail.
        output.push(frame.iter().sum::<f32>() / channels as f32);
    }
    frames
}

/// Splits interleaved samples into one plane per channel.
///
/// A trailing partial frame is dropped, so every returned plane has the same
/// length. `channels == 0` yields no planes.
///
/// ```
/// use decibri_decode::deinterleave;
///
/// let planes = deinterleave(&[1.0, -1.0, 2.0, -2.0], 2);
/// assert_eq!(planes, vec![vec![1.0, 2.0], vec![-1.0, -2.0]]);
/// ```
pub fn deinterleave(samples: &[f32], channels: u16) -> Vec<Vec<f32>> {
    let channels = usize::from(channels);
    if channels == 0 {
        return Vec::new();
    }
    let frames = samples.len() / channels;
    let mut planes = vec![Vec::with_capacity(frames); channels];
    for frame in samples.chunks_exact(channels) {
        for (plane, &sample) in planes.iter_mut().zip(frame) {
            plane.push(sample);
        }
    }
    planes
}

/// Interleaves per-channel planes into `output`, and returns how many samples
/// it appended.
///
/// The frame count is the shortest plane's length, so planes of unequal length
/// interleave to the part they all cover rather than to a ragged buffer or a
/// panic. No planes appends nothing.
///
/// ```
/// use decibri_decode::interleave;
///
/// let mut out = Vec::new();
/// interleave(&[vec![1.0, 2.0], vec![-1.0, -2.0]], &mut out);
/// assert_eq!(out, [1.0, -1.0, 2.0, -2.0]);
/// ```
pub fn interleave<P: AsRef<[f32]>>(planes: &[P], output: &mut Vec<f32>) -> usize {
    let frames = planes
        .iter()
        .map(|plane| plane.as_ref().len())
        .min()
        .unwrap_or(0);
    let appended = frames * planes.len();
    output.reserve(appended);
    for frame in 0..frames {
        for plane in planes {
            output.push(plane.as_ref()[frame]);
        }
    }
    appended
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every format, for the tests that have to cover all of them rather than
    /// the one that was being thought about.
    const ALL: [SampleFormat; 12] = [
        SampleFormat::U8,
        SampleFormat::I8,
        SampleFormat::I16Le,
        SampleFormat::I16Be,
        SampleFormat::I24Le,
        SampleFormat::I24Be,
        SampleFormat::I32Le,
        SampleFormat::I32Be,
        SampleFormat::F32Le,
        SampleFormat::F32Be,
        SampleFormat::F64Le,
        SampleFormat::F64Be,
    ];

    /// The multi-byte formats, paired little-endian with big-endian.
    const ENDIAN_PAIRS: [(SampleFormat, SampleFormat); 5] = [
        (SampleFormat::I16Le, SampleFormat::I16Be),
        (SampleFormat::I24Le, SampleFormat::I24Be),
        (SampleFormat::I32Le, SampleFormat::I32Be),
        (SampleFormat::F32Le, SampleFormat::F32Be),
        (SampleFormat::F64Le, SampleFormat::F64Be),
    ];

    fn decode_one(format: SampleFormat, bytes: &[u8]) -> f32 {
        let mut out = Vec::new();
        assert_eq!(format.decode(bytes, &mut out), 1);
        out[0]
    }

    fn encode_one(format: SampleFormat, sample: f32) -> Vec<u8> {
        let mut out = Vec::new();
        format.encode(&[sample], &mut out);
        out
    }

    // -- Gate 5: exhaustive i16 round trip --------------------------------

    /// All 65,536 `i16` values to `f32` and back, bit-identical. The complete
    /// input domain, not a sample of it.
    #[test]
    fn every_i16_value_round_trips_bit_identically() {
        for raw in i16::MIN..=i16::MAX {
            let as_float = i16_to_f32(raw);
            assert_eq!(
                f32_to_i16(as_float),
                raw,
                "{raw} did not survive the round trip (became {as_float})"
            );
        }
    }

    /// The scale factor, stated as the two values that distinguish `/32768`
    /// from `/32767`. This is the assertion that would fail if somebody
    /// "corrected" the asymmetry.
    #[test]
    fn i16_is_scaled_by_32768_not_32767() {
        assert_eq!(i16_to_f32(i16::MIN), -1.0);
        assert_eq!(i16_to_f32(i16::MAX), 32767.0 / 32768.0);
        assert!(i16_to_f32(i16::MAX) < 1.0);
        assert_eq!(i16_to_f32(0), 0.0);
        assert_eq!(i16_to_f32(16_384), 0.5);
        // decibri's own worked example: 0.5 * 32768 = 16384.
        assert_eq!(f32_to_i16(0.5), 16_384);
    }

    // -- Gate 6: exhaustive u8 round trip ---------------------------------

    /// All 256 `u8` values.
    #[test]
    fn every_u8_value_round_trips_bit_identically() {
        for raw in u8::MIN..=u8::MAX {
            let as_float = u8_to_f32(raw);
            assert_eq!(
                f32_to_u8(as_float),
                raw,
                "{raw} did not survive the round trip (became {as_float})"
            );
        }
    }

    /// The 128 offset, at the three points where getting it wrong is loudest.
    #[test]
    fn u8_is_unsigned_and_offset_by_128() {
        assert_eq!(u8_to_f32(0), -1.0);
        assert_eq!(u8_to_f32(128), 0.0);
        assert_eq!(u8_to_f32(255), 127.0 / 128.0);
        // Silence in 8-bit WAV is 0x80, not 0x00. Reading it as signed puts
        // silence at -1.0: a DC offset of half full scale.
        assert_eq!(decode_one(SampleFormat::U8, &[0x80]), 0.0);
        assert_eq!(encode_one(SampleFormat::U8, 0.0), [0x80]);
    }

    /// All 256 `i8` values: the exhaustive gate for the format AIFF carries.
    #[test]
    fn every_i8_value_round_trips_bit_identically() {
        for raw in i8::MIN..=i8::MAX {
            let as_float = i8_to_f32(raw);
            assert_eq!(
                f32_to_i8(as_float),
                raw,
                "{raw} did not survive the round trip (became {as_float})"
            );
        }
    }

    /// The signed convention, anchored at the bytes where confusing it with
    /// the unsigned one is loudest. `0x00` is silence in AIFF and `-1.0` in
    /// WAV; `0x80` is the reverse. A shared implementation that read both
    /// containers' 8-bit through one function would fail one of these two
    /// blocks whichever convention it picked.
    #[test]
    fn i8_is_signed_and_the_exact_inverse_of_u8() {
        assert_eq!(i8_to_f32(0), 0.0);
        assert_eq!(i8_to_f32(i8::MIN), -1.0);
        assert_eq!(i8_to_f32(i8::MAX), 127.0 / 128.0);
        assert_eq!(i8_to_f32(64), 0.5);

        // Silence in AIFF's 8-bit is 0x00, not 0x80.
        assert_eq!(decode_one(SampleFormat::I8, &[0x00]), 0.0);
        assert_eq!(encode_one(SampleFormat::I8, 0.0), [0x00]);
        // 0x80 is the most negative value, not the midpoint.
        assert_eq!(decode_one(SampleFormat::I8, &[0x80]), -1.0);
        // And every byte decodes exactly half full scale away from the same
        // byte read as WAV's unsigned 8-bit, wrapped into range: the two
        // conventions differ by the sign bit and nothing else.
        for byte in 0..=u8::MAX {
            assert_eq!(
                decode_one(SampleFormat::I8, &[byte]),
                decode_one(SampleFormat::U8, &[byte ^ 0x80]),
                "byte {byte:#04x}: I8 and U8 are not sign-bit twins"
            );
        }
    }

    // -- Gate 7: i24 and i32 round trips ----------------------------------

    /// Every one of the 16,777,216 packed 24-bit values. Exhaustive is
    /// affordable here, so the strided sweep the plan allowed is not needed:
    /// `f32` holds 24 significand bits, so the whole domain is representable
    /// and the round trip is exact across all of it.
    #[test]
    fn every_i24_value_round_trips_bit_identically() {
        for raw in I24_MIN..=I24_MAX {
            assert_eq!(f32_to_i24(i24_to_f32(raw)), raw, "{raw} did not round trip");
        }
    }

    /// `i32` cannot round trip exhaustively through `f32` and this states the
    /// exact boundary of what does. Below `2^24` every value is exact; above
    /// it, the value lands on the nearest `f32` and the error is bounded by half
    /// an ulp at that magnitude, never wrapped, never sign-flipped.
    #[test]
    fn i32_round_trips_exactly_within_the_f32_significand_and_lands_near_outside_it() {
        // Exact below 2^24: the whole range an f32 represents integrally.
        for raw in [0, 1, -1, 1 << 20, -(1 << 20), (1 << 24) - 1, -(1 << 24)] {
            assert_eq!(f32_to_i32(i32_to_f32(raw)), raw, "{raw} must be exact");
        }
        // Boundaries: the two values where a wrap would show up first.
        assert_eq!(i32_to_f32(i32::MIN), -1.0);
        assert_eq!(f32_to_i32(-1.0), i32::MIN);
        // i32::MAX is not representable in f32; it rounds to 2^31, which the
        // clamp brings back to i32::MAX. Exact by way of the clamp.
        assert_eq!(i32_to_f32(i32::MAX), 1.0);
        assert_eq!(f32_to_i32(1.0), i32::MAX);

        // A strided sweep over the whole domain. The stride is 65,537, an odd
        // number coprime with every power of two, so successive samples land on
        // every bit position of the low half rather than repeating one residue
        // class the way a power-of-two stride would. 65,536 iterations covers
        // the range in a test that runs in milliseconds; exhaustive would be
        // 4.29 billion.
        const STRIDE: i64 = 65_537;
        let mut raw = i32::MIN as i64;
        while raw <= i32::MAX as i64 {
            let value = raw as i32;
            let recovered = f32_to_i32(i32_to_f32(value));
            // Half an ulp of f32 at this magnitude, and never more.
            let ulp = (value.unsigned_abs().max(1 << 24) >> 23) as i64;
            let error = (recovered as i64 - raw).abs();
            assert!(
                error <= ulp,
                "{value} recovered as {recovered}, off by {error} (ulp {ulp})"
            );
            // Whatever the precision loss, the sign never flips.
            assert!(
                recovered.signum() == value.signum() || value.unsigned_abs() <= 1,
                "{value} changed sign to {recovered}"
            );
            raw += STRIDE;
        }
    }

    /// The packed 24-bit boundaries, including the sign-extension edge that is
    /// this format's classic bug: `0x800000` is the most negative value, not
    /// the largest positive one.
    #[test]
    fn i24_sign_extension_is_correct_at_the_boundaries() {
        assert_eq!(i24_to_f32(I24_MIN), -1.0);
        assert_eq!(i24_to_f32(I24_MAX), 8_388_607.0 / 8_388_608.0);
        assert_eq!(i24_to_f32(0), 0.0);
        // 0x800000 little-endian is I24_MIN. Zero-extending instead of
        // sign-extending would read it as +8388608, i.e. +1.0.
        assert_eq!(decode_one(SampleFormat::I24Le, &[0x00, 0x00, 0x80]), -1.0);
        assert_eq!(decode_one(SampleFormat::I24Be, &[0x80, 0x00, 0x00]), -1.0);
        // 0xFFFFFF is -1 in 24-bit two's complement, not +16777215.
        assert_eq!(
            decode_one(SampleFormat::I24Le, &[0xff, 0xff, 0xff]),
            -1.0 / 8_388_608.0
        );
        assert_eq!(
            decode_one(SampleFormat::I24Be, &[0xff, 0xff, 0xff]),
            -1.0 / 8_388_608.0
        );
        // And the largest positive, which must not read as negative.
        assert_eq!(
            decode_one(SampleFormat::I24Le, &[0xff, 0xff, 0x7f]),
            8_388_607.0 / 8_388_608.0
        );
    }

    // -- Gate 8: clamping -------------------------------------------------

    /// A float past full scale converts to the integer extreme and never
    /// wraps. A wrap turns a loud passage into a loud passage of the opposite
    /// sign, which is the worst failure mode available here; decibri guards the
    /// same property at `crates/decibri/src/sample.rs:188` because its echo
    /// canceller really does emit 1.41 against a capture peaking at 0.81.
    #[test]
    fn a_float_beyond_full_scale_clamps_and_never_wraps() {
        for over in [1.000_01_f32, 1.41, 2.0, 8.5, 1e9, f32::MAX, f32::INFINITY] {
            assert_eq!(f32_to_u8(over), u8::MAX, "{over} must clamp in u8");
            assert_eq!(f32_to_u8(-over), u8::MIN, "-{over} must clamp in u8");
            assert_eq!(f32_to_i8(over), i8::MAX, "{over} must clamp in i8");
            assert_eq!(f32_to_i8(-over), i8::MIN, "-{over} must clamp in i8");
            assert_eq!(f32_to_i16(over), i16::MAX, "{over} must clamp in i16");
            assert_eq!(f32_to_i16(-over), i16::MIN, "-{over} must clamp in i16");
            assert_eq!(f32_to_i24(over), I24_MAX, "{over} must clamp in i24");
            assert_eq!(f32_to_i24(-over), I24_MIN, "-{over} must clamp in i24");
            assert_eq!(f32_to_i32(over), i32::MAX, "{over} must clamp in i32");
            assert_eq!(f32_to_i32(-over), i32::MIN, "-{over} must clamp in i32");

            // Through the byte-level API, which is what a caller actually hits.
            assert_eq!(encode_one(SampleFormat::I16Le, over), [0xff, 0x7f]);
            assert_eq!(encode_one(SampleFormat::I16Le, -over), [0x00, 0x80]);
            assert_eq!(encode_one(SampleFormat::I24Be, over), [0x7f, 0xff, 0xff]);
            assert_eq!(encode_one(SampleFormat::I24Be, -over), [0x80, 0x00, 0x00]);
        }
    }

    /// The float formats do not clamp. An `f32` consumer is delivered what the
    /// chain produced, overshoot included, and decibri's `f32_to_f32_le_bytes`
    /// documents the same thing at `crates/decibri/src/sample.rs:23`, and the
    /// overshoot is a statement about the canceller's residual rather than a
    /// defect to hide.
    #[test]
    fn the_float_formats_pass_values_through_unclamped() {
        for over in [1.41_f32, -2.0, 100.0] {
            for format in [SampleFormat::F32Le, SampleFormat::F32Be] {
                let bytes = encode_one(format, over);
                assert_eq!(decode_one(format, &bytes), over);
            }
            for format in [SampleFormat::F64Le, SampleFormat::F64Be] {
                let bytes = encode_one(format, over);
                assert_eq!(decode_one(format, &bytes), over);
            }
        }
    }

    /// NaN becomes silence, not a wrapped extreme. Rust's float-to-integer cast
    /// saturates and sends NaN to zero; this pins that as intended behaviour
    /// rather than as something that happens to hold.
    #[test]
    fn nan_quantizes_to_silence() {
        assert_eq!(f32_to_i16(f32::NAN), 0);
        assert_eq!(f32_to_i32(f32::NAN), 0);
        assert_eq!(f32_to_i24(f32::NAN), 0);
        assert_eq!(f32_to_i8(f32::NAN), 0);
        // Silence in unsigned 8-bit is the 128 midpoint.
        assert_eq!(f32_to_u8(f32::NAN), 128);
    }

    /// A NaN arriving in an `f64` stream becomes silence at the point of
    /// narrowing, rather than becoming whichever NaN the target's conversion
    /// instruction happens to produce.
    ///
    /// IEEE 754 specifies the narrowing exactly for every value except NaN,
    /// whose payload through a conversion is left to the implementation. That
    /// made a decoded `f64` NaN the one output in this crate that could
    /// legitimately differ between architectures; normalising it to silence
    /// removes the exception, and matches what the integer path has always done.
    #[test]
    fn an_f64_nan_narrows_to_silence_rather_than_to_an_unspecified_nan() {
        for format in [SampleFormat::F64Le, SampleFormat::F64Be] {
            for nan in [f64::NAN, -f64::NAN, f64::from_bits(0x7ff8_0000_dead_beef)] {
                let bytes = match format {
                    SampleFormat::F64Le => nan.to_le_bytes(),
                    _ => nan.to_be_bytes(),
                };
                let decoded = decode_one(format, &bytes);
                assert!(!decoded.is_nan(), "{format:?} passed a NaN through");
                assert_eq!(decoded.to_bits(), 0.0_f32.to_bits(), "{format:?}");
            }
        }
        // A signalling NaN too: the quiet-bit handling on a narrowing is the
        // part that differs most between architectures.
        let signalling = f64::from_bits(0x7ff0_0000_0000_0001);
        assert!(signalling.is_nan());
        assert_eq!(
            decode_one(SampleFormat::F64Le, &signalling.to_le_bytes()),
            0.0
        );

        // Infinity is specified and is not touched: it is a real value that a
        // decoder must not quietly turn into silence.
        assert_eq!(
            decode_one(SampleFormat::F64Le, &f64::INFINITY.to_le_bytes()),
            f32::INFINITY
        );

        // The 32-bit float formats involve no narrowing, so a NaN there is
        // exactly the bit pattern the file carried, on every target.
        let bits = 0x7fc0_dead_u32;
        assert_eq!(
            decode_one(SampleFormat::F32Le, &bits.to_le_bytes()).to_bits(),
            bits
        );
    }

    /// Halfway between two integers goes toward zero, because truncation is
    /// what decibri's cast does (`crates/decibri/src/sample.rs:12`). Stated as
    /// a test so a later "improvement" to round-half-up has to argue with
    /// decibri rather than with a comment.
    #[test]
    fn a_halfway_value_truncates_toward_zero() {
        // 1.5 / 32768 scales to exactly 1.5.
        assert_eq!(f32_to_i16(1.5 / 32768.0), 1);
        assert_eq!(f32_to_i16(-1.5 / 32768.0), -1);
        // And just under a whole number rounds down, not to nearest.
        assert_eq!(f32_to_i16(0.999_999 * 2.0 / 32768.0), 1);
    }

    // -- Gate 9: endianness, against hand-written bytes -------------------

    /// Byte sequences written by hand from the format definitions, not
    /// produced by this crate's own encoder. An encoder and a decoder that
    /// share a byte-order mistake agree with each other perfectly.
    #[test]
    fn endianness_is_proven_against_hand_written_bytes() {
        // 0x1234 = 4660. As f32: 4660 / 32768.
        assert_eq!(
            decode_one(SampleFormat::I16Le, &[0x34, 0x12]),
            4660.0 / 32768.0
        );
        assert_eq!(
            decode_one(SampleFormat::I16Be, &[0x12, 0x34]),
            4660.0 / 32768.0
        );

        // 0x123456 = 1193046. As f32: 1193046 / 8388608.
        assert_eq!(
            decode_one(SampleFormat::I24Le, &[0x56, 0x34, 0x12]),
            1_193_046.0 / 8_388_608.0
        );
        assert_eq!(
            decode_one(SampleFormat::I24Be, &[0x12, 0x34, 0x56]),
            1_193_046.0 / 8_388_608.0
        );

        // 0x12345678 = 305419896. As f32 this rounds; compare against the same
        // rounding applied to the integer, which is the documented behaviour.
        assert_eq!(
            decode_one(SampleFormat::I32Le, &[0x78, 0x56, 0x34, 0x12]),
            305_419_896_f32 / 2_147_483_648.0
        );
        assert_eq!(
            decode_one(SampleFormat::I32Be, &[0x12, 0x34, 0x56, 0x78]),
            305_419_896_f32 / 2_147_483_648.0
        );

        // IEEE 754 binary32 for 1.0 is 0x3F800000.
        assert_eq!(
            decode_one(SampleFormat::F32Le, &[0x00, 0x00, 0x80, 0x3f]),
            1.0
        );
        assert_eq!(
            decode_one(SampleFormat::F32Be, &[0x3f, 0x80, 0x00, 0x00]),
            1.0
        );
        // -0.5 is 0xBF000000.
        assert_eq!(
            decode_one(SampleFormat::F32Le, &[0x00, 0x00, 0x00, 0xbf]),
            -0.5
        );
        assert_eq!(
            decode_one(SampleFormat::F32Be, &[0xbf, 0x00, 0x00, 0x00]),
            -0.5
        );

        // IEEE 754 binary64 for 1.0 is 0x3FF0000000000000.
        assert_eq!(
            decode_one(
                SampleFormat::F64Le,
                &[0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f]
            ),
            1.0
        );
        assert_eq!(
            decode_one(
                SampleFormat::F64Be,
                &[0x3f, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
            ),
            1.0
        );
        // -0.25 is 0xBFD0000000000000.
        assert_eq!(
            decode_one(
                SampleFormat::F64Be,
                &[0xbf, 0xd0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
            ),
            -0.25
        );
    }

    /// `f64` narrows to `f32` on the way in, round to nearest, ties to even.
    /// The crate works in `f32`, so a 64-bit source loses the bits `f32` has no
    /// room for, deterministically, and at the point of decode rather than
    /// somewhere downstream.
    #[test]
    fn f64_narrows_to_the_nearest_f32() {
        // 0.1 is representable in neither, and the two nearest values differ.
        let mut bytes = Vec::new();
        SampleFormat::F64Le.encode(&[0.1], &mut bytes);
        assert_eq!(decode_one(SampleFormat::F64Le, &bytes), 0.1_f32);
        // A value with more precision than f32 can hold lands on the nearest.
        let precise = 1.0_f64 / 3.0;
        assert_eq!(
            decode_one(SampleFormat::F64Be, &precise.to_be_bytes()),
            precise as f32
        );
        // And one too large for f32 becomes infinity rather than wrapping.
        assert_eq!(
            decode_one(SampleFormat::F64Le, &1.0e300_f64.to_le_bytes()),
            f32::INFINITY
        );
    }

    /// The two byte orders are byte-reversals of each other, for every
    /// multi-byte format and a spread of values. Catches a format whose two
    /// variants were implemented independently and diverged.
    #[test]
    fn the_two_byte_orders_are_reversals_of_each_other() {
        for (little, big) in ENDIAN_PAIRS {
            for sample in [0.0_f32, 0.5, -0.5, 1.0, -1.0, 0.123_456_79, -0.987_654_3] {
                let mut le = encode_one(little, sample);
                let be = encode_one(big, sample);
                le.reverse();
                assert_eq!(le, be, "{little:?} and {big:?} disagree on {sample}");
            }
        }
    }

    // -- Round trips through the byte-level API ---------------------------

    /// Every format survives f32 -> bytes -> f32 for values it can represent.
    /// The 8-bit format is coarse enough to need its own tolerance; the rest
    /// are exact at these values.
    #[test]
    fn every_format_round_trips_through_bytes() {
        for format in ALL {
            let samples = [0.0_f32, 0.5, -0.5, 0.25, -0.75];
            let mut bytes = Vec::new();
            let written = format.encode(&samples, &mut bytes);
            assert_eq!(written, samples.len() * format.bytes_per_sample());
            assert_eq!(bytes.len(), written);

            let mut back = Vec::new();
            assert_eq!(format.decode(&bytes, &mut back), samples.len());
            for (original, recovered) in samples.iter().zip(&back) {
                let tolerance = 1.0 / 128.0;
                assert!(
                    (original - recovered).abs() <= tolerance,
                    "{format:?}: {original} came back as {recovered}"
                );
            }
            // Everything but the 8-bit format is exact at these values.
            if format != SampleFormat::U8 {
                assert_eq!(back, samples, "{format:?} was not exact");
            }
        }
    }

    /// A trailing partial sample is ignored by the slice API rather than
    /// rejected or read past. The decision about whether it is an error belongs
    /// to the caller, because on an open stream it is not one.
    #[test]
    fn a_trailing_partial_sample_is_ignored_by_the_slice_api() {
        let mut out = Vec::new();
        // Five bytes of a three-byte format: one whole sample, two bytes over.
        assert_eq!(
            SampleFormat::I24Le.decode(&[0, 0, 0, 0xff, 0xff], &mut out),
            1
        );
        assert_eq!(out, [0.0]);

        out.clear();
        assert_eq!(SampleFormat::F64Be.decode(&[0; 7], &mut out), 0);
        assert!(out.is_empty());
        assert_eq!(SampleFormat::U8.decode(&[], &mut out), 0);
    }

    /// `decode` and `encode` append rather than replace, which is what lets a
    /// decoder's output vector feed a resampler without a copy.
    #[test]
    fn decode_and_encode_append_rather_than_clear() {
        let mut samples = vec![99.0_f32];
        SampleFormat::I16Le.decode(&[0x00, 0x40], &mut samples);
        assert_eq!(samples, [99.0, 0.5]);

        let mut bytes = vec![0xaa];
        SampleFormat::U8.encode(&[0.0], &mut bytes);
        assert_eq!(bytes, [0xaa, 0x80]);
    }

    // -- Format metadata --------------------------------------------------

    #[test]
    fn format_widths_are_what_the_wire_carries() {
        assert_eq!(SampleFormat::U8.bytes_per_sample(), 1);
        assert_eq!(SampleFormat::I16Be.bytes_per_sample(), 2);
        // Packed, not padded into four.
        assert_eq!(SampleFormat::I24Le.bytes_per_sample(), 3);
        assert_eq!(SampleFormat::I24Le.bits_per_sample(), 24);
        assert_eq!(SampleFormat::I32Le.bytes_per_sample(), 4);
        assert_eq!(SampleFormat::F64Le.bytes_per_sample(), 8);

        for format in ALL {
            assert!(
                format.bytes_per_sample() <= MAX_BYTES_PER_SAMPLE,
                "{format:?} does not fit the partial-sample buffer"
            );
            assert_eq!(
                format.is_float(),
                matches!(
                    format,
                    SampleFormat::F32Le
                        | SampleFormat::F32Be
                        | SampleFormat::F64Le
                        | SampleFormat::F64Be
                ),
                "{format:?} is on the wrong side of is_float"
            );
        }
    }

    // -- Q3: downmix, matched to decibri ----------------------------------

    /// decibri's own downmix test vectors, from
    /// `crates/decibri/src/sample.rs:313-356`, run against this
    /// implementation. If the two ever diverge, the same file decoded through
    /// each path gives two different mono results, and this is where that shows
    /// up.
    #[test]
    fn the_downmix_matches_decibris_vectors_exactly() {
        let mut mono = Vec::new();
        // test_downmix_stereo_averages_and_halves
        assert_eq!(downmix_to_mono(&[0.5, 0.3, 0.4, 0.6], 2, &mut mono), 2);
        assert!((mono[0] - 0.4).abs() < 1e-6);
        assert!((mono[1] - 0.5).abs() < 1e-6);

        // test_downmix_six_channel: a 5.1 frame summing to zero.
        mono.clear();
        downmix_to_mono(&[0.0, 0.6, 0.3, -0.3, 1.2, -1.8], 6, &mut mono);
        assert_eq!(mono.len(), 1);
        assert!((mono[0] - 0.0).abs() < 1e-6);

        // test_downmix_preserves_sign_and_magnitude
        mono.clear();
        downmix_to_mono(&[1.0, -1.0], 2, &mut mono);
        assert_eq!(mono, [0.0]);
        mono.clear();
        downmix_to_mono(&[-0.5, -0.5], 2, &mut mono);
        assert_eq!(mono, [-0.5]);

        // test_downmix_drops_trailing_partial_frame: 5 samples, 2 channels.
        mono.clear();
        assert_eq!(downmix_to_mono(&[0.2, 0.4, 0.6, 0.8, 0.9], 2, &mut mono), 2);
        assert!((mono[0] - 0.3).abs() < 1e-6);
        assert!((mono[1] - 0.7).abs() < 1e-6);

        // test_downmix_mono_passthrough, including the degenerate zero.
        mono.clear();
        downmix_to_mono(&[0.1, -0.2, 0.3], 1, &mut mono);
        assert_eq!(mono, [0.1, -0.2, 0.3]);
        mono.clear();
        downmix_to_mono(&[0.1, -0.2, 0.3], 0, &mut mono);
        assert_eq!(mono, [0.1, -0.2, 0.3]);

        // test_downmix_empty
        mono.clear();
        assert_eq!(downmix_to_mono(&[], 2, &mut mono), 0);
        assert!(mono.is_empty());
    }

    /// The formula is the mean and not the sum. Two identical channels
    /// reproduce the mono signal at its original level rather than at twice it,
    /// which is the property decibri's `open_reads_header_and_downmixes` test
    /// asserts on the file path (`crates/decibri/src/file.rs:1095`).
    #[test]
    fn the_downmix_is_a_mean_and_not_a_sum() {
        let mut mono = Vec::new();
        downmix_to_mono(&[0.7, 0.7, -0.3, -0.3], 2, &mut mono);
        assert_eq!(mono, [0.7, -0.3]);
    }

    /// The fold order is part of the answer. `f32` addition does not associate,
    /// so a downmix summing in a different order, or in `f64` and narrowed,
    /// would differ from decibri's in the last bit on some inputs. This pins
    /// the left-to-right `f32` fold.
    #[test]
    fn the_downmix_sums_left_to_right_in_f32() {
        // Chosen so that (a + b) + c and a + (b + c) differ in f32.
        let a = 1.0_f32;
        let b = 1e-8_f32;
        let c = -1.0_f32;
        let left_to_right = ((a + b) + c) / 3.0;
        let mut mono = Vec::new();
        downmix_to_mono(&[a, b, c], 3, &mut mono);
        assert_eq!(mono[0].to_bits(), left_to_right.to_bits());
    }

    // -- Interleaving -----------------------------------------------------

    #[test]
    fn interleaving_round_trips_through_planes() {
        let interleaved = [1.0_f32, -1.0, 2.0, -2.0, 3.0, -3.0];
        let planes = deinterleave(&interleaved, 2);
        assert_eq!(planes, vec![vec![1.0, 2.0, 3.0], vec![-1.0, -2.0, -3.0]]);

        let mut back = Vec::new();
        assert_eq!(interleave(&planes, &mut back), interleaved.len());
        assert_eq!(back, interleaved);
    }

    #[test]
    fn deinterleaving_drops_a_trailing_partial_frame_and_handles_the_edges() {
        // Seven samples at three channels: two whole frames, one sample over.
        let planes = deinterleave(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 3);
        assert_eq!(planes, vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]);
        // Mono is one plane, unchanged.
        assert_eq!(deinterleave(&[1.0, 2.0], 1), vec![vec![1.0, 2.0]]);
        // No channels is no planes, not a division by zero.
        assert!(deinterleave(&[1.0, 2.0], 0).is_empty());
        assert_eq!(deinterleave(&[], 2), vec![Vec::<f32>::new(); 2]);
    }

    #[test]
    fn interleaving_uneven_planes_covers_the_shortest() {
        let mut out = Vec::new();
        // Three frames in one plane, two in the other: two frames interleave.
        assert_eq!(
            interleave(&[vec![1.0, 2.0, 3.0], vec![-1.0, -2.0]], &mut out),
            4
        );
        assert_eq!(out, [1.0, -1.0, 2.0, -2.0]);

        out.clear();
        let empty: [Vec<f32>; 0] = [];
        assert_eq!(interleave(&empty, &mut out), 0);
        assert!(out.is_empty());

        // Slices work as well as owned planes, without a reshaping copy.
        out.clear();
        let left: &[f32] = &[1.0, 2.0];
        let right: &[f32] = &[-1.0, -2.0];
        assert_eq!(interleave(&[left, right], &mut out), 4);
        assert_eq!(out, [1.0, -1.0, 2.0, -2.0]);
    }

    /// Interleaving appends, like everything else that writes into a caller's
    /// buffer here.
    #[test]
    fn interleave_appends_rather_than_clears() {
        let mut out = vec![42.0_f32];
        interleave(&[vec![1.0], vec![2.0]], &mut out);
        assert_eq!(out, [42.0, 1.0, 2.0]);
    }
}
