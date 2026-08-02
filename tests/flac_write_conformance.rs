#![forbid(unsafe_code)]
//! FLAC write conformance: the writer's round trip, its edge cases, its
//! typed rejections, and the pinned witnesses over its exact bytes.
//!
//! # What the oracles here can and cannot establish
//!
//! Round-tripping [`FlacWriter`] through this crate's own reader is
//! necessary and deliberately weak: encoder and decoder share the
//! prediction arithmetic on purpose, so an error in that shared arithmetic
//! is invisible to every round trip in this file. That is coverage lesson 4
//! stated for a codec. Three things carry the independence instead:
//!
//! - The decode side of the shared arithmetic is anchored by RFC 9639's own
//!   worked examples and the conformance corpus in `flac_conformance.rs`
//!   and `flac_corpus.rs`, which the encoder cannot influence.
//! - The witness hashes below pin the writer's exact bytes, so any change
//!   in arithmetic, search or serialisation moves a pinned number.
//! - The reference decoder (`flac -t`, `flac -d`) is run over this writer's
//!   output as a release gate, recorded in the pass reports; it shares no
//!   code with this crate.
//!
//! # The input dimensions, enumerated before the tests
//!
//! 1. bit depth: 4, 8, 12, 16, 20, 24 and 32, spanning both direct frame
//!    header codes and the depths only streaminfo can express
//! 2. channel count: 1, 2, 3 and 8, spanning the stereo searches and the
//!    independent path
//! 3. compression level: 0, 5 and 8 in the matrix, all nine in the level
//!    sweep, spanning both block sizes and every search bound
//! 4. total length against the block size: shorter, exactly one block, one
//!    sample over, and several blocks with a short last frame
//! 5. content: silence, constant, tonal, noise, correlated stereo and
//!    wasted-bits material, so constant, fixed, LPC and verbatim subframes
//!    and every stereo mode are all reachable
//! 6. sample parity: the generator masks nothing, so odd values are as
//!    common as even ones, which is dimension 14 of `flac_conformance.rs`
//!    and the mid/side losslessness bit in particular

use decibri_decode::{
    identify, AudioSpec, Container, DecodeError, FlacFrameReader, FlacReader, FlacWriter, WavCodec,
    WavReader, WavWriter,
};

// -- Deterministic audio ------------------------------------------------------

/// SplitMix64, so every fixture is the same on every machine and toolchain.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Integer samples at `bits`, mixing tone, sinusoid, noise and silence so
/// every subframe type has something to win on. Nothing masks the low
/// bits: odd values appear as often as even ones.
///
/// The sinusoid segment exists for the linear predictor specifically, and
/// its absence was found by a negative control: a triangle wave is a fixed
/// predictor's home ground, so with only the other segments the encoder
/// never emitted an LPC subframe into the witness fixtures, and a break in
/// the LPC arithmetic left the encode witness green. A sinusoid follows
/// `x[n] = 2cos(w) x[n-1] - x[n-2]` with a non-integer `2cos(w)`, which no
/// fixed order can express and a quantised predictor can. The recurrence
/// is two multiplications and a subtraction, so it is deterministic
/// without a libm sine.
fn mixed_audio(count: usize, bits: u32, seed: u64) -> Vec<i64> {
    let quarter = 1i64 << (bits - 2);
    let mut random = SplitMix64(seed | 1);
    let mut sine_previous = 0.3 * quarter as f64;
    let mut sine_before = 0.0f64;
    (0..count)
        .map(|index| {
            match (index / 512) % 5 {
                // A triangle wave, a fixed predictor's home ground.
                0 => {
                    let phase = (index % 128) as i64;
                    (phase - 64) * (quarter / 64)
                }
                // Noise, which no predictor helps.
                1 => ((random.next() >> 33) as i64 % quarter) - quarter / 2,
                // Silence.
                2 => 0,
                // Quiet noise with shared low zero bits: wasted-bits
                // material. The step guard keeps the modulus non-zero at
                // the narrowest depths, where the segment degenerates to
                // silence.
                3 => {
                    let step = (quarter / 16).max(1);
                    let value = ((random.next() >> 40) as i64 % step) << 4;
                    value - quarter / 32 * 16
                }
                // The sinusoid, linear prediction's home ground. 1.9 is
                // 2cos(w) for about a fifth of the sample rate, and the
                // amplitude the seeds give stays under `quarter`.
                _ => {
                    let next = 1.9 * sine_previous - sine_before;
                    sine_before = sine_previous;
                    sine_previous = next;
                    next as i64
                }
            }
        })
        .collect()
}

/// The `f32` form of integer samples at `bits`, the crate's own scaling.
fn to_f32(samples: &[i64], bits: u32) -> Vec<f32> {
    let divisor = (1u64 << (bits - 1)) as f32;
    samples.iter().map(|&s| s as f32 / divisor).collect()
}

/// Interleaves per-channel integer audio into one `f32` stream at `bits`.
fn interleaved_f32(channels: &[Vec<i64>], bits: u32) -> Vec<f32> {
    let divisor = (1u64 << (bits - 1)) as f32;
    let frames = channels[0].len();
    let mut out = Vec::with_capacity(frames * channels.len());
    for index in 0..frames {
        for channel in channels {
            out.push(channel[index] as f32 / divisor);
        }
    }
    out
}

/// FNV-1a, the same construction every witness in this suite uses.
fn fnv1a<'a>(chunks: impl IntoIterator<Item = &'a [u8]>) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for chunk in chunks {
        for &byte in chunk {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
    }
    hash
}

// -- The round-trip matrix ----------------------------------------------------

/// Bit depth crossed with channel count crossed with level, every cell
/// encoded and decoded back to bit-identical `f32`.
///
/// The expected samples are computed here from the integers and the
/// documented scaling rule, not taken from the writer, so the round trip is
/// anchored at both ends. The MD5 the writer computed is verified by the
/// reader as part of `decode_to_end`, so every cell also exercises the
/// checksum path.
#[test]
fn the_round_trip_matrix() {
    for &bits in &[4u32, 8, 12, 16, 20, 24, 32] {
        for &channels in &[1usize, 2, 3, 8] {
            for &level in &[0u8, 5, 8] {
                let frames = 5000usize;
                let per_channel: Vec<Vec<i64>> = (0..channels)
                    .map(|channel| mixed_audio(frames, bits, 0x1000 * bits as u64 + channel as u64))
                    .collect();
                let samples = interleaved_f32(&per_channel, bits);
                let spec = AudioSpec::new(44_100, channels as u16);
                let file = FlacWriter::new(spec, bits as u8)
                    .with_level(level)
                    .to_bytes(&samples)
                    .unwrap_or_else(|e| panic!("{bits} bits {channels}ch level {level}: {e}"));

                assert!(matches!(identify(&file), Ok(Container::Flac)));
                let reader = FlacReader::new(&file)
                    .unwrap_or_else(|e| panic!("{bits} bits {channels}ch level {level}: {e}"));
                let info = reader.stream_info();
                assert_eq!(info.bits_per_sample, bits as u8);
                assert_eq!(info.spec, spec);
                assert_eq!(reader.frames(), Some(frames as u64));
                assert!(
                    info.md5.is_some(),
                    "the MD5 must be computed, not left unset"
                );
                let decoded = reader
                    .decode_to_end()
                    .unwrap_or_else(|e| panic!("{bits} bits {channels}ch level {level}: {e}"));
                assert_eq!(
                    decoded.samples(),
                    &samples[..],
                    "{bits} bits, {channels} channels, level {level}"
                );
            }
        }
    }
}

/// Lengths against the block size: shorter, exact, one over, and empty.
///
/// Level 5's block is 4096 and level 0's is 1152, so the same lengths land
/// differently against each.
#[test]
fn every_length_against_the_block_boundary_round_trips() {
    for &level in &[0u8, 5] {
        for &frames in &[0usize, 1, 15, 1151, 1152, 1153, 4095, 4096, 4097, 9000] {
            let audio = mixed_audio(frames, 16, 77);
            let samples = to_f32(&audio, 16);
            let file = FlacWriter::new(AudioSpec::mono(48_000), 16)
                .with_level(level)
                .to_bytes(&samples)
                .unwrap_or_else(|e| panic!("{frames} frames at level {level}: {e}"));
            let reader = FlacReader::new(&file)
                .unwrap_or_else(|e| panic!("{frames} frames at level {level}: {e}"));
            if frames == 0 {
                // Zero total samples is stored as the zero that means
                // unknown; the field cannot say which, and the reader
                // reports what the field can carry.
                assert_eq!(reader.frames(), None);
            } else {
                assert_eq!(reader.frames(), Some(frames as u64));
            }
            let decoded = reader
                .decode_to_end()
                .unwrap_or_else(|e| panic!("{frames} frames at level {level}: {e}"));
            assert_eq!(
                decoded.samples(),
                &samples[..],
                "{frames} frames at level {level}"
            );
        }
    }
}

// -- Edge-case content --------------------------------------------------------

/// The local transcription of the crate's quantisation rule, so expected
/// values do not come from the code under test: clamp, scale, truncate
/// toward zero, clamp in integer space.
fn quantise_16(sample: f32) -> i64 {
    let clamped = sample.clamp(-1.0, 1.0);
    ((clamped * 32_768.0) as i64).clamp(-32_768, 32_767)
}

/// Silence, constants, full scale and wasted bits, each round-tripping and
/// the compressible ones actually compressing.
///
/// The subframe type chosen for each is not observable through the public
/// API; the size bounds are the behavioural assertion. Silence and constant
/// input must land within a few bytes per frame, which only the constant
/// subframe can do; wasted-bit input must beat the same input with its low
/// bits populated.
#[test]
fn silence_constants_and_wasted_bits_compress_as_their_subframes() {
    let writer = FlacWriter::new(AudioSpec::mono(44_100), 16);

    // 5000 frames of silence: two frames of constant subframes plus the 42
    // header bytes. Anything above a few hundred bytes means verbatim was
    // chosen, which would be correct and poor.
    let silence = vec![0.0f32; 5000];
    let file = writer.to_bytes(&silence).expect("silence encodes");
    assert!(
        file.len() < 300,
        "silence took {} bytes; a constant subframe was not chosen",
        file.len()
    );
    let decoded = FlacReader::new(&file)
        .expect("parse")
        .decode_to_end()
        .expect("decode");
    assert_eq!(decoded.samples(), &silence[..]);

    // A constant non-zero value.
    let constant = vec![0.25f32; 5000];
    let file = writer.to_bytes(&constant).expect("constant encodes");
    assert!(
        file.len() < 300,
        "a constant signal took {} bytes",
        file.len()
    );
    let decoded = FlacReader::new(&file)
        .expect("parse")
        .decode_to_end()
        .expect("decode");
    assert_eq!(decoded.samples(), &constant[..]);

    // Full-scale input: +1.0 clamps to 32767 and comes back as 32767/32768,
    // -1.0 is exactly representable. The expected values are computed by
    // the local transcription of the rule.
    let full_scale: Vec<f32> = (0..4096)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let expected: Vec<f32> = full_scale
        .iter()
        .map(|&s| quantise_16(s) as f32 / 32_768.0)
        .collect();
    let file = writer.to_bytes(&full_scale).expect("full scale encodes");
    let decoded = FlacReader::new(&file)
        .expect("parse")
        .decode_to_end()
        .expect("decode");
    assert_eq!(decoded.samples(), &expected[..]);

    // The same noise with and without six low zero bits. With the
    // declaration the shifted copy costs about what the unshifted one
    // costs; without it every sample would carry six more bits, 6,144
    // bytes here. The bound splits the two outcomes.
    let mut random = SplitMix64(99);
    let noisy: Vec<i64> = (0..8192)
        .map(|_| ((random.next() >> 55) as i64) - 256)
        .collect();
    let shifted: Vec<i64> = noisy.iter().map(|&v| v << 6).collect();
    let noisy_file = writer.to_bytes(&to_f32(&noisy, 16)).expect("noise encodes");
    let shifted_file = writer
        .to_bytes(&to_f32(&shifted, 16))
        .expect("shifted noise encodes");
    assert!(
        shifted_file.len() < noisy_file.len() + 1024,
        "wasted bits were not declared: {} vs {}",
        shifted_file.len(),
        noisy_file.len()
    );
    let decoded = FlacReader::new(&shifted_file)
        .expect("parse")
        .decode_to_end()
        .expect("decode");
    assert_eq!(decoded.samples(), &to_f32(&shifted, 16)[..]);
}

/// Tonal material compresses hard; incompressible material stays within a
/// verbatim-sized envelope. These are the gross behavioural bounds a broken
/// search cannot meet.
#[test]
fn compression_actually_happens_where_it_can() {
    // A sawtooth: linear along each period, so prediction removes nearly
    // everything except the discontinuities.
    let tonal: Vec<i64> = (0..16_384).map(|i| ((i % 200) as i64 - 100) * 80).collect();
    let samples = to_f32(&tonal, 16);
    let file = FlacWriter::new(AudioSpec::mono(44_100), 16)
        .to_bytes(&samples)
        .expect("tonal encodes");
    let raw_bytes = tonal.len() * 2;
    // Measured at 9,910 bytes against 32,768 raw; the bound leaves room for
    // search changes without letting a broken search (which lands near the
    // raw size) pass.
    assert!(
        file.len() < raw_bytes / 3,
        "a sawtooth compressed to only {} of {raw_bytes} bytes",
        file.len()
    );

    // Full-depth noise is incompressible; the file must stay within a small
    // envelope of the raw size rather than ballooning.
    let mut random = SplitMix64(123);
    let noise: Vec<i64> = (0..16_384)
        .map(|_| ((random.next() >> 48) as i64) - 32_768)
        .collect();
    let samples = to_f32(&noise, 16);
    let file = FlacWriter::new(AudioSpec::mono(44_100), 16)
        .to_bytes(&samples)
        .expect("noise encodes");
    assert!(
        file.len() < raw_bytes + raw_bytes / 32 + 256,
        "noise took {} bytes against {raw_bytes} raw",
        file.len()
    );
}

// -- Levels -------------------------------------------------------------------

/// Every level round-trips the same audio, and searching harder does not
/// produce a larger file on compressible stereo material.
#[test]
fn every_level_round_trips_and_the_search_pays() {
    let frames = 12_000usize;
    let left = mixed_audio(frames, 16, 0xA);
    // A correlated right channel, so the stereo search has something real
    // to find.
    let right: Vec<i64> = left
        .iter()
        .enumerate()
        .map(|(i, &l)| l - (i as i64 % 7) + 3)
        .collect();
    let samples = interleaved_f32(&[left, right], 16);
    let spec = AudioSpec::new(44_100, 2);

    let mut sizes = Vec::new();
    for level in 0u8..=8 {
        let file = FlacWriter::new(spec, 16)
            .with_level(level)
            .to_bytes(&samples)
            .unwrap_or_else(|e| panic!("level {level}: {e}"));
        let decoded = FlacReader::new(&file)
            .unwrap_or_else(|e| panic!("level {level}: {e}"))
            .decode_to_end()
            .unwrap_or_else(|e| panic!("level {level}: {e}"));
        assert_eq!(decoded.samples(), &samples[..], "level {level}");
        sizes.push(file.len());
    }
    println!("level sizes: {sizes:?}");
    assert!(
        sizes[8] <= sizes[0],
        "level 8 ({}) produced more bytes than level 0 ({})",
        sizes[8],
        sizes[0]
    );
    assert!(
        sizes[5] <= sizes[2],
        "level 5 ({}) produced more bytes than level 2 ({})",
        sizes[5],
        sizes[2]
    );
}

// -- Independent anchors ------------------------------------------------------

/// The same integers through the WAV writer and the FLAC writer decode to
/// the same samples.
///
/// The WAV path's scaling is anchored against decibri and against
/// hand-built fixtures, so agreement here anchors the FLAC writer's
/// quantisation to something outside this file, which is coverage lesson 4:
/// an oracle must be independent of the path under test.
#[test]
fn flac_and_wav_agree_on_the_same_audio() {
    let audio = mixed_audio(6000, 16, 0xBEEF);
    let samples = to_f32(&audio, 16);
    let spec = AudioSpec::new(32_000, 1);

    let wav = WavWriter::new(spec, WavCodec::PcmI16)
        .to_bytes(&samples)
        .expect("wav encodes");
    let flac = FlacWriter::new(spec, 16)
        .to_bytes(&samples)
        .expect("flac encodes");

    let from_wav = WavReader::new(&wav).expect("wav parses").decode_to_end();
    let from_flac = FlacReader::new(&flac)
        .expect("flac parses")
        .decode_to_end()
        .expect("flac decodes");
    assert_eq!(from_wav.samples(), from_flac.samples());
}

/// The writer's frames are self-describing: stripped of the 42-byte header,
/// they decode through the bare frame reader with every property derived
/// from the first frame.
#[test]
fn the_written_frames_stand_alone() {
    let audio = mixed_audio(6000, 16, 0xF00D);
    let samples = to_f32(&audio, 16);
    // 44100 Hz and 16 bits both have direct frame header codes, so nothing
    // in the frames defers to the streaminfo block being stripped.
    let file = FlacWriter::new(AudioSpec::mono(44_100), 16)
        .to_bytes(&samples)
        .expect("encodes");
    let (decoded, report) = FlacFrameReader::new(&file[42..])
        .expect("the frames open at a frame boundary")
        .decode_to_end()
        .expect("the frames decode alone");
    assert_eq!(decoded.samples(), &samples[..]);
    assert_eq!(report.samples, samples.len());
}

/// What streaminfo declares is what was written: block sizes, frame size
/// bounds, the total and the checksum.
#[test]
fn streaminfo_states_what_was_written() {
    let audio = mixed_audio(10_000, 16, 0xACE);
    let samples = to_f32(&audio, 16);
    let file = FlacWriter::new(AudioSpec::mono(44_100), 16)
        .to_bytes(&samples)
        .expect("encodes");
    let reader = FlacReader::new(&file).expect("parses");
    let info = reader.stream_info();
    // Level 5's nominal block, stated as both bounds because every frame
    // but the last is exactly nominal.
    assert_eq!(info.min_block_size, 4096);
    assert_eq!(info.max_block_size, 4096);
    assert_eq!(info.total_samples, Some(10_000));
    assert!(info.md5.is_some());
    let smallest = info.min_frame_size.expect("frame sizes are stated") as usize;
    let largest = info.max_frame_size.expect("frame sizes are stated") as usize;
    assert!(smallest <= largest);
    assert!(largest < file.len());
    // The smallest possible frame is a constant-subframe frame: header,
    // one sample, footer. Nothing can be smaller than ten bytes.
    assert!(smallest >= 10);
}

// -- Typed rejections ---------------------------------------------------------

/// Impossible configurations are typed errors naming the constraint, never
/// a panic and never a file.
#[test]
fn impossible_configurations_are_typed_errors() {
    let samples = [0.0f32; 8];

    let error = FlacWriter::new(AudioSpec::new(44_100, 0), 16)
        .to_bytes(&samples)
        .expect_err("no channels");
    assert!(matches!(
        error,
        DecodeError::UnsupportedChannelLayout { channels: 0 }
    ));

    let error = FlacWriter::new(AudioSpec::new(44_100, 9), 16)
        .to_bytes(&[0.0; 9])
        .expect_err("nine channels");
    assert!(matches!(
        error,
        DecodeError::UnsupportedChannelLayout { channels: 9 }
    ));

    let error = FlacWriter::new(AudioSpec::mono(0), 16)
        .to_bytes(&samples)
        .expect_err("zero rate");
    assert!(matches!(error, DecodeError::Malformed { .. }));

    // One past streaminfo's 20-bit field.
    let error = FlacWriter::new(AudioSpec::mono(1_048_576), 16)
        .to_bytes(&samples)
        .expect_err("rate past 20 bits");
    assert!(matches!(error, DecodeError::Malformed { .. }));

    for bits in [0u8, 3, 33, 255] {
        let error = FlacWriter::new(AudioSpec::mono(44_100), bits)
            .to_bytes(&samples)
            .expect_err("bit depth outside 4..=32");
        assert!(
            matches!(
                error,
                DecodeError::UnsupportedSampleFormat {
                    bits_per_sample, ..
                } if bits_per_sample == u16::from(bits)
            ),
            "bits {bits}: {error}"
        );
    }

    let error = FlacWriter::new(AudioSpec::mono(44_100), 16)
        .with_level(9)
        .to_bytes(&samples)
        .expect_err("level 9");
    assert!(matches!(error, DecodeError::Malformed { .. }));

    // A partial trailing frame: three samples into two channels.
    let error = FlacWriter::new(AudioSpec::new(44_100, 2), 16)
        .to_bytes(&[0.0, 0.0, 0.0])
        .expect_err("partial frame");
    assert!(matches!(error, DecodeError::Truncated { .. }));

    // The extremes that are legal must stay legal.
    for (rate, bits) in [(1u32, 4u8), (1_048_575, 32)] {
        let file = FlacWriter::new(AudioSpec::mono(rate), bits)
            .to_bytes(&samples)
            .unwrap_or_else(|e| panic!("rate {rate} bits {bits}: {e}"));
        let decoded = FlacReader::new(&file)
            .expect("parses")
            .decode_to_end()
            .expect("decodes");
        assert_eq!(decoded.samples().len(), samples.len());
    }
}

/// Values a `f32` can hold that no sample should: NaN becomes silence and
/// infinities clamp to full scale, the crate's one quantisation rule.
#[test]
fn hostile_float_values_quantise_by_the_stated_rule() {
    let samples = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 2.0, -2.0];
    let expected: Vec<f32> = [0i64, 32_767, -32_768, 32_767, -32_768]
        .iter()
        .map(|&v| v as f32 / 32_768.0)
        .collect();
    let file = FlacWriter::new(AudioSpec::mono(8_000), 16)
        .to_bytes(&samples)
        .expect("encodes");
    let decoded = FlacReader::new(&file)
        .expect("parses")
        .decode_to_end()
        .expect("decodes");
    assert_eq!(decoded.samples(), &expected[..]);
}

// -- Determinism --------------------------------------------------------------

/// The writer's exact bytes, pinned.
///
/// The same fixtures at every level hash to the same value on every
/// platform, toolchain and optimisation level, which is the encoder half of
/// the crate's byte-identical claim. A hash that moves on unchanged input
/// is a determinism break or an intended output change; it is re-pinned
/// only for the second, stated in the changelog.
#[test]
fn flac_writer_output_is_bit_identical_to_a_pinned_witness() {
    // Stereo 16-bit through every level.
    let frames = 9000usize;
    let left = mixed_audio(frames, 16, 0x5EED_0001);
    let right: Vec<i64> = left
        .iter()
        .zip(mixed_audio(frames, 16, 0x5EED_0002))
        .map(|(&l, r)| (l + r) / 2)
        .collect();
    let stereo = interleaved_f32(&[left, right], 16);
    let mut outputs: Vec<Vec<u8>> = Vec::new();
    for level in 0u8..=8 {
        outputs.push(
            FlacWriter::new(AudioSpec::new(44_100, 2), 16)
                .with_level(level)
                .to_bytes(&stereo)
                .unwrap_or_else(|e| panic!("level {level}: {e}")),
        );
    }

    // 24-bit mono and 32-bit stereo at the default level: the depths where
    // integer width and quantisation differ most.
    let mono24 = to_f32(&mixed_audio(7000, 24, 0x5EED_0003), 24);
    outputs.push(
        FlacWriter::new(AudioSpec::mono(96_000), 24)
            .to_bytes(&mono24)
            .expect("24-bit encodes"),
    );
    let wide = interleaved_f32(
        &[
            mixed_audio(5000, 32, 0x5EED_0004),
            mixed_audio(5000, 32, 0x5EED_0005),
        ],
        32,
    );
    outputs.push(
        FlacWriter::new(AudioSpec::new(192_000, 2), 32)
            .to_bytes(&wide)
            .expect("32-bit encodes"),
    );

    let witness = fnv1a(outputs.iter().map(Vec::as_slice));
    assert_eq!(
        witness, 0x7d49_2b0a_1148_06b4,
        "the FLAC writer's bytes moved: witness {witness:#018x}"
    );

    // The same calls twice produce identical bytes within one process too.
    let again = FlacWriter::new(AudioSpec::new(44_100, 2), 16)
        .with_level(5)
        .to_bytes(&stereo)
        .expect("re-encode");
    assert_eq!(outputs[5], again);
}
