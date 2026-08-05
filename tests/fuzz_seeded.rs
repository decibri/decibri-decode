#![forbid(unsafe_code)]
//! A seeded, deterministic fuzz gate over every public entry point that takes
//! arbitrary bytes.
//!
//! # What this is, and what it is not
//!
//! The crate's other suites assert what correct and near-correct inputs
//! decode to. This one asserts something narrower over a much wilder input
//! space: **no input, however damaged, panics, hangs the caller loop, or
//! makes an error path misbehave**. Every input is derived from a fixed seed
//! corpus by a fixed mutation schedule under a hand-rolled PRNG, so the whole
//! run is a pure function of the constants in this file: a failure here
//! reproduces forever, on every machine, with no corpus to fetch and no
//! dependency to build.
//!
//! Deterministic seeding is this gate's strength and its limit. It replays
//! byte-for-byte in CI, and it discovers nothing a mutation of its fixed
//! seeds cannot reach; the wider randomized exploration ran out of tree
//! during the pre-publish audit, and anything such a run finds is committed
//! into [`REGRESSIONS`] as literal bytes, so the finding outlives the
//! generator that found it.
//!
//! # The seed corpus
//!
//! Built in-process from the crate's own writers, one well-formed file per
//! container family plus RFC 9639 appendix D.3's worked example and a bare
//! frame run cut from it. Mutations start from structurally real files
//! because random bytes rarely survive a magic check: the depth of a parser
//! a fuzzer reaches is bounded by how real its inputs look.
//!
//! # The headerless decoders
//!
//! [`PcmDecoder`] and [`G711Decoder`] parse nothing. A headerless stream
//! carries no header to check a caller's claim against, so the format, the
//! law and the layout are asserted at construction and every byte after that
//! is audio. They are driven here for the property the file's title states:
//! taking arbitrary bytes is what they have in common with a parser, and the
//! stall a mis-implemented `feed` produces is the same stall a mis-implemented
//! `push` produces.
//!
//! The sample format, the companding law and the channel count are a varied
//! dimension rather than a fixed choice: drawn from a generator of their own
//! over the mutation run, so the mutation schedule above them is unchanged by
//! their presence, and cycled over the much shorter unmutated-seed run, where
//! a draw is too short to be relied on. Both runs assert at the end which
//! values were actually reached, rather than reading the spread off the
//! tables they came from.

use decibri_decode::{
    decode, identify, AiffCodec, AiffStreamDecoder, AiffWriter, AudioSpec, AudioStreamDecoder,
    Decoder, FlacFrameReader, FlacReader, FlacRecovery, FlacStreamDecoder, FlacWriter, G711Decoder,
    G711Law, PcmDecoder, SampleFormat, StreamSource, WavCodec, WavHeaderStyle, WavStreamDecoder,
    WavWriter,
};

/// How many mutated inputs each seed family contributes. Sized so the whole
/// gate runs in seconds in a debug build while still walking every parser
/// several hundred times per seed.
const ITERATIONS: usize = 1_200;

/// Inputs that once crashed a decoder, kept as the bytes that did it.
///
/// Every crash the pre-publish out-of-tree fuzzer finds is reduced and
/// committed here, so the regression outlives the schedule that found it.
///
/// An empty list is a statement rather than a placeholder. The pre-publish
/// audit ran 446,685,717 executions across six decode targets and 1,063,494
/// more over the three writers, under a release profile with
/// `overflow-checks` and `debug-assertions` both on, and found no panic to
/// record. The one defect that campaign did find was disproportionate work
/// rather than a crash, so its regression is a cumulative-allocation bound in
/// `allocation_ceiling.rs` and not a byte sequence here.
const REGRESSIONS: &[&[u8]] = &[];

/// xorshift64*: deterministic, dependency-free, and stated in full so the
/// stream can never drift with a library version.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

/// Values worth planting where a size or count field might sit: the zeros,
/// the ones-past-a-limit, and the sign and wrap boundaries.
const INTERESTING: [u32; 12] = [
    0,
    1,
    2,
    16,
    255,
    256,
    65_535,
    65_536,
    0x00FF_FFFF,
    0x7FFF_FFFF,
    0x8000_0000,
    0xFFFF_FFFF,
];

/// One mutated input, derived from `seeds` under `rng`.
fn mutate(rng: &mut Rng, seeds: &[Vec<u8>]) -> Vec<u8> {
    // One input in sixteen is pure noise, so the magic checks and the
    // shallowest rejection paths stay exercised too.
    if rng.below(16) == 0 {
        let len = rng.below(512) + 1;
        return (0..len).map(|_| rng.next() as u8).collect();
    }
    let mut data = seeds[rng.below(seeds.len())].clone();
    let rounds = 1 + rng.below(12);
    for _ in 0..rounds {
        if data.is_empty() {
            data.push(rng.next() as u8);
        }
        match rng.below(8) {
            0 => {
                let at = rng.below(data.len());
                data[at] ^= 1 << rng.below(8);
            }
            1 => {
                let at = rng.below(data.len());
                data[at] = rng.next() as u8;
            }
            2 => {
                let to = rng.below(data.len());
                data.truncate(to.max(1));
            }
            3 => {
                let at = rng.below(data.len());
                let n = (1 + rng.below(8)).min(data.len() - at);
                data.drain(at..at + n);
            }
            4 => {
                let at = rng.below(data.len() + 1);
                let n = 1 + rng.below(8);
                let junk: Vec<u8> = (0..n).map(|_| rng.next() as u8).collect();
                data.splice(at..at, junk);
            }
            5 => {
                if data.len() >= 4 {
                    let at = rng.below(data.len() - 3);
                    let value = INTERESTING[rng.below(INTERESTING.len())];
                    let bytes = if rng.below(2) == 0 {
                        value.to_le_bytes()
                    } else {
                        value.to_be_bytes()
                    };
                    data[at..at + 4].copy_from_slice(&bytes);
                }
            }
            6 => {
                // Splice this input's head onto another seed's tail.
                let other = &seeds[rng.below(seeds.len())];
                if !other.is_empty() {
                    let cut = rng.below(data.len());
                    let from = rng.below(other.len());
                    data.truncate(cut.max(1));
                    data.extend_from_slice(&other[from..]);
                }
            }
            _ => {
                // Overwrite a run with one byte: the shape of a bad sector.
                let at = rng.below(data.len());
                let n = (1 + rng.below(64)).min(data.len() - at);
                let fill = rng.next() as u8;
                data[at..at + n].fill(fill);
            }
        }
    }
    // Recovery scans every byte and trial-decodes every plausible sync
    // point, so unbounded inputs would make this gate's cost quadratic in
    // the worst case rather than a few seconds.
    data.truncate(16_384);
    data
}

/// RFC 9639 appendix D.3's worked example: 8-bit mono, 24 samples, with its
/// audio frame starting at byte 42.
const D3: [u8; 73] = [
    0x66, 0x4c, 0x61, 0x43, 0x80, 0x00, 0x00, 0x22, 0x10, 0x00, 0x10, 0x00, 0x00, 0x00, 0x1f, 0x00,
    0x00, 0x1f, 0x07, 0xd0, 0x00, 0x70, 0x00, 0x00, 0x00, 0x18, 0xf8, 0xf9, 0xe3, 0x96, 0xf5, 0xcb,
    0xcf, 0xc6, 0xdc, 0x80, 0x7f, 0x99, 0x77, 0x90, 0x6b, 0x32, 0xff, 0xf8, 0x68, 0x02, 0x00, 0x17,
    0xe9, 0x44, 0x00, 0x4f, 0x6f, 0x31, 0x3d, 0x10, 0x47, 0xd2, 0x27, 0xcb, 0x6d, 0x09, 0x08, 0x31,
    0x45, 0x2b, 0xdc, 0x28, 0x22, 0x22, 0x80, 0x57, 0xa3,
];

/// The seed corpus: one well-formed file per shape the crate reads.
fn seeds() -> Vec<Vec<u8>> {
    let mono: Vec<f32> = (0..400)
        .map(|i| ((i % 200) as f32 - 100.0) / 128.0)
        .collect();
    let stereo: Vec<f32> = (0..800)
        .map(|i| ((i % 320) as f32 - 160.0) / 256.0)
        .collect();

    let mut corpus: Vec<Vec<u8>> = vec![
        WavWriter::new(AudioSpec::mono(16_000), WavCodec::PcmI16)
            .to_bytes(&mono)
            .expect("a seed WAV writes"),
        WavWriter::new(AudioSpec::new(44_100, 2), WavCodec::PcmI24)
            .with_header_style(WavHeaderStyle::Extensible)
            .to_bytes(&stereo)
            .expect("a seed extensible WAV writes"),
        WavWriter::new(AudioSpec::mono(8_000), WavCodec::MuLaw)
            .to_bytes(&mono)
            .expect("a seed mu-law WAV writes"),
        AiffWriter::new(AudioSpec::mono(16_000), AiffCodec::PcmI16)
            .to_bytes(&mono)
            .expect("a seed AIFF writes"),
        AiffWriter::new(AudioSpec::new(44_100, 2), AiffCodec::PcmI24Sowt)
            .to_bytes(&stereo)
            .expect("a seed sowt AIFF-C writes"),
        AiffWriter::new(AudioSpec::mono(48_000), AiffCodec::Float32)
            .to_bytes(&mono)
            .expect("a seed float AIFF-C writes"),
        FlacWriter::new(AudioSpec::mono(16_000), 16)
            .with_level(0)
            .to_bytes(&mono)
            .expect("a seed FLAC writes"),
        FlacWriter::new(AudioSpec::new(44_100, 2), 24)
            .to_bytes(&stereo)
            .expect("a seed stereo FLAC writes"),
        D3.to_vec(),
        D3[42..].to_vec(),
    ];
    // A damaged recovery shape: junk that almost syncs, then real frames.
    let mut junk_front = vec![0xFF, 0xF8, 0x00, 0x11, 0x22];
    junk_front.extend_from_slice(&D3[42..]);
    corpus.push(junk_front);
    corpus
}

/// Drives a [`StreamSource`] over `data` in `chunk`-byte pieces the way the
/// trait documents, tolerating errors and stopping on the documented
/// terminal state, so the only way this fails is a panic or a stall the
/// bounded loop turns into one.
fn drive(source: &mut dyn StreamSource, data: &[u8], chunk: usize, out: &mut Vec<f32>) {
    for piece in data.chunks(chunk.max(1)) {
        let mut offset = 0;
        let mut stalls = 0u32;
        while offset < piece.len() {
            match source.push(&piece[offset..]) {
                Ok(0) => match source.pull(out, usize::MAX) {
                    // Nothing taken and nothing ready is the documented end
                    // state of a failed stream; anything else pushing on
                    // would loop forever, which is itself the finding.
                    Ok(0) => {
                        stalls += 1;
                        assert!(
                            stalls <= 2,
                            "a live stream took no bytes and produced no samples"
                        );
                    }
                    Ok(_) => stalls = 0,
                    Err(_) => return,
                },
                Ok(taken) => {
                    offset += taken;
                    stalls = 0;
                    if source.pull(out, usize::MAX).is_err() {
                        return;
                    }
                }
                Err(_) => return,
            }
        }
    }
    let _ = source.finish(out);
    let _ = source.pull(out, usize::MAX);
}

/// Every sample format the crate reads: each width once per byte order, and
/// the two one-byte formats that have no byte order to vary.
const PCM_FORMATS: [SampleFormat; 12] = [
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

/// Both companding laws, so neither G.711 table is the one the gate skips.
const G711_LAWS: [G711Law; 2] = [G711Law::MuLaw, G711Law::ALaw];

/// The channel counts the headerless decoders are driven at.
///
/// Zero is in the list because [`AudioSpec`] accepts it and both decoders
/// document what they do with it, so it is a reachable state rather than an
/// impossible one. Three divides none of the widths above, so an input that
/// ends on a sample boundary at three channels rarely ends on a frame one.
const HEADERLESS_CHANNELS: [u16; 5] = [0, 1, 2, 3, 8];

/// The rate every headerless draw is made at.
///
/// Fixed rather than varied: no code path in either decoder reads it, so a
/// second value would widen the matrix without widening the coverage.
const HEADERLESS_RATE: u32 = 16_000;

/// One draw of the headerless dimension: what a caller claims the bytes are.
///
/// Nothing in either decoder can check any of it, so every combination is a
/// legal thing to assert and the gate has to survive all of them.
#[derive(Clone, Copy)]
struct Headerless {
    format: SampleFormat,
    law: G711Law,
    spec: AudioSpec,
}

impl Headerless {
    /// A draw from `rng`, which is a generator of its own so that adding this
    /// dimension leaves the mutation stream above it byte-for-byte unchanged,
    /// and so that the format cannot lock to one chunk size the way indexing
    /// both by the iteration number would.
    ///
    /// Used where the run is long enough for a draw to reach every value,
    /// which is asserted rather than assumed.
    fn draw(rng: &mut Rng) -> Self {
        Self {
            format: PCM_FORMATS[rng.below(PCM_FORMATS.len())],
            law: G711_LAWS[rng.below(G711_LAWS.len())],
            spec: AudioSpec::new(
                HEADERLESS_RATE,
                HEADERLESS_CHANNELS[rng.below(HEADERLESS_CHANNELS.len())],
            ),
        }
    }

    /// The `index`th step of a cycle through all three tables at once.
    ///
    /// Used where the run is a few dozen inputs rather than a few thousand.
    /// The table lengths are 12, 2 and 5, so a cycle long enough to exhaust
    /// the longest exhausts all three, and no run of that length has to rely
    /// on a draw happening to be even.
    fn cycle(index: usize) -> Self {
        Self {
            format: PCM_FORMATS[index % PCM_FORMATS.len()],
            law: G711_LAWS[index % G711_LAWS.len()],
            spec: AudioSpec::new(
                HEADERLESS_RATE,
                HEADERLESS_CHANNELS[index % HEADERLESS_CHANNELS.len()],
            ),
        }
    }
}

/// What a run of draws actually reached, so the spread is asserted from
/// observation rather than from the tables it was drawn from.
#[derive(Default)]
struct Reached {
    formats: Vec<SampleFormat>,
    laws: Vec<G711Law>,
    channels: Vec<u16>,
}

impl Reached {
    /// Notes one draw.
    fn record(&mut self, draw: &Headerless) {
        if !self.formats.contains(&draw.format) {
            self.formats.push(draw.format);
        }
        if !self.laws.contains(&draw.law) {
            self.laws.push(draw.law);
        }
        if !self.channels.contains(&draw.spec.channels) {
            self.channels.push(draw.spec.channels);
        }
    }

    /// Fails unless every value in every table above was actually drawn.
    ///
    /// A drawn dimension that happened to miss half its table would report a
    /// spread it did not have, and the miss would be silent.
    fn assert_every_value_was_reached(&self) {
        for format in PCM_FORMATS {
            assert!(
                self.formats.contains(&format),
                "no headerless dimension reached {format:?}"
            );
        }
        for law in G711_LAWS {
            assert!(
                self.laws.contains(&law),
                "no headerless dimension reached {law:?}"
            );
        }
        for channels in HEADERLESS_CHANNELS {
            assert!(
                self.channels.contains(&channels),
                "no headerless dimension reached {channels} channel(s)"
            );
        }
    }
}

/// Drives a [`Decoder`] over `data` in `chunk`-byte pieces the way the trait
/// documents, tolerating errors and stopping on the documented terminal
/// state, so the only way this fails is a panic or a stall the bounded loop
/// turns into one.
fn drive_decoder(decoder: &mut dyn Decoder, data: &[u8], chunk: usize, out: &mut Vec<f32>) {
    for piece in data.chunks(chunk.max(1)) {
        let mut offset = 0;
        let mut stalls = 0u32;
        while offset < piece.len() {
            match decoder.feed(&piece[offset..]) {
                Ok(0) => match decoder.decode(out) {
                    // Nothing taken and nothing ready leaves a caller
                    // following the documented loop with no way forward, so
                    // pushing on would loop forever, which is the finding.
                    Ok(0) => {
                        stalls += 1;
                        assert!(
                            stalls <= 2,
                            "a live decoder took no bytes and produced no samples"
                        );
                    }
                    Ok(_) => stalls = 0,
                    Err(_) => return,
                },
                Ok(taken) => {
                    offset += taken;
                    stalls = 0;
                    if decoder.decode(out).is_err() {
                        return;
                    }
                }
                Err(_) => return,
            }
        }
    }
    let _ = decoder.flush(out);
    // `reset` returns the decoder to its just-constructed state, so the same
    // bytes have to be drivable again through the same instance, including
    // after the flush above rejected them.
    decoder.reset();
    let _ = decoder.feed(data);
    let _ = decoder.decode(out);
    let _ = decoder.flush(out);
}

/// Every parser in the crate and both headerless decoders, over one input.
/// Errors are the expected currency here and are discarded; a panic is the
/// failure.
fn exercise(data: &[u8], chunk: usize, headerless: Headerless, out: &mut Vec<f32>) {
    let _ = identify(data);
    let _ = decode(data);
    if let Ok(reader) = FlacReader::new(data) {
        let _ = reader.decode_to_end();
    }
    if let Ok(reader) = FlacFrameReader::new(data) {
        out.clear();
        let _ = reader.decode(out);
    }
    out.clear();
    let _ = FlacRecovery::new(data).decode(out);

    out.clear();
    drive(&mut AudioStreamDecoder::new(), data, chunk, out);
    out.clear();
    drive(&mut WavStreamDecoder::new(), data, chunk, out);
    out.clear();
    drive(&mut AiffStreamDecoder::new(), data, chunk, out);
    out.clear();
    drive(&mut FlacStreamDecoder::new(), data, chunk, out);
    out.clear();
    drive(&mut FlacStreamDecoder::frames(), data, chunk, out);

    out.clear();
    drive_decoder(
        &mut PcmDecoder::new(headerless.format, headerless.spec),
        data,
        chunk,
        out,
    );
    out.clear();
    drive_decoder(
        &mut G711Decoder::new(headerless.law, headerless.spec),
        data,
        chunk,
        out,
    );
}

#[test]
fn no_seeded_mutation_panics_any_parser() {
    let seeds = seeds();
    let mut reached = Reached::default();
    // Two fixed generators rather than one, so a schedule change in half the
    // run cannot silently shorten the other half's coverage.
    for seed in [0x5EED_0001_u64, 0xD15E_A5ED] {
        let mut rng = Rng(u64::from(seed as u32) | 1);
        let mut dimension = Rng(seed ^ 0x0DEC_0DE5_0DEC_0DE5);
        let mut out = Vec::new();
        for iteration in 0..ITERATIONS {
            let data = mutate(&mut rng, &seeds);
            // Chunk sizes cycle so streaming state machines meet every input
            // at several boundaries, including mid-header ones.
            let chunk = [1, 3, 7, 64, 1_021, usize::MAX][iteration % 6];
            let headerless = Headerless::draw(&mut dimension);
            reached.record(&headerless);
            exercise(&data, chunk, headerless, &mut out);
        }
    }
    reached.assert_every_value_was_reached();
}

#[test]
fn the_unmutated_seeds_and_every_regression_input_decode_without_panicking() {
    let mut out = Vec::new();
    let mut reached = Reached::default();
    let mut step = 0usize;
    for seed in seeds() {
        for chunk in [1, 7, usize::MAX] {
            let headerless = Headerless::cycle(step);
            step += 1;
            reached.record(&headerless);
            exercise(&seed, chunk, headerless, &mut out);
        }
    }
    for input in REGRESSIONS {
        for chunk in [1, 7, usize::MAX] {
            let headerless = Headerless::cycle(step);
            step += 1;
            reached.record(&headerless);
            exercise(input, chunk, headerless, &mut out);
        }
    }
    reached.assert_every_value_was_reached();
}
