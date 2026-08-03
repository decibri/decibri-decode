<!-- markdownlint-disable MD033 MD038 MD041 -->

<p align="center">
  <a href="https://decibri.com">
    <img
      src="https://github.com/user-attachments/assets/6584c5b5-007d-4f6a-8d9c-60ac7cbd97b3"
      alt="Decibri Decode"
      width="100%">
  </a>
</p>

# decibri-decode

Turn audio into the format you need to work with.

Get back `f32` samples, with the sample rate and channel count travelling
with them, from any WAV, AIFF or FLAC file, a stream arriving over a socket,
or raw telephony audio with no header at all. Pure Rust, and the same bytes
give you the same samples on every platform.

<a href="https://crates.io/crates/decibri-decode"><img src="https://img.shields.io/crates/v/decibri-decode.svg" alt="Crates.io"></a>&nbsp;
<a href="https://github.com/decibri/decibri-decode/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="Apache 2.0 License"></a>

## What you can do with it

**Automatically detects formats for you.** WAV, AIFF and FLAC, in every common
sample format. One call reads the leading twelve bytes, works out which format
it has from the file itself rather than from the extension, and hands it to the
right reader, so a mislabelled file still opens. Twelve rather than four
because `RIFF` at the front of a file means a container family and not a
format, and an AVI is a RIFF file too. A container this crate does not read is
an error that names what it found.

**Decode audio as it arrives.** Push bytes in as they come off a socket and
pull samples out as they become ready, with no need to hold the whole file.

**Handle telephony audio.** G.711 mu-law and A-law, both directions, including
the headerless form that phone systems actually send.

**Read FLAC with no header on it.** A FLAC frame describes itself, so a bare
run of frames decodes with nothing in front of it, which is what a container
such as Ogg or Matroska hands over. You can supply the stream's `STREAMINFO`
block separately if you have one, and the decoder tells you whether it managed
to verify the audio against a checksum. Headerless input is always something
you state rather than something the crate guesses at, and the same goes for
raw PCM and raw G.711.

**Recover audio from a damaged file.** Ask for it explicitly and the decoder
will find frame boundaries in bytes that start part way through a frame, carry
junk, or are corrupted in the middle. It tells you exactly which byte ranges
produced nothing and which frames you did not get; it never quietly hands
back short audio. Recovery works on a buffer you already hold, so a live
stream that goes bad part way through stops with an error rather than picking
itself back up.

**Write WAV, AIFF and FLAC files.** WAV and AIFF cover every encoding the
readers decode. WAV is written as plain RIFF, or as RF64 when the file
outgrows RIFF's 32-bit sizes. AIFF is written as plain AIFF, or as AIFF-C
when the encoding needs the compression field. FLAC is written at
compression levels 0 through 8 with 5 the default, at any bit depth from 4
to 32, with the audio's checksum computed and stored so any decoder can
verify what it reads. On a fifteen file selection of the conformance
corpus, the output at level 5 measured 0.04 percent smaller in aggregate
than flac 1.5.0 at the same level.

## Formats

| Format | Read | Write |
| --- | --- | --- |
| WAV | yes | yes |
| AIFF and AIFF-C | yes | yes |
| FLAC | yes | yes |
| Raw PCM, no header | yes | yes |
| G.711, no header | yes | yes |

Files are identified from their contents rather than their names. Headerless
audio has no signature to identify, so you tell the crate what it is.

## Mixing and rate conversion

Samples come back the way the source holds them, interleaved and at the
source's own rate, with the rate and channel count attached so nothing
downstream has to guess. Mixing to mono is `downmix_to_mono`, which this crate
provides. Changing the sample rate is
[`decibri-resampler`](https://crates.io/crates/decibri-resampler), this
crate's only dependency, so it is already in your tree. The first example
below does both.

A caller who wants stereo keeps it, and a caller who wants the source rate
keeps it.

## Examples

Every Rust block on this page is compiled by `cargo test`, through a
`doc = include_str!` hook in `lib.rs` that turns this file into doctests. All
but the first are run as well. The first is compile-checked only, because it
needs a file that exists on your machine.

### Open a file, any supported format

The format comes from the bytes. You do not tell it, and it never looks at the
name.

```rust,no_run
use decibri_decode::decibri_resampler::{PolyphaseResampler, Resampler};
use decibri_decode::{decode, downmix_to_mono};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let decoded = decode(&std::fs::read("input.wav")?)?;

    // Interleaved samples at the file's own rate, to mono at 16 kHz.
    let mut mono = Vec::new();
    downmix_to_mono(decoded.samples(), decoded.channels(), &mut mono);
    let mut resampler = PolyphaseResampler::new(decoded.sample_rate(), 16_000)?;
    let mut samples = Vec::new();
    resampler.process(&mono, &mut samples)?;
    resampler.flush(&mut samples);

    println!("{} samples of mono f32 at 16000 Hz", samples.len());
    Ok(())
}
```

If you want to know what a file is without decoding it, `identify` answers
that on its own, and `AudioStreamDecoder` does the same job as `decode` for
bytes that arrive in pieces.

### Decode a byte stream

Bytes pushed in as they arrive, samples pulled out as they become ready. This
is the shape of a socket feed or an HTTP chunk stream. The writer here stands
in for the network.

```rust
use decibri_decode::{AudioSpec, StreamSource, WavCodec, WavStreamDecoder, WavWriter};

fn main() -> Result<(), decibri_decode::DecodeError> {
    // A small WAV standing in for bytes arriving over a socket.
    let file = WavWriter::new(AudioSpec::mono(16_000), WavCodec::PcmI16)
        .to_bytes(&[0.0, 0.25, -0.25, 0.5])?;

    let mut stream = WavStreamDecoder::new();
    let mut samples = Vec::new();

    // Push each arriving piece, pull whatever is ready. A short return from
    // push is back pressure, answered by pulling and offering the rest again.
    for piece in file.chunks(7) {
        let mut offset = 0;
        while offset < piece.len() {
            offset += stream.push(&piece[offset..])?;
            while stream.pull(&mut samples, usize::MAX)? > 0 {}
        }
    }
    stream.finish(&mut samples)?;

    assert_eq!(samples, [0.0, 0.25, -0.25, 0.5]);
    Ok(())
}
```

### Headerless G.711 at 8 kHz

Telephony sends no header at all. The companding law and the sample rate
arrive out of band, so the caller states both. Nothing here defaults to
8000 Hz, because G.711 is a sample format rather than a rate.

```rust
use decibri_decode::{AudioSpec, Decoder, G711Decoder, G711Law};

fn main() -> Result<(), decibri_decode::DecodeError> {
    let mut decoder = G711Decoder::new(G711Law::MuLaw, AudioSpec::mono(8_000));
    let mut samples = Vec::new();

    // Two mu-law codes. 0xFF is silence and 0x80 is the loudest positive code.
    decoder.feed(&[0xFF, 0x80])?;
    decoder.decode(&mut samples)?;
    decoder.flush(&mut samples)?;

    assert_eq!(samples, [0.0, 32_124.0 / 32_768.0]);
    Ok(())
}
```

### Write a WAV

```rust
use decibri_decode::{AudioSpec, WavCodec, WavWriter};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // One hundred frames of a quiet ramp, mono 16-bit at 16 kHz.
    let samples: Vec<f32> = (0..100).map(|i| i as f32 / 400.0).collect();
    let bytes = WavWriter::new(AudioSpec::mono(16_000), WavCodec::PcmI16)
        .to_bytes(&samples)?;

    let path = std::env::temp_dir().join("ramp.wav");
    std::fs::write(&path, bytes)?;
    println!("wrote {}", path.display());
    Ok(())
}
```

## What you can rely on

**The same bytes give the same samples, everywhere.** Decoding is bit-exact
and identical across platforms, toolchains and optimisation levels. This is
checked rather than asserted, by pinned hashes over the decoded output of
every format, run in CI across three operating systems, two CPU
architectures and two toolchains.

**Samples always carry their rate and channel count.** Nothing downstream has
to guess or be told separately, which is how audio ends up played back at the
wrong speed.

**You get the sample count you expect.** Decoding gives you exactly the number
of frames the file declares. A file that ends part way through a frame is an
error, not silently shortened audio.

**A broken file gives you an error, not a crash.** A size written in a file is
a claim rather than a fact, so nothing here allocates against one. Malformed
input returns a typed error with enough detail to act on. There is no panic,
no hang, and no four gigabyte allocation because a two kilobyte file said so.

FLAC is the one format with a buffer that has to follow a number from the
file, because a frame states how many samples it holds and a decoder must hold
them to reconstruct them. That buffer is capped at **4,194,240 bytes**, which
is the format's own largest frame, 65,535 samples in 8 channels at 8 bytes
each, and a file's `STREAMINFO` narrows it further when it permits less. Both
figures are measured by a test that counts what the allocator hands out.

**A FLAC file is checked against its own checksum, every time.** Every FLAC
stream carries a checksum of its own audio, so the decoder can verify what it
produced on every file it ever sees rather than only on test files. It does,
whether you read the whole file or stream it, and a mismatch is an error
instead of wrong audio. The per frame checksums are verified as well. A file
that leaves its checksum unset is allowed and is not treated as a failure.

**Your audio comes back unchanged.** Values round-trip bit-identically for
every format at or below 24 significant bits, which covers 8-bit, 16-bit,
24-bit and `f32` itself. Values in `i32` and `f64` carry more precision than
`f32` holds and land on the nearest representable value. Nothing wraps and no
sign flips.

**No hidden rate policy.** Nothing defaults a sample rate, nothing converts
one behind your back, and nothing branches on a file name.

**No unsafe code in the library.** One integration test implements
`GlobalAlloc`, a trait that is unsafe to implement by definition, to measure
what the reader allocates on malformed input. Every other file forbids it.

## Two things to know when streaming

The whole-file readers accept chunks in any order, including `data` before
`fmt ` and `SSND` before `COMM`. The streaming readers need the describing
chunk first and return an error otherwise, because decoding a payload before
reading its header would mean buffering the whole payload, and avoiding an
unbounded buffer is the point of streaming. So a file that opens fine can fail
when streamed.

Recovery from damage is a whole-buffer operation. A streaming decode that
meets a corrupted frame stops with an error, and getting the audio after the
damage means collecting the bytes and running recovery over them. Finding a
frame boundary safely needs the bytes after the candidate as well as the
candidate itself, so there is no honest way to do it as the stream arrives.

## License

Apache-2.0 &copy; 2026 [Decibri](https://github.com/decibri).
