# craft-voice

Push-to-talk questions for the ArtCraft apps (GridCraft, WordCraft, DeckCraft): hold a key, speak,
point at what you mean while speaking, release. The question goes on in the background.

- **`Recorder`**: the default microphone into memory (cpal), 16 kHz mono. `position()` is the
  time base for pointing events.
- **`Transcriber`**: OpenAI Whisper on a worker thread with [candle](https://github.com/huggingface/candle)
  (Metal on macOS, the CPU elsewhere). Downloads the model on first use, reports progress, warms
  the GPU up, then transcribes jobs in order. Speech longer than Whisper's 30-second window is
  stitched without repeating the phrase cut at the boundary; a decoding that loops (compresses
  too well) or is unsure is redone at rising temperatures, as OpenAI's Whisper does. Results are `Event`s with a `Transcript`: text,
  language and timed segments.
- **`splice(transcript, pointers)`**: puts `⟦n⟧` into the text for every pointer, where the user
  was in their speech at that moment (interpolated over the segment's characters by time, rounded
  to a word boundary; a click in silence goes to the nearer end of the speech around it). Generic
  over the app's anchor type; the same anchor reuses its number. Pointing is never interpreted:
  the reader (an agent) resolves "this" from context.
- **`Model::{Small, Turbo}`**: Whisper small (q5_1, 190 MB download) and large-v3-turbo (q5_0,
  574 MB).

Pure Rust (no whisper.cpp, no Python). Native only for recording and transcription; `splice`,
`Audio` and the WAV reader build for wasm too.

## Privacy

Audio never leaves the machine and is never written anywhere: it stays in memory until its
transcription is done, then it is dropped. Only text goes on. The only network access is the
one-time model download.

## Models

On first use the model is downloaded from Hugging Face
([`ggerganov/whisper.cpp`](https://huggingface.co/ggerganov/whisper.cpp), pinned revision
`5359861c`), checked against its SHA-256, and converted once into a GGUF file candle loads (tensor
names mapped to candle's Whisper, mel filters and vocabulary kept inside). The download is then
removed. Models live in one folder shared by every ArtCraft app:

| | |
| --- | --- |
| macOS | `~/Library/Application Support/ArtCraft/voice-models` |
| Windows | `%APPDATA%\ArtCraft\voice-models` |
| Linux, BSD | `$XDG_DATA_HOME/artcraft/voice-models` or `~/.local/share/artcraft/voice-models` |

`CRAFT_VOICE_MODELS` moves it; `CRAFT_VOICE_SOURCE_DIR` points at a folder that already holds the
upstream `ggml-*.bin` files (offline installs). `CRAFT_VOICE_DEVICE=cpu` forces the CPU;
`CRAFT_VOICE_PROFILE=1` prints the time of each stage.

| Model | Upstream file | Size | SHA-256 | Licence |
| --- | --- | --- | --- | --- |
| Whisper small | `ggml-small-q5_1.bin` | 190,085,487 | `ae85e4a9…120411bb` | MIT (OpenAI Whisper weights; whisper.cpp conversion) |
| Whisper large-v3-turbo | `ggml-large-v3-turbo-q5_0.bin` | 574,041,195 | `39422170…5ffa7e2` | MIT (OpenAI Whisper weights; whisper.cpp conversion) |

## Speed (MacBook Pro M1 Pro, candle 0.11 on Metal)

Synthetic speech (macOS `say`), padded to 5 and 15 s, median of 3 runs, after the warm-up:

| | load + warm-up | memory (RSS) | 5 s EN | 5 s RU | 15 s EN | 15 s RU |
| --- | --- | --- | --- | --- | --- | --- |
| small q5_1 | 1.3 s | ~0.65 GB | 1.14 s | 1.47 s | 2.76 s | 3.36 s |
| large-v3-turbo q5_0 | 2.6 s | ~1.3 GB | 2.07 s | 2.20 s | 3.05 s | 3.49 s |

Where the time goes: the encoder always sees a 30-second window (small 0.3 s, turbo 1.5 s); the
decoder costs ~39 ms a token for small (12 layers) and ~24 ms for turbo (4 layers). So turbo is
slower on short phrases, about the same on long ones, and more accurate (numbers, rare words).

`cargo run --release --example bench -- small a.wav b.wav` measures it on your machine.

## Using it

```rust
let rec = craft_voice::Recorder::start()?;              // key down
let clicked_at = rec.position();                          // a click while speaking
let audio = rec.stop()?;                                  // key up
let t = craft_voice::Transcriber::start(craft_voice::Model::Small, craft_voice::default_dir().unwrap(), Default::default(), || ui.repaint());
t.transcribe(1, audio);
// later, in the UI loop:
for e in t.poll() {
    if let craft_voice::Event::Text { id, transcript, .. } = e {
        let q = craft_voice::splice(&transcript, &[craft_voice::Pointer { at: clicked_at, anchor: "Sheet1!C16" }]);
        // q.text: "why is ⟦1⟧ this so high", q.refs: [(1, "Sheet1!C16")]
    }
}
```

## Dependency note

Every published candle (0.9.2 and later) depends on `zip` 7.x/8.x, and every `zip` release before
9.0.3 has been yanked from crates.io. Cargo still uses a yanked version that a `Cargo.lock` already
names, so this repository's lock seeds `zip 8.6.0`; an app that depends on craft-voice does the
same: add the `[[package]] name = "zip" version = "8.6.0"` entry (with its checksum) to its own
`Cargo.lock` once, or run `cargo update -p zip --precise 8.6.0` when a newer zip is locked.

## Licence

MIT OR Apache-2.0. Parts of the decoding follow OpenAI's Whisper (MIT) and candle's Whisper
example (MIT OR Apache-2.0).
