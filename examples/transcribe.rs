//! Transcribes WAV files: `cargo run --release --example transcribe -- small|turbo a.wav [b.wav…]`.
//! Models go to `craft_voice::default_dir()` (downloaded on first use).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::time::Duration;

use craft_voice::{Audio, Event, Model, Options, Transcriber};

fn main() {
    let mut args = std::env::args().skip(1);
    let model = args.next().and_then(|m| Model::from_id(&m)).unwrap_or(Model::Small);
    let files: Vec<String> = args.collect();
    let dir = craft_voice::default_dir().expect("no models folder");
    let t = Transcriber::start(model, dir, Options::default(), || {});
    for (id, f) in files.iter().enumerate() {
        let audio = Audio::from_wav(&std::fs::read(f).expect("read")).expect("wav");
        println!("{f}: {:.1} s of audio", audio.seconds());
        t.transcribe(id as u64, audio);
    }
    let mut left = files.len();
    while left > 0 {
        match t.wait(Duration::from_secs(1800)) {
            Some(Event::Downloading { done, total, .. }) => eprint!("\rdownloading {:.0}%", done as f64 * 100.0 / total.max(1) as f64),
            Some(Event::Text { id, transcript, secs }) => {
                left -= 1;
                println!("[{}] {secs:.2} s, {}: {}", files[id as usize], transcript.language, transcript.text);
                for s in &transcript.segments {
                    println!("    {:5.2}–{:5.2}  {}", s.start, s.end, s.text);
                }
            }
            Some(Event::Failed { id: Some(_), error }) => {
                left -= 1;
                println!("failed: {error}");
            }
            Some(e) => println!("{e:?}"),
            None => break,
        }
    }
}
