//! Records the microphone for a few seconds and transcribes it (a live check of the whole crate):
//! `cargo run --release --example record -- [seconds] [small|turbo]`. Nothing is saved.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::time::Duration;

use craft_voice::{Event, Model, Options, Recorder, Transcriber};

fn main() {
    let mut args = std::env::args().skip(1);
    let secs: f32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(4.0);
    let model = args.next().and_then(|m| Model::from_id(&m)).unwrap_or(Model::Small);
    let t = Transcriber::start(model, craft_voice::default_dir().expect("models folder"), Options::default(), || {});
    let rec = Recorder::start().unwrap_or_else(|e| panic!("{e}"));
    eprintln!("recording {secs} s — speak now");
    while rec.position() < secs {
        std::thread::sleep(Duration::from_millis(100));
        eprint!("\r{:4.1} s  level {:.3}   ", rec.position(), rec.level());
    }
    let audio = rec.stop().unwrap_or_else(|e| panic!("\n{e}"));
    eprintln!("\n{:.1} s of audio, RMS {:.4}; transcribing…", audio.seconds(), audio.rms());
    t.transcribe(1, audio);
    loop {
        match t.wait(Duration::from_secs(600)) {
            Some(Event::Text { transcript, secs, .. }) => {
                println!("[{}] {secs:.2} s: {}", transcript.language, transcript.text);
                break;
            }
            Some(Event::Failed { error, .. }) => panic!("{error}"),
            Some(_) => {}
            None => panic!("timed out"),
        }
    }
}
