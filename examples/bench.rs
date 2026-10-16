//! Load time, memory and latency of one model:
//! `cargo run --release --example bench -- small|turbo a.wav [b.wav…]` (each file padded with
//! silence to a whole number of seconds, run 3 times; the median is reported).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::time::{Duration, Instant};

use craft_voice::{Audio, Event, Model, Options, Transcriber};

fn rss_mb() -> f64 {
    let pid = std::process::id().to_string();
    let out = std::process::Command::new("ps").args(["-o", "rss=", "-p", &pid]).output().ok();
    out.and_then(|o| String::from_utf8_lossy(&o.stdout).trim().parse::<f64>().ok()).unwrap_or(0.0) / 1024.0
}

fn main() {
    let mut args = std::env::args().skip(1);
    let model = args.next().and_then(|m| Model::from_id(&m)).unwrap_or(Model::Small);
    let files: Vec<String> = args.collect();
    let dir = craft_voice::default_dir().expect("no models folder");
    let before = rss_mb();
    let t0 = Instant::now();
    let t = Transcriber::start(model, dir, Options::default(), || {});
    loop {
        match t.wait(Duration::from_secs(1800)) {
            Some(Event::Ready { load_secs, .. }) => {
                println!(
                    "{}: ready in {:.2} s (load + warm-up {load_secs:.2} s), RSS {:.0} MB → {:.0} MB",
                    model.label(),
                    t0.elapsed().as_secs_f32(),
                    before,
                    rss_mb()
                );
                break;
            }
            Some(Event::Failed { error, .. }) => panic!("{error}"),
            Some(_) => {}
            None => panic!("timeout"),
        }
    }
    let mut id = 0u64;
    for f in &files {
        let mut audio = Audio::from_wav(&std::fs::read(f).expect("read")).expect("wav");
        let whole = audio.seconds().ceil();
        audio.samples.resize((whole * craft_voice::SAMPLE_RATE as f32) as usize, 0.0);
        let mut times = vec![];
        let mut text = String::new();
        for _ in 0..3 {
            let start = Instant::now();
            t.transcribe(id, audio.clone());
            id += 1;
            match t.wait(Duration::from_secs(600)) {
                Some(Event::Text { transcript, .. }) => {
                    times.push(start.elapsed().as_secs_f64());
                    text = format!("[{}] {}", transcript.language, transcript.text);
                }
                other => panic!("{other:?}"),
            }
        }
        times.sort_by(f64::total_cmp);
        let name = std::path::Path::new(f).file_name().unwrap().to_string_lossy().to_string();
        println!("  {name} ({whole:.0} s): median {:.2} s (min {:.2}, max {:.2}), RSS {:.0} MB  {text}", times[1], times[0], times[2], rss_mb());
    }
}
