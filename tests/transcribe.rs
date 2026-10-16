//! Runs a real model on synthetic speech. Skipped unless the small model is already in the models
//! folder (`craft_voice::default_dir()`, or `CRAFT_VOICE_MODELS`) and macOS `say` can make the
//! speech. Nothing is downloaded by the tests.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::PathBuf;
use std::time::Duration;

use craft_voice::{Audio, Event, Model, Options, Pointer, Transcriber, markers_in, splice};

fn speech(voice: &str, text: &str) -> Option<Audio> {
    let dir = std::env::temp_dir().join(format!("craft-voice-say-{}", std::process::id()));
    std::fs::create_dir_all(&dir).ok()?;
    let wav: PathBuf = dir.join(format!("{voice}.wav"));
    let ok = std::process::Command::new("say").args(["-v", voice, "-o"]).arg(&wav).args(["--data-format=LEI16@16000", text]).status().ok()?.success();
    let audio = ok.then(|| std::fs::read(&wav).ok()).flatten().and_then(|b| Audio::from_wav(&b).ok());
    let _ = std::fs::remove_dir_all(&dir);
    audio
}

fn ready_model() -> Option<(Model, PathBuf)> {
    let dir = craft_voice::default_dir()?;
    Model::Small.is_ready(&dir).then_some((Model::Small, dir))
}

#[test]
fn speech_becomes_text_with_markers() {
    let Some((model, dir)) = ready_model() else {
        eprintln!("skipped: no small model in the models folder");
        return;
    };
    let Some(en) = speech("Samantha", "Why is the revenue in this year so much higher than in that one?") else {
        eprintln!("skipped: no `say`");
        return;
    };
    let ru = speech("Milena", "Почему выручка здесь выше, чем вот тут?");
    let t = Transcriber::start(model, dir, Options { languages: vec!["en".into(), "ru".into()] }, || {});
    t.transcribe(1, en.clone());
    if let Some(ru) = ru.clone() {
        t.transcribe(2, ru);
    }
    t.transcribe(3, Audio { samples: vec![0.0; 16_000] });
    let mut got = std::collections::HashMap::new();
    while got.len() < 2 + usize::from(ru.is_some()) {
        match t.wait(Duration::from_secs(300)) {
            Some(Event::Text { id, transcript, .. }) => {
                got.insert(id, transcript);
            }
            Some(Event::Failed { error, .. }) => panic!("{error}"),
            Some(_) => {}
            None => panic!("timed out"),
        }
    }
    let en_t = &got[&1];
    assert_eq!(en_t.language, "en");
    assert!(en_t.text.to_lowercase().contains("revenue"), "{}", en_t.text);
    assert!(!en_t.segments.is_empty());
    let last = en_t.segments.last().unwrap();
    assert!(last.end > 1.0 && last.end <= en.seconds() + 0.5, "{last:?}");
    // Pointing at two cells, at the start and near the end of the phrase.
    let s = splice(en_t, &[Pointer { at: 0.0, anchor: "B5" }, Pointer { at: en.seconds() * 0.85, anchor: "C5" }]);
    assert_eq!(markers_in(&s.text), vec![1, 2], "{}", s.text);
    assert!(s.text.starts_with("⟦1⟧"), "{}", s.text);
    if ru.is_some() {
        let ru_t = &got[&2];
        assert_eq!(ru_t.language, "ru");
        assert!(ru_t.text.to_lowercase().contains("выручк"), "{}", ru_t.text);
    }
    // Silence is nothing, not a hallucinated "Thank you".
    assert!(got[&3].is_empty(), "{:?}", got[&3]);
}
