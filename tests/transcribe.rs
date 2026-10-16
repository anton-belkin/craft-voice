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

/// Phrases spoken one after another with a pause between them: the audio and when each phrase
/// starts and ends.
fn phrases(voice: &str, texts: &[&str], gap: f32) -> Option<(Audio, Vec<(f32, f32)>)> {
    let mut samples = vec![];
    let mut spans = vec![];
    for t in texts {
        let a = speech(voice, t)?;
        let start = samples.len() as f32 / 16_000.0;
        samples.extend_from_slice(&a.samples);
        spans.push((start, samples.len() as f32 / 16_000.0));
        samples.extend(std::iter::repeat_n(0.0, (gap * 16_000.0) as usize));
    }
    Some((Audio { samples }, spans))
}

fn transcribe_one(audio: Audio) -> Option<craft_voice::Transcript> {
    let (model, dir) = ready_model()?;
    let t = Transcriber::start(model, dir, Options { languages: vec!["ru".into(), "en".into()] }, || {});
    t.transcribe(1, audio);
    loop {
        match t.wait(Duration::from_secs(600)) {
            Some(Event::Text { transcript, .. }) => return Some(transcript),
            Some(Event::Failed { error, .. }) => panic!("{error}"),
            Some(_) => {}
            None => panic!("timed out"),
        }
    }
}

fn count(hay: &str, needle: &str) -> usize {
    hay.to_lowercase().matches(&needle.to_lowercase()).count()
}

/// Speech longer than Whisper's 30-second window, with a phrase across the boundary: each phrase
/// comes out once (the unfinished phrase at the end of a window was decoded twice before).
#[test]
fn long_speech_is_not_repeated_across_windows() {
    if ready_model().is_none() {
        eprintln!("skipped: no small model");
        return;
    }
    let texts = [
        "А почему здесь девятьсот тысяч?",
        "Так, а здесь два с половиной миллиона. Почему?",
        "Валовая прибыль, кажется, выглядит нормально.",
        "Посмотри на маркетинг вот в этой строке.",
        "Откуда взялся рост клиентов в марте?",
        "И почему отток такой маленький?",
        "Проверь, пожалуйста, цены за место.",
        "Сколько стоит обслуживание одного клиента и кто его посчитал?",
        "А здесь операционные расходы почти не растут.",
        "Объясни, как считается средняя база клиентов.",
    ];
    let Some((mut audio, mut spans)) = phrases("Milena", &texts, 0.6) else {
        eprintln!("skipped: no `say`");
        return;
    };
    // A longer pause before phrase 8 so that it starts at 28.5 s and runs across the 30-second
    // window boundary.
    if spans[7].0 < 28.5 {
        let pad = 28.5 - spans[7].0;
        let at = (spans[7].0 * 16_000.0) as usize;
        audio.samples.splice(at..at, std::iter::repeat_n(0.0, (pad * 16_000.0) as usize));
        for s in spans.iter_mut().skip(7) {
            *s = (s.0 + pad, s.1 + pad);
        }
    }
    let straddles = spans[7].0 < 29.0 && spans[7].1 > 30.5;
    assert!(straddles, "{spans:?}");
    eprintln!("{:.1} s, phrase across 30 s: {straddles}", audio.seconds());
    let t = transcribe_one(audio).unwrap();
    eprintln!("{}", t.text);
    for w in ["девятьсот", "900", "миллион", "маркетинг", "отток", "обслуживание", "операционные", "средняя"]
    {
        assert!(count(&t.text, w) <= 1, "`{w}` said more than once: {}", t.text);
    }
    assert!(count(&t.text, "обслуживание") == 1 && count(&t.text, "средняя") == 1, "{}", t.text);
    // Segments go forward in time.
    assert!(t.segments.windows(2).all(|w| w[1].start >= w[0].start - 0.05), "{:?}", t.segments);
}

/// Clicks at known moments land in the phrase being spoken at that moment.
#[test]
fn markers_land_in_the_phrase_being_spoken() {
    if ready_model().is_none() {
        eprintln!("skipped: no small model");
        return;
    }
    let texts = ["Почему выручка здесь такая большая?", "Расходы почти не меняются два года.", "А маржа выглядит слишком хорошо."];
    let Some((audio, spans)) = phrases("Milena", &texts, 1.0) else {
        eprintln!("skipped: no `say`");
        return;
    };
    let mid = |(a, b): (f32, f32)| a + (b - a) * 0.5;
    let pointers = [Pointer { at: mid(spans[0]), anchor: "B15" }, Pointer { at: mid(spans[2]), anchor: "B19" }];
    let t = transcribe_one(audio).unwrap();
    let s = splice(&t, &pointers);
    eprintln!("{}\n{:?}", s.text, t.segments);
    let lower = s.text.to_lowercase();
    let at = |w: &str| lower.find(w).unwrap_or_else(|| panic!("no `{w}` in {}", s.text));
    // ⟦1⟧ inside the first phrase, ⟦2⟧ inside the third.
    assert!(at("⟦1⟧") > at("почему") && at("⟦1⟧") < at("расходы"), "{}", s.text);
    assert!(at("⟦2⟧") > at("марж"), "{}", s.text);
}
