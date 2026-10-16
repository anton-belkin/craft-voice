//! Transcription off the UI thread: one worker per model that downloads it if needed, loads it,
//! and transcribes jobs in order. Results come back as [`Event`]s.

use std::panic::AssertUnwindSafe;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{Duration, Instant};

use crate::audio::Audio;
use crate::model::{self, Model};
use crate::splice::Transcript;
use crate::whisper::{self, Engine};

/// How the worker transcribes.
#[derive(Clone, Debug, Default)]
pub struct Options {
    /// Languages to choose from when detecting the spoken one (`["ru", "en"]`); empty: any of
    /// Whisper's languages.
    pub languages: Vec<String>,
}

/// What the worker reports.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// Downloading the model: bytes so far and in all.
    Downloading { model: Model, done: u64, total: u64 },
    /// Downloaded; converting and loading.
    Loading { model: Model },
    /// Ready to transcribe; how long loading took (seconds, download excluded).
    Ready { model: Model, load_secs: f32 },
    /// A job is done; how long it took (seconds).
    Text { id: u64, transcript: Transcript, secs: f32 },
    /// A job failed (or, with no id, the model couldn't be made ready: every job will fail).
    Failed { id: Option<u64>, error: String },
}

enum Job {
    Transcribe(u64, Audio),
}

/// A model loaded on a worker thread. Dropping it stops the worker once its current job is done.
pub struct Transcriber {
    model: Model,
    jobs: Option<Sender<Job>>,
    events: Receiver<Event>,
}

impl Transcriber {
    /// Starts a worker for `model`, keeping models in `dir`. `wake` is called after each event
    /// (to repaint a UI).
    pub fn start(model: Model, dir: PathBuf, options: Options, wake: impl Fn() + Send + 'static) -> Transcriber {
        let (job_tx, job_rx) = mpsc::channel::<Job>();
        let (ev_tx, ev_rx) = mpsc::channel::<Event>();
        let spawned = std::thread::Builder::new().name(format!("craft-voice-{}", model.id())).spawn(move || {
            let send = |e: Event| {
                let _ = ev_tx.send(e);
                wake();
            };
            worker(model, dir, options, job_rx, &send);
        });
        // Without a worker thread `transcribe` returns false.
        let jobs = spawned.ok().map(|_| job_tx);
        Transcriber { model, jobs, events: ev_rx }
    }

    pub fn model(&self) -> Model {
        self.model
    }

    /// Queues speech for transcription; the result comes back as [`Event::Text`] or
    /// [`Event::Failed`] with this id. Returns false when the worker is gone.
    pub fn transcribe(&self, id: u64, audio: Audio) -> bool {
        self.jobs.as_ref().is_some_and(|j| j.send(Job::Transcribe(id, audio)).is_ok())
    }

    /// Events since the last call.
    pub fn poll(&self) -> Vec<Event> {
        self.events.try_iter().collect()
    }

    /// Waits up to `timeout` for the next event (for tools and tests).
    pub fn wait(&self, timeout: Duration) -> Option<Event> {
        self.events.recv_timeout(timeout).ok()
    }
}

fn worker(model: Model, dir: PathBuf, options: Options, jobs: Receiver<Job>, send: &dyn Fn(Event)) {
    let mut last = Instant::now();
    let load = |last: &mut Instant| -> Option<Engine> {
        let r = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let path = model::ensure(model, &dir, &mut |done, total| {
                // A few updates a second are enough for a progress bar.
                if done == 0 || done >= total || last.elapsed() >= Duration::from_millis(200) {
                    *last = Instant::now();
                    send(Event::Downloading { model, done, total });
                }
            })?;
            send(Event::Loading { model });
            let t0 = Instant::now();
            let engine = Engine::load(&path, whisper::device(), options.languages.clone())?;
            Ok::<_, crate::Error>((engine, t0.elapsed().as_secs_f32()))
        }));
        match r {
            Ok(Ok((engine, secs))) => {
                send(Event::Ready { model, load_secs: secs });
                Some(engine)
            }
            Ok(Err(e)) => {
                send(Event::Failed { id: None, error: e.to_string() });
                None
            }
            Err(_) => {
                send(Event::Failed { id: None, error: "the speech model crashed while loading".into() });
                None
            }
        }
    };
    let mut engine = load(&mut last);
    while let Ok(job) = jobs.recv() {
        match job {
            Job::Transcribe(id, audio) => {
                // A failed load is tried again with the next recording (the network may be back).
                if engine.is_none() {
                    engine = load(&mut last);
                }
                let Some(e) = engine.as_mut() else {
                    send(Event::Failed { id: Some(id), error: "the speech model isn't available".into() });
                    continue;
                };
                let t0 = Instant::now();
                let r = std::panic::catch_unwind(AssertUnwindSafe(|| e.transcribe(&audio)));
                // The audio is dropped here: nothing is kept.
                drop(audio);
                match r {
                    Ok(Ok(transcript)) => send(Event::Text { id, transcript, secs: t0.elapsed().as_secs_f32() }),
                    Ok(Err(err)) => send(Event::Failed { id: Some(id), error: err.to_string() }),
                    Err(_) => {
                        send(Event::Failed { id: Some(id), error: "the speech model crashed on this recording".into() });
                        // The model may be in a bad state after a panic: load it again next time.
                        engine = None;
                    }
                }
            }
        }
    }
}
