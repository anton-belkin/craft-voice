//! Push-to-talk questions for the ArtCraft apps (GridCraft, WordCraft, DeckCraft).
//!
//! Hold a key, speak, point at what you mean while speaking; release, and the question goes on in
//! the background:
//!
//! - [`Recorder`] records the default microphone into memory (16 kHz mono [`Audio`]); its
//!   [`position`](Recorder::position) is the time base for pointing events.
//! - [`Transcriber`] runs Whisper ([`Model::Small`] or [`Model::Turbo`]) with candle on a worker
//!   thread (Metal on macOS, the CPU elsewhere), downloading the model on first use, and sends
//!   back a [`Transcript`] with timed [`Segment`]s.
//! - [`splice`] puts a marker `⟦n⟧` into the text for every [`Pointer`], where the user was in
//!   their speech when they pointed. The anchor type is the app's.
//!
//! Audio never leaves the machine and is never written anywhere: it lives in memory until the
//! transcription is done. Only the text goes on.
//!
//! [`splice`] and [`Audio`] build everywhere (wasm too); recording and transcription are native.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]
#![forbid(unsafe_code)]

mod audio;
mod error;
mod splice;

#[cfg(not(target_arch = "wasm32"))]
mod model;
#[cfg(not(target_arch = "wasm32"))]
mod transcriber;
#[cfg(not(target_arch = "wasm32"))]
mod whisper;

pub use audio::{Audio, MAX_SECONDS, SAMPLE_RATE};
pub use error::{Error, Result};
pub use splice::{Pointer, Segment, Spliced, Transcript, marker, markers_in, splice};

#[cfg(not(target_arch = "wasm32"))]
pub use audio::Recorder;
#[cfg(not(target_arch = "wasm32"))]
pub use model::{Model, SOURCE_REPO, SOURCE_REVISION, default_dir, ensure};
#[cfg(not(target_arch = "wasm32"))]
pub use transcriber::{Event, Options, Transcriber};
#[cfg(not(target_arch = "wasm32"))]
pub use whisper::LANGUAGES;
