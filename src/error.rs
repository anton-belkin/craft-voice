use std::fmt;

/// What went wrong, in words a person can act on.
#[derive(Debug)]
pub enum Error {
    /// No input device.
    NoMicrophone,
    /// The microphone couldn't be opened, or delivered nothing (often: access not allowed).
    Microphone(String),
    /// The model couldn't be downloaded or verified.
    Download(String),
    /// The model file couldn't be read or loaded.
    Model(String),
    /// Transcription failed.
    Transcribe(String),
    Io(std::io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NoMicrophone => write!(f, "no microphone found"),
            Error::Microphone(m) => write!(f, "microphone: {m}"),
            Error::Download(m) => write!(f, "model download: {m}"),
            Error::Model(m) => write!(f, "speech model: {m}"),
            Error::Transcribe(m) => write!(f, "transcription: {m}"),
            Error::Io(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl From<candle_core::Error> for Error {
    fn from(e: candle_core::Error) -> Self {
        Error::Transcribe(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;
