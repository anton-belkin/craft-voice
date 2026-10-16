//! Audio in memory: 16 kHz mono samples, what Whisper takes. Nothing here writes audio anywhere.

use crate::error::{Error, Result};

/// The sample rate of [`Audio`].
pub const SAMPLE_RATE: u32 = 16_000;
/// Longest recording kept (longer speech is cut at this point).
pub const MAX_SECONDS: u32 = 300;

/// Speech at 16 kHz, mono, in memory.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Audio {
    pub samples: Vec<f32>,
}

impl Audio {
    /// Interleaved samples at any rate and channel count, mixed down and resampled to 16 kHz.
    pub fn from_pcm(interleaved: &[f32], rate: u32, channels: u16) -> Audio {
        let ch = usize::from(channels.max(1));
        let mono: Vec<f32> =
            if ch == 1 { interleaved.to_vec() } else { interleaved.chunks(ch).map(|f| f.iter().sum::<f32>() / f.len().max(1) as f32).collect() };
        Audio { samples: resample(&mono, rate, SAMPLE_RATE) }
    }

    /// Mono samples at `rate`, resampled to 16 kHz.
    pub fn from_mono(mono: &[f32], rate: u32) -> Audio {
        Audio { samples: resample(mono, rate, SAMPLE_RATE) }
    }

    /// A WAV file (PCM 8/16/24/32-bit or 32-bit float), mixed down and resampled.
    pub fn from_wav(bytes: &[u8]) -> Result<Audio> {
        let bad = |m: &str| Error::Transcribe(format!("not a WAV file I can read: {m}"));
        if bytes.get(0..4) != Some(b"RIFF") || bytes.get(8..12) != Some(b"WAVE") {
            return Err(bad("no RIFF/WAVE header"));
        }
        let (mut fmt, mut data): (Option<&[u8]>, Option<&[u8]>) = (None, None);
        let mut i = 12usize;
        while let Some(head) = bytes.get(i..i.saturating_add(8)) {
            let id = head.get(0..4).unwrap_or_default();
            let len = head.get(4..8).and_then(|b| b.try_into().ok()).map(u32::from_le_bytes).unwrap_or(0) as usize;
            let start = i.saturating_add(8);
            let body = bytes.get(start..start.saturating_add(len).min(bytes.len())).unwrap_or_default();
            match id {
                b"fmt " => fmt = Some(body),
                b"data" => data = Some(body),
                _ => {}
            }
            i = start.saturating_add(len).saturating_add(len & 1);
        }
        let fmt = fmt.ok_or_else(|| bad("no fmt chunk"))?;
        let data = data.ok_or_else(|| bad("no data chunk"))?;
        let u16_at = |o: usize| fmt.get(o..o + 2).and_then(|b| b.try_into().ok()).map(u16::from_le_bytes).unwrap_or(0);
        let u32_at = |o: usize| fmt.get(o..o + 4).and_then(|b| b.try_into().ok()).map(u32::from_le_bytes).unwrap_or(0);
        let mut format = u16_at(0);
        let channels = u16_at(2);
        let rate = u32_at(4);
        let bits = u16_at(14);
        if format == 0xFFFE {
            // WAVE_FORMAT_EXTENSIBLE: the real format is the first two bytes of the sub-format GUID.
            format = u16_at(24);
        }
        if channels == 0 || rate == 0 || rate > 384_000 {
            return Err(bad("bad channel count or rate"));
        }
        let samples: Vec<f32> = match (format, bits) {
            (1, 8) => data.iter().map(|&b| (f32::from(b) - 128.0) / 128.0).collect(),
            (1, 16) => data.as_chunks::<2>().0.iter().map(|b| f32::from(i16::from_le_bytes(*b)) / 32768.0).collect(),
            (1, 24) => data.as_chunks::<3>().0.iter().map(|&[a, b, c]| (i32::from_le_bytes([0, a, b, c]) >> 8) as f32 / 8_388_608.0).collect(),
            (1, 32) => data.as_chunks::<4>().0.iter().map(|b| i32::from_le_bytes(*b) as f32 / 2_147_483_648.0).collect(),
            (3, 32) => data.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect(),
            _ => return Err(bad(&format!("format {format}, {bits} bits"))),
        };
        Ok(Audio::from_pcm(&samples, rate, channels))
    }

    pub fn seconds(&self) -> f32 {
        self.samples.len() as f32 / SAMPLE_RATE as f32
    }

    /// Root mean square level (0 = silence).
    pub fn rms(&self) -> f32 {
        rms(&self.samples)
    }
}

pub(crate) fn rms(s: &[f32]) -> f32 {
    if s.is_empty() {
        return 0.0;
    }
    (s.iter().map(|x| x * x).sum::<f32>() / s.len() as f32).sqrt()
}

/// Windowed-sinc resampling (Hann window, 16 zero crossings), low-passed below the lower
/// Nyquist frequency.
pub(crate) fn resample(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || from == 0 || to == 0 || x.is_empty() {
        return x.to_vec();
    }
    let ratio = f64::from(from) / f64::from(to);
    let out_len = ((x.len() as f64) / ratio).floor() as usize;
    // Cut-off as a fraction of the input rate's Nyquist.
    let cutoff = (1.0 / ratio).min(1.0) * 0.95;
    const ZEROS: f64 = 16.0;
    let half = (ZEROS / cutoff).ceil() as isize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let center = i as f64 * ratio;
        let c = center.floor() as isize;
        let mut acc = 0.0f64;
        let mut wsum = 0.0f64;
        for k in (c - half)..=(c + half) {
            let Some(&v) = usize::try_from(k).ok().and_then(|k| x.get(k)) else { continue };
            let t = (k as f64 - center) * cutoff;
            let sinc = if t.abs() < 1e-9 { 1.0 } else { (std::f64::consts::PI * t).sin() / (std::f64::consts::PI * t) };
            let w = 0.5 + 0.5 * (std::f64::consts::PI * (k as f64 - center) / (half as f64 + 1.0)).cos();
            let h = sinc * w;
            acc += f64::from(v) * h;
            wsum += h;
        }
        out.push(if wsum.abs() > 1e-9 { (acc / wsum) as f32 } else { 0.0 });
    }
    out
}

#[cfg(not(target_arch = "wasm32"))]
pub use recorder::Recorder;

#[cfg(not(target_arch = "wasm32"))]
mod recorder {
    use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::sync::{Arc, Mutex};
    use std::thread::JoinHandle;
    use std::time::Duration;

    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use cpal::{FromSample, SampleFormat, SizedSample};

    use super::{Audio, MAX_SECONDS};
    use crate::error::{Error, Result};

    #[derive(Default)]
    struct Shared {
        /// Mono samples at the device rate.
        buf: Mutex<Vec<f32>>,
        len: AtomicUsize,
        rate: AtomicU32,
        /// Recent level (f32 bits).
        level: AtomicU32,
        /// Any non-zero sample arrived (a denied microphone delivers zeros).
        heard: AtomicBool,
        failed: Mutex<Option<String>>,
    }

    /// Push-to-talk recording from the default microphone into memory. Start it when the key
    /// goes down, [`stop`](Recorder::stop) it when the key comes up. The audio stays in memory.
    pub struct Recorder {
        shared: Arc<Shared>,
        stop: Option<mpsc::Sender<()>>,
        thread: Option<JoinHandle<()>>,
    }

    impl Recorder {
        /// Opens the default input device and starts recording (returns once audio flows or the
        /// device refused).
        pub fn start() -> Result<Recorder> {
            let shared = Arc::new(Shared::default());
            let (stop_tx, stop_rx) = mpsc::channel::<()>();
            let (ready_tx, ready_rx) = mpsc::sync_channel::<Result<()>>(1);
            let sh = shared.clone();
            let thread = std::thread::Builder::new()
                .name("craft-voice-mic".into())
                .spawn(move || {
                    // The stream lives on this thread (it isn't Send everywhere).
                    let stream = match open(sh) {
                        Ok(s) => s,
                        Err(e) => {
                            let _ = ready_tx.send(Err(e));
                            return;
                        }
                    };
                    let _ = ready_tx.send(Ok(()));
                    let _ = stop_rx.recv();
                    drop(stream);
                })
                .map_err(Error::Io)?;
            match ready_rx.recv_timeout(Duration::from_secs(10)) {
                Ok(Ok(())) => Ok(Recorder { shared, stop: Some(stop_tx), thread: Some(thread) }),
                Ok(Err(e)) => Err(e),
                Err(_) => Err(Error::Microphone("the microphone didn't start".into())),
            }
        }

        /// Seconds of audio recorded so far: the time base of pointing events.
        pub fn position(&self) -> f32 {
            let rate = self.shared.rate.load(Ordering::Relaxed);
            if rate == 0 { 0.0 } else { self.shared.len.load(Ordering::Relaxed) as f32 / rate as f32 }
        }

        /// The current input level, 0..1 (for a level meter).
        pub fn level(&self) -> f32 {
            f32::from_bits(self.shared.level.load(Ordering::Relaxed))
        }

        /// Stops recording and returns the audio at 16 kHz mono.
        pub fn stop(mut self) -> Result<Audio> {
            self.finish();
            if let Some(m) = self.shared.failed.lock().ok().and_then(|g| g.clone()) {
                return Err(Error::Microphone(m));
            }
            let rate = self.shared.rate.load(Ordering::Relaxed);
            let buf = match self.shared.buf.lock() {
                Ok(mut g) => std::mem::take(&mut *g),
                Err(p) => std::mem::take(&mut *p.into_inner()),
            };
            if !buf.is_empty() && !self.shared.heard.load(Ordering::Relaxed) {
                return Err(Error::Microphone("only silence came in: is microphone access allowed for this app?".into()));
            }
            Ok(Audio::from_mono(&buf, rate))
        }

        fn finish(&mut self) {
            if let Some(tx) = self.stop.take() {
                let _ = tx.send(());
            }
            if let Some(t) = self.thread.take() {
                let _ = t.join();
            }
        }
    }

    impl Drop for Recorder {
        fn drop(&mut self) {
            self.finish();
            // Nothing is kept: the samples go with the recorder.
            if let Ok(mut g) = self.shared.buf.lock() {
                g.clear();
                g.shrink_to_fit();
            }
        }
    }

    fn open(sh: Arc<Shared>) -> Result<cpal::Stream> {
        let host = cpal::default_host();
        let device = host.default_input_device().ok_or(Error::NoMicrophone)?;
        let supported = device.default_input_config().map_err(|e| Error::Microphone(e.to_string()))?;
        let format = supported.sample_format();
        let config = supported.config();
        sh.rate.store(config.sample_rate, Ordering::Relaxed);
        let stream = match format {
            SampleFormat::F32 => build::<f32>(&device, config, sh),
            SampleFormat::I16 => build::<i16>(&device, config, sh),
            SampleFormat::U16 => build::<u16>(&device, config, sh),
            SampleFormat::I32 => build::<i32>(&device, config, sh),
            SampleFormat::I8 => build::<i8>(&device, config, sh),
            SampleFormat::U8 => build::<u8>(&device, config, sh),
            SampleFormat::F64 => build::<f64>(&device, config, sh),
            other => Err(Error::Microphone(format!("unsupported sample format {other:?}"))),
        }?;
        stream.play().map_err(|e| Error::Microphone(e.to_string()))?;
        Ok(stream)
    }

    fn build<T>(device: &cpal::Device, config: cpal::StreamConfig, sh: Arc<Shared>) -> Result<cpal::Stream>
    where
        T: SizedSample,
        f32: FromSample<T>,
    {
        let channels = usize::from(config.channels.max(1));
        let cap = (config.sample_rate as usize).saturating_mul(MAX_SECONDS as usize);
        let data_sh = sh.clone();
        let err_sh = sh;
        device
            .build_input_stream::<T, _, _>(
                config,
                move |data: &[T], _| {
                    let mut sum = 0.0f32;
                    let mut n = 0usize;
                    let mut heard = false;
                    let Ok(mut buf) = data_sh.buf.lock() else { return };
                    for frame in data.chunks(channels) {
                        let v = frame.iter().map(|s| s.to_sample::<f32>()).sum::<f32>() / frame.len().max(1) as f32;
                        heard |= v != 0.0;
                        sum += v * v;
                        n += 1;
                        if buf.len() < cap {
                            buf.push(v);
                        }
                    }
                    data_sh.len.store(buf.len(), Ordering::Relaxed);
                    drop(buf);
                    if heard {
                        data_sh.heard.store(true, Ordering::Relaxed);
                    }
                    if n > 0 {
                        let level = (sum / n as f32).sqrt().min(1.0);
                        data_sh.level.store(level.to_bits(), Ordering::Relaxed);
                    }
                },
                move |e| {
                    if let Ok(mut f) = err_sh.failed.lock() {
                        f.get_or_insert_with(|| e.to_string());
                    }
                },
                None,
            )
            .map_err(|e| Error::Microphone(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(freq: f32, rate: u32, secs: f32) -> Vec<f32> {
        (0..(rate as f32 * secs) as usize).map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / rate as f32).sin()).collect()
    }

    #[test]
    fn resampling_keeps_length_and_a_speech_band_tone() {
        let x = tone(440.0, 48_000, 1.0);
        let a = Audio::from_mono(&x, 48_000);
        assert!((a.samples.len() as i64 - 16_000).abs() <= 1);
        // A 440 Hz sine keeps its level (RMS of a unit sine ≈ 0.707).
        assert!((a.rms() - 0.707).abs() < 0.02, "rms {}", a.rms());
        // A 12 kHz tone is above 8 kHz: filtered out.
        let hi = Audio::from_mono(&tone(12_000.0, 48_000, 1.0), 48_000);
        assert!(hi.rms() < 0.05, "rms {}", hi.rms());
    }

    #[test]
    fn stereo_is_mixed_down() {
        let l = tone(300.0, 16_000, 0.5);
        let inter: Vec<f32> = l.iter().flat_map(|&v| [v, -v]).collect();
        let a = Audio::from_pcm(&inter, 16_000, 2);
        assert_eq!(a.samples.len(), l.len());
        assert!(a.rms() < 1e-6);
    }

    #[test]
    fn wav_reading() {
        let samples: Vec<i16> = tone(200.0, 16_000, 0.25).iter().map(|v| (v * 16_000.0) as i16).collect();
        let data: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let mut w = b"RIFF".to_vec();
        w.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        w.extend_from_slice(b"WAVEfmt ");
        w.extend_from_slice(&16u32.to_le_bytes());
        w.extend_from_slice(&1u16.to_le_bytes());
        w.extend_from_slice(&1u16.to_le_bytes());
        w.extend_from_slice(&16_000u32.to_le_bytes());
        w.extend_from_slice(&32_000u32.to_le_bytes());
        w.extend_from_slice(&2u16.to_le_bytes());
        w.extend_from_slice(&16u16.to_le_bytes());
        w.extend_from_slice(b"data");
        w.extend_from_slice(&(data.len() as u32).to_le_bytes());
        w.extend_from_slice(&data);
        let a = Audio::from_wav(&w).unwrap_or_default();
        assert_eq!(a.samples.len(), 4_000);
        assert!((a.seconds() - 0.25).abs() < 1e-6);
        // Garbage is an error, never a panic.
        assert!(Audio::from_wav(b"RIFF").is_err());
        assert!(Audio::from_wav(&w[..30]).is_err());
        let mut odd = w.clone();
        odd[16] = 0xFF;
        let _ = Audio::from_wav(&odd);
    }
}
