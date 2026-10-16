//! Whisper inference with candle: log-mel spectrogram, encoder, greedy decoding with timestamp
//! tokens (OpenAI's timestamp rules), segments with their time spans.

use std::path::Path;

use candle_core::quantized::gguf_file::{self, Value};
use candle_core::{Device, IndexOp, Tensor};
use candle_transformers::models::whisper::{self as m, Config, audio, quantized_model};

use crate::audio::Audio;
use crate::error::{Error, Result};
use crate::model::{FORMAT, Hparams};
use crate::splice::{Segment, Transcript};

/// Whisper's languages in token order (the first 99 for models before large-v3).
pub const LANGUAGES: [&str; 100] = [
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms",
    "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be",
    "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha",
    "ba", "jw", "su", "yue",
];

/// Most tokens decoded per 30-second window.
const MAX_TOKENS: usize = 224;
/// Seconds per timestamp token.
const TS_STEP: f32 = 0.02;
/// Quieter than this is silence (RMS).
const SILENCE_RMS: f32 = 0.002;

/// Special token ids, computed the way whisper.cpp does from the vocabulary size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Special {
    pub eot: u32,
    pub sot: u32,
    pub transcribe: u32,
    pub no_speech: u32,
    pub no_timestamps: u32,
    pub ts_begin: u32,
    pub n_languages: u32,
}

impl Special {
    pub fn for_vocab(n_vocab: u32) -> Special {
        let multilingual = n_vocab >= 51865;
        let n_languages = n_vocab.saturating_sub(51765).saturating_sub(u32::from(multilingual));
        let dt = n_languages.saturating_sub(98);
        let (eot, sot) = if multilingual { (50257, 50258) } else { (50256, 50257) };
        Special {
            eot,
            sot,
            transcribe: 50358 + dt,
            no_speech: 50361 + dt,
            no_timestamps: 50362 + dt,
            ts_begin: 50363 + dt,
            n_languages: if multilingual { n_languages.min(LANGUAGES.len() as u32) } else { 0 },
        }
    }

    pub fn language_token(&self, code: &str) -> Option<u32> {
        let i = LANGUAGES.iter().position(|l| *l == code)? as u32;
        (i < self.n_languages).then_some(self.sot + 1 + i)
    }
}

/// A loaded model, ready to transcribe.
pub(crate) struct Engine {
    model: quantized_model::Whisper,
    config: Config,
    filters: Vec<f32>,
    vocab: Vec<Vec<u8>>,
    special: Special,
    device: Device,
    /// Restrict language detection to these (empty: any).
    languages: Vec<String>,
    max_tokens: usize,
    profile: bool,
}

/// The device to run on: Metal on macOS (unless `CRAFT_VOICE_DEVICE=cpu`), else the CPU.
pub(crate) fn device() -> Device {
    let want_cpu = std::env::var("CRAFT_VOICE_DEVICE").is_ok_and(|v| v.eq_ignore_ascii_case("cpu"));
    #[cfg(target_os = "macos")]
    if !want_cpu && let Ok(d) = Device::new_metal(0) {
        return d;
    }
    let _ = want_cpu;
    Device::Cpu
}

fn meta_u32(meta: &std::collections::HashMap<String, Value>, k: &str) -> Result<u32> {
    meta.get(k).and_then(|v| v.to_u32().ok()).ok_or_else(|| Error::Model(format!("the model file has no {k}")))
}

impl Engine {
    pub fn load(path: &Path, device: Device, languages: Vec<String>) -> Result<Engine> {
        let mut f = std::fs::File::open(path)?;
        let content = gguf_file::Content::read(&mut f).map_err(|e| Error::Model(e.to_string()))?;
        if meta_u32(&content.metadata, "craftvoice.format")? != FORMAT {
            return Err(Error::Model("the model file is from another version".into()));
        }
        let mut hv = [0u32; 10];
        for (v, k) in hv.iter_mut().zip(Hparams::KEYS) {
            *v = meta_u32(&content.metadata, &format!("whisper.{k}"))?;
        }
        let hp = Hparams::from_values(hv);
        let bytes: Vec<u8> = match content.metadata.get("whisper.vocab.bytes") {
            Some(Value::Array(a)) => a.iter().map(|v| v.to_u8().unwrap_or(b'?')).collect(),
            _ => return Err(Error::Model("the model file has no vocabulary".into())),
        };
        let ends: Vec<usize> = match content.metadata.get("whisper.vocab.ends") {
            Some(Value::Array(a)) => a.iter().map(|v| v.to_u32().unwrap_or(0) as usize).collect(),
            _ => return Err(Error::Model("the model file has no vocabulary".into())),
        };
        let mut vocab = Vec::with_capacity(ends.len());
        let mut start = 0usize;
        for e in ends {
            vocab.push(bytes.get(start..e).unwrap_or_default().to_vec());
            start = e.max(start);
        }
        let filters = content
            .tensor(&mut f, "mel_filters", &Device::Cpu)
            .and_then(|t| t.dequantize(&Device::Cpu))
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| Error::Model(e.to_string()))?;
        drop(f);
        let config = Config {
            num_mel_bins: hp.n_mels as usize,
            max_source_positions: hp.n_audio_ctx as usize,
            d_model: hp.n_audio_state as usize,
            encoder_attention_heads: hp.n_audio_head as usize,
            encoder_layers: hp.n_audio_layer as usize,
            vocab_size: hp.n_vocab as usize,
            max_target_positions: hp.n_text_ctx as usize,
            decoder_attention_heads: hp.n_text_head as usize,
            decoder_layers: hp.n_text_layer as usize,
            suppress_tokens: vec![],
        };
        let vb = quantized_model::VarBuilder::from_gguf(path, &device).map_err(|e| Error::Model(e.to_string()))?;
        let model = quantized_model::Whisper::load(&vb, config.clone()).map_err(|e| Error::Model(e.to_string()))?;
        let profile = std::env::var_os("CRAFT_VOICE_PROFILE").is_some();
        let mut e =
            Engine { model, config, filters, vocab, special: Special::for_vocab(hp.n_vocab), device, languages, max_tokens: MAX_TOKENS, profile };
        e.warm_up()?;
        Ok(e)
    }

    /// Runs the model once on a second of noise, so the GPU kernels are compiled and the first
    /// real question is as fast as the next ones.
    fn warm_up(&mut self) -> Result<()> {
        let mut seed = 0x2545_f491u32;
        let noise: Vec<f32> = (0..m::SAMPLE_RATE)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                (seed as f32 / u32::MAX as f32 - 0.5) * 0.2
            })
            .collect();
        let (max, profile) = (self.max_tokens, self.profile);
        self.max_tokens = 4;
        self.profile = false;
        let r = self.transcribe(&Audio { samples: noise });
        self.max_tokens = max;
        self.profile = profile;
        r.map(|_| ())
    }

    /// Transcribes speech (any length; 30-second windows).
    pub fn transcribe(&mut self, a: &Audio) -> Result<Transcript> {
        if a.samples.is_empty() || a.rms() < SILENCE_RMS {
            return Ok(Transcript::default());
        }
        let n_mels = self.config.num_mel_bins;
        let content_frames = a.samples.len() / m::HOP_LENGTH;
        // Pad with 30 s of silence so every window is a full one, as Whisper was trained.
        let mut pcm = a.samples.clone();
        pcm.extend(std::iter::repeat_n(0.0f32, m::N_SAMPLES));
        let t_mel = std::time::Instant::now();
        let mel = audio::pcm_to_mel(&self.config, &pcm, &self.filters);
        if self.profile {
            eprintln!("craft-voice: mel {:.0} ms", t_mel.elapsed().as_secs_f64() * 1e3);
        }
        let frames = mel.len() / n_mels.max(1);
        let mel = Tensor::from_vec(mel, (1, n_mels, frames), &self.device)?;

        let mut segments: Vec<Segment> = vec![];
        let mut language = String::new();
        let mut seek = 0usize;
        while seek < content_frames {
            let window = m::N_FRAMES.min(frames.saturating_sub(seek));
            if window == 0 {
                break;
            }
            let offset = seek as f32 * m::HOP_LENGTH as f32 / m::SAMPLE_RATE as f32;
            let in_window = (content_frames - seek).min(m::N_FRAMES);
            let window_secs = in_window as f32 * m::HOP_LENGTH as f32 / m::SAMPLE_RATE as f32;
            let mel_w = mel.narrow(2, seek, window)?;
            let t_enc = std::time::Instant::now();
            let features = self.model.encoder.forward(&mel_w, true)?;
            if self.profile {
                self.device.synchronize()?;
                eprintln!("craft-voice: encoder {:.0} ms", t_enc.elapsed().as_secs_f64() * 1e3);
            }
            if language.is_empty() {
                language = self.detect_language(&features)?;
            }
            let lang_token = self.special.language_token(&language);
            // Greedy first; when the text loops (compresses too well) or the model is unsure,
            // again at rising temperatures, as OpenAI's Whisper does.
            let mut best: Option<Decoded> = None;
            for (k, &temperature) in m::TEMPERATURES.iter().enumerate() {
                let d = self.decode(&features, lang_token, temperature, (seek as u64) << 8 | k as u64)?;
                let silent = d.no_speech > m::NO_SPEECH_THRESHOLD && d.avg_logprob < m::LOGPROB_THRESHOLD;
                let looping = compression_ratio(&self.text_of(&d.tokens)) > m::COMPRESSION_RATIO_THRESHOLD;
                let unsure = d.avg_logprob < m::LOGPROB_THRESHOLD;
                let done = silent || (!looping && !unsure);
                if self.profile {
                    eprintln!("craft-voice: t={temperature} logprob {:.2} looping {looping} silent {silent}", d.avg_logprob);
                }
                best = Some(d);
                if done {
                    break;
                }
            }
            let Some(d) = best else { break };
            let parsed = if d.no_speech > m::NO_SPEECH_THRESHOLD && d.avg_logprob < m::LOGPROB_THRESHOLD {
                Parsed::default()
            } else {
                parse_segments(&d.tokens, &self.special, &self.vocab, offset, window_secs)
            };
            let (keep, advance) = stitch(parsed, content_frames - seek > m::N_FRAMES, in_window);
            segments.extend(keep);
            seek += advance.max(1);
        }
        Ok(Transcript::from_segments(dedupe(segments), language))
    }

    fn text_of(&self, tokens: &[u32]) -> String {
        let bytes: Vec<u8> =
            tokens.iter().filter(|&&t| t < self.special.eot).filter_map(|&t| self.vocab.get(t as usize)).flatten().copied().collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn logits(&mut self, tokens: &[u32], features: &Tensor, flush: bool) -> Result<Vec<f32>> {
        Ok(self.logits_at(tokens, features, flush, false)?.0)
    }

    /// Logits after the last token, and with `first` also after the first one (where Whisper
    /// says how likely it is that there is no speech at all).
    fn logits_at(&mut self, tokens: &[u32], features: &Tensor, flush: bool, first: bool) -> Result<(Vec<f32>, Vec<f32>)> {
        let t = Tensor::new(tokens, &self.device)?.unsqueeze(0)?;
        let ys = self.model.decoder.forward(&t, features, flush)?;
        let (_, seq, _) = ys.dims3()?;
        let last = self.model.decoder.final_linear(&ys.i((..1, seq - 1..))?)?.i(0)?.i(0)?.to_vec1::<f32>()?;
        let head = if first { self.model.decoder.final_linear(&ys.i((..1, 0..1))?)?.i(0)?.i(0)?.to_vec1::<f32>()? } else { vec![] };
        Ok((last, head))
    }

    fn detect_language(&mut self, features: &Tensor) -> Result<String> {
        let s = self.special;
        if s.n_languages == 0 {
            return Ok("en".into());
        }
        let logits = self.logits(&[s.sot], features, true)?;
        let allowed = |code: &str| self.languages.is_empty() || self.languages.iter().any(|l| l == code);
        let best = LANGUAGES
            .iter()
            .take(s.n_languages as usize)
            .enumerate()
            .filter(|(_, c)| allowed(c))
            .filter_map(|(i, c)| logits.get((s.sot + 1 + i as u32) as usize).map(|v| (*c, *v)))
            .max_by(|a, b| a.1.total_cmp(&b.1));
        Ok(best.map(|(c, _)| c.to_string()).unwrap_or_else(|| "en".into()))
    }

    /// Decodes one window with timestamps: greedy at temperature 0, sampled above it.
    fn decode(&mut self, features: &Tensor, lang: Option<u32>, temperature: f64, seed: u64) -> Result<Decoded> {
        let s = self.special;
        let mut tokens: Vec<u32> = vec![s.sot];
        tokens.extend(lang);
        tokens.push(s.transcribe);
        let begin = tokens.len();
        let vocab = self.config.vocab_size;
        let mut sum_logprob = 0.0f64;
        let mut no_speech = 0.0f64;
        let mut rng = Rng(seed ^ 0x9E37_79B9_7F4A_7C15);
        let t_dec = std::time::Instant::now();
        let mut steps = 0usize;
        for i in 0..self.max_tokens {
            steps += 1;
            let (mut logits, head) = self.logits_at(&tokens, features, i == 0, i == 0)?;
            if logits.len() < vocab {
                return Err(Error::Transcribe("unexpected model output".into()));
            }
            logits.truncate(vocab);
            if i == 0 {
                no_speech = softmax_at(&head, s.no_speech as usize);
            }
            apply_rules(&mut logits, tokens.get(begin..).unwrap_or_default(), &s, i == 0);
            let pick = if temperature > 0.0 { sample(&logits, temperature, &mut rng) } else { argmax(&logits) };
            sum_logprob += log_softmax_at(&logits, pick);
            let tok = pick as u32;
            if tok == s.eot {
                break;
            }
            tokens.push(tok);
            if tokens.len() >= self.config.max_target_positions.saturating_sub(1) {
                break;
            }
        }
        if self.profile {
            eprintln!("craft-voice: decoder {steps} steps, {:.0} ms", t_dec.elapsed().as_secs_f64() * 1e3);
        }
        let generated: Vec<u32> = tokens.get(begin..).unwrap_or_default().to_vec();
        let avg_logprob = sum_logprob / (generated.len().max(1) as f64);
        Ok(Decoded { tokens: generated, avg_logprob, no_speech })
    }
}

/// One window's decoding.
struct Decoded {
    tokens: Vec<u32>,
    avg_logprob: f64,
    no_speech: f64,
}

/// A window's segments: those closed by a timestamp, the unfinished one after them, and the
/// last closing timestamp (seconds into the window).
#[derive(Debug, Default, PartialEq)]
pub(crate) struct Parsed {
    pub closed: Vec<Segment>,
    pub trailing: Option<Segment>,
    pub last_closed: Option<f32>,
}

/// Timestamp tokens around text → segments (times include `offset`).
pub(crate) fn parse_segments(generated: &[u32], s: &Special, vocab: &[Vec<u8>], offset: f32, window_secs: f32) -> Parsed {
    let mut out = Parsed::default();
    let mut start: Option<f32> = None;
    let mut text: Vec<u8> = vec![];
    let seg = |st: f32, end: f32, text: &[u8]| Segment {
        start: offset + st,
        end: offset + end.max(st),
        text: String::from_utf8_lossy(text).trim().to_string(),
    };
    for &t in generated {
        if t >= s.ts_begin {
            let at = (t - s.ts_begin) as f32 * TS_STEP;
            if !text.is_empty() {
                // Text since the last timestamp: it ends here (and the next text starts here,
                // whether or not the model repeats this timestamp).
                let st = start.unwrap_or(at);
                out.closed.push(seg(st, at, &text));
                text.clear();
                out.last_closed = Some(at);
            }
            start = Some(at);
        } else if t < s.eot
            && let Some(b) = vocab.get(t as usize)
        {
            text.extend_from_slice(b);
        }
    }
    if !text.is_empty() {
        let st = start.unwrap_or(0.0);
        out.trailing = Some(seg(st, window_secs, &text));
    }
    out.closed.retain(|g| !g.text.is_empty());
    out.trailing = out.trailing.filter(|g| !g.text.is_empty());
    out
}

/// What a window adds and how far the next one starts (in mel frames). When speech goes on past
/// this window, the next one starts right after the last complete segment and the unfinished one
/// is left to it: keeping it here too would say it twice.
pub(crate) fn stitch(parsed: Parsed, more: bool, in_window: usize) -> (Vec<Segment>, usize) {
    let mut keep = parsed.closed;
    match parsed.last_closed {
        Some(t) if more && t > 1.0 => (keep, ((t * m::SAMPLE_RATE as f32 / m::HOP_LENGTH as f32) as usize).min(in_window)),
        _ => {
            keep.extend(parsed.trailing);
            (keep, in_window)
        }
    }
}

/// Drops a segment that repeats the one before it over the same stretch of audio (a window
/// boundary decoded twice); a sentence really said twice has its own time and stays.
pub(crate) fn dedupe(segments: Vec<Segment>) -> Vec<Segment> {
    let norm = |t: &str| t.chars().filter(|c| c.is_alphanumeric()).flat_map(char::to_lowercase).collect::<String>();
    let mut out: Vec<Segment> = Vec::with_capacity(segments.len());
    for g in segments {
        if let Some(prev) = out.last() {
            let overlap = (prev.end.min(g.end) - prev.start.max(g.start)).max(0.0);
            let shorter = (g.end - g.start).min(prev.end - prev.start).max(0.01);
            if overlap / shorter > 0.5 && norm(&prev.text) == norm(&g.text) {
                continue;
            }
        }
        out.push(g);
    }
    out
}

/// How well text compresses (zlib): Whisper's sign of a decoding loop above 2.4.
pub(crate) fn compression_ratio(text: &str) -> f64 {
    use std::io::Write;
    if text.is_empty() {
        return 0.0;
    }
    let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    let compressed = z.write_all(text.as_bytes()).and_then(|()| z.finish()).map(|v| v.len()).unwrap_or(text.len());
    text.len() as f64 / compressed.max(1) as f64
}

fn argmax(logits: &[f32]) -> usize {
    logits.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |acc, (k, &v)| if v > acc.1 { (k, v) } else { acc }).0
}

/// A small deterministic generator (xorshift), so a retry is reproducible.
struct Rng(u64);

impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn sample(logits: &[f32], temperature: f64, rng: &mut Rng) -> usize {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return argmax(logits);
    }
    let w: Vec<f64> = logits.iter().map(|&v| (f64::from(v - max) / temperature).exp()).collect();
    let total: f64 = w.iter().sum();
    let mut x = rng.next_f64() * total;
    for (k, p) in w.iter().enumerate() {
        x -= p;
        if x <= 0.0 && *p > 0.0 {
            return k;
        }
    }
    argmax(logits)
}

fn softmax_at(logits: &[f32], k: usize) -> f64 {
    log_softmax_at(logits, k).exp()
}

fn log_softmax_at(logits: &[f32], k: usize) -> f64 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return f64::NEG_INFINITY;
    }
    let sum: f64 = logits.iter().map(|&v| f64::from(v - max).exp()).sum();
    logits.get(k).map_or(f64::NEG_INFINITY, |&v| f64::from(v - max) - sum.ln())
}

/// OpenAI's decoding rules: no special tokens in the text, timestamps in pairs and never going
/// back, a timestamp first (at most 1 s in), and a timestamp whenever all of them together are
/// likelier than any single text token.
pub(crate) fn apply_rules(logits: &mut [f32], generated: &[u32], s: &Special, first: bool) {
    let n = logits.len();
    let ts = s.ts_begin as usize;
    let mut kill = |range: std::ops::Range<usize>| {
        for v in logits.iter_mut().take(range.end.min(n)).skip(range.start) {
            *v = f32::NEG_INFINITY;
        }
    };
    // Special tokens other than end-of-text and timestamps.
    kill(s.eot as usize + 1..ts);
    if first {
        // No blank and no end at the very start.
        kill(220..221);
        kill(s.eot as usize..s.eot as usize + 1);
    }
    let last_ts = generated.last().is_some_and(|&t| t >= s.ts_begin);
    let penult_ts = generated.len() < 2 || generated.get(generated.len() - 2).is_some_and(|&t| t >= s.ts_begin);
    if last_ts {
        if penult_ts {
            kill(ts..n);
        } else {
            kill(0..s.eot as usize);
        }
    }
    if let Some(&last) = generated.iter().rev().find(|&&t| t >= s.ts_begin) {
        let floor = if last_ts && !penult_ts { last } else { last + 1 };
        kill(ts..floor as usize);
    }
    if generated.is_empty() {
        kill(0..ts);
        // At most 1 s of silence before the first word.
        kill(ts + 51..n);
    }
    // When the timestamps together beat every text token, a timestamp it is.
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max.is_finite() {
        let ln_sum: f64 = logits.iter().map(|&v| f64::from(v - max).exp()).sum::<f64>().ln();
        let lp = |v: f32| f64::from(v - max) - ln_sum;
        let ts_lse = {
            let tail: Vec<f64> = logits.iter().skip(ts).map(|&v| lp(v)).filter(|v| v.is_finite()).collect();
            let m = tail.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if m.is_finite() { m + tail.iter().map(|v| (v - m).exp()).sum::<f64>().ln() } else { f64::NEG_INFINITY }
        };
        let text_max = logits.iter().take(ts).map(|&v| lp(v)).fold(f64::NEG_INFINITY, f64::max);
        if ts_lse > text_max {
            for v in logits.iter_mut().take(ts) {
                *v = f32::NEG_INFINITY;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn special_tokens_match_whisper_cpp() {
        // small (multilingual, 99 languages).
        let s = Special::for_vocab(51865);
        assert_eq!((s.eot, s.sot, s.transcribe, s.no_timestamps, s.ts_begin), (50257, 50258, 50359, 50363, 50364));
        assert_eq!(s.language_token("ru"), Some(50258 + 1 + 4));
        assert_eq!(s.language_token("yue"), None);
        // large-v3 / turbo (100 languages).
        let t = Special::for_vocab(51866);
        assert_eq!((t.transcribe, t.no_speech, t.no_timestamps, t.ts_begin), (50360, 50363, 50364, 50365));
        assert_eq!(t.language_token("yue"), Some(50258 + 100));
    }

    fn vocab() -> Vec<Vec<u8>> {
        // Token k is the word "w<k> " for the few ids the tests use.
        (0..200).map(|k| format!(" w{k}").into_bytes()).collect()
    }

    #[test]
    fn segments_from_timestamp_tokens() {
        let s = Special::for_vocab(51865);
        let ts = |sec: f32| s.ts_begin + (sec / TS_STEP).round() as u32;
        // <|0.00|> w1 w2 <|2.00|><|2.00|> w3 <|3.50|> w4 (unfinished: speech goes on)
        let tokens = [ts(0.0), 1, 2, ts(2.0), ts(2.0), 3, ts(3.5), 4];
        let p = parse_segments(&tokens, &s, &vocab(), 30.0, 28.0);
        assert_eq!(p.closed.len(), 2);
        assert_eq!((p.closed[0].start, p.closed[0].end, p.closed[0].text.as_str()), (30.0, 32.0, "w1 w2"));
        assert_eq!((p.closed[1].start, p.closed[1].end), (32.0, 33.5));
        assert_eq!(p.last_closed, Some(3.5));
        let t = p.trailing.unwrap();
        assert_eq!((t.start, t.end, t.text.as_str()), (33.5, 58.0, "w4"));
        // A single timestamp between two pieces of text still ends one and starts the next.
        let p = parse_segments(&[ts(0.0), 1, ts(1.0), 2, ts(2.0)], &s, &vocab(), 0.0, 5.0);
        assert_eq!(p.closed.iter().map(|g| (g.start, g.end)).collect::<Vec<_>>(), vec![(0.0, 1.0), (1.0, 2.0)]);
        assert!(p.trailing.is_none());
    }

    #[test]
    fn an_unfinished_segment_is_left_to_the_next_window() {
        let g = |a: f32, b: f32, t: &str| Segment { start: a, end: b, text: t.into() };
        let parsed = || Parsed { closed: vec![g(0.0, 26.7, "one")], trailing: Some(g(26.7, 30.0, "two, cut")), last_closed: Some(26.7) };
        // More speech after this window: "two" is decoded again by the next one, from 26.7 s.
        let (keep, advance) = stitch(parsed(), true, 3000);
        assert_eq!(keep.iter().map(|g| g.text.as_str()).collect::<Vec<_>>(), vec!["one"]);
        assert_eq!(advance, 2670);
        // The last window: everything stays.
        let (keep, advance) = stitch(parsed(), false, 1200);
        assert_eq!(keep.len(), 2);
        assert_eq!(advance, 1200);
    }

    #[test]
    fn a_window_boundary_decoded_twice_is_said_once() {
        let g = |a: f32, b: f32, t: &str| Segment { start: a, end: b, text: t.into() };
        let out = dedupe(vec![g(0.0, 3.0, "А почему здесь 900?"), g(0.2, 3.1, "а почему здесь 900"), g(3.0, 5.0, "Так.")]);
        assert_eq!(out.len(), 2);
        // Said twice for real (one after the other): both stay.
        let out = dedupe(vec![g(0.0, 3.0, "Почему здесь 900?"), g(3.2, 6.0, "Почему здесь 900?")]);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn loops_compress_well() {
        assert!(compression_ratio(&"а почему здесь 900? ".repeat(12)) > 2.4);
        assert!(compression_ratio("Почему выручка в двадцать восьмом году намного выше, чем вот эта цифра?") < 2.4);
        assert_eq!(compression_ratio(""), 0.0);
    }

    #[test]
    fn timestamps_come_in_pairs_and_never_go_back() {
        let s = Special::for_vocab(51865);
        let n = 51865usize;
        let ts = s.ts_begin as usize;
        // At the start: only timestamps up to 1 s.
        let mut l = vec![0.0f32; n];
        apply_rules(&mut l, &[], &s, true);
        assert!(l[..ts].iter().all(|v| v.is_infinite()));
        assert!(l[ts..=ts + 50].iter().all(|v| v.is_finite()));
        assert!(l[ts + 51..].iter().all(|v| v.is_infinite()));
        // After "<|0.50|> hello <|1.00|>" a timestamp must follow (or end), not text, and not
        // earlier than 1.00.
        let generated = [s.ts_begin + 25, 31373, s.ts_begin + 50];
        let mut l = vec![0.0f32; n];
        l[31373] = 5.0;
        apply_rules(&mut l, &generated, &s, false);
        assert!(l[31373].is_infinite());
        assert!(l[ts + 49].is_infinite());
        assert!(l[ts + 50].is_finite());
        // After a pair, text (not another timestamp).
        let generated = [s.ts_begin + 25, 31373, s.ts_begin + 50, s.ts_begin + 50];
        let mut l = vec![0.0f32; n];
        apply_rules(&mut l, &generated, &s, false);
        assert!(l[ts..].iter().all(|v| v.is_infinite()));
        assert!(l[100].is_finite());
    }
}
