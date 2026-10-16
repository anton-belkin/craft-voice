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
            let (segs, last_ts) = self.decode(&features, lang_token, offset, window_secs)?;
            segments.extend(segs);
            // Next window: right after the last complete segment, or the whole window.
            let advance = match last_ts {
                Some(t) if content_frames - seek > m::N_FRAMES && t > 1.0 => (t * m::SAMPLE_RATE as f32 / m::HOP_LENGTH as f32) as usize,
                _ => in_window,
            };
            seek += advance.max(1);
        }
        Ok(Transcript::from_segments(segments, language))
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

    /// Greedy decoding of one window with timestamps. Returns its segments (times include
    /// `offset`) and the last timestamp that closed a segment (seconds into the window).
    fn decode(&mut self, features: &Tensor, lang: Option<u32>, offset: f32, window_secs: f32) -> Result<(Vec<Segment>, Option<f32>)> {
        let s = self.special;
        let mut tokens: Vec<u32> = vec![s.sot];
        tokens.extend(lang);
        tokens.push(s.transcribe);
        let begin = tokens.len();
        let vocab = self.config.vocab_size;
        let mut sum_logprob = 0.0f64;
        let mut no_speech = 0.0f64;
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
            apply_rules(&mut logits, &tokens[begin..], &s, i == 0);
            let (best, _) = logits.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |acc, (k, &v)| if v > acc.1 { (k, v) } else { acc });
            sum_logprob += log_softmax_at(&logits, best);
            let tok = best as u32;
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
        let generated = &tokens[begin..];
        let avg_logprob = sum_logprob / (generated.len().max(1) as f64);
        if no_speech > m::NO_SPEECH_THRESHOLD && avg_logprob < m::LOGPROB_THRESHOLD {
            return Ok((vec![], None));
        }
        // Timestamp pairs around text.
        let mut segments = vec![];
        let mut start: Option<f32> = None;
        let mut text: Vec<u8> = vec![];
        let mut last_closed = None;
        for &t in generated {
            if t >= s.ts_begin {
                let at = (t - s.ts_begin) as f32 * TS_STEP;
                match start {
                    Some(st) if !text.is_empty() => {
                        segments.push(Segment {
                            start: offset + st,
                            end: offset + at.max(st),
                            text: String::from_utf8_lossy(&text).trim().to_string(),
                        });
                        text.clear();
                        start = None;
                        last_closed = Some(at);
                    }
                    _ => start = Some(at),
                }
            } else if t < s.eot
                && let Some(b) = self.vocab.get(t as usize)
            {
                text.extend_from_slice(b);
            }
        }
        if !text.is_empty() {
            let st = start.unwrap_or(0.0);
            segments.push(Segment { start: offset + st, end: offset + window_secs.max(st), text: String::from_utf8_lossy(&text).trim().to_string() });
        }
        segments.retain(|g| !g.text.is_empty());
        Ok((segments, last_closed))
    }
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
