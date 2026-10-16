//! The two speech models, where they come from and where they are kept.
//!
//! Both are OpenAI's Whisper (MIT) in the quantized `ggml` files the whisper.cpp project
//! publishes on Hugging Face (MIT), pinned to a revision and checked against their SHA-256.
//! On first use the file is downloaded into the models folder, verified, and converted once into
//! a GGUF file candle loads directly (tensor names mapped, the mel filters and the vocabulary
//! kept in it); the download itself is then removed.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use candle_core::Device;
use candle_core::quantized::gguf_file::{self, Value};
use candle_core::quantized::{GgmlDType, QTensor, ggml_file};
use sha2::{Digest, Sha256};

use crate::error::{Error, Result};

/// The Hugging Face repository and revision the models come from.
pub const SOURCE_REPO: &str = "ggerganov/whisper.cpp";
pub const SOURCE_REVISION: &str = "5359861c739e955e79d9a303bcbc70fb988958b1";
/// Version of the converted file's layout (bump to convert again).
pub(crate) const FORMAT: u32 = 1;

/// A speech model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Model {
    /// Whisper small, 5-bit (q5_1), about 180 MB. Fast.
    Small,
    /// Whisper large-v3-turbo, 5-bit (q5_0), about 550 MB. Better, slower.
    Turbo,
}

impl Model {
    pub const ALL: [Model; 2] = [Model::Small, Model::Turbo];

    /// A stable id (`small`, `turbo`) for settings.
    pub fn id(self) -> &'static str {
        match self {
            Model::Small => "small",
            Model::Turbo => "turbo",
        }
    }

    pub fn from_id(id: &str) -> Option<Model> {
        Model::ALL.into_iter().find(|m| m.id() == id)
    }

    /// A name for people.
    pub fn label(self) -> &'static str {
        match self {
            Model::Small => "Whisper small",
            Model::Turbo => "Whisper large-v3-turbo",
        }
    }

    /// The file published upstream.
    pub fn source_file(self) -> &'static str {
        match self {
            Model::Small => "ggml-small-q5_1.bin",
            Model::Turbo => "ggml-large-v3-turbo-q5_0.bin",
        }
    }

    pub fn source_sha256(self) -> &'static str {
        match self {
            Model::Small => "ae85e4a935d7a567bd102fe55afc16bb595bdb618e11b2fc7591bc08120411bb",
            Model::Turbo => "394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2",
        }
    }

    /// Download size in bytes.
    pub fn download_size(self) -> u64 {
        match self {
            Model::Small => 190_085_487,
            Model::Turbo => 574_041_195,
        }
    }

    pub fn source_url(self) -> String {
        format!("https://huggingface.co/{SOURCE_REPO}/resolve/{SOURCE_REVISION}/{}", self.source_file())
    }

    /// The converted model in `dir`.
    pub fn path_in(self, dir: &Path) -> PathBuf {
        dir.join(format!("whisper-{}-v{FORMAT}.gguf", self.id()))
    }

    /// Downloaded and converted already.
    pub fn is_ready(self, dir: &Path) -> bool {
        self.path_in(dir).is_file()
    }
}

/// Where models are kept unless the app says otherwise: `$CRAFT_VOICE_MODELS`, else the
/// shared ArtCraft data folder (`~/Library/Application Support/ArtCraft/voice-models` on macOS,
/// `%APPDATA%\ArtCraft\voice-models` on Windows, `$XDG_DATA_HOME/artcraft/voice-models` or
/// `~/.local/share/artcraft/voice-models` elsewhere), so every ArtCraft app shares one download.
pub fn default_dir() -> Option<PathBuf> {
    if let Some(d) = std::env::var_os("CRAFT_VOICE_MODELS").filter(|d| !d.is_empty()) {
        return Some(PathBuf::from(d));
    }
    let home = std::env::var_os("HOME").filter(|h| !h.is_empty()).map(PathBuf::from);
    if cfg!(target_os = "macos") {
        return home.map(|h| h.join("Library/Application Support/ArtCraft/voice-models"));
    }
    if cfg!(windows) {
        return std::env::var_os("APPDATA").map(|a| PathBuf::from(a).join("ArtCraft").join("voice-models"));
    }
    std::env::var_os("XDG_DATA_HOME")
        .filter(|d| !d.is_empty())
        .map(PathBuf::from)
        .or_else(|| home.map(|h| h.join(".local/share")))
        .map(|d| d.join("artcraft").join("voice-models"))
}

/// Makes sure `model` is in `dir`, downloading and converting it if needed. `progress` gets
/// (bytes so far, total) during the download.
pub fn ensure(model: Model, dir: &Path, progress: &mut dyn FnMut(u64, u64)) -> Result<PathBuf> {
    let out = model.path_in(dir);
    if out.is_file() {
        return Ok(out);
    }
    fs::create_dir_all(dir)?;
    let part = dir.join(format!("{}.part", model.source_file()));
    let local = std::env::var_os("CRAFT_VOICE_SOURCE_DIR").map(|d| PathBuf::from(d).join(model.source_file())).filter(|p| p.is_file());
    let src = match local {
        // A copy of the upstream file already on disk (offline installs, tests).
        Some(p) => {
            verify(&p, model.source_sha256())?;
            p
        }
        None => {
            download(&model.source_url(), &part, model.source_sha256(), model.download_size(), progress)?;
            part.clone()
        }
    };
    let tmp = dir.join(format!("whisper-{}.gguf.tmp", model.id()));
    let r = convert(&src, &tmp).and_then(|()| fs::rename(&tmp, &out).map_err(Error::Io));
    let _ = fs::remove_file(&part);
    if r.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    r.map(|()| out)
}

fn verify(path: &Path, sha256: &str) -> Result<()> {
    let mut f = fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(buf.get(..n).unwrap_or_default());
    }
    let got = hex(&h.finalize());
    if got != sha256 {
        return Err(Error::Download(format!("{} doesn't match its checksum", path.display())));
    }
    Ok(())
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn download(url: &str, to: &Path, sha256: &str, size: u64, progress: &mut dyn FnMut(u64, u64)) -> Result<()> {
    let resp = ureq::get(url).call().map_err(|e| Error::Download(e.to_string()))?;
    let total = resp.body().content_length().unwrap_or(size);
    // Bounded: a little over the size we expect.
    let mut reader = resp.into_body().into_with_config().limit(size.saturating_add(size / 10).max(1 << 20)).reader();
    let mut file = fs::File::create(to)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut done = 0u64;
    progress(0, total);
    loop {
        let n = reader.read(&mut buf).map_err(|e| Error::Download(e.to_string()))?;
        if n == 0 {
            break;
        }
        let chunk = buf.get(..n).unwrap_or_default();
        file.write_all(chunk)?;
        h.update(chunk);
        done = done.saturating_add(n as u64);
        progress(done, total);
    }
    file.flush()?;
    drop(file);
    if hex(&h.finalize()) != sha256 {
        let _ = fs::remove_file(to);
        return Err(Error::Download("the downloaded file doesn't match its checksum".into()));
    }
    Ok(())
}

// ---------------------------------------------------------------- ggml (whisper.cpp) → GGUF

/// Hyper-parameters of a whisper.cpp model file.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Hparams {
    pub n_vocab: u32,
    pub n_audio_ctx: u32,
    pub n_audio_state: u32,
    pub n_audio_head: u32,
    pub n_audio_layer: u32,
    pub n_text_ctx: u32,
    pub n_text_state: u32,
    pub n_text_head: u32,
    pub n_text_layer: u32,
    pub n_mels: u32,
}

impl Hparams {
    pub const KEYS: [&'static str; 10] = [
        "n_vocab",
        "n_audio_ctx",
        "n_audio_state",
        "n_audio_head",
        "n_audio_layer",
        "n_text_ctx",
        "n_text_state",
        "n_text_head",
        "n_text_layer",
        "n_mels",
    ];

    pub fn values(&self) -> [u32; 10] {
        [
            self.n_vocab,
            self.n_audio_ctx,
            self.n_audio_state,
            self.n_audio_head,
            self.n_audio_layer,
            self.n_text_ctx,
            self.n_text_state,
            self.n_text_head,
            self.n_text_layer,
            self.n_mels,
        ]
    }

    pub fn from_values(v: [u32; 10]) -> Hparams {
        let [n_vocab, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer, n_text_ctx, n_text_state, n_text_head, n_text_layer, n_mels] = v;
        Hparams { n_vocab, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer, n_text_ctx, n_text_state, n_text_head, n_text_layer, n_mels }
    }

    fn sane(&self) -> bool {
        let v = self.values();
        v.iter().all(|&x| x > 0 && x <= 1 << 20) && self.n_mels <= 512 && self.n_audio_layer <= 64 && self.n_text_layer <= 64
    }
}

struct Cursor<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.i.checked_add(n).ok_or_else(|| Error::Model("truncated model file".into()))?;
        let s = self.b.get(self.i..end).ok_or_else(|| Error::Model("truncated model file".into()))?;
        self.i = end;
        Ok(s)
    }

    fn u32(&mut self) -> Result<u32> {
        let s = self.take(4)?;
        Ok(u32::from_le_bytes([
            s.first().copied().unwrap_or(0),
            s.get(1).copied().unwrap_or(0),
            s.get(2).copied().unwrap_or(0),
            s.get(3).copied().unwrap_or(0),
        ]))
    }

    fn done(&self) -> bool {
        self.i >= self.b.len()
    }
}

/// whisper.cpp tensor names → the names candle's Whisper expects.
pub(crate) fn candle_name(ggml: &str) -> Option<String> {
    let block_part = |rest: &str| -> Option<String> {
        let (layer, name) = rest.split_once('.')?;
        let _: u32 = layer.parse().ok()?;
        let mapped = [
            ("attn_ln.", "self_attn_layer_norm."),
            ("attn.query.", "self_attn.q_proj."),
            ("attn.key.", "self_attn.k_proj."),
            ("attn.value.", "self_attn.v_proj."),
            ("attn.out.", "self_attn.out_proj."),
            ("cross_attn_ln.", "encoder_attn_layer_norm."),
            ("cross_attn.query.", "encoder_attn.q_proj."),
            ("cross_attn.key.", "encoder_attn.k_proj."),
            ("cross_attn.value.", "encoder_attn.v_proj."),
            ("cross_attn.out.", "encoder_attn.out_proj."),
            ("mlp_ln.", "final_layer_norm."),
            ("mlp.0.", "fc1."),
            ("mlp.2.", "fc2."),
        ]
        .iter()
        .find_map(|(from, to)| name.strip_prefix(from).map(|tail| format!("{to}{tail}")))?;
        Some(format!("layers.{layer}.{mapped}"))
    };
    if let Some(rest) = ggml.strip_prefix("encoder.") {
        let tail = match rest {
            "positional_embedding" => return None,
            r if r.starts_with("conv1.") || r.starts_with("conv2.") => r.to_string(),
            r if r.starts_with("ln_post.") => r.replacen("ln_post.", "layer_norm.", 1),
            r => block_part(r.strip_prefix("blocks.")?)?,
        };
        return Some(format!("model.encoder.{tail}"));
    }
    if let Some(rest) = ggml.strip_prefix("decoder.") {
        let tail = match rest {
            "positional_embedding" => "embed_positions.weight".to_string(),
            "token_embedding.weight" => "embed_tokens.weight".to_string(),
            r if r.starts_with("ln.") => r.replacen("ln.", "layer_norm.", 1),
            r => block_part(r.strip_prefix("blocks.")?)?,
        };
        return Some(format!("model.decoder.{tail}"));
    }
    None
}

/// The ggml type ids whisper.cpp's files use.
fn ggml_type(t: u32) -> Option<GgmlDType> {
    Some(match t {
        0 => GgmlDType::F32,
        1 => GgmlDType::F16,
        2 => GgmlDType::Q4_0,
        3 => GgmlDType::Q4_1,
        6 => GgmlDType::Q5_0,
        7 => GgmlDType::Q5_1,
        8 => GgmlDType::Q8_0,
        _ => return None,
    })
}

/// Converts a whisper.cpp `ggml` model into the GGUF file this crate loads.
pub(crate) fn convert(src: &Path, dst: &Path) -> Result<()> {
    let bytes = fs::read(src)?;
    let mut c = Cursor { b: &bytes, i: 0 };
    if c.u32()? != 0x6767_6d6c {
        return Err(Error::Model("not a whisper.cpp model file".into()));
    }
    let mut hv = [0u32; 10];
    for v in hv.iter_mut() {
        *v = c.u32()?;
    }
    let hp = Hparams::from_values(hv);
    let _ftype = c.u32()?;
    if !hp.sane() {
        return Err(Error::Model("unexpected model dimensions".into()));
    }
    // Mel filters.
    let n_mel = c.u32()? as usize;
    let n_fft = c.u32()? as usize;
    if n_mel != hp.n_mels as usize || n_fft == 0 || n_fft > 4096 {
        return Err(Error::Model("unexpected mel filter bank".into()));
    }
    let filters = c.take(n_mel * n_fft * 4)?.to_vec();
    // Vocabulary: the byte string of each text token.
    let n_tokens = c.u32()? as usize;
    if n_tokens > hp.n_vocab as usize {
        return Err(Error::Model("unexpected vocabulary".into()));
    }
    let mut vocab_bytes: Vec<Value> = vec![];
    let mut vocab_ends: Vec<Value> = Vec::with_capacity(n_tokens);
    let mut total = 0u32;
    for _ in 0..n_tokens {
        let len = c.u32()? as usize;
        if len > 1024 {
            return Err(Error::Model("unexpected vocabulary entry".into()));
        }
        let tok = c.take(len)?;
        vocab_bytes.extend(tok.iter().map(|&b| Value::U8(b)));
        total = total.saturating_add(len as u32);
        vocab_ends.push(Value::U32(total));
    }
    // Tensors.
    let cpu = Device::Cpu;
    let mut tensors: Vec<(String, QTensor)> = vec![];
    tensors.push(("mel_filters".into(), ggml_file::qtensor_from_ggml(GgmlDType::F32, &filters, vec![n_mel, n_fft], &cpu)?));
    while !c.done() {
        let n_dims = c.u32()? as usize;
        let name_len = c.u32()? as usize;
        let ttype = c.u32()?;
        if n_dims == 0 || n_dims > 4 || name_len == 0 || name_len > 256 {
            return Err(Error::Model("unexpected tensor header".into()));
        }
        let mut dims = vec![];
        for _ in 0..n_dims {
            dims.push(c.u32()? as usize);
        }
        let name = String::from_utf8_lossy(c.take(name_len)?).to_string();
        let dtype = ggml_type(ttype).ok_or_else(|| Error::Model(format!("unsupported tensor type {ttype} for {name}")))?;
        let elems = dims.iter().try_fold(1usize, |a, &d| a.checked_mul(d)).ok_or_else(|| Error::Model("tensor too large".into()))?;
        if dims.contains(&0) || elems % dtype.block_size() != 0 {
            return Err(Error::Model(format!("unexpected shape for {name}")));
        }
        let size = elems / dtype.block_size() * dtype.type_size();
        let data = c.take(size)?;
        // ggml lists dimensions innermost first; candle wants them outermost first.
        dims.reverse();
        // Conv biases are stored as [n, 1]; candle wants [n].
        if name.ends_with(".bias") {
            dims.retain(|&d| d != 1);
            if dims.is_empty() {
                dims.push(1);
            }
        }
        let Some(cname) = candle_name(&name) else { continue };
        tensors.push((cname, ggml_file::qtensor_from_ggml(dtype, data, dims, &cpu)?));
    }
    let needed = [
        "model.encoder.conv1.weight",
        "model.decoder.embed_tokens.weight",
        "model.decoder.embed_positions.weight",
        "model.decoder.layer_norm.weight",
    ];
    if let Some(missing) = needed.iter().find(|n| !tensors.iter().any(|(t, _)| t == *n)) {
        return Err(Error::Model(format!("the model file has no {missing}")));
    }
    drop(bytes);

    let mut meta: Vec<(String, Value)> = vec![("craftvoice.format".into(), Value::U32(FORMAT))];
    for (k, v) in Hparams::KEYS.iter().zip(hp.values()) {
        meta.push((format!("whisper.{k}"), Value::U32(v)));
    }
    meta.push(("whisper.vocab.bytes".into(), Value::Array(vocab_bytes)));
    meta.push(("whisper.vocab.ends".into(), Value::Array(vocab_ends)));
    let meta_refs: Vec<(&str, &Value)> = meta.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let tensor_refs: Vec<(&str, &QTensor)> = tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let mut out = std::io::BufWriter::new(fs::File::create(dst)?);
    gguf_file::write(&mut out, &meta_refs, &tensor_refs)?;
    out.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_names_map_onto_candle() {
        let m = |s: &str| candle_name(s).unwrap_or_default();
        assert_eq!(m("encoder.conv1.weight"), "model.encoder.conv1.weight");
        assert_eq!(m("encoder.blocks.3.attn.query.bias"), "model.encoder.layers.3.self_attn.q_proj.bias");
        assert_eq!(m("encoder.blocks.0.mlp.2.weight"), "model.encoder.layers.0.fc2.weight");
        assert_eq!(m("encoder.ln_post.bias"), "model.encoder.layer_norm.bias");
        assert_eq!(m("decoder.blocks.11.cross_attn.key.weight"), "model.decoder.layers.11.encoder_attn.k_proj.weight");
        assert_eq!(m("decoder.blocks.1.cross_attn_ln.weight"), "model.decoder.layers.1.encoder_attn_layer_norm.weight");
        assert_eq!(m("decoder.positional_embedding"), "model.decoder.embed_positions.weight");
        assert_eq!(m("decoder.token_embedding.weight"), "model.decoder.embed_tokens.weight");
        assert_eq!(m("decoder.ln.weight"), "model.decoder.layer_norm.weight");
        assert_eq!(candle_name("encoder.positional_embedding"), None);
        assert_eq!(candle_name("decoder.blocks.x.attn.query.weight"), None);
        assert_eq!(candle_name("something"), None);
    }

    #[test]
    fn garbage_is_an_error() {
        let dir = std::env::temp_dir().join(format!("craft-voice-test-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let src = dir.join("bad.bin");
        let _ = fs::write(&src, b"lmggxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx");
        assert!(convert(&src, &dir.join("out.gguf")).is_err());
        let mut head = 0x6767_6d6cu32.to_le_bytes().to_vec();
        head.extend(std::iter::repeat_n(0xFFu8, 60));
        let _ = fs::write(&src, &head);
        assert!(convert(&src, &dir.join("out.gguf")).is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn ids_round_trip() {
        for m in Model::ALL {
            assert_eq!(Model::from_id(m.id()), Some(m));
            assert!(m.source_url().ends_with(m.source_file()));
        }
    }
}
