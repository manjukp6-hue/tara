//! Multimedia skills native Rust implementation.
//! Replaces:
//! - TARA/SKILLS/multimedia/imagegen/scripts/image_gen.py
//! - TARA/SKILLS/multimedia/screenshot/scripts/take_screenshot.py
//! - TARA/SKILLS/multimedia/speech/scripts/text_to_speech.py
//! - TARA/SKILLS/multimedia/transcribe/scripts/transcribe_diarize.py

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::Digest;
use std::fs;
#[cfg(target_os = "linux")]
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
#[cfg(target_os = "linux")]
use std::process::Stdio;

// ── 1. Image Generation ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenerationResult {
    pub prompt: String,
    pub width: u32,
    pub height: u32,
    pub format: String,
    pub file_path: Option<String>,
    pub svg_data: String,
}

pub fn generate_image_svg(
    prompt: &str,
    width: u32,
    height: u32,
    out_path: Option<&Path>,
) -> Result<ImageGenerationResult, String> {
    let clean_prompt = prompt.trim();
    if clean_prompt.is_empty() || width == 0 || height == 0 || width > 8192 || height > 8192 {
        return Err(
            "prompt must be non-empty and dimensions must be between 1 and 8192 pixels".to_string(),
        );
    }
    let hash_bytes = sha2::Sha256::digest(clean_prompt.as_bytes());
    let hash = hex::encode(&hash_bytes[..8]);

    // Derive deterministic procedural palette and geometry from prompt entropy
    let c1 = format!(
        "#{:02x}{:02x}{:02x}",
        (hash_bytes[0] % 60) + 15,
        (hash_bytes[1] % 60) + 20,
        (hash_bytes[2] % 80) + 40
    );
    let c2 = format!(
        "#{:02x}{:02x}{:02x}",
        (hash_bytes[3] % 80) + 30,
        (hash_bytes[4] % 100) + 80,
        (hash_bytes[5] % 120) + 100
    );
    let c_accent = format!(
        "#{:02x}{:02x}{:02x}",
        (hash_bytes[6] % 150) + 100,
        (hash_bytes[7] % 150) + 100,
        (hash_bytes[8] % 150) + 100
    );

    let cx = width / 2;
    let cy = height / 2;
    let base_r = (width.min(height) / 4) as f32;

    // Generate procedural concentric geometric arcs and polygon points
    let mut elements = Vec::new();
    let num_nodes = 6 + (hash_bytes[9] as usize % 6);
    let mut poly_points = Vec::new();
    for i in 0..num_nodes {
        let angle = (i as f32 * 2.0 * std::f32::consts::PI) / (num_nodes as f32);
        let variance = 0.8 + ((hash_bytes[(10 + i) % 32] as f32 / 255.0) * 0.4);
        let r = base_r * variance;
        let px = cx as f32 + r * angle.cos();
        let py = cy as f32 + r * angle.sin();
        poly_points.push(format!("{:.1},{:.1}", px, py));
        elements.push(format!(
            r#"  <circle cx="{:.1}" cy="{:.1}" r="{:.1}" fill="{}" opacity="0.85" />"#,
            px,
            py,
            4.0 + (variance * 3.0),
            c_accent
        ));
    }

    let escaped_prompt = clean_prompt
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;");

    let svg = format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">
  <defs>
    <linearGradient id="grad_{id}" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:{c1};stop-opacity:1" />
      <stop offset="100%" style="stop-color:{c2};stop-opacity:1" />
    </linearGradient>
  </defs>
  <rect width="100%" height="100%" fill="url(#grad_{id})" />
  <polygon points="{poly}" fill="none" stroke="{c_accent}" stroke-width="2" opacity="0.6" />
{nodes}
  <circle cx="{cx}" cy="{cy}" r="{r_inner}" fill="none" stroke="{c2}" stroke-width="1.5" stroke-dasharray="4,4" opacity="0.7" />
  <text x="50%" y="{sub_y}" text-anchor="middle" fill="#94a3b8" font-family="system-ui, sans-serif" font-size="{sub_font_size}">TARA AI Engine</text>
  <text x="50%" y="{text_y}" text-anchor="middle" fill="#f8fafc" font-family="system-ui, sans-serif" font-size="{font_size}" font-weight="600">{title}</text>
</svg>"##,
        w = width,
        h = height,
        id = hash,
        c1 = c1,
        c2 = c2,
        c_accent = c_accent,
        poly = poly_points.join(" "),
        nodes = elements.join("\n"),
        cx = cx,
        cy = cy,
        r_inner = (base_r * 0.6) as u32,
        sub_y = (height / 10).max(24),
        sub_font_size = (width.min(height) / 36).max(11),
        text_y = height.saturating_sub((height / 10).max(24)),
        font_size = (width.min(height) / 28).max(12),
        title = escaped_prompt
    );

    let saved_path = if let Some(p) = out_path {
        if let Some(parent) = p.parent().filter(|p| !p.as_os_str().is_empty()) {
            let _ = fs::create_dir_all(parent);
        }
        fs::write(p, &svg).map_err(|e| e.to_string())?;
        Some(p.to_string_lossy().to_string())
    } else {
        None
    };

    Ok(ImageGenerationResult {
        prompt: clean_prompt.to_string(),
        width,
        height,
        format: "svg".to_string(),
        file_path: saved_path,
        svg_data: svg,
    })
}

// ── 2. Cross-Platform Screenshot Helper ─────────────────────────────────────────

pub fn parse_region_rect(s: &str) -> Result<(i32, i32, u32, u32), String> {
    let parts: Vec<&str> = s.split(',').map(|p| p.trim()).collect();
    if parts.len() != 4 {
        return Err("Region must be x,y,w,h".to_string());
    }
    let x = parts[0].parse::<i32>().map_err(|e| e.to_string())?;
    let y = parts[1].parse::<i32>().map_err(|e| e.to_string())?;
    let w = parts[2].parse::<u32>().map_err(|e| e.to_string())?;
    let h = parts[3].parse::<u32>().map_err(|e| e.to_string())?;
    Ok((x, y, w, h))
}

pub fn save_screenshot_file(
    out_path: &Path,
    region: Option<(i32, i32, u32, u32)>,
) -> Result<PathBuf, String> {
    if let Some(p) = out_path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(p).map_err(|e| e.to_string())?;
    }
    #[cfg(target_os = "windows")]
    {
        let mut command = Command::new("powershell.exe");
        command.args([
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
        ]);
        let script = "$ErrorActionPreference='Stop'; Add-Type -AssemblyName System.Windows.Forms; Add-Type -AssemblyName System.Drawing; $b=[System.Windows.Forms.Screen]::PrimaryScreen.Bounds; $bmp=New-Object System.Drawing.Bitmap($b.Width,$b.Height); $g=[System.Drawing.Graphics]::FromImage($bmp); $g.CopyFromScreen($b.Location,[System.Drawing.Point]::Empty,$b.Size); if($env:TARA_SCREENSHOT_REGION -ne ''){$r=$env:TARA_SCREENSHOT_REGION.Split(',')|ForEach-Object{[int]$_}; if($r[2] -le 0 -or $r[3] -le 0){throw 'Invalid crop size'}; $crop=$bmp.Clone([System.Drawing.Rectangle]::new($r[0],$r[1],$r[2],$r[3]), $bmp.PixelFormat); $bmp.Dispose(); $bmp=$crop}; $bmp.Save($env:TARA_SCREENSHOT_OUT,[System.Drawing.Imaging.ImageFormat]::Png); $g.Dispose(); $bmp.Dispose();";
        command.env("TARA_SCREENSHOT_OUT", out_path).env(
            "TARA_SCREENSHOT_REGION",
            region
                .map(|(x, y, w, h)| format!("{x},{y},{w},{h}"))
                .unwrap_or_default(),
        );
        command.arg(script);
        let output = command
            .output()
            .map_err(|e| format!("Could not start Windows screen capture: {e}"))?;
        if !output.status.success() {
            return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
        }
    }
    #[cfg(target_os = "linux")]
    {
        let mut command = if let Some((x, y, w, h)) = region {
            let crop = format!("{w}x{h}+{x}+{y}");
            let mut c = Command::new("import");
            c.args(["-window", "root", "-crop", &crop]);
            c.arg(out_path);
            c
        } else {
            let mut c = Command::new("gnome-screenshot");
            c.args(["--file"]).arg(out_path);
            c
        };
        let output = command
            .output()
            .map_err(|e| format!("Could not start screen capture utility: {e}"))?;
        if !output.status.success() {
            return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
        }
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        let _ = region;
        return Err("Screen capture is unsupported on this operating system".to_string());
    }
    Ok(out_path.to_path_buf())
}

// ── 3. Audio & Speech Synthesis (TTS) ──────────────────────────────────────────

/// Generates valid RIFF WAV audio data natively with sine/chirp PCM synthesis.
pub fn generate_wav_pcm(duration_secs: f32, sample_rate: u32, freq_hz: f32) -> Vec<u8> {
    let num_samples = (duration_secs * sample_rate as f32) as usize;
    let mut pcm_samples = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        let sample_f = (2.0 * std::f32::consts::PI * freq_hz * t).sin() * 0.5;
        let sample_i16 = (sample_f * 32767.0) as i16;
        pcm_samples.push(sample_i16);
    }

    let mut wav_bytes = Vec::new();
    let data_len = (pcm_samples.len() * 2) as u32;
    let riff_len = 36 + data_len;

    // RIFF header
    wav_bytes.extend_from_slice(b"RIFF");
    wav_bytes.extend_from_slice(&riff_len.to_le_bytes());
    wav_bytes.extend_from_slice(b"WAVE");

    // fmt chunk
    wav_bytes.extend_from_slice(b"fmt ");
    wav_bytes.extend_from_slice(&16u32.to_le_bytes()); // Subchunk1Size
    wav_bytes.extend_from_slice(&1u16.to_le_bytes()); // AudioFormat (PCM = 1)
    wav_bytes.extend_from_slice(&1u16.to_le_bytes()); // NumChannels (1 = Mono)
    wav_bytes.extend_from_slice(&sample_rate.to_le_bytes());
    let byte_rate = sample_rate * 2;
    wav_bytes.extend_from_slice(&byte_rate.to_le_bytes());
    wav_bytes.extend_from_slice(&2u16.to_le_bytes()); // BlockAlign
    wav_bytes.extend_from_slice(&16u16.to_le_bytes()); // BitsPerSample

    // data chunk
    wav_bytes.extend_from_slice(b"data");
    wav_bytes.extend_from_slice(&data_len.to_le_bytes());
    for s in pcm_samples {
        wav_bytes.extend_from_slice(&s.to_le_bytes());
    }

    wav_bytes
}

pub fn synthesize_speech(
    text: &str,
    voice: &str,
    out_path: Option<&Path>,
) -> Result<Vec<u8>, String> {
    let clean_text = text.trim();
    if clean_text.is_empty() {
        return Err("Input text is empty".to_string());
    }

    #[cfg(target_os = "windows")]
    let wav_data = {
        let path = out_path.map(Path::to_path_buf).unwrap_or_else(|| {
            std::env::temp_dir().join(format!("tara-{}.wav", uuid::Uuid::new_v4()))
        });
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let text_b64 = base64::Engine::encode(
            &base64::engine::general_purpose::STANDARD,
            clean_text.as_bytes(),
        );
        let script = "$ErrorActionPreference='Stop'; Add-Type -AssemblyName System.Speech; $text=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($env:TARA_TTS_TEXT)); $s=New-Object System.Speech.Synthesis.SpeechSynthesizer; if ($env:TARA_TTS_VOICE) { $v=$s.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo } | Where-Object { $_.Name -eq $env:TARA_TTS_VOICE -or $_.Culture.Name -eq $env:TARA_TTS_VOICE } | Select-Object -First 1; if (-not $v) { throw \"Requested voice is not installed: $env:TARA_TTS_VOICE\" }; $s.SelectVoice($v.Name) }; $s.SetOutputToWaveFile($env:TARA_TTS_PATH); $s.Speak($text); $s.Dispose();";
        let output = Command::new("powershell.exe")
            .args([
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                script,
            ])
            .env("TARA_TTS_TEXT", text_b64)
            .env("TARA_TTS_VOICE", voice)
            .env("TARA_TTS_PATH", &path)
            .output()
            .map_err(|e| format!("Could not start Windows speech synthesis: {e}"))?;
        if !output.status.success() {
            return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
        }
        let bytes = fs::read(&path).map_err(|e| e.to_string())?;
        if out_path.is_none() {
            let _ = fs::remove_file(path);
        }
        bytes
    };
    #[cfg(target_os = "linux")]
    let wav_data = {
        let executable = if Command::new("espeak-ng").arg("--version").output().is_ok() {
            "espeak-ng"
        } else {
            "espeak"
        };
        let mut child = Command::new(executable)
            .args(["--stdout", "-v", voice])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("No speech synthesizer is installed (espeak-ng/espeak): {e}"))?;
        child
            .stdin
            .take()
            .ok_or("speech synthesizer stdin unavailable")?
            .write_all(clean_text.as_bytes())
            .map_err(|e| e.to_string())?;
        let output = child.wait_with_output().map_err(|e| e.to_string())?;
        if !output.status.success() {
            return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
        }
        output.stdout
    };
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    let wav_data: Vec<u8> =
        return Err("Speech synthesis is unsupported on this operating system".to_string());
    if !wav_data.starts_with(b"RIFF") || wav_data.len() < 44 {
        return Err("Speech synthesizer returned invalid WAV data".to_string());
    }
    if let Some(path) = out_path {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        fs::write(path, &wav_data).map_err(|e| e.to_string())?;
    }
    Ok(wav_data)
}

// ── 4. Transcription & Diarization ─────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiarizationSegment {
    pub speaker: String,
    pub start_sec: f64,
    pub end_sec: f64,
    pub text: String,
}

/// Transcribes WAV audio to text using a locally installed Whisper-compatible binary.
///
/// # Dependency
/// This function requires a locally installed Whisper-compatible CLI binary.
/// - Default executable name: `whisper`
/// - Override with env var: `TARA_WHISPER_BIN=/path/to/whisper`
/// - Whisper is NOT bundled with TARA. Install it separately:
///   `pip install openai-whisper` or use a compiled Whisper.cpp binary.
///
/// Audio input must be a valid RIFF WAV file (validated before processing).
pub fn transcribe_audio_bytes(
    audio: &[u8],
    language: &str,
) -> Result<Vec<DiarizationSegment>, String> {
    if audio.len() < 44 || &audio[0..4] != b"RIFF" || &audio[8..12] != b"WAVE" {
        return Err("Audio transcription requires a valid WAV audio file".to_string());
    }

    let executable = std::env::var_os("TARA_WHISPER_BIN").unwrap_or_else(|| "whisper".into());

    // Pre-validate: check if the whisper binary is reachable before creating temp files.
    // This gives a clear, actionable error instead of a confusing OS error.
    let probe = Command::new(&executable).arg("--help").output();
    if probe.is_err() {
        return Err(format!(
            "Whisper transcription engine not found: '{}'. \
             Install it with 'pip install openai-whisper' or set TARA_WHISPER_BIN \
             to the path of a compatible whisper binary. \
             Transcription requires a locally installed whisper, which is not bundled with TARA.",
            executable.to_string_lossy()
        ));
    }

    let temp_dir = std::env::temp_dir().join(format!("tara-stt-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&temp_dir).map_err(|e| e.to_string())?;
    let audio_path = temp_dir.join("input.wav");
    fs::write(&audio_path, audio).map_err(|e| e.to_string())?;
    let lang_code = language.split('-').next().unwrap_or(language);
    let output = Command::new(&executable)
        .arg(&audio_path)
        .args(["--output_dir"])
        .arg(&temp_dir)
        .args(["--output_format", "txt", "--language"])
        .arg(lang_code)
        .output()
        .map_err(|e| format!("Could not start Whisper transcription engine: {e}"))?;
    if !output.status.success() {
        let _ = fs::remove_dir_all(&temp_dir);
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
    }
    let transcript_path = temp_dir.join("input.txt");
    let transcript = fs::read_to_string(&transcript_path)
        .map_err(|e| format!("Transcription output missing: {e}"))?;
    let _ = fs::remove_dir_all(&temp_dir);
    if transcript.trim().is_empty() {
        return Ok(Vec::new());
    }
    let sample_rate = u32::from_le_bytes(
        audio[24..28]
            .try_into()
            .map_err(|_| "Invalid WAV sample rate")?,
    );
    let channels = u16::from_le_bytes(
        audio[22..24]
            .try_into()
            .map_err(|_| "Invalid WAV channels")?,
    ) as u32;
    let bits = u16::from_le_bytes(
        audio[34..36]
            .try_into()
            .map_err(|_| "Invalid WAV bit depth")?,
    ) as u32;
    let data_bytes = u32::from_le_bytes(
        audio[40..44]
            .try_into()
            .map_err(|_| "Invalid WAV data length")?,
    ) as u64;
    let bytes_per_second = sample_rate as u64 * channels as u64 * bits as u64 / 8;
    if bytes_per_second == 0 {
        return Err("WAV stream has invalid audio format metadata".to_string());
    }
    let duration = data_bytes as f64 / bytes_per_second as f64;
    Ok(vec![DiarizationSegment {
        speaker: "unknown".to_string(),
        start_sec: 0.0,
        end_sec: duration,
        text: transcript.trim().to_string(),
    }])
}

// ── JSON Dispatcher ─────────────────────────────────────────────────────────────

pub fn handle_multimedia_skill(action: &str, params: Value) -> Value {
    match action {
        // Procedural SVG vector image generation.
        // This is NOT a neural diffusion image generator.
        // It produces deterministic geometric SVG art derived from the prompt hash.
        "generate_image_svg" | "generate_image" => {
            let prompt = params
                .get("prompt")
                .and_then(|v| v.as_str())
                .unwrap_or("TARA vector graphic");
            let w = params.get("width").and_then(|v| v.as_u64()).unwrap_or(512) as u32;
            let h = params.get("height").and_then(|v| v.as_u64()).unwrap_or(512) as u32;
            let out = params
                .get("out_path")
                .and_then(|v| v.as_str())
                .map(PathBuf::from);

            match generate_image_svg(prompt, w, h, out.as_deref()) {
                Ok(res) => json!({
                    "status": "SUCCESS",
                    "generator_type": "procedural_svg",
                    "note": "Output is deterministic SVG vector art derived from prompt hash. Not a neural diffusion image.",
                    "result": res,
                }),
                Err(e) => json!({ "status": "ERROR", "error": e }),
            }
        }
        "take_screenshot" => {
            let out = params
                .get("out_path")
                .and_then(|v| v.as_str())
                .unwrap_or("screenshot.png");
            let region = match params.get("region").and_then(Value::as_str) {
                Some(value) => match parse_region_rect(value) {
                    Ok(rect) => Some(rect),
                    Err(error) => return json!({"status":"ERROR","error":error}),
                },
                None => None,
            };
            match save_screenshot_file(Path::new(out), region) {
                Ok(p) => json!({ "status": "SUCCESS", "file_path": p.to_string_lossy() }),
                Err(e) => json!({ "status": "ERROR", "error": e }),
            }
        }
        "text_to_speech" => {
            let text = params.get("text").and_then(|v| v.as_str()).unwrap_or("");
            let voice = params
                .get("voice")
                .and_then(|v| v.as_str())
                .unwrap_or("cedar");
            let out = params
                .get("out_path")
                .and_then(|v| v.as_str())
                .map(PathBuf::from);

            match synthesize_speech(text, voice, out.as_deref()) {
                Ok(bytes) => {
                    json!({ "status": "SUCCESS", "bytes_generated": bytes.len(), "voice": voice })
                }
                Err(e) => json!({ "status": "ERROR", "error": e }),
            }
        }
        "transcribe" => {
            let Some(audio_b64) = params.get("audio_data").and_then(Value::as_str) else {
                return json!({"status":"ERROR","error":"audio_data (base64 WAV) is required"});
            };
            let audio = match base64::Engine::decode(
                &base64::engine::general_purpose::STANDARD,
                audio_b64,
            ) {
                Ok(audio) => audio,
                Err(error) => {
                    return json!({"status":"ERROR","error":format!("Invalid audio_data: {error}")})
                }
            };
            let language = params
                .get("language")
                .and_then(Value::as_str)
                .unwrap_or("en");
            match transcribe_audio_bytes(&audio, language) {
                Ok(segments) => {
                    json!({"status":"SUCCESS","segments":segments,"count":segments.len()})
                }
                Err(error) => json!({"status":"ERROR","error":error}),
            }
        }
        _ => {
            json!({ "status": "ERROR", "error": format!("Unknown multimedia action: {}", action) })
        }
    }
}
