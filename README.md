# YT-Dubber

Automatically download Chinese YouTube videos, dub them into English, and upload to your YouTube channel. Runs entirely on your local GPU — no API costs.

## Pipeline

```
YouTube URL
    |
    v
1. Download video (yt-dlp)
    |
    v
2. Extract audio (ffmpeg)
    |
    v
3. Separate vocals from background music (demucs, GPU)
    |
    v
4. Transcribe Chinese speech from isolated vocals (Whisper large, GPU)
       - Word-level timestamps
       - Sentence splitting at punctuation boundaries
       - Time-based fallback split (max 8s per segment)
    |
    v
5. Translate Chinese to English (Helsinki-NLP by default, GPU)
       - Title translated separately for best quality
       - Optional: --translator gemma (requires >8GB free VRAM)
    |
    v
6. Detect speaker gender from vocal pitch -> auto-select English voice
    |
    v
7. Generate English speech (edge-tts, sequential + up to 5 retry passes)
    |
    v
8. Mix English speech over original background music (pydub)
       - Background at -5 dB, ducked -4 dB under speech
       - Gentle speedup (max 1.3x) if clip exceeds time slot
       - Fade-out on hard trim to avoid abrupt cuts
    |
    v
9. Merge audio into video (ffmpeg)
    |
    v
[optional] Upload to YouTube (YouTube Data API v3)
           - Private by default
           - Description includes original title, source URL, English transcript
```

## Requirements

- Windows 10/11
- Python 3.12
- ffmpeg
- NVIDIA GPU with CUDA 12.x driver

### 1. Install Python 3.12

`winget` often leaves the Microsoft Store stub as the active `python` command. Use the setup script to install and fix PATH automatically:

```powershell
winget install Python.Python.3.12 --accept-package-agreements
.\setup_python.ps1   # locates the real executable and adds it to user PATH
```

Open a new terminal and verify:
```powershell
python --version   # should print Python 3.12.x
pip --version
```

> **Why not just winget?** After install, `python` often still points to the Windows Store stub. `setup_python.ps1` uses the `py` launcher (`py -3.12`) to locate the real executable and patches your user PATH correctly.

### 2. Install ffmpeg

```powershell
winget install Gyan.FFmpeg
```

### 3. Install Python dependencies

```powershell
.\install.bat
```

## Usage

### Basic — dub only
```
run.bat https://youtu.be/VIDEO_ID
```

### Dub and upload to YouTube
```
run.bat https://youtu.be/VIDEO_ID --upload
```

### Upload an already-dubbed video (all steps skipped)
```
run.bat https://youtu.be/VIDEO_ID --upload
```

### Options
| Flag | Default | Description |
|------|---------|-------------|
| `--whisper` | `large` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large` |
| `--translator` | `helsinki` | `helsinki` (fast, ~300MB) or `gemma` (better quality, requires >8GB free VRAM) |
| `--voice` | auto | Edge-TTS voice — auto-detected from speaker gender pitch analysis |
| `--out` | `output` | Output folder |
| `--upload` | off | Upload to YouTube after dubbing |
| `--redo-from N` | off | Redo from step N onwards (clears N and all later steps) |
| `--redo-step N` | off | Redo only step N (all other steps stay cached) |

### List available English voices
```
py -3.12 list_voices.py
```

## Resumability

The pipeline is fully resumable. If it crashes at any step, re-run the same command — completed steps are skipped automatically.

Redo from a specific step (clears that step and all later ones):
```
run.bat https://youtu.be/VIDEO_ID --redo-from N
```

Redo only one step (all other steps stay cached):
```
run.bat https://youtu.be/VIDEO_ID --redo-step N
```

| N | Step | `--redo-from` clears... | `--redo-step` clears... |
|---|------|------------------------|------------------------|
| 2 | Audio extraction | Steps 2-8 | Step 2 only |
| 3 | Demucs separation | Steps 3-8 | Step 3 only |
| 4 | Transcription | Steps 4-8 | Step 4 only |
| 5 | Translation | Steps 5-8 | Step 5 only |
| 6 | TTS generation | Steps 6-8 | Step 6 only |
| 7 | Audio assembly | Steps 7-8 | Step 7 only |
| 8 | Final merge | Step 8 only | Step 8 only |

## Output files

```
output/
    VIDEO_ID_EN_dubbed.mp4       # Final dubbed video
    VIDEO_ID_transcript.json     # English transcript with timestamps
    _tmp/                        # Intermediate files (safe to delete after)
        VIDEO_ID.mp4             # Original downloaded video
        audio_source.wav         # Full quality stereo audio
        audio_whisper.wav        # Mono 16kHz (fallback)
        vocals_whisper.wav       # Isolated vocals 16kHz (used for transcription)
        VIDEO_ID_raw.json        # Raw Chinese transcript (sentence-level)
        demucs/                  # Separated vocals + background (mp3)
        tts/                     # Individual English TTS clips
        dubbed_audio.mp3         # Assembled English audio track
```

## YouTube Upload Setup

First-time setup:

1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Create a project
3. Enable **YouTube Data API v3**
4. Go to **Credentials** -> **Create OAuth 2.0 Client ID** (Desktop app)
5. Download the JSON and save as `client_secrets.json` in this folder
6. Go to **OAuth consent screen** -> **Test users** -> add your Gmail address

On first upload, a browser window opens for Google login. Credentials are cached in `token.pickle` — subsequent uploads skip the login.

Uploaded videos are set to **private** by default. Change visibility in YouTube Studio when ready.

## Notes

- **Gemma translator**: requires >8GB free VRAM. Even the E2B 4-bit quantized model (~1.2GB) needs ~4GB free VRAM at inference time. Run translation as a separate step (`--redo-step 5 --translator gemma`) to ensure other models are unloaded first.
- **Whisper transcribes isolated vocals**: demucs runs before Whisper so background music doesn't interfere with speech detection.
- **Gender detection**: pitch analysis on the separated vocals track — no extra model needed. Female voice (>165Hz avg) maps to `en-US-JennyNeural`, male to `en-US-ChristopherNeural`. Override with `--voice`.

## Hugging Face Token (optional)

Avoids rate limiting when downloading translation models:

1. Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Set it permanently:
```
setx HF_TOKEN "your_token_here"
```
