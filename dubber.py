"""
YT-Dubber: Download Chinese YouTube videos, dub them in English, and upload to your channel.
Uses local GPU (CUDA) for fast processing.
"""

import os
import sys
import json
import asyncio
import subprocess
import tempfile
from pathlib import Path

# Force UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import yt_dlp
import whisper
from transformers import MarianMTModel, MarianTokenizer
import edge_tts
from pydub import AudioSegment


# ── GPU check ────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def banner(msg):
    print(f"\n{'-'*54}")
    print(f"  {msg}")
    print(f"{'-'*54}")


# ── Resumability helper ───────────────────────────────────────────────────────
def done(*paths: Path) -> bool:
    """True if every path exists and is non-empty."""
    return all(p.exists() and p.stat().st_size > 0 for p in paths)


# ── Step 1: Download ──────────────────────────────────────────────────────────
def download_video(url: str, out_dir: Path) -> tuple[Path, str, float]:
    """Download best quality mp4, resuming partial downloads automatically."""
    # Fetch info first (no download) to determine the video id and check if already done
    ydl_opts_info = {"quiet": True, "noplaylist": True}
    with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
        info = ydl.extract_info(url, download=False)
        video_id = info["id"]
        title = info.get("title", video_id)
        duration = float(info.get("duration", 0))

    video_path = out_dir / f"{video_id}.mp4"

    if done(video_path):
        print(f"  [SKIP] Already downloaded: {video_path}")
        return video_path, title, duration

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        "continuedl": True,   # resume partial downloads
        "quiet": False,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    if not video_path.exists():
        matches = list(out_dir.glob(f"{video_id}.*"))
        if not matches:
            raise FileNotFoundError(f"Could not find downloaded file for {video_id}")
        video_path = matches[0]

    return video_path, title, duration


# ── Step 2: Extract audio ─────────────────────────────────────────────────────
def extract_audio(video_path: Path, out_dir: Path) -> Path:
    """Extract stereo 44.1 kHz WAV for demucs + a mono 16 kHz copy for Whisper."""
    # Full quality stereo for demucs
    wav_path = out_dir / "audio_source.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-vn",
         "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", str(wav_path)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    # Mono 16 kHz for Whisper
    whisper_wav = out_dir / "audio_whisper.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav_path), "-ar", "16000", "-ac", "1", str(whisper_wav)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return wav_path, whisper_wav


# ── Step 2b: Separate vocals from background music (demucs) ──────────────────
def separate_audio(wav_path: Path, out_dir: Path) -> tuple[Path, Path]:
    """Use demucs htdemucs to split vocals and background music. GPU-accelerated."""
    import shutil as _shutil

    sep_dir = out_dir / "demucs"
    sep_dir.mkdir(exist_ok=True)

    # Ensure ffmpeg is findable by demucs subprocess
    env = os.environ.copy()
    ffmpeg_bin = _shutil.which("ffmpeg")
    if ffmpeg_bin:
        env["PATH"] = str(Path(ffmpeg_bin).parent) + os.pathsep + env.get("PATH", "")

    print("  Running demucs (first run downloads ~80 MB model)...")
    result = subprocess.run(
        [
            sys.executable, "-m", "demucs",
            "--two-stems", "vocals",
            "--mp3",                  # use mp3 output (lameenc) — avoids torchaudio WAV backend issues
            "--device", DEVICE,
            "--out", str(sep_dir.resolve()),
            str(wav_path.resolve()),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr[-2000:])
        raise RuntimeError("demucs failed — see error above")

    stem = wav_path.stem
    vocals_path = sep_dir / "htdemucs" / stem / "vocals.mp3"
    background_path = sep_dir / "htdemucs" / stem / "no_vocals.mp3"
    return vocals_path, background_path


# ── Gender detection via pitch analysis ───────────────────────────────────────
def detect_gender(vocals_path: Path) -> str:
    """
    Estimate speaker gender from average pitch of the vocal track.
    Uses autocorrelation on short frames — no extra model needed.
    Returns 'female' or 'male'.
    """
    import numpy as np
    from pydub import AudioSegment

    audio = AudioSegment.from_file(str(vocals_path)).set_channels(1).set_frame_rate(16000)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    samples /= max(np.max(np.abs(samples)), 1)

    frame_size = 1600   # 100ms frames at 16kHz
    min_lag = int(16000 / 300)  # 300 Hz max (below female range)
    max_lag = int(16000 / 60)   # 60 Hz min (below male range)

    pitches = []
    for start in range(0, len(samples) - frame_size, frame_size):
        frame = samples[start:start + frame_size]
        if np.max(np.abs(frame)) < 0.01:  # skip silence
            continue
        # Autocorrelation
        corr = np.correlate(frame, frame, mode="full")[len(frame):]
        peak_range = corr[min_lag:max_lag]
        if peak_range.size == 0:
            continue
        lag = np.argmax(peak_range) + min_lag
        pitches.append(16000 / lag)

    if not pitches:
        print("  [WARN] Could not detect pitch — defaulting to male voice")
        return "male"

    avg_pitch = np.median(pitches)
    gender = "female" if avg_pitch > 165 else "male"
    print(f"  Detected pitch: {avg_pitch:.0f} Hz -> {gender} voice")
    return gender


VOICE_MAP = {
    "en": {
        "female": "en-US-JennyNeural",
        "male":   "en-US-ChristopherNeural",
    },
    "zh": {
        "female": "zh-CN-XiaoxiaoNeural",
        "male":   "zh-CN-YunxiNeural",
    },
}

HELSINKI_MODEL = {
    "zh-en": "Helsinki-NLP/opus-mt-zh-en",
    "en-zh": "Helsinki-NLP/opus-mt-en-zh",
}

DUBBED_SUFFIX = {
    "en": "_EN_dubbed",
    "zh": "_ZH_dubbed",
}


# ── Language detection via Whisper ────────────────────────────────────────────
def detect_language(vocals_wav: Path, whisper_model) -> str:
    """
    Use Whisper's built-in language detector on the first 30 s of vocal audio.
    Returns a language code such as 'zh', 'en', 'ja', etc.
    """
    audio = whisper.load_audio(str(vocals_wav))
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio, n_mels=whisper_model.dims.n_mels).to(DEVICE)
    _, probs = whisper_model.detect_language(mel)
    lang = max(probs, key=probs.get)
    print(f"  Detected language: {lang} (confidence: {probs[lang]:.2%})")
    return lang


# ── Step 3: Transcribe with Whisper (GPU) ─────────────────────────────────────
def transcribe(wav_path: Path, whisper_model, language: str = "zh") -> list[dict]:
    """
    Transcribe Chinese audio with word-level timestamps, then split into
    sentence-level segments at punctuation boundaries so each TTS clip
    covers exactly one sentence.
    """
    # Encourage sentence-ending punctuation for the source language
    _initial_prompts = {
        "zh": "请使用标点符号。例如：你好！今天天气很好。我们去吃饭吧？",
        "en": "Please use proper punctuation. For example: Hello! How are you? Let's go.",
    }
    initial_prompt = _initial_prompts.get(language, "")

    print(f"  Running on: {DEVICE.upper()}")
    result = whisper_model.transcribe(
        str(wav_path),
        language=language,
        task="transcribe",
        word_timestamps=True,
        fp16=(DEVICE == "cuda"),
        verbose=False,
        initial_prompt=initial_prompt,
    )

    SENTENCE_END = set("。！？…!?")
    MAX_SEG_SECS = 8.0   # hard cap — split any segment longer than this by time

    def flush(buf_words, buf_start):
        text = "".join(w["word"] for w in buf_words).strip()
        if not text:
            return []
        end = buf_words[-1]["end"]
        # If still too long, split by time buckets
        if end - buf_start <= MAX_SEG_SECS:
            return [{"start": buf_start, "end": end, "text": text}]
        # Time-based split: bucket words into MAX_SEG_SECS chunks
        result_segs = []
        bucket, b_start = [], buf_words[0]["start"]
        for w in buf_words:
            bucket.append(w)
            if w["end"] - b_start >= MAX_SEG_SECS:
                t = "".join(x["word"] for x in bucket).strip()
                if t:
                    result_segs.append({"start": b_start, "end": w["end"], "text": t})
                bucket, b_start = [], w["end"]
        if bucket:
            t = "".join(x["word"] for x in bucket).strip()
            if t:
                result_segs.append({"start": b_start, "end": bucket[-1]["end"], "text": t})
        return result_segs

    sentences = []
    for seg in result["segments"]:
        words = seg.get("words", [])
        if not words:
            text = seg["text"].strip()
            if text:
                sentences.append({"start": seg["start"], "end": seg["end"], "text": text})
            continue

        buf_words, buf_start = [], None
        for w in words:
            if buf_start is None:
                buf_start = w["start"]
            buf_words.append(w)
            if any(c in w["word"] for c in SENTENCE_END):
                sentences.extend(flush(buf_words, buf_start))
                buf_words, buf_start = [], None

        if buf_words:
            sentences.extend(flush(buf_words, buf_start))

    print(f"  Found {len(sentences)} sentences (from {len(result['segments'])} raw segments)")
    return sentences


# ── Step 4a: Translate with Helsinki-NLP (GPU) ────────────────────────────────
def translate_title(title: str, translate_fn) -> str:
    """
    Translate a title by splitting at common delimiters if it is long.
    Each piece is translated independently then rejoined with ' | '.
    Splitters tried in order: fullwidth pipe, regular pipe, em-dash,
    fullwidth colon, regular colon, slash. Falls back to full title if
    no splitter is found or the title is short (<= 30 chars).
    """
    import re
    SPLITTERS = [r'\uff5c', r'\|', r'\u2014', r'\u2013', r'\uff1a', r':', r'/']
    MAX_CHARS = 30

    if len(title) <= MAX_CHARS:
        return translate_fn([title])[0]

    for pattern in SPLITTERS:
        parts = [p.strip() for p in re.split(pattern, title) if p.strip()]
        if len(parts) > 1:
            translated_parts = translate_fn(parts)
            return " | ".join(translated_parts)

    # No splitter found — translate whole title
    return translate_fn([title])[0]


def translate_helsinki(texts: list[str], tokenizer, model) -> list[str]:
    """Batch-translate a list of Chinese strings to English using Helsinki-NLP."""
    batch_size = 16
    translated = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(**inputs, num_beams=4)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translated.extend(decoded)
        print(f"  Translated {min(i + batch_size, len(texts))}/{len(texts)}", end="\r")
    print()
    return translated


# ── Step 4b: Translate with TranslateGemma-4B (4-bit, GPU) ───────────────────
def translate_gemma(texts: list[str], src_lang: str = "Chinese", tgt_lang: str = "English") -> list[str]:
    """Translate a list of strings using TranslateGemma-4B (4-bit quantized)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # NOTE: Even the E2B 4-bit quantized model requires ~4GB VRAM.
    # An 8GB GPU is NOT enough when other models (Whisper, demucs) are also loaded.
    # Free VRAM first by running translation as a separate step (--redo-step 5).
    model_id = "unsloth/gemma-4-E2B-it-unsloth-bnb-4bit"
    print(f"  Loading {model_id} (first run ~1.2 GB)...")

    # Already quantized to 4-bit by unsloth — no BitsAndBytesConfig needed
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
    )
    model.eval()

    translated = []
    total = len(texts)
    for i, text in enumerate(texts):
        messages = [{"role": "user", "content": f"Translate the following text from {src_lang} to {tgt_lang}:\n{text}"}]
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(DEVICE)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][input_ids.shape[-1]:]
        translated.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        print(f"  Translated {i+1}/{total}", end="\r")
    print()

    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return translated


# ── Step 5: Generate TTS with edge-tts ───────────────────────────────────────
# TODO: look into VibeVoice as a higher-quality TTS backend
async def _tts_one(text: str, path: Path, voice: str) -> bool:
    """Attempt a single TTS generation. Returns True on success."""
    try:
        communicate = edge_tts.Communicate(text, voice=voice)
        await communicate.save(str(path))
        return path.exists() and path.stat().st_size > 0
    except Exception:
        return False


async def generate_tts(segments: list[dict], tts_dir: Path, voice: str, max_passes: int = 5) -> list[dict]:
    """
    Generate TTS sequentially, then do up to max_passes retry passes
    over any failed segments until all are covered or passes exhausted.
    """
    tts_dir.mkdir(exist_ok=True)
    paths = [tts_dir / f"seg_{i:05d}.mp3" for i in range(len(segments))]
    total = len(segments)

    # Pass 1: sequential generation — skip already completed segments
    already = sum(1 for p in paths if p.exists() and p.stat().st_size > 0)
    if already == total:
        print(f"  [SKIP] All {total} TTS segments already exist")
        return [{**seg, "tts_path": paths[i]} for i, seg in enumerate(segments)]

    print(f"  Pass 1/{max_passes} - generating {total - already} segments ({already} already done)...")
    for i, seg in enumerate(segments):
        if paths[i].exists() and paths[i].stat().st_size > 0:
            continue
        print(f"  [{i+1:04d}/{total}] generating...  {seg['translated'][:55]}", end="\r")
        ok = await _tts_one(seg["translated"], paths[i], voice)
        status = "ok " if ok else "FAIL"
        print(f"  [{i+1:04d}/{total}] {status}  {seg['translated'][:55]}")
    print(f"  Pass 1 done.")

    # Retry passes for any failures
    for pass_num in range(2, max_passes + 1):
        failed = [i for i, p in enumerate(paths) if not p.exists() or p.stat().st_size == 0]
        if not failed:
            break
        print(f"  Pass {pass_num}/{max_passes} - retrying {len(failed)} failed segments...")
        await asyncio.sleep(2)  # brief pause before retrying
        for idx, i in enumerate(failed):
            # Remove corrupt/empty file before retry
            if paths[i].exists():
                paths[i].unlink()
            print(f"  [retry {idx+1:03d}/{len(failed)}] generating...  {segments[i]['translated'][:50]}", end="\r")
            ok = await _tts_one(segments[i]["translated"], paths[i], voice)
            status = "ok " if ok else "FAIL"
            print(f"  [retry {idx+1:03d}/{len(failed)}] {status}  {segments[i]['translated'][:50]}")
        print(f"  Pass {pass_num} done.")

    # Final coverage report
    still_missing = [i for i, p in enumerate(paths) if not p.exists() or p.stat().st_size == 0]
    coverage = (total - len(still_missing)) / total * 100
    print(f"  Coverage: {total - len(still_missing)}/{total} segments ({coverage:.1f}%)")
    if still_missing:
        print(f"  [WARN] {len(still_missing)} segments could not be generated after {max_passes} passes")

    return [{**seg, "tts_path": paths[i]} for i, seg in enumerate(segments)]


# ── Step 6: Assemble dubbed audio track ──────────────────────────────────────
def assemble_audio(segments: list[dict], total_duration: float, background_path: Path = None) -> AudioSegment:
    """
    Mix dubbed speech over original background music.
    - Background plays at -5 dB throughout.
    - TTS clips placed at natural speed with no forced speedup.
    - Background ducked an extra -4 dB while speech is playing.
    """
    total_ms = int(total_duration * 1000) + 2000

    # Load background music or fall back to silence
    if background_path and background_path.exists():
        background = AudioSegment.from_file(str(background_path))
        if len(background) < total_ms:
            background = background + AudioSegment.silent(duration=total_ms - len(background))
        else:
            background = background[:total_ms]
        background = background - 5  # global -5 dB
        print(f"  Background music loaded ({len(background)/1000:.1f}s)")
    else:
        background = AudioSegment.silent(duration=total_ms)
        print("  [WARN] No background track - using silence")

    timeline = background
    FADE_MS = 150

    valid_segs = [s for s in segments if Path(s["tts_path"]).exists() and Path(s["tts_path"]).stat().st_size > 0]
    missing = len(segments) - len(valid_segs)

    if valid_segs:
        first_start = valid_segs[0]["start"]
        print(f"  First segment at {first_start:.1f}s — {valid_segs[0].get('translated','')[:60]}")

    for idx, seg in enumerate(valid_segs):
        clip = AudioSegment.from_mp3(str(seg["tts_path"]))
        start_ms = int(seg["start"] * 1000)

        # Slot = time until next segment starts (or end of video)
        if idx + 1 < len(valid_segs):
            next_start_ms = int(valid_segs[idx + 1]["start"] * 1000)
        else:
            next_start_ms = total_ms
        slot_ms = next_start_ms - start_ms

        # Strategy depends on slot size:
        # - Long slots (>10s): let clip play naturally, it's fine if it overlaps slightly
        # - Medium slots (3-10s): gentle speedup up to 1.3x, then fade
        # - Short slots (<3s): speedup up to 1.5x, then fade
        if slot_ms >= 10000:
            # Long segment — don't constrain, English will naturally be longer than Chinese
            max_clip_ms = int(slot_ms * 1.3)  # allow 30% overflow into next segment's silence
        elif slot_ms >= 3000:
            max_clip_ms = slot_ms - 150        # 150ms gap for medium segments
        else:
            max_clip_ms = max(slot_ms - 50, 300)  # tight but keep at least 300ms

        max_speed = 1.3 if slot_ms >= 3000 else 1.5

        if len(clip) > max_clip_ms:
            speed = min(len(clip) / max_clip_ms, max_speed)
            clip = clip.speedup(playback_speed=speed)

        if len(clip) > max_clip_ms:
            clip = clip[:max_clip_ms].fade_out(min(FADE_MS, max_clip_ms // 2))

        speech_len = min(len(clip), total_ms - start_ms)
        if speech_len <= 0:
            continue

        # Duck background -4 dB under speech window
        end_ms = start_ms + speech_len
        ducked = timeline[start_ms:end_ms] - 4
        timeline = timeline[:start_ms] + ducked + timeline[end_ms:]

        timeline = timeline.overlay(clip, position=start_ms)

    if missing:
        print(f"  [WARN] {missing} segments had no audio - gaps will appear in output")

    return timeline


# ── Step 7: Merge dubbed audio into video ────────────────────────────────────
def merge(video_path: Path, dubbed_audio_path: Path, output_path: Path):
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(dubbed_audio_path),
            "-c:v", "copy",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            str(output_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ── Archive helper ────────────────────────────────────────────────────────────
def archive_output(out_dir: Path, video_id: str) -> Path:
    """
    Move *out_dir* to  <out_dir.parent>/archive/<video_id>/
    then recreate an empty *out_dir*.  Returns the archive destination path.
    """
    import shutil

    dest = out_dir.parent / "archive" / video_id
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(out_dir), str(dest))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [ARCHIVE] Moved output -> {dest}")
    print(f"  [ARCHIVE] Recreated empty {out_dir}")
    return dest


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run(url: str, out_dir: Path, whisper_size: str = "large", voice: str = None, translator: str = "helsinki", archive: bool = True):
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

    # ── 1. Download
    banner("Step 1/8 - Downloading video")
    video_path, title, duration = download_video(url, tmp_dir)
    print(f"  Title   : {title}")
    print(f"  Duration: {duration:.0f}s")
    print(f"  File    : {video_path}")
    video_id = video_path.stem

    # ── 2. Extract audio
    banner("Step 2/8 - Extracting audio")
    wav_path     = tmp_dir / "audio_source.wav"
    whisper_wav  = tmp_dir / "audio_whisper.wav"
    if done(wav_path, whisper_wav):
        print("  [SKIP] Audio already extracted")
    else:
        wav_path, whisper_wav = extract_audio(video_path, tmp_dir)

    # ── 3. Separate vocals / background
    sep_dir        = tmp_dir / "demucs" / "htdemucs" / "audio_source"
    background_path = sep_dir / "no_vocals.mp3"
    vocals_path     = sep_dir / "vocals.mp3"
    banner("Step 3/8 - Separating vocals from background music (demucs)")
    if done(background_path, vocals_path):
        print("  [SKIP] Separation already done")
    else:
        vocals_path, background_path = separate_audio(wav_path, tmp_dir)
    print(f"  Background: {background_path}")

    # ── 4. Transcribe (with auto language detection)
    raw_transcript_path = tmp_dir / f"{video_id}_raw.json"
    banner(f"Step 4/8 - Transcribing (Whisper {whisper_size})")

    # Use isolated vocals for transcription — cleaner than mixed audio
    vocals_wav = tmp_dir / "vocals_whisper.wav"
    if not vocals_wav.exists():
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(vocals_path),
             "-ar", "16000", "-ac", "1", str(vocals_wav)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    print("  Loading Whisper model (first run downloads ~3 GB)...")
    whisper_model = whisper.load_model(whisper_size, device=DEVICE)

    # Detect language from vocals, then decide translation direction
    src_lang = detect_language(vocals_wav, whisper_model)
    # Collapse regional variants (e.g. 'yue' Cantonese → treat as 'zh')
    if src_lang in ("yue", "zh-TW", "wuu"):
        src_lang = "zh"
    tgt_lang = "en" if src_lang == "zh" else "zh"
    print(f"  Direction: {src_lang} → {tgt_lang}")

    if done(raw_transcript_path):
        print("  [SKIP] Transcription already done — loading from file")
        with open(raw_transcript_path, encoding="utf-8") as f:
            segments = json.load(f)
        del whisper_model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    else:
        segments = transcribe(vocals_wav, whisper_model, language=src_lang)
        del whisper_model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        with open(raw_transcript_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)

    # Auto-detect gender from vocals if voice not manually specified
    if voice is None:
        gender = detect_gender(vocals_path)
        voice = VOICE_MAP[tgt_lang][gender]
        print(f"  Auto-selected voice: {voice}")

    # ── 5. Translate
    transcript_path = out_dir / f"{video_id}_transcript.json"
    _lang_names = {"zh": "Chinese", "en": "English"}
    src_name, tgt_name = _lang_names.get(src_lang, src_lang), _lang_names.get(tgt_lang, tgt_lang)
    banner(f"Step 5/8 - Translating {src_name} -> {tgt_name}")
    if done(transcript_path):
        print("  [SKIP] Translation already done — loading from file")
        with open(transcript_path, encoding="utf-8") as f:
            saved = json.load(f)
        if isinstance(saved, dict):
            title_translated = saved.get("title", title)
            translated_segments = saved["segments"]
        else:
            title_translated = title
            translated_segments = saved
        for seg, ts in zip(segments, translated_segments):
            seg["translated"] = ts["text"]
    else:
        seg_texts = [s["text"] for s in segments]
        direction_key = f"{src_lang}-{tgt_lang}"

        if translator == "gemma":
            print(f"  Using TranslateGemma-4B")
            title_translated = translate_title(title, lambda t: translate_gemma(t, src_name, tgt_name))
            seg_translations = translate_gemma(seg_texts, src_name, tgt_name)
        else:
            model_name = HELSINKI_MODEL.get(direction_key, HELSINKI_MODEL["zh-en"])
            print(f"  Using {model_name} (first run downloads ~300 MB)...")
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            mt_model = MarianMTModel.from_pretrained(model_name).to(DEVICE)
            title_translated = translate_title(title, lambda texts: translate_helsinki(texts, tokenizer, mt_model))
            seg_translations = translate_helsinki(seg_texts, tokenizer, mt_model)
            del mt_model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        print(f"  Title ({tgt_lang.upper()}): {title_translated}")

        for seg, t in zip(segments, seg_translations):
            seg["translated"] = t

        translated_segments = [{"start": s["start"], "end": s["end"], "text": s["translated"]} for s in segments]
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump({"title": title_translated, "segments": translated_segments}, f, ensure_ascii=False, indent=2)
        print(f"  Saved: {transcript_path}")

    # ── 6. TTS  (generate_tts already skips existing files internally)
    banner(f"Step 6/8 - Generating {tgt_name} speech [{voice}]")
    tts_dir = tmp_dir / "tts"
    segments = asyncio.run(generate_tts(segments, tts_dir, voice))

    # ── 7. Assemble audio
    dubbed_audio_path = tmp_dir / "dubbed_audio.mp3"
    banner("Step 7/8 - Assembling dubbed audio track")
    if done(dubbed_audio_path):
        print("  [SKIP] Dubbed audio already assembled")
    else:
        total_dur = translated_segments[-1]["end"] if translated_segments else duration
        dubbed_audio = assemble_audio(segments, total_dur, background_path=background_path)
        dubbed_audio.export(str(dubbed_audio_path), format="mp3", bitrate="192k")

    # ── 8. Merge
    output_path = out_dir / f"{video_id}{DUBBED_SUFFIX[tgt_lang]}.mp4"
    banner("Step 8/8 - Merging audio into video")
    if done(output_path):
        print("  [SKIP] Final video already merged")
    else:
        merge(video_path, dubbed_audio_path, output_path)

    # ── Archive: move out_dir to archive/<original_title>/ then recreate empty out_dir
    if archive:
        banner("Archiving output directory")
        archive_dest = archive_output(out_dir, video_id)
        output_path    = archive_dest / output_path.name
        transcript_path = archive_dest / transcript_path.name

    banner("Done!")
    print(f"  Output: {output_path}\n")
    return output_path, title_translated, url, transcript_path, duration


# ── Step cache map (used by --redo-from) ─────────────────────────────────────
def clear_from_step(step: int, out_dir: Path, video_id: str):
    """Delete cached outputs for steps >= step so the pipeline reruns them."""
    import shutil
    tmp_dir = out_dir / "_tmp"

    step_files = {
        2: [tmp_dir / "audio_source.wav", tmp_dir / "audio_whisper.wav"],
        3: [tmp_dir / "demucs"],
        4: [tmp_dir / f"{video_id}_raw.json"],
        5: [out_dir / f"{video_id}_transcript.json"],
        6: [tmp_dir / "tts"],
        7: [tmp_dir / "dubbed_audio.mp3"],
        8: [out_dir / f"{video_id}_EN_dubbed.mp4", out_dir / f"{video_id}_ZH_dubbed.mp4"],
    }

    cleared = []
    for s in range(step, 9):
        for p in step_files.get(s, []):
            p = Path(p)
            if p.exists():
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
                cleared.append(str(p))

    if cleared:
        print(f"  Cleared {len(cleared)} cached file(s) from step {step} onwards:")
        for c in cleared:
            print(f"    - {c}")
    else:
        print(f"  Nothing to clear from step {step} onwards.")


def clear_step(step: int, out_dir: Path, video_id: str):
    """Delete cached outputs for exactly one step."""
    import shutil
    tmp_dir = out_dir / "_tmp"

    step_files = {
        2: [tmp_dir / "audio_source.wav", tmp_dir / "audio_whisper.wav"],
        3: [tmp_dir / "demucs"],
        4: [tmp_dir / f"{video_id}_raw.json"],
        5: [out_dir / f"{video_id}_transcript.json"],
        6: [tmp_dir / "tts"],
        7: [tmp_dir / "dubbed_audio.mp3"],
        8: [out_dir / f"{video_id}_EN_dubbed.mp4", out_dir / f"{video_id}_ZH_dubbed.mp4"],
    }

    cleared = []
    for p in step_files.get(step, []):
        p = Path(p)
        if p.exists():
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
            cleared.append(str(p))

    if cleared:
        print(f"  Cleared {len(cleared)} cached file(s) for step {step}:")
        for c in cleared:
            print(f"    - {c}")
    else:
        print(f"  Nothing to clear for step {step}.")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dub a Chinese YouTube video into English")
    parser.add_argument("url", help="YouTube URL")
    parser.add_argument("--out", default="output", help="Output folder (default: output)")
    parser.add_argument(
        "--whisper",
        default="large",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: large, best quality)",
    )
    parser.add_argument(
        "--voice",
        default=None,
        help="Edge-TTS voice (default: auto-detected from speaker gender)",
    )
    parser.add_argument(
        "--translator",
        default="helsinki",
        choices=["helsinki", "gemma"],
        help="Translation engine: helsinki (default, fast, ~300MB) or gemma (better quality, requires ~4GB VRAM)",
    )
    parser.add_argument("--upload", action="store_true", help="Upload to YouTube after dubbing")
    parser.add_argument(
        "--no-skip-long",
        action="store_true",
        default=False,
        help="Upload even if video is longer than 15 minutes (skip-long is on by default for new account limits)",
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        default=False,
        help="Skip archiving output dir to archive/<title>/ after completion (archive is on by default)",
    )
    parser.add_argument(
        "--redo-from",
        type=int,
        choices=range(2, 9),
        metavar="N",
        help="Redo from step N onwards (2=audio extract, 3=demucs, 4=transcribe, 5=translate, 6=TTS, 7=assemble, 8=merge)",
    )
    parser.add_argument(
        "--redo-step",
        type=int,
        choices=range(2, 9),
        metavar="N",
        help="Redo only step N (same step numbers as --redo-from)",
    )
    args = parser.parse_args()

    print(f"\n{'='*54}")
    print("  YT-Dubber  |  Chinese -> English Dubbing Pipeline")
    print(f"  GPU: {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU only'}")
    print(f"{'='*54}")

    if args.redo_from or args.redo_step:
        with yt_dlp.YoutubeDL({"quiet": True, "noplaylist": True}) as ydl:
            info = ydl.extract_info(args.url, download=False)
            video_id = info["id"]
        if args.redo_from:
            banner(f"Clearing cache from step {args.redo_from} onwards")
            clear_from_step(args.redo_from, Path(args.out), video_id)
        if args.redo_step:
            banner(f"Clearing cache for step {args.redo_step} only")
            clear_step(args.redo_step, Path(args.out), video_id)

    out_path, title, source_url, transcript_path, duration = run(
        url=args.url,
        out_dir=Path(args.out),
        whisper_size=args.whisper,
        voice=args.voice,
        translator=args.translator,
        archive=not args.no_archive,
    )

    if args.upload:
        if duration > 15 * 60 and not args.no_skip_long:
            print(f"\n  [SKIP UPLOAD] Video is {duration/60:.1f} min — over 15-min new-account limit. Use --no-skip-long to override.")
        else:
            from uploader import upload_to_youtube
            upload_to_youtube(str(out_path), f"{title} [English Dubbed]", source_url=source_url, source_title=title, transcript_path=str(transcript_path))
