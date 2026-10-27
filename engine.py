"""
engine.py - តក្កវិជ្ជាក្រោយខ្នងរបស់ BUNYIM Dubbing Studio (Desktop)
គ្មាន GUI ក្នុងឯកសារនេះ ដូច្នេះសាកល្បងបានដោយឡែក
"""
from __future__ import annotations



import asyncio
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
import wave
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# ថេរ (constants)
# ---------------------------------------------------------------------------
SPEAKER_LABELS = {"male": "ប្រុស", "female": "ស្រី", "narrator": "អ្នកនាំរឿង"}
SPEAKER_KEYS = {v: k for k, v in SPEAKER_LABELS.items()}
DEFAULT_SPEAKER = SPEAKER_LABELS["narrator"]

COLUMNS = ["start", "end", "speaker", "original", "khmer"]

EDGE_VOICES = {
    "km-KH-PisethNeural": "ពិសិដ្ឋ (Piseth) - សំឡេងប្រុស",
    "km-KH-SreymomNeural": "ស្រីមុំ (Sreymom) - សំឡេងស្រី",
}

SOURCE_LANGS = {
    "ចិន (Chinese)": "Chinese",
    "អង់គ្លេស (English)": "English",
    "កូរ៉េ (Korean)": "Korean",
    "ជប៉ុន (Japanese)": "Japanese",
    "ថៃ (Thai)": "Thai",
    "ស្វ័យប្រវត្តិ (Auto-detect)": "the original language (detect it automatically)",
}

TTS_SR = 24000  # sample rate នៃបទសំឡេងខ្មែរ


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------
def empty_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "start": pd.Series(dtype="float64"),
            "end": pd.Series(dtype="float64"),
            "speaker": pd.Series(dtype="object"),
            "original": pd.Series(dtype="object"),
            "khmer": pd.Series(dtype="object"),
        }
    )


def clean_df(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """ធ្វើឲ្យតារាងស្អាត: ប្រភេទទិន្នន័យត្រឹមត្រូវ, លុបជួរទទេ, តម្រៀបតាមពេលវេលា"""
    if df is None or len(df) == 0:
        return empty_df()
    d = df.copy()
    for c in COLUMNS:
        if c not in d.columns:
            d[c] = None
    d = d[COLUMNS]
    d["start"] = pd.to_numeric(d["start"], errors="coerce")
    d["end"] = pd.to_numeric(d["end"], errors="coerce")
    for c in ("original", "khmer"):
        d[c] = d[c].fillna("").astype(str).str.strip()
    d["speaker"] = d["speaker"].where(d["speaker"].isin(SPEAKER_KEYS), DEFAULT_SPEAKER)
    # ជួរដែលគ្មានអត្ថបទទាំងពីរ ត្រូវលុប
    d = d[(d["original"] != "") | (d["khmer"] != "")]
    d["start"] = d["start"].fillna(0.0).clip(lower=0.0)
    d["end"] = d["end"].fillna(d["start"] + 2.0)
    d["end"] = np.maximum(d["end"], d["start"] + 0.3)
    d = d.sort_values("start", kind="stable").reset_index(drop=True)
    return d


def demo_df() -> pd.DataFrame:
    rows = [
        (0.5, 3.5, "ប្រុស", "你好，你过得好吗？", "សួស្តី តើអ្នកសុខសប្បាយជាទេ?"),
        (4.0, 6.5, "ស្រី", "我很好，谢谢。", "ខ្ញុំសុខសប្បាយ អរគុណ។"),
        (7.0, 10.0, "អ្នកនាំរឿង", "我们一起去看电影吧！", "តោះទៅមើលរឿងជាមួយគ្នា!"),
    ]
    return clean_df(pd.DataFrame(rows, columns=COLUMNS))


# ---------------------------------------------------------------------------
# SRT
# ---------------------------------------------------------------------------
def fmt_srt_time(sec: float) -> str:
    ms = int(round(max(0.0, float(sec)) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def fmt_clock(sec: float) -> str:
    sec = max(0.0, float(sec))
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


_SRT_TIME = re.compile(
    r"\s*(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)"
)


def _to_sec(h: str, m: str, s: str, frac: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(frac) / (10 ** len(frac))


def parse_srt(text: str) -> list[dict]:
    text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("﻿")
    out: list[dict] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.split("\n")
        idx = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if idx is None:
            continue
        m = _SRT_TIME.match(lines[idx])
        if not m:
            continue
        g = m.groups()
        start = _to_sec(*g[0:4])
        end = _to_sec(*g[4:8])
        body = " ".join(ln.strip() for ln in lines[idx + 1:] if ln.strip())
        body = re.sub(r"<[^>]+>", "", body).strip()
        if body:
            out.append({"start": start, "end": end, "text": body})
    return out


def to_srt(df: pd.DataFrame, prefer: str = "khmer") -> str:
    d = clean_df(df)
    parts = []
    n = 0
    for _, r in d.iterrows():
        text = r[prefer] or r["original" if prefer == "khmer" else "khmer"]
        if not text:
            continue
        n += 1
        parts.append(
            f"{n}\n{fmt_srt_time(r['start'])} --> {fmt_srt_time(r['end'])}\n{text}\n"
        )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------
def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return _sp_run(cmd, capture_output=True)


def probe_duration(path: Path | str) -> float:
    r = _sp_run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def _has_stream(path: Path | str, kind: str) -> bool:
    r = _sp_run(
        ["ffprobe", "-v", "error", "-select_streams", kind,
         "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    return bool(r.stdout.strip())


def has_audio(path: Path | str) -> bool:
    return _has_stream(path, "a")


def has_video(path: Path | str) -> bool:
    # រូបភាពក្របមុខ (cover art) របស់ mp3 ក៏ជា video stream ដែរ ដូច្នេះពិនិត្យថា codec មិនមែន mjpeg/png
    r = _sp_run(
        ["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
         "stream=codec_name", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    codecs = [c.strip() for c in r.stdout.split() if c.strip()]
    return any(c not in ("mjpeg", "png", "bmp", "gif") for c in codecs)


def video_size(path: Path | str) -> str:
    r = _sp_run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=s=x:p=0", str(path)],
        capture_output=True, text=True,
    )
    return r.stdout.strip() or "-"


def extract_audio_chunk(src: Path, dst: Path, start: float, length: float) -> None:
    r = _run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{length:.3f}",
         "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k", str(dst)]
    )
    if r.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        raise RuntimeError("ទាញសំឡេងពីវីដេអូមិនបាន: " + r.stderr.decode(errors="ignore")[-400:])


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------
TRANSCRIBE_PROMPT = """You are a professional subtitle translator and dubbing-script writer.
Listen to this audio. The spoken language is: {lang}.
Transcribe every spoken line, then translate it into natural, spoken Khmer that is suitable
for dubbing a movie or drama: short, conversational, keeping the emotion and meaning
(never translate word-for-word, and use polite pronouns that fit the characters).

Return ONLY a JSON array. Each item must have exactly these keys:
  "start": start time in total SECONDS from the beginning of THIS audio (a plain number such as 83.5, never "MM:SS"),
  "end": end time in total SECONDS (a plain number),
  "speaker": one of "male", "female", "narrator",
  "original": the transcript in the original language,
  "khmer": the Khmer translation.
Keep each line under about 12 seconds. Ignore music and noise that has no speech.
If there is no speech at all, return []."""

TRANSLATE_PROMPT = """Translate each of the following {lang} lines into natural, spoken Khmer
suitable for dubbing a movie or drama (short, conversational, keep the emotion; not word-for-word).
Return ONLY a JSON array of Khmer strings, in exactly the same order and with exactly {n} items.

Lines (JSON):
{lines}"""


def get_client(api_key: str):
    from google import genai  # នាំចូលពេលប្រើ ដើម្បីឲ្យ កម្មវិធីសាកល្បងបានដោយគ្មាន SDK

    return genai.Client(api_key=api_key)


def parse_json_loose(text: str):
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    return json.loads(t)


def _wait_active(client, f, timeout: float = 180.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        f = client.files.get(name=f.name)
        state = getattr(getattr(f, "state", None), "name", None) or str(getattr(f, "state", ""))
        if state in ("", "None", "ACTIVE"):
            return f
        if state == "FAILED":
            raise RuntimeError("Gemini ដំណើរការឯកសារសំឡេងមិនបាន (FAILED)")
        time.sleep(2)
    raise RuntimeError("Gemini ដំណើរការឯកសារយូរពេក (timeout)")


def _to_seconds(v) -> float:
    """ទទួលលេខ ឬ "MM:SS" / "HH:MM:SS" ហើយត្រឡប់វិនាទី"""
    if isinstance(v, (int, float)):
        return float(v)
    t = str(v).strip().replace(",", ".")
    if ":" in t:
        parts = [float(x) for x in t.split(":")]
        sec = 0.0
        for x in parts:
            sec = sec * 60 + x
        return sec
    return float(t)


def _normalize_items(items, offset: float, limit: Optional[float] = None) -> tuple[list[dict], int]:
    """ត្រឡប់ (ជួរ, ចំនួនឃ្លាដែលត្រូវទម្លាក់ព្រោះពេលវេលាលើសប្រវែងផ្នែក)"""
    rows = []
    dropped = 0
    if isinstance(items, dict):
        items = items.get("lines") or items.get("segments") or []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        try:
            s = _to_seconds(it.get("start", 0))
            e = _to_seconds(it.get("end", s + 2))
        except (TypeError, ValueError):
            continue
        if limit is not None and s > limit + 1.0:
            dropped += 1
            continue
        if limit is not None:
            e = min(e, limit)
        s += offset
        e += offset
        spk = str(it.get("speaker", "narrator")).strip().lower()
        rows.append(
            {
                "start": s,
                "end": max(e, s + 0.3),
                "speaker": SPEAKER_LABELS.get(spk, DEFAULT_SPEAKER),
                "original": str(it.get("original", "") or "").strip(),
                "khmer": str(it.get("khmer", "") or "").strip(),
            }
        )
    return rows, dropped


def _is_transient(e: Exception) -> bool:
    t = str(e)
    return any(x in t for x in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500",
                                "INTERNAL", "504", "DEADLINE", "overloaded", "high demand"))


def _is_missing(e: Exception) -> bool:
    t = str(e)
    return "404" in t or "NOT_FOUND" in t


def gen_json(client, models: list[str], contents, note: Optional[Callable[[str], None]] = None):
    """
    ហៅ Gemini ហើយទទួល JSON។
    - 503/429 (ម៉ាស៊ីនបម្រើរវល់): រង់ចាំរួចសាកម្តងទៀត (រហូតដល់ ៥ ដង) រួចប្តូរទៅម៉ូដែលបន្ទាប់
    - 404 (ម៉ូដែលត្រូវបានបិទ): ប្តូរទៅម៉ូដែលបន្ទាប់ភ្លាមៗ
    ត្រឡប់ (ទិន្នន័យ, ម៉ូដែលដែលបានប្រើ)
    """
    from google.genai import types

    last: Exception | None = None
    for m in models:
        for attempt in range(5):
            try:
                resp = client.models.generate_content(
                    model=m,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json", temperature=0.3
                    ),
                )
                return parse_json_loose(resp.text), m
            except Exception as e:
                last = e
                if _is_missing(e):
                    break  # ម៉ូដែលនេះប្រើមិនបាន -> ម៉ូដែលបន្ទាប់
                if _is_transient(e) or isinstance(e, ValueError):
                    wait = min(4 * 2 ** attempt, 30)
                    if note:
                        note(f"Gemini រវល់ ({m}) កំពុងរង់ចាំ {wait} វិនាទី ហើយសាកម្តងទៀត...")
                    time.sleep(wait)
                    continue
                raise
    raise RuntimeError(f"Gemini មិនឆ្លើយត្រឹមត្រូវ (សាកគ្រប់ម៉ូដែលហើយ): {last}")


def _model_chain(model: str, fallbacks: Optional[list[str]]) -> list[str]:
    return [model] + [m for m in (fallbacks or []) if m != model]


def transcribe_translate(
    client,
    model: str,
    media: Path,
    lang_name: str,
    workdir: Path,
    chunk_seconds: int = 120,
    progress: Optional[Callable[[float, str], None]] = None,
    fallbacks: Optional[list[str]] = None,
) -> pd.DataFrame:

    total = probe_duration(media)
    if total <= 0:
        raise RuntimeError("អានរយៈពេលឯកសារមិនបាន")
    n = max(1, math.ceil(total / chunk_seconds))
    rows: list[dict] = []
    for i in range(n):
        off = i * chunk_seconds
        if progress:
            progress(i / n, f"កំពុងចាប់សំឡេង និងបកប្រែ ផ្នែក {i + 1}/{n}")
        chunk = workdir / f"chunk_{i}.mp3"
        extract_audio_chunk(media, chunk, off, chunk_seconds)
        uploaded = client.files.upload(file=str(chunk))
        try:
            uploaded = _wait_active(client, uploaded)
            models = _model_chain(model, fallbacks)
            note = (lambda t: progress(i / n, t)) if progress else None
            items, _used = gen_json(
                client, models, [uploaded, TRANSCRIBE_PROMPT.format(lang=lang_name)], note
            )
            this_len = min(chunk_seconds, max(total - off, 1.0))
            part_rows, dropped = _normalize_items(items, off, limit=this_len)
            rows.extend(part_rows)
            if dropped and progress:
                progress(i / n, f"ផ្នែក {i + 1}: ទម្លាក់ {dropped} ឃ្លាដែលពេលវេលាខុស")
            if i + 1 < n:
                time.sleep(2)  # កុំហៅលឿនពេក ដើម្បីជៀស 429
        finally:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass
    if progress:
        progress(1.0, "ចាប់សំឡេង និងបកប្រែរួចរាល់")
    return clean_df(pd.DataFrame(rows, columns=COLUMNS))


LANG_CODES = {
    "Chinese": "zh", "English": "en", "Korean": "ko", "Japanese": "ja", "Thai": "th",
}


def groq_transcribe(
    api_key: str,
    media: Path,
    lang_name: str,
    workdir: Path,
    model: str = "whisper-large-v3-turbo",
    chunk_seconds: int = 600,
    progress: Optional[Callable[[float, str], None]] = None,
) -> pd.DataFrame:
    """ចាប់សំឡេងដោយ Whisper លើ Groq (ឥតគិតថ្លៃ លឿន) ត្រឡប់តារាងដែលមានតែអត្ថបទដើម"""
    import requests

    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    total = probe_duration(media)
    if total <= 0:
        raise RuntimeError("អានរយៈពេលឯកសារមិនបាន")
    n = max(1, math.ceil(total / chunk_seconds))
    rows: list[dict] = []
    for i in range(n):
        off = i * chunk_seconds
        if progress:
            progress(i / n, f"កំពុងចាប់សំឡេង (Whisper) ផ្នែក {i + 1}/{n}")
        chunk = workdir / f"gchunk_{i}.mp3"
        extract_audio_chunk(media, chunk, off, chunk_seconds)
        data = {
            "model": model,
            "response_format": "verbose_json",
            "timestamp_granularities[]": "segment",
            "temperature": "0",
        }
        code = LANG_CODES.get(lang_name)
        if code:
            data["language"] = code
        js = None
        last = ""
        for attempt in range(5):
            with open(chunk, "rb") as fh:
                r = requests.post(
                    url,
                    headers={"Authorization": f"Bearer {api_key}"},
                    files={"file": (chunk.name, fh, "audio/mpeg")},
                    data=data,
                    timeout=300,
                )
            if r.status_code == 200:
                js = r.json()
                break
            last = f"{r.status_code} {r.text[:300]}"
            if r.status_code in (401, 403):
                raise RuntimeError("Groq API Key មិនត្រឹមត្រូវ: " + last)
            if r.status_code == 429 or r.status_code >= 500:
                try:
                    wait = float(r.headers.get("retry-after", ""))
                except ValueError:
                    wait = min(8 * 2 ** attempt, 60)
                wait = min(max(wait, 3.0), 90.0)
                if progress:
                    progress(i / n, f"Groq រវល់ កំពុងរង់ចាំ {int(wait)} វិនាទី...")
                time.sleep(wait)
                continue
            raise RuntimeError("Groq មានកំហុស: " + last)
        if js is None:
            raise RuntimeError("Groq មិនឆ្លើយតប: " + last)
        for seg in js.get("segments") or []:
            text = str(seg.get("text", "")).strip()
            if not text or float(seg.get("no_speech_prob", 0) or 0) > 0.9:
                continue
            st_ = float(seg.get("start", 0)) + off
            en_ = float(seg.get("end", 0)) + off
            rows.append({
                "start": st_, "end": max(en_, st_ + 0.3), "speaker": DEFAULT_SPEAKER,
                "original": text, "khmer": "",
            })
    if progress:
        progress(1.0, "ចាប់សំឡេងរួចរាល់")
    return clean_df(pd.DataFrame(rows, columns=COLUMNS))


def translate_lines(
    client,
    model: str,
    lines: list[str],
    lang_name: str,
    batch: int = 40,
    progress: Optional[Callable[[float, str], None]] = None,
    fallbacks: Optional[list[str]] = None,
) -> list[str]:

    out: list[str] = []
    for b in range(0, len(lines), batch):
        part = lines[b:b + batch]
        if progress:
            progress(b / max(1, len(lines)), f"កំពុងបកប្រែ {b + 1}-{b + len(part)} / {len(lines)}")
        prompt = TRANSLATE_PROMPT.format(
            lang=lang_name, n=len(part), lines=json.dumps(part, ensure_ascii=False)
        )
        result = None
        last_err: Exception | None = None
        models = _model_chain(model, fallbacks)
        note = (lambda t: progress(b / max(1, len(lines)), t)) if progress else None
        for _try in range(3):  # ចំនួនឃ្លាត្រូវតែត្រូវគ្នា
            data, _used = gen_json(client, models, prompt, note)
            if isinstance(data, list) and len(data) == len(part):
                result = [str(x) for x in data]
                break
            last_err = ValueError(f"ចំនួនឃ្លាមិនត្រូវគ្នា ({len(data) if hasattr(data, '__len__') else '?'} ≠ {len(part)})")
        if result is None:
            raise RuntimeError(f"បកប្រែមិនបាន: {last_err}")
        out.extend(result)
    if progress:
        progress(1.0, "បកប្រែរួចរាល់")
    return out


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------
def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(1) as ex:
        return ex.submit(asyncio.run, coro).result()


async def _edge_save(text: str, voice: str, rate: str, pitch: str, out: str) -> None:
    import edge_tts

    await edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).save(out)


def tts_clip(
    text: str, voice: str, rate_pct: int, pitch_hz: int, cache_dir: Path
) -> Path:
    """បង្កើតឯកសារសំឡេង mp3 សម្រាប់ឃ្លាមួយ (មាន cache)"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{voice}|{rate_pct}|{pitch_hz}|{text}".encode()).hexdigest()
    out = cache_dir / f"{key}.mp3"
    if out.exists() and out.stat().st_size > 0:
        return out
    rate = f"{int(rate_pct):+d}%"
    pitch = f"{int(pitch_hz):+d}Hz"
    edge_err: Exception | None = None
    for attempt in range(2):
        try:
            _run_async(_edge_save(text, voice, rate, pitch, str(out)))
            if out.exists() and out.stat().st_size > 0:
                return out
        except Exception as e:
            edge_err = e
            time.sleep(1.5)
    # ផ្លូវបម្រុង: Google TTS (សំឡេងតែមួយ មិនអាចជ្រើសប្រុស/ស្រី)
    try:
        from gtts import gTTS

        gTTS(text=text, lang="km").save(str(out))
        if out.exists() and out.stat().st_size > 0:
            return out
    except Exception as e:
        raise RuntimeError(
            f"បង្កើតសំឡេងខ្មែរមិនបាន។ Edge TTS: {edge_err} | gTTS: {e}"
        ) from e
    raise RuntimeError(f"បង្កើតសំឡេងខ្មែរមិនបាន: {edge_err}")


def decode_clip(path: Path, tempo: float = 1.0, sr: int = TTS_SR) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-i", str(path)]
    if abs(tempo - 1.0) > 0.01:
        cmd += ["-af", f"atempo={min(max(tempo, 0.5), 2.0):.3f}"]
    cmd += ["-ac", "1", "-ar", str(sr), "-f", "s16le", "-"]
    r = _run(cmd)
    if r.returncode != 0 or not r.stdout:
        raise RuntimeError("ឌីកូដសំឡេងមិនបាន: " + r.stderr.decode(errors="ignore")[-300:])
    return np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def write_wav(path: Path, samples: np.ndarray, sr: int = TTS_SR) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def build_dub_track(
    df: pd.DataFrame,
    out_wav: Path,
    cache_dir: Path,
    voices: dict[str, str],
    rate_pct: int,
    narrator_pitch: int,
    max_speedup: float,
    total_seconds: float,
    clip_fn: Callable[[str, str, int, int, Path], Path] = tts_clip,
    progress: Optional[Callable[[float, str], None]] = None,
) -> int:
    """
    បង្កើតបទសំឡេងខ្មែរមួយ (wav) ដោយដាក់ឃ្លានីមួយៗតាមពេលចាប់ផ្តើមរបស់វា។
    ប្រសិនបើសំឡេងវែងជាងចន្លោះដែលមាន នឹងបង្កើនល្បឿន (atempo) រហូតដល់ max_speedup។
    ត្រឡប់ចំនួនឃ្លាដែលបានដាក់។
    """
    d = clean_df(df)
    d = d[d["khmer"] != ""].reset_index(drop=True)
    if d.empty:
        raise RuntimeError("មិនមានអត្ថបទខ្មែរសម្រាប់បង្កើតសំឡេងទេ")

    last_end = float(d["end"].max())
    track = np.zeros(int((max(total_seconds, last_end) + 5) * TTS_SR), dtype=np.float32)
    max_pos = 0  # ចុងសំឡេងបំផុត (sample) ដើម្បីកាត់ស្ងាត់ដែលលើសចេញ

    for i, r in d.iterrows():
        if progress:
            progress(i / len(d), f"កំពុងបង្កើតសំឡេងខ្មែរ {i + 1}/{len(d)}")
        key = SPEAKER_KEYS.get(r["speaker"], "narrator")
        pitch = narrator_pitch if key == "narrator" else 0
        clip = clip_fn(r["khmer"], voices[key], rate_pct, pitch, cache_dir)
        audio = decode_clip(clip)
        dur = len(audio) / TTS_SR

        start = float(r["start"])
        slot = max(float(r["end"]) - start, 0.3)
        if i + 1 < len(d):
            available = max(slot, float(d.loc[i + 1, "start"]) - start - 0.05)
        else:
            available = slot + 2.0
        if dur > available:
            tempo = min(dur / available, max_speedup)
            if tempo > 1.01:
                audio = decode_clip(clip, tempo=tempo)

        pos = int(start * TTS_SR)
        end_pos = pos + len(audio)
        if end_pos > len(track):
            track = np.concatenate([track, np.zeros(end_pos - len(track) + TTS_SR, np.float32)])
        track[pos:end_pos] += audio
        max_pos = max(max_pos, end_pos)

    keep = max(int(total_seconds * TTS_SR), max_pos)
    track = track[:keep]
    peak = float(np.max(np.abs(track)))
    if peak > 1.0:
        track = track / peak
    write_wav(out_wav, track)
    if progress:
        progress(1.0, "បង្កើតសំឡេងខ្មែររួចរាល់")
    return len(d)


# ---------------------------------------------------------------------------
# Mux: លាយសំឡេងខ្មែរចូលវីដេអូ
# ---------------------------------------------------------------------------
def mux_dub(
    media: Path,
    dub_wav: Path,
    out_path: Path,
    orig_volume: float,
) -> Path:
    """
    media មានរូបភាព  -> out_path (.mp4) រូបភាពដើម + សំឡេងដើម (ស្រាល) + សំឡេងខ្មែរ
    media ជា audio  -> out_path (.m4a)
    """
    keep_orig = orig_volume > 0.0 and has_audio(media)
    video = has_video(media)

    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(media), "-i", str(dub_wav)]
    if keep_orig:
        fc = (
            f"[0:a]volume={orig_volume:.2f}[o];"
            "[1:a]volume=1.0[d];"
            "[o][d]amix=inputs=2:duration=longest:normalize=0[a]"
        )
        cmd += ["-filter_complex", fc, "-map", "[a]"]
    else:
        cmd += ["-map", "1:a"]

    if video:
        base = cmd + ["-map", "0:v:0"]
        attempt1 = base + ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
                           "-movflags", "+faststart", str(out_path)]
        r = _run(attempt1)
        if r.returncode != 0:
            attempt2 = base + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                               "-shortest", "-movflags", "+faststart", str(out_path)]
            r = _run(attempt2)
    else:
        r = _run(cmd + ["-c:a", "aac", "-b:a", "192k", str(out_path)])

    if r.returncode != 0 or not out_path.exists():
        raise RuntimeError("Export មិនបាន: " + r.stderr.decode(errors="ignore")[-500:])
    return out_path




_NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _sp_run(cmd, **kw):
    if _NOWIN:
        kw.setdefault("creationflags", _NOWIN)
    return subprocess.run(cmd, **kw)


# ---------------------------------------------------------------------------
# ពេលវេលា mm:ss.xx
# ---------------------------------------------------------------------------
def fmt_ts(sec: float) -> str:
    sec = max(0.0, float(sec))
    m, s = divmod(sec, 60.0)
    h, m = divmod(int(m), 60)
    return f"{h}:{m:02d}:{s:05.2f}" if h else f"{m:02d}:{s:05.2f}"


def parse_ts(text: str) -> float:
    """"1:23.5" | "01:02:03.25" | "83.5" -> វិនាទី (ValueError បើអានមិនបាន)"""
    t = str(text).strip().replace(",", ".")
    if not t:
        raise ValueError("empty")
    sec = 0.0
    for part in t.split(":"):
        sec = sec * 60.0 + float(part)
    if sec < 0:
        raise ValueError("negative")
    return sec


# ---------------------------------------------------------------------------
# Whisper លើកុំព្យូទ័រ (faster-whisper) - គ្មាន API, គ្មាន 503
# ---------------------------------------------------------------------------
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3"]


def extract_full_wav(media: Path, dst: Path) -> None:
    r = _run(["ffmpeg", "-y", "-v", "error", "-i", str(media), "-vn", "-ac", "1",
              "-ar", "16000", "-f", "wav", str(dst)])
    if r.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        raise RuntimeError("ទាញសំឡេងពីវីដេអូមិនបាន: " + r.stderr.decode(errors="ignore")[-400:])


def whisper_device() -> str:
    try:
        import ctranslate2

        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:
        return "cpu"


def whisper_local(
    media: Path,
    lang_name: str,
    workdir: Path,
    model_size: str = "small",
    progress: Optional[Callable[[float, str], None]] = None,
    model_factory: Optional[Callable] = None,
) -> pd.DataFrame:
    """ចាប់សំឡេងដោយ faster-whisper លើកុំព្យូទ័រផ្ទាល់។ model_factory សម្រាប់សាកល្បង"""
    total = probe_duration(media)
    if total <= 0:
        raise RuntimeError("អានរយៈពេលឯកសារមិនបាន")
    wav = workdir / "whisper_in.wav"
    if progress:
        progress(0.0, "កំពុងរៀបចំសំឡេង...")
    extract_full_wav(media, wav)

    if model_factory is None:
        from faster_whisper import WhisperModel

        dev = whisper_device()

        def model_factory(name):
            return WhisperModel(name, device=dev, compute_type="float16" if dev == "cuda" else "int8")

    if progress:
        progress(0.02, f"កំពុងផ្ទុកម៉ូដែល Whisper «{model_size}» (លើកដំបូងត្រូវទាញយក ត្រូវការអ៊ីនធឺណិត)...")
    model = model_factory(model_size)
    code = LANG_CODES.get(lang_name)
    segments, _info = model.transcribe(
        str(wav), language=code, vad_filter=True, beam_size=5, condition_on_previous_text=False
    )
    rows: list[dict] = []
    for seg in segments:
        text = str(getattr(seg, "text", "")).strip()
        if progress:
            progress(min(0.98, 0.05 + 0.93 * float(seg.end) / total), f"កំពុងចាប់សំឡេង {fmt_clock(seg.end)} / {fmt_clock(total)}")
        if not text or float(getattr(seg, "no_speech_prob", 0.0) or 0.0) > 0.9:
            continue
        rows.append({
            "start": float(seg.start), "end": max(float(seg.end), float(seg.start) + 0.3),
            "speaker": DEFAULT_SPEAKER, "original": text, "khmer": "",
        })
    if progress:
        progress(1.0, "ចាប់សំឡេងរួចរាល់")
    return clean_df(pd.DataFrame(rows, columns=COLUMNS))


# ---------------------------------------------------------------------------
# បកប្រែ: Gemini ជាមុន បើបរាជ័យប្រើ Google Translate ឥតគិតថ្លៃ
# ---------------------------------------------------------------------------
def translate_google_free(
    lines: list[str], lang_name: str,
    progress: Optional[Callable[[float, str], None]] = None, translator=None,
) -> list[str]:
    if translator is None:
        from deep_translator import GoogleTranslator

        src = {"zh": "zh-CN"}.get(LANG_CODES.get(lang_name, ""), LANG_CODES.get(lang_name, "auto"))
        translator = GoogleTranslator(source=src, target="km")
    out: list[str] = []
    step = 15
    for i in range(0, len(lines), step):
        part = lines[i:i + step]
        if progress:
            progress(i / max(1, len(lines)), f"Google Translate {i + 1}-{i + len(part)} / {len(lines)}")
        got = None
        for attempt in range(3):
            try:
                got = translator.translate_batch(part)
                if got and len(got) == len(part):
                    break
            except Exception:
                time.sleep(2 * (attempt + 1))
        if not got or len(got) != len(part):
            got = []
            for x in part:  # បកម្តងមួយឃ្លា ពេល batch បរាជ័យ
                for attempt in range(2):
                    try:
                        got.append(translator.translate(x))
                        break
                    except Exception:
                        time.sleep(1.5)
                else:
                    got.append("")
        out.extend([("" if g is None else str(g)) for g in got])
    if len(out) != len(lines):
        raise RuntimeError("Google Translate ឆ្លើយមិនគ្រប់ឃ្លា")
    return out


def translate_smart(
    client, models: list[str], lines: list[str], lang_name: str,
    use_google_fallback: bool = True,
    progress: Optional[Callable[[float, str], None]] = None,
) -> tuple[list[str], str]:
    """ត្រឡប់ (ឃ្លាបកប្រែ, ឈ្មោះម៉ាស៊ីនដែលបានប្រើ)"""
    err: Exception | None = None
    if client is not None and models:
        try:
            res = translate_lines(client, models[0], lines, lang_name, progress=progress, fallbacks=models[1:])
            return res, "Gemini"
        except Exception as e:  # noqa: BLE001
            err = e
            if not use_google_fallback:
                raise
    elif not use_google_fallback:
        raise RuntimeError("មិនមាន Gemini key ហើយមិនបានអនុញ្ញាត Google Translate")
    if progress:
        progress(0.0, "Gemini មិនដំណើរការ → ប្រើ Google Translate ជំនួស...")
    try:
        return translate_google_free(lines, lang_name, progress=progress), "Google Translate"
    except Exception as e2:  # noqa: BLE001
        raise RuntimeError(f"បកប្រែមិនបាន។ Gemini: {err} | Google: {e2}") from e2


# ---------------------------------------------------------------------------
# Waveform សម្រាប់ timeline
# ---------------------------------------------------------------------------
def waveform_envelope(media: Path, buckets: int = 4000) -> np.ndarray:
    """ត្រឡប់អារេ 0..1 ប្រវែង buckets ជាកម្រិតសំឡេងរបស់បទដើម"""
    r = _run(["ffmpeg", "-v", "error", "-i", str(media), "-vn", "-ac", "1", "-ar", "4000",
              "-f", "s16le", "-"])
    if r.returncode != 0 or not r.stdout:
        return np.zeros(buckets, dtype=np.float32)
    a = np.abs(np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32))
    if a.size == 0:
        return np.zeros(buckets, dtype=np.float32)
    per = max(1, int(math.ceil(a.size / buckets)))
    pad = per * buckets - a.size
    if pad > 0:
        a = np.concatenate([a, np.zeros(pad, dtype=np.float32)])
    env = a[: per * buckets].reshape(buckets, per).max(axis=1)
    peak = float(env.max())
    return (env / peak) if peak > 0 else env


# ---------------------------------------------------------------------------
# Config + Project
# ---------------------------------------------------------------------------
def config_dir() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home() / ".config")
    d = Path(base) / "BunyimStudio"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_config() -> dict:
    p = config_dir() / "config.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_config(cfg: dict) -> None:
    try:
        (config_dir() / "config.json").write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def save_project(path: Path, df: pd.DataFrame, meta: dict) -> None:
    d = clean_df(df)
    payload = {"version": 1, "meta": meta, "lines": d.to_dict(orient="records")}
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def load_project(path: Path) -> tuple[pd.DataFrame, dict]:
    js = json.loads(Path(path).read_text(encoding="utf-8"))
    df = clean_df(pd.DataFrame(js.get("lines", []), columns=COLUMNS))
    return df, js.get("meta", {})


def setup_ffmpeg_path() -> None:
    """ពេលជា .exe: ដាក់ថត bundle (ដែលមាន ffmpeg.exe) ចូល PATH"""
    import sys

    base = getattr(sys, "_MEIPASS", None)
    if base:
        os.environ["PATH"] = str(base) + os.pathsep + os.environ.get("PATH", "")
