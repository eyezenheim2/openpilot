#!/usr/bin/env python3
"""Render the drive-narration voice clips with Piper.

  pip install piper-tts
  curl -LO https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-en-us-libritts-high.tar.gz
  tar xzf voice-en-us-libritts-high.tar.gz
  python3 tools/narration/make_voices.py en-us-libritts-high.onnx

Writes openpilot/selfdrive/assets/sounds/narration/<voice dir>/<clip>.wav (mono, 16-bit, 22.05 kHz) for every
voice in narration_text.VOICES: one clip per phrase and per number. Clips are trimmed to the speech and peak-
normalized so they join cleanly; soundd resamples them to its output rate when it loads them.
"""
import argparse
import importlib.util
import io
import os
import sys
import wave

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(ROOT, "openpilot", "selfdrive", "assets", "sounds", "narration")
SAMPLE_RATE = 22050
PEAK = 0.89  # -1 dBFS
TRIM_LEVEL = 400  # int16 amplitude treated as silence at the ends
LEAD, TAIL = 0.01, 0.02  # s kept before the first and after the last sound


def load_text_module():
  # load narration_text.py directly: importing it as a package would pull in the UI
  path = os.path.join(ROOT, "openpilot", "selfdrive", "ui", "narration_text.py")
  spec = importlib.util.spec_from_file_location("narration_text", path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def finish(x: np.ndarray, rate: int) -> np.ndarray:
  x = x.astype(np.float32)
  if rate != SAMPLE_RATE:
    t = np.arange(len(x)) / rate
    x = np.interp(np.arange(0, t[-1], 1 / SAMPLE_RATE), t, x)
  loud = np.where(np.abs(x) > TRIM_LEVEL)[0]
  if len(loud):
    x = x[max(0, loud[0] - int(LEAD * SAMPLE_RATE)): loud[-1] + int(TAIL * SAMPLE_RATE)]
  return (x / (np.abs(x).max() + 1e-9) * PEAK * 32767).astype(np.int16)


def make_chime() -> np.ndarray:
  """Soft two-note chime (G5 then C6, rising), bell-like decay, ~0.75 s, peak ~40% of the voice clips."""
  t = np.arange(int(0.75 * SAMPLE_RATE)) / SAMPLE_RATE
  out = np.zeros_like(t)
  for start, freq in ((0.0, 783.99), (0.16, 1046.5)):
    u = np.clip(t - start, 0, None)
    on = t >= start
    attack = np.minimum(u / 0.008, 1.0)  # no click
    tone = np.sin(2 * np.pi * freq * u) + 0.25 * np.sin(2 * np.pi * 2 * freq * u) * np.exp(-u * 9)  # touch of overtone
    out += on * attack * np.exp(-u * 5.5) * tone
  out *= np.minimum(1.0, (t[-1] - t) / 0.05)  # fade the tail to silence
  return (out / np.abs(out).max() * 0.36 * 32767).astype(np.int16)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("model", help="Piper LibriTTS model (.onnx, with its .onnx.json next to it)")
  ap.add_argument("--voice", action="append", help="only these voice dirs (default: all)")
  ap.add_argument("--only-missing", action="store_true", help="keep existing clips, render only new phrases")
  args = ap.parse_args()

  from piper import PiperVoice, SynthesisConfig
  text = load_text_module()
  model = PiperVoice.load(args.model)
  texts = text.all_clip_texts()

  os.makedirs(OUT_DIR, exist_ok=True)
  with wave.open(os.path.join(OUT_DIR, text.CHIME_FILE), "wb") as wf:  # deterministic: same file every run
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(make_chime().tobytes())

  for voice_dir, label, speaker in text.VOICES:
    if args.voice and voice_dir not in args.voice:
      continue
    out = os.path.join(OUT_DIR, voice_dir)
    os.makedirs(out, exist_ok=True)
    keep = set()
    for t in texts:
      name = text.clip_name(t) + ".wav"
      keep.add(name)
      if args.only_missing and os.path.isfile(os.path.join(out, name)):
        continue
      buf = io.BytesIO()
      with wave.open(buf, "wb") as wf:
        model.synthesize_wav(t, wf, syn_config=SynthesisConfig(speaker_id=speaker))
      buf.seek(0)
      with wave.open(buf) as w:
        x = finish(np.frombuffer(w.readframes(w.getnframes()), np.int16), w.getframerate())
      with wave.open(os.path.join(out, name), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(x.tobytes())
    for stale in set(os.listdir(out)) - keep:  # clips for phrases that no longer exist
      os.remove(os.path.join(out, stale))
    print(f"{label} (speaker {speaker}): {len(keep)} clips -> {os.path.relpath(out, ROOT)}", file=sys.stderr)


if __name__ == "__main__":
  main()
