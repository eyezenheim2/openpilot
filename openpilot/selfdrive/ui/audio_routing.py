"""Sound output routing for soundd: USB speaker, alarm backup, and drive-narration playback.

- With UsbAudioOutput on and a USB audio device present, every sound plays through the USB device.
- With UsbAudioAlarmBackup on, the urgent alarms (the ones Quiet Mode never silences) also play on the
  built-in speaker, so a muted or failed USB speaker can't hide a take-control alarm.
- Drive narration (DriveNarrationSay, written by the UI) is mixed into the main output from pre-rendered
  clips (selfdrive/assets/sounds/narration/<voice>/), and never over an alert sound.

Kept free of sounddevice and messaging so it can be tested without audio hardware.
"""
import os
import wave
from pathlib import Path

import numpy as np

from openpilot.common.basedir import BASEDIR
from openpilot.selfdrive.ui.narration_text import clip_name, pause_after

ASOUND_CARDS = Path("/proc/asound/cards")
NARRATION_DIR = Path(BASEDIR) / "openpilot" / "selfdrive" / "assets" / "sounds" / "narration"
USB_HINT = "usb"
NARRATION_MIN_GAIN = 0.5  # narration never drops below this share of its set volume in a quiet cabin


def usb_card_present(cards_path: Path = ASOUND_CARDS) -> bool:
  """A USB audio card is registered with ALSA (cheap to poll: one small procfs read)."""
  try:
    return USB_HINT in cards_path.read_text().lower()
  except OSError:
    return False


def find_usb_output(devices) -> int | None:
  """Index of the first USB output device in sounddevice.query_devices(), or None."""
  for i, dev in enumerate(devices):
    if dev.get("max_output_channels", 0) > 0 and USB_HINT in str(dev.get("name", "")).lower():
      return i
  return None


def plan_outputs(devices, usb_enabled: bool, backup_enabled: bool) -> tuple[int | None, bool]:
  """(device for the main stream, whether to also open the built-in backup stream).
  None means the system default device, the built-in speaker."""
  usb = find_usb_output(devices) if usb_enabled else None
  return usb, usb is not None and backup_enabled


def _read_clip(path: Path) -> tuple[np.ndarray, int]:
  with wave.open(str(path), "rb") as w:
    if w.getnchannels() != 1 or w.getsampwidth() != 2:
      raise ValueError(f"{path}: expected mono 16-bit")
    return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0, w.getframerate()


def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
  if src == dst or len(x) < 2:
    return x
  n = int(round(len(x) * dst / src))
  return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


class NarrationPlayer:
  """Joins clips into one buffer and hands it out block by block from soundd's audio callback."""

  def __init__(self, sample_rate: int, clip_dir: Path = NARRATION_DIR):
    self.sample_rate = sample_rate
    self.clip_dir = clip_dir
    self._cache: dict[tuple[str, str], np.ndarray] = {}
    self._buf = np.zeros(0, np.float32)
    self._pos = 0
    self.last_id = None
    self.volume: int | None = None  # % sent with the current line, overrides the DriveNarrationVolume setting

  def _clip(self, voice: str, text: str) -> np.ndarray:
    key = (voice, text)
    if key not in self._cache:
      x, rate = _read_clip(self.clip_dir / voice / f"{clip_name(text)}.wav")
      self._cache[key] = _resample(x, rate, self.sample_rate)
    return self._cache[key]

  def build(self, voice: str, parts: list[str]) -> np.ndarray:
    out = []
    for i, part in enumerate(parts):
      out.append(self._clip(voice, part))
      if i < len(parts) - 1:
        out.append(np.zeros(int(pause_after(part) * self.sample_rate), np.float32))
    return np.concatenate(out) if out else np.zeros(0, np.float32)

  def handle_request(self, request, now: float | None = None, max_age: float = 5.0) -> bool:
    """Start a DriveNarrationSay request ({"id", "time", "voice", "parts", optional "volume"}) once. Returns True if it started playing.
    Requests older than max_age seconds (time.monotonic, shared by all processes) are skipped, so a restart never replays an old line."""
    if not isinstance(request, dict) or request.get("id") is None or request.get("id") == self.last_id:
      return False
    self.last_id = request["id"]
    if now is not None and not (now - max_age <= float(request.get("time", 0)) <= now + max_age):
      return False
    voice, parts = str(request.get("voice", "")), request.get("parts") or []
    if not voice or os.sep in voice or not isinstance(parts, list):
      return False
    try:
      buf = self.build(voice, [str(p) for p in parts])
    except (OSError, ValueError, EOFError):
      return False  # missing or bad clip: skip this line rather than play half of it
    vol = request.get("volume")
    self.volume = int(vol) if isinstance(vol, (int, float)) and not isinstance(vol, bool) else None
    self._buf, self._pos = buf, 0  # a new line replaces whatever was still playing
    return len(buf) > 0

  @property
  def playing(self) -> bool:
    return self._pos < len(self._buf)

  def stop(self) -> None:
    self._pos = len(self._buf)

  def read(self, frames: int) -> np.ndarray:
    out = np.zeros(frames, np.float32)
    n = min(frames, len(self._buf) - self._pos)
    if n > 0:
      out[:n] = self._buf[self._pos:self._pos + n]
      self._pos += n
    return out


def narration_gain(volume_pct: int, alert_volume: float) -> float:
  """Narration level: the user's volume setting, following cabin noise like the alerts but never too quiet to hear."""
  return max(0.0, min(1.0, volume_pct / 100.0)) * max(NARRATION_MIN_GAIN, min(1.0, alert_volume))
