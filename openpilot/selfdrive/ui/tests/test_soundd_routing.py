import time

import numpy as np
import pytest

from openpilot.selfdrive.ui.audio_routing import NarrationPlayer
from openpilot.selfdrive.ui.soundd import AudibleAlert, SAMPLE_RATE, Soundd

FRAMES = 4096


def make_soundd(alert=AudibleAlert.none):
  """A Soundd with synthetic tones instead of the LFS sound files and no audio hardware."""
  s = object.__new__(Soundd)
  s.enabled = False  # quiet mode off
  s._frame = 0
  s.loaded_sounds = {a: np.full(1000, 0.5, np.float32) for a in
                     (AudibleAlert.engage, AudibleAlert.prompt, AudibleAlert.promptRepeat, AudibleAlert.promptDistracted,
                      AudibleAlert.warningSoft, AudibleAlert.warningImmediate)}
  s.current_alert, s.current_sound_frame, s.pending_stop = alert, 0, False
  s.current_volume = 1.0
  s.narration = NarrationPlayer(SAMPLE_RATE)
  s.narration_volume = 100
  s.backup_alert, s.backup_sound_frame = AudibleAlert.none, 0
  return s


def run(callback, frames=FRAMES):
  out = np.zeros((frames, 1), np.float32)
  callback(out, frames, None, None)
  return out[:, 0]


def say(s, parts):
  return s.narration.handle_request({"id": time.monotonic_ns(), "time": time.monotonic(), "voice": "voice_a", "parts": parts},
                                    now=time.monotonic())


def test_narration_plays_when_no_alert():
  s = make_soundd()
  assert say(s, ["Car ahead slowing.", "I'm slowing down."])
  assert np.abs(run(s.callback)).max() > 0.05


def test_alert_stops_narration():
  s = make_soundd()
  say(s, ["Car ahead slowing.", "I'm slowing down."])
  run(s.callback)
  s.current_alert = AudibleAlert.warningSoft
  out = run(s.callback)
  assert not s.narration.playing
  assert np.allclose(out, 0.5)  # just the alert tone, nothing mixed in


def test_backup_plays_only_urgent_alarms():
  for alert in (AudibleAlert.warningSoft, AudibleAlert.warningImmediate, AudibleAlert.promptDistracted, AudibleAlert.promptRepeat):
    s = make_soundd(alert)
    assert np.allclose(run(s.backup_callback), 0.5), alert
  for alert in (AudibleAlert.none, AudibleAlert.engage, AudibleAlert.prompt):
    s = make_soundd(alert)
    assert np.allclose(run(s.backup_callback), 0.0), alert


def test_backup_ignores_narration_and_loops_alarm():
  s = make_soundd(AudibleAlert.warningImmediate)
  say(s, ["Stop sign.", "Stopping."])
  for _ in range(5):  # several blocks past the 1000-sample tone: keeps looping
    assert np.allclose(run(s.backup_callback), 0.5)


def test_backup_follows_volume():
  s = make_soundd(AudibleAlert.warningSoft)
  s.current_volume = 0.2
  assert np.allclose(run(s.backup_callback), 0.1)


def test_line_volume_overrides_setting():
  quiet, loud = make_soundd(), make_soundd()
  for s, vol in ((quiet, 20), (loud, 100)):
    s.narration_volume = 100
    s.narration.handle_request({"id": time.monotonic_ns(), "time": time.monotonic(), "voice": "voice_a", "volume": vol,
                                "parts": ["Car ahead slowing."]}, now=time.monotonic())
  assert np.abs(run(loud.callback)).max() == pytest.approx(5 * np.abs(run(quiet.callback)).max(), rel=1e-3)
