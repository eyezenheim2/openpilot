import time

import numpy as np
import pytest

from openpilot.selfdrive.ui.audio_routing import (NarrationPlayer, find_usb_output, narration_gain, plan_outputs,
                                                   usb_card_present)

BUILTIN = {"name": "sdm845-tavil-snd-card: - (hw:0,0)", "max_output_channels": 2}
USB = {"name": "USB2.0 Device: Audio (hw:1,0)", "max_output_channels": 2}
USB_MIC = {"name": "USB Microphone: Audio (hw:2,0)", "max_output_channels": 0}


def test_usb_card_present(tmp_path):
  cards = tmp_path / "cards"
  cards.write_text(" 0 [sdm845tavilsndc]: sdm845-tavil-sn - sdm845-tavil-snd-card\n")
  assert not usb_card_present(cards)
  cards.write_text(cards.read_text() + " 1 [Device         ]: USB-Audio - USB2.0 Device\n")
  assert usb_card_present(cards)
  assert not usb_card_present(tmp_path / "missing")


def test_plan_outputs():
  assert find_usb_output([BUILTIN, USB_MIC, USB]) == 2  # skips input-only devices
  assert plan_outputs([BUILTIN, USB], usb_enabled=True, backup_enabled=True) == (1, True)
  assert plan_outputs([BUILTIN, USB], usb_enabled=True, backup_enabled=False) == (1, False)
  assert plan_outputs([BUILTIN, USB], usb_enabled=False, backup_enabled=True) == (None, False)
  assert plan_outputs([BUILTIN], usb_enabled=True, backup_enabled=True) == (None, False)  # no backup without USB


def request(parts, voice="voice_a", age=0.0, rid=None):
  return {"id": rid or f"{time.monotonic_ns()}", "time": time.monotonic() - age, "voice": voice, "parts": parts}


def test_player_joins_real_clips_at_output_rate():
  p = NarrationPlayer(48000)
  assert p.handle_request(request(["Set to", "65", "Speeding up."]), now=time.monotonic())
  assert p.playing
  total = 0
  while p.playing:
    block = p.read(4096)
    assert block.dtype == np.float32 and np.abs(block).max() <= 1.0
    total += len(block)
  assert 1.0 < total / 48000 < 4.0  # a couple of seconds of speech


def test_player_skips_stale_duplicate_and_bad_requests():
  p, now = NarrationPlayer(48000), time.monotonic()
  assert not p.handle_request(request(["Stop sign."], age=30.0), now=now)  # left over from before a restart
  r = request(["Stop sign."])
  assert p.handle_request(r, now=now)
  p.stop()
  assert not p.handle_request(r, now=now)  # same id never plays twice
  assert not p.handle_request(request(["Stop sign."], voice="../voice_a"), now=now)
  assert not p.handle_request(request(["not a phrase we have"]), now=now)
  assert not p.handle_request(None, now=now) and not p.handle_request({"parts": ["x"]}, now=now)


def test_new_line_replaces_current():
  p, now = NarrationPlayer(48000), time.monotonic()
  p.handle_request(request(["Car ahead slowing.", "I'm slowing down."]), now=now)
  p.read(4096)
  p.handle_request(request(["Car cutting in.", "Making room."]), now=now)
  assert p._pos == 0


@pytest.mark.parametrize("vol,alert,expected", [(80, 0.067, 0.4), (80, 1.0, 0.8), (100, 0.7, 0.7), (0, 1.0, 0.0), (150, 1.0, 1.0)])
def test_narration_gain(vol, alert, expected):
  assert narration_gain(vol, alert) == pytest.approx(expected)


def test_line_carries_its_volume():
  p, now = NarrationPlayer(48000), time.monotonic()
  assert p.handle_request({**request(["Stop sign."]), "volume": 40}, now=now) and p.volume == 40
  assert p.handle_request(request(["Stop sign."]), now=now) and p.volume is None  # falls back to the setting
  assert p.handle_request({**request(["Stop sign."]), "volume": "loud"}, now=now) and p.volume is None


def test_chime_is_shared_and_quieter_than_the_voice():
  p = NarrationPlayer(48000)
  chime = p.build("voice_a", ["[chime]"])
  assert 0.5 < len(chime) / 48000 < 1.0
  assert p.build("voice_b", ["[chime]"]) is not None  # same file for every voice
  voice = p.build("voice_a", ["It's"])
  assert np.abs(chime).max() < 0.5 * np.abs(voice).max()
  line = p.build("voice_a", ["[chime]", "It's", "8", "A.M."])
  assert len(line) > len(chime) + len(voice)
