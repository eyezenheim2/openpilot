from pathlib import Path

from openpilot.selfdrive.ui.narration_text import (GREETINGS, Msg, NOT_SPOKEN, NUMBERS, SPOKEN, TEXT, VOICES, SpeechTrigger,
                                                               all_clip_texts, clip_name, greeting_band, greeting_choices, spoken_parts)

CLIP_ROOT = Path(__file__).resolve().parents[2] / "assets" / "sounds" / "narration"


def test_every_message_has_caption_and_speech():
  for m in Msg:
    if m != Msg.NONE:
      assert m in TEXT and m in SPOKEN, m


def test_spoken_parts_fill_numbers():
  assert spoken_parts(Msg.SET_SPEED_UP, 65) == ["Set to", "65", "Speeding up."]
  assert spoken_parts(Msg.SLOWING) == ["Car ahead slowing.", "I'm slowing down."]
  assert spoken_parts(Msg.SPEED_LIMIT, 0) == []  # no clip for 0
  assert spoken_parts(Msg.SPEED_LIMIT, max(NUMBERS) + 1) == []
  for m in NOT_SPOKEN:
    assert spoken_parts(m, 30) == []


def test_clip_names_unique_and_safe():
  names = [clip_name(t) for t in all_clip_texts()]
  assert len(names) == len(set(names))
  assert all(n and n.replace("_", "").isalnum() for n in names)
  assert clip_name("65") == "n065" and clip_name("I'm slowing down.") == "im_slowing_down"


def test_every_voice_has_every_clip():
  needed = {clip_name(t) + ".wav" for t in all_clip_texts()}
  for voice_dir, _, _ in VOICES:
    have = {p.name for p in (CLIP_ROOT / voice_dir).glob("*.wav")}
    assert needed <= have, (voice_dir, sorted(needed - have)[:5])
    assert not have - needed, (voice_dir, sorted(have - needed)[:5])  # no stale clips


def test_greetings():
  assert greeting_band(7) == "morning" and greeting_band(13) == "afternoon"
  assert greeting_band(19) == "evening" and greeting_band(23) == "night" and greeting_band(3) == "night"
  for hour in range(24):
    choices = greeting_choices(hour)
    assert set(GREETINGS["any"]) <= set(choices) and len(choices) > len(GREETINGS["any"])


def run_trigger(frames, dt=0.05):
  trig, t, said = SpeechTrigger(), 0.0, []
  for msg, value, dur in frames:
    for _ in range(round(dur / dt)):
      if parts := trig.update(msg, value, t):
        said.append((round(t, 2), parts))
      t += dt
  return said


def test_speaks_once_per_caption():
  said = run_trigger([(Msg.SLOWING, 0, 3.0)])
  assert [p for _, p in said] == [["Car ahead slowing.", "I'm slowing down."]]


def test_no_quick_repeat_but_repeats_later():
  said = run_trigger([(Msg.SLOWING, 0, 1.0), (Msg.NONE, 0, 2.0), (Msg.SLOWING, 0, 1.0), (Msg.NONE, 0, 10.0), (Msg.SLOWING, 0, 1.0)])
  assert len(said) == 2 and said[1][0] >= 10.0


def test_numbers_wait_to_settle():
  # set speed clicked up 55 -> 60 -> 65 quickly: only the final value is spoken
  said = run_trigger([(Msg.SET_SPEED_UP, 55, 0.3), (Msg.SET_SPEED_UP, 60, 0.3), (Msg.SET_SPEED_UP, 65, 2.0)])
  assert [p for _, p in said] == [["Set to", "65", "Speeding up."]]


def test_new_caption_interrupts():
  said = run_trigger([(Msg.SLOWING, 0, 1.0), (Msg.CUT_IN, 0, 1.0)])
  assert [p[0] for _, p in said] == ["Car ahead slowing.", "Car cutting in."]


def test_planted_shown_not_spoken():
  assert run_trigger([(Msg.PLANTED, 0, 3.0)]) == []
