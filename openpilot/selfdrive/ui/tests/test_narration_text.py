from pathlib import Path

from openpilot.selfdrive.ui.narration_text import (CHIME, CHIME_FILE, GREETINGS, Msg, NOT_SPOKEN, NUMBERS, SPOKEN, TEXT, VOICES, SpeechTrigger,
                                                               all_clip_texts, clip_name, greeting_band, greeting_choices, spoken_parts,
                                                     time_caption, time_parts, weather_parts, StopAndGo)

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


def test_time_parts():
  assert time_parts(15, 22) == [CHIME, "It's", "3", "22", "P.M."]
  assert time_parts(8, 5) == [CHIME, "It's", "8", "oh", "5", "A.M."]
  assert time_parts(8, 0) == [CHIME, "It's", "8", "A.M."]
  assert time_parts(0, 0) == [CHIME, "It's", "12", "A.M."]  # midnight
  assert time_parts(12, 30) == [CHIME, "It's", "12", "30", "P.M."]  # noon is P.M.
  assert time_caption(8, 5) == "8:05 AM" and time_caption(12, 22) == "12:22 PM" and time_caption(0, 45) == "12:45 AM"
  texts = set(all_clip_texts()) | {CHIME}
  for hour in range(24):
    for minute in range(60):
      assert set(time_parts(hour, minute)) <= texts, (hour, minute)  # every part has a clip
  assert (CLIP_ROOT / CHIME_FILE).is_file()


def run_traffic(speeds, dt=0.5):
  """speeds: list of (m/s, seconds). Returns the in-traffic flag at the end."""
  sg, t, out = StopAndGo(), 0.0, False
  for v, dur in speeds:
    for _ in range(round(dur / dt)):
      out = sg.update(v, t)
      t += dt
  return out


def test_stop_and_go_detected():
  crawl = [(6.0, 20), (0.0, 15)]
  assert run_traffic([(25.0, 30)] + [(10.0, 130)] + crawl * 2)  # highway, slowed down, then two stops


def test_single_stop_is_not_traffic():
  assert not run_traffic([(15.0, 60), (0.0, 30)])


def test_recent_highway_speed_is_not_traffic():
  assert not run_traffic([(6.0, 20), (0.0, 15), (25.0, 30), (6.0, 20), (0.0, 15)])


def test_old_stops_expire():
  assert not run_traffic([(6.0, 20), (0.0, 15), (12.0, 400), (0.0, 15)])


def test_creeping_does_not_count_as_new_stops():
  # inching forward at walking pace between stops doesn't add stops
  assert not run_traffic([(10.0, 130), (0.0, 10), (1.0, 5), (0.0, 10), (1.0, 5), (0.0, 10)])


def test_muted_lines_still_captioned():
  for m in (Msg.STEER_LIMIT, Msg.GAS_OVERRIDE):
    assert m in NOT_SPOKEN and m in TEXT
  assert run_trigger([(Msg.STEER_LIMIT, 0, 3.0), (Msg.GAS_OVERRIDE, 0, 3.0)]) == []


def test_weather_parts():
  assert weather_parts(58.4, 45, True, 67.2, 10) == ["Currently", "58", "degrees and foggy.", "High of", "67", "today."]
  assert weather_parts(52, 0, False, 67, 10, band="evening") == ["Currently", "52", "degrees and clear."]  # no high at night
  assert weather_parts(60, 0, True, 72, 10)[2] == "degrees and sunny."
  assert weather_parts(60, 2, True, 60, 70) == ["Currently", "60", "degrees and partly cloudy.", "Rain is likely later."]
  assert weather_parts(55, 63, True, 58, 90) == ["Currently", "55", "degrees and raining.", "High of", "58", "today."]  # already wet
  assert weather_parts(0.3, 0, True) == [] and weather_parts(-5, 71, True) == []  # no clip below 1
  assert weather_parts(60, 999, True) == []  # unknown code


def test_weather_parts_have_clips():
  texts = set(all_clip_texts())
  for code in (0, 1, 2, 3, 45, 48, 51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 71, 73, 75, 77, 80, 81, 82, 85, 86, 95, 96, 99):
    for is_day in (True, False):
      parts = weather_parts(70, code, is_day, 80, 60)
      assert parts and set(parts) <= texts, (code, parts)
