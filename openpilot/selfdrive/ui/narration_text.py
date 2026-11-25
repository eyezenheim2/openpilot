"""Drive narration wording: on-screen captions and spoken lines.

Plain Python with no UI or cereal imports, so the caption renderer, soundd and the clip generator
(tools/narration/make_voices.py) all read the same table.

Spoken lines are built from short clips: one per phrase below, plus one per number for lines that
include a speed. Clip files live in selfdrive/assets/sounds/narration/<voice dir>/<clip name>.wav.
"""
import re
from enum import IntEnum


class Msg(IntEnum):
  # higher value wins when several apply at once
  NONE = 0
  PLANTED = 1
  LANES_FADED = 2
  WAITING = 3
  PULLING_AWAY = 4
  GOING = 5
  SIGN_GO = 6
  GREEN_LIGHT = 7
  ROAD_CLEAR = 8
  SET_SPEED_UP = 9
  SET_SPEED_DOWN = 10
  MORE_ROOM = 11
  LESS_ROOM = 12
  SPEED_LIMIT = 13
  CURVE = 14
  SLOWING = 15
  STOP_AHEAD = 16
  STOP_SIGN = 17
  STOPPING = 18
  MERGING = 19
  STEER_LIMIT = 20
  CUT_IN = 21
  GAS_OVERRIDE = 22
  EYES = 23


# on-screen caption: (first part, second part); {n} is a speed in the display units
TEXT = {
  Msg.PLANTED: ("Planted steering", "Holding steady"),
  Msg.LANES_FADED: ("Lane lines faded", "Following the road"),
  Msg.WAITING: ("Waiting for", "car ahead"),
  Msg.PULLING_AWAY: ("Car ahead pulling away", "Keeping up"),
  Msg.GOING: ("Car ahead moving", "Going"),
  Msg.SIGN_GO: ("Full stop made", "Going"),
  Msg.GREEN_LIGHT: ("Light's green", "Going"),
  Msg.ROAD_CLEAR: ("Road's clear", "Back to {n}"),
  Msg.SET_SPEED_UP: ("Set to {n}", "Speeding up"),
  Msg.SET_SPEED_DOWN: ("Set to {n}", "Slowing down"),
  Msg.MORE_ROOM: ("Farther follow", "Giving more room"),
  Msg.LESS_ROOM: ("Closer follow", "Closing the gap"),
  Msg.SPEED_LIMIT: ("Speed limit {n}", "Slowing to match"),
  Msg.CURVE: ("Curve ahead", "Easing off"),
  Msg.SLOWING: ("Car ahead slowing", "I'm slowing down"),
  Msg.STOP_AHEAD: ("Stopping ahead", "Slowing down"),
  Msg.STOP_SIGN: ("Stop sign", "Stopping"),
  Msg.STOPPING: ("Car ahead stopped", "I'm stopping"),
  Msg.MERGING: ("Car merging in", "Watching it"),
  Msg.STEER_LIMIT: ("Near steering limit", "Be ready"),
  Msg.CUT_IN: ("Car cutting in", "Making room"),
  Msg.GAS_OVERRIDE: ("You're on the gas", "I'll take back over"),
  Msg.EYES: ("Eyes on the road", "please"),
}

# spoken line as clips; "{n}" is the number clip
SPOKEN = {
  Msg.PLANTED: ["Planted steering.", "Holding steady."],
  Msg.LANES_FADED: ["Lane lines faded.", "Following the road."],
  Msg.WAITING: ["Waiting for the car ahead."],
  Msg.PULLING_AWAY: ["Car ahead pulling away.", "Keeping up."],
  Msg.GOING: ["Car ahead moving.", "Going."],
  Msg.SIGN_GO: ["Full stop made.", "Going."],
  Msg.GREEN_LIGHT: ["Light's green.", "Going."],
  Msg.ROAD_CLEAR: ["Road's clear.", "Back to", "{n}"],
  Msg.SET_SPEED_UP: ["Set to", "{n}", "Speeding up."],
  Msg.SET_SPEED_DOWN: ["Set to", "{n}", "Slowing down."],
  Msg.MORE_ROOM: ["Farther follow.", "Giving more room."],
  Msg.LESS_ROOM: ["Closer follow.", "Closing the gap."],
  Msg.SPEED_LIMIT: ["Speed limit", "{n}", "Slowing to match."],
  Msg.CURVE: ["Curve ahead.", "Easing off."],
  Msg.SLOWING: ["Car ahead slowing.", "I'm slowing down."],
  Msg.STOP_AHEAD: ["Stopping ahead.", "Slowing down."],
  Msg.STOP_SIGN: ["Stop sign.", "Stopping."],
  Msg.STOPPING: ["Car ahead stopped.", "I'm stopping."],
  Msg.MERGING: ["Car merging in.", "Watching it."],
  Msg.STEER_LIMIT: ["Near steering limit.", "Be ready."],
  Msg.CUT_IN: ["Car cutting in.", "Making room."],
  Msg.GAS_OVERRIDE: ["You're on the gas.", "I'll take back over."],
  Msg.EYES: ["Eyes on the road, please."],
}

# spoken once each time openpilot goes onroad and has loaded; one picked at random from the time-of-day band + ANY.
# Original lines in a dry, polite AI-butler style.
GREETINGS = {
  "morning": [  # 05:00-11:59
    "Good morning. All systems are online.",
    "Good morning. Ready when you are.",
    "Morning. Everything checks out. Let's go.",
    "Good morning. The road is waiting.",
  ],
  "afternoon": [  # 12:00-16:59
    "Good afternoon. All systems are online.",
    "Good afternoon. Ready when you are.",
    "Afternoon. Systems checked and standing by.",
  ],
  "evening": [  # 17:00-21:59
    "Good evening. All systems are online.",
    "Good evening. Let's get you home.",
    "Evening. Everything is ready.",
  ],
  "night": [  # 22:00-04:59
    "Working late, I see. All systems are online.",
    "Late drive tonight. I'll keep watch.",
    "Still up? Systems are ready.",
  ],
  "any": [
    "Welcome back. Systems online and ready.",
    "Hello again. Everything checks out.",
    "All systems nominal. Whenever you're ready.",
    "Welcome back. I'll keep an eye on the road.",
  ],
}


def greeting_band(hour: int) -> str:
  if 5 <= hour < 12:
    return "morning"
  if 12 <= hour < 17:
    return "afternoon"
  if 17 <= hour < 22:
    return "evening"
  return "night"


def greeting_choices(hour: int) -> list[str]:
  return GREETINGS[greeting_band(hour)] + GREETINGS["any"]


# quarter-hour time announcement: "It's" + hour + minutes ("It's" + "3" + "15"), or "It's" + "12" + "o'clock."
TIME_PREFIX = "It's"
TIME_OCLOCK = "o'clock."


def time_parts(hour: int, minute: int) -> list[str]:
  """Spoken time on a 12-hour clock: (15, 30) -> ["It's", "3", "30"], (0, 0) -> ["It's", "12", "o'clock."]."""
  h12 = hour % 12 or 12
  return [TIME_PREFIX, str(h12), TIME_OCLOCK if minute == 0 else str(minute)]


# shown but not spoken: fires often at low speed and is a tuning aid rather than news
NOT_SPOKEN = {Msg.PLANTED}

NUMBERS = range(1, 161)  # speeds that can be spoken (mph or km/h)

# voices, by the DriveNarrationVoiceId param; all from Piper's LibriTTS voice (CC BY 4.0), see tools/narration/README.md
VOICES = [
  # (directory, label, LibriTTS speaker)
  ("voice_a", "Voice A", 832),
  ("voice_b", "Voice B", 348),
  ("voice_c", "Voice C", 444),
  ("voice_d", "Voice D", 656),
]

PAUSE_AFTER_SENTENCE = 0.18  # s of silence after a part ending in . or ,
PAUSE_IN_SENTENCE = 0.06  # s between parts of one sentence ("Set to" + "65")


def clip_name(text: str) -> str:
  """File stem for a phrase or number clip: "Car ahead slowing." -> car_ahead_slowing, "65" -> n065."""
  if text.isdigit():
    return f"n{int(text):03d}"
  return re.sub(r"[^a-z0-9]+", "_", text.lower().replace("'", "")).strip("_")


def spoken_parts(msg: Msg, value: int = 0) -> list[str]:
  """Clip texts for a message, with the number filled in. Empty when the line can't be spoken."""
  if msg in NOT_SPOKEN or msg not in SPOKEN:
    return []
  parts = []
  for part in SPOKEN[msg]:
    if part == "{n}":
      if value not in NUMBERS:
        return []
      parts.append(str(value))
    else:
      parts.append(part)
  return parts


def pause_after(part: str) -> float:
  return PAUSE_AFTER_SENTENCE if part[-1] in ".," else PAUSE_IN_SENTENCE


def all_clip_texts() -> list[str]:
  """Every clip a voice needs."""
  phrases = {p for parts in SPOKEN.values() for p in parts if p != "{n}"}
  phrases |= {g for lines in GREETINGS.values() for g in lines}
  phrases |= {TIME_PREFIX, TIME_OCLOCK}
  return sorted(phrases) + [str(n) for n in NUMBERS]


class SpeechTrigger:
  """Decides when a caption gets spoken: once when it appears, numbers only after they settle, no quick repeats."""
  NUMBER_SETTLE = 0.8  # s a number has to hold (set speed while the button is still being pressed)
  REPEAT_GAP = 10.0  # s before the same line can be spoken again

  def __init__(self):
    self.pending: tuple[Msg, int, float] | None = None
    self.pending_spoken = False
    self.last_spoken: dict[Msg, tuple[float, int]] = {}

  def update(self, msg: Msg, value: int, now: float) -> list[str] | None:
    """Call every frame with the caption on screen (Msg.NONE when none). Returns clip texts to speak, or None."""
    if msg == Msg.NONE:
      self.pending = None
      return None
    if self.pending is None or self.pending[:2] != (msg, value):
      self.pending, self.pending_spoken = (msg, value, now), False
    if self.pending_spoken:
      return None
    settle = self.NUMBER_SETTLE if "{n}" in SPOKEN.get(msg, []) else 0.0
    if now - self.pending[2] < settle:
      return None
    self.pending_spoken = True
    last = self.last_spoken.get(msg)
    if last is not None and now - last[0] < self.REPEAT_GAP and last[1] == value:
      return None
    parts = spoken_parts(msg, value)
    if not parts:
      return None
    self.last_spoken[msg] = (now, value)
    return parts
