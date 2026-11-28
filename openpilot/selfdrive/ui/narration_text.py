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
  CUT_IN = 21
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
  Msg.CUT_IN: ("Car cutting in", "Making room"),
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
  Msg.CUT_IN: ["Car cutting in.", "Making room."],
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


# names said before the greeting ("Eyezenheim, good morning..."). Clips are pre-rendered, so a name is only spoken
# when it's listed here as (name, how Piper should say it); any other typed name is shown, not spoken.
# The list is also the sunnylink "Your Name" dropdown (DriveNarrationNameId = position + 1, 0 = typed on the device),
# so only ever add rows at the end. To add one: add a row, add the option in settings_ui_src/pages/developer.yaml,
# and run tools/narration/make_voices.py --only-missing.
NAMES = [
  ("Eyezenheim", "Eyezenheim,"),
]


def name_clip(name: str) -> str | None:
  key = name.strip().lower()
  return next((spoken for display, spoken in NAMES if display.lower() == key), None)


def resolve_name(name_id: int, typed: str) -> str:
  """Name to greet: the sunnylink pick when one is chosen, else what was typed on the device."""
  if 1 <= name_id <= len(NAMES):
    return NAMES[name_id - 1][0]
  return typed.strip()


def greeting_caption(greeting: str, name: str) -> tuple[str, str]:
  """On-screen line for a named greeting: ("Good morning", "Eyezenheim")."""
  first = re.split(r"(?<=[.?!])\s", greeting.strip(), maxsplit=1)[0].rstrip(".!")
  return first, name.strip()


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


# weather, said right after the greeting: "Currently" + temp + condition (+ rain warning)
WEATHER_PREFIX = "Currently"
WEATHER_RAIN_LATER = "Rain is likely later."
WEATHER_RAIN_CHANCE = 50  # % chance today that earns the rain warning
# condition phrase by WMO weather code (Open-Meteo's weather_code); (day, night)
WEATHER_CONDITIONS = [
  ((0,), ("degrees and sunny.", "degrees and clear.")),
  ((1, 2), ("degrees and partly cloudy.", "degrees and partly cloudy.")),
  ((3,), ("degrees and cloudy.", "degrees and cloudy.")),
  ((45, 48), ("degrees and foggy.", "degrees and foggy.")),
  ((51, 53, 55, 56, 57), ("degrees and drizzling.", "degrees and drizzling.")),
  ((61, 63, 65, 66, 67, 80, 81, 82), ("degrees and raining.", "degrees and raining.")),
  ((71, 73, 75, 77, 85, 86), ("degrees and snowing.", "degrees and snowing.")),
  ((95, 96, 99), ("degrees with thunderstorms.", "degrees with thunderstorms.")),
]
WET_CODES = {c for codes, (day, _) in WEATHER_CONDITIONS for c in codes if c >= 51}


def weather_condition(code: int, is_day: bool) -> str | None:
  for codes, (day, night) in WEATHER_CONDITIONS:
    if code in codes:
      return day if is_day else night
  return None


def weather_parts(temp: float, code: int, is_day: bool, rain_chance: int | None = None) -> list[str]:
  """Clip texts for the weather line: (58, 45, True, 10) -> ["Currently", "58", "degrees and foggy."].
  The rain warning is only said when it isn't wet already.
  Empty when the temperature has no clip (below 1 degree) or the code is unknown."""
  t, condition = round(temp), weather_condition(code, is_day)
  if t not in NUMBERS or condition is None:
    return []
  parts = [WEATHER_PREFIX, str(t), condition]
  if rain_chance is not None and rain_chance >= WEATHER_RAIN_CHANCE and code not in WET_CODES:
    parts.append(WEATHER_RAIN_LATER)
  return parts


# quarter-hour time announcement: "It's" + hour (+ minutes) + "A.M."/"P.M." ("A.M." is what Piper reads as "ay em")
TIME_PREFIX = "It's"
TIME_AM, TIME_PM = "A.M.", "P.M."
CHIME = "[chime]"  # soft two-note chime, shared by all voices (narration/chime.wav, made by tools/narration/make_voices.py)
CHIME_FILE = "chime.wav"

# chestnut (big model GPU) status, said once per drive: when it starts loading, then when the big model is running
CHESTNUT_CONNECTING = "Chestnut connecting."
CHESTNUT_CONNECTED = "Chestnut connected."
PAUSE_AFTER_CHIME = 0.15  # s between the chime and the voice


TIME_OH = "oh"  # "eight oh five"


def time_parts(hour: int, minute: int) -> list[str]:
  """Chime, then the time on a 12-hour clock: (15, 22) -> [CHIME, "It's", "3", "22", "P.M."],
  (8, 5) -> [CHIME, "It's", "8", "oh", "5", "A.M."], (8, 0) -> [CHIME, "It's", "8", "A.M."]."""
  h12 = hour % 12 or 12
  mins = [] if minute == 0 else [TIME_OH, str(minute)] if minute < 10 else [str(minute)]
  return [CHIME, TIME_PREFIX, str(h12)] + mins + [TIME_AM if hour < 12 else TIME_PM]


def time_caption(hour: int, minute: int) -> str:
  return f"{hour % 12 or 12}:{minute:02d} {'AM' if hour < 12 else 'PM'}"


# live road info (narration_live.py): fetched while driving, spoken from these clips plus the number clips
LIVE_RAIN = "Rain starting in about"  # + n + LIVE_MINUTES
LIVE_MINUTES = "minutes."
LIVE_QUAKE = "Earthquake nearby."  # + "Magnitude" + 4 (+ "point" + 2) + distance "away"
LIVE_MAGNITUDE = "Magnitude"
LIVE_POINT = "point"
LIVE_TRAFFIC = {  # + distance "ahead"
  "crash": "Crash reported ahead.",
  "stalled": "Stalled vehicle ahead.",
  "hazard": "Road hazard ahead.",
  "closed": "Road closed ahead.",
  "lane": "Lane closure ahead.",
  "incident": "Incident reported ahead.",
}
LIVE_TOLL = "Express lane toll is"  # + dollars (+ cents)
_LIVE_UNITS = {(u, s, plural): f"{u}{'s' if plural else ''} {s}." for u in ("mile", "kilometer") for s in ("ahead", "away")
               for plural in (False, True)}
_LIVE_LESS = {(u, s): f"Less than a {u} {s}." for u in ("mile", "kilometer") for s in ("ahead", "away")}
_LIVE_MONEY = ("dollar", "dollars", "cents.")


def live_unit(unit: str, n: int, suffix: str) -> str:
  """("mile", 3, "ahead") -> "miles ahead.", ("kilometer", 1, "away") -> "kilometer away." """
  return _LIVE_UNITS[(unit, suffix, n != 1)]


def live_less_than(unit: str, suffix: str) -> str:
  return _LIVE_LESS[(unit, suffix)]


def live_toll_parts(price: float) -> list[str]:
  """4.5 -> ["Express lane toll is", "4", "dollars", "50"], 0.75 -> [..., "75", "cents."], 1.0 -> [..., "1", "dollar"]"""
  cents_total = int(round(price * 100))
  dollars, cents = divmod(cents_total, 100)
  dollars = min(dollars, max(NUMBERS))
  if dollars == 0:
    return [LIVE_TOLL, str(cents), "cents."]
  return [LIVE_TOLL, str(dollars), "dollar" if dollars == 1 else "dollars"] + ([str(cents)] if cents else [])


def live_clip_texts() -> set[str]:
  return ({LIVE_RAIN, LIVE_MINUTES, LIVE_QUAKE, LIVE_MAGNITUDE, LIVE_POINT, LIVE_TOLL} | set(LIVE_TRAFFIC.values()) |
          set(_LIVE_UNITS.values()) | set(_LIVE_LESS.values()) | set(_LIVE_MONEY))


# shown but not spoken: PLANTED fires often at low speed and is a tuning aid rather than news; GOING by request
NOT_SPOKEN = {Msg.PLANTED, Msg.GOING}
# shown but not spoken in stop-and-go traffic (see StopAndGo), where they'd repeat every few car lengths
QUIET_IN_TRAFFIC = {Msg.SLOWING, Msg.STOPPING}

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
  if part == CHIME:
    return PAUSE_AFTER_CHIME
  return PAUSE_AFTER_SENTENCE if part[-1] in ".," else PAUSE_IN_SENTENCE


def all_clip_texts() -> list[str]:
  """Every clip a voice needs (the shared chime is separate)."""
  phrases = {p for parts in SPOKEN.values() for p in parts if p != "{n}"}
  phrases |= {g for lines in GREETINGS.values() for g in lines}
  phrases |= {TIME_PREFIX, TIME_AM, TIME_PM, TIME_OH}
  phrases |= {WEATHER_PREFIX, WEATHER_RAIN_LATER}
  phrases |= {spoken for _, spoken in NAMES}
  phrases |= {c for _, conds in WEATHER_CONDITIONS for c in conds}
  phrases |= live_clip_texts()
  phrases |= {CHESTNUT_CONNECTING, CHESTNUT_CONNECTED}
  return sorted(phrases) + [str(n) for n in NUMBERS]


class SpeechTrigger:
  """Decides when a caption gets spoken: once when it appears, numbers only after they settle, no quick repeats."""
  NUMBER_SETTLE = 0.8  # s a number has to hold (set speed while the button is still being pressed)
  REPEAT_GAP = 10.0  # s before the same line can be spoken again

  def __init__(self):
    self.pending: tuple[Msg, int, float] | None = None
    self.pending_spoken = False
    self.last_spoken: dict[Msg, tuple[float, int]] = {}

  def update(self, msg: Msg, value: int, now: float, in_traffic: bool = False) -> list[str] | None:
    """Call every frame with the caption on screen (Msg.NONE when none). Returns clip texts to speak, or None.
    In stop-and-go traffic the QUIET_IN_TRAFFIC captions are skipped, not held back to be said later."""
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
    if in_traffic and msg in QUIET_IN_TRAFFIC:
      return None
    last = self.last_spoken.get(msg)
    if last is not None and now - last[0] < self.REPEAT_GAP and last[1] == value:
      return None
    parts = spoken_parts(msg, value)
    if not parts:
      return None
    self.last_spoken[msg] = (now, value)
    return parts


class StopAndGo:
  """Stop-and-go traffic: at least STOPS_NEEDED stops in the last STOP_WINDOW seconds and nothing faster than
  FAST_SPEED for the last FAST_WINDOW seconds. A single stop at a light or highway driving doesn't count."""
  STOP_SPEED = 0.5  # m/s: stopped
  MOVING_SPEED = 2.0  # m/s: has to get going again before the next stop counts
  STOPS_NEEDED = 2
  STOP_WINDOW = 300.0  # s
  FAST_SPEED = 20.0  # m/s (~45 mph)
  FAST_WINDOW = 120.0  # s

  def __init__(self):
    self.stops: list[float] = []
    self.moving = True
    self.last_fast = -1e9

  def update(self, v_ego: float, now: float) -> bool:
    if v_ego > self.FAST_SPEED:
      self.last_fast = now
    if v_ego > self.MOVING_SPEED:
      self.moving = True
    elif v_ego < self.STOP_SPEED and self.moving:
      self.moving = False
      self.stops.append(now)
    self.stops = [t for t in self.stops if now - t <= self.STOP_WINDOW]
    return len(self.stops) >= self.STOPS_NEEDED and now - self.last_fast > self.FAST_WINDOW
