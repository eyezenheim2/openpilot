"""Live road info for the drive narration, fetched while driving (over a phone hotspot or other connection).

  rain starting soon    Open-Meteo 15-minutely forecast (free, no key), every RAIN_EVERY s
  earthquakes nearby    USGS past-hour M2.5+ feed (free, no key), every QUAKE_EVERY s
  incidents ahead       511 SF Bay traffic events (free token in the Narration511Token param), every TRAFFIC_EVERY s
  express-lane tolls    511 SF Bay toll programs (same token), every TOLL_EVERY s

511 allows 60 requests an hour per token by default; traffic + tolls here use about 36.
Everything is best-effort: no connection, a timeout or an odd reply just means nothing is said. Each incident, quake,
toll price and rain spell is announced once. The narrator picks announcements up with pop() when it's quiet; ones that
wait longer than MAX_WAIT (behind alerts, or the voice muted by a tap) are dropped, since they'd be stale.

The decision functions are plain Python on plain dicts, so they're tested off-device with recorded replies.
"""
import json
import math
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from openpilot.selfdrive.ui import narration_text as nt

RAIN_URL = "https://api.open-meteo.com/v1/forecast"
QUAKE_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_hour.geojson"
TRAFFIC_URL = "https://api.511.org/traffic/events"
TOLL_URL = "https://api.511.org/toll/programs"
TOKEN_PARAM = "Narration511Token"

RAIN_EVERY, QUAKE_EVERY, TRAFFIC_EVERY, TOLL_EVERY = 300., 120., 150., 300.  # s
TIMEOUT = 5.  # s per request
MAX_WAIT = 60.  # s an announcement may wait for a quiet moment
IDLE_STOP = 10.  # s without update() (offroad: the onroad view isn't drawn) before the thread stops itself

RAIN_MM = 0.2  # mm in a 15-minute slot that counts as rain (light rain)
RAIN_LOOKAHEAD = 60.  # min
RAIN_REPEAT = 1800.  # s before another "rain starting" can be said

QUAKE_MIN_MAG = 3.0  # felt by some people
QUAKE_RADIUS_KM = 80.
QUAKE_MAX_AGE = 1800.  # s: older quakes aren't news any more

TRAFFIC_RADIUS_M = 10000  # asked of 511 around the car
AHEAD_MIN_KM, AHEAD_MAX_KM = 0.3, 5.
AHEAD_CONE = 25.  # deg either side of the heading
MIN_SPEED = 5.  # m/s: below this the GPS heading isn't trustworthy
TOLL_RADIUS_M = 3000
TOLL_MIN_SPEED = 13.  # m/s (~30 mph): on the freeway

KM_PER_MI = 1.609344
_CARDINALS = {"north": 0., "east": 90., "south": 180., "west": 270.}
_SIGN_DIR = {"NB": "north", "EB": "east", "SB": "south", "WB": "west"}


@dataclass
class Announcement:
  key: str  # dedupe: never said twice
  parts: list[str]  # clip texts for soundd
  caption: tuple[str, str]  # on-screen (title, detail); any text, it isn't spoken
  made: float = field(default=0.)


# ---- geometry ----

def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
  p1, p2 = math.radians(lat1), math.radians(lat2)
  a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
  return 6371.0 * 2 * math.asin(min(1., math.sqrt(a)))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
  p1, p2, dl = math.radians(lat1), math.radians(lat2), math.radians(lon2 - lon1)
  y = math.sin(dl) * math.cos(p2)
  x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
  return math.degrees(math.atan2(y, x)) % 360.


def angle_off(a: float, b: float) -> float:
  return abs((a - b + 180.) % 360. - 180.)


def ahead_km(car: tuple[float, float], heading: float, lat: float, lon: float) -> float | None:
  """Distance to a point if it's in front of the car (inside AHEAD_CONE and the distance window), else None."""
  d = distance_km(car[0], car[1], lat, lon)
  if not (AHEAD_MIN_KM <= d <= AHEAD_MAX_KM):
    return None
  return d if angle_off(bearing_deg(car[0], car[1], lat, lon), heading) <= AHEAD_CONE else None


def same_direction(direction_text: str, heading: float) -> bool:
  """511 road direction ("Eastbound", "Northbound and Southbound", "") vs the car's heading. Unknown counts as yes."""
  named = [deg for name, deg in _CARDINALS.items() if name in (direction_text or "").lower()]
  return not named or any(angle_off(deg, heading) < 67.5 for deg in named)


def _distance_parts(km: float, metric: bool, suffix: str) -> list[str]:
  """suffix 'ahead' or 'away' -> clip texts: ["3", "miles ahead."], ["1", "mile ahead."], ["Less than a mile ahead."]"""
  n = km if metric else km / KM_PER_MI
  unit = "kilometer" if metric else "mile"
  if n < 0.95:
    return [nt.live_less_than(unit, suffix)]
  n = min(round(n), max(nt.NUMBERS))
  return [str(n), nt.live_unit(unit, n, suffix)]


def _distance_text(km: float, metric: bool) -> str:
  return f"{km:.0f} km" if metric else f"{km / KM_PER_MI:.0f} mi" if km >= 1.5 else f"{km / KM_PER_MI:.1f} mi"


# ---- rain ----

def rain_request(lat: float, lon: float) -> dict:
  return {"latitude": round(lat, 3), "longitude": round(lon, 3), "minutely_15": "precipitation",
          "forecast_minutely_15": 8, "timeformat": "unixtime"}


def rain_soon(data: dict, now: float) -> Announcement | None:
  """'Rain starting in about 15 minutes.' when it's dry now and a 15-minute slot within the hour is wet."""
  try:
    m = data["minutely_15"]
    slots = [(int(t), float(p or 0.)) for t, p in zip(m["time"], m["precipitation"], strict=True)]
  except (KeyError, TypeError, ValueError):
    return None
  current = [p for t, p in slots if t <= now < t + 900]
  if current and current[0] >= RAIN_MM:
    return None  # already raining: the greeting / the windshield already said so
  for t, p in slots:
    if t > now and p >= RAIN_MM:
      minutes = (t - now) / 60.
      if minutes > RAIN_LOOKAHEAD:
        return None
      n = max(5, int(round(minutes / 5.) * 5))
      return Announcement(f"rain:{t}", [nt.LIVE_RAIN, str(n), nt.LIVE_MINUTES], ("Rain starting soon", f"in about {n} min"))
  return None


# ---- earthquakes ----

def quakes_nearby(data: dict, pos: tuple[float, float], now: float, metric: bool) -> list[Announcement]:
  out = []
  try:
    features = data["features"]
  except (KeyError, TypeError):
    return out
  for f in features:
    try:
      props, coords = f["properties"], f["geometry"]["coordinates"]
      mag, when, qid = float(props["mag"]), float(props["time"]) / 1000., str(f["id"])
      lon, lat = float(coords[0]), float(coords[1])
    except (KeyError, TypeError, ValueError, IndexError):
      continue
    if mag < QUAKE_MIN_MAG or now - when > QUAKE_MAX_AGE:
      continue
    d = distance_km(pos[0], pos[1], lat, lon)
    if d > QUAKE_RADIUS_KM:
      continue
    whole, tenth = int(mag), int(round((mag - int(mag)) * 10))
    if tenth == 10:
      whole, tenth = whole + 1, 0
    mag_parts = [nt.LIVE_MAGNITUDE, str(whole)] + ([nt.LIVE_POINT, str(tenth)] if tenth else [])
    out.append(Announcement(f"quake:{qid}", [nt.LIVE_QUAKE] + mag_parts + _distance_parts(d, metric, "away"),
                            (f"Earthquake M{mag:.1f}", f"{_distance_text(d, metric)} away")))
  return out


# ---- 511 traffic incidents ----

def traffic_request(token: str, lat: float, lon: float) -> dict:
  return {"api_key": token, "format": "json", "status": "ACTIVE", "in_effect_on": "now",
          "event_type": "INCIDENT,ROAD_CONDITION,CONSTRUCTION",
          "geography": f"POINT({lon:.5f} {lat:.5f})", "tolerance": TRAFFIC_RADIUS_M}


def classify_event(ev: dict) -> str | None:
  """Which phrase (narration_text.LIVE_TRAFFIC key) fits an event, or None to stay quiet."""
  kind = str(ev.get("event_type", "")).upper()
  states = {str(r.get("state", "")).upper() for r in ev.get("roads") or [] if isinstance(r, dict)}
  words = (str(ev.get("headline", "")) + " " + " ".join(map(str, ev.get("event_subtypes") or []))).lower()
  if "CLOSED" in states:
    return "closed"
  if kind == "INCIDENT":
    if re.search(r"collision|crash|accident|overturn|spin.?out", words):
      return "crash"
    if re.search(r"stall|disabled|broken.?down|flat tire", words):
      return "stalled"
    if re.search(r"debris|object|hazard|animal|pedestrian|fire|spill", words):
      return "hazard"
    return "lane" if states & {"SOME_LANES_CLOSED", "SINGLE_LANE_ALTERNATING"} else "incident"
  if states & {"SOME_LANES_CLOSED", "SINGLE_LANE_ALTERNATING"}:
    return "lane"  # construction or a road condition taking lanes away
  if kind == "ROAD_CONDITION" and re.search(r"flood|ice|snow|chain|debris|hazard", words):
    return "hazard"
  return None  # planned work with all lanes open, special events, ...


def traffic_ahead(events: list[dict], car: tuple[float, float], heading: float, metric: bool) -> list[Announcement]:
  out = []
  for ev in events:
    try:
      lon, lat = float(ev["geography"]["coordinates"][0]), float(ev["geography"]["coordinates"][1])
      eid = str(ev["id"])
    except (KeyError, TypeError, ValueError, IndexError):
      continue  # line geometries and odd entries: skip rather than guess where they are
    phrase = classify_event(ev)
    if phrase is None:
      continue
    roads = [r for r in ev.get("roads") or [] if isinstance(r, dict)]
    if roads and not any(same_direction(str(r.get("direction", "")), heading) for r in roads):
      continue  # the other side of the freeway
    d = ahead_km(car, heading, lat, lon)
    if d is None:
      continue
    title = nt.LIVE_TRAFFIC[phrase].rstrip(".")
    road = str(roads[0].get("name", "")).strip() if roads else ""
    detail = f"{road} · {_distance_text(d, metric)}" if road else _distance_text(d, metric)
    out.append(Announcement(f"traffic:{eid}", [nt.LIVE_TRAFFIC[phrase]] + _distance_parts(d, metric, "ahead"),
                            (title, detail)))
  return out


# ---- 511 express-lane tolls ----

def toll_request(token: str, lat: float, lon: float) -> dict:
  return {"api_key": token, "format": "json", "geography": f"POINT({lon:.5f} {lat:.5f})", "tolerance": TOLL_RADIUS_M}


def _sign_point(sign: dict) -> tuple[float, float] | None:
  geo = sign.get("geography") or sign.get("location")
  try:
    lon, lat = float(geo["coordinates"][0]), float(geo["coordinates"][1])
    return lat, lon
  except (KeyError, TypeError, ValueError, IndexError):
    return None


def tolls_ahead(data: dict, car: tuple[float, float], heading: float) -> list[Announcement]:
  """Express-lane toll signs near the car (511 already filtered by distance) for the direction it's going."""
  out = []
  try:
    programs = data["toll-programs"]
  except (KeyError, TypeError):
    return out
  for prog in programs:
    for sign in (prog or {}).get("toll-signs") or []:
      try:
        name, sid = str(sign.get("name", "")), str(sign["id"])
        dests = sorted(sign.get("toll-destinations") or [], key=lambda d: int(d.get("order") or 99))
        price = float(dests[0]["toll-rates"][0]["toll-price"])
      except (KeyError, TypeError, ValueError, IndexError):
        continue
      m = re.search(r"\b(NB|SB|EB|WB)\b", name)
      if m is None or angle_off(_CARDINALS[_SIGN_DIR[m.group(1)]], heading) >= 67.5 or price <= 0:
        continue
      point = _sign_point(sign)
      if point is not None and ahead_km(car, heading, *point) is None:
        continue  # behind us or off to the side
      road = name.split(" - ", maxsplit=1)[0].strip()
      out.append(Announcement(f"toll:{sid}:{price:.2f}", nt.live_toll_parts(price), ("Express lane toll", f"${price:.2f} · {road}")))
  return out


# ---- fetching ----

def http_get_json(url: str, params: dict, timeout: float) -> dict:
  import requests
  r = requests.get(url, params=params, timeout=timeout)
  r.raise_for_status()
  return json.loads(r.content.decode("utf-8-sig"))  # 511 replies start with a byte-order mark


@dataclass
class _Car:
  pos: tuple[float, float] | None = None
  heading: float = 0.
  speed: float = 0.
  metric: bool = False
  token: str = ""


class LiveFeeds:
  def __init__(self, get: Callable[[str, dict, float], dict] | None = None,
               clock=time.time):  # noqa: TID251  wall clock: quake times and forecast slots are wall-clock
    self._get = get or http_get_json
    self._clock = clock
    self._lock = threading.Lock()
    self._car = _Car()
    self._queue: deque[Announcement] = deque(maxlen=4)
    self._said: set[str] = set()
    self._last_rain = -1e9
    self._events: list[dict] = []
    self._due = dict.fromkeys(("rain", "quake", "traffic", "toll"), 0.)
    self._thread: threading.Thread | None = None
    self._stop = threading.Event()
    self._last_update = -1e9  # time.monotonic() of the last update()

  def update(self, pos: tuple[float, float] | None, heading: float, speed: float, metric: bool, token: str) -> None:
    """Every frame from the UI: where the car is. Starts the fetch thread the first time."""
    with self._lock:
      self._car = _Car(pos, heading, speed, metric, token.strip())
      self._last_update = time.monotonic()
    if self._thread is None or not self._thread.is_alive():
      self._stop.clear()
      self._thread = threading.Thread(target=self._run, daemon=True, name="narration_live")
      self._thread.start()

  def stop(self) -> None:
    """Offroad: stop fetching and forget the queue (incidents seen stay 'said' until the UI restarts)."""
    self._stop.set()
    with self._lock:
      self._queue.clear()
      self._events = []

  def pop(self, now: float | None = None) -> Announcement | None:
    now = self._clock() if now is None else now
    with self._lock:
      while self._queue:
        a = self._queue.popleft()
        if now - a.made <= MAX_WAIT:
          return a
    return None

  def _push(self, items: list[Announcement], now: float) -> None:
    with self._lock:
      for a in items:
        if a.key not in self._said:
          self._said.add(a.key)
          a.made = now
          self._queue.append(a)

  def step(self, now: float) -> None:
    """One pass: fetch whatever is due, then look for incidents now ahead. Runs ~1 Hz on the thread; tests call it."""
    with self._lock:
      car = self._car
    if car.pos is None:
      return
    moving = car.speed >= MIN_SPEED
    try:
      if now >= self._due["rain"]:
        self._due["rain"] = now + RAIN_EVERY
        a = rain_soon(self._get(RAIN_URL, rain_request(*car.pos), TIMEOUT), now)
        if a is not None and now - self._last_rain > RAIN_REPEAT:
          self._last_rain = now
          self._push([a], now)
    except Exception:
      pass
    try:
      if now >= self._due["quake"]:
        self._due["quake"] = now + QUAKE_EVERY
        self._push(quakes_nearby(self._get(QUAKE_URL, {}, TIMEOUT), car.pos, now, car.metric), now)
    except Exception:
      pass
    if car.token:
      try:
        if now >= self._due["traffic"]:
          self._due["traffic"] = now + TRAFFIC_EVERY
          events = self._get(TRAFFIC_URL, traffic_request(car.token, *car.pos), TIMEOUT).get("events") or []
          with self._lock:
            self._events = [e for e in events if isinstance(e, dict)]
      except Exception:
        pass
      try:
        if now >= self._due["toll"] and moving and car.speed >= TOLL_MIN_SPEED:
          self._due["toll"] = now + TOLL_EVERY
          self._push(tolls_ahead(self._get(TOLL_URL, toll_request(car.token, *car.pos), TIMEOUT), car.pos, car.heading), now)
      except Exception:
        pass
    if moving:
      with self._lock:
        events = list(self._events)
      self._push(traffic_ahead(events, car.pos, car.heading, car.metric), now)

  def _run(self) -> None:
    while not self._stop.is_set():
      with self._lock:
        idle = time.monotonic() - self._last_update > IDLE_STOP
      if idle:
        self.stop()
        return
      self.step(self._clock())
      self._stop.wait(1.0)
