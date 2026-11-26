"""Weather for the onroad greeting, from Open-Meteo (free, no API key).

Every time openpilot goes onroad a fresh forecast is fetched (hourly, today and tomorrow) and saved to the
DriveNarrationWeatherCache param, replacing the old one. The greeting waits a few seconds for it; when there's no
internet (or it's slow) the saved forecast is used instead, so a drive without a connection still gets this hour's
weather. No saved forecast for this hour or this area either just means the greeting has no weather.
"""
import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

URL = "https://api.open-meteo.com/v1/forecast"
CACHE_PARAM = "DriveNarrationWeatherCache"
TIMEOUT = 4.0  # s per request
FORECAST_DAYS = 2  # covers a drive past midnight
MAX_DIST = 0.3  # deg (~30 km): a forecast for farther away than this isn't used
RAIN_HOURS = 12  # "Rain is likely later." looks this far ahead


@dataclass
class Weather:
  temp: float  # current, in the requested unit
  code: int  # WMO weather code
  is_day: bool
  rain_chance: int | None = None  # highest precipitation probability over the next RAIN_HOURS, %


def request_params(lat: float, lon: float) -> dict:
  return {
    "latitude": round(lat, 3), "longitude": round(lon, 3),  # ~100 m is plenty, and no more precise than needed
    "hourly": "temperature_2m,weather_code,is_day,precipitation_probability",
    "temperature_unit": "celsius",  # stored in C, converted when spoken so the unit setting can change any time
    "timeformat": "unixtime", "forecast_days": FORECAST_DAYS,
  }


def parse(data: dict, lat: float, lon: float, fetched: float) -> dict | None:
  """Open-Meteo reply -> what gets stored."""
  try:
    h = data["hourly"]
    n = len(h["time"])
    cache = {
      "lat": lat, "lon": lon, "fetched": fetched,
      "time": [int(t) for t in h["time"]],
      "temp": [None if v is None else float(v) for v in h["temperature_2m"]],
      "code": [None if v is None else int(v) for v in h["weather_code"]],
      "is_day": [bool(v) for v in h.get("is_day") or [1] * n],
      "rain": [None if v is None else int(v) for v in h.get("precipitation_probability") or [None] * n],
    }
  except (KeyError, TypeError, ValueError):
    return None
  if n == 0 or any(len(cache[k]) != n for k in ("temp", "code", "is_day", "rain")):
    return None
  return cache


def _hour_index(cache: dict, now: float) -> int | None:
  for i, t in enumerate(cache["time"]):
    if t <= now < t + 3600:
      return i
  return None


def usable(cache, pos: tuple[float, float] | None, now: float) -> bool:
  """Has the current hour, and (when we know where the car is) is for around here."""
  if not isinstance(cache, dict):
    return False
  try:
    i = _hour_index(cache, now)
    if i is None or cache["temp"][i] is None or cache["code"][i] is None:
      return False
    if pos is not None and (abs(cache["lat"] - pos[0]) > MAX_DIST or abs(cache["lon"] - pos[1]) > MAX_DIST):
      return False
  except (KeyError, TypeError, IndexError):
    return False
  return True


def weather_at(cache: dict, now: float, metric: bool) -> Weather | None:
  i = _hour_index(cache, now)
  if i is None or cache["temp"][i] is None or cache["code"][i] is None:
    return None
  temp = cache["temp"][i] if metric else cache["temp"][i] * 9 / 5 + 32
  ahead = [r for r in cache["rain"][i:i + RAIN_HOURS] if r is not None]
  return Weather(temp=temp, code=cache["code"][i], is_day=cache["is_day"][i], rain_chance=max(ahead) if ahead else None)


def position_from_param(raw) -> tuple[float, float] | None:
  """LastGPSPositionLLK: JSON {"latitude", "longitude", ...}."""
  if not raw:
    return None
  try:
    pos = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    lat, lon = float(pos["latitude"]), float(pos["longitude"])
  except (KeyError, TypeError, ValueError):
    return None
  if not (-90 <= lat <= 90 and -180 <= lon <= 180) or (lat == 0 and lon == 0):
    return None
  return lat, lon


class WeatherFetcher:
  def __init__(self, load: Callable[[], dict | None], save: Callable[[dict], None], get=None,
               clock=time.time):  # noqa: TID251  wall clock: matched against the forecast hours
    self._load, self._save = load, save  # stored forecast (the param), so it survives reboots
    self._get = get  # (url, params, timeout) -> dict; injectable for tests
    self._clock = clock
    self._lock = threading.Lock()
    self._thread: threading.Thread | None = None
    self._cache: dict | None = None

  def start(self, pos: tuple[float, float] | None) -> None:
    """Load the saved forecast as a fallback and fetch a fresh one in the background (at the saved forecast's place
    when there's no GPS fix yet)."""
    with self._lock:
      if self._thread is not None and self._thread.is_alive():
        return
      try:
        self._cache = self._load()
      except Exception:
        self._cache = None
      now = self._clock()
      have = usable(self._cache, pos, now)
      if not have and pos is not None:
        self._cache = None  # for somewhere else: don't speak it
      if pos is None and have:
        pos = (self._cache["lat"], self._cache["lon"])
      if pos is None:
        return
      self._thread = threading.Thread(target=self._run, args=pos, daemon=True, name="narration_weather")
      self._thread.start()

  @property
  def busy(self) -> bool:
    """Fresh forecast still on its way (the greeting waits a few seconds for it)."""
    with self._lock:
      return self._thread is not None and self._thread.is_alive()

  def result(self, metric: bool) -> Weather | None:
    with self._lock:
      now = self._clock()
      return weather_at(self._cache, now, metric) if usable(self._cache, None, now) else None

  def _run(self, lat: float, lon: float) -> None:
    try:
      get = self._get or _http_get
      cache = parse(get(URL, request_params(lat, lon), TIMEOUT), lat, lon, self._clock())
    except Exception:
      cache = None
    if cache is None:
      return  # offline: keep whatever was stored
    with self._lock:
      self._cache = cache
    try:
      self._save(cache)
    except Exception:
      pass


def _http_get(url: str, params: dict, timeout: float) -> dict:
  import requests
  r = requests.get(url, params=params, timeout=timeout)
  r.raise_for_status()
  return r.json()
