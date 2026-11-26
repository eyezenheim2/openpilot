"""Current weather for the onroad greeting, from Open-Meteo (free, no API key).

The fetch runs on a background thread when openpilot goes onroad, so the UI never waits on the network. No
connection, no GPS fix yet or a slow reply just means the greeting is said without the weather.
"""
import json
import threading
import time
from dataclasses import dataclass

URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT = 4.0  # s per request
CACHE_S = 20 * 60  # reuse a result this long (restarting the car a few minutes later doesn't refetch)
CACHE_DIST = 0.05  # deg (~5 km): a cached result farther than this from the car is fetched again


@dataclass
class Weather:
  temp: float  # current, in the requested unit
  code: int  # WMO weather code
  is_day: bool
  rain_chance: int | None = None  # today's highest precipitation probability, %


def request_params(lat: float, lon: float, metric: bool) -> dict:
  return {
    "latitude": round(lat, 3), "longitude": round(lon, 3),  # ~100 m is plenty, and no more precise than needed
    "current": "temperature_2m,weather_code,is_day",
    "daily": "precipitation_probability_max",
    "temperature_unit": "celsius" if metric else "fahrenheit",
    "timezone": "auto", "forecast_days": 1,
  }


def parse(data: dict) -> Weather | None:
  try:
    cur = data["current"]
    daily = data.get("daily") or {}
    rain = daily.get("precipitation_probability_max") or [None]
    return Weather(temp=float(cur["temperature_2m"]), code=int(cur["weather_code"]), is_day=bool(cur.get("is_day", 1)),
                   rain_chance=None if rain[0] is None else int(rain[0]))
  except (KeyError, TypeError, ValueError, IndexError):
    return None


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
  def __init__(self, get=None):
    self._get = get  # (url, params, timeout) -> dict; injectable for tests
    self._lock = threading.Lock()
    self._thread: threading.Thread | None = None
    self._result: Weather | None = None
    self._key: tuple[float, float, bool] | None = None
    self._at = -1e9

  def start(self, lat: float, lon: float, metric: bool) -> None:
    """Begin a fetch unless one is running or a recent nearby result is cached."""
    with self._lock:
      if self._thread is not None and self._thread.is_alive():
        return
      if self._fresh(lat, lon, metric):
        return
      self._result = None
      self._thread = threading.Thread(target=self._run, args=(lat, lon, metric), daemon=True, name="narration_weather")
      self._thread.start()

  def _fresh(self, lat: float, lon: float, metric: bool) -> bool:
    if self._result is None or self._key is None or time.monotonic() - self._at > CACHE_S:
      return False
    klat, klon, kmetric = self._key
    return kmetric == metric and abs(klat - lat) < CACHE_DIST and abs(klon - lon) < CACHE_DIST

  @property
  def busy(self) -> bool:
    with self._lock:
      return self._thread is not None and self._thread.is_alive()

  def result(self) -> Weather | None:
    with self._lock:
      return self._result

  def _run(self, lat: float, lon: float, metric: bool) -> None:
    try:
      get = self._get or _http_get
      weather = parse(get(URL, request_params(lat, lon, metric), TIMEOUT))
    except Exception:
      weather = None
    with self._lock:
      self._result, self._key, self._at = weather, (lat, lon, metric), time.monotonic()


def _http_get(url: str, params: dict, timeout: float) -> dict:
  import requests
  r = requests.get(url, params=params, timeout=timeout)
  r.raise_for_status()
  return r.json()
