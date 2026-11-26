import time

from openpilot.selfdrive.ui.narration_text import weather_parts
from openpilot.selfdrive.ui.narration_weather import REFRESH_S, WeatherFetcher, parse, position_from_param, request_params, usable, weather_at

T0 = 1_764_000_000 // 3600 * 3600  # an hour boundary
HOME = (37.5, -122.2)


def reply(hours=48, temp_c=14.5, code=45, rain=10):
  return {"hourly": {
    "time": [T0 + 3600 * i for i in range(hours)],
    "temperature_2m": [temp_c] * hours,
    "weather_code": [code] * hours,
    "is_day": [1] * hours,
    "precipitation_probability": [rain] * hours,
  }}


class Store:
  def __init__(self, value=None):
    self.value = value

  def load(self):
    return self.value

  def save(self, v):
    self.value = v


def wait(f):
  for _ in range(200):
    if f._thread is None or not f._thread.is_alive():
      return
    time.sleep(0.01)


def test_request_params():
  p = request_params(37.774929, -122.419416)
  assert p["latitude"] == 37.775 and p["longitude"] == -122.419 and p["temperature_unit"] == "celsius"
  assert "hourly" in p and p["forecast_days"] == 2


def test_parse_and_read_hour():
  cache = parse(reply(), *HOME, fetched=T0)
  assert usable(cache, HOME, T0 + 5 * 3600 + 120) and not usable(cache, HOME, T0 + 49 * 3600)
  assert not usable(cache, (38.5, -122.2), T0)  # stored for somewhere else
  w = weather_at(cache, T0 + 600, metric=False)
  assert round(w.temp) == 58 and w.code == 45 and w.is_day and w.rain_chance == 10
  assert round(weather_at(cache, T0, metric=True).temp) == 14  # 14.5 C rounds half to even
  assert parse({}, *HOME, T0) is None and parse({"hourly": {"time": [], "temperature_2m": [], "weather_code": []}}, *HOME, T0) is None


def test_rain_looks_ahead():
  r = reply()
  r["hourly"]["precipitation_probability"] = [0] * 8 + [70] + [0] * 39  # rain in 8 h
  cache = parse(r, *HOME, T0)
  assert weather_parts(**vars(weather_at(cache, T0, False)))[-1] == "Rain is likely later."
  assert weather_at(cache, T0 + 9 * 3600, False).rain_chance == 0  # already past it


def test_position_from_param():
  assert position_from_param('{"latitude": 37.5, "longitude": -122.2, "altitude": 3}') == (37.5, -122.2)
  assert position_from_param(None) is None and position_from_param("garbage") is None
  assert position_from_param('{"latitude": 0, "longitude": 0}') is None


def test_fetch_stores_forecast():
  store, calls, now = Store(), [], [T0 + 100]
  f = WeatherFetcher(store.load, store.save, get=lambda *a: calls.append(a) or reply(), clock=lambda: now[0])
  f.start(HOME)
  wait(f)
  assert len(calls) == 1 and store.value["time"][0] == T0
  assert round(f.result(metric=False).temp) == 58


def test_offline_later_uses_stored_forecast():
  store = Store(parse(reply(), *HOME, fetched=T0))
  def offline(*_):
    raise OSError("no network")
  now = T0 + 10 * 3600  # hours later, past the refresh age: tries to refresh, can't, still has weather
  f = WeatherFetcher(store.load, store.save, get=offline, clock=lambda: now)
  f.start(HOME)
  assert not f.busy  # greeting doesn't wait: the stored hour is right there
  wait(f)
  assert f.result(metric=False) is not None and store.value["fetched"] == T0


def test_fresh_forecast_not_refetched():
  store, calls = Store(parse(reply(), *HOME, fetched=T0)), []
  f = WeatherFetcher(store.load, store.save, get=lambda *a: calls.append(a) or reply(), clock=lambda: T0 + REFRESH_S - 60)
  f.start(HOME)
  wait(f)
  assert calls == [] and f.result(metric=False) is not None


def test_no_gps_uses_stored():
  store = Store(parse(reply(), *HOME, fetched=T0))
  f = WeatherFetcher(store.load, store.save, get=lambda *a: reply(), clock=lambda: T0 + 3600)
  f.start(None)
  assert f.result(metric=False) is not None


def test_stored_for_elsewhere_not_spoken_offline():
  store = Store(parse(reply(), 34.0, -118.2, fetched=T0))  # LA forecast, now in the Bay Area with no network
  def offline(*_):
    raise OSError
  f = WeatherFetcher(store.load, store.save, get=offline, clock=lambda: T0 + 3600)
  f.start(HOME)
  wait(f)
  assert f.result(metric=False) is None


def test_nothing_stored_and_offline():
  def offline(*_):
    raise OSError
  f = WeatherFetcher(Store().load, Store().save, get=offline, clock=lambda: T0)
  f.start(HOME)
  wait(f)
  assert f.result(metric=False) is None
