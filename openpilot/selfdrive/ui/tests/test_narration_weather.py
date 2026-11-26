import time

from openpilot.selfdrive.ui.narration_weather import WeatherFetcher, parse, position_from_param, request_params

SAMPLE = {
  "current": {"temperature_2m": 58.3, "weather_code": 45, "is_day": 1},
  "daily": {"temperature_2m_max": [67.1], "precipitation_probability_max": [10]},
}


def test_parse():
  w = parse(SAMPLE)
  assert (w.temp, w.code, w.is_day, w.high, w.rain_chance) == (58.3, 45, True, 67.1, 10)
  assert parse({"current": {"temperature_2m": 50, "weather_code": 3}}).high is None  # no daily block
  assert parse({}) is None and parse({"current": {"temperature_2m": None, "weather_code": 1}}) is None


def test_request_params():
  p = request_params(37.774929, -122.419416, metric=False)
  assert p["latitude"] == 37.775 and p["longitude"] == -122.419 and p["temperature_unit"] == "fahrenheit"
  assert request_params(37.7, -122.4, metric=True)["temperature_unit"] == "celsius"


def test_position_from_param():
  assert position_from_param('{"latitude": 37.5, "longitude": -122.2, "altitude": 3}') == (37.5, -122.2)
  assert position_from_param(None) is None and position_from_param("garbage") is None
  assert position_from_param('{"latitude": 0, "longitude": 0}') is None


def wait(f):
  for _ in range(100):
    if not f.busy:
      return
    time.sleep(0.01)


def test_fetch_and_cache():
  calls = []
  f = WeatherFetcher(get=lambda url, params, timeout: calls.append(params) or SAMPLE)
  f.start(37.5, -122.2, False)
  wait(f)
  assert f.result().temp == 58.3 and len(calls) == 1
  f.start(37.51, -122.21, False)  # nearby, recent: cached
  wait(f)
  assert len(calls) == 1
  f.start(38.5, -122.2, False)  # far away: refetch
  wait(f)
  assert len(calls) == 2


def test_fetch_failure_gives_none():
  def boom(*_):
    raise OSError("no network")
  f = WeatherFetcher(get=boom)
  f.start(37.5, -122.2, False)
  wait(f)
  assert f.result() is None
