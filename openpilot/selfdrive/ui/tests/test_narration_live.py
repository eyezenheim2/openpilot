from openpilot.selfdrive.ui import narration_live as nl
from openpilot.selfdrive.ui import narration_text as nt

CAR = (37.8000, -122.3000)  # Bay Area
NOW = 1_800_000_000.0
KM_LAT = 1 / 111.2  # degrees of latitude per km


def north_of(km: float, east_km: float = 0.) -> tuple[float, float]:
  return CAR[0] + km * KM_LAT, CAR[1] + east_km * KM_LAT / 0.79


# ---- geometry ----

def test_ahead_only_in_front_and_in_range():
  assert abs(nl.ahead_km(CAR, 0., *north_of(2.0)) - 2.0) < 0.05  # 2 km straight ahead, heading north
  assert nl.ahead_km(CAR, 180., *north_of(2.0)) is None  # behind
  assert nl.ahead_km(CAR, 0., *north_of(2.0, east_km=2.0)) is None  # 45 deg off to the side
  assert nl.ahead_km(CAR, 0., *north_of(0.1)) is None and nl.ahead_km(CAR, 0., *north_of(9.0)) is None


def test_same_direction():
  assert nl.same_direction("Northbound", 10.) and not nl.same_direction("Southbound", 10.)
  assert nl.same_direction("Northbound and Southbound", 190.) and nl.same_direction("", 123.)


# ---- rain ----

def rain_reply(mm_by_slot):
  start = int(NOW // 900 * 900)
  return {"minutely_15": {"time": [start + 900 * i for i in range(len(mm_by_slot))], "precipitation": mm_by_slot}}


def test_rain_starting_soon():
  a = nl.rain_soon(rain_reply([0, 0, 0.5, 1.0]), NOW)  # dry now, wet from the third slot
  assert a.parts[0] == nt.LIVE_RAIN and a.parts[2] == nt.LIVE_MINUTES and int(a.parts[1]) in (20, 25, 30)
  assert nl.rain_soon(rain_reply([0.6, 0.6, 0.6]), NOW) is None  # already raining
  assert nl.rain_soon(rain_reply([0, 0, 0, 0, 0, 0]), NOW) is None
  assert nl.rain_soon({"oops": 1}, NOW) is None


# ---- earthquakes ----

def quake(mag, km, minutes_ago, qid="nc1"):
  lat, lon = north_of(km)
  return {"id": qid, "properties": {"mag": mag, "time": (NOW - minutes_ago * 60) * 1000},
          "geometry": {"coordinates": [lon, lat, 8.0]}}


def test_quake_nearby():
  [a] = nl.quakes_nearby({"features": [quake(4.2, 20., 5)]}, CAR, NOW, metric=False)
  assert a.parts == [nt.LIVE_QUAKE, nt.LIVE_MAGNITUDE, "4", nt.LIVE_POINT, "2", "12", "miles away."]
  [a] = nl.quakes_nearby({"features": [quake(3.0, 20., 5)]}, CAR, NOW, metric=True)
  assert a.parts == [nt.LIVE_QUAKE, nt.LIVE_MAGNITUDE, "3", "20", "kilometers away."]


def test_quake_ignored_when_small_old_or_far():
  feats = [quake(2.6, 10., 5, "a"), quake(4.5, 10., 90, "b"), quake(5.0, 300., 5, "c")]
  assert nl.quakes_nearby({"features": feats}, CAR, NOW, metric=False) == []


# ---- 511 traffic ----

def event(eid, km, headline, kind="INCIDENT", direction="Northbound", state="SOME_LANES_CLOSED", east_km=0.):
  lat, lon = north_of(km, east_km)
  return {"id": f"511.org/{eid}", "event_type": kind, "headline": headline, "event_subtypes": [],
          "geography": {"type": "Point", "coordinates": [lon, lat]},
          "roads": [{"name": "I-80", "direction": direction, "state": state}]}


def test_crash_ahead():
  [a] = nl.traffic_ahead([event(1, 3.3, "Collision on I-80 NB at Ashby")], CAR, 0., metric=False)
  assert a.parts == ["Crash reported ahead.", "2", "miles ahead."]
  assert a.caption == ("Crash reported ahead", "I-80 · 2 mi")


def test_traffic_kinds():
  def kind(**kw):
    out = nl.traffic_ahead([event(9, 2.0, **kw)], CAR, 0., metric=False)
    return out[0].parts[0] if out else None
  assert kind(headline="Stalled vehicle in #2 lane") == "Stalled vehicle ahead."
  assert kind(headline="Debris on roadway") == "Road hazard ahead."
  assert kind(headline="Full closure", state="CLOSED") == "Road closed ahead."
  assert kind(headline="Lane closure for paving", kind="CONSTRUCTION") == "Lane closure ahead."
  assert kind(headline="Paving", kind="CONSTRUCTION", state="ALL_LANES_OPEN") is None
  assert kind(headline="Something", kind="SPECIAL_EVENT", state="ALL_LANES_OPEN") is None


def test_traffic_ignored_behind_or_other_direction_or_close():
  ev = [event(1, 2.0, "Collision", direction="Southbound"), event(2, 0.1, "Collision"), event(3, 2.0, "Collision")]
  out = nl.traffic_ahead(ev, CAR, 180., metric=False)  # heading south: #1 is behind, #3 behind, #2 too close
  assert out == []
  assert nl.traffic_ahead([event(4, 0.5, "Collision")], CAR, 0., metric=False)[0].parts[-1] == "Less than a mile ahead."


# ---- 511 tolls ----

def toll_reply(name, price, sid="s1"):
  return {"toll-programs": [{"toll-signs": [{"id": sid, "name": name, "toll-destinations": [
    {"id": "d2", "order": 2, "toll-rates": [{"toll-price": 9.0}]},
    {"id": "d1", "order": 1, "toll-rates": [{"toll-price": price}]},
  ]}]}]}


def test_toll_for_our_direction():
  [a] = nl.tolls_ahead(toll_reply("I-680 NB - Livorna - 5", 4.5), CAR, 5.)
  assert a.parts == [nt.LIVE_TOLL, "4", "dollars", "50"] and a.caption == ("Express lane toll", "$4.50 · I-680 NB")
  assert nl.tolls_ahead(toll_reply("I-680 NB - Livorna - 5", 4.5), CAR, 185.) == []  # going south
  assert nl.tolls_ahead(toll_reply("I-680 NB - Livorna - 5", 0.0), CAR, 5.) == []  # free right now


def test_toll_parts():
  assert nt.live_toll_parts(1.0) == [nt.LIVE_TOLL, "1", "dollar"]
  assert nt.live_toll_parts(0.75) == [nt.LIVE_TOLL, "75", "cents."]


# ---- the fetch loop ----

class FakeNet:
  def __init__(self):
    self.calls = []
    self.replies = {nl.RAIN_URL: rain_reply([0, 0, 0.5]), nl.QUAKE_URL: {"features": [quake(4.2, 20., 5)]},
                    nl.TRAFFIC_URL: {"events": [event(1, 3.3, "Collision")]},
                    nl.TOLL_URL: toll_reply("I-80 NB - x - 1", 2.0)}

  def get(self, url, params, timeout):
    self.calls.append(url)
    return self.replies[url]


def test_step_fetches_dedupes_and_needs_token_for_511():
  net = FakeNet()
  feeds = nl.LiveFeeds(get=net.get, clock=lambda: NOW)
  feeds._car = nl._Car(CAR, 0., 20., False, "")
  feeds.step(NOW)
  assert set(net.calls) == {nl.RAIN_URL, nl.QUAKE_URL}  # no token: no 511 requests
  feeds._car = nl._Car(CAR, 0., 20., False, "tok")
  feeds.step(NOW + 1)
  said = []
  while (a := feeds.pop(NOW + 2)) is not None:
    said.append(a.parts[0])
  assert set(said) == {nt.LIVE_RAIN, nt.LIVE_QUAKE, "Crash reported ahead.", nt.LIVE_TOLL}
  feeds._due = dict.fromkeys(feeds._due, 0.)
  feeds.step(NOW + 400)  # same quake, crash, toll and rain spell again: nothing new
  assert feeds.pop(NOW + 401) is None


def test_stale_announcements_dropped():
  feeds = nl.LiveFeeds(get=FakeNet().get, clock=lambda: NOW)
  feeds._car = nl._Car(CAR, 0., 20., False, "")
  feeds.step(NOW)
  assert feeds.pop(NOW + nl.MAX_WAIT + 1) is None


def test_offline_is_quiet():
  def offline(url, params, timeout):
    raise OSError("no network")
  feeds = nl.LiveFeeds(get=offline, clock=lambda: NOW)
  feeds._car = nl._Car(CAR, 0., 20., False, "tok")
  feeds.step(NOW)
  assert feeds.pop(NOW) is None


def test_every_live_phrase_has_a_clip_name():
  names = [nt.clip_name(t) for t in nt.live_clip_texts()]
  assert len(names) == len(set(names))
