"""Drive narration: short one-line captions that say what openpilot is doing and why.

  Car ahead slowing · I'm slowing down       Stop sign · Stopping
  Car cutting in · Making room               Curve ahead · Easing off
  Set to 65 · Speeding up                    Car merging in · Watching it
  ...

The decision part (NarratorLogic) only sees plain numbers (NarratorInputs) so it can be tested and
replayed off-device; DriveNarrator pulls those numbers out of the UI's SubMaster and draws the caption.

Openpilot's own alerts always win: captions hide under them. That's also why lane changes, blind-spot
warnings and the lane-change nudge prompt aren't narrated here; openpilot already shows alerts for them.
"""
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pyray as rl

from openpilot.cereal import custom, log
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget
from openpilot.selfdrive.ui.narration_text import (Msg, TEXT, VOICES, SpeechTrigger,
                                                               greeting_caption, greeting_choices, name_clip, resolve_name,
                                                               spoken_parts, time_caption, time_parts, weather_parts, StopAndGo)
from openpilot.common.params import Params
from openpilot.selfdrive.ui.narration_weather import CACHE_PARAM, WeatherFetcher, position_from_param

NARRATION_PARAM = "DriveNarration"
GREETING_DELAY = 1.5  # s after openpilot has loaded (controls and model running) before the onroad greeting
WEATHER_WAIT = 4.0  # s past GREETING_DELAY the greeting waits for a fresh forecast before using the saved one
GREETING_TZ = ZoneInfo("America/Los_Angeles")  # same zone as the HUD clock
TIME_QUIET_S = 3.0  # and only once nothing has been said for this long
GREETING_CAPTION_S = 4.0  # s the "Good morning · Eyezenheim" caption stays up
TIME_CAPTION_S = 3.0  # s the "It's · 3:15" caption stays up (a driving caption takes its place right away)
PLANTED_STEERING_PARAM = "PlantedSteering"


WHITE, BLUE, GREEN, AMBER, ORANGE, RED, TEAL = ((200, 210, 225), (90, 170, 255), (0, 230, 90), (255, 190, 40),
                                                (255, 150, 30), (255, 90, 70), (60, 220, 200))
TIME_ACCENT = (200, 210, 225)
ACCENT = {
  Msg.PLANTED: TEAL, Msg.LANES_FADED: AMBER, Msg.WAITING: WHITE, Msg.PULLING_AWAY: GREEN, Msg.GOING: GREEN,
  Msg.SIGN_GO: GREEN, Msg.GREEN_LIGHT: GREEN, Msg.ROAD_CLEAR: GREEN, Msg.SET_SPEED_UP: GREEN, Msg.SET_SPEED_DOWN: BLUE,
  Msg.MORE_ROOM: BLUE, Msg.LESS_ROOM: GREEN, Msg.SPEED_LIMIT: BLUE, Msg.CURVE: BLUE, Msg.SLOWING: BLUE,
  Msg.STOP_AHEAD: RED, Msg.STOP_SIGN: RED, Msg.STOPPING: RED, Msg.MERGING: AMBER,
  Msg.CUT_IN: ORANGE, Msg.EYES: AMBER,
}

# minimum time a caption stays up once shown, so it can be read
MIN_HOLD = dict.fromkeys(Msg, 1.5)
MIN_HOLD.update({Msg.CUT_IN: 3.0, Msg.MERGING: 2.0, Msg.GOING: 2.0, Msg.SIGN_GO: 2.0, Msg.GREEN_LIGHT: 2.0,
                 Msg.ROAD_CLEAR: 2.5, Msg.SET_SPEED_UP: 2.0, Msg.SET_SPEED_DOWN: 2.0, Msg.MORE_ROOM: 2.0,
                 Msg.LESS_ROOM: 2.0, Msg.PLANTED: 2.0, Msg.LANES_FADED: 2.5})
# how long a condition must hold before its caption appears (stops flicker); events show at once
CONFIRM = dict.fromkeys(Msg, 0.3)
CONFIRM.update({Msg.CUT_IN: 0.0, Msg.LANES_FADED: 1.5, Msg.PLANTED: 0.5, Msg.EYES: 0.5, Msg.CURVE: 0.5})

# --- thresholds ---
SLOW_ACCEL = -0.4  # m/s^2 commanded decel that counts as "slowing down"
CLOSING_SPEED = -0.3  # m/s, vRel below this means we're closing on the lead
LEAD_BRAKING = -0.6  # m/s^2 lead accel that counts as "car ahead slowing"
STOPPED_LEAD_SPEED = 1.0  # m/s
WAITING_MAX_DIST = 15.0  # m
GO_ACCEL = 0.2  # m/s^2 commanded accel that counts as "going"
CUT_IN_MIN_SPEED = 3.0  # m/s
CUT_IN_HEADWAY = 1.5  # s, a newly-appeared lead closer than this is a cut-in
CUT_IN_ABSENT = 1.0  # s with no lead before a new one counts as "appeared" (ignores radar dropouts)
CUT_IN_JUMP = 6.0  # m, lead distance dropping by at least this between frames means a new car took the spot
MERGE_MIN_SPEED = 5.0  # m/s
MERGE_MIN_OFFSET = 0.9  # m from our path
MERGE_MAX_OFFSET = 3.5  # m
MERGE_INWARD_SPEED = 0.35  # m/s drifting toward our path
MERGE_MAX_DIST = 60.0  # m
LAT_RATE_TAU = 0.3  # s, smoothing on the lead's lateral speed
E2E_LEAD_HEADWAY = 3.0  # s: in Experimental mode the plan doesn't say why it brakes; a closing lead this near is the reason
STOP_AHEAD_MAX_DIST = 80.0  # m: model planning a stop within this, with no lead before it
STOP_SIGN_MEMORY = 3.0  # s since the car's sign recognition last reported a stop sign
SIGN_GO_WINDOW = 15.0  # s after a stop-sign stop in which pulling away says "full stop made"
CURVE_LAT_ACCEL = 1.6  # m/s^2 planned lateral accel ahead that counts as a curve
CURVE_AHEAD_MARGIN = 0.5  # m/s^2 the curve ahead must be sharper than what we're in now
CURVE_BRAKE = -0.2  # m/s^2
SET_SPEED_WINDOW = 4.0  # s after a set-speed change
SET_SPEED_GAP = 2  # display units: still this far from the new set speed
ROAD_CLEAR_WINDOW = (0.5, 6.0)  # s after the lead left
ROAD_CLEAR_GAP = 3  # display units below set speed
PERSONALITY_WINDOW = 8.0  # s after a follow-distance change
PULL_AWAY_LEAD = 1.0  # m/s lead opening on us
PULL_AWAY_ACCEL = 0.3  # m/s^2
EYES_AWARENESS = (62, 95)  # %: distracted and draining, but before openpilot's first alert (~61%)
LANES_FADED_PROB = 0.3
LANES_FADED_MIN_SPEED = 11.0  # m/s (~25 mph)
# planted steering's low-speed wobble gate (mirrors latcontrol_pid.WobbleGate)
GATE_SPEED_BP = [6.7, 9.0]
GATE_STRAIGHT_BP = [5., 10.]
GATE_AVG_RC = 1.0
GATE_WEIGHT_RC = 0.3
GATE_SHOW_WEIGHT = 0.6
GATE_SHOW_DEV = 1.5  # deg of wiggle the gate is dropping
GATE_MIN_SPEED = 2.0  # m/s


@dataclass
class NarratorInputs:
  engaged: bool = True
  long_active: bool = False  # openpilot in control of gas/brake right now
  has_long: bool = True  # openpilot longitudinal (not stock ACC)
  lat_active: bool = False
  v_ego: float = 0.0  # m/s
  v_ego_disp: float = 0.0  # display units (mph or km/h)
  standstill: bool = False
  accel_cmd: float = 0.0  # m/s^2 openpilot is asking for
  should_stop: bool = False
  # car ahead (radarState.leadOne)
  lead_present: bool = False
  lead_d: float = 0.0  # m
  lead_v: float = 0.0  # m/s absolute
  lead_v_rel: float = 0.0  # m/s, negative = closing
  lead_a: float = 0.0  # m/s^2
  lead_offset: float = 0.0  # m from our planned path at the lead's distance, either side
  # planner
  plan_source: str = "lead"  # "lead", "e2e", "cruise"
  model_stop_dist: float | None = None  # m to where the model's plan comes to a stop
  plan_max_lat_accel: float = 0.0  # m/s^2, sharpest planned lateral accel ahead
  cur_lat_accel: float = 0.0  # m/s^2 now
  scc_curve: bool = False  # sunnypilot vision/map turn speed control is limiting speed
  speed_limit_adapting: bool = False  # sunnypilot speed limit assist is slowing us
  speed_limit: int = 0  # display units
  set_speed: int = 0  # display units, 0 = not set
  green_light: bool = False  # sunnypilot green-light alert
  stop_sign: bool = False  # car's sign recognition reported a stop sign this frame
  personality: int = 1  # 0 aggressive, 1 standard, 2 relaxed
  # driver monitoring
  dm_distracted: bool = False
  dm_awareness: int = 100  # %
  dm_alert: int = 0
  # steering
  lane_probs: tuple[float, float] = (1.0, 1.0)  # left, right inner lane lines
  planted_on: bool = False
  steer_request_deg: float = 0.0


class NarratorLogic:
  def __init__(self):
    self.reset()

  def reset(self):
    self.msg = Msg.NONE
    self.value = 0
    self._shown_at = 0.0
    self._candidate = Msg.NONE
    self._candidate_since = 0.0
    self._prev_t: float | None = None
    # lead tracking
    self._prev_lead_present = False
    self._prev_lead_d = 0.0
    self._lead_lost_at: float | None = 0.0  # None while a lead is present
    self._was_following = False
    self._prev_offset: float | None = None
    self._lat_rate = 0.0
    self._waiting = False
    self._cut_in_until = 0.0
    # other trackers
    self._sign_seen_at = -1e9
    self._sign_stop_at = -1e9  # when we last came to a stop for a stop sign
    self._green_at = -1e9
    self._prev_set_speed = 0
    self._set_changed_at = -1e9
    self._set_dir = 0
    self._prev_personality: int | None = None
    self._personality_changed_at = -1e9
    self._personality_dir = 0
    self._gate_avg: float | None = None
    self._gate_w = 0.0

  # ---- trackers ----

  def _track_lead(self, i: NarratorInputs, now: float, dt: float) -> tuple[bool, bool]:
    """Returns (cut_in_event, merging) from how the lead changed since last frame."""
    cut_in = False
    if i.lead_present and i.long_active and i.v_ego > CUT_IN_MIN_SPEED:
      close = i.lead_d < i.v_ego * CUT_IN_HEADWAY
      appeared = not self._prev_lead_present and self._lead_lost_at is not None and now - self._lead_lost_at > CUT_IN_ABSENT
      replaced = self._prev_lead_present and self._prev_lead_d - i.lead_d > max(CUT_IN_JUMP, 0.3 * self._prev_lead_d)
      cut_in = close and (appeared or replaced)

    merging = False
    if i.lead_present and self._prev_lead_present and self._prev_offset is not None and 0.0 < dt < 0.5:
      rate = (i.lead_offset - self._prev_offset) / dt
      self._lat_rate += dt / (LAT_RATE_TAU + dt) * (rate - self._lat_rate)
      inward = -self._lat_rate * np.sign(i.lead_offset)  # positive when heading toward our path
      merging = (i.v_ego > MERGE_MIN_SPEED and i.lead_d < MERGE_MAX_DIST and
                 MERGE_MIN_OFFSET < abs(i.lead_offset) < MERGE_MAX_OFFSET and inward > MERGE_INWARD_SPEED)
    else:
      self._lat_rate = 0.0

    if i.lead_present:
      self._lead_lost_at = None
      self._was_following = self._lead_caused(i) or i.lead_d < max(i.v_ego * E2E_LEAD_HEADWAY, 12.0)
    elif self._lead_lost_at is None:
      self._lead_lost_at = now
    self._prev_lead_present = i.lead_present
    self._prev_lead_d = i.lead_d
    self._prev_offset = i.lead_offset if i.lead_present else None
    return cut_in, merging

  def _track_settings(self, i: NarratorInputs, now: float):
    if i.set_speed and self._prev_set_speed and i.set_speed != self._prev_set_speed:
      self._set_changed_at = now
      self._set_dir = 1 if i.set_speed > self._prev_set_speed else -1
    self._prev_set_speed = i.set_speed

    if self._prev_personality is not None and i.personality != self._prev_personality:
      self._personality_changed_at = now
      self._personality_dir = 1 if i.personality > self._prev_personality else -1
    self._prev_personality = i.personality

    if i.stop_sign:
      self._sign_seen_at = now
    if i.green_light:
      self._green_at = now

  def _track_gate(self, i: NarratorInputs, dt: float) -> bool:
    """Planted steering's wobble gate is engaged and actually dropping wiggle."""
    if not (i.planted_on and i.lat_active) or dt <= 0.0:
      self._gate_avg, self._gate_w = None, 0.0
      return False
    if self._gate_avg is None:
      self._gate_avg = i.steer_request_deg
    self._gate_avg += dt / (GATE_AVG_RC + dt) * (i.steer_request_deg - self._gate_avg)
    target = float(np.interp(i.v_ego, GATE_SPEED_BP, [1., 0.]) * np.interp(abs(self._gate_avg), GATE_STRAIGHT_BP, [1., 0.]))
    self._gate_w += dt / (GATE_WEIGHT_RC + dt) * (target - self._gate_w)
    dev = abs(i.steer_request_deg - self._gate_avg)
    return i.v_ego > GATE_MIN_SPEED and self._gate_w > GATE_SHOW_WEIGHT and dev > GATE_SHOW_DEV

  # ---- conditions ----

  @staticmethod
  def _lead_caused(i: NarratorInputs) -> bool:
    """Is the planner reacting to the lead? Experimental mode doesn't say, so a close, closing/slow lead counts."""
    if not i.lead_present:
      return False
    if i.plan_source == "lead":
      return True
    if i.plan_source == "e2e":
      near = i.lead_d < max(i.v_ego * E2E_LEAD_HEADWAY, 12.0)
      return near and (i.lead_v_rel < CLOSING_SPEED or i.lead_v < STOPPED_LEAD_SPEED or i.lead_a < LEAD_BRAKING)
    return False

  def _conditions(self, i: NarratorInputs, now: float, merging: bool, gate: bool) -> list[tuple[Msg, int]]:
    out: list[tuple[Msg, int]] = []
    if not i.engaged:
      self._waiting = False
      return out

    # driver and steering: apply whenever openpilot is steering
    if i.dm_distracted and EYES_AWARENESS[0] < i.dm_awareness < EYES_AWARENESS[1] and i.dm_alert == 0:
      out.append((Msg.EYES, 0))
    if i.lat_active and i.v_ego > LANES_FADED_MIN_SPEED and max(i.lane_probs) < LANES_FADED_PROB:
      out.append((Msg.LANES_FADED, 0))
    if gate:
      out.append((Msg.PLANTED, 0))

    if not i.has_long:
      return out
    if not i.long_active:
      self._waiting = False
      return out

    braking = i.accel_cmd < SLOW_ACCEL or i.should_stop
    going = i.accel_cmd > GO_ACCEL
    lead = self._lead_caused(i)
    stopped = i.standstill or i.v_ego < 0.3

    if now < self._cut_in_until and i.lead_present:
      out.append((Msg.CUT_IN, 0))
    if merging:
      out.append((Msg.MERGING, 0))

    # stopped / pulling away
    if stopped:
      if now - self._sign_seen_at < STOP_SIGN_MEMORY + 5.0 and not i.lead_present:
        self._sign_stop_at = now
      if i.lead_present and i.lead_d < WAITING_MAX_DIST:
        if self._waiting and i.lead_v > STOPPED_LEAD_SPEED and going:
          out.append((Msg.GOING, 0))
        else:
          self._waiting = True
          out.append((Msg.WAITING, 0))
    else:
      if self._waiting and going and i.lead_present and i.lead_v > STOPPED_LEAD_SPEED:
        out.append((Msg.GOING, 0))
      elif not going:
        self._waiting = False
      if going and now - self._sign_stop_at < SIGN_GO_WINDOW and i.v_ego < 6.0:
        out.append((Msg.SIGN_GO, 0))
    if going and now - self._green_at < 3.0:
      out.append((Msg.GREEN_LIGHT, 0))

    # slowing for the car ahead
    if braking and lead and not stopped:
      out.append((Msg.STOPPING if i.lead_v < STOPPED_LEAD_SPEED else Msg.SLOWING, 0))

    # stopping with no car in the way: stop sign (car's sign recognition) or light / line (model plans a stop)
    # (only when the Experimental-mode plan is the one braking: in chill mode openpilot doesn't stop for these)
    clear_to_stop = (i.plan_source == "e2e" and i.model_stop_dist is not None and i.model_stop_dist < STOP_AHEAD_MAX_DIST and
                     not (i.lead_present and i.lead_d < i.model_stop_dist + 5.0))
    if braking and not stopped and clear_to_stop:
      out.append((Msg.STOP_SIGN if now - self._sign_seen_at < STOP_SIGN_MEMORY else Msg.STOP_AHEAD, 0))

    # curves and speed limits
    curve_ahead = (i.plan_max_lat_accel > CURVE_LAT_ACCEL and i.plan_max_lat_accel > i.cur_lat_accel + CURVE_AHEAD_MARGIN)
    if i.accel_cmd < CURVE_BRAKE and not lead and not clear_to_stop and (i.scc_curve or (curve_ahead and i.plan_source == "e2e")):
      out.append((Msg.CURVE, 0))
    if i.speed_limit_adapting and i.speed_limit > 0 and i.accel_cmd < CURVE_BRAKE:
      out.append((Msg.SPEED_LIMIT, i.speed_limit))

    # your settings
    if now - self._set_changed_at < SET_SPEED_WINDOW and i.set_speed:
      gap = i.set_speed - i.v_ego_disp
      if self._set_dir > 0 and going and gap > SET_SPEED_GAP:
        out.append((Msg.SET_SPEED_UP, i.set_speed))
      elif self._set_dir < 0 and i.accel_cmd < -0.1 and gap < -SET_SPEED_GAP:
        out.append((Msg.SET_SPEED_DOWN, i.set_speed))
    if now - self._personality_changed_at < PERSONALITY_WINDOW and i.lead_present:
      if self._personality_dir > 0 and i.accel_cmd < -0.1:
        out.append((Msg.MORE_ROOM, 0))
      elif self._personality_dir < 0 and going:
        out.append((Msg.LESS_ROOM, 0))

    # car ahead gone or getting away
    if (not i.lead_present and self._was_following and self._lead_lost_at is not None and
        ROAD_CLEAR_WINDOW[0] < now - self._lead_lost_at < ROAD_CLEAR_WINDOW[1] and
        going and i.set_speed and i.set_speed - i.v_ego_disp > ROAD_CLEAR_GAP):
      out.append((Msg.ROAD_CLEAR, i.set_speed))
    if (i.lead_present and not stopped and i.accel_cmd > PULL_AWAY_ACCEL and
        (i.lead_v_rel > PULL_AWAY_LEAD or i.lead_a > 0.5)):
      out.append((Msg.PULLING_AWAY, 0))
    return out

  def update(self, i: NarratorInputs, now: float) -> Msg:
    dt = 0.0 if self._prev_t is None else now - self._prev_t
    self._prev_t = now

    self._track_settings(i, now)
    cut_in, merging = self._track_lead(i, now, dt)
    if cut_in:
      self._cut_in_until = now + MIN_HOLD[Msg.CUT_IN]
    gate = self._track_gate(i, dt)

    conds = self._conditions(i, now, merging, gate)
    cond, value = max(conds, key=lambda c: c[0]) if conds else (Msg.NONE, 0)

    # a condition has to hold briefly before it's shown
    if cond != self._candidate:
      self._candidate = cond
      self._candidate_since = now
    confirmed = cond if now - self._candidate_since >= CONFIRM[cond] else Msg.NONE

    held = self.msg != Msg.NONE and now - self._shown_at < MIN_HOLD[self.msg]
    if confirmed != Msg.NONE and confirmed != self.msg and (not held or confirmed > self.msg):
      self.msg, self.value = confirmed, value
      self._shown_at = now
    elif confirmed == Msg.NONE and not held:
      self.msg, self.value = Msg.NONE, 0
    elif confirmed == self.msg:
      self.value = value
      self._shown_at = max(self._shown_at, now - MIN_HOLD[self.msg] + 0.5)  # linger 0.5 s after it ends
    return self.msg


# ---- helpers for pulling inputs out of messages ----

def path_offset(lead_d: float, lead_y_rel: float, path_x, path_y) -> float:
  """Lead's lateral distance from our planned path (m). radar yRel is left-positive; the model frame is right-positive."""
  if len(path_x) < 2:
    return -lead_y_rel
  return float(-lead_y_rel - np.interp(lead_d, path_x, path_y))


def model_stop_dist(vel_x, pos_x, stop_v: float = 0.3) -> float | None:
  """Distance along the plan at which the model's planned speed first reaches ~0."""
  for v, x in zip(vel_x, pos_x, strict=False):
    if v < stop_v:
      return float(x)
  return None


def plan_max_lat_accel(vel_x, yaw_rate, max_t_idx: int = 25) -> float:
  """Sharpest planned lateral accel over roughly the next 6 s of the model's plan."""
  n = min(len(vel_x), len(yaw_rate), max_t_idx)
  if n == 0:
    return 0.0
  return float(max(abs(v * w) for v, w in zip(vel_x[:n], yaw_rate[:n], strict=True)))


def fill_text(msg: Msg, value: int) -> tuple[str, str]:
  a, b = TEXT[msg]
  return a.format(n=value), b.format(n=value)


class DriveNarrator(Widget):
  def __init__(self):
    super().__init__()
    self._logic = NarratorLogic()
    self._shown: tuple[str, str, tuple] | None = None  # (title, sub, accent) on screen, kept through the fade-out
    self._time_caption: tuple[str, str] | None = None
    self._time_caption_until = 0.0
    self._alpha = 0.0
    self._enabled = True
    self._planted = False
    self._voice = True
    self._voice_dir = VOICES[1][0]
    self._greeting = True
    self._volume = 80
    self._param_checked = -1.0
    self._speech = SpeechTrigger()
    self._say_id = 0
    self._greeted = False
    self._loaded_at: float | None = None
    self._time_on = True
    self._weather_on = True
    # the fetch thread saves with its own Params handle
    self._weather = WeatherFetcher(load=lambda: ui_state.params.get(CACHE_PARAM), save=lambda c: Params().put(CACHE_PARAM, c))
    self._weather_started = False
    self._traffic = StopAndGo()
    self._last_said_at = -1e9
    self._time_announced: tuple[int, int, int] | None = None  # (day, hour, quarter) already announced

  def _inputs(self) -> NarratorInputs:
    sm = ui_state.sm
    cs, cc, ss = sm['carState'], sm['carControl'], sm['selfdriveState']
    plan, model = sm['longitudinalPlan'], sm['modelV2']
    lead = sm['radarState'].leadOne
    present = bool(sm.alive['radarState'] and lead.present)
    model_ok = sm.alive['modelV2']
    speed_conv = 3.6 if ui_state.is_metric else 2.236936

    offset = 0.0
    if present and model_ok:
      offset = path_offset(lead.dRel, lead.yRel, model.position.x, model.position.y)

    src = plan.longitudinalPlanSource
    Src = log.LongitudinalPlan.LongitudinalPlanSource
    plan_source = "lead" if src in (Src.lead0, Src.lead1, Src.lead2) else "e2e" if src == Src.e2e else "cruise"

    v_cruise = cs.vCruiseCluster if cs.vCruiseCluster != 0.0 else sm['controlsState'].deprecated.vCruise  # km/h
    set_speed = 0 if not (0 < v_cruise < 250) else round(v_cruise if ui_state.is_metric else v_cruise * 0.621371)

    # sunnypilot extras, when present
    scc_curve = speed_limit_adapting = green = False
    speed_limit = 0
    if sm.alive.get('longitudinalPlanSP', False):
      sp = sm['longitudinalPlanSP']
      SpSrc = custom.LongitudinalPlanSP.LongitudinalPlanSource
      scc_curve = sp.longitudinalPlanSource in (SpSrc.sccVision, SpSrc.sccMap)
      speed_limit_adapting = (sp.longitudinalPlanSource == SpSrc.speedLimitAssist or
                              sp.speedLimit.assist.state == custom.LongitudinalPlanSP.SpeedLimit.AssistState.adapting)
      speed_limit = round(sp.speedLimit.resolver.speedLimitFinal * speed_conv)
      green = bool(sp.e2eAlerts.greenLightAlert)

    stop_sign = False
    if sm.updated.get('customReservedRawData1', False):
      raw = bytes(sm['customReservedRawData1'])
      stop_sign = len(raw) >= 2 and raw[0] == 2 and raw[1] == 121  # honda_objectd v2, TSR code for a stop sign

    dm = sm['driverMonitoringState']
    vision_dm = dm.activePolicy == log.DriverMonitoringState.MonitoringPolicy.vision
    lat = sm['controlsState'].lateralControlState
    which = lat.which()
    steer_request = float(lat.pidState.steeringAngleDesiredDeg) if which == 'pidState' else 0.0

    vel_x = list(model.velocity.x) if model_ok else []
    yaw = list(model.orientationRate.z) if model_ok else []
    probs = list(model.laneLineProbs) if model_ok else []
    return NarratorInputs(
      engaged=bool(ss.enabled),
      long_active=bool(ui_state.has_longitudinal_control and cc.longActive),
      has_long=bool(ui_state.has_longitudinal_control),
      lat_active=bool(cc.latActive),
      v_ego=float(cs.vEgo),
      v_ego_disp=float(cs.vEgo) * speed_conv,
      standstill=bool(cs.standstill),
      accel_cmd=float(cc.actuators.accel),
      should_stop=bool(sm.alive['longitudinalPlan'] and plan.shouldStop),
      lead_present=present,
      lead_d=float(lead.dRel),
      lead_v=float(lead.vLead),
      lead_v_rel=float(lead.vRel),
      lead_a=float(lead.aLeadK),
      lead_offset=offset,
      plan_source=plan_source,
      model_stop_dist=model_stop_dist(vel_x, model.position.x) if model_ok else None,
      plan_max_lat_accel=plan_max_lat_accel(vel_x, yaw),
      cur_lat_accel=abs(float(cs.vEgo) * yaw[0]) if yaw else 0.0,
      scc_curve=scc_curve,
      speed_limit_adapting=speed_limit_adapting,
      speed_limit=speed_limit,
      set_speed=set_speed,
      green_light=green,
      stop_sign=stop_sign,
      personality=int(ss.personality.raw),
      dm_distracted=bool(vision_dm and dm.visionPolicyState.isDistracted),
      dm_awareness=int(dm.visionPolicyState.awarenessPercent) if vision_dm else 100,
      dm_alert=int(dm.alertLevel.raw),
      lane_probs=(float(probs[1]), float(probs[2])) if len(probs) >= 3 else (1.0, 1.0),
      planted_on=self._planted,
      steer_request_deg=steer_request,
    )

  def _say(self, parts: list[str]) -> None:
    """Hand a line to soundd, which plays it on the narration output (see audio_routing.py)."""
    self._say_id += 1
    self._last_said_at = rl.get_time()
    ui_state.params.put("DriveNarrationSay", {"id": f"{time.monotonic_ns()}-{self._say_id}", "time": time.monotonic(),
                                              "voice": self._voice_dir, "volume": self._volume, "parts": parts})

  def preview(self, volume: int) -> None:
    """Test line for the onroad volume bar, at the new level right away (soundd only rereads the setting every 2 s)."""
    self._volume = volume
    voice_id = int(ui_state.params.get("DriveNarrationVoiceId", return_default=True) or 0)
    self._voice_dir = VOICES[voice_id][0] if 0 <= voice_id < len(VOICES) else VOICES[1][0]
    self._say(spoken_parts(Msg.SLOWING))

  def _update_greeting(self, now: float) -> None:
    """Speak one greeting per onroad session, shortly after openpilot has loaded."""
    sm = ui_state.sm
    if not ui_state.started:
      self._greeted, self._loaded_at, self._weather_started = False, None, False
      return
    if self._greeted or not (self._voice and self._greeting):
      return
    if self._weather_on and not self._weather_started:  # fetch while openpilot is still loading
      self._weather_started = True
      self._weather.start(position_from_param(ui_state.params.get("LastGPSPositionLLK")))
    loaded = sm.recv_frame['carControl'] >= ui_state.started_frame and sm.alive['modelV2']
    if not loaded:
      return
    if self._loaded_at is None:
      self._loaded_at = now
    elif now - self._loaded_at >= GREETING_DELAY:
      if self._weather_on and self._weather.busy and now - self._loaded_at < GREETING_DELAY + WEATHER_WAIT:
        return
      self._greeted = True
      t = datetime.now(GREETING_TZ)
      greeting = random.choice(greeting_choices(t.hour))
      parts = [greeting]
      params = ui_state.params
      name = ""
      if params.get_bool("DriveNarrationGreetName"):
        name = resolve_name(int(params.get("DriveNarrationNameId", return_default=True) or 0), params.get("DriveNarrationName") or "")
      if name:
        if spoken := name_clip(name):
          parts.insert(0, spoken)
        self._time_caption = greeting_caption(greeting, name)  # shown even when the name has no clip
        self._time_caption_until = now + GREETING_CAPTION_S
      if self._weather_on and (w := self._weather.result(ui_state.params.get_bool("IsMetric"))):
        parts += weather_parts(w.temp, w.code, w.is_day, w.rain_chance)
      if self._time_on:  # chime + "It's 7:42 A.M." to close the greeting; counts as this quarter hour's time announcement
        parts += time_parts(t.hour, t.minute)
        self._time_announced = (t.toordinal(), t.hour, t.minute // 15)
      self._say(parts)

  def _update_time(self, now: float, i: NarratorInputs, in_traffic: bool) -> None:
    """In stop-and-go traffic, say the exact time once per quarter hour while stopped behind a car, in a quiet moment
    (no recent line, no openpilot alert, after the greeting)."""
    if not (self._voice and self._time_on and self._greeted):
      return
    stopped_behind_car = (i.standstill or i.v_ego < StopAndGo.STOP_SPEED) and i.lead_present
    if not (in_traffic and stopped_behind_car):
      return
    t = datetime.now(GREETING_TZ)
    key = (t.toordinal(), t.hour, t.minute // 15)
    if key == self._time_announced or now - self._last_said_at < TIME_QUIET_S:
      return
    if ui_state.sm['selfdriveState'].alertSize != log.SelfdriveState.AlertSize.none:
      return
    self._time_announced = key
    self._say(time_parts(t.hour, t.minute))
    self._time_caption = ("It's", time_caption(t.hour, t.minute))
    self._time_caption_until = now + TIME_CAPTION_S

  def _render(self, rect: rl.Rectangle) -> None:
    now = rl.get_time()
    if now - self._param_checked > 1.0:
      self._param_checked = now
      params = ui_state.params
      self._enabled = params.get_bool(NARRATION_PARAM)
      self._planted = params.get_bool(PLANTED_STEERING_PARAM)
      self._voice = params.get_bool("DriveNarrationVoice")
      self._greeting = params.get_bool("DriveNarrationGreeting")
      self._time_on = params.get_bool("DriveNarrationTime")
      self._weather_on = params.get_bool("DriveNarrationWeather")
      self._volume = int(params.get("DriveNarrationVolume", return_default=True) or 0)
      voice_id = int(params.get("DriveNarrationVoiceId", return_default=True) or 0)
      self._voice_dir = VOICES[voice_id][0] if 0 <= voice_id < len(VOICES) else VOICES[1][0]

    self._update_greeting(now)

    sm = ui_state.sm
    if not self._enabled or not ui_state.started or sm.recv_frame['carControl'] < ui_state.started_frame:
      self._logic.reset()
      self._speech = SpeechTrigger()
      self._traffic = StopAndGo()
      self._alpha = 0.0
      return

    inputs = self._inputs()
    msg = self._logic.update(inputs, now)
    in_traffic = self._traffic.update(inputs.v_ego, now)
    self._update_time(now, inputs, in_traffic)
    # stay out from under openpilot's own alerts
    if sm['selfdriveState'].alertSize != log.SelfdriveState.AlertSize.none:
      msg = Msg.NONE

    if self._voice and (parts := self._speech.update(msg, self._logic.value, now, in_traffic)):
      self._say(parts)

    visible = msg != Msg.NONE
    time_up = (self._time_caption is not None and now < self._time_caption_until and
               sm['selfdriveState'].alertSize == log.SelfdriveState.AlertSize.none)
    if time_up and msg in (Msg.NONE, Msg.WAITING):  # the time is said while waiting in traffic: show it over "Waiting for"
      self._shown, visible = (*self._time_caption, TIME_ACCENT), True
    elif visible:
      self._shown = (*fill_text(msg, self._logic.value), ACCENT[msg])
    target = 1.0 if visible else 0.0
    self._alpha += (target - self._alpha) * (0.35 if target else 0.15)
    if self._alpha < 0.02 or self._shown is None:
      return
    self._draw(rect, *self._shown, self._alpha)

  @staticmethod
  def _draw(rect: rl.Rectangle, title: str, sub: str, accent: tuple, alpha: float) -> None:
    font = gui_app.font(FontWeight.DISPLAY)
    font_sub = gui_app.font(FontWeight.SEMI_BOLD)
    sep = "  ·  "

    # one line in the strip between the speed/date row and the PRNDL block; shrink if it won't fit
    size = 22
    widths = lambda sz: (measure_text_cached(font, title, sz, 0), measure_text_cached(font_sub, sep, sz, 0),  # noqa: E731
                         measure_text_cached(font_sub, sub, sz, 0))
    t_ext, sep_ext, s_ext = widths(size)
    pad_x, pad_y = 14, 4
    max_w = rect.width * 0.92
    total = t_ext.x + sep_ext.x + s_ext.x
    if total + pad_x * 2 > max_w:
      size = max(14, math.floor(size * (max_w - pad_x * 2) / total))
      t_ext, sep_ext, s_ext = widths(size)
      total = t_ext.x + sep_ext.x + s_ext.x

    w, h = total + pad_x * 2, t_ext.y + pad_y * 2
    cx = rect.x + rect.width / 2
    box = rl.Rectangle(cx - w / 2, rect.y + 110, w, h)

    a = lambda k: int(255 * alpha * k)  # noqa: E731
    rl.draw_rectangle_rounded(box, 0.5, 8, rl.Color(0, 0, 0, a(0.8)))
    rl.draw_rectangle_rounded_lines_ex(box, 0.5, 8, 2.0, rl.Color(*accent, a(0.9)))
    x, y = box.x + pad_x, box.y + pad_y
    rl.draw_text_ex(font, title, rl.Vector2(x, y), size, 0, rl.Color(255, 255, 255, a(1.0)))
    x += t_ext.x
    rl.draw_text_ex(font_sub, sep, rl.Vector2(x, y), size, 0, rl.Color(160, 165, 170, a(1.0)))
    x += sep_ext.x
    rl.draw_text_ex(font_sub, sub, rl.Vector2(x, y), size, 0, rl.Color(*accent, a(1.0)))
