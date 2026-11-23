"""Stop-line assist: firmer, earlier braking when Experimental Mode stops for a light, stop line or stop sign.

In Experimental Mode the driving model decides where to stop. Drive logs and video showed two problems:
  - red lights: the model's stopping point can be a few metres past the line (stopped on the crosswalk at night)
  - stop signs: the model plans a rolling stop, slowing to ~2 m/s and driving on, never planning zero speed
This aims ~5 m short of the model's stopping point and, at stop signs, turns the rolling stop into a full stop.

Two triggers, both only with no lead car at or before the stop:
  - full stop: the model's planned speed reaches ~0 within its horizon (lights, lines)
  - rolling stop (unless CaliforniaStop is on): the model plans to slow below ROLL_V close ahead while we are at moderate speed,
    AND the car's own Traffic Sign Recognition reported a stop sign within SIGN_MEMORY (fresh messages only).
    Without that confirmation (other cars, decoder not running, stale data) no stop-sign stop happens.
    Once committed, the stop point is latched by odometry and held until standstill + HOLD_TIME, because the
    model plans to accelerate through and would otherwise pull the car along.

Only ever adds braking, never throttle: the output is min(model accel, assist command). The assist command is capped
(speed-dependent, -3.0 m/s2 at speed easing to -1.8 near standstill) and jerk-limited. Gas, a lead appearing before the
stop, leaving Experimental Mode or disengaging cancels it. Any internal error disables it for the drive.
"""
import numpy as np

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog

STOP_V = 0.3  # m/s: model speed treated as a planned full stop
STOP_FRACTION = 0.90  # aim for 90% of the model's stopping distance ...
STOP_MARGIN = 4.5  # ... minus this many metres: converges ~5 m short of the model's stop. Drive video (night, faint
                   # line) showed the model's stop point 3-5 m too far: the car stopped on the crosswalk.
MIN_TARGET_DIST = 0.3  # m
MIN_SPEED = 1.0  # m/s: below this the normal stopping logic finishes a full-stop plan
ENGAGE_DECEL = -0.6  # m/s2: full-stop trigger only steps in once the required decel is at least this firm

# rolling stops (stop signs): the model slows below ROLL_V close ahead and drives on
ROLL_V = 2.6  # m/s: planned minimum speed that counts as a rolling stop
ROLL_MAX_DIST = 35.0  # m: only when the slow point is this close (red lights commit to a full stop before this)
ROLL_MIN_DIST = 2.0  # m
ROLL_SPEED = (3.0, 12.0)  # m/s: our speed range for the rolling-stop trigger
HOLD_TIME = 1.0  # s at standstill before handing back to the model
COOLDOWN_DIST = 15.0  # m to travel after a stop-sign stop before another can trigger
STANDSTILL_V = 0.15  # m/s
HOLD_ACCEL = -1.0  # m/s2 while holding at the stop
OVERSHOOT_RELEASE = 3.0  # m: if we are this far past the latched stop and still moving, let go (don't stop mid-intersection)
LATCH_TIMEOUT = 10.0  # s: a latched stop-sign stop never lasts longer than this

A_MIN = -3.0  # m/s2: firmest the assist will ask for (car limit is -3.5)
# ...and less near the end of the stop, so the extra braking happens early rather than as a jab at walking pace
A_MIN_BP_V = [2.0, 4.0, 8.0]  # m/s
A_MIN_V = [-1.8, -2.2, A_MIN]  # m/s2
JERK = 2.5  # m/s3: how fast the assist command may firm up
RELEASE_JERK = 4.0  # m/s3: how fast it may let go
LEAD_CLEARANCE = 5.0  # m: a lead within stop point + this means the stop is for the lead, not a line
TSR_SERVICE = 'customReservedRawData1'  # honda_objectd: header (version=2, tsr_code) + tracked objects
TSR_VERSION, TSR_STOP = 2, 121  # car's Traffic Sign Recognition code for a stop sign
SIGN_MEMORY = 3.0  # s: a stop sign recognised this recently confirms a rolling stop as a stop sign
SIGN_COMFORT_DECEL = 2.2  # m/s2: if the sign is confirmed late, stop a little closer (never past the slow point) rather than brake harder
SIGN_SLOW_POINT_GAP = 1.0  # m: the stop is always at least this far before the model's slow point


def _floor(v_ego: float) -> float:
  return float(np.interp(v_ego, A_MIN_BP_V, A_MIN_V))


class StopLineAssist:
  def __init__(self, dt: float):
    self.dt = dt
    self.params = Params()
    self.enabled = self.params.get_bool("StopLineAssist")
    self.sign_enabled = not self.params.get_bool("CaliforniaStop")
    self.counter = 0
    self.failed = False  # any unexpected error switches the assist off for the drive; planning continues untouched

    self.odo = 0.0
    self.cmd = 0.0  # assist's own acceleration command while active
    self.active = False
    self.sign_stop_at: float | None = None  # odometer position of a latched stop-sign stop
    self.hold_t = 0.0
    self.latch_t = 0.0
    self.cooldown_until = -1.0
    self.t = 0.0
    self.sign_seen_t = -1e9
    self.tsr_mono = 0

  # ------------------------------------------------------------ triggers
  @staticmethod
  def _plan(sm):
    model = sm['modelV2']
    v = np.asarray(model.velocity.x, dtype=np.float64)
    x = np.asarray(model.position.x, dtype=np.float64)
    if v.size == 0 or x.size != v.size:
      return None, None
    return v, x

  @staticmethod
  def _lead_before(sm, dist: float) -> bool:
    lead = sm['radarState'].leadOne
    return bool(lead.present and lead.dRel < dist + LEAD_CLEARANCE)

  def _full_stop_target(self, sm, v_ego: float, v, x) -> float | None:
    """Decel to stop before a light/line the model already plans to stop at."""
    if v_ego < MIN_SPEED:
      return None
    stopped = np.nonzero(v < STOP_V)[0]
    if stopped.size == 0:
      return None
    stop_x = float(x[stopped[0]])
    if self._lead_before(sm, stop_x):
      return None
    target = max(MIN_TARGET_DIST, STOP_FRACTION * stop_x - STOP_MARGIN)
    a_req = -v_ego * v_ego / (2.0 * target)
    if a_req > ENGAGE_DECEL:
      return None
    return max(a_req, _floor(v_ego))

  def _rolling_stop_dist(self, sm, v_ego: float, v, x) -> float | None:
    """Distance to stop at if the model is planning a rolling stop (stop sign), else None."""
    if not (ROLL_SPEED[0] < v_ego < ROLL_SPEED[1]) or self.odo < self.cooldown_until:
      return None
    i = int(np.argmin(v))
    if v[i] >= ROLL_V or v[i] < STOP_V:  # a planned full stop is handled by the full-stop trigger
      return None
    slow_x = float(x[i])
    if not (ROLL_MIN_DIST < slow_x < ROLL_MAX_DIST) or self._lead_before(sm, slow_x):
      return None
    dist = max(MIN_TARGET_DIST, STOP_FRACTION * slow_x - STOP_MARGIN)
    # confirmed late (close to the sign): ease the stop point toward the slow point instead of braking harder
    comfortable = v_ego * v_ego / (2.0 * SIGN_COMFORT_DECEL)
    return max(dist, min(comfortable, slow_x - SIGN_SLOW_POINT_GAP))

  def _update_sign(self, sm):
    """Remember when a *fresh* message last reported a stop sign; stale data can't confirm anything."""
    mono = sm.logMonoTime[TSR_SERVICE] if hasattr(sm, 'logMonoTime') else 0
    if mono == 0 or mono == self.tsr_mono:
      return
    self.tsr_mono = mono
    raw = bytes(sm[TSR_SERVICE])
    if len(raw) >= 2 and raw[0] == TSR_VERSION and raw[1] == TSR_STOP:
      self.sign_seen_t = self.t

  def _sign_confirmed(self) -> bool:
    return self.t - self.sign_seen_t < SIGN_MEMORY

  # ------------------------------------------------------------ update
  def _target(self, sm, v_ego: float) -> float | None:
    ss = sm['selfdriveState']
    if not ss.experimentalMode or not ss.enabled or sm['carState'].gasPressed:
      self.sign_stop_at = None
      return None
    v, x = self._plan(sm)
    if v is None:
      return None

    # latched stop-sign stop: brake to the latched point, hold briefly, then hand back to the model
    if self.sign_stop_at is not None:
      remaining = self.sign_stop_at - self.odo
      self.latch_t += self.dt
      overshot = remaining < -OVERSHOOT_RELEASE and v_ego > STANDSTILL_V
      if self._lead_before(sm, max(remaining, 0.0)) or overshot or self.latch_t > LATCH_TIMEOUT:
        self.sign_stop_at = None
        self.cooldown_until = self.odo + COOLDOWN_DIST
        return None
      if v_ego < STANDSTILL_V:
        self.hold_t += self.dt
        if self.hold_t >= HOLD_TIME:
          self.sign_stop_at = None
          self.cooldown_until = self.odo + COOLDOWN_DIST
          return None
        return HOLD_ACCEL
      self.hold_t = 0.0
      return max(-v_ego * v_ego / (2.0 * max(remaining, MIN_TARGET_DIST)), _floor(v_ego))

    full = self._full_stop_target(sm, v_ego, v, x)
    if full is not None:
      return full

    if self.sign_enabled and self._sign_confirmed():
      dist = self._rolling_stop_dist(sm, v_ego, v, x)
      if dist is not None:
        self.sign_stop_at = self.odo + dist
        self.hold_t = self.latch_t = 0.0
        return max(-v_ego * v_ego / (2.0 * dist), _floor(v_ego))
    return None

  def update(self, sm, v_ego: float, a_e2e: float) -> float:
    """Returns the e2e acceleration with the assist applied (never higher than a_e2e)."""
    if self.counter % 50 == 0:
      self.enabled = self.params.get_bool("StopLineAssist")
      self.sign_enabled = not self.params.get_bool("CaliforniaStop")
    self.counter += 1
    self.t += self.dt
    self.odo += max(v_ego, 0.0) * self.dt

    target = None
    if self.enabled and not self.failed:
      try:
        self._update_sign(sm)
        target = self._target(sm, v_ego)
      except Exception:
        cloudlog.exception("stop_line_assist: disabled after an error")
        self.failed = True
        self.sign_stop_at = None

    if target is not None:
      if not self.active:
        self.active, self.cmd = True, a_e2e  # start from where the model is, then firm up smoothly
      self.cmd = float(np.clip(target, self.cmd - JERK * self.dt, self.cmd + RELEASE_JERK * self.dt))
    elif self.active:
      self.cmd += RELEASE_JERK * self.dt  # let go smoothly
      if self.cmd >= a_e2e:
        self.active = False

    if not self.active:
      return a_e2e
    self.cmd = max(self.cmd, A_MIN)
    return min(a_e2e, self.cmd)  # never less braking than the model asked for
