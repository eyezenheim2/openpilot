"""Stop-line assist: firmer, earlier braking when experimental mode stops for a light or stop line.

In experimental mode the driving model decides where to stop (modelV2.velocity reaching ~0 at some distance).
Drive logs showed it commits late and plans gentle stops (peak ~-1.9 m/s2), so the car can roll past the line.
This aims the stop a bit before the model's stopping point and asks for the deceleration that needs.

Only ever adds braking, never throttle. Only acts when:
  - experimental mode is on and the model's own plan comes to a stop within its horizon
  - there is no lead car near or before that stop (lead stops stay with the normal planner)
  - the car is still moving faster than a crawl
The extra braking is capped (A_MIN) and jerk-limited, and turns off with the StopLineAssist param.
"""
import numpy as np

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog

STOP_V = 0.3  # m/s: model speed treated as stopped
STOP_FRACTION = 0.80  # aim for 80% of the model's stopping distance ...
STOP_MARGIN = 2.0  # ... minus this many metres
MIN_TARGET_DIST = 0.5  # m
MIN_SPEED = 1.0  # m/s: below this the normal stopping logic finishes the stop
ENGAGE_DECEL = -0.6  # m/s2: only step in once the required decel is at least this firm (avoid early creeping brakes)
A_MIN = -3.0  # m/s2: firmest the assist will ask for (car limit is -3.5)
JERK = 2.5  # m/s3: how fast the extra braking may build
RELEASE_JERK = 4.0  # m/s3: how fast it may let go
LEAD_CLEARANCE = 5.0  # m: a lead within stop point + this means the stop is for the lead, not a line


class StopLineAssist:
  def __init__(self, dt: float):
    self.dt = dt
    self.params = Params()
    self.enabled = self.params.get_bool("StopLineAssist")
    self.counter = 0
    self.a_assist = 0.0  # extra braking currently applied (<= 0), jerk-limited
    self.active = False
    self.failed = False  # any unexpected error switches the assist off for the drive; planning continues untouched

  def _target(self, sm, v_ego: float) -> float | None:
    """Deceleration needed to stop before the line, or None if the assist should not act."""
    if not sm['selfdriveState'].experimentalMode or v_ego < MIN_SPEED:
      return None
    model = sm['modelV2']
    v = np.asarray(model.velocity.x, dtype=np.float64)
    x = np.asarray(model.position.x, dtype=np.float64)
    if v.size == 0 or x.size != v.size:
      return None
    stopped = np.nonzero(v < STOP_V)[0]
    if stopped.size == 0:
      return None  # the model is not planning to stop
    stop_x = float(x[stopped[0]])

    lead = sm['radarState'].leadOne
    if lead.present and lead.dRel < stop_x + LEAD_CLEARANCE:
      return None  # stopping for a car, not a line

    target = max(MIN_TARGET_DIST, STOP_FRACTION * stop_x - STOP_MARGIN)
    a_req = -v_ego * v_ego / (2.0 * target)
    if a_req > ENGAGE_DECEL:
      return None
    return max(a_req, A_MIN)

  def update(self, sm, v_ego: float, a_e2e: float) -> float:
    """Returns the e2e acceleration with the assist applied (never higher than a_e2e)."""
    if self.counter % 50 == 0:
      self.enabled = self.params.get_bool("StopLineAssist")
    self.counter += 1

    target = None
    if self.enabled and not self.failed:
      try:
        target = self._target(sm, v_ego)
      except Exception:
        cloudlog.exception("stop_line_assist: disabled after an error")
        self.failed = True
    # extra braking wanted beyond what the model already asks for
    want = min(0.0, target - a_e2e) if target is not None else 0.0
    # jerk-limit the extra braking both ways so it builds and releases smoothly
    self.a_assist = float(np.clip(want, self.a_assist - JERK * self.dt, self.a_assist + RELEASE_JERK * self.dt))
    self.a_assist = min(0.0, self.a_assist)
    self.active = self.a_assist < -0.05
    if not self.active:
      return a_e2e
    # never less braking than the model asked for, and the assist itself never goes past A_MIN
    return min(a_e2e, max(A_MIN, a_e2e + self.a_assist))
