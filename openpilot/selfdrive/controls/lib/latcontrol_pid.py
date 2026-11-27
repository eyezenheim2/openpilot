import math
import time

import numpy as np

from openpilot.cereal import log
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.common.pid import PIDController

# "Planted steering": a firmer variant of the car's stock PID tune, applied on top of whatever the
# car port sets so it works for every PID car. Each piece targets one "floaty" symptom:
#   integral up        -> holds the lane center instead of slowly drifting off it
#   proportional up    -> corrects small errors sooner
#   feedforward down   -> settles into curves instead of turning in too hard and being pulled back
#                         (logs showed stock*1.3 feedforward was ~2.3-2.5x the torque curves needed,
#                         with the integrator constantly fighting it)
#   desired smoothing  -> the wheel stops chasing the model's small ~1 Hz twitches, so it holds
#                         its line on straights
#   damping (D) term   -> resists steering-wheel motion the plan didn't ask for (crosswind, ruts,
#                         road crown). The wheel rate it uses is filtered: the raw rate signal is
#                         coarse (whole deg/s), and an unfiltered D turned that into torque flutter
PLANTED_STEERING_PARAM = "PlantedSteering"
PLANTED_KP_SCALE = 1.2
PLANTED_KI_SCALE = 1.5
PLANTED_KF_SCALE = 0.6  # was 1.3
PLANTED_KD_SECONDS = 0.05  # damping gain as a time constant: kd = kp * this. 0 = no damping
PLANTED_DESIRED_FILTER_RC = 0.15  # s, low-pass on the desired steering angle. 0 = off
PLANTED_RATE_FILTER_RC = 0.1  # s, smooths the desired-angle rate used by the damping term
PLANTED_WHEEL_RATE_FILTER_RC = 0.15  # s, smooths the measured wheel rate used by the damping term
PARAM_POLL_INTERVAL = 1.0  # s

# "Hands on the wheel" wobble gate. At low speed the model's request wiggles +-2-4 deg at under 1 Hz,
# and the Honda torque ramp (0 -> full in 0.33 s) can't reverse the wheel fast enough to follow it,
# so the wheel ends up late, overshoots, the model steers against the swing, and it wobbles.
# On a straight, low-speed road the gate holds the wheel at the request's ~1 s average and
# ignores wiggle inside a deadband. Anything bigger (a real correction, a turn starting) passes
# through, minus the deadband, while the average catches up. The gate fades out with speed and
# as soon as the average says the road is bending. steeringAngleDesiredDeg still logs the raw request;
# angleError is measured against the gated + smoothed target. Replayed on routes 593a234151 seg 18 and
# 162eed8477 seg 3: model wiggle in the wobble stretches +-3.0-3.8 -> +-1.1-1.4 deg, path vs the
# model's within 3 cm over any 2 s, gate mostly off through the tight turns.
GATE_SPEED_BP = [6.7, 9.0]  # m/s (15, 20 mph): full gate below, none above
GATE_STRAIGHT_BP = [5., 10.]  # deg, |average request|: full gate below, none above
GATE_DEADBAND = 3.0  # deg of steering wheel ignored around the average
GATE_AVG_RC = 1.0  # s, averaging time for the "where the road actually goes" estimate
GATE_WEIGHT_RC = 0.3  # s, smooths gate engage/release


# Oversteer gate. On slow, tight turns the model's request runs far past what the wheel reaches (route 5bb5f789ca seg 11:
# up to 147 deg asked, 80 deg reached), so P pins the output at full torque through the whole turn. Two things then feel
# like oversteer:
#   1. the driver holds the wheel against openpilot (wrong or early turn-in) and openpilot keeps pulling at 100%:
#      while the driver's torque opposes openpilot's, openpilot's torque fades down to OVERRIDE_FLOOR;
#   2. on the way out of a turn the 0.15 s desired smoothing lags the unwinding request by 10-15 deg, so the wheel is
#      still held in at full torque after the road straightens: while the request heads back toward center inside a
#      real turn, the smoothing drops to UNWIND_FILTER_RC. Straights keep the full smoothing (wobble handling unchanged).
OVERRIDE_FLOOR = 0.3  # share of openpilot's torque kept while the driver holds against it
OVERRIDE_ATTACK_RC = 0.1  # s, how fast the fade comes in once the driver opposes
OVERRIDE_RELEASE_RC = 0.5  # s, how fast full torque comes back after the driver lets go (no snap-back)
OVERRIDE_AGREE_RC = 0.05  # s, how fast it comes back when openpilot turns to push the same way as the driver
UNWIND_MIN_ANGLE = 10.0  # deg, |smoothed target| above which a request heading back to center counts as a turn exit
UNWIND_FILTER_RC = 0.03  # s, desired smoothing while unwinding out of a turn


class OversteerGate:
  def __init__(self, dt):
    self.dt = dt
    self.weight = 0.0  # 0 = full torque, 1 = faded to OVERRIDE_FLOOR

  def reset(self):
    self.weight = 0.0

  def update(self, output_torque: float, driver_torque: float, steering_pressed: bool) -> float:
    """Scale openpilot's torque down while the driver is pressing the wheel the other way."""
    opposing = steering_pressed and driver_torque * output_torque < 0.
    agreeing = steering_pressed and driver_torque * output_torque > 0.
    target = 1.0 if opposing else 0.0
    if target > self.weight:
      rc = OVERRIDE_ATTACK_RC
    else:
      rc = OVERRIDE_AGREE_RC if agreeing else OVERRIDE_RELEASE_RC
    self.weight += (target - self.weight) * self.dt / (rc + self.dt)
    return output_torque * (1. - self.weight * (1. - OVERRIDE_FLOOR))


def unwinding(request: float, smoothed: float) -> bool:
  """The request is heading back toward center from inside a real turn (same side, smaller than the smoothed target)."""
  return abs(smoothed) > UNWIND_MIN_ANGLE and request * smoothed >= 0. and abs(request) < abs(smoothed)


# Wind-in stop: prevent oversteer in the first place. On slow turns the model's steering command runs ~1.7x its own
# planned path (seg 11 of 5bb5f789ca: 147 deg asked, the wheel topped out near 80 at full torque), so the wheel is always
# behind the request and keeps winding in for ~2 s after the turn has peaked (66 -> 80 deg while the request was already
# falling). Once the request is clearly easing off (WINDIN_EASE_DEG below its peak for this turn and falling faster than
# WINDIN_EASE_RATE), the target stops being the request itself and becomes the request scaled by how far behind the wheel
# was at that moment (wheel / target, e.g. 66 / 141): the target starts right at the wheel, so nothing more winds in, and
# then comes down in step with the request, so the wheel unwinds with it instead of ~1 s after it. Self-calibrating: if
# the wheel is keeping up (ratio ~1) it changes nothing. It lets go when the turn ends, when the request tightens again
# by WINDIN_EASE_DEG (a turn that keeps tightening gets the full request back) or above WINDIN_SPEED_MAX.
WINDIN_SPEED_MAX = 11.0  # m/s (~25 mph): off above
WINDIN_MIN_REQUEST = 10.0  # deg, |request| below this isn't a turn
WINDIN_EASE_DEG = 10.0  # deg the request must have dropped from its peak in this turn (and rise again to let go)
WINDIN_EASE_RATE = 20.0  # deg/s the request must be heading back to center at
WINDIN_RATE_RC = 0.2  # s, smoothing on the request rate (the model's request wiggles a few degrees at ~1 Hz)
WINDIN_MIN_RATIO = 0.4  # never scale the request below this share
WINDIN_RELEASE_RC = 0.3  # s, how gently the scale returns to 1 when it lets go


class WindInStop:
  def __init__(self, dt):
    self.dt = dt
    self.reset()

  def reset(self):
    self.prev: float | None = None
    self.rate = 0.0
    self.peak = 0.0  # |request| peak in the current turn
    self.side = 0.0
    self.ratio: float | None = None  # latched wheel/target ratio while easing out of a turn
    self.low = 0.0  # |request| low since the ratio latched
    self.scale = 1.0

  def update(self, request: float, target: float, wheel: float, v_ego: float) -> float:
    """Scale (0..1) to apply to the smoothed target. 1 = untouched."""
    if self.prev is not None:
      self.rate += ((request - self.prev) / self.dt - self.rate) * self.dt / (WINDIN_RATE_RC + self.dt)
    self.prev = request

    side = math.copysign(1., request) if abs(request) >= WINDIN_MIN_REQUEST else 0.
    if side == 0. or side != self.side or v_ego >= WINDIN_SPEED_MAX:  # straight, a new turn the other way, or fast
      self.side, self.peak, self.ratio = side, abs(request), None
    else:
      self.peak = max(self.peak, abs(request))
      if self.ratio is None:
        easing = (self.peak - abs(request) >= WINDIN_EASE_DEG and request * self.rate < 0. and
                  abs(self.rate) >= WINDIN_EASE_RATE)
        behind = wheel * target > 0. and abs(target) > abs(wheel)
        if easing and behind:
          self.ratio = max(abs(wheel) / abs(target), WINDIN_MIN_RATIO)
          self.low = abs(request)
      else:
        self.low = min(self.low, abs(request))
        if abs(request) - self.low >= WINDIN_EASE_DEG:  # tightening again: give the full request back
          self.ratio, self.peak = None, abs(request)

    if self.ratio is not None:
      self.scale = self.ratio
    else:
      self.scale += (1. - self.scale) * self.dt / (WINDIN_RELEASE_RC + self.dt)
    return self.scale


def _scale_gain(gain, scale):
  bp, v = gain
  return list(bp), [x * scale for x in v]


class WobbleGate:
  def __init__(self, dt):
    self.avg = FirstOrderFilter(0.0, GATE_AVG_RC, dt, initialized=False)
    self.weight = FirstOrderFilter(0.0, GATE_WEIGHT_RC, dt)

  def reset(self):
    self.avg.initialized = False
    self.weight.x = 0.0

  def update(self, request, v_ego):
    avg = self.avg.update(request)
    dev = request - avg
    target = float(np.interp(v_ego, GATE_SPEED_BP, [1., 0.]) * np.interp(abs(avg), GATE_STRAIGHT_BP, [1., 0.]))
    w = self.weight.update(target)
    gated = avg + math.copysign(max(abs(dev) - GATE_DEADBAND, 0.), dev)
    return w * gated + (1. - w) * request, w


class LatControlPID(LatControl):
  def __init__(self, CP, CP_SP, CI, dt):
    super().__init__(CP, CP_SP, CI, dt)
    self._stock_kp = (list(CP.lateralTuning.pid.kpBP), list(CP.lateralTuning.pid.kpV))
    self._stock_ki = (list(CP.lateralTuning.pid.kiBP), list(CP.lateralTuning.pid.kiV))
    self._stock_kf = CP.lateralTuning.pid.kf
    self.pid = PIDController(self._stock_kp, self._stock_ki,
                             pos_limit=self.steer_max, neg_limit=-self.steer_max)
    self.ff_factor = self._stock_kf
    self.get_steer_feedforward = CI.get_steer_feedforward_function()

    self._params = Params()
    self._param_poll_time = 0.0
    self.planted: bool | None = None
    self._desired_angle = FirstOrderFilter(0.0, PLANTED_DESIRED_FILTER_RC, dt, initialized=False)
    self._desired_rate = FirstOrderFilter(0.0, PLANTED_RATE_FILTER_RC, dt)
    self._wheel_rate = FirstOrderFilter(0.0, PLANTED_WHEEL_RATE_FILTER_RC, dt, initialized=False)
    self._last_angle_des: float | None = None
    self._wobble_gate = WobbleGate(dt)
    self._oversteer_gate = OversteerGate(dt)
    self._wind_in_stop = WindInStop(dt)
    self._update_planted()

  def _reset_planted_filters(self):
    self._wobble_gate.reset()
    self._oversteer_gate.reset()
    self._wind_in_stop.reset()
    self._desired_angle.initialized = False
    self._wheel_rate.initialized = False
    self._desired_rate.x = 0.0
    self._last_angle_des = None

  def _update_planted(self):
    now = time.monotonic()
    if now < self._param_poll_time:
      return
    self._param_poll_time = now + PARAM_POLL_INTERVAL

    planted = self._params.get_bool(PLANTED_STEERING_PARAM)
    if planted == self.planted:
      return
    self.planted = planted
    self._reset_planted_filters()

    if planted:
      kp = _scale_gain(self._stock_kp, PLANTED_KP_SCALE)
      self.pid._k_p = kp
      self.pid._k_i = _scale_gain(self._stock_ki, PLANTED_KI_SCALE)
      self.pid._k_d = _scale_gain(kp, PLANTED_KD_SECONDS)
      self.ff_factor = self._stock_kf * PLANTED_KF_SCALE
    else:
      self.pid._k_p = self._stock_kp
      self.pid._k_i = self._stock_ki
      self.pid._k_d = ([0.], [0.])
      self.ff_factor = self._stock_kf

  def update(self, active, CS, VM, params, steer_limited_by_safety, desired_curvature, calibrated_pose, curvature_limited, lat_delay):
    self._update_planted()

    pid_log = log.ControlsState.LateralPIDState.new_message()
    pid_log.steeringAngleDeg = float(CS.steeringAngleDeg)
    pid_log.steeringRateDeg = float(CS.steeringRateDeg)

    # The angle the plan asks for, before any smoothing. Logged as steeringAngleDesiredDeg so logs
    # still show the raw request; angleError is measured against the (smoothed) target actually tracked.
    angle_request_no_offset = math.degrees(VM.get_steer_from_curvature(-desired_curvature, CS.vEgo, params.roll))
    pid_log.steeringAngleDesiredDeg = angle_request_no_offset + params.angleOffsetDeg

    if not active:
      self._reset_planted_filters()

    if self.planted and active and PLANTED_DESIRED_FILTER_RC > 0.:
      gated_request, _ = self._wobble_gate.update(angle_request_no_offset, CS.vEgo)
      exiting = self._desired_angle.initialized and unwinding(gated_request, self._desired_angle.x)
      self._desired_angle.update_alpha(UNWIND_FILTER_RC if exiting else PLANTED_DESIRED_FILTER_RC)
      angle_steers_des_no_offset = self._desired_angle.update(gated_request)
      rate_ref_no_offset = angle_steers_des_no_offset
      wheel_no_offset = CS.steeringAngleDeg - params.angleOffsetDeg
      # turn has peaked: stop winding in and bring the target down with the request (see WindInStop)
      angle_steers_des_no_offset *= self._wind_in_stop.update(angle_request_no_offset, angle_steers_des_no_offset,
                                                              wheel_no_offset, CS.vEgo)
    else:
      angle_steers_des_no_offset = angle_request_no_offset
      rate_ref_no_offset = angle_steers_des_no_offset
    angle_steers_des = angle_steers_des_no_offset + params.angleOffsetDeg
    # the damping term follows the smoothed request, not the wind-in hold, so engaging the hold doesn't kick the wheel
    rate_ref = rate_ref_no_offset + params.angleOffsetDeg
    error = angle_steers_des - CS.steeringAngleDeg
    pid_log.angleError = error

    if not active:
      output_torque = 0.0
      pid_log.active = False

    else:
      # Rate of the steering error, for the damping term: (desired rate) - (actual wheel rate).
      if self._last_angle_des is None:
        self._desired_rate.x = 0.0
      else:
        self._desired_rate.update((rate_ref - self._last_angle_des) / self.dt)
      self._last_angle_des = rate_ref
      error_rate = self._desired_rate.x - self._wheel_rate.update(CS.steeringRateDeg)

      # offset does not contribute to resistive torque
      ff = self.ff_factor * self.get_steer_feedforward(angle_steers_des_no_offset, CS.vEgo)
      freeze_integrator = steer_limited_by_safety or CS.steeringPressed or CS.vEgo < 5

      output_torque = self.pid.update(error,
                                error_rate=error_rate,
                                feedforward=ff,
                                speed=CS.vEgo,
                                freeze_integrator=freeze_integrator)
      if self.planted:
        output_torque = self._oversteer_gate.update(output_torque, CS.steeringTorque, CS.steeringPressed)

      pid_log.active = True
      pid_log.p = float(self.pid.p)
      pid_log.i = float(self.pid.i)
      pid_log.f = float(self.pid.f)
      pid_log.output = float(output_torque)
      pid_log.saturated = bool(self._check_saturation(self.steer_max - abs(output_torque) < 1e-3, CS, steer_limited_by_safety, curvature_limited))

    return output_torque, angle_steers_des, pid_log
