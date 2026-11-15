import math
import time

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


def _scale_gain(gain, scale):
  bp, v = gain
  return list(bp), [x * scale for x in v]


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
    self._update_planted()

  def _reset_planted_filters(self):
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
      angle_steers_des_no_offset = self._desired_angle.update(angle_request_no_offset)
    else:
      angle_steers_des_no_offset = angle_request_no_offset
    angle_steers_des = angle_steers_des_no_offset + params.angleOffsetDeg
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
        self._desired_rate.update((angle_steers_des - self._last_angle_des) / self.dt)
      self._last_angle_des = angle_steers_des
      error_rate = self._desired_rate.x - self._wheel_rate.update(CS.steeringRateDeg)

      # offset does not contribute to resistive torque
      ff = self.ff_factor * self.get_steer_feedforward(angle_steers_des_no_offset, CS.vEgo)
      freeze_integrator = steer_limited_by_safety or CS.steeringPressed or CS.vEgo < 5

      output_torque = self.pid.update(error,
                                error_rate=error_rate,
                                feedforward=ff,
                                speed=CS.vEgo,
                                freeze_integrator=freeze_integrator)

      pid_log.active = True
      pid_log.p = float(self.pid.p)
      pid_log.i = float(self.pid.i)
      pid_log.f = float(self.pid.f)
      pid_log.output = float(output_torque)
      pid_log.saturated = bool(self._check_saturation(self.steer_max - abs(output_torque) < 1e-3, CS, steer_limited_by_safety, curvature_limited))

    return output_torque, angle_steers_des, pid_log
