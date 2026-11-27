from openpilot.selfdrive.controls.lib.latcontrol_pid import (OVERRIDE_FLOOR, OversteerGate, UNWIND_MIN_ANGLE,
                                                              unwinding)

DT = 0.01


def run_gate(frames, gate=None):
  """frames: (seconds, output_torque, driver_torque, pressed). Returns (gate, last gated output)."""
  gate = gate or OversteerGate(DT)
  out = 0.0
  for seconds, torque, driver, pressed in frames:
    for _ in range(round(seconds / DT)):
      out = gate.update(torque, driver, pressed)
  return gate, out


def test_no_driver_full_torque():
  _, out = run_gate([(2.0, 1.0, 0.0, False)])
  assert out == 1.0


def test_driver_holding_against_fades_to_floor():
  _, out = run_gate([(1.0, -1.0, 1200.0, True)])
  assert abs(out - (-OVERRIDE_FLOOR)) < 0.01


def test_driver_pushing_same_way_is_not_gated():
  _, out = run_gate([(1.0, -1.0, -1200.0, True)])
  assert out == -1.0


def test_light_touch_below_threshold_is_not_gated():
  _, out = run_gate([(1.0, 1.0, -300.0, False)])
  assert out == 1.0


def test_comes_back_fast_when_openpilot_turns_to_agree():
  gate, _ = run_gate([(1.0, 1.0, -1500.0, True)])  # driver unwinding against openpilot
  _, out = run_gate([(0.15, -1.0, -1500.0, True)], gate)  # openpilot reverses to help
  assert out < -0.9


def test_comes_back_slowly_when_driver_lets_go():
  gate, _ = run_gate([(1.0, 1.0, -1500.0, True)])
  _, out = run_gate([(0.15, 1.0, 0.0, False)], gate)
  assert out < 0.8  # no snap back to full torque the instant the hands come off
  _, out = run_gate([(2.0, 1.0, 0.0, False)], gate)
  assert out > 0.98


def test_unwinding():
  assert unwinding(60.0, 80.0) and unwinding(-60.0, -80.0)  # heading back toward center from inside a turn
  assert not unwinding(90.0, 80.0)  # still turning in
  assert not unwinding(-5.0, 20.0)  # crossed to the other side: normal smoothing
  assert not unwinding(2.0, UNWIND_MIN_ANGLE - 1.0)  # small angles on a straight keep the wobble smoothing


def run_wind_in(requests, wheel, v=7.0):
  """requests: list of (seconds, start_deg, end_deg) ramps; the target is taken as the request. Returns scales."""
  from openpilot.selfdrive.controls.lib.latcontrol_pid import WindInStop
  stop, scales = WindInStop(DT), []
  for seconds, a, b in requests:
    n = round(seconds / DT)
    for k in range(n):
      r = a + (b - a) * k / n
      scales.append(stop.update(r, r, wheel, v))
  return scales


def test_wind_in_stop_untouched_while_turning_in():
  assert all(s == 1.0 for s in run_wind_in([(2.0, 0., 140.)], wheel=70.))


def test_wind_in_stop_follows_the_request_down_once_the_turn_peaks():
  # request climbs to 140 deg, then eases off at 60 deg/s while the wheel (70 deg) is behind it
  scales = run_wind_in([(2.0, 0., 140.), (1.0, 140., 80.)], wheel=70.)
  latched = [s for s in scales[200:] if s < 1.0]
  assert latched and abs(latched[0] * 127.0 - 70.) < 10.  # the target starts near the wheel, not above it


def test_wind_in_stop_ignores_mid_turn_wiggle():
  frames = [(1.5, 0., 120.)] + [(0.25, 120., 117.), (0.25, 117., 120.)] * 8
  assert all(s == 1.0 for s in run_wind_in(frames, wheel=70.))


def test_wind_in_stop_gives_full_request_back_if_turn_tightens():
  scales = run_wind_in([(2.0, 0., 140.), (1.0, 140., 90.), (1.0, 90., 140.), (2.0, 140., 140.)], wheel=70.)
  assert min(scales) < 0.9 and scales[-1] > 0.99


def test_wind_in_stop_off_when_keeping_up_or_fast():
  assert min(run_wind_in([(2.0, 0., 140.), (1.0, 140., 80.)], wheel=139.)) > 0.95  # wheel kept up: ratio ~1
  assert all(s == 1.0 for s in run_wind_in([(2.0, 0., 140.), (1.0, 140., 80.)], wheel=70., v=15.))


def test_wind_in_stop_ignores_a_quick_dip_that_comes_back():
  # slowing for a stop: the request dips 14 deg in 0.1 s and comes straight back (seg 1 of route 252)
  scales = run_wind_in([(1.5, 0., 76.), (0.1, 76., 62.), (0.4, 62., 72.), (1.0, 72., 72.)], wheel=25.)
  assert all(s == 1.0 for s in scales)
