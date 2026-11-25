from openpilot.selfdrive.ui.mici.onroad.drive_narrator import Msg, NarratorInputs, NarratorLogic

DT = 0.05  # 20 Hz UI


def run(frames):
  """frames: list of (duration_s, NarratorInputs). Returns the list of (t, msg) whenever the caption changes."""
  logic, t, out = NarratorLogic(), 0.0, []
  for duration, inp in frames:
    for _ in range(round(duration / DT)):
      msg = logic.update(inp, t)
      if not out or out[-1][1] != msg:
        out.append((round(t, 2), msg))
      t += DT
  return out


def shown(out):
  return [m for _, m in out if m != Msg.NONE]


def follow(**kw):
  base = {'long_active': True, 'v_ego': 25.0, 'lead_present': True, 'lead_d': 35.0, 'lead_v': 25.0}
  base.update(kw)
  return NarratorInputs(**base)


def test_steady_following_is_quiet():
  assert shown(run([(10.0, follow())])) == []


def test_not_in_control_is_quiet():
  assert shown(run([(5.0, follow(long_active=False, accel_cmd=-2.0, lead_v_rel=-5.0))])) == []


def test_lead_braking_says_slowing():
  out = run([(2.0, follow()), (3.0, follow(accel_cmd=-1.5, lead_v_rel=-3.0, lead_a=-2.0)), (4.0, follow())])
  assert shown(out) == [Msg.SLOWING]
  assert out[-1][1] == Msg.NONE  # clears once the braking is over


def test_brief_blip_does_not_flicker():
  assert shown(run([(2.0, follow()), (0.1, follow(accel_cmd=-1.0, lead_v_rel=-1.0)), (2.0, follow())])) == []


def test_cut_in_from_no_lead():
  out = run([(3.0, follow(lead_present=False)), (0.5, follow(lead_d=15.0, lead_v=22.0, lead_v_rel=-3.0)),
             (4.0, follow(lead_d=15.0, accel_cmd=-1.0, lead_v_rel=-3.0))])
  assert shown(out)[0] == Msg.CUT_IN


def test_cut_in_in_front_of_existing_lead():
  out = run([(2.0, follow(lead_d=45.0)), (4.0, follow(lead_d=14.0, lead_v=22.0, lead_v_rel=-3.0, accel_cmd=-1.0))])
  assert shown(out)[0] == Msg.CUT_IN


def test_radar_dropout_is_not_a_cut_in():
  out = run([(2.0, follow(lead_d=30.0)), (0.2, follow(lead_present=False)), (3.0, follow(lead_d=30.0))])
  assert shown(out) == []


def test_merge_anticipation():
  # car 2.5 m to the side drifting in at ~0.6 m/s
  frames = [(1.0, follow(lead_d=30.0, lead_offset=2.5))]
  offset = 2.5
  for _ in range(30):
    offset -= 0.6 * DT
    frames.append((DT, follow(lead_d=30.0, lead_offset=offset)))
  assert Msg.MERGING in shown(run(frames))


def test_parallel_car_is_quiet():
  assert shown(run([(5.0, follow(lead_offset=2.0))])) == []


def test_stop_and_go():
  out = run([
    (2.0, follow(v_ego=10.0, lead_d=20.0, lead_v=0.0, lead_v_rel=-10.0, accel_cmd=-2.0)),  # stopping for it
    (4.0, follow(v_ego=0.0, standstill=True, lead_d=5.0, lead_v=0.0, accel_cmd=-0.5)),  # waiting
    (2.0, follow(v_ego=0.0, standstill=True, lead_d=6.0, lead_v=2.0, accel_cmd=0.8)),  # lead pulls away
  ])
  assert shown(out) == [Msg.STOPPING, Msg.WAITING, Msg.GOING]
