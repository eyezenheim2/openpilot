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


# ---- lead-source check ----

def test_cruise_braking_with_lead_far_is_not_blamed_on_lead():
  # planner source says cruise (e.g. braking to a lower set speed) while a lead happens to be closing far ahead
  out = run([(3.0, follow(plan_source="cruise", lead_d=90.0, accel_cmd=-1.0, lead_v_rel=-2.0))])
  assert Msg.SLOWING not in shown(out)


def test_experimental_mode_close_closing_lead_counts():
  out = run([(3.0, follow(plan_source="e2e", lead_d=30.0, accel_cmd=-1.0, lead_v_rel=-3.0))])
  assert shown(out) == [Msg.SLOWING]


def test_experimental_mode_distant_lead_does_not_count():
  out = run([(3.0, follow(plan_source="e2e", lead_d=120.0, accel_cmd=-1.0, lead_v_rel=-3.0))])
  assert Msg.SLOWING not in shown(out)


# ---- stops and lights ----

def no_lead(**kw):
  base = {'long_active': True, 'v_ego': 12.0, 'v_ego_disp': 27.0, 'plan_source': "e2e", 'set_speed': 35}
  base.update(kw)
  return NarratorInputs(**base)


def test_stop_ahead_light_or_line():
  assert shown(run([(3.0, no_lead(accel_cmd=-1.5, model_stop_dist=40.0))])) == [Msg.STOP_AHEAD]


def test_stop_ahead_needs_experimental_plan():
  assert shown(run([(3.0, no_lead(plan_source="cruise", accel_cmd=-1.5, model_stop_dist=40.0))])) == []


def test_stop_sign_then_full_stop_go():
  out = run([
    (0.1, no_lead(stop_sign=True, accel_cmd=-1.5, model_stop_dist=30.0)),
    (2.5, no_lead(accel_cmd=-1.5, model_stop_dist=20.0)),
    (2.0, no_lead(v_ego=0.0, v_ego_disp=0.0, standstill=True, accel_cmd=-1.0)),
    (2.0, no_lead(v_ego=1.0, v_ego_disp=2.0, accel_cmd=1.0)),
  ])
  assert shown(out) == [Msg.STOP_SIGN, Msg.SIGN_GO]


def test_green_light():
  out = run([(2.0, no_lead(v_ego=0.0, standstill=True, accel_cmd=-0.5)),
             (0.1, no_lead(v_ego=0.0, standstill=True, accel_cmd=0.8, green_light=True)),
             (2.0, no_lead(v_ego=1.5, accel_cmd=0.8))])
  assert shown(out) == [Msg.GREEN_LIGHT]


# ---- curves and speed ----

def test_curve_ahead():
  assert shown(run([(3.0, no_lead(accel_cmd=-0.6, plan_max_lat_accel=2.5, cur_lat_accel=0.3))])) == [Msg.CURVE]


def test_already_in_the_curve_is_quiet():
  assert shown(run([(3.0, no_lead(accel_cmd=-0.6, plan_max_lat_accel=2.5, cur_lat_accel=2.3))])) == []


def test_sunnypilot_curve_control():
  assert shown(run([(3.0, no_lead(plan_source="cruise", accel_cmd=-0.6, scc_curve=True))])) == [Msg.CURVE]


def test_speed_limit():
  out = run([(3.0, no_lead(plan_source="cruise", accel_cmd=-0.6, speed_limit_adapting=True, speed_limit=45))])
  assert shown(out) == [Msg.SPEED_LIMIT]


def test_set_speed_up_and_value():
  logic, t = NarratorLogic(), 0.0
  for inp, dur in ((no_lead(plan_source="cruise", set_speed=55, v_ego_disp=55), 1.0),
                   (no_lead(plan_source="cruise", set_speed=65, v_ego_disp=55, accel_cmd=0.8), 1.0)):
    for _ in range(round(dur / DT)):
      logic.update(inp, t)
      t += DT
  assert logic.msg == Msg.SET_SPEED_UP and logic.value == 65


def test_set_speed_down():
  out = run([(1.0, no_lead(plan_source="cruise", set_speed=65, v_ego_disp=65)),
             (2.0, no_lead(plan_source="cruise", set_speed=55, v_ego_disp=64, accel_cmd=-0.5))])
  assert shown(out) == [Msg.SET_SPEED_DOWN]


def test_road_clear():
  out = run([(3.0, follow(v_ego_disp=45.0, set_speed=65)),
             (3.0, follow(lead_present=False, v_ego_disp=46.0, set_speed=65, accel_cmd=0.8, plan_source="cruise"))])
  assert shown(out) == [Msg.ROAD_CLEAR]


def test_follow_distance_change():
  out = run([(1.0, follow(personality=1)), (3.0, follow(personality=2, accel_cmd=-0.4))])
  assert shown(out) == [Msg.MORE_ROOM]
  out = run([(1.0, follow(personality=1)), (3.0, follow(personality=0, accel_cmd=0.5))])
  assert shown(out) == [Msg.LESS_ROOM]


def test_lead_pulling_away():
  assert shown(run([(3.0, follow(accel_cmd=0.6, lead_v_rel=2.0))])) == [Msg.PULLING_AWAY]


# ---- driver, steering ----

def test_gas_override():
  assert shown(run([(2.0, follow(long_active=False, gas_pressed=True))])) == [Msg.GAS_OVERRIDE]


def test_eyes_on_road_before_openpilot_alert():
  assert shown(run([(2.0, follow(dm_distracted=True, dm_awareness=80))])) == [Msg.EYES]
  # openpilot's own alert has taken over
  assert shown(run([(2.0, follow(dm_distracted=True, dm_awareness=55, dm_alert=1))])) == []


def test_steer_limit():
  assert shown(run([(2.0, follow(lat_active=True, steer_output=0.92))])) == [Msg.STEER_LIMIT]
  assert shown(run([(2.0, follow(lat_active=True, steer_output=0.6))])) == []


def test_lanes_faded_needs_to_persist():
  assert shown(run([(1.0, follow(lat_active=True, lane_probs=(0.1, 0.2)))])) == []
  assert shown(run([(3.0, follow(lat_active=True, lane_probs=(0.1, 0.2)))])) == [Msg.LANES_FADED]


def planted_drive(wiggle_deg, v=5.0, base=1.0, dur=4.0):
  frames, t = [], 0.0
  while t < dur:
    req = base + wiggle_deg * (1 if int(t / 0.25) % 2 else -1)
    frames.append((DT, NarratorInputs(engaged=True, lat_active=True, planted_on=True, v_ego=v, steer_request_deg=req)))
    t += DT
  return run(frames)


def test_planted_steering_gate():
  assert shown(planted_drive(3.0)) == [Msg.PLANTED]
  assert shown(planted_drive(0.5)) == []  # nothing to hold
  assert shown(planted_drive(3.0, v=15.0)) == []  # gate only works at low speed
  assert shown(planted_drive(3.0, base=20.0)) == []  # turning, not wobbling


def test_disengaged_is_quiet():
  assert shown(run([(3.0, follow(engaged=False, accel_cmd=-2.0, lead_v_rel=-3.0, gas_pressed=True))])) == []


def test_priority_cut_in_over_slowing():
  out = run([(2.0, follow(lead_d=45.0)), (6.0, follow(lead_d=14.0, lead_v=22.0, lead_v_rel=-3.0, accel_cmd=-1.0))])
  assert shown(out)[0] == Msg.CUT_IN and Msg.SLOWING in shown(out)


def test_every_message_has_text_and_fits():
  from openpilot.selfdrive.ui.mici.onroad.drive_narrator import ACCENT, TEXT, fill_text
  for m in Msg:
    if m == Msg.NONE:
      continue
    assert m in TEXT and m in ACCENT
    a, b = fill_text(m, 65)
    assert len(a) + len(b) <= 40, (a, b)
