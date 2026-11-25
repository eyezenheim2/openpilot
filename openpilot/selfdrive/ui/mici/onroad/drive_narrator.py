"""Drive narration: short captions that say what openpilot is doing about the car ahead.

  "Car ahead slowing / I'm slowing down"
  "Car cutting in / Making room"
  "Car merging in / Watching it"
  ...

The decision part (NarratorLogic) only looks at plain numbers so it can be tested and replayed
off-device; DriveNarrator pulls those numbers out of the UI's SubMaster and draws the caption.
Captions only show while openpilot is in control of gas/brake, since "I'm slowing down" would
be a lie under stock ACC.
"""
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pyray as rl

from openpilot.cereal import log
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget

NARRATION_PARAM = "DriveNarration"


class Msg(IntEnum):
  # higher value wins when several apply at once
  NONE = 0
  WAITING = 1
  GOING = 2
  SLOWING = 3
  STOPPING = 4
  MERGING = 5
  CUT_IN = 6


TEXT = {
  Msg.WAITING: ("Waiting for", "car ahead"),
  Msg.GOING: ("Car ahead moving", "Going"),
  Msg.SLOWING: ("Car ahead slowing", "I'm slowing down"),
  Msg.STOPPING: ("Car ahead stopped", "I'm stopping"),
  Msg.MERGING: ("Car merging in", "Watching it"),
  Msg.CUT_IN: ("Car cutting in", "Making room"),
}

# (r, g, b) accent per message: calm white/blue for routine, amber for other cars doing something
ACCENT = {
  Msg.WAITING: (200, 210, 225),
  Msg.GOING: (0, 230, 90),
  Msg.SLOWING: (90, 170, 255),
  Msg.STOPPING: (255, 90, 70),
  Msg.MERGING: (255, 190, 40),
  Msg.CUT_IN: (255, 150, 30),
}

# minimum time a caption stays up once shown, so it can be read
MIN_HOLD = {
  Msg.WAITING: 1.5, Msg.GOING: 2.0, Msg.SLOWING: 1.5,
  Msg.STOPPING: 1.5, Msg.MERGING: 2.0, Msg.CUT_IN: 3.0,
}
CONFIRM_TIME = 0.3  # s a condition must hold before its caption appears (stops flicker)

# thresholds
SLOW_ACCEL = -0.4  # m/s^2 commanded decel that counts as "slowing down"
CLOSING_SPEED = -0.3  # m/s, vRel below this means we're closing on the lead
LEAD_BRAKING = -0.6  # m/s^2 lead accel that counts as "car ahead slowing"
STOPPED_LEAD_SPEED = 1.0  # m/s
WAITING_MAX_DIST = 15.0  # m
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


@dataclass
class NarratorInputs:
  long_active: bool = False
  v_ego: float = 0.0
  standstill: bool = False
  accel_cmd: float = 0.0  # m/s^2 openpilot is asking for
  should_stop: bool = False
  lead_present: bool = False
  lead_d: float = 0.0  # m
  lead_v: float = 0.0  # m/s absolute
  lead_v_rel: float = 0.0  # m/s, negative = closing
  lead_a: float = 0.0  # m/s^2
  lead_offset: float = 0.0  # m from our planned path at the lead's distance, either side


class NarratorLogic:
  def __init__(self):
    self.reset()

  def reset(self):
    self.msg = Msg.NONE
    self._shown_at = 0.0
    self._candidate = Msg.NONE
    self._candidate_since = 0.0
    self._prev_t: float | None = None
    self._prev_lead_present = False
    self._prev_lead_d = 0.0
    self._lead_lost_at: float | None = 0.0  # None while a lead is present
    self._prev_offset: float | None = None
    self._lat_rate = 0.0
    self._waiting = False
    self._cut_in_until = 0.0

  def _track_lead(self, i: NarratorInputs, now: float) -> tuple[bool, bool]:
    """Returns (cut_in_event, merging) from how the lead changed since last frame."""
    dt = 0.0 if self._prev_t is None else now - self._prev_t
    self._prev_t = now

    cut_in = False
    if i.lead_present and i.long_active and i.v_ego > CUT_IN_MIN_SPEED:
      close = i.lead_d < i.v_ego * CUT_IN_HEADWAY
      appeared = not self._prev_lead_present and self._lead_lost_at is not None and now - self._lead_lost_at > CUT_IN_ABSENT
      replaced = self._prev_lead_present and self._prev_lead_d - i.lead_d > max(CUT_IN_JUMP, 0.3 * self._prev_lead_d)
      cut_in = close and (appeared or replaced)

    merging = False
    if i.lead_present and self._prev_lead_present and self._prev_offset is not None and 0.0 < dt < 0.5:
      rate = (i.lead_offset - self._prev_offset) / dt
      k = dt / (LAT_RATE_TAU + dt)
      self._lat_rate += k * (rate - self._lat_rate)
      inward = -self._lat_rate * np.sign(i.lead_offset)  # positive when heading toward our path
      merging = (i.v_ego > MERGE_MIN_SPEED and i.lead_d < MERGE_MAX_DIST and
                 MERGE_MIN_OFFSET < abs(i.lead_offset) < MERGE_MAX_OFFSET and inward > MERGE_INWARD_SPEED)
    else:
      self._lat_rate = 0.0

    if i.lead_present:
      self._lead_lost_at = None
    elif self._lead_lost_at is None:
      self._lead_lost_at = now
    self._prev_lead_present = i.lead_present
    self._prev_lead_d = i.lead_d
    self._prev_offset = i.lead_offset if i.lead_present else None
    return cut_in, merging

  def _condition(self, i: NarratorInputs, now: float, merging: bool) -> Msg:
    if not i.long_active or not i.lead_present:
      self._waiting = False
      return Msg.NONE

    if now < self._cut_in_until:
      return Msg.CUT_IN
    if merging:
      return Msg.MERGING

    braking = i.accel_cmd < SLOW_ACCEL or i.should_stop
    if i.standstill or i.v_ego < 0.3:
      if i.lead_d < WAITING_MAX_DIST:
        if self._waiting and i.lead_v > STOPPED_LEAD_SPEED and i.accel_cmd > 0.2:
          return Msg.GOING
        self._waiting = True
        return Msg.WAITING
      return Msg.NONE

    if self._waiting and i.accel_cmd > 0.2 and i.lead_v > STOPPED_LEAD_SPEED:
      return Msg.GOING  # just pulled away behind the lead
    self._waiting = False

    if braking and i.lead_v < STOPPED_LEAD_SPEED:
      return Msg.STOPPING
    if braking and (i.lead_v_rel < CLOSING_SPEED or i.lead_a < LEAD_BRAKING):
      return Msg.SLOWING
    return Msg.NONE

  def update(self, i: NarratorInputs, now: float) -> Msg:
    cut_in, merging = self._track_lead(i, now)
    if cut_in:
      self._cut_in_until = now + MIN_HOLD[Msg.CUT_IN]

    cond = self._condition(i, now, merging)

    # a condition has to hold briefly before it's shown (cut-ins are already an event, show at once)
    if cond != self._candidate:
      self._candidate = cond
      self._candidate_since = now
    confirmed = cond if (cond == Msg.CUT_IN or now - self._candidate_since >= CONFIRM_TIME) else Msg.NONE

    held = self.msg != Msg.NONE and now - self._shown_at < MIN_HOLD[self.msg]
    if confirmed != Msg.NONE and confirmed != self.msg and (not held or confirmed > self.msg):
      self.msg = confirmed
      self._shown_at = now
    elif confirmed == Msg.NONE and not held:
      self.msg = Msg.NONE
    elif confirmed == self.msg:
      self._shown_at = max(self._shown_at, now - MIN_HOLD[self.msg] + 0.5)  # linger 0.5 s after it ends
    return self.msg


def path_offset(lead_d: float, lead_y_rel: float, path_x, path_y) -> float:
  """Lead's lateral distance from our planned path (m). radar yRel is left-positive; the model frame is right-positive."""
  if len(path_x) < 2:
    return -lead_y_rel
  return float(-lead_y_rel - np.interp(lead_d, path_x, path_y))


class DriveNarrator(Widget):
  def __init__(self):
    super().__init__()
    self._logic = NarratorLogic()
    self._shown = Msg.NONE  # what's on screen, kept through the fade-out
    self._alpha = 0.0
    self._enabled = True
    self._param_checked = -1.0

  def _inputs(self) -> NarratorInputs:
    sm = ui_state.sm
    cs, cc = sm['carState'], sm['carControl']
    plan = sm['longitudinalPlan']
    lead = sm['radarState'].leadOne
    present = bool(sm.alive['radarState'] and lead.present)
    offset = 0.0
    if present and sm.alive['modelV2']:
      pos = sm['modelV2'].position
      offset = path_offset(lead.dRel, lead.yRel, pos.x, pos.y)
    return NarratorInputs(
      long_active=bool(ui_state.has_longitudinal_control and cc.longActive),
      v_ego=float(cs.vEgo),
      standstill=bool(cs.standstill),
      accel_cmd=float(cc.actuators.accel),
      should_stop=bool(sm.alive['longitudinalPlan'] and plan.shouldStop),
      lead_present=present,
      lead_d=float(lead.dRel),
      lead_v=float(lead.vLead),
      lead_v_rel=float(lead.vRel),
      lead_a=float(lead.aLeadK),
      lead_offset=offset,
    )

  def _render(self, rect: rl.Rectangle) -> None:
    now = rl.get_time()
    if now - self._param_checked > 1.0:
      self._param_checked = now
      self._enabled = ui_state.params.get_bool(NARRATION_PARAM)

    sm = ui_state.sm
    if not self._enabled or not ui_state.started or sm.recv_frame['carControl'] < ui_state.started_frame:
      self._logic.reset()
      self._alpha = 0.0
      return

    msg = self._logic.update(self._inputs(), now)
    # stay out from under openpilot's own alerts
    if sm['selfdriveState'].alertSize != log.SelfdriveState.AlertSize.none:
      msg = Msg.NONE

    if msg != Msg.NONE:
      self._shown = msg
    target = 1.0 if msg != Msg.NONE else 0.0
    self._alpha += (target - self._alpha) * (0.35 if target else 0.15)
    if self._alpha < 0.02 or self._shown == Msg.NONE:
      return
    self._draw(rect, self._shown, self._alpha)

  @staticmethod
  def _draw(rect: rl.Rectangle, msg: Msg, alpha: float) -> None:
    title, sub = TEXT[msg]
    accent = ACCENT[msg]
    font = gui_app.font(FontWeight.DISPLAY)
    font_sub = gui_app.font(FontWeight.SEMI_BOLD)
    sep = "  \u00b7  "

    # one line in the strip between the speed/date row and the PRNDL block; shrink if it won't fit
    size = 22
    widths = lambda sz: (measure_text_cached(font, title, sz, 0), measure_text_cached(font_sub, sep, sz, 0),  # noqa: E731
                         measure_text_cached(font_sub, sub, sz, 0))
    t_ext, sep_ext, s_ext = widths(size)
    pad_x, pad_y = 14, 4
    max_w = rect.width * 0.92
    total = t_ext.x + sep_ext.x + s_ext.x
    if total + pad_x * 2 > max_w:
      size = max(14, int(size * (max_w - pad_x * 2) / total))
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
