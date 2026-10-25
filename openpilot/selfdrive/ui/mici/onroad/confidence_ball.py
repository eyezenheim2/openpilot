import math
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import pyray as rl
from openpilot.selfdrive.ui.ui_state import ui_state, UIStatus
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.common.filter_simple import FirstOrderFilter

LEAD_TUNNEL_SPEED = 0.45
LEAD_TUNNEL_BRIGHTNESS = 0.38
CLOCK_TIMEZONE = ZoneInfo('America/Los_Angeles')
# DATE DISPLAY - sizes are fractions of HUD height; position is relative to HUD.
DATE_NUMBER_SIZE = 0.21
DATE_WEEKDAY_SIZE = 0.08
DATE_CENTER_X = 0.86
DATE_CENTER_Y = 0.82
DATE_LINE_GAP = 0.01


class ConfidenceBall(Widget):
  def __init__(self, demo: bool = False):
    Widget.__init__(self)
    self._demo = demo
    self._confidence_filter = FirstOrderFilter(-0.5, 0.5, 1 / gui_app.target_fps)

  @staticmethod
  def get_animate_status_probs():
    if ui_state.status == UIStatus.LAT_ONLY:
      return ui_state.sm['modelV2'].meta.disengagePredictions.steerOverrideProbs
    return ui_state.sm['modelV2'].meta.disengagePredictions.brakeDisengageProbs

  @staticmethod
  def get_lat_long_dot_color():
    if ui_state.status == UIStatus.LAT_ONLY:
      return rl.Color(0x00, 0xC8, 0xC8, 0xFF)
    return rl.Color(0x96, 0x1C, 0xA8, 0xFF)

  def update_filter(self, value: float):
    self._confidence_filter.update(value)

  def randomize_face(self) -> None:
    """Compatibility with existing callers; the tunnel has a fixed design."""
    pass

  def draw_hud_face(self, rect: rl.Rectangle) -> None:
    """Continuous onroad triangle tunnel, underneath HUD and alerts."""
    if not ui_state.started or rect.width <= 0 or rect.height <= 0:
      return
    self._draw_face_background(rect)
    phase = (rl.get_time() * LEAD_TUNNEL_SPEED) % 1.0
    sm = ui_state.sm
    face_detected = (sm.all_checks(["driverMonitoringState"]) and
                     sm.recv_frame["driverMonitoringState"] > ui_state.started_frame and
                     sm["driverMonitoringState"].visionPolicyState.faceDetected)
    self._draw_lead_tunnel(rect, phase, 1.0, colorful=face_detected)
    timestamp = time.time()
    self._draw_date(rect, timestamp)

  def _draw_face_background(self, rect: rl.Rectangle):
    rl.draw_rectangle_rec(rect, rl.BLACK)

  def release_face_texture(self):
    pass

  @staticmethod
  def _draw_lead_tunnel(panel: rl.Rectangle, phase: float, alpha: float, colorful: bool = False) -> None:
    """Perspective triangle outlines; clipped without altering the caller's scissor."""
    if alpha <= 0:
      return
    unit = min(panel.width, panel.height)
    inset = unit * 0.045
    left, right = panel.x + inset, panel.x + panel.width - inset
    top, bottom = panel.y + inset, panel.y + panel.height - inset
    cx, cy = panel.x + panel.width / 2, panel.y + panel.height / 2

    def clipped_line(a, b, width, color):
      dx, dy = b[0] - a[0], b[1] - a[1]
      lo, hi = 0.0, 1.0
      for p, q in ((-dx, a[0]-left), (dx, right-a[0]), (-dy, a[1]-top), (dy, bottom-a[1])):
        if abs(p) < 1e-9:
          if q < 0:
            return
        elif p < 0:
          lo = max(lo, q/p)
        else:
          hi = min(hi, q/p)
        if lo > hi:
          return
      rl.draw_line_ex(rl.Vector2(a[0]+lo*dx, a[1]+lo*dy),
                      rl.Vector2(a[0]+hi*dx, a[1]+hi*dy), width, color)

    # Geometric spacing expands continuously toward the viewer. Fade both ends
    # of the depth range so recycled rings do not pop at the loop boundary.
    count = 22
    for i in range(count):
      depth = (i + phase) / count
      radius = unit * 0.045 * (45.0 ** depth)
      edge_fade = min(1.0, depth * 10, (1-depth) * 10)
      opacity = int(255 * LEAD_TUNNEL_BRIGHTNESS * alpha * edge_fade * (0.25 + 0.75*depth))
      width = max(1.0, unit * 0.005 * depth)
      for inverted in (True, False):
        angle = math.pi / 2 if inverted else -math.pi / 2
        points = [(cx + radius*math.cos(angle+j*math.tau/3),
                   cy + radius*math.sin(angle+j*math.tau/3)) for j in range(3)]
        color = rl.Color(180, 185, 190, opacity // 4) if inverted else rl.Color(235, 240, 245, opacity)
        if colorful:
          # Rainbow colors travel with the rings; inverted triangles use complementary hues.
          hue = (depth * 360.0 + (180.0 if inverted else 0.0)) % 360.0
          color = rl.color_from_hsv(hue, 0.85, 1.0)
          color.a = opacity // 4 if inverted else opacity
        for j in range(3):
          clipped_line(points[j], points[(j+1)%3], width, color)

  @staticmethod
  def _draw_date(rect: rl.Rectangle, timestamp: float) -> None:
    if rect.width <= 0 or rect.height <= 0:
      return
    local = datetime.fromtimestamp(timestamp, CLOCK_TIMEZONE)
    weekday = ('MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN')[local.weekday()]
    number_size = rect.height * DATE_NUMBER_SIZE
    weekday_size = rect.height * DATE_WEEKDAY_SIZE
    gap = rect.height * DATE_LINE_GAP
    center_x = rect.x + rect.width * DATE_CENTER_X
    y = rect.y + rect.height * DATE_CENTER_Y - (number_size + gap + weekday_size) / 2
    for label, size, weight in ((str(local.day), number_size, FontWeight.DISPLAY),
                                 (weekday, weekday_size, FontWeight.SEMI_BOLD)):
      font = gui_app.font(weight)
      measured = rl.measure_text_ex(font, label, size, 0)
      pos = rl.Vector2(center_x - measured.x / 2, y)
      rl.draw_text_ex(font, label, rl.Vector2(pos.x + 2, pos.y + 2), size, 0, rl.BLACK)
      rl.draw_text_ex(font, label, pos, size, 0, rl.WHITE)
      y += size + gap

  def _update_state(self):
    if self._demo:
      return

    # animate status dot in from bottom
    if ui_state.status == UIStatus.DISENGAGED:
      self._confidence_filter.update(-0.5)
    elif ui_state.status in (UIStatus.LAT_ONLY, UIStatus.LONG_ONLY):
      self._confidence_filter.update(1 - max(self.get_animate_status_probs() or [1]))
    else:
      self._confidence_filter.update((1 - max(ui_state.sm['modelV2'].meta.disengagePredictions.brakeDisengageProbs or [1])) *
                                                        (1 - max(ui_state.sm['modelV2'].meta.disengagePredictions.steerOverrideProbs or [1])))

  def _confidence_color(self) -> rl.Color:
    """Match the original ball's top color, including non-engaged UI states."""
    if ui_state.status == UIStatus.ENGAGED or self._demo:
      if self._confidence_filter.x > 0.5:
        return rl.Color(0,255,204,255)
      if self._confidence_filter.x > 0.2:
        return rl.Color(255,200,0,255)
      return rl.Color(255,0,21,255)
    if ui_state.status in (UIStatus.LAT_ONLY, UIStatus.LONG_ONLY):
      return self.get_lat_long_dot_color()
    if ui_state.status == UIStatus.OVERRIDE:
      return rl.Color(255,255,255,255)
    return rl.Color(50,50,50,255)

  def _render(self, _):
    # Widget.render still updates confidence; the former side ball is hidden.
    pass
