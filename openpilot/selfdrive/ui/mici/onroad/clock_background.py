import math
import time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import pyray as rl
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight


LEAD_TUNNEL_SPEED = 0.45
LEAD_TUNNEL_BRIGHTNESS = 0.38
CLOCK_TIMEZONE = ZoneInfo('America/Los_Angeles')
# DATE DISPLAY - sizes are fractions of HUD height; position is relative to HUD.
DATE_NUMBER_SIZE = 0.15
DATE_WEEKDAY_SIZE = 0.055
DATE_CENTER_X = 0.74
DATE_CENTER_Y = 0.635
DATE_LINE_GAP = 0.01
CLOCK_BACKGROUND = Path(__file__).with_name('contour_clock_face.png')


class ClockBackground:
  """Clock and animated triangle background ported from the mici onroad UI."""

  def __init__(self):
    self._face_texture = None
    self._face_texture_checked = False

  def draw_hud_face(self, rect: rl.Rectangle, draw_small_clock: bool = True) -> None:
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
    self._draw_date(rect, timestamp, draw_small_clock)

  def _draw_face_background(self, rect: rl.Rectangle):
    if not self._face_texture_checked:
      self._face_texture_checked = True
      if CLOCK_BACKGROUND.is_file():
        texture = rl.load_texture(str(CLOCK_BACKGROUND))
        if texture.id:
          self._face_texture = texture
          rl.set_texture_filter(texture, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
    rl.draw_rectangle_rec(rect, rl.BLACK)
    if self._face_texture is not None:
      texture = self._face_texture
      # Scale the complete landscape dial so no edge numerals are cropped.
      source = rl.Rectangle(0, 0, texture.width, texture.height)
      rl.draw_texture_pro(texture, source, rect, rl.Vector2(0, 0), 0, rl.WHITE)

  def release_face_texture(self):
    if self._face_texture is not None:
      rl.unload_texture(self._face_texture)
      self._face_texture = None
    self._face_texture_checked = False

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
  def _clock_angles(timestamp: float) -> tuple[float, float, float]:
    """Pacific time with automatic PST/PDT and smoothly moving hands."""
    local = datetime.fromtimestamp(timestamp, CLOCK_TIMEZONE)
    seconds = local.second + local.microsecond / 1_000_000
    minutes = local.minute + seconds / 60.0
    hours = local.hour % 12 + minutes / 60.0
    return tuple(value * math.tau - math.pi / 2 for value in
                 (hours / 12.0, minutes / 60.0, seconds / 60.0))

  @staticmethod
  def draw_clock_hands(center_x: float, center_y: float, width: float, height: float, timestamp: float) -> None:
    """Draw live white hour/minute hands and a red seconds needle."""
    if width <= 0 or height <= 0:
      return
    w, h = width, height
    cx, cy = center_x, center_y
    white = rl.Color(250, 250, 245, 255)
    red = rl.Color(244, 48, 40, 255)
    hour, minute, second = ClockBackground._clock_angles(timestamp)

    def point(angle, length):
      return rl.Vector2(cx + math.cos(angle) * w * 0.445 * length,
                        cy + math.sin(angle) * h * 0.42 * length)

    # Rounded, broad white hands with narrow stems like the reference.
    for angle, length, thickness in ((hour, 0.49, h * 0.026), (minute, 0.76, h * 0.021)):
      root, start, end = point(angle, 0.0), point(angle, 0.12), point(angle, length)
      rl.draw_line_ex(root, end, thickness + 3, rl.BLACK)
      rl.draw_line_ex(root, end, max(1, thickness * 0.42), white)
      rl.draw_line_ex(start, end, thickness, white)
      rl.draw_circle_v(start, thickness / 2, white)
      rl.draw_circle_v(end, thickness / 2, white)
    rl.draw_line_ex(point(second, -0.12), point(second, 0.86), max(1, h * 0.003), red)
    rl.draw_circle_v(rl.Vector2(cx, cy), h * 0.013, white)
    rl.draw_circle_v(rl.Vector2(cx, cy), h * 0.009, red)
    rl.draw_circle_v(rl.Vector2(cx, cy), h * 0.004, rl.BLACK)

  @staticmethod
  def _draw_clock(rect: rl.Rectangle, timestamp: float) -> None:
    """Compatibility wrapper for drawing hands over a rectangular dial."""
    ClockBackground.draw_clock_hands(rect.x + rect.width / 2, rect.y + rect.height / 2,
                                     rect.width, rect.height, timestamp)

  @staticmethod
  def _draw_date(rect: rl.Rectangle, timestamp: float, draw_small_clock: bool = True) -> None:
    if rect.width <= 0 or rect.height <= 0:
      return
    local = datetime.fromtimestamp(timestamp, CLOCK_TIMEZONE)
    current_time = local.strftime("%I:%M")
    weekday = ('MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN')[local.weekday()]
    number_size = rect.height * DATE_NUMBER_SIZE
    weekday_size = rect.height * DATE_WEEKDAY_SIZE
    time_size = weekday_size
    gap = rect.height * DATE_LINE_GAP
    center_x = rect.x + rect.width * DATE_CENTER_X
    stack_height = time_size + gap + number_size + gap + weekday_size
    time_y = rect.y + rect.height * DATE_CENTER_Y - stack_height / 2
    date_y = time_y + time_size + gap
    date_label = str(local.day)
    time_font = gui_app.font(FontWeight.SEMI_BOLD)
    time_width = rl.measure_text_ex(time_font, current_time, time_size, 0).x
    time_pos = rl.Vector2(center_x - time_width / 2, time_y)
    rl.draw_text_ex(time_font, current_time, rl.Vector2(time_pos.x + 2, time_pos.y + 2),
                    time_size, 0, rl.BLACK)
    rl.draw_text_ex(time_font, current_time, time_pos, time_size, 0, rl.WHITE)

    date_font = gui_app.font(FontWeight.DISPLAY)
    date_width = rl.measure_text_ex(date_font, date_label, number_size, 0).x
    date_pos = rl.Vector2(center_x - date_width / 2, date_y)
    rl.draw_text_ex(date_font, date_label, rl.Vector2(date_pos.x + 2, date_pos.y + 2),
                    number_size, 0, rl.BLACK)
    rl.draw_text_ex(date_font, date_label, date_pos, number_size, 0, rl.WHITE)

    if draw_small_clock:
      # Place the small analog clock where the steering-wheel icon sat.
      hand_radius = min(26.0, rect.height * 0.06)
      hand_center_x = rect.x + 21 + 25
      hand_center_y = rect.y + rect.height - 14 - 25
      seconds = local.second + local.microsecond / 1_000_000
      minutes = local.minute + seconds / 60.0
      hours = local.hour % 12 + minutes / 60.0
      hands = ((-90 + hours * 30, 0.55, rect.height * 0.009, rl.WHITE),
               (-90 + minutes * 6, 0.92, rect.height * 0.006, rl.WHITE),
               (-90 + seconds * 6, 0.92, rect.height * 0.003, rl.Color(255, 45, 58, 255)))
      for angle, length, thickness, color in hands:
        end = rl.Vector2(hand_center_x + math.cos(math.radians(angle)) * hand_radius * length,
                         hand_center_y + math.sin(math.radians(angle)) * hand_radius * length)
        rl.draw_line_ex(rl.Vector2(hand_center_x + 1, hand_center_y + 1),
                        rl.Vector2(end.x + 1, end.y + 1), thickness + 1, rl.BLACK)
        rl.draw_line_ex(rl.Vector2(hand_center_x, hand_center_y), end, thickness, color)
    weekday_font = gui_app.font(FontWeight.SEMI_BOLD)
    weekday_width = rl.measure_text_ex(weekday_font, weekday, weekday_size, 0).x
    weekday_pos = rl.Vector2(center_x - weekday_width / 2, date_y + number_size + gap)
    rl.draw_text_ex(weekday_font, weekday, rl.Vector2(weekday_pos.x + 2, weekday_pos.y + 2),
                    weekday_size, 0, rl.BLACK)
    rl.draw_text_ex(weekday_font, weekday, weekday_pos, weekday_size, 0, rl.WHITE)

  @staticmethod
  def brake_percent() -> int:
    accel = float(ui_state.sm["carControl"].actuators.accel)
    if not math.isfinite(accel):
      return 0
    return round(max(0.0, min(1.0, -accel / 4.0)) * 100)

