import math
from datetime import datetime

import pyray as rl

from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import FontWeight, gui_app


class BrakePressureBall:
  """Circular pressure gauge with a synchronized ring and analog needle."""

  @staticmethod
  def brake_percent() -> int:
    accel = float(ui_state.sm["carControl"].actuators.accel)
    if not math.isfinite(accel):
      return 0
    return round(max(0.0, min(1.0, -accel / 4.0)) * 100)

  def render_at(self, center_x: float, center_y: float, radius: float, percent: int,
                show_readout: bool = True) -> None:
    percent = max(0, min(100, percent))
    active = percent > 0
    center = rl.Vector2(center_x, center_y)
    inner_radius, outer_radius = radius * 0.82, radius * 0.98
    track_color = rl.Color(75, 78, 84, 220) if not active else rl.Color(105, 105, 105, 255)
    brake_color = rl.Color(255, 35, 75, 255)
    end_angle = -90 + 360 * percent / 100

    rl.draw_circle_v(center, radius + 5, rl.Color(45, 45, 45, 255))
    rl.draw_ring(center, inner_radius, outer_radius, 0, 360, 96, track_color)
    if active:
      rl.draw_ring(center, inner_radius, outer_radius, -90, end_angle,
                   max(2, round(96 * percent / 100)), brake_color)
      cap_radius = (outer_radius - inner_radius) / 2
      cap_distance = (outer_radius + inner_radius) / 2
      cap_angle = math.radians(end_angle)
      rl.draw_circle_v(rl.Vector2(center_x + math.cos(cap_angle) * cap_distance,
                                  center_y + math.sin(cap_angle) * cap_distance),
                       cap_radius, brake_color)

    inner_color = rl.Color(24, 25, 29, 235) if not active else rl.Color(28, 28, 28, 255)
    rl.draw_circle_v(center, radius * 0.76, inner_color)
    now = datetime.now().astimezone()
    seconds = now.second + now.microsecond / 1_000_000
    minutes = now.minute + seconds / 60.0
    hours = now.hour % 12 + minutes / 60.0
    hand_angles = (-90 + hours * 30, -90 + minutes * 6, -90 + seconds * 6)
    for tick in range(12):
      angle = math.radians(-90 + tick * 30)
      tick_start = rl.Vector2(center_x + math.cos(angle) * radius * 0.66,
                              center_y + math.sin(angle) * radius * 0.66)
      tick_end = rl.Vector2(center_x + math.cos(angle) * radius * 0.72,
                            center_y + math.sin(angle) * radius * 0.72)
      rl.draw_line_ex(tick_start, tick_end, max(1.0, radius * 0.008), rl.Color(145, 150, 160, 210))
    hands = ((hand_angles[0], 0.36, max(3.0, radius * 0.035), rl.WHITE),
             (hand_angles[1], 0.54, max(2.0, radius * 0.022), rl.WHITE),
             (hand_angles[2], 0.54, max(1.5, radius * 0.010), rl.Color(255, 45, 58, 255)))
    for angle, length, thickness, hand_color in hands:
      end = rl.Vector2(center_x + math.cos(math.radians(angle)) * radius * length,
                       center_y + math.sin(math.radians(angle)) * radius * length)
      rl.draw_line_ex(center, end, thickness, hand_color)
    if not show_readout:
      return

    font = gui_app.font(FontWeight.DISPLAY)
    value = f"{percent}%"
    value_font_size = max(30.0, radius * 0.32)
    value_size = rl.measure_text_ex(font, value, value_font_size, 0)
    value_pos = rl.Vector2(center_x - value_size.x / 2,
                           center_y + radius * 0.10 - value_size.y / 2)
    value_color = rl.WHITE if active else rl.Color(145, 149, 157, 255)
    if active:
      rl.draw_text_ex(font, value, rl.Vector2(value_pos.x + 2, value_pos.y + 2),
                      value_font_size, 0, rl.BLACK)
    rl.draw_text_ex(font, value, value_pos, value_font_size, 0, value_color)

    label = "BRAKE"
    label_size = max(18.0, radius * 0.14)
    label_font = gui_app.font(FontWeight.SEMI_BOLD)
    measured = rl.measure_text_ex(label_font, label, label_size, 0)
    label_pos = rl.Vector2(center_x - measured.x / 2,
                           center_y + radius * 0.52 - measured.y / 2)
    label_color = rl.Color(125, 130, 138, 220) if not active else rl.Color(185, 190, 200, 230)
    rl.draw_text_ex(label_font, label, label_pos, label_size, 0, label_color)
