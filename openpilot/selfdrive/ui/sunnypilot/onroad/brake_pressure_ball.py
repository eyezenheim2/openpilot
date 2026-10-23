import math

import pyray as rl

from openpilot.selfdrive.ui.sunnypilot.onroad.clock_background import ClockBackground
from openpilot.system.ui.lib.application import FontWeight, gui_app


class BrakePressureBall:
  """Circular brake pressure gauge driven by commanded braking percentage."""

  @staticmethod
  def brake_percent() -> int:
    return ClockBackground.brake_percent()

  def render_at(self, center_x: float, center_y: float, radius: float, percent: int,
                show_readout: bool = True) -> None:
    percent = max(0, min(100, percent))
    center = rl.Vector2(center_x, center_y)
    track_color = rl.Color(90, 34, 48, 220)
    brake_color = rl.Color(255, 35, 75, 255)
    inner_radius, outer_radius = radius * 0.82, radius * 0.98

    rl.draw_circle_v(center, radius + 5, rl.Color(0, 0, 0, 150))
    # Full muted track with a red arc that grows clockwise from 12 o'clock.
    rl.draw_ring(center, inner_radius, outer_radius, 0, 360, 96, track_color)
    if percent > 0:
      end_angle = -90 + 360 * percent / 100
      rl.draw_ring(center, inner_radius, outer_radius, -90, end_angle, max(2, round(96 * percent / 100)), brake_color)
      cap_radius = (outer_radius - inner_radius) / 2
      cap_distance = (outer_radius + inner_radius) / 2
      cap_angle = math.radians(end_angle)
      rl.draw_circle_v(rl.Vector2(center_x + math.cos(cap_angle) * cap_distance,
                                  center_y + math.sin(cap_angle) * cap_distance), cap_radius, brake_color)

    # Keep the center clean, and hide the text while camera-offset controls are shown.
    rl.draw_circle_v(center, radius * 0.76, rl.Color(10, 13, 18, 235))
    if show_readout:
      value = f"{percent}%"
      font_size = radius * 0.58
      font = gui_app.font(FontWeight.DISPLAY)
      measured = rl.measure_text_ex(font, value, font_size, 0)
      value_pos = rl.Vector2(center_x - measured.x / 2, center_y - radius * 0.45)
      rl.draw_text_ex(font, value, rl.Vector2(value_pos.x + 2, value_pos.y + 2), font_size, 0, rl.BLACK)
      rl.draw_text_ex(font, value, value_pos, font_size, 0, rl.WHITE)

      label = "BRAKE"
      label_size = radius * 0.16
      label_font = gui_app.font(FontWeight.SEMI_BOLD)
      label_width = rl.measure_text_ex(label_font, label, label_size, 0).x
      rl.draw_text_ex(label_font, label,
                      rl.Vector2(center_x - label_width / 2, center_y + radius * 0.24),
                      label_size, 0, rl.Color(185, 190, 200, 230))
