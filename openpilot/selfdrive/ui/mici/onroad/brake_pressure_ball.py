import math

import pyray as rl

from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import FontWeight, gui_app


class BrakePressureBall:
  """Circular brake pressure gauge for the compact Mici HUD."""

  @staticmethod
  def brake_percent() -> int:
    accel = float(ui_state.sm["carControl"].actuators.accel)
    if not math.isfinite(accel):
      return 0
    return round(max(0.0, min(1.0, -accel / 4.0)) * 100)

  def render_at(self, center_x: float, center_y: float, radius: float, percent: int,
                show_readout: bool = True) -> None:
    percent = max(0, min(100, percent))
    center = rl.Vector2(center_x, center_y)
    inner_radius, outer_radius = radius * 0.82, radius * 0.98
    active = percent > 0
    track_color = rl.Color(75, 78, 84, 220) if not active else rl.Color(105, 105, 105, 255)
    brake_color = rl.Color(255, 35, 75, 255)

    # Neutral grayscale backing keeps the pressure gauge legible over camera imagery.
    rl.draw_circle_v(center, radius + 5, rl.Color(45, 45, 45, 255))
    rl.draw_ring(center, inner_radius, outer_radius, 0, 360, 96, track_color)
    if percent:
      end_angle = -90 + 360 * percent / 100
      rl.draw_ring(center, inner_radius, outer_radius, -90, end_angle,
                   max(2, round(96 * percent / 100)), brake_color)
      cap_radius = (outer_radius - inner_radius) / 2
      cap_distance = (outer_radius + inner_radius) / 2
      cap_angle = math.radians(end_angle)
      rl.draw_circle_v(rl.Vector2(center_x + math.cos(cap_angle) * cap_distance,
                                  center_y + math.sin(cap_angle) * cap_distance), cap_radius, brake_color)

    inner_color = rl.Color(24, 25, 29, 235) if not active else rl.Color(28, 28, 28, 255)
    rl.draw_circle_v(center, radius * 0.76, inner_color)
    if not show_readout:
      return

    value = f"{percent}%"
    font_size = radius * 0.58
    font = gui_app.font(FontWeight.DISPLAY)
    value_size = rl.measure_text_ex(font, value, font_size, 0)

    label = "BRAKE"
    label_size = radius * 0.16
    label_font = gui_app.font(FontWeight.SEMI_BOLD)
    label_size_measured = rl.measure_text_ex(label_font, label, label_size, 0)

    gap = radius * 0.06
    group_height = value_size.y + gap + label_size_measured.y
    group_top = center_y - group_height / 2
    value_pos = rl.Vector2(center_x - value_size.x / 2, group_top)
    value_color = rl.WHITE if active else rl.Color(145, 149, 157, 255)
    if active:
      rl.draw_text_ex(font, value, rl.Vector2(value_pos.x + 2, value_pos.y + 2), font_size, 0, rl.BLACK)
    rl.draw_text_ex(font, value, value_pos, font_size, 0, value_color)

    label_pos = rl.Vector2(center_x - label_size_measured.x / 2,
                           group_top + value_size.y + gap)
    rl.draw_text_ex(label_font, label,
                    label_pos, label_size, 0,
                    rl.Color(125, 130, 138, 220) if not active else rl.Color(185, 190, 200, 230))
