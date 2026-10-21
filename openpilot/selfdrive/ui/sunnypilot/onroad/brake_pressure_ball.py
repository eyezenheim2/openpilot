import time

import pyray as rl

from openpilot.selfdrive.ui.sunnypilot.onroad.clock_background import ClockBackground


class BrakePressureBall:
  """Brake disc icon driven by openpilot's commanded braking percentage."""

  @staticmethod
  def brake_percent() -> int:
    return ClockBackground.brake_percent()

  def render_at(self, center_x: float, center_y: float, radius: float, percent: int) -> None:
    percent = max(0, min(100, percent))
    center = rl.Vector2(center_x, center_y)

    # Brake-disc symbol: black at 0% and white at 100% brake pressed.
    shade = round(255 * percent / 100)
    icon_color = rl.Color(shade, shade, shade, 255)
    cutout = rl.Color(10, 13, 18, 240)
    rl.draw_circle_v(center, radius + 5, rl.Color(0, 0, 0, 150))
    rl.draw_circle_v(center, radius, rl.Color(112, 118, 126, 235))
    rl.draw_ring(center, radius * 0.68, radius * 0.96, 0, 360, 64, icon_color)
    rl.draw_ring(center, radius * 0.58, radius * 0.64, 0, 360, 64, icon_color)
    rl.draw_circle_v(center, radius * 0.53, cutout)
    # Keep a dark center badge for the clock hands.
    rl.draw_circle_v(center, radius * 0.36, cutout)
    rl.draw_circle_lines(int(center_x), int(center_y), radius * 0.36, icon_color)
    for i in range(8):
      angle = i * math.tau / 8
      hole = rl.Vector2(center_x + math.cos(angle) * radius * 0.40,
                        center_y + math.sin(angle) * radius * 0.40)
      rl.draw_circle_v(hole, radius * 0.075, cutout)

    # Keep the live clock hands inside the brake disc's center badge. The
    # surrounding disc ring continues to show commanded braking intensity.
    ClockBackground.draw_clock_hands(center_x, center_y, radius * 0.72, radius * 0.72, time.time())
