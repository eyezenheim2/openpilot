import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pyray as rl
from openpilot.cereal import log
from opendbc.car.structs import car
from openpilot.cereal.visionipc import VisionStreamType
from openpilot.selfdrive.ui.ui_state import ui_state, UIStatus
from openpilot.selfdrive.ui.mici.onroad import SIDE_PANEL_WIDTH
from openpilot.selfdrive.ui.mici.onroad.alert_renderer import AlertRenderer
from openpilot.selfdrive.ui.mici.onroad.driver_state import DriverStateRenderer
from openpilot.selfdrive.ui.mici.onroad.hud_renderer import HudRenderer
from openpilot.selfdrive.ui.mici.onroad.long_indicator import LongIndicator
from openpilot.selfdrive.ui.mici.onroad.model_renderer import ModelRenderer
from openpilot.selfdrive.ui.mici.onroad.navigation import NavigationPanel, PANEL_WIDTH
from openpilot.selfdrive.ui.mici.onroad.confidence_ball import ConfidenceBall
from openpilot.selfdrive.ui.mici.onroad.cameraview import CameraView
from openpilot.system.ui.lib.application import FontWeight, gui_app, MousePos, MouseEvent, TextAlignment, TextAlignmentVertical
from openpilot.system.ui.widgets.label import UnifiedLabel
from openpilot.system.ui.widgets import Widget
from openpilot.common.filter_simple import BounceFilter, FirstOrderFilter
from openpilot.common.transformations.camera import DEVICE_CAMERAS, DeviceCameraConfig, view_frame_from_device_frame
from openpilot.common.transformations.orientation import rot_from_euler
from enum import IntEnum

if gui_app.sunnypilot_ui():
  from openpilot.selfdrive.ui.sunnypilot.mici.onroad.hud_renderer import HudRendererSP as HudRenderer
  from openpilot.selfdrive.ui.sunnypilot.ui_state import OnroadTimerStatus

OpState = log.SelfdriveState.OpenpilotState
CALIBRATED = log.ExtrinsicsCalibration.Status.calibrated
NARROW_ROAD_CAM = VisionStreamType.VISION_STREAM_NARROW_ROAD
WIDE_CAM = VisionStreamType.VISION_STREAM_WIDE_ROAD
DEFAULT_DEVICE_CAMERA = DEVICE_CAMERAS["tici", "ar0231"]


class BookmarkState(IntEnum):
  HIDDEN = 0
  DRAGGING = 1
  TRIGGERED = 2

WIDE_CAM_MAX_SPEED = 5.0  # m/s (11 mph)
ROAD_CAM_MIN_SPEED = 10.0  # m/s (22 mph)
SCREENSHOT_LONG_TOUCH_SECONDS = 5
SCREENSHOT_STATUS_SECONDS = 3.0

CAM_Y_OFFSET = 20


class BookmarkIcon(Widget):
  PEEK_THRESHOLD = 150 # If icon peeks out this much, snap it fully visible
  FULL_VISIBLE_OFFSET = 200  # How far onscreen when fully visible
  HIDDEN_OFFSET = -50  # How far offscreen when hidden

  def __init__(self, bookmark_callback):
    super().__init__()
    self._bookmark_callback = bookmark_callback
    self._icon = gui_app.texture("icons_mici/onroad/bookmark.png", 180, 180)
    self._filled_icon = gui_app.texture("icons_mici/onroad/bookmark_fill.png", 180, 180)
    self._offset_filter = BounceFilter(0.0, 0.1, 1 / gui_app.target_fps)
    self._active_alpha = FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps)

    # State
    self._interacting = False
    self._state = BookmarkState.HIDDEN
    self._swipe_start_x = 0.0
    self._swipe_current_x = 0.0
    self._is_swiping = False
    self._is_swiping_left: bool = False
    self._triggered_time: float = 0.0
    self._triggered_duration = 1.5
    self._swipe_expired = False

  def is_swiping_left(self) -> bool:
    """Check if currently swiping left (for scroller to disable)."""
    return self._is_swiping_left

  def interacting(self):
    interacting, self._interacting = self._interacting, False
    return interacting

  def _update_state(self):
    if self._state == BookmarkState.TRIGGERED and rl.get_time() - self._triggered_time >= self._triggered_duration:
      self._state = BookmarkState.HIDDEN
      self._swipe_expired = self._is_swiping

    swipe_offset = self._swipe_start_x - self._swipe_current_x
    armed = self._state == BookmarkState.DRAGGING and swipe_offset > self.PEEK_THRESHOLD
    if self._state == BookmarkState.DRAGGING:
      # Snap to the released position when armed, while allowing further dragging or cancellation.
      if armed:
        swipe_offset += self.FULL_VISIBLE_OFFSET - self.PEEK_THRESHOLD
      swipe_offset = min(swipe_offset, self.FULL_VISIBLE_OFFSET + 50)
      self._offset_filter.update(swipe_offset)

    elif self._state == BookmarkState.TRIGGERED:
      # Let another left swipe move the same bookmark.
      offset = self.FULL_VISIBLE_OFFSET
      if self._is_swiping and self._is_swiping_left:
        offset += min(swipe_offset, 50)
      self._offset_filter.update(offset)

    elif self._state == BookmarkState.HIDDEN:
      self._offset_filter.update(self.HIDDEN_OFFSET)

      if self._offset_filter.x < 1e-3:
        self._interacting = False

    self._active_alpha.update(float(armed or self._state == BookmarkState.TRIGGERED))

  def _handle_mouse_event(self, mouse_event: MouseEvent):
    if not ui_state.started:
      return

    if mouse_event.left_pressed:
      if self._state == BookmarkState.TRIGGERED:
        self._triggered_time = rl.get_time()
        self._triggered_duration = 0.5

      # Store relative position within widget
      self._swipe_start_x = mouse_event.pos.x
      self._swipe_current_x = mouse_event.pos.x
      self._is_swiping = True
      self._is_swiping_left = False
      self._swipe_expired = False

    elif mouse_event.left_down and self._is_swiping:
      self._swipe_current_x = mouse_event.pos.x
      swipe_offset = self._swipe_start_x - self._swipe_current_x
      self._is_swiping_left = swipe_offset > 0
      if self._is_swiping_left:
        self._interacting = True
        if self._state == BookmarkState.HIDDEN and not self._swipe_expired:
          self._state = BookmarkState.DRAGGING

    elif mouse_event.left_released:
      if self._is_swiping and self._state == BookmarkState.DRAGGING:
        swipe_distance = self._swipe_start_x - self._swipe_current_x

        # If peeking past threshold, transition to animating to fully visible and bookmark
        if swipe_distance > self.PEEK_THRESHOLD:
          self._state = BookmarkState.TRIGGERED
          self._triggered_time = rl.get_time()
          self._triggered_duration = 1.5
          self._bookmark_callback()
        else:
          # Otherwise, transition back to hidden
          self._state = BookmarkState.HIDDEN

      # Reset swipe state
      self._is_swiping = False
      self._is_swiping_left = False

  def _render(self, _):
    """Render the bookmark icon."""
    if self._offset_filter.x > 0:
      icon_x = self.rect.x + self.rect.width - round(self._offset_filter.x)
      icon_y = self.rect.y + (self.rect.height - self._icon.height) / 2  # Vertically centered
      for icon, alpha in ((self._icon, 1.0 - self._active_alpha.x), (self._filled_icon, self._active_alpha.x)):
        rl.draw_texture_ex(icon, rl.Vector2(icon_x, icon_y), 0.0, 1.0, rl.Color(255, 255, 255, round(255 * alpha)))


class AugmentedRoadView(CameraView):
  def __init__(self, bookmark_callback=None, stream_type: VisionStreamType = VisionStreamType.VISION_STREAM_NARROW_ROAD):
    super().__init__("camerad", stream_type)
    self._bookmark_callback = bookmark_callback
    self._set_placeholder_color(rl.BLACK)

    self.device_camera: DeviceCameraConfig | None = None
    self.view_from_calib = view_frame_from_device_frame.copy()
    self.view_from_wide_calib = view_frame_from_device_frame.copy()

    self._matrix_cache_key: tuple | None = None
    self._cached_matrix: np.ndarray | None = None
    self._content_rect = rl.Rectangle()
    self._last_click_time = 0.0
    self._control_press: tuple[float, float, float] | None = None
    self._offset_controls_until = 0.0
    self._steer_ratio_controls_until = 0.0
    self._camera_offset_value = float(ui_state.params.get("CameraOffset", return_default=True))
    self._screenshot_status = ""
    self._screenshot_status_until = 0.0
    self._last_active_gear = None
    self._gear_popup_letter = None
    self._gear_popup_until = 0.0
    self._clock_icon_size = 100

    # Bookmark icon with swipe gesture
    self._bookmark_icon = BookmarkIcon(bookmark_callback)

    self._navigation = NavigationPanel()
    self._model_renderer = ModelRenderer()
    self._hud_renderer = HudRenderer()
    self._long_indicator = LongIndicator()
    self._alert_renderer = AlertRenderer()
    self._driver_state_renderer = DriverStateRenderer()
    self._confidence_ball = ConfidenceBall()
    self._offroad_label = UnifiedLabel("start the car to\nuse sunnypilot", 54, FontWeight.DISPLAY,
                                       text_color=rl.Color(255, 255, 255, int(255 * 0.9)),
                                       alignment=TextAlignment.CENTER,
                                       alignment_vertical=TextAlignmentVertical.MIDDLE)

    self._fade_texture = gui_app.texture("icons_mici/onroad/onroad_fade.png")

  def is_swiping_left(self) -> bool:
    """Check if currently swiping left (for scroller to disable)."""
    return self._bookmark_icon.is_swiping_left()

  def _update_state(self):
    super()._update_state()

    # update offroad label
    if ui_state.panda_type == log.PandaState.PandaType.unknown:
      self._offroad_label.set_text("system booting")
    elif ui_state.ignition and not ui_state.started:
      self._offroad_label.set_text("openpilot can't start\ncheck alerts")
    else:
      self._offroad_label.set_text("start the car to\nuse sunnypilot")

  def _handle_mouse_press(self, mouse_pos: MousePos):
    self._control_press = (rl.get_time(), mouse_pos.x, mouse_pos.y)

  def _handle_mouse_release(self, mouse_pos: MousePos):
    # The bookmark swipe has priority over HUD gestures.
    if self._bookmark_icon.interacting():
      self._control_press = None
      return

    start = self._control_press
    self._control_press = None
    if start is None or not ui_state.started:
      super()._handle_mouse_release(mouse_pos)
      return

    pressed_at, start_x, start_y = start
    dx = mouse_pos.x - start_x
    dy = mouse_pos.y - start_y
    now = rl.get_time()

    if now - pressed_at >= SCREENSHOT_LONG_TOUCH_SECONDS and dx * dx + dy * dy <= 30 ** 2:
      self._request_screenshot()
      return

    # Hold for 0.6 seconds anywhere on the HUD to open fixed steer-ratio controls.
    if now - pressed_at >= 0.6 and (dx * dx + dy * dy) <= 45 ** 2:
      self._offset_controls_until = 0.0
      self._steer_ratio_controls_until = now + 5.0
      return

    if abs(dx) > 30 or abs(dy) > 30:
      super()._handle_mouse_release(mouse_pos)
      return

    if now < self._steer_ratio_controls_until:
      third = self.rect.x + self.rect.width / 3
      two_thirds = self.rect.x + self.rect.width * 2 / 3
      if mouse_pos.x < third:
        self._hud_renderer.adjust_steer_ratio("-")
      elif mouse_pos.x > two_thirds:
        self._hud_renderer.adjust_steer_ratio("+")
      else:
        self._hud_renderer.toggle_steer_ratio()
      self._steer_ratio_controls_until = now + 5.0
      return

    if now < self._offset_controls_until:
      midpoint = self.rect.x + self.rect.width / 2
      step = -0.01 if mouse_pos.x < midpoint else 0.01
      value = float(ui_state.params.get("CameraOffset", return_default=True))
      value = round(max(-0.35, min(0.35, value + step)), 2)
      ui_state.params.put("CameraOffset", value)
      self._camera_offset_value = value
      self._model_renderer._camera_offset = value
      return

    # A tap opens the camera-offset control bar.
    self._steer_ratio_controls_until = 0.0
    self._camera_offset_value = float(ui_state.params.get("CameraOffset", return_default=True))
    self._offset_controls_until = now + 5.0

  def _render(self, _):
    # Draw text if not onroad
    if not ui_state.started:
      self._content_rect = self.rect
      rl.draw_rectangle_rec(self.rect, rl.BLACK)
      self._offroad_label.render(self._rect)
      self._draw_analog_clock_icon()
      return

    self._switch_stream_if_needed(ui_state.sm)

    # Update calibration before rendering
    self._update_calibration()

    alert_to_render, not_animating_out = self._alert_renderer.will_render()
    show_navigation = ui_state.navigation_enabled and ui_state.nav_destination_set and alert_to_render is None

    # Create inner content area with border padding
    self._content_rect = rl.Rectangle(
      self.rect.x,
      self.rect.y,
      self.rect.width - (PANEL_WIDTH if show_navigation else SIDE_PANEL_WIDTH),
      self.rect.height,
    )

    # Enable scissor mode to clip all rendering within content rectangle boundaries
    # This creates a rendering viewport that prevents graphics from drawing outside the border
    rl.begin_scissor_mode(
      int(self._content_rect.x),
      int(self._content_rect.y),
      int(self._content_rect.width),
      int(self._content_rect.height)
    )

    # Render the base camera view
    super()._render(self._content_rect)

    self._draw_hud_backing(self._content_rect)

    # Draw all UI overlays
    self._model_renderer.render(self._content_rect)

    # Fade out bottom of overlays for looks
    rl.draw_texture_ex(self._fade_texture, rl.Vector2(self._content_rect.x, self._content_rect.y), 0.0, 1.0, rl.WHITE)

    # Hide DMoji when disengaged unless AlwaysOnDM is enabled
    should_draw_dmoji = (not self._hud_renderer.drawing_top_icons() and
                         (ui_state.status != UIStatus.DISENGAGED or ui_state.always_on_dm))
    self._driver_state_renderer.set_should_draw(should_draw_dmoji)
    dm_size = self._driver_state_renderer.BASE_SIZE
    dm_margin = 16
    self._driver_state_renderer.set_position(
      self._rect.x + self._rect.width - dm_size - dm_margin,
      self._rect.y + self._rect.height - dm_size - dm_margin,
    )
    self._driver_state_renderer.render()
    self._long_indicator.set_should_draw(not self._hud_renderer.drawing_top_icons())
    self._long_indicator.render(self._content_rect)

    self._hud_renderer.set_can_draw_top_icons(alert_to_render is None)
    self._hud_renderer.set_wheel_critical_icon(alert_to_render is not None and not not_animating_out and
                                               alert_to_render.visual_alert == car.CarControl.HUDControl.VisualAlert.steerRequired)
    self._hud_renderer.render(self._content_rect)
    self._draw_analog_clock_icon()
    self._draw_day_date()
    self._draw_ai_label()
    self._draw_fuel_gauge()
    self._draw_brake_pressure_percent()
    self._draw_gear_change_popup()
    # Keep alert popups above the HUD text and background.
    self._alert_renderer.render(self._content_rect)
    self._draw_camera_and_steer_controls()
    self._draw_screenshot_status()

    # Draw fake rounded border
    rl.draw_rectangle_rounded_lines_ex(self._content_rect, 0.2 * 1.02, 10, 50, rl.BLACK)

    # End clipping region
    rl.end_scissor_mode()

    # Custom UI extension point - add custom overlays here
    # Use self._content_rect for positioning within camera bounds
    if show_navigation:
      self._navigation.render(rl.Rectangle(self._content_rect.x + self._content_rect.width, self.rect.y, PANEL_WIDTH, self.rect.height))
    else:
      self._confidence_ball.render(self.rect)

    self._bookmark_icon.render(self.rect)

  def _request_screenshot(self) -> None:
    self._control_press = None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    gui_app.request_screenshot(
      Path("/data/media/0/screenshot") / f"openpilot_{timestamp}.png",
      on_saved=lambda: self._set_screenshot_status("SAVED"),
      on_failed=lambda: self._set_screenshot_status("SAVE FAILED"),
    )

  def _set_screenshot_status(self, status: str) -> None:
    self._screenshot_status = status
    self._screenshot_status_until = rl.get_time() + SCREENSHOT_STATUS_SECONDS

  def _draw_screenshot_status(self) -> None:
    if not self._screenshot_status or rl.get_time() >= self._screenshot_status_until:
      return

    font = gui_app.font(FontWeight.DISPLAY)
    size = 40
    text = self._screenshot_status
    extent = rl.measure_text_ex(font, text, size, 0)
    x = self.rect.x + (self.rect.width - extent.x) / 2
    y = self.rect.y + self.rect.height * 0.15
    padding_x, padding_y = 24, 14
    rl.draw_rectangle_rec(
      rl.Rectangle(x - padding_x, y - padding_y, extent.x + padding_x * 2, extent.y + padding_y * 2),
      rl.Color(0, 0, 0, 220),
    )
    color = rl.GREEN if text == "SAVED" else rl.RED
    rl.draw_text_ex(font, text, rl.Vector2(x, y), size, 0, color)

  def _draw_hud_backing(self, rect: rl.Rectangle) -> None:
    top_height = min(170, max(110, int(rect.height * 0.22)))
    bottom_height = min(220, max(140, int(rect.height * 0.36)))
    rl.draw_rectangle_rec(rl.Rectangle(rect.x, rect.y, rect.width, top_height), rl.BLACK)
    rl.draw_rectangle_rec(
      rl.Rectangle(rect.x, rect.y + rect.height - bottom_height, rect.width, bottom_height), rl.BLACK)


  def _draw_analog_clock_icon(self) -> None:
    rect = self._content_rect
    clock_size = self._clock_icon_size
    radius = clock_size * 0.40
    hand_scale = 1.5
    hand_reach = radius * 0.84 * hand_scale
    if ui_state.started:
      center_x = rect.x + hand_reach + 8
    else:
      center_x = rect.x + rect.width - hand_reach - 110
    center_y = rect.y + rect.height - hand_reach

    now = datetime.now(ZoneInfo("America/Los_Angeles"))
    seconds = now.second + now.microsecond / 1_000_000
    minutes = now.minute + seconds / 60
    hours = (now.hour % 12) + minutes / 60
    hands = (
      (-90 + hours * 30, radius * 0.50 * hand_scale, 4.0 * hand_scale, rl.WHITE),
      (-90 + minutes * 6, radius * 0.72 * hand_scale, 3.0 * hand_scale, rl.WHITE),
      (-90 + seconds * 6, radius * 0.48 * hand_scale, 1.8 * hand_scale, rl.RED),
    )
    center = rl.Vector2(center_x, center_y)
    for angle, length, thickness, color in hands:
      end = rl.Vector2(center_x + math.cos(math.radians(angle)) * length,
                       center_y + math.sin(math.radians(angle)) * length)
      rl.draw_line_ex(center, end, thickness, color)

  def _draw_day_date(self) -> None:
    rect = self._content_rect
    now = datetime.now(ZoneInfo("America/Los_Angeles"))
    weekday = now.strftime("%a").upper()
    day = now.strftime("%d").lstrip("0") or "0"

    font = gui_app.font(FontWeight.DISPLAY)
    center_x = rect.x + rect.width - 24 - min(105.0, rect.height * 0.21)
    weekday_size = min(24.0, rect.height * 0.055)
    day_size = min(88.0, rect.height * 0.19)
    weekday_extent = rl.measure_text_ex(font, weekday, weekday_size, 0)
    day_extent = rl.measure_text_ex(font, day, day_size, 0)

    rl.draw_text_ex(font, weekday,
                    rl.Vector2(center_x - weekday_extent.x / 2, rect.y + 22),
                    weekday_size, 0, rl.Color(220, 228, 238, 255))
    rl.draw_text_ex(font, day,
                    rl.Vector2(center_x - day_extent.x / 2,
                               rect.y + 24 + weekday_extent.y),
                    day_size, 0, rl.WHITE)

  def _draw_ai_label(self) -> None:
    rect = self._content_rect
    now = datetime.now(ZoneInfo("America/Los_Angeles"))
    font = gui_app.font(FontWeight.DISPLAY)
    text = "AI"
    font_size = 36
    text_extent = rl.measure_text_ex(font, text, font_size, 0)
    radius = min(105.0, rect.height * 0.21)
    center_x = rect.x + rect.width - 24 - radius
    day = now.strftime("%d").lstrip("0") or "0"
    day_size = min(88.0, rect.height * 0.19)
    day_extent = rl.measure_text_ex(font, day, day_size, 0)
    weekday = now.strftime("%a").upper()
    weekday_size = min(24.0, rect.height * 0.055)
    weekday_extent = rl.measure_text_ex(font, weekday, weekday_size, 0)

    x = center_x + day_extent.x / 2 + 12
    if x + text_extent.x > rect.x + rect.width - 8:
      x = center_x - day_extent.x / 2 - 12 - text_extent.x
    y = rect.y + 24 + weekday_extent.y + (day_extent.y - text_extent.y) / 2

    car_state = ui_state.sm["carState"]
    car_control = ui_state.sm["carControl"]
    pedal_override = car_state.gasPressed or car_state.brakePressed or any(
      event.name == log.OnroadEvent.EventName.gasPressedOverride
      for event in ui_state.sm["onroadEvents"]
    )
    if pedal_override:
      color = rl.GRAY
    elif (self._hud_renderer.is_cruise_set and ui_state.has_longitudinal_control and
          car_control.longActive):
      color = rl.Color(0, 255, 80, 255)
    elif car_control.latActive:
      color = rl.Color(0, 170, 55, 255)
    else:
      color = rl.WHITE

    rl.draw_text_ex(font, text, rl.Vector2(x + 2, y + 2), font_size, 0, rl.Color(0, 0, 0, 190))
    rl.draw_text_ex(font, text, rl.Vector2(x, y), font_size, 0, color)

  def _ai_brake_pressure(self) -> float:
    car_control = ui_state.sm["carControl"]
    if not ui_state.has_longitudinal_control or not car_control.longActive:
      return 0.0

    # The longitudinal controller expresses braking as negative acceleration.
    return max(0.0, min(1.0, -float(car_control.actuators.accel) / 4.0))

  @staticmethod
  def _brake_pressure_color(brake_pressure: float) -> rl.Color:
    color_stops = ((255, 205, 105), (255, 145, 0), (255, 45, 20), (255, 0, 35))
    color_position = brake_pressure * (len(color_stops) - 1)
    color_index = min(int(color_position), len(color_stops) - 2)
    blend = color_position - color_index
    start_color, end_color = color_stops[color_index:color_index + 2]
    red, green, blue = (int(start + (end - start) * blend)
                        for start, end in zip(start_color, end_color))
    return rl.Color(red, green, blue, 255)

  def _draw_brake_pressure_percent(self) -> None:
    brake_pressure = self._ai_brake_pressure()
    if brake_pressure <= 0.01:
      return

    rect = self._content_rect
    percent_text = f"{round(brake_pressure * 100)}%"
    font = gui_app.font(FontWeight.DISPLAY)
    max_radius = min(32.0, rect.height * 0.08)
    percent_font_size = max_radius * 1.8
    percent_extent = rl.measure_text_ex(font, percent_text, percent_font_size, 0)
    gauge_center_x = rect.x + rect.width / 2
    gauge_y = rect.y + 32 + 25 + 24
    percent_x = gauge_center_x - percent_extent.x / 2
    percent_y = gauge_y + max_radius + 10 - percent_extent.y / 2
    glow_alpha = int(35 + 160 * brake_pressure)
    color = self._brake_pressure_color(brake_pressure)
    glow = rl.Color(color.r, color.g, color.b, glow_alpha)
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
      rl.draw_text_ex(font, percent_text, rl.Vector2(percent_x + dx, percent_y + dy),
                      percent_font_size, 0, glow)
    rl.draw_text_ex(font, percent_text, rl.Vector2(percent_x, percent_y),
                    percent_font_size, 0, color)

  def _draw_fuel_gauge(self) -> None:
    rect = self._content_rect
    car_state = ui_state.sm["carState"]
    fuel = float(car_state.fuelGauge)
    if fuel > 1.0:
      fuel /= 100.0
    fuel = max(0.0, min(1.0, fuel))
    gear = car_state.gearShifter
    gear_shifter = car.CarState.GearShifter
    active_gear = {
      gear_shifter.park: "P",
      gear_shifter.reverse: "R",
      gear_shifter.neutral: "N",
      gear_shifter.drive: "D",
      gear_shifter.low: "L",
    }.get(gear)
    if active_gear is None:
      gear_name = str(gear).lower().rsplit(".", 1)[-1]
      active_gear = {"park": "P", "reverse": "R", "neutral": "N", "drive": "D", "low": "L"}.get(gear_name)
    if active_gear is not None and active_gear != self._last_active_gear:
      self._gear_popup_letter = active_gear
      self._gear_popup_until = 0.0 if active_gear == "R" else rl.get_time() + 3.0
    self._last_active_gear = active_gear

    # Center PRNDL over an E-to-F dash gauge.
    gear_step = 36.0
    gear_width = gear_step * 5
    gear_left = rect.x + (rect.width - gear_width) / 2
    gear_center_y = rect.y + 32 + 25
    gray = rl.Color(150, 155, 160, 255)

    font = gui_app.font(FontWeight.DISPLAY)
    for index, letter in enumerate("PRNDL"):
      color = rl.Color(0, 255, 80, 255) if letter == active_gear else gray
      extent = rl.measure_text_ex(font, letter, 34, 0)
      x = gear_left + index * gear_step + (gear_step - extent.x) / 2
      y = gear_center_y - extent.y / 2
      rl.draw_text_ex(font, letter, rl.Vector2(x, y), 34, 0, color)

    dash_count = 10
    dash_width = 9.0
    dash_gap = 4.0
    dash_total_width = dash_count * dash_width + (dash_count - 1) * dash_gap
    fuel_label_size = 14
    zero_extent = rl.measure_text_ex(font, "0%", fuel_label_size, 0)
    full_extent = rl.measure_text_ex(font, "100%", fuel_label_size, 0)
    gauge_width = zero_extent.x + 7 + dash_total_width + 7 + full_extent.x
    gauge_left = rect.x + (rect.width - gauge_width) / 2
    gauge_y = gear_center_y + 24
    rl.draw_text_ex(font, "0%", rl.Vector2(gauge_left, gauge_y - zero_extent.y / 2),
                    fuel_label_size, 0, gray)
    dash_left = gauge_left + zero_extent.x + 7
    filled_dashes = math.ceil(fuel * dash_count) if fuel > 0.0 else 0
    brake_pressure = self._ai_brake_pressure()
    braking = brake_pressure > 0.01
    if braking:
      filled_dashes = max(1, round(brake_pressure * dash_count))
      fill_color = self._brake_pressure_color(brake_pressure)
    else:
      fill_color = rl.Color(0, 255, 80, 255)
    for index in range(dash_count):
      color = fill_color if index < filled_dashes else gray
      rl.draw_rectangle_rec(rl.Rectangle(dash_left + index * (dash_width + dash_gap),
                                           gauge_y - 2, dash_width, 4), color)
    full_x = dash_left + dash_total_width + 7
    rl.draw_text_ex(font, "100%", rl.Vector2(full_x, gauge_y - full_extent.y / 2),
                    fuel_label_size, 0, gray)

  def _draw_gear_change_popup(self) -> None:
    if (self._gear_popup_letter is None or
        (self._gear_popup_letter != "R" and rl.get_time() >= self._gear_popup_until)):
      return

    rect = self._content_rect
    size = min(190.0, rect.height * 0.62)
    center_x = rect.x + rect.width / 2
    center_y = rect.y + rect.height * 0.60
    box = rl.Rectangle(center_x - size / 2, center_y - size / 2, size, size)
    green = rl.Color(0, 255, 80, 255)
    for padding, thickness, alpha in ((9.0, 12.0, 30), (5.0, 7.0, 80)):
      glow = rl.Rectangle(box.x - padding, box.y - padding,
                          box.width + padding * 2, box.height + padding * 2)
      rl.draw_rectangle_rounded_lines_ex(
        glow, 0.18, 10, thickness, rl.Color(0, 255, 80, alpha))
    rl.draw_rectangle_rounded(box, 0.18, 10, rl.Color(0, 0, 0, 235))
    rl.draw_rectangle_rounded_lines_ex(box, 0.18, 10, 6.0, green)

    font = gui_app.font(FontWeight.DISPLAY)
    font_size = size * 0.68
    extent = rl.measure_text_ex(font, self._gear_popup_letter, font_size, 0)
    rl.draw_text_ex(font, self._gear_popup_letter,
                    rl.Vector2(box.x + (box.width - extent.x) / 2,
                               box.y + (box.height - extent.y) / 2),
                    font_size, 0, rl.WHITE)

  def _draw_camera_and_steer_controls(self) -> None:
    now = rl.get_time()
    if now >= self._offset_controls_until:
      self._offset_controls_until = 0.0
    if now >= self._steer_ratio_controls_until:
      self._steer_ratio_controls_until = 0.0
    if not self._offset_controls_until and not self._steer_ratio_controls_until:
      return

    # Use the complete HUD rectangle so the black bar reaches both HUD edges.
    rect = self.rect
    height = min(rect.height * 0.80, max(160.0, rect.height * 0.40))
    top = rect.y + (rect.height - height) / 2
    rl.draw_rectangle_rec(rl.Rectangle(rect.x, top, rect.width, height), rl.BLACK)

    font = gui_app.font(FontWeight.DISPLAY)
    ratio_mode = self._steer_ratio_controls_until > now
    if ratio_mode:
      value = f"{self._hud_renderer.steer_ratio_value():.1f}"
      ratio_state = "ON" if self._hud_renderer.steer_ratio_enabled() else "OFF"
      status = f"FIXED STEER RATIO {ratio_state}"
      symbol_color = (rl.Color(0, 220, 80, 255) if
                      self._hud_renderer.steer_ratio_adjustment_allowed() else
                      rl.Color(120, 125, 125, 255))
    else:
      value = f"{self._camera_offset_value:+.2f}"
      status = "CAMERA OFFSET"
      symbol_color = rl.Color(0, 220, 80, 255)

    value_size = 82
    measured = rl.measure_text_ex(font, value, value_size, 0)
    rl.draw_text_ex(font, value,
                    rl.Vector2(rect.x + (rect.width - measured.x) / 2,
                               top + height * (0.24 if status else 0.34)),
                    value_size, 0, rl.WHITE)

    symbol_size = 68
    for symbol, x in (("-", rect.x + rect.width * 0.16),
                      ("+", rect.x + rect.width * 0.84)):
      symbol_measured = rl.measure_text_ex(font, symbol, symbol_size, 0)
      rl.draw_text_ex(font, symbol,
                      rl.Vector2(x - symbol_measured.x / 2,
                                 top + (height - symbol_measured.y) / 2),
                      symbol_size, 0, symbol_color)

    if status:
      status_size = 22
      status_measured = rl.measure_text_ex(font, status, status_size, 0)
      rl.draw_text_ex(font, status,
                      rl.Vector2(rect.x + (rect.width - status_measured.x) / 2,
                                 top + height * 0.72),
                      status_size, 0, rl.WHITE)

  def close(self):
    self._navigation.close()
    super().close()

  def _switch_stream_if_needed(self, sm):
    if sm['carParams'].openpilotLongitudinalControl and WIDE_CAM in self.available_streams:
      v_ego = sm['carState'].vEgo
      if v_ego < WIDE_CAM_MAX_SPEED:
        target = WIDE_CAM
      elif v_ego > ROAD_CAM_MIN_SPEED:
        target = NARROW_ROAD_CAM
      else:
        # Hysteresis zone - keep current stream
        target = self.stream_type
    else:
      target = NARROW_ROAD_CAM

    if self.stream_type != target:
      self.switch_stream(target)

  def _update_calibration(self):
    # Update device camera if not already set
    sm = ui_state.sm
    if not self.device_camera and sm.seen['narrowRoadCameraState'] and sm.seen['deviceState']:
      self.device_camera = DEVICE_CAMERAS[(str(sm['deviceState'].deviceType), str(sm['narrowRoadCameraState'].sensor))]

    # Check if camera calibration data is available and valid
    if not (sm.updated["extrinsicsCalibration"] and sm.valid['extrinsicsCalibration']):
      return

    calib = sm['extrinsicsCalibration']
    if len(calib.rpyCalib) != 3 or calib.calStatus != CALIBRATED:
      return

    # Update view_from_calib matrix
    device_from_calib = rot_from_euler(calib.rpyCalib)
    self.view_from_calib = view_frame_from_device_frame @ device_from_calib

    # Update wide calibration if available
    if hasattr(calib, 'wideFromDeviceEuler') and len(calib.wideFromDeviceEuler) == 3:
      wide_from_device = rot_from_euler(calib.wideFromDeviceEuler)
      self.view_from_wide_calib = view_frame_from_device_frame @ wide_from_device @ device_from_calib

  def _calc_frame_matrix(self, rect: rl.Rectangle) -> np.ndarray:
    cache_key = (
      ui_state.sm.recv_frame['extrinsicsCalibration'],
      int(self._content_rect.width),
      int(self._content_rect.height),
      self.stream_type,
      round(ui_state.sm['carState'].vEgo, 1),
    )

    if cache_key == self._matrix_cache_key and self._cached_matrix is not None:
      return self._cached_matrix

    # Get camera configuration
    device_camera = self.device_camera or DEFAULT_DEVICE_CAMERA
    is_wide_camera = self.stream_type == WIDE_CAM
    intrinsic = device_camera.wide_road.intrinsics if is_wide_camera else device_camera.narrow_road.intrinsics
    calibration = self.view_from_wide_calib if is_wide_camera else self.view_from_calib
    if is_wide_camera:
      zoom = 0.7 * 1.5
    else:
      zoom = np.interp(ui_state.sm['carState'].vEgo, [10, 30], [0.8, 1.0])

    # Calculate transforms for vanishing point
    inf_point = np.array([1000.0, 0.0, 0.0])
    calib_transform = intrinsic @ calibration
    kep = calib_transform @ inf_point

    # Calculate center points and dimensions
    w, h = self._content_rect.width, self._content_rect.height
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    # Ensure zoom views the whole area
    zoom = max(zoom, w / (2 * cx), h / (2 * cy))

    # Calculate max allowed offsets with margins
    margin = 5
    max_x_offset = max(0.0, cx * zoom - w / 2 - margin)
    max_y_offset = max(0.0, cy * zoom - h / 2 - margin)

    # Calculate and clamp offsets to prevent out-of-bounds issues
    try:
      if abs(kep[2]) > 1e-6:
        x_offset = np.clip((kep[0] / kep[2] - cx) * zoom, -max_x_offset, max_x_offset)
        y_offset = np.clip((kep[1] / kep[2] - cy) * zoom + CAM_Y_OFFSET, -max_y_offset, max_y_offset)
      else:
        x_offset, y_offset = 0, 0
    except (ZeroDivisionError, OverflowError):
      x_offset, y_offset = 0, 0

    # Cache the computed transformation matrix to avoid recalculations
    self._matrix_cache_key = cache_key
    self._cached_matrix = np.array([
      [zoom * 2 * cx / w, 0, -x_offset / w * 2],
      [0, zoom * 2 * cy / h, -y_offset / h * 2],
      [0, 0, 1.0]
    ])

    # built without rect.x/y so cache stays hot during scroll. ModelRenderer adds offset at draw time
    video_transform = np.array([
      [zoom, 0.0, (w / 2 - x_offset) - (cx * zoom)],
      [0.0, zoom, (h / 2 - y_offset) - (cy * zoom)],
      [0.0, 0.0, 1.0]
    ])
    self._model_renderer.set_transform(video_transform @ calib_transform)

    return self._cached_matrix

  def show_event(self):
    if gui_app.sunnypilot_ui():
      ui_state.reset_onroad_sleep_timer(OnroadTimerStatus.RESUME)

  def hide_event(self):
    if gui_app.sunnypilot_ui():
      ui_state.reset_onroad_sleep_timer(OnroadTimerStatus.PAUSE)


if __name__ == "__main__":
  gui_app.init_window("OnRoad Camera View")
  road_camera_view = AugmentedRoadView(lambda: None, stream_type=NARROW_ROAD_CAM)
  print("***press space to switch camera view***")
  try:
    for _ in gui_app.render():
      ui_state.update()
      if rl.is_key_released(rl.KeyboardKey.KEY_SPACE):
        if WIDE_CAM in road_camera_view.available_streams:
          stream = NARROW_ROAD_CAM if road_camera_view.stream_type == WIDE_CAM else WIDE_CAM
          road_camera_view.switch_stream(stream)
      road_camera_view.render(rl.Rectangle(0, 0, gui_app.width, gui_app.height))
  finally:
    road_camera_view.close()
