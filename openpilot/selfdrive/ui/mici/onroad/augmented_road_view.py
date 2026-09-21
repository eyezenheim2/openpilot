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
from openpilot.selfdrive.ui.mici.onroad.model_renderer import ModelRenderer
from openpilot.selfdrive.ui.mici.onroad.confidence_ball import ConfidenceBall
from openpilot.selfdrive.ui.mici.onroad.cameraview import CameraView
from openpilot.system.ui.lib.application import FontWeight, gui_app, MousePos, MouseEvent, TextAlignment, TextAlignmentVertical
from openpilot.system.ui.widgets.label import UnifiedLabel
from openpilot.system.ui.widgets import Widget
from openpilot.common.filter_simple import BounceFilter
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

WIDE_CAM_MAX_SPEED = 5.0  # m/s (10 mph)
ROAD_CAM_MIN_SPEED = 10  # m/s (25 mph)

CAM_Y_OFFSET = 20

# CAMERA OFFSET DISPLAY - edit these font sizes (pixels).
CAMERA_OFFSET_VALUE_FONT_SIZE = 72
CAMERA_OFFSET_SYMBOL_FONT_SIZE = 58

# SPEEDOMETER DISPLAY - centered in the marked area between 8 and 9.
SPEEDOMETER_VALUE_FONT_SIZE = 112
SPEEDOMETER_UNIT_FONT_SIZE = 36
SPEEDOMETER_CENTER_X = 0.275  # Fraction of HUD width.
SPEEDOMETER_CENTER_Y = 0.635  # Fraction of HUD height.
SPEEDOMETER_UNIT_GAP = 6  # Pixels between the value and unit.
# LARGE CENTERED SPEEDOMETER - selected by a 0.6-second long press.
SPEEDOMETER_LARGE_VALUE_FONT_SIZE = 224
SPEEDOMETER_LARGE_UNIT_FONT_SIZE = 48
SPEEDOMETER_LARGE_CENTER_X = 0.50
SPEEDOMETER_LARGE_CENTER_Y = 0.50




class SpeedometerHudRenderer(HudRenderer):
  show_current_speed = False

  def _draw_set_speed(self, rect: rl.Rectangle) -> None:
    if self.show_current_speed:
      # Current speed occupies this slot; do not draw set speed underneath it.
      self._set_speed_alpha_filter.update(0)
      return
    super()._draw_set_speed(rect)


class BookmarkIcon(Widget):
  PEEK_THRESHOLD = 50  # If icon peeks out this much, snap it fully visible
  FULL_VISIBLE_OFFSET = 200  # How far onscreen when fully visible
  HIDDEN_OFFSET = -50  # How far offscreen when hidden

  def __init__(self, bookmark_callback):
    super().__init__()
    self._bookmark_callback = bookmark_callback
    self._icon = gui_app.texture("icons_mici/onroad/bookmark.png", 180, 180)
    self._offset_filter = BounceFilter(0.0, 0.1, 1 / gui_app.target_fps)

    # State
    self._interacting = False
    self._state = BookmarkState.HIDDEN
    self._swipe_start_x = 0.0
    self._swipe_current_x = 0.0
    self._is_swiping = False
    self._is_swiping_left: bool = False
    self._triggered_time: float = 0.0

  def is_swiping_left(self) -> bool:
    """Check if currently swiping left (for scroller to disable)."""
    return self._is_swiping_left

  def interacting(self):
    interacting, self._interacting = self._interacting, False
    return interacting

  def _update_state(self):
    if self._state == BookmarkState.DRAGGING:
      # Allow pulling past activated position with rubber band effect
      swipe_offset = self._swipe_start_x - self._swipe_current_x
      swipe_offset = min(swipe_offset, self.FULL_VISIBLE_OFFSET + 50)
      self._offset_filter.update(swipe_offset)

    elif self._state == BookmarkState.TRIGGERED:
      # Continue animating to fully visible
      self._offset_filter.update(self.FULL_VISIBLE_OFFSET)
      # Stay in TRIGGERED state for 1 second
      if rl.get_time() - self._triggered_time >= 1.5:
        self._state = BookmarkState.HIDDEN

    elif self._state == BookmarkState.HIDDEN:
      self._offset_filter.update(self.HIDDEN_OFFSET)

      if self._offset_filter.x < 1e-3:
        self._interacting = False

  def _handle_mouse_event(self, mouse_event: MouseEvent):
    if not ui_state.started:
      return

    if mouse_event.left_pressed:
      # Store relative position within widget
      self._swipe_start_x = mouse_event.pos.x
      self._swipe_current_x = mouse_event.pos.x
      self._is_swiping = True
      self._is_swiping_left = False
      self._state = BookmarkState.DRAGGING

    elif mouse_event.left_down and self._is_swiping:
      self._swipe_current_x = mouse_event.pos.x
      swipe_offset = self._swipe_start_x - self._swipe_current_x
      self._is_swiping_left = swipe_offset > 0
      if self._is_swiping_left:
        self._interacting = True

    elif mouse_event.left_released:
      if self._is_swiping:
        swipe_distance = self._swipe_start_x - self._swipe_current_x

        # If peeking past threshold, transition to animating to fully visible and bookmark
        if swipe_distance > self.PEEK_THRESHOLD:
          self._state = BookmarkState.TRIGGERED
          self._triggered_time = rl.get_time()
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
      rl.draw_texture_ex(self._icon, rl.Vector2(icon_x, icon_y), 0.0, 1.0, rl.WHITE)


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
    self._face_press: tuple[float, float, float] | None = None
    self._hold_toggled = False
    self._offset_press_active = False
    self._offset_controls_until = 0.0
    self._camera_offset_value = float(ui_state.params.get('CameraOffset', return_default=True))
    self._speedometer_visible = True
    self._speedometer_large = False

    # Bookmark icon with swipe gesture
    self._bookmark_icon = BookmarkIcon(bookmark_callback)

    self._model_renderer = ModelRenderer()
    self._hud_renderer = SpeedometerHudRenderer()
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

    if not ui_state.started:
      self._face_press = None
      self._hold_toggled = False
      self._offset_controls_until = 0.0

    # update offroad label
    if ui_state.panda_type == log.PandaState.PandaType.unknown:
      self._offroad_label.set_text("system booting")
    elif ui_state.ignition and not ui_state.started:
      self._offroad_label.set_text("openpilot can't start\ncheck alerts")
    else:
      self._offroad_label.set_text("start the car to\nuse sunnypilot")

  def _inside_face(self, pos: MousePos) -> bool:
    rect = self.rect
    return (rect.x <= pos.x < rect.x + rect.width - SIDE_PANEL_WIDTH and
            rect.y <= pos.y < rect.y + rect.height)

  def _handle_mouse_press(self, mouse_pos: MousePos):
    if not ui_state.started:
      super()._handle_mouse_press(mouse_pos)
      return
    self._face_press = None
    if not self._inside_face(mouse_pos):
      return
    now = rl.get_time()
    self._face_press = (now, mouse_pos.x, mouse_pos.y)
    self._hold_toggled = False
    self._offset_press_active = now < self._offset_controls_until

  def _toggle_speedometer_for_hold(self, now: float) -> None:
    if self._face_press is not None and not self._hold_toggled and now - self._face_press[0] >= 0.6:
      self._speedometer_large = not self._speedometer_large
      self._speedometer_visible = True
      self._hold_toggled = True
      self._offset_controls_until = 0.0

  def _handle_mouse_event(self, event: MouseEvent):
    super()._handle_mouse_event(event)
    press = self._face_press
    if press is None:
      return
    if (not ui_state.started or not self._inside_face(event.pos) or
        (event.pos.x - press[1]) ** 2 + (event.pos.y - press[2]) ** 2 > 30 ** 2):
      self._face_press = None
      return
    if event.left_down and not event.left_pressed:
      self._toggle_speedometer_for_hold(rl.get_time())

  def _handle_mouse_release(self, mouse_pos: MousePos):
    if not ui_state.started:
      self._face_press = None
      super()._handle_mouse_release(mouse_pos)
      return
    press = self._face_press
    if press is None:
      return
    if (self._bookmark_icon.interacting() or not self._inside_face(mouse_pos) or
        (mouse_pos.x - press[1]) ** 2 + (mouse_pos.y - press[2]) ** 2 > 30 ** 2):
      self._face_press = None
      return
    now = rl.get_time()
    self._toggle_speedometer_for_hold(now)
    self._face_press = None
    if self._hold_toggled:
      return
    # Offset controls take priority over the speedometer.
    self._speedometer_visible = False
    # First tap reveals controls; subsequent taps adjust the selected half.
    if self._offset_press_active and ui_state.active_bundle is not None:
      midpoint = self.rect.x + (self.rect.width - SIDE_PANEL_WIDTH) / 2
      step = -0.01 if mouse_pos.x < midpoint else 0.01
      value = float(ui_state.params.get('CameraOffset', return_default=True))
      value = round(max(-0.35, min(0.35, value + step)), 2)
      ui_state.params.put('CameraOffset', value)
      self._camera_offset_value = value
      self._model_renderer._camera_offset = value
      self._model_renderer._transform_dirty = True
    else:
      self._camera_offset_value = float(ui_state.params.get('CameraOffset', return_default=True))
    self._offset_controls_until = now + 5.0

  def _draw_touch_controls(self):
    if rl.get_time() >= self._offset_controls_until:
      if self._offset_controls_until > 0:
        self._offset_controls_until = 0.0
        self._speedometer_visible = True
      return
    rect = self._content_rect
    font = gui_app.font(FontWeight.DISPLAY_REGULAR)
    active = ui_state.active_bundle is not None
    color = rl.WHITE if active else rl.Color(160, 160, 160, 255)
    # Font sizes are configured near the top of this file.
    height = max(150.0, CAMERA_OFFSET_VALUE_FONT_SIZE + 32, CAMERA_OFFSET_SYMBOL_FONT_SIZE + 32)
    top = rect.y + (rect.height - height) / 2
    rl.draw_rectangle_rec(rl.Rectangle(rect.x, top, rect.width, height), rl.BLACK)
    value = self._camera_offset_value
    value = 0.0 if abs(value) < 0.005 else value
    for label, fraction, size in (
      ('-', 0.18, CAMERA_OFFSET_SYMBOL_FONT_SIZE),
      (f'{value:.2f}', 0.50, CAMERA_OFFSET_VALUE_FONT_SIZE),
      ('+', 0.82, CAMERA_OFFSET_SYMBOL_FONT_SIZE),
    ):
      measured = rl.measure_text_ex(font, label, size, 0)
      pos = rl.Vector2(rect.x + rect.width * fraction - measured.x / 2,
                       top + (height - measured.y) / 2)
      rl.draw_text_ex(font, label, pos, size, 0, color)

  def _draw_speedometer(self):
    if not self._speedometer_visible:
      return
    sm = ui_state.sm
    if (not sm.valid['carState'] or not sm.alive['carState'] or
        sm.recv_frame['carState'] <= ui_state.started_frame):
      return
    # HudRenderer updates speed using cluster speed and the selected units.
    rect = self._content_rect
    if self._speedometer_large:
      value_size, unit_size = SPEEDOMETER_LARGE_VALUE_FONT_SIZE, SPEEDOMETER_LARGE_UNIT_FONT_SIZE
      x_fraction, y_fraction = SPEEDOMETER_LARGE_CENTER_X, SPEEDOMETER_LARGE_CENTER_Y
    else:
      value_size, unit_size = SPEEDOMETER_VALUE_FONT_SIZE, SPEEDOMETER_UNIT_FONT_SIZE
      x_fraction, y_fraction = SPEEDOMETER_CENTER_X, SPEEDOMETER_CENTER_Y
    center_x = rect.x + rect.width * x_fraction
    center_y = rect.y + rect.height * y_fraction
    total_height = value_size + SPEEDOMETER_UNIT_GAP + unit_size
    y = center_y - total_height / 2
    for label, size, weight in (
      (str(round(self._hud_renderer.speed)), value_size, FontWeight.DISPLAY),
      ('km/h' if ui_state.is_metric else 'mph', unit_size, FontWeight.SEMI_BOLD),
    ):
      font = gui_app.font(weight)
      measured = rl.measure_text_ex(font, label, size, 0)
      pos = rl.Vector2(center_x - measured.x / 2, y)
      rl.draw_text_ex(font, label, rl.Vector2(pos.x + 2, pos.y + 2), size, 0, rl.BLACK)
      rl.draw_text_ex(font, label, pos, size, 0, rl.WHITE)
      y += size + SPEEDOMETER_UNIT_GAP

  def _render(self, _):
    # Offroad retains the normal startup/status screen.
    if not ui_state.started:
      rl.draw_rectangle_rec(self.rect, rl.BLACK)
      self._offroad_label.render(self._rect)
      return

    self._switch_stream_if_needed(ui_state.sm)

    # Update calibration before rendering
    self._update_calibration()

    # Create inner content area with border padding
    self._content_rect = rl.Rectangle(
      self.rect.x,
      self.rect.y,
      self.rect.width - SIDE_PANEL_WIDTH,
      self.rect.height,
    )

    # Update and draw the side indicator independently of face visibility.
    self._confidence_ball.render(self.rect)

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

    # Draw all UI overlays
    self._model_renderer.render(self._content_rect)

    # Fade out bottom of overlays for looks
    rl.draw_texture_ex(self._fade_texture, rl.Vector2(self._content_rect.x, self._content_rect.y), 0.0, 1.0, rl.WHITE)

    # Continuous triangle animation sits below speed, driver monitoring, steering, and alerts.
    self._confidence_ball.draw_hud_face(self._content_rect)
    self._draw_touch_controls()

    alert_to_render, not_animating_out = self._alert_renderer.will_render()

    # Hide DMoji when disengaged unless AlwaysOnDM is enabled
    should_draw_dmoji = (not self._hud_renderer.drawing_top_icons() and
                         (ui_state.status != UIStatus.DISENGAGED or ui_state.always_on_dm))
    self._driver_state_renderer.set_should_draw(should_draw_dmoji)
    self._driver_state_renderer.set_position(self._rect.x + 16, self._rect.y + 10)
    self._driver_state_renderer.render()

    self._hud_renderer.show_current_speed = self._speedometer_visible
    self._hud_renderer.set_can_draw_top_icons(self._speedometer_visible and alert_to_render is None)
    self._hud_renderer.set_wheel_critical_icon(alert_to_render is not None and not not_animating_out and
                                               alert_to_render.visual_alert == car.CarControl.HUDControl.VisualAlert.steerRequired)
    self._hud_renderer.render(self._content_rect)
    self._draw_speedometer()
    self._alert_renderer.render(self._content_rect)

    # Draw fake rounded border
    rl.draw_rectangle_rounded_lines_ex(self._content_rect, 0.2 * 1.02, 10, 50, rl.BLACK)

    # End clipping region
    rl.end_scissor_mode()

    # Custom UI extension point - add custom overlays here
    # Use self._content_rect for positioning within camera bounds

    self._bookmark_icon.render(self.rect)

  def _switch_stream_if_needed(self, sm):
    if sm['selfdriveState'].experimentalMode and WIDE_CAM in self.available_streams:
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

  def close(self):
    self._confidence_ball.release_face_texture()
    super().close()

  def show_event(self):
    if gui_app.sunnypilot_ui():
      ui_state.reset_onroad_sleep_timer(OnroadTimerStatus.RESUME)

  def hide_event(self):
    self._confidence_ball.release_face_texture()
    self._face_press = None
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
