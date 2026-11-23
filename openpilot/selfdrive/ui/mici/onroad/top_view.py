"""Bird's-eye 3D view in the style of the sunnylink.wiki driving sim.

Everything comes from what openpilot actually perceives through the road camera:
  - road surface between the model's road edges
  - lane lines from the model (fading with their confidence)
  - the planned path (teal, like the sim) up to the lead car
  - the lead car from the vision model (modelV2.leadsV3), at its measured distance and lateral position
Nothing is invented: the vision model only reports the lead in our lane, so that is the only other car drawn.

World frame for rendering (raylib, right-handed, Y up): X = right, Y = up, Z = backwards (forward is -Z).
Model frame (openpilot calibrated): x forward, y right, z down.
"""
import math

import numpy as np
import pyray as rl

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.ui_state import ui_state, UIStatus
from openpilot.system.ui.lib.application import gui_app

# Night palette from the sim (highway)
BG = (21, 29, 43)
ROAD = (45, 56, 73)
SHOULDER = (31, 42, 57)
LINE = (204, 214, 227)
PATH = (40, 182, 154)  # sim's teal path
PATH_ALPHA = 0.38

EGO_PAINT = (246, 248, 247)  # sim default: pearl white
LEAD_PAINT = (137, 157, 164)
GLASS = (52, 71, 78)
UNDER = (59, 73, 75)
TRIM = (88, 103, 103)
TIRE = (38, 49, 49)
RIM = (174, 186, 186)

FOG_NEAR, FOG_FAR = 30.0, 95.0
BEHIND = 20.0  # m of road drawn behind us
EGO_CENTER_X = -1.2  # the camera sits near the windshield, the car's centre is behind it
CAR_LEN = 4.65
LEAD_MIN_PROB = 0.5

# Bird's-eye chase camera, relative to our car (metres)
CAM_HEIGHT = 14.0
CAM_BACK = 13.8
CAM_LOOK_AHEAD = 7.0
CAM_FOVY = 46.0

# Face shading for flat low-poly lighting (sun from above, slightly left/front)
SHADE = {"top": 1.0, "left": 0.84, "right": 0.66, "front": 0.92, "back": 0.74, "bottom": 0.5}


def _w(x: float, y: float, h: float = 0.0) -> rl.Vector3:
  """Model-frame (x fwd, y right) + height -> raylib world."""
  return rl.Vector3(y, h, -x)


def _rgba(rgb, a: float = 1.0, k: float = 1.0) -> rl.Color:
  return rl.Color(min(255, int(rgb[0] * k)), min(255, int(rgb[1] * k)), min(255, int(rgb[2] * k)), int(max(0.0, min(1.0, a)) * 255))


def _tri(a, b, c, col):
  rl.draw_triangle_3d(a, b, c, col)


def _quad(a, b, c, d, col):
  _tri(a, b, c, col)
  _tri(a, c, d, col)


class TopView:
  def __init__(self):
    self._rt: rl.RenderTexture | None = None
    self._rt_size = (0, 0)
    self._odometer = 0.0
    self._fog_ref = rl.Vector3(0, 0, 0)

    self._path = np.zeros((0, 2), dtype=np.float32)
    self._lanes = [np.zeros((0, 2), dtype=np.float32) for _ in range(4)]
    self._edges = [np.zeros((0, 2), dtype=np.float32) for _ in range(2)]
    self._lane_probs = np.zeros(4, dtype=np.float32)
    self._edge_stds = np.ones(2, dtype=np.float32)

    dt = 1 / gui_app.target_fps
    self._lead_x = FirstOrderFilter(0.0, 0.15, dt, initialized=False)
    self._lead_y = FirstOrderFilter(0.0, 0.15, dt, initialized=False)
    self._lead_alpha = FirstOrderFilter(0.0, 0.15, dt)
    self._camera_offset = 0.0
    self._counter = 0

  # ---------------------------------------------------------------- data
  def _update_data(self):
    sm = ui_state.sm
    if self._counter % 180 == 0:
      self._camera_offset = float(ui_state.params.get("CameraOffset", return_default=True)) if ui_state.active_bundle else 0.0
    self._counter += 1

    v_ego = max(0.0, sm['carState'].vEgo)
    self._odometer += v_ego / gui_app.target_fps

    if sm.updated['modelV2'] or self._path.size == 0:
      model = sm['modelV2']
      off = self._camera_offset
      self._path = np.column_stack((model.position.x, np.array(model.position.y) + off)).astype(np.float32)
      for i, line in enumerate(model.laneLines):
        self._lanes[i] = np.column_stack((line.x, np.array(line.y) + off)).astype(np.float32)
      for i, edge in enumerate(model.roadEdges):
        self._edges[i] = np.column_stack((edge.x, np.array(edge.y) + off)).astype(np.float32)
      self._lane_probs = np.array(model.laneLineProbs, dtype=np.float32)
      self._edge_stds = np.array(model.roadEdgeStds, dtype=np.float32)

    # lead from the vision model (camera), not radar
    leads = sm['modelV2'].leadsV3
    lead = leads[0] if len(leads) else None
    visible = lead is not None and lead.prob > LEAD_MIN_PROB and len(lead.x) > 0 and 0.0 < lead.x[0] < 120.0
    if visible:
      x, y = float(lead.x[0]), float(lead.y[0]) + self._camera_offset
      if abs(y - self._lead_y.x) > 3.0 or abs(x - self._lead_x.x) > 15.0:
        self._lead_x.initialized = self._lead_y.initialized = False  # a different car: snap, don't slide
      self._lead_x.update(x)
      self._lead_y.update(y)
    self._lead_alpha.update(1.0 if visible else 0.0)

  # ---------------------------------------------------------------- helpers
  def _fog(self, rgb, a: float, x: float, y: float = 0.0) -> rl.Color:
    """Fade toward the background with distance from the camera, like the sim's fog."""
    p = _w(x, y)
    d = math.dist((p.x, p.y, p.z), (self._fog_ref.x, self._fog_ref.y, self._fog_ref.z))
    t = min(1.0, max(0.0, (d - FOG_NEAR) / (FOG_FAR - FOG_NEAR)))
    return rl.Color(int(rgb[0] + (BG[0] - rgb[0]) * t), int(rgb[1] + (BG[1] - rgb[1]) * t),
                    int(rgb[2] + (BG[2] - rgb[2]) * t), int(255 * max(0.0, min(1.0, a))))  # colour carries the fog; alpha stays put (no seams)

  @staticmethod
  def _extend_back(line: np.ndarray) -> np.ndarray:
    """Model lines start at us; continue them straight behind so the chase camera sees road under the car."""
    if line.shape[0] == 0:
      return line
    back = np.array([[-BEHIND, line[0, 1]], [min(-0.5, line[0, 0] - 0.5), line[0, 1]]], dtype=np.float32)
    return np.vstack((back, line[line[:, 0] >= 0]))

  def _ribbon(self, line: np.ndarray, half: float, x0: float, x1: float, rgb, a: float, h: float, dash: tuple | None = None):
    if line.shape[0] < 2 or x1 <= x0:
      return
    xs = np.arange(x0, x1, 2.0, dtype=np.float32)
    xs = np.append(xs, np.float32(x1))
    ys = np.interp(xs, line[:, 0], line[:, 1])
    for i in range(len(xs) - 1):
      xa, xb = float(xs[i]), float(xs[i + 1])
      if dash is not None:
        period, on = dash
        mid = (xa + xb) / 2 + self._odometer
        if (mid % period) > on:
          continue
      ya, yb = float(ys[i]), float(ys[i + 1])
      col = self._fog(rgb, a, (xa + xb) / 2, (ya + yb) / 2)
      if col.a == 0:
        continue
      _quad(_w(xa, ya - half, h), _w(xb, yb - half, h), _w(xb, yb + half, h), _w(xa, ya + half, h), col)

  def _box(self, cx, cy, cz, sx, sy, sz, rgb, a=1.0):
    """Axis-aligned box in raylib world, flat-shaded per face."""
    x0, x1 = cx - sx / 2, cx + sx / 2
    y0, y1 = cy - sy / 2, cy + sy / 2
    z0, z1 = cz - sz / 2, cz + sz / 2
    V = rl.Vector3
    _quad(V(x0, y1, z0), V(x1, y1, z0), V(x1, y1, z1), V(x0, y1, z1), _rgba(rgb, a, SHADE["top"]))
    _quad(V(x0, y0, z0), V(x0, y1, z0), V(x0, y1, z1), V(x0, y0, z1), _rgba(rgb, a, SHADE["left"]))
    _quad(V(x1, y0, z0), V(x1, y0, z1), V(x1, y1, z1), V(x1, y1, z0), _rgba(rgb, a, SHADE["right"]))
    _quad(V(x0, y0, z0), V(x1, y0, z0), V(x1, y1, z0), V(x0, y1, z0), _rgba(rgb, a, SHADE["front"]))
    _quad(V(x0, y0, z1), V(x0, y1, z1), V(x1, y1, z1), V(x1, y0, z1), _rgba(rgb, a, SHADE["back"]))

  def _car(self, x_center: float, y: float, paint, a: float = 1.0, brake: bool = False):
    """Low-poly car after the sim's buildCar: body, underbody, hood, glass cabin, lights, wheels.
    Sim coordinates (x right, y up, z forward) are mapped into our world around the car's centre."""
    W = _w(x_center, y)
    ox, oz = W.x, W.z

    def box(x, yy, z, sx, sy, sz, rgb):
      self._box(ox + x, yy, oz - z, sx, sy, sz, rgb, a)

    # contact shadow
    _quad(rl.Vector3(ox - 1.15, 0.02, oz - 2.6), rl.Vector3(ox + 1.15, 0.02, oz - 2.6),
          rl.Vector3(ox + 1.15, 0.02, oz + 2.6), rl.Vector3(ox - 1.15, 0.02, oz + 2.6), rl.Color(0, 0, 0, int(110 * a)))

    box(0, 0.57, 0, 1.88, 0.62, 4.65, paint)  # body
    box(0, 0.38, 0, 1.79, 0.16, 4.4, UNDER)  # underbody
    box(0, 0.85, 1.52, 1.72, 0.2, 1.35, paint)  # hood

    # glass cabin: sloped windshield and rear glass, curved roof (as a flat-shaded prism)
    profile = [(-1.45, 0.8), (-0.9, 1.42), (0.67, 1.43), (1.33, 0.82)]  # (z forward, height)
    half = 0.775
    g = [(rl.Vector3(ox - half, hh, oz - z), rl.Vector3(ox + half, hh, oz - z)) for z, hh in profile]
    shades = (SHADE["back"], SHADE["top"], SHADE["front"])
    for i in range(3):
      (l0, r0), (l1, r1) = g[i], g[i + 1]
      _quad(l0, r0, r1, l1, _rgba(GLASS, a, shades[i] * 1.1))
    for side, k in ((0, SHADE["left"]), (1, SHADE["right"])):
      pts = [p[side] for p in g]
      _quad(pts[0], pts[1], pts[2], pts[3], _rgba(GLASS, a, k))
    for s in (-1, 1):
      box(s * 0.73, 1.49, -0.13, 0.07, 0.08, 1.45, paint)  # roof rails
      box(s * 1.02, 1.0, 0.78, 0.28, 0.14, 0.35, paint)  # mirrors
      box(s * 0.66, 0.73, 2.26, 0.52, 0.095, 0.04, (246, 255, 249))  # headlights
      box(s * 0.63, 0.77, -2.26, 0.57, 0.11, 0.05, (255, 34, 20) if brake else (64, 19, 17))  # tail lights
    box(0, 1.45, -1.35, 0.44, 0.045, 0.05, (255, 34, 20) if brake else (64, 19, 17))  # CHMSL

    for z in (1.45, -1.48):
      for s in (-1, 1):
        c = rl.Vector3(ox + s * 0.91, 0.36, oz - z)
        rl.draw_cylinder_ex(rl.Vector3(c.x - 0.12, c.y, c.z), rl.Vector3(c.x + 0.12, c.y, c.z), 0.35, 0.35, 10, _rgba(TIRE, a))
        rl.draw_cylinder_ex(rl.Vector3(c.x + s * 0.12, c.y, c.z), rl.Vector3(c.x + s * 0.135, c.y, c.z), 0.23, 0.23, 8, _rgba(RIM, a))

  # ---------------------------------------------------------------- render
  def _scene(self):
    status = ui_state.status
    sm = ui_state.sm

    # road: shoulder band, then the surface between the model's road edges
    left, right = self._extend_back(self._edges[0]), self._extend_back(self._edges[1])
    if left.shape[0] >= 2 and right.shape[0] >= 2:
      far = float(min(left[-1, 0], right[-1, 0], 140.0))
      xs = np.append(np.arange(-BEHIND, far, 3.0, dtype=np.float32), np.float32(far))
      yl = np.interp(xs, left[:, 0], left[:, 1])
      yr = np.interp(xs, right[:, 0], right[:, 1])
      for i in range(len(xs) - 1):
        xa, xb = float(xs[i]), float(xs[i + 1])
        mid = (xa + xb) / 2
        col_s = self._fog(SHOULDER, 1.0, mid)
        _quad(_w(xa, yl[i] - 1.2), _w(xb, yl[i + 1] - 1.2), _w(xb, yr[i + 1] + 1.2), _w(xa, yr[i] + 1.2), col_s)
        col = self._fog(ROAD, 1.0, mid)
        _quad(_w(xa, yl[i], 0.01), _w(xb, yl[i + 1], 0.01), _w(xb, yr[i + 1], 0.01), _w(xa, yr[i], 0.01), col)

    # road edges (soft) and lane lines: ours solid, the outer ones dashed and flowing with distance
    for i, edge in enumerate(self._edges):
      e = self._extend_back(edge)
      if e.shape[0]:
        self._ribbon(e, 0.06, -BEHIND, float(min(e[-1, 0], 140.0)), LINE, 0.38 * (1.0 - min(1.0, float(self._edge_stds[i]))), 0.03)
    for i, lane in enumerate(self._lanes):
      ln = self._extend_back(lane)
      if ln.shape[0] == 0:
        continue
      prob = float(self._lane_probs[i]) if i < len(self._lane_probs) else 0.0
      dash = None if i in (1, 2) else (12.0, 3.0)
      self._ribbon(ln, 0.06, -BEHIND, float(min(ln[-1, 0], 120.0)), LINE, 0.85 * min(1.0, prob * 1.4), 0.04, dash)

    # planned path: teal, from our front bumper to the lead (or as far as the model plans)
    lead_a = self._lead_alpha.x if self._lead_x.initialized else 0.0
    path = self._path
    if path.shape[0] >= 2 and status != UIStatus.DISENGAGED:
      end = float(min(path[-1, 0], 85.0))
      if lead_a > 0.5:
        end = min(end, self._lead_x.x - 0.5)
      self._ribbon(path, 0.75, EGO_CENTER_X + 2.4, end, PATH, PATH_ALPHA, 0.06)

    # lead car from the camera model
    if lead_a > 0.02:
      self._car(self._lead_x.x + CAR_LEN / 2, self._lead_y.x, LEAD_PAINT, lead_a)

    # our car
    cc = sm['carControl']
    braking = sm['carState'].brakePressed or (cc.longActive and cc.actuators.accel < -0.5)
    y0 = float(np.interp(0.0, path[:, 0], path[:, 1])) if path.shape[0] >= 2 else 0.0
    self._car(EGO_CENTER_X, y0, EGO_PAINT, 1.0, braking)

  def prepare(self, rect: rl.Rectangle):
    """Render the 3D scene into an offscreen texture. Call before any scissor mode is active."""
    self._update_data()
    w, h = max(1, int(rect.width)), max(1, int(rect.height))
    if self._rt is None or self._rt_size != (w, h):
      if self._rt is not None:
        rl.unload_render_texture(self._rt)
      self._rt = rl.load_render_texture(w, h)
      rl.set_texture_filter(self._rt.texture, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
      self._rt_size = (w, h)

    cam_pos = _w(EGO_CENTER_X - CAM_BACK, 0.0, CAM_HEIGHT)
    target = _w(EGO_CENTER_X + CAM_LOOK_AHEAD, 0.0, 0.0)
    camera = rl.Camera3D(cam_pos, target, rl.Vector3(0, 1, 0), CAM_FOVY, rl.CameraProjection.CAMERA_PERSPECTIVE)
    self._fog_ref = cam_pos

    outer = getattr(gui_app, "_render_texture", None)
    rl.begin_texture_mode(self._rt)
    rl.clear_background(_rgba(BG))
    rl.begin_mode_3d(camera)
    rl.rl_disable_backface_culling()
    self._scene()
    rl.rl_draw_render_batch_active()  # flush while culling is still off
    rl.rl_enable_backface_culling()
    rl.end_mode_3d()
    rl.end_texture_mode()
    if outer is not None:
      rl.begin_texture_mode(outer)  # hand the app's own framebuffer back

  def draw(self, rect: rl.Rectangle):
    if self._rt is None:
      return
    tex = self._rt.texture
    # render textures are stored upside down
    rl.draw_texture_rec(tex, rl.Rectangle(0, 0, tex.width, -tex.height), rl.Vector2(rect.x, rect.y), rl.WHITE)
