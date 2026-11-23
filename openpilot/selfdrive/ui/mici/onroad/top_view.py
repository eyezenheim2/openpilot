"""Bird's-eye view in the style of the sunnylink.wiki driving sim.

Everything comes from what openpilot actually perceives through the road camera:
  - road surface between the model's road edges
  - lane lines from the model (fading with their confidence)
  - the planned path (teal, like the sim) up to the lead car
  - the lead car from the vision model (modelV2.leadsV3), at its measured distance and lateral position
Nothing is invented: the vision model only reports the lead in our lane, so that is the only other car drawn.

Rendering is plain 2D: the 3D projection is done here in numpy (like the sim's SVG renderer) and the
result is drawn with ordinary 2D triangles in back-to-front order. No render textures, no 3D mode and no
depth buffer, so it uses exactly the same GL path as the rest of the HUD.

World frame: X = right, Y = up, Z = backwards (forward is -Z). Model frame: x forward, y right.
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
TIRE = (38, 49, 49)
BRAKE_ON, BRAKE_OFF = (255, 34, 20), (64, 19, 17)

FOG_NEAR, FOG_FAR = 30.0, 95.0
FOG_BANDS = 10  # fog is stepped so long ribbons go out in a few draw calls
BEHIND = 12.0  # m of road drawn behind us (the virtual camera sits further back than this)
EGO_CENTER_X = -1.2  # the camera sits near the windshield, the car's centre is behind it
CAR_LEN = 4.65
LEAD_MIN_PROB = 0.5
NEAR = 0.5  # m: anything closer to the virtual camera is skipped

# Bird's-eye chase camera, relative to our car (metres)
CAM_HEIGHT = 14.0
CAM_BACK = 13.8
CAM_LOOK_AHEAD = 7.0
CAM_FOVY = 46.0

# Flat low-poly lighting: sun from above, slightly left/front
LIGHT = np.array([-0.35, 0.85, -0.4])
LIGHT /= np.linalg.norm(LIGHT)


def _wp(x, y, h=0.0):
  """Model-frame (x fwd, y right) + height -> world point."""
  return np.array([y, h, -x], dtype=np.float64)


def _rgba(rgb, a: float = 1.0, k: float = 1.0) -> rl.Color:
  return rl.Color(min(255, int(rgb[0] * k)), min(255, int(rgb[1] * k)), min(255, int(rgb[2] * k)), int(max(0.0, min(1.0, a)) * 255))


class _Camera:
  def __init__(self, eye: np.ndarray, target: np.ndarray, fovy_deg: float, rect: rl.Rectangle):
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, np.array([0.0, 1.0, 0.0]))
    r /= np.linalg.norm(r)
    self.eye, self.f, self.r, self.u = eye, f, r, np.cross(r, f)
    self.focal = (rect.height / 2) / math.tan(math.radians(fovy_deg) / 2)
    self.cx, self.cy = rect.x + rect.width / 2, rect.y + rect.height / 2

  def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N,3) world -> (N,2) screen, (N,) depth along the view direction."""
    d = pts - self.eye
    z = d @ self.f
    zs = np.where(z > 1e-6, z, 1e-6)
    sx = self.cx + self.focal * (d @ self.r) / zs
    sy = self.cy - self.focal * (d @ self.u) / zs
    return np.column_stack((sx, sy)), z


def _vec2_array(pts: np.ndarray):
  """numpy (N,2) -> raylib Vector2* (the buffer stays alive as long as the returned tuple)."""
  pts = np.ascontiguousarray(pts, dtype=np.float32)
  arr = rl.ffi.new("Vector2[]", len(pts))
  rl.ffi.memmove(arr, rl.ffi.from_buffer(pts), pts.nbytes)
  return arr, rl.ffi.cast("Vector2 *", arr)


def _strip2d(left: np.ndarray, right: np.ndarray, col: rl.Color):
  """One draw call for a ribbon: interleave its two edges into a triangle strip."""
  n = len(left)
  if n < 2:
    return
  pts = np.empty((2 * n, 2), dtype=np.float32)
  pts[0::2], pts[1::2] = left, right
  _keep, ptr = _vec2_array(pts)
  rl.draw_triangle_strip(ptr, 2 * n, col)


def _fan2d(pts: np.ndarray, col: rl.Color):
  _keep, ptr = _vec2_array(pts)
  rl.draw_triangle_fan(ptr, len(pts), col)


class TopView:
  def __init__(self):
    self._odometer = 0.0
    self._cam: _Camera | None = None
    self._car_template = self._build_car_template()

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
  @staticmethod
  def _fog(rgb, a: float, depth: float) -> rl.Color:
    """Fade toward the background with distance from the camera, like the sim's fog."""
    t = min(1.0, max(0.0, (depth - FOG_NEAR) / (FOG_FAR - FOG_NEAR)))
    return rl.Color(int(rgb[0] + (BG[0] - rgb[0]) * t), int(rgb[1] + (BG[1] - rgb[1]) * t),
                    int(rgb[2] + (BG[2] - rgb[2]) * t), int(255 * max(0.0, min(1.0, a))))

  @staticmethod
  def _extend_back(line: np.ndarray) -> np.ndarray:
    """Model lines start at us; continue them straight behind so the chase camera sees road under the car."""
    if line.shape[0] == 0:
      return line
    back = np.array([[-BEHIND, line[0, 1]], [min(-0.5, line[0, 0] - 0.5), line[0, 1]]], dtype=np.float32)
    return np.vstack((back, line[line[:, 0] >= 0]))

  def _strip(self, xs, y_left, y_right, rgb, a: float, dash: tuple | None = None):
    """Ground strip between two lateral offsets. Segments sharing a fog shade go out as one triangle strip."""
    xs = np.asarray(xs, dtype=np.float64)
    n = len(xs)
    world = np.zeros((2 * n, 3))
    world[:n, 0], world[n:, 0] = y_left, y_right
    world[:n, 2] = world[n:, 2] = -xs
    scr, z = self._cam.project(world)
    L, R, zl, zr = scr[:n], scr[n:], z[:n], z[n:]
    seg_ok = np.minimum(np.minimum(zl[:-1], zl[1:]), np.minimum(zr[:-1], zr[1:])) >= NEAR
    if dash is not None:
      period, on = dash
      seg_ok &= ((xs[:-1] + xs[1:]) / 2 + self._odometer) % period <= on
    depth = (zl[:-1] + zl[1:] + zr[:-1] + zr[1:]) / 4
    band = np.round(np.clip((depth - FOG_NEAR) / (FOG_FAR - FOG_NEAR), 0.0, 1.0) * FOG_BANDS).astype(int)
    i = 0
    while i < n - 1:
      if not seg_ok[i]:
        i += 1
        continue
      j = i
      while j + 1 < n - 1 and seg_ok[j + 1] and band[j + 1] == band[i]:
        j += 1
      col = self._fog(rgb, a, FOG_NEAR + band[i] / FOG_BANDS * (FOG_FAR - FOG_NEAR))
      _strip2d(L[i:j + 2], R[i:j + 2], col)
      i = j + 1

  def _ribbon(self, line: np.ndarray, half: float, x0: float, x1: float, rgb, a: float, dash: tuple | None = None):
    if line.shape[0] < 2 or x1 <= x0 or a <= 0.01:
      return
    xs = np.append(np.arange(x0, x1, 2.0), x1)
    ys = np.interp(xs, line[:, 0], line[:, 1])
    self._strip(xs, ys - half, ys + half, rgb, a, dash)

  @staticmethod
  def _build_car_template():
    """Low-poly car after the sim's buildCar, in local coords around the car's centre (world axes).
    Returns quads (F,4,3), outward normals (F,3) and colour keys ('paint', 'brake' or an rgb)."""
    quads, normals, keys = [], [], []

    def box(x, y, z, sx, sy, sz, key):
      c = np.array([x, y, -z])  # sim z is forward; world forward is -Z
      hx, hy, hz = sx / 2, sy / 2, sz / 2
      for n, (u, v) in (((0, 1, 0), ((1, 0, 0), (0, 0, 1))), ((-1, 0, 0), ((0, 1, 0), (0, 0, 1))), ((1, 0, 0), ((0, 1, 0), (0, 0, 1))),
                        ((0, 0, -1), ((1, 0, 0), (0, 1, 0))), ((0, 0, 1), ((1, 0, 0), (0, 1, 0)))):
        n, u, v = np.array(n, float), np.array(u, float), np.array(v, float)
        h = np.array([hx, hy, hz])
        fc = c + n * h
        du, dv = u * h, v * h
        quads.append([fc - du - dv, fc + du - dv, fc + du + dv, fc - du + dv])
        normals.append(n)
        keys.append(key)

    box(0, 0.38, 0, 1.79, 0.16, 4.4, UNDER)
    box(0, 0.57, 0, 1.88, 0.62, 4.65, 'paint')
    box(0, 0.85, 1.52, 1.72, 0.2, 1.35, 'paint')
    for z in (1.45, -1.48):
      for s in (-1, 1):
        box(s * 0.91, 0.36, z, 0.24, 0.7, 0.7, TIRE)
    for s in (-1, 1):
      box(s * 0.73, 1.49, -0.13, 0.07, 0.08, 1.45, 'paint')  # roof rails
      box(s * 1.02, 1.0, 0.78, 0.28, 0.14, 0.35, 'paint')  # mirrors
      box(s * 0.63, 0.77, -2.26, 0.57, 0.11, 0.05, 'brake')  # tail lights
    box(0, 1.45, -1.35, 0.44, 0.045, 0.05, 'brake')  # CHMSL

    # glass cabin: sloped rear glass, curved roof, windshield, two sides (a convex prism)
    profile = [(-1.45, 0.8), (-0.9, 1.42), (0.67, 1.43), (1.33, 0.82)]  # (sim z forward, height)
    half = 0.775
    Lp = [np.array([-half, h, -z]) for z, h in profile]
    Rp = [np.array([half, h, -z]) for z, h in profile]
    centre = np.mean(Lp + Rp, axis=0)
    faces = [[Lp[i], Rp[i], Rp[i + 1], Lp[i + 1]] for i in range(3)] + [Lp, Rp]
    for f in faces:
      f = np.array(f)
      n = np.cross(f[1] - f[0], f[2] - f[0])
      n /= np.linalg.norm(n)
      if np.dot(n, f.mean(axis=0) - centre) < 0:
        n = -n
      quads.append(f)
      normals.append(n)
      keys.append(GLASS)

    normals = np.array(normals)
    shade = 0.55 + 0.45 * np.clip(normals @ LIGHT, 0.0, 1.0)
    return np.array(quads, dtype=np.float64), normals, keys, shade

  def _draw_car(self, x_center: float, y: float, paint, alpha: float, brake: bool):
    """Translate the template, drop faces turned away from the camera, paint the rest far to near."""
    quads, normals, keys, shade = self._car_template
    pts = quads + _wp(x_center, y)
    centres = pts.mean(axis=1)
    to_eye = self._cam.eye - centres
    visible = np.einsum('ij,ij->i', normals, to_eye) > 0
    scr, z = self._cam.project(pts.reshape(-1, 3))
    scr, z = scr.reshape(-1, 4, 2), z.reshape(-1, 4)
    visible &= z.min(axis=1) >= NEAR
    dist = np.linalg.norm(to_eye, axis=1)
    for i in np.argsort(-dist):
      if not visible[i]:
        continue
      key = keys[i]
      rgb = paint if key == 'paint' else (BRAKE_ON if brake else BRAKE_OFF) if key == 'brake' else key
      k = shade[i]
      _fan2d(scr[i], self._fog((int(rgb[0] * k), int(rgb[1] * k), int(rgb[2] * k)), alpha, float(dist[i])))

  def _shadow(self, x_center: float, y: float, alpha: float):
    o = _wp(x_center, y, 0.02)
    pts = np.array([o + [-1.15, 0, -2.6], o + [1.15, 0, -2.6], o + [1.15, 0, 2.6], o + [-1.15, 0, 2.6]])
    scr, z = self._cam.project(pts)
    if z.min() >= NEAR:
      _fan2d(scr, rl.Color(0, 0, 0, int(110 * alpha)))

  # ---------------------------------------------------------------- render
  def prepare(self, rect: rl.Rectangle):
    self._update_data()

  def draw(self, rect: rl.Rectangle):
    eye = _wp(EGO_CENTER_X - CAM_BACK, 0.0, CAM_HEIGHT)
    target = _wp(EGO_CENTER_X + CAM_LOOK_AHEAD, 0.0, 0.0)
    self._cam = _Camera(eye, target, CAM_FOVY, rect)

    rl.draw_rectangle_rec(rect, _rgba(BG))
    rl.rl_draw_render_batch_active()
    rl.rl_disable_backface_culling()
    try:
      self._draw_scene()
    finally:
      rl.rl_draw_render_batch_active()  # flush while culling is still off
      rl.rl_enable_backface_culling()

  def _draw_scene(self):
    status = ui_state.status
    sm = ui_state.sm

    # road: shoulder band, then the surface between the model's road edges
    left, right = self._extend_back(self._edges[0]), self._extend_back(self._edges[1])
    if left.shape[0] >= 2 and right.shape[0] >= 2:
      far = float(min(left[-1, 0], right[-1, 0], 140.0))
      xs = np.append(np.arange(-BEHIND, far, 3.0), far)
      yl = np.interp(xs, left[:, 0], left[:, 1])
      yr = np.interp(xs, right[:, 0], right[:, 1])
      self._strip(xs, yl - 1.2, yr + 1.2, SHOULDER, 1.0)
      self._strip(xs, yl, yr, ROAD, 1.0)

    # road edges (soft) and lane lines: ours solid, the outer ones dashed and flowing with distance
    for i, edge in enumerate(self._edges):
      e = self._extend_back(edge)
      if e.shape[0]:
        self._ribbon(e, 0.06, -BEHIND, float(min(e[-1, 0], 140.0)), LINE, 0.38 * (1.0 - min(1.0, float(self._edge_stds[i]))))
    for i, lane in enumerate(self._lanes):
      ln = self._extend_back(lane)
      if ln.shape[0] == 0:
        continue
      prob = float(self._lane_probs[i]) if i < len(self._lane_probs) else 0.0
      dash = None if i in (1, 2) else (12.0, 3.0)
      self._ribbon(ln, 0.06, -BEHIND, float(min(ln[-1, 0], 120.0)), LINE, 0.85 * min(1.0, prob * 1.4), dash)

    # planned path: teal, from our front bumper to the lead (or as far as the model plans)
    lead_a = self._lead_alpha.x if self._lead_x.initialized else 0.0
    path = self._path
    if path.shape[0] >= 2 and status != UIStatus.DISENGAGED:
      end = float(min(path[-1, 0], 85.0))
      if lead_a > 0.5:
        end = min(end, self._lead_x.x - 0.5)
      self._ribbon(path, 0.75, EGO_CENTER_X + 2.4, end, PATH, PATH_ALPHA)

    # cars: shadows on the road first, then the bodies far to near
    cc = sm['carControl']
    braking = sm['carState'].brakePressed or (cc.longActive and cc.actuators.accel < -0.5)
    y0 = float(np.interp(0.0, path[:, 0], path[:, 1])) if path.shape[0] >= 2 else 0.0
    if lead_a > 0.02:
      lead_c = self._lead_x.x + CAR_LEN / 2
      self._shadow(lead_c, self._lead_y.x, lead_a)
      self._draw_car(lead_c, self._lead_y.x, LEAD_PAINT, lead_a, False)
    self._shadow(EGO_CENTER_X, y0, 1.0)
    self._draw_car(EGO_CENTER_X, y0, EGO_PAINT, 1.0, braking)
