"""Bird's-eye view in the style of the sunnylink.wiki driving sim.

Everything comes from what openpilot actually perceives through the road camera:
  - road surface between the model's road edges
  - lane lines from the model (fading with their confidence)
  - the planned path (teal, like the sim) up to the lead car
  - the lead car from the vision model (modelV2.leadsV3), at its measured distance and lateral position
  - on the 11th-gen Accord, the other cars the car's own sensors track (decoded by honda_objectd from the
    driver-assist bus and published on customReservedRawData1, UI only)
Nothing is invented: elsewhere the vision model only reports the lead in our lane, so that is the only other car drawn.

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
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached

# Night palette from the sim (highway)
BG = (21, 29, 43)
ROAD = (45, 56, 73)
SHOULDER = (31, 42, 57)
LINE = (204, 214, 227)
PATH = (40, 182, 154)  # sim's teal path
PATH_ALPHA = 0.38

EGO_PAINT = (246, 248, 247)  # sim default: pearl white
LEAD_PAINT = (246, 248, 247)  # other cars in white too, so they stand out on the dark road
TRAFFIC_PAINT = [(246, 248, 247)]
HALO = (190, 210, 235)  # soft light pool under other cars
DISPLAY_DIST_SCALE = 0.5  # other cars are drawn at this fraction of their real gap to us, so they sit where they can be seen
MIN_DISPLAY_GAP = 1.0  # m of drawn gap kept between our bumper and the nearest car
IN_LANE_Y = 1.4  # m from our path: a car this close sideways is in our lane
FAR_BOOST = (10.0, 40.0, 1.6)  # other cars drawn up to 1.6x larger with distance (position unchanged) so they read on the small screen
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
RADAR_TO_CAMERA = 1.52  # m, object distances are from the front bumper, model distances from the camera
OBJ_DTYPE = np.dtype([('id', 'u1'), ('dist', '<f4'), ('lat', '<f4'), ('vrel', '<f4')])  # honda_objectd record
PAYLOAD_VERSION = 2  # honda_objectd header: (version, tsr_code)
TSR_STOP = 121  # car's Traffic Sign Recognition code for a stop sign
# stop sign post: placed at the estimated stop line on the right curb (the car's camera says a sign is there, not where)
SIGN_ROLL_V, SIGN_STOP_V = 2.6, 0.3  # m/s: model plan speeds that mark where it stops / rolls (same as the planner)
SIGN_DEFAULT_DIST = 20.0  # m: where to show it if the model's plan doesn't say
SIGN_CURB_OFFSET = 0.6  # m outside the right road edge
SIGN_POST_H, SIGN_RADIUS = 2.1, 0.38  # m: real post height to the sign's centre, 30-inch octagon
SIGN_BOOST = 2.2  # drawn larger, like the other cars, so it reads on the small screen
OBJ_TIMEOUT = 0.4  # s without an update before a tracked car fades out
LEAD_MATCH_X, LEAD_MATCH_Y = 6.0, 1.8  # m: a tracked car this close to the camera lead is the same car

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
    self._objects: dict[int, dict] = {}  # id -> {'x', 'y' filters, 'alpha' filter, 'seen' time}
    self._tsr_code = 0
    self._plan_v = np.zeros(0, dtype=np.float32)
    self._sign_x = FirstOrderFilter(SIGN_DEFAULT_DIST, 0.3, dt)
    self._stop_sign_alpha = FirstOrderFilter(0.0, 0.15, dt)

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
      self._plan_v = np.array(model.velocity.x, dtype=np.float32)
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
    self._update_objects()

  def _update_objects(self):
    """Tracked cars from the car's own sensors (Accord 11G), smoothed per track id."""
    sm = ui_state.sm
    now = rl.get_time()
    dt = 1 / gui_app.target_fps
    if sm.updated['customReservedRawData1'] and sm.recv_frame['customReservedRawData1'] >= ui_state.started_frame:
      raw = bytes(sm['customReservedRawData1'])
      if len(raw) >= 2 and raw[0] == PAYLOAD_VERSION and (len(raw) - 2) % OBJ_DTYPE.itemsize == 0:
        self._tsr_code, raw = raw[1], raw[2:]
      recs = np.frombuffer(raw[:len(raw) - len(raw) % OBJ_DTYPE.itemsize], dtype=OBJ_DTYPE)
      for r in recs:
        oid = int(r['id'])
        x, y = float(r['dist']) + RADAR_TO_CAMERA, -float(r['lat'])  # model frame: y right-positive
        o = self._objects.get(oid)
        if o is None or abs(o['x'].x - x) > 10.0 or abs(o['y'].x - y) > 3.0:
          o = {'x': FirstOrderFilter(x, 0.1, dt), 'y': FirstOrderFilter(y, 0.1, dt), 'alpha': FirstOrderFilter(0.0, 0.15, dt)}
          self._objects[oid] = o
        o['x'].update(x)
        o['y'].update(y)
        o['seen'] = now
    self._stop_sign_alpha.update(1.0 if self._tsr_code == TSR_STOP else 0.0)
    for oid in list(self._objects):
      o = self._objects[oid]
      o['alpha'].update(1.0 if now - o['seen'] < OBJ_TIMEOUT else 0.0)
      if now - o['seen'] > OBJ_TIMEOUT and o['alpha'].x < 0.02:
        del self._objects[oid]

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

  def _draw_car(self, x_center: float, y: float, paint, alpha: float, brake: bool, scale: float = 1.0):
    """Translate the template, drop faces turned away from the camera, paint the rest far to near."""
    quads, normals, keys, shade = self._car_template
    pts = quads * scale + _wp(x_center, y)
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
      _fan2d(scr[i], _rgba(rgb, alpha, k))

  def _halo(self, x_center: float, y: float, alpha: float, scale: float = 1.0):
    """Light pool on the road around a car, so it pops against the dark asphalt even when far away."""
    o = _wp(x_center, y, 0.015)
    for (hx, hz), a in (((2.2, 4.0), 0.10), ((1.7, 3.3), 0.16)):
      hx, hz = hx * scale, hz * scale
      pts = np.array([o + [-hx, 0, -hz], o + [hx, 0, -hz], o + [hx, 0, hz], o + [-hx, 0, hz]])
      scr, z = self._cam.project(pts)
      if z.min() >= NEAR:
        _fan2d(scr, _rgba(HALO, a * alpha))

  def _sign_distance(self) -> float:
    """Estimated stop line ahead (model frame x): where the model's plan stops or rolls, a few metres short of it."""
    v, x = self._plan_v, self._path[:, 0] if self._path.shape[0] else np.zeros(0)
    if v.size and v.size == x.size:
      stopped = np.nonzero(v < SIGN_STOP_V)[0]
      i = int(stopped[0]) if stopped.size else int(np.argmin(v))
      if v[i] < SIGN_ROLL_V and x[i] > 2.0:
        return max(3.0, 0.9 * float(x[i]) - 2.5)
    return SIGN_DEFAULT_DIST

  def _draw_stop_sign_post(self, x: float, y_curb: float, alpha: float, k: float):
    """Red octagon on a grey post at the curb, in true perspective (it faces back toward us)."""
    base = _wp(x, y_curb)
    post_top = SIGN_POST_H * k
    w = 0.06 * k
    # post: a thin vertical box, two visible faces
    for dx in (-w, w):
      pts = np.array([base + [dx, 0, 0], base + [dx, post_top, 0], base + [dx, post_top, -w * 2], base + [dx, 0, -w * 2]])
      scr, z = self._cam.project(pts)
      if z.min() >= NEAR:
        _fan2d(scr, _rgba((150, 156, 160), alpha, 0.8 if dx < 0 else 0.6))
    # sign plate: white rim and red face, octagons in the vertical plane facing back toward our car
    centre = base + [0, post_top, 0.02]
    for radius, rgb in ((SIGN_RADIUS * k, (245, 245, 245)), (SIGN_RADIUS * k * 0.86, (200, 16, 32))):
      ang = np.radians(22.5 + 45.0 * np.arange(8))
      ring = centre + np.column_stack((np.cos(ang) * radius, np.sin(ang) * radius, np.zeros(8)))
      ring[:, 2] += 0.01 if rgb[0] < 240 else 0.0  # face sits just in front of the rim
      scr, z = self._cam.project(ring)
      if z.min() < NEAR:
        return
      _fan2d(scr, _rgba(rgb, alpha))
    # "STOP", sized to the projected plate
    scr, z = self._cam.project(np.array([centre, centre + [SIGN_RADIUS * k, 0, 0]]))
    size = int(max(5.0, abs(scr[1, 0] - scr[0, 0]) * 0.75))
    font = gui_app.font(FontWeight.BOLD)
    m = measure_text_cached(font, "STOP", size)
    rl.draw_text_ex(font, "STOP", rl.Vector2(float(scr[0, 0] - m.x / 2), float(scr[0, 1] - m.y / 2)), size, 0,
                    rl.Color(255, 255, 255, int(255 * alpha)))

  def _shadow(self, x_center: float, y: float, alpha: float, scale: float = 1.0):
    o = _wp(x_center, y, 0.02)
    hx, hz = 1.15 * scale, 2.6 * scale
    pts = np.array([o + [-hx, 0, -hz], o + [hx, 0, -hz], o + [hx, 0, hz], o + [-hx, 0, hz]])
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
    self._draw_stop_sign_badge(rect)

  def _draw_stop_sign_badge(self, rect: rl.Rectangle):
    """Red octagon while the car's own camera (Traffic Sign Recognition) sees a stop sign."""
    a = self._stop_sign_alpha.x
    if a < 0.02:
      return
    r = 24.0
    c = rl.Vector2(rect.x + rect.width - 62, rect.y + rect.height * 0.58)
    rl.draw_poly(c, 8, r + 3, 22.5, rl.Color(255, 255, 255, int(255 * a)))
    rl.draw_poly(c, 8, r, 22.5, rl.Color(200, 16, 32, int(255 * a)))
    font = gui_app.font(FontWeight.BOLD)
    size = 13
    m = measure_text_cached(font, "STOP", size)
    rl.draw_text_ex(font, "STOP", rl.Vector2(c.x - m.x / 2, c.y - m.y / 2), size, 0, rl.Color(255, 255, 255, int(255 * a)))

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

    # cars to draw: tracked cars from the car's sensors, plus the camera lead unless one of them is that car
    lead_a = self._lead_alpha.x if self._lead_x.initialized else 0.0
    cars = []  # (x_rear, y, paint, alpha)
    lead_matched = False
    for oid, o in self._objects.items():
      x, y, a = o['x'].x, o['y'].x, o['alpha'].x
      if a < 0.02 or x < EGO_CENTER_X + CAR_LEN:
        continue
      if lead_a > 0.3 and abs(x - self._lead_x.x) < LEAD_MATCH_X and abs(y - self._lead_y.x) < LEAD_MATCH_Y:
        lead_matched = True
        cars.append((x, y, LEAD_PAINT, a))
      else:
        cars.append((x, y, TRAFFIC_PAINT[oid % len(TRAFFIC_PAINT)], a))
    if lead_a > 0.02 and not lead_matched:
      cars.append((self._lead_x.x, self._lead_y.x, LEAD_PAINT, lead_a))

    # draw other cars closer than they are (DISPLAY_DIST_SCALE of the real gap), keeping their lane:
    # their offset from our path is preserved, so on curves they stay on the road
    path = self._path
    ego_front = EGO_CENTER_X + CAR_LEN / 2

    def to_display(x: float, y: float) -> tuple[float, float]:
      x_d = ego_front + max(MIN_DISPLAY_GAP, (x - ego_front) * DISPLAY_DIST_SCALE)
      if path.shape[0] >= 2:
        y += float(np.interp(x_d, path[:, 0], path[:, 1]) - np.interp(x, path[:, 0], path[:, 1]))
      return x_d, y

    cars = [(*to_display(x, y), paint, a) for x, y, paint, a in cars]
    # the path stops at the nearest drawn car in our lane, whether the camera or the car's sensors reported it
    in_lane = [x for x, y, _, a in cars if a > 0.5 and path.shape[0] >= 2 and abs(y - float(np.interp(x, path[:, 0], path[:, 1]))) < IN_LANE_Y]
    lead_disp_x = min(in_lane) if in_lane else None

    # planned path: teal, from our front bumper to the (drawn) lead, or as far as the model plans
    if path.shape[0] >= 2 and status != UIStatus.DISENGAGED:
      end = float(min(path[-1, 0], 85.0))
      if lead_disp_x is not None:
        end = min(end, lead_disp_x - 0.5)
      self._ribbon(path, 0.75, EGO_CENTER_X + 2.4, end, PATH, PATH_ALPHA)

    # cars: shadows on the road first, then the bodies far to near
    cc = sm['carControl']
    braking = sm['carState'].brakePressed or (cc.longActive and cc.actuators.accel < -0.5)
    y0 = float(np.interp(0.0, path[:, 0], path[:, 1])) if path.shape[0] >= 2 else 0.0
    for x, y, paint, a in sorted(cars, key=lambda c: -c[0]):  # far to near so nearer cars overlap correctly
      near, far, most = FAR_BOOST
      k = 1.0 + (most - 1.0) * min(1.0, max(0.0, (x - near) / (far - near)))
      cx = x + CAR_LEN * k / 2  # keep the rear bumper where the sensors put it
      self._halo(cx, y, a, k)
      self._shadow(cx, y, a, k)
      self._draw_car(cx, y, paint, a, False, k)
    self._shadow(EGO_CENTER_X, y0, 1.0)
    sign_a = self._stop_sign_alpha.x
    if sign_a > 0.02:
      sx = self._sign_x.update(self._sign_distance())
      if sx > ego_front:
        x_d, _ = to_display(sx, 0.0)
        edge = self._edges[1]
        if edge.shape[0] >= 2:
          y_curb = float(np.interp(x_d, edge[:, 0], edge[:, 1])) + SIGN_CURB_OFFSET
        else:
          y_curb = (float(np.interp(x_d, path[:, 0], path[:, 1])) if path.shape[0] >= 2 else 0.0) + 3.5
        self._draw_stop_sign_post(x_d, y_curb, sign_a, SIGN_BOOST)
    self._draw_car(EGO_CENTER_X, y0, EGO_PAINT, 1.0, braking)
