"""Cartoon Tesla-style onroad view over the live camera: our car, the lead car, and a blue path between them.

Everything is drawn through ModelRenderer's car-space -> screen transform, so the path and
the lead car line up with the same road geometry the normal overlay uses. Our own car can't be
projected (it's under the camera), so it is pinned to the bottom of the screen and the path is
started at the ground distance that lands under it.
"""
import numpy as np
import pyray as rl

from openpilot.selfdrive.ui.ui_state import ui_state, UIStatus
from openpilot.system.ui.lib.shader_polygon import draw_polygon, Gradient

CAR_WIDTH = 1.85  # m
CAR_HEIGHT = 1.45  # m
PATH_HALF_WIDTH = 0.8  # m
LEAD_GAP = 0.4  # m, path stops just short of the lead's bumper
NO_LEAD_MIN_DIST = 20.0  # m
NO_LEAD_TIME = 4.0  # s of travel shown when there's no lead
EGO_WIDTH_FRAC = 0.20  # our car's width as a fraction of the view width
EGO_ASPECT = 0.72  # height / width of our car sprite
EGO_BOTTOM_MARGIN = 4  # px

BLUE = (30, 120, 255)
BLUE_LIGHT = (120, 190, 255)
OUTLINE = (12, 14, 28)
EGO_BODY = (250, 250, 252)
LEAD_BODY = (255, 138, 36)


def _color(rgb, alpha: float) -> rl.Color:
  return rl.Color(rgb[0], rgb[1], rgb[2], int(max(0.0, min(1.0, alpha)) * 255))


def _shade(rgb, k: float):
  return tuple(min(255, int(c * k)) for c in rgb)


def _quad(tl, tr, br, bl, color):
  # raylib wants counter-clockwise vertices
  rl.draw_triangle(tl, bl, br, color)
  rl.draw_triangle(tl, br, tr, color)


def _grow(r: rl.Rectangle, o: float) -> rl.Rectangle:
  return rl.Rectangle(r.x - o, r.y - o, r.width + 2 * o, r.height + 2 * o)


def _rrect(r: rl.Rectangle, roundness: float, fill: rl.Color, line: rl.Color, o: float):
  """Cartoon rounded rect: dark outline underneath, flat fill on top."""
  rl.draw_rectangle_rounded(_grow(r, o), roundness, 8, line)
  rl.draw_rectangle_rounded(r, roundness, 8, fill)


def draw_car_rear(cx: float, bottom: float, w: float, h: float, body, alpha: float = 1.0, brake: bool = False):
  """Cartoon rear view of a car. (cx, bottom) is where the tires meet the road."""
  if w < 2 or h < 2 or alpha <= 0.01:
    return

  o = max(1.0, w * 0.03)  # outline thickness scales with the car
  line_c = _color(OUTLINE, alpha)
  body_c = _color(body, alpha)
  hi_c = _color(_shade(body, 1.25) if max(body) < 230 else (255, 255, 255), alpha)
  dark_c = _color(_shade(body, 0.55), alpha)
  glass_c = _color((70, 150, 230), alpha)
  shine_c = rl.Color(255, 255, 255, int(170 * alpha))

  rl.draw_ellipse(int(cx), int(bottom), w * 0.62, max(2.0, h * 0.11), rl.Color(0, 0, 0, int(110 * alpha)))

  if w < 14:
    _rrect(rl.Rectangle(cx - w / 2, bottom - h * 0.6, w, h * 0.6), 0.4, body_c, line_c, 1.0)
    rl.draw_rectangle_rec(rl.Rectangle(cx - w * 0.34, bottom - h, w * 0.68, h * 0.4), glass_c)
    return

  # chunky tires
  tw, th = w * 0.18, h * 0.22
  for side in (-1, 1):
    tx = cx + side * (w / 2 - tw * 0.55) - tw / 2
    _rrect(rl.Rectangle(tx, bottom - th, tw, th), 0.5, rl.Color(30, 30, 36, int(255 * alpha)), line_c, o)

  # cabin: outline, body-colored frame, glass, shine streak
  body_top = bottom - h * 0.60
  roof = bottom - h
  base = body_top + h * 0.08
  tl, tr = rl.Vector2(cx - w * 0.30, roof), rl.Vector2(cx + w * 0.30, roof)
  br, bl = rl.Vector2(cx + w * 0.44, base), rl.Vector2(cx - w * 0.44, base)
  _quad(rl.Vector2(tl.x - o, tl.y - o), rl.Vector2(tr.x + o, tr.y - o),
        rl.Vector2(br.x + o, br.y), rl.Vector2(bl.x - o, bl.y), line_c)
  _quad(tl, tr, br, bl, body_c)
  i = max(1.5, w * 0.04)
  _quad(rl.Vector2(tl.x + i, tl.y + i * 1.3), rl.Vector2(tr.x - i, tr.y + i * 1.3),
        rl.Vector2(br.x - i * 1.6, br.y - i * 0.3), rl.Vector2(bl.x + i * 1.6, bl.y - i * 0.3), glass_c)
  rl.draw_line_ex(rl.Vector2(cx - w * 0.20, base - h * 0.04), rl.Vector2(cx - w * 0.06, roof + h * 0.12), max(1.5, w * 0.035), shine_c)

  # body: outline, fill, top highlight band, bumper
  body = rl.Rectangle(cx - w / 2, body_top, w, h * 0.46)
  _rrect(body, 0.5, body_c, line_c, o)
  rl.draw_rectangle_rounded(rl.Rectangle(body.x + w * 0.08, body_top + h * 0.03, w * 0.84, h * 0.06), 1.0, 6, hi_c)
  rl.draw_rectangle_rounded(rl.Rectangle(cx - w * 0.36, bottom - h * 0.24, w * 0.72, h * 0.09), 0.8, 6, dark_c)

  # big round tail lights
  r = max(2.0, h * 0.075)
  ty = body_top + h * 0.17
  tail_c = rl.Color(255, 60, 60, int(255 * alpha)) if brake else rl.Color(210, 20, 40, int(255 * alpha))
  for x in (cx - w * 0.37, cx + w * 0.37):
    if brake:
      rl.draw_circle(int(x), int(ty), r * 2.0, rl.Color(255, 60, 60, int(80 * alpha)))
    rl.draw_circle(int(x), int(ty), r + o, line_c)
    rl.draw_circle(int(x), int(ty), r, tail_c)
    rl.draw_circle(int(x - r * 0.35), int(ty - r * 0.35), max(1.0, r * 0.3), shine_c)


class LeadCarView:
  def __init__(self, model_renderer):
    self._mr = model_renderer

  def _ground(self, xs: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    path = self._mr._path.raw_points
    if y is None:
      y = np.interp(xs, path[:, 0], path[:, 1])
    z = np.interp(xs, path[:, 0], path[:, 2]) + self._mr._path_offset_z
    return np.column_stack((xs, y, z)).astype(np.float32)

  def _project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = self._mr._car_space_transform @ pts.T
    ok = np.abs(p[2]) > 1e-6
    scr = np.zeros((pts.shape[0], 2), dtype=np.float32)
    scr[ok] = (p[:2, ok] / p[2, ok]).T
    return scr, ok

  def render(self, rect: rl.Rectangle):
    path = self._mr._path.raw_points
    if path.shape[0] < 2 or path[-1, 0] < 2.0:
      return

    sm = ui_state.sm
    v_ego = max(0.0, sm['carState'].vEgo)
    status = ui_state.status

    # where on the road our car sits: the ground distance that projects to the bottom of the view
    xs = np.linspace(max(0.5, float(path[0, 0])), float(path[-1, 0]), 160)
    scr, ok = self._project(self._ground(xs))
    if not ok.any():
      return
    xs, scr = xs[ok], scr[ok]
    ys = np.minimum.accumulate(scr[:, 1])  # screen y should only go up the screen as distance grows
    ego_bottom = rect.height - EGO_BOTTOM_MARGIN
    if ys[0] <= ego_bottom:
      d_ego = float(xs[0])
    else:
      d_ego = float(np.interp(-ego_bottom, -ys, xs))
    ego_cx = float(np.interp(d_ego, xs, scr[:, 0]))

    # lead car
    lead = self._mr._lead_vehicles[0]
    lead_alpha = lead.fade_filter.x / 0.8 if lead.d_filter.initialized else 0.0
    lead_d = float(lead.d_filter.x)
    has_lead = lead_alpha > 0.3 and lead_d > d_ego + 2.0

    # blue path from under our car to the lead (or a few seconds ahead)
    if has_lead:
      end_d = lead_d - LEAD_GAP
    else:
      end_d = min(float(path[-1, 0]), max(NO_LEAD_MIN_DIST, v_ego * NO_LEAD_TIME))
    if end_d > d_ego + 0.5:
      self._draw_path(rect, d_ego, end_d, has_lead, status)

    if lead_alpha > 0.01 and lead_d > d_ego + 1.0:
      self._draw_lead(rect, lead_d, float(lead.y_filter.x), lead_alpha)

    # our car, pinned to the bottom of the view and following the path's lateral position
    w = rect.width * EGO_WIDTH_FRAC
    cc = sm['carControl']
    braking = sm['carState'].brakePressed or (cc.longActive and cc.actuators.accel < -0.5)
    draw_car_rear(rect.x + ego_cx, rect.y + ego_bottom, w, w * EGO_ASPECT, EGO_BODY, 1.0, braking)

  def _ribbon(self, xs: np.ndarray, half_width: float) -> np.ndarray:
    line = self._ground(xs)
    line[:, 2] -= self._mr._path_offset_z  # _map_line_to_polygon adds the height offset itself
    return self._mr._map_line_to_polygon(line, half_width, self._mr._path_offset_z, len(xs) - 1, allow_invert=False)

  def _draw_path(self, rect: rl.Rectangle, start_d: float, end_d: float, has_lead: bool, status):
    xs = np.linspace(start_d, end_d, 40)
    offset = np.array([rect.x, rect.y], dtype=np.float32)
    outer = self._ribbon(xs, PATH_HALF_WIDTH + 0.14)
    fill = self._ribbon(xs, PATH_HALF_WIDTH)
    stripe = self._ribbon(xs, PATH_HALF_WIDTH * 0.22)
    if fill.shape[0] < 4 or outer.shape[0] < 4:
      return

    if status == UIStatus.DISENGAGED:
      rgb, light = (150, 150, 160), (210, 210, 220)
    elif status == UIStatus.OVERRIDE:
      rgb, light = (235, 238, 245), (255, 255, 255)
    else:
      rgb, light = BLUE, BLUE_LIGHT
    fade_alpha = 0.55 if status == UIStatus.DISENGAGED else 1.0

    bottom = float(outer[:, 1].max()) / rect.height
    top = float(outer[:, 1].min()) / rect.height

    def grad(c, a):
      g = Gradient(start=(0.0, bottom), end=(0.0, top), colors=[], stops=[])
      if has_lead:
        g.colors, g.stops = [_color(c, a), _color(c, a)], [0.0, 1.0]
      else:
        g.colors, g.stops = [_color(c, a), _color(c, a), _color(c, 0.0)], [0.0, 0.6, 1.0]
      return g

    draw_polygon(rect, outer + offset, gradient=grad(OUTLINE, 0.9 * fade_alpha))
    draw_polygon(rect, fill + offset, gradient=grad(rgb, 0.95 * fade_alpha))
    if stripe.shape[0] >= 4:
      draw_polygon(rect, stripe + offset, gradient=grad(light, 0.9 * fade_alpha))

  def _draw_lead(self, rect: rl.Rectangle, d: float, y_rel: float, alpha: float):
    # radar yRel is left-positive; the model frame is right-positive
    y = -y_rel + self._mr._camera_offset
    ground = self._ground(np.array([d, d, d]), np.array([y - CAR_WIDTH / 2, y + CAR_WIDTH / 2, y]))
    ground[2, 2] -= CAR_HEIGHT  # roof (z is down)
    scr, ok = self._project(ground)
    if not ok.all():
      return
    w = abs(float(scr[1, 0] - scr[0, 0]))
    bottom = float(scr[0:2, 1].mean())
    h = bottom - float(scr[2, 1])
    if w > rect.width * 2 or h <= 0:
      return
    draw_car_rear(rect.x + float(scr[0:2, 0].mean()), rect.y + bottom, w, h, LEAD_BODY, alpha)
