"""Tesla-style onroad view over the GTA-graded live camera: our car, the lead car, and a blue path between them.

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
EGO_ASPECT = 0.92  # height / width of our car sprite (taller: chase-cam view shows the roof)
EGO_BOTTOM_MARGIN = 4  # px

BLUE = (30, 120, 255)
BLUE_LIGHT = (120, 190, 255)
EGO_BODY = (196, 201, 209)  # silver, like the Tesla visualization
LEAD_BODY = (120, 125, 134)


def _color(rgb, alpha: float) -> rl.Color:
  return rl.Color(rgb[0], rgb[1], rgb[2], int(max(0.0, min(1.0, alpha)) * 255))


def _shade(rgb, k: float):
  return tuple(min(255, int(c * k)) for c in rgb)


def _quad(tl, tr, br, bl, color):
  # raylib wants counter-clockwise vertices
  rl.draw_triangle(tl, bl, br, color)
  rl.draw_triangle(tl, br, tr, color)


def _vgrad_rrect(r: rl.Rectangle, roundness: float, top: rl.Color, bottom: rl.Color, steps: int = 6):
  """Rounded rect with a vertical shade, built from stacked rounded slices (raylib has no rounded gradient)."""
  rl.draw_rectangle_rounded(r, roundness, 8, bottom)
  for i in range(steps):
    k = i / steps
    c = rl.Color(int(top.r + (bottom.r - top.r) * k), int(top.g + (bottom.g - top.g) * k),
                 int(top.b + (bottom.b - top.b) * k), int(top.a + (bottom.a - top.a) * k))
    sh = r.height * (1.0 - k)
    rl.draw_rectangle_rounded(rl.Rectangle(r.x, r.y, r.width, sh), min(1.0, roundness * r.height / max(sh, 1.0)), 8, c)


def draw_car_rear(cx: float, bottom: float, w: float, h: float, body, alpha: float = 1.0, brake: bool = False,
                  roof: float = 0.0):
  """Tesla-visualization style rear view. (cx, bottom) is where the tires meet the road.
  roof: fraction of the height showing the roof from above (chase-cam look for our own car)."""
  if w < 2 or h < 2 or alpha <= 0.01:
    return

  A = lambda rgb, k=1.0: _color(rgb, alpha * k)  # noqa: E731
  body_hi = _shade(body, 1.12)
  body_lo = _shade(body, 0.72)
  edge = A(_shade(body, 0.45), 0.9)
  glass = (28, 32, 40)

  # soft contact shadow
  rl.draw_ellipse(int(cx), int(bottom - h * 0.02), w * 0.58, max(2.0, h * 0.07), rl.Color(0, 0, 0, int(140 * alpha)))

  if w < 14:
    rl.draw_rectangle_rec(rl.Rectangle(cx - w / 2, bottom - h * 0.55, w, h * 0.55), A(body))
    rl.draw_rectangle_rec(rl.Rectangle(cx - w * 0.36, bottom - h, w * 0.72, h * 0.45), A(glass))
    rl.draw_rectangle_rec(rl.Rectangle(cx - w * 0.48, bottom - h * 0.45, w * 0.96, max(1.0, h * 0.06)), rl.Color(200, 20, 30, int(255 * alpha)))
    return

  # tires peeking out under the body
  tw, th = w * 0.15, h * 0.13
  for side in (-1, 1):
    rl.draw_rectangle_rounded(rl.Rectangle(cx + side * (w * 0.5 - tw * 0.62) - tw / 2, bottom - th, tw, th), 0.35, 4,
                              rl.Color(14, 14, 16, int(255 * alpha)))

  roof_h = h * roof
  greenhouse_top = bottom - h + roof_h
  shoulder = bottom - h * 0.50  # where the rear glass meets the trunk

  # roof seen from above (our car): tapered panel, dark glass roof like a Model Y
  if roof_h > 1.0:
    rt = rl.Vector2(cx - w * 0.22, bottom - h)
    rtr = rl.Vector2(cx + w * 0.22, bottom - h)
    rbr = rl.Vector2(cx + w * 0.30, greenhouse_top)
    rbl = rl.Vector2(cx - w * 0.30, greenhouse_top)
    _quad(rt, rtr, rbr, rbl, A(body_lo))
    i = max(1.0, w * 0.025)
    _quad(rl.Vector2(rt.x + i, rt.y + i), rl.Vector2(rtr.x - i, rtr.y + i),
          rl.Vector2(rbr.x - i, rbr.y), rl.Vector2(rbl.x + i, rbl.y), A(glass, 0.95))

  # C-pillars / sides of the cabin in body color, rear glass inset
  tl, tr = rl.Vector2(cx - w * 0.30, greenhouse_top), rl.Vector2(cx + w * 0.30, greenhouse_top)
  br, bl = rl.Vector2(cx + w * 0.45, shoulder), rl.Vector2(cx - w * 0.45, shoulder)
  _quad(tl, tr, br, bl, A(body_hi))
  corner = max(1.0, w * 0.05)  # round the top of the cabin
  rl.draw_rectangle_rounded(rl.Rectangle(tl.x - corner * 0.3, tl.y - corner * 0.4, tr.x - tl.x + corner * 0.6, corner * 1.6), 1.0, 6, A(body_hi))
  i = max(1.0, w * 0.04)
  gtl, gtr = rl.Vector2(tl.x + i, tl.y + i * 0.6), rl.Vector2(tr.x - i, tr.y + i * 0.6)
  gbr, gbl = rl.Vector2(br.x - i * 2.2, br.y - i * 0.4), rl.Vector2(bl.x + i * 2.2, bl.y - i * 0.4)
  _quad(gtl, gtr, gbr, gbl, A(glass))
  # sky reflection on the rear glass
  _quad(rl.Vector2(gtl.x + (gtr.x - gtl.x) * 0.10, gtl.y + 1), rl.Vector2(gtl.x + (gtr.x - gtl.x) * 0.38, gtl.y + 1),
        rl.Vector2(gbl.x + (gbr.x - gbl.x) * 0.22, gbl.y - 1), rl.Vector2(gbl.x + (gbr.x - gbl.x) * 0.02, gbl.y - 1),
        rl.Color(255, 255, 255, int(28 * alpha)))

  # rear body: smooth shaded shell, lighter on top like painted metal under sky light
  shell = rl.Rectangle(cx - w / 2, shoulder - h * 0.03, w, bottom - h * 0.05 - (shoulder - h * 0.03))
  rl.draw_rectangle_rounded(rl.Rectangle(shell.x - 0.6, shell.y - 0.6, shell.width + 1.2, shell.height + 1.2), 0.6, 10, edge)
  _vgrad_rrect(shell, 0.6, A(body_hi), A(body_lo))
  # trunk lip highlight
  rl.draw_rectangle_rounded(rl.Rectangle(cx - w * 0.40, shell.y + h * 0.015, w * 0.80, max(1.0, h * 0.025)), 1.0, 4,
                            rl.Color(255, 255, 255, int(70 * alpha)))

  # Tesla-style thin wrap-around tail lights joined by a dark trim line
  ly = shell.y + shell.height * 0.24
  lh = max(1.5, h * 0.045)
  red = rl.Color(255, 45, 45, int(255 * alpha)) if brake else rl.Color(190, 16, 30, int(255 * alpha))
  rl.draw_rectangle_rec(rl.Rectangle(cx - w * 0.30, ly + lh * 0.25, w * 0.60, max(1.0, lh * 0.5)), A(_shade(body, 0.35)))
  for x0, x1 in ((cx - w * 0.495, cx - w * 0.30), (cx + w * 0.30, cx + w * 0.495)):
    rl.draw_rectangle_rounded(rl.Rectangle(x0, ly, x1 - x0, lh), 0.8, 4, red)
    if brake:
      rl.draw_rectangle_rounded(rl.Rectangle(x0 - 3, ly - 3, x1 - x0 + 6, lh + 6), 0.8, 4, rl.Color(255, 40, 40, int(60 * alpha)))

  # dark lower bumper / diffuser
  rl.draw_rectangle_rounded(rl.Rectangle(cx - w * 0.42, bottom - h * 0.15, w * 0.84, h * 0.10), 0.7, 6,
                            rl.Color(22, 24, 28, int(235 * alpha)))


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
    draw_car_rear(rect.x + ego_cx, rect.y + ego_bottom, w, w * EGO_ASPECT, EGO_BODY, 1.0, braking, roof=0.24)

  def _ribbon(self, xs: np.ndarray, half_width: float) -> np.ndarray:
    line = self._ground(xs)
    line[:, 2] -= self._mr._path_offset_z  # _map_line_to_polygon adds the height offset itself
    return self._mr._map_line_to_polygon(line, half_width, self._mr._path_offset_z, len(xs) - 1, allow_invert=False)

  def _draw_path(self, rect: rl.Rectangle, start_d: float, end_d: float, has_lead: bool, status):
    xs = np.linspace(start_d, end_d, 40)
    offset = np.array([rect.x, rect.y], dtype=np.float32)
    fill = self._ribbon(xs, PATH_HALF_WIDTH)
    stripe = self._ribbon(xs, PATH_HALF_WIDTH * 0.22)
    if fill.shape[0] < 4:
      return

    if status == UIStatus.DISENGAGED:
      rgb, light = (150, 150, 160), (210, 210, 220)
    elif status == UIStatus.OVERRIDE:
      rgb, light = (235, 238, 245), (255, 255, 255)
    else:
      rgb, light = BLUE, BLUE_LIGHT
    fade_alpha = 0.55 if status == UIStatus.DISENGAGED else 1.0

    bottom = float(fill[:, 1].max()) / rect.height
    top = float(fill[:, 1].min()) / rect.height

    def grad(c, a):
      g = Gradient(start=(0.0, bottom), end=(0.0, top), colors=[], stops=[])
      if has_lead:
        g.colors, g.stops = [_color(c, a), _color(c, a)], [0.0, 1.0]
      else:
        g.colors, g.stops = [_color(c, a), _color(c, a), _color(c, 0.0)], [0.0, 0.6, 1.0]
      return g

    draw_polygon(rect, fill + offset, gradient=grad(rgb, 0.95 * fade_alpha))
    if stripe.shape[0] >= 4:
      draw_polygon(rect, stripe + offset, gradient=grad(light, 0.35 * fade_alpha))

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
    draw_car_rear(rect.x + float(scr[0:2, 0].mean()), rect.y + bottom, w, h, LEAD_BODY, alpha, roof=0.06)
