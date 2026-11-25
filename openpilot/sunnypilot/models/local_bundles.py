"""
Local entries for the small-model (qcom) list: models that run on the stock runner from the
bundled driving_supercombo.onnx, so they need no download.

An entry only shows when the bundled onnx on this device is the model it names (checked by
sha256), so the list can't claim a model the device isn't carrying. If the file changes, the
entry disappears and an active selection of it falls back to the default.
"""
import hashlib
import os

from openpilot.common.basedir import BASEDIR
from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot.models.helpers import REQUIRED_JSON_VERSION

BUNDLED_ONNX = os.path.join(BASEDIR, "openpilot", "selfdrive", "modeld", "models", "driving_supercombo.onnx")

# same shape as the sunnypilot catalog's bundles; "onnx_sha256" is local-only.
# Modelled on the catalog's Sharp Cheddar (SC, index 78) entry: same folder, generation and flags.
# is_20hz and the lat/long smoothing overrides are only read by sunnypilot's own runner (modeld_v2);
# these entries run on the stock runner, so they're kept for parity and have no effect.
LOCAL_QCOM_BUNDLES = [
  {
    "index": 1000,
    "short_name": "CH3",
    "display_name": "Crumbly Cheddar (CH3) (November 24, 2026)",
    "ref": "local-ch3-962396e7",
    "onnx_sha256": "962396e7909556af6a9c7bfb7ad6db01a8d2c8da0760379eb71efda004cb3548",
    "runner": "stock",
    "models": [],
    "generation": "12",
    "environment": "development",
    "is_20hz": True,
    "is_big": False,
    "minimum_selector_version": str(REQUIRED_JSON_VERSION),
    "overrides": {"folder": "2026 Deep RL Models", "lat": ".0", "long": ".3"},
  },
]

_hash_cache: dict[str, tuple[tuple[int, int], str | None]] = {}


def bundled_onnx_sha256(path: str | None = None) -> str | None:
  """sha256 of the bundled model, recomputed only when its size or mtime changes."""
  path = path or BUNDLED_ONNX
  try:
    st = os.stat(path)
  except OSError:
    return None
  key = (st.st_size, st.st_mtime_ns)
  cached = _hash_cache.get(path)
  if cached and cached[0] == key:
    return cached[1]
  digest = None
  try:
    h = hashlib.sha256()
    with open(path, "rb") as f:
      for block in iter(lambda: f.read(1 << 20), b""):
        h.update(block)
    digest = h.hexdigest()
  except OSError:
    cloudlog.exception(f"could not hash {path}")
  _hash_cache[path] = (key, digest)
  return digest


def local_bundle_dicts(source: str, path: str | None = None) -> list[dict]:
  if source != "qcom":
    return []
  digest = bundled_onnx_sha256(path)
  return [{k: v for k, v in b.items() if k != "onnx_sha256"} for b in LOCAL_QCOM_BUNDLES if b["onnx_sha256"] == digest]
