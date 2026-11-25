import hashlib
import json

import pytest

from openpilot.cereal import custom
from openpilot.sunnypilot.models import local_bundles
from openpilot.sunnypilot.models.fetcher import ModelParser, with_local_bundles
from openpilot.sunnypilot.models.local_bundles import LOCAL_QCOM_BUNDLES, local_bundle_dicts

CH3 = LOCAL_QCOM_BUNDLES[0]
REMOTE = {"bundles": [{
  "short_name": "SC", "display_name": "Sharp Cheddar (November 17, 2026)", "is_20hz": True, "is_big": False,
  "ref": "da491579b8ea8be37c8f14e79beb1d87ead5dbb6", "environment": "development", "runner": "tinygrad", "index": 78,
  "minimum_selector_version": "21", "generation": "12",
  "overrides": {"folder": "2026 Deep RL Models", "lat": ".0", "long": ".3"},
  "models": [{"type": "chunked", "artifact": {"file_name": "sc.pkl", "download_uri": {"url": "https://x/sc.pkl", "sha256": "ab" * 32}}}],
}]}


@pytest.fixture
def onnx(tmp_path, monkeypatch):
  """A stand-in bundled onnx whose sha256 we control."""
  path = tmp_path / "driving_supercombo.onnx"

  def write(content: bytes, claim_ch3: bool):
    path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    monkeypatch.setitem(CH3, "onnx_sha256", digest if claim_ch3 else "0" * 64)
    monkeypatch.setattr(local_bundles, "BUNDLED_ONNX", str(path))
    local_bundles._hash_cache.clear()
    return str(path)
  return write


def remote():
  return ModelParser.parse_models(REMOTE)


def test_shows_when_bundled_model_is_ch3(onnx):
  path = onnx(b"crumbly", claim_ch3=True)
  bundles = with_local_bundles("qcom", remote())
  assert [b.internalName for b in bundles] == ["SC", "CH3"]
  ch3 = bundles[1]
  assert ch3.runner == custom.ModelManagerSP.Runner.stock
  assert ch3.displayName == "Crumbly Cheddar (CH3) (November 24, 2026)"
  assert len(ch3.models) == 0
  assert local_bundle_dicts("qcom", path)


def test_matches_sharp_cheddar_fields(onnx):
  onnx(b"crumbly", claim_ch3=True)
  sc, ch3 = (b.to_dict() for b in with_local_bundles("qcom", remote()))
  for key in ("generation", "environment", "is20hz", "minimumSelectorVersion", "overrides"):
    assert ch3[key] == sc[key], key


def test_hidden_when_bundled_model_is_something_else(onnx):
  onnx(b"oct 28 model", claim_ch3=False)
  assert [b.internalName for b in with_local_bundles("qcom", remote())] == ["SC"]


def test_not_added_to_big_model_list(onnx):
  onnx(b"crumbly", claim_ch3=True)
  assert with_local_bundles("chestnut", remote()) == remote() or \
    [b.internalName for b in with_local_bundles("chestnut", remote())] == ["SC"]


def test_offline_list_stays_empty(onnx):
  # an empty list tells validate_active_bundles the fetch failed; adding CH3 would make it reset a catalog selection
  onnx(b"crumbly", claim_ch3=True)
  assert with_local_bundles("qcom", []) == []


def test_selection_survives_validation(onnx, monkeypatch):
  from openpilot.sunnypilot.models import helpers
  onnx(b"crumbly", claim_ch3=True)
  bundles = with_local_bundles("qcom", remote())
  ch3 = bundles[1]
  assert not helpers._bundle_needs_reset(ch3, bundles)  # no files to verify, ref/runner match

  # the stored form (params JSON round-trip) parses back as a valid stock selection
  stored = json.loads(json.dumps(ch3.to_dict()))
  parsed = helpers._parse_active_bundle(stored)
  assert parsed is not None and parsed.runner == custom.ModelManagerSP.Runner.stock


def test_selection_resets_if_file_changes(onnx):
  from openpilot.sunnypilot.models import helpers
  onnx(b"crumbly", claim_ch3=True)
  ch3 = with_local_bundles("qcom", remote())[1]
  onnx(b"oct 28 model", claim_ch3=False)
  assert helpers._bundle_needs_reset(ch3, with_local_bundles("qcom", remote()))


def test_hash_is_cached(onnx, monkeypatch):
  path = onnx(b"crumbly", claim_ch3=True)
  local_bundles.bundled_onnx_sha256(path)
  calls = []
  real = hashlib.sha256
  monkeypatch.setattr(local_bundles.hashlib, "sha256", lambda *a: calls.append(1) or real(*a))
  for _ in range(5):
    local_bundles.bundled_onnx_sha256(path)
  assert calls == []
