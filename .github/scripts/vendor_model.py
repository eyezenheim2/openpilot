#!/usr/bin/env python3
"""Put a comma driving model into this branch as a regular git file (not LFS).

Reads .github/vendor-model.json:
  {"oid": "<sha256>", "size": <bytes>, "name": "<display name>", "source": "<where it came from>"}

Downloads the object from comma's LFS (Hugging Face commaai/openpilot-lfs), checks size and
sha256, writes driving_supercombo.onnx, exempts that one path from LFS in .gitattributes, and
updates sunnypilot's default-model name and model_hash. Run by .github/workflows/vendor-model.yaml.

Why: the fork's LFS is sunnypilot's, which doesn't have comma's newest models, so as an LFS
pointer the device can't fetch it. As a regular file a plain git pull (and the updater) gets it.
"""
import hashlib
import json
import os
import sys
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REQUEST = os.path.join(ROOT, ".github", "vendor-model.json")
MODEL = "openpilot/selfdrive/modeld/models/driving_supercombo.onnx"
MODEL_NAME_PY = os.path.join(ROOT, "openpilot", "sunnypilot", "models", "model_name.py")
MODEL_HASH = os.path.join(ROOT, "openpilot", "sunnypilot", "models", "tests", "model_hash")
GITATTRIBUTES = os.path.join(ROOT, ".gitattributes")
COMMA_LFS = "https://huggingface.co/commaai/openpilot-lfs.git/info/lfs/objects/batch"
LFS_EXEMPT = f"{MODEL} -filter -diff -merge -text"


def sha256_file(path: str) -> str | None:
  if not os.path.isfile(path):
    return None
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for block in iter(lambda: f.read(1 << 20), b""):
      h.update(block)
  return h.hexdigest()


def download(oid: str, size: int) -> bytes:
  body = json.dumps({"operation": "download", "transfers": ["basic"], "objects": [{"oid": oid, "size": size}]}).encode()
  req = urllib.request.Request(COMMA_LFS, data=body, method="POST", headers={
    "Accept": "application/vnd.git-lfs+json", "Content-Type": "application/vnd.git-lfs+json"})
  with urllib.request.urlopen(req, timeout=60) as r:
    obj = json.load(r)["objects"][0]
  if "error" in obj:
    sys.exit(f"comma LFS has no object {oid}: {obj['error']}")
  action = obj["actions"]["download"]
  req = urllib.request.Request(action["href"], headers=action.get("header", {}))
  with urllib.request.urlopen(req, timeout=900) as r:
    data = r.read()
  if len(data) != size:
    sys.exit(f"size mismatch: got {len(data)}, expected {size}")
  if hashlib.sha256(data).hexdigest() != oid:
    sys.exit("sha256 mismatch: download corrupted")
  return data


def set_default_name(name: str) -> str:
  lines = open(MODEL_NAME_PY).read().splitlines()
  out, ref = [], ""
  for line in lines:
    if line.startswith("DEFAULT_MODEL ="):
      line = f'DEFAULT_MODEL = "{name}"'
    elif line.startswith("DEFAULT_MODEL_REF ="):
      ref = line.split("=", 1)[1].strip().strip('"')
    out.append(line)
  open(MODEL_NAME_PY, "w").write("\n".join(out) + "\n")
  return ref


def main() -> None:
  req = json.load(open(REQUEST))
  oid, size, name = req["oid"].lower(), int(req["size"]), req["name"]
  path = os.path.join(ROOT, MODEL)

  if sha256_file(path) == oid:
    print("model file already in place")
  else:
    print(f"downloading {oid} ({size} bytes) from comma LFS")
    data = download(oid, size)
    with open(path, "wb") as f:
      f.write(data)

  attrs = open(GITATTRIBUTES).read()
  if LFS_EXEMPT not in attrs:
    open(GITATTRIBUTES, "a").write(("" if attrs.endswith("\n") else "\n") +
                                   "# comma's newest driving model, kept as a regular file (see .github/scripts/vendor_model.py)\n" +
                                   LFS_EXEMPT + "\n")

  # same fingerprint sunnypilot's default_model.update_model_hash() writes
  ref = set_default_name(name)
  with open(MODEL_HASH, "w") as f:
    f.write(hashlib.sha256(f"{oid}:{name}:{ref}".encode()).hexdigest())
  print(f"done: {name}")


if __name__ == "__main__":
  main()
