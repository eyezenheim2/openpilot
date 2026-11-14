#!/usr/bin/env python3
"""
timezoned: sets the device's local time zone automatically.

Once the device has internet, the IANA time zone (e.g. "America/Chicago") is looked up
from the device's public IP address and applied to the system. The lookup is repeated
periodically so the time zone follows the car when crossing zones.
"""
import os
import subprocess
import time
from typing import NoReturn

import requests

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.common.hardware import AGNOS

ZONEINFO_PATH = "/usr/share/zoneinfo"
LOOKUP_URLS = (
  "https://ipapi.co/timezone",
  "https://ipinfo.io/timezone",
)
REQUEST_TIMEOUT_S = 10
RETRY_INTERVAL_S = 60        # no internet / lookup failed
REFRESH_INTERVAL_S = 60 * 60  # after a successful lookup


def is_valid_timezone(tz: str) -> bool:
  if not tz or tz.startswith("/") or ".." in tz.split("/"):
    return False
  return os.path.isfile(os.path.join(ZONEINFO_PATH, tz))


def get_current_timezone() -> str | None:
  try:
    if AGNOS:
      with open("/data/etc/timezone") as f:
        return f.read().strip()
    link = os.path.realpath("/etc/localtime")
    if link.startswith(ZONEINFO_PATH + "/"):
      return os.path.relpath(link, ZONEINFO_PATH)
  except OSError:
    pass
  return None


def lookup_timezone() -> str | None:
  """Returns the IANA time zone for this device's public IP, or None if offline/failed."""
  for url in LOOKUP_URLS:
    try:
      r = requests.get(url, timeout=REQUEST_TIMEOUT_S, headers={"User-Agent": "sunnypilot-timezoned"})
      if r.status_code != 200:
        cloudlog.debug(f"timezoned: {url} returned {r.status_code}")
        continue
      tz = r.text.strip()
      if is_valid_timezone(tz):
        return tz
      cloudlog.warning(f"timezoned: {url} returned invalid timezone {tz!r}")
    except requests.exceptions.RequestException as e:
      cloudlog.debug(f"timezoned: {url} lookup failed: {e}")
  return None


def set_timezone(tz: str) -> bool:
  if not is_valid_timezone(tz):
    cloudlog.error(f"timezoned: refusing to set invalid timezone {tz!r}")
    return False

  cloudlog.info(f"timezoned: setting timezone to {tz}")
  try:
    if AGNOS:
      tzpath = os.path.join(ZONEINFO_PATH, tz)
      subprocess.check_call(["sudo", "ln", "-snf", tzpath, "/data/etc/tmptime"])
      subprocess.check_call(["sudo", "mv", "-T", "/data/etc/tmptime", "/data/etc/localtime"])
      subprocess.run(["sudo", "tee", "/data/etc/timezone"], input=f"{tz}\n", text=True,
                     stdout=subprocess.DEVNULL, check=True)
    else:
      subprocess.check_call(["sudo", "timedatectl", "set-timezone", tz])
  except (subprocess.CalledProcessError, OSError):
    cloudlog.exception("timezoned: failed to set timezone")
    return False
  return True


def main() -> NoReturn:
  params = Params()

  while True:
    tz = lookup_timezone()
    if tz is None:
      time.sleep(RETRY_INTERVAL_S)
      continue

    if tz != get_current_timezone():
      if not set_timezone(tz):
        time.sleep(RETRY_INTERVAL_S)
        continue

    if params.get("Timezone") != tz:
      params.put("Timezone", tz)

    time.sleep(REFRESH_INTERVAL_S)


if __name__ == "__main__":
  main()
