"""Keep the Wi-Fi services at high CPU priority, so joining a saved network and keeping it up isn't slowed by
everything else running onroad.

Raises NetworkManager and wpa_supplicant to nice -10 (default 0). openpilot's control processes run SCHED_FIFO
(see common/realtime.py), which always outranks any nice level, so this can't take time from them.
Checked every 10 s by hardwared; renice only runs when a service has (re)started at a lower priority.
"""
import os
import subprocess
from collections.abc import Callable

from openpilot.common.swaglog import cloudlog

PARAM = "WifiHighPriority"
SERVICES = ("NetworkManager", "wpa_supplicant")
NICE = -10


def find_pids(names=SERVICES, proc: str = "/proc") -> dict[int, str]:
  pids = {}
  for entry in os.listdir(proc):
    if not entry.isdigit():
      continue
    try:
      with open(os.path.join(proc, entry, "comm")) as f:
        comm = f.read().strip()
    except OSError:
      continue  # exited
    if comm in names:
      pids[int(entry)] = comm
  return pids


def _renice(pid: int, nice: int) -> bool:
  return subprocess.run(["sudo", "renice", "-n", str(nice), "-p", str(pid)], capture_output=True, timeout=5).returncode == 0


def boost(enabled: bool, proc: str = "/proc", get_nice: Callable[[int], int] | None = None,
          renice: Callable[[int, int], bool] = _renice) -> list[int]:
  """Raise (or, when turned off, restore) the Wi-Fi services' priority. Returns the pids changed."""
  get_nice = get_nice or (lambda pid: os.getpriority(os.PRIO_PROCESS, pid))
  target = NICE if enabled else 0
  changed = []
  for pid, name in find_pids(proc=proc).items():
    try:
      current = get_nice(pid)
    except OSError:
      continue
    if enabled and current <= target:
      continue  # already high enough (or someone set it higher)
    if not enabled and current != NICE:
      continue  # only undo our own change
    if renice(pid, target):
      changed.append(pid)
      cloudlog.info(f"wifi_priority: {name} ({pid}) nice {current} -> {target}")
    else:
      cloudlog.warning(f"wifi_priority: renice failed for {name} ({pid})")
  return changed
