from openpilot.system.hardware import wifi_priority


def fake_proc(tmp_path, procs):
  for pid, comm in procs.items():
    (tmp_path / str(pid)).mkdir()
    (tmp_path / str(pid) / "comm").write_text(comm + "\n")
  (tmp_path / "self").mkdir()
  return str(tmp_path)


def test_find_pids(tmp_path):
  proc = fake_proc(tmp_path, {101: "NetworkManager", 102: "wpa_supplicant", 103: "ui", 104: "NetworkManager-d"})
  assert wifi_priority.find_pids(proc=proc) == {101: "NetworkManager", 102: "wpa_supplicant"}


def test_boost_only_when_needed(tmp_path):
  proc = fake_proc(tmp_path, {101: "NetworkManager", 102: "wpa_supplicant"})
  nice = {101: 0, 102: -10}
  calls = []
  def renice(pid, n):
    calls.append((pid, n))
    nice[pid] = n
    return True
  assert wifi_priority.boost(True, proc=proc, get_nice=nice.__getitem__, renice=renice) == [101]
  assert calls == [(101, -10)]
  assert wifi_priority.boost(True, proc=proc, get_nice=nice.__getitem__, renice=renice) == []  # nothing left to do


def test_turning_off_restores_only_ours(tmp_path):
  proc = fake_proc(tmp_path, {101: "NetworkManager", 102: "wpa_supplicant"})
  nice = {101: -10, 102: -5}  # 102 was set by someone else
  calls = []
  wifi_priority.boost(False, proc=proc, get_nice=nice.__getitem__, renice=lambda pid, n: calls.append((pid, n)) or True)
  assert calls == [(101, 0)]


def test_failed_renice_not_reported(tmp_path):
  proc = fake_proc(tmp_path, {101: "NetworkManager"})
  assert wifi_priority.boost(True, proc=proc, get_nice=lambda pid: 0, renice=lambda pid, n: False) == []
