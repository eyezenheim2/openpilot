DM_HUD_HOLD_SECONDS = 3.0
DM_MAX_SAMPLE_GAP_SECONDS = 0.5


class DmHudToggle:
  """One HUD toggle per face-loss gesture; face detection is a cover proxy."""

  def __init__(self):
    self.reset()

  def reset(self) -> None:
    self._armed = False
    self._blocked_since: float | None = None
    self._last_sample: float | None = None

  def update(self, face_detected: bool, timestamp: float) -> bool:
    if self._last_sample is not None:
      gap = timestamp - self._last_sample
      if gap <= 0 or gap > DM_MAX_SAMPLE_GAP_SECONDS:
        self.reset()
    self._last_sample = timestamp

    if face_detected:
      self._armed = True
      self._blocked_since = None
    elif self._armed:
      if self._blocked_since is None:
        self._blocked_since = timestamp
      elif timestamp >= self._blocked_since + DM_HUD_HOLD_SECONDS:
        self._armed = False
        self._blocked_since = None
        return True
    return False
