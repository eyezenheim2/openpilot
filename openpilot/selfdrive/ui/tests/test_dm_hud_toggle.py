import unittest

from openpilot.selfdrive.ui.lib.dm_hud_toggle import DmHudToggle


class TestDmHudToggle(unittest.TestCase):
  def setUp(self):
    self.gesture = DmHudToggle()

  def samples(self, visible, start, end):
    return [self.gesture.update(visible, tick / 20) for tick in range(start, end + 1)]

  def test_three_seconds_and_one_toggle_per_cover(self):
    self.assertFalse(self.gesture.update(True, 0.0))
    self.assertFalse(any(self.samples(False, 1, 60)))
    self.assertTrue(self.gesture.update(False, 3.05))
    self.assertFalse(any(self.samples(False, 62, 200)))
    self.assertFalse(self.gesture.update(True, 10.05))
    self.assertEqual(sum(self.samples(False, 202, 280)), 1)

  def test_uncover_cancels_short_cover(self):
    self.gesture.update(True, 0.0)
    self.assertFalse(any(self.samples(False, 1, 50)))
    self.gesture.update(True, 2.55)
    self.assertFalse(any(self.samples(False, 52, 111)))
    self.assertTrue(self.gesture.update(False, 5.6))

  def test_startup_covered_does_not_toggle(self):
    self.assertFalse(any(self.samples(False, 0, 200)))

  def test_reset_requires_visible_face(self):
    self.gesture.update(True, 0.0)
    self.samples(False, 1, 50)
    self.gesture.reset()
    self.assertFalse(any(self.samples(False, 51, 200)))
    self.gesture.update(True, 10.05)
    self.assertEqual(sum(self.samples(False, 202, 280)), 1)

  def test_gap_or_nonincreasing_timestamp_cancels_gesture(self):
    for timestamp in (4.0, 2.5, 0.0):
      with self.subTest(timestamp=timestamp):
        self.gesture.reset()
        self.gesture.update(True, 0.0)
        self.samples(False, 1, 50)
        self.assertFalse(self.gesture.update(False, timestamp))
        self.assertFalse(any(self.gesture.update(False, timestamp + tick / 20) for tick in range(1, 100)))


if __name__ == '__main__':
  unittest.main()
