import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from openpilot.selfdrive.locationd.paramsd import initialize_fixed_steer_ratio, read_fixed_steer_ratio


class TestFixedSteerRatio(unittest.TestCase):
  def setUp(self):
    self.directory = tempfile.TemporaryDirectory()
    self.addCleanup(self.directory.cleanup)
    self.path = Path(self.directory.name) / 'FixedSteerRatio'
    self.cp = SimpleNamespace(steerRatio=15.25)

  def test_missing_file_initializes_fixed_ratio(self):
    initialize_fixed_steer_ratio(self.path, self.cp)
    self.assertEqual(self.path.read_text(), '15.25')
    self.assertEqual(read_fixed_steer_ratio(self.path, self.cp), 15.25)

  def test_existing_file_is_not_overwritten(self):
    for value in ('16.1', '', 'invalid'):
      with self.subTest(value=value):
        self.path.write_text(value)
        initialize_fixed_steer_ratio(self.path, self.cp)
        self.assertEqual(self.path.read_text(), value)

  def test_default_respects_hard_cap(self):
    self.cp.steerRatio = 24.0
    initialize_fixed_steer_ratio(self.path, self.cp)
    self.assertEqual(read_fixed_steer_ratio(self.path, self.cp), 23.33)

  def test_read_picks_up_edits_and_removal(self):
    initialize_fixed_steer_ratio(self.path, self.cp)
    self.path.write_text('16.1')
    self.assertEqual(read_fixed_steer_ratio(self.path, self.cp), 16.1)
    self.path.unlink()
    self.assertIsNone(read_fixed_steer_ratio(self.path, self.cp))
    self.assertFalse(self.path.exists())

  def test_creation_error_does_not_stop_paramsd(self):
    with patch.object(Path, 'open', side_effect=PermissionError('read-only')):
      initialize_fixed_steer_ratio(self.path, self.cp)
    self.assertFalse(self.path.exists())
