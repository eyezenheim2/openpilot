import math
import numpy as np
import time
import wave


from openpilot.cereal import log, messaging, custom
from openpilot.common.basedir import BASEDIR
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import Ratekeeper
from openpilot.common.utils import retry
from openpilot.common.swaglog import cloudlog

from openpilot.system import micd
from openpilot.common.hardware import HARDWARE

from openpilot.sunnypilot.selfdrive.ui.quiet_mode import QuietMode, ALERTS_ALWAYS_PLAY
from openpilot.selfdrive.ui.audio_routing import NarrationPlayer, narration_gain, plan_outputs, usb_card_present

SAMPLE_RATE = 48000
SAMPLE_BUFFER = 4096 # (approx 100ms)
MAX_VOLUME = 1.0
MIN_VOLUME = 0.1
ALERT_RAMP_TIME = 4 # seconds to ramp to max volume for warningImmediate
SELFDRIVE_STATE_TIMEOUT = 5 # 5 seconds
ROUTING_CHECK_FRAMES = 40  # 2 s at 20 Hz: look for a USB speaker being plugged/unplugged or the setting changing
NARRATION_POLL_FRAMES = 2  # 10 Hz
FILTER_DT = 1. / (micd.SAMPLE_RATE / micd.FFT_SAMPLES)

AMBIENT_DB = 26 # DB where MIN_VOLUME is applied
DB_SCALE = 30 # AMBIENT_DB + DB_SCALE is where MAX_VOLUME is applied

VOLUME_BASE = 20
if HARDWARE.get_device_type() == "tizi":
  AMBIENT_DB = 30
  VOLUME_BASE = 10

AudibleAlert = log.SelfdriveState.AudibleAlert
AudibleAlertSP = custom.SelfdriveStateSP.AudibleAlert


sound_list_sp: dict[int, tuple[str, int | None, float]] = {
  # AudibleAlertSP, file name, play count (none for infinite)
  AudibleAlertSP.promptSingleLow: ("prompt_single_low.wav", 1, MAX_VOLUME),
  AudibleAlertSP.promptSingleHigh: ("prompt_single_high.wav", 1, MAX_VOLUME),
}

sound_list: dict[int, tuple[str, int | None, float]] = {
  # AudibleAlert, file name, play count (none for infinite)
  AudibleAlert.engage: ("engage.wav", 1, MAX_VOLUME),
  AudibleAlert.disengage: ("disengage.wav", 1, MAX_VOLUME),
  AudibleAlert.refuse: ("refuse.wav", 1, MAX_VOLUME),

  AudibleAlert.prompt: ("warning.wav", 1, MAX_VOLUME),
  AudibleAlert.promptRepeat: ("warning.wav", None, MAX_VOLUME),
  AudibleAlert.promptDistracted: ("dm_warning.wav", None, MAX_VOLUME),

  AudibleAlert.preAlert: ("pre_alert.wav", 1, MAX_VOLUME),

  AudibleAlert.warningSoft: ("critical.wav", None, MAX_VOLUME),
  AudibleAlert.warningImmediate: ("dm_critical.wav", None, MAX_VOLUME),

  **sound_list_sp,
}

def check_selfdrive_timeout_alert(sm):
  ss_missing = time.monotonic() - sm.recv_time['selfdriveState']

  if ss_missing > SELFDRIVE_STATE_TIMEOUT:
    if (sm['selfdriveState'].enabled or sm['selfdriveStateSP'].mads.enabled) and (ss_missing - SELFDRIVE_STATE_TIMEOUT) < 10:
      return True

  return False


class Soundd(QuietMode):
  def __init__(self):
    super().__init__()

    self.load_sounds()

    self.current_alert = AudibleAlert.none
    self.current_volume = MIN_VOLUME
    self.current_sound_frame = 0

    self.ramp_start_volume = MIN_VOLUME
    self.ramp_start_time = 0.

    self.selfdrive_timeout_alert = False
    self.pending_stop = False

    self.spl_filter_weighted = FirstOrderFilter(0, 2.5, FILTER_DT, initialized=False)

    # USB speaker routing and drive narration (see audio_routing.py)
    self.narration = NarrationPlayer(SAMPLE_RATE)
    self.narration_volume = 80
    self.backup_alert = AudibleAlert.none
    self.backup_sound_frame = 0

  def load_sounds(self):
    self.loaded_sounds: dict[int, np.ndarray] = {}

    # Load all sounds
    for sound in sound_list:
      filename, play_count, volume = sound_list[sound]

      with wave.open(BASEDIR + "/openpilot/selfdrive/assets/sounds/" + filename, 'r') as wavefile:
        assert wavefile.getnchannels() == 1
        assert wavefile.getsampwidth() == 2
        assert wavefile.getframerate() == SAMPLE_RATE

        length = wavefile.getnframes()
        self.loaded_sounds[sound] = np.frombuffer(wavefile.readframes(length), dtype=np.int16).astype(np.float32) / (2**16/2)

  def get_sound_data(self, frames): # get "frames" worth of data from the current alert sound, looping when required

    ret = np.zeros(frames, dtype=np.float32)

    if self.should_play_sound(self.current_alert):
      num_loops = sound_list[self.current_alert][1]
      sound_data = self.loaded_sounds[self.current_alert]
      written_frames = 0

      current_sound_frame = self.current_sound_frame % len(sound_data)
      loops = self.current_sound_frame // len(sound_data)

      while written_frames < frames and (num_loops is None or loops < num_loops):
        available_frames = sound_data.shape[0] - current_sound_frame
        frames_to_write = min(available_frames, frames - written_frames)
        ret[written_frames:written_frames+frames_to_write] = sound_data[current_sound_frame:current_sound_frame+frames_to_write]
        written_frames += frames_to_write
        self.current_sound_frame += frames_to_write
        current_sound_frame = self.current_sound_frame % len(sound_data)
        loops = self.current_sound_frame // len(sound_data)
        if self.pending_stop and current_sound_frame == 0:
          self.current_alert = AudibleAlert.none
          self.pending_stop = False
          break

    return ret * self.current_volume

  def callback(self, data_out: np.ndarray, frames: int, time, status) -> None:
    if status:
      cloudlog.warning(f"soundd stream over/underflow: {status}")
    data = self.get_sound_data(frames)
    if self.should_play_sound(self.current_alert):
      self.narration.stop()  # never talk over an alert
    elif self.narration.playing:
      volume = self.narration.volume if self.narration.volume is not None else self.narration_volume
      data = data + self.narration.read(frames) * narration_gain(volume, self.current_volume)
    data_out[:frames, 0] = data

  def backup_callback(self, data_out: np.ndarray, frames: int, time, status) -> None:
    """Built-in speaker while the USB speaker is the main output: urgent alarms only (all of them loop)."""
    alert = self.current_alert
    if alert != self.backup_alert:
      self.backup_alert, self.backup_sound_frame = alert, 0
    if alert not in ALERTS_ALWAYS_PLAY or alert not in self.loaded_sounds:
      data_out[:frames, 0] = 0
      return
    sound = self.loaded_sounds[alert]
    idx = (self.backup_sound_frame + np.arange(frames)) % len(sound)
    self.backup_sound_frame += frames
    data_out[:frames, 0] = sound[idx] * self.current_volume

  def update_alert(self, new_alert):
    current_alert_played_once = self.current_alert == AudibleAlert.none or self.current_sound_frame >= len(self.loaded_sounds[self.current_alert])
    # let looping sounds finish the current loop instead of cutting off mid tone
    if new_alert == AudibleAlert.none and self.current_alert != AudibleAlert.none and sound_list[self.current_alert][1] is None:
      if current_alert_played_once:
        self.pending_stop = True
      else:
        self.current_alert = AudibleAlert.none
        self.current_sound_frame = 0
      return
    self.pending_stop = False
    if self.current_alert != new_alert and (new_alert != AudibleAlert.none or current_alert_played_once):
      if new_alert == AudibleAlert.warningImmediate:
        self.ramp_start_volume = self.current_volume
        self.ramp_start_time = time.monotonic()
      self.current_alert = new_alert
      self.current_sound_frame = 0

  def get_audible_alert(self, sm):
    if sm.updated['selfdriveState']:
      new_alert = sm['selfdriveState'].alertSound.raw
      self.update_alert(new_alert)
    elif check_selfdrive_timeout_alert(sm):
      self.update_alert(AudibleAlert.warningImmediate)
      self.selfdrive_timeout_alert = True
    elif self.selfdrive_timeout_alert:
      self.update_alert(AudibleAlert.none)
      self.selfdrive_timeout_alert = False

  def calculate_volume(self, weighted_db):
    volume = ((weighted_db - AMBIENT_DB) / DB_SCALE) * (MAX_VOLUME - MIN_VOLUME) + MIN_VOLUME
    return math.pow(VOLUME_BASE, (np.clip(volume, MIN_VOLUME, MAX_VOLUME) - 1))

  def routing_settings(self) -> tuple[bool, bool]:
    return self.params.get_bool("UsbAudioOutput"), self.params.get_bool("UsbAudioAlarmBackup")

  @retry(attempts=10, delay=3)
  def get_streams(self, sd, usb_enabled: bool, backup_enabled: bool):
    """Open the output streams: (main, backup or None, main is USB)."""
    # reload sounddevice to reinitialize portaudio (also picks up a newly plugged USB speaker)
    sd._terminate()
    sd._initialize()
    try:
      devices = list(sd.query_devices())
    except Exception:
      cloudlog.exception("soundd: could not list audio devices")
      devices = []
    usb_device, want_backup = plan_outputs(devices, usb_enabled, backup_enabled)

    def open_stream(device, callback):
      return sd.OutputStream(device=device, channels=1, samplerate=SAMPLE_RATE, callback=callback, blocksize=SAMPLE_BUFFER)

    if usb_device is not None:
      try:
        main = open_stream(usb_device, self.callback)
      except Exception:
        cloudlog.exception(f"soundd: USB speaker {devices[usb_device].get('name')} failed to open, using the built-in speaker")
        return open_stream(None, self.callback), None, False
      backup = None
      if want_backup:
        try:
          backup = open_stream(None, self.backup_callback)
        except Exception:
          cloudlog.exception("soundd: built-in backup stream failed to open")
      return main, backup, True
    return open_stream(None, self.callback), None, False

  def soundd_thread(self):
    # sounddevice must be imported after forking processes
    import sounddevice as sd
    micd.patch_sounddevice(sd)

    sm = messaging.SubMaster(['selfdriveState', 'selfdriveStateSP', 'soundPressure'])

    while True:
      settings = self.routing_settings()
      usb_present = usb_card_present()
      main, backup, on_usb = self.get_streams(sd, *settings)
      streams = [s for s in (main, backup) if s is not None]
      try:
        for s in streams:
          s.start()
        cloudlog.info(f"soundd stream started: {main.samplerate=} {main.channels=} {main.dtype=} {main.device=}, {main.blocksize=}, " +
                      f"usb={on_usb} backup={backup is not None}")
        self.run_streams(sm, streams, on_usb, settings, usb_present)
      finally:
        for s in streams:
          try:
            s.close()
          except Exception:
            cloudlog.exception("soundd: stream close failed")

  def run_streams(self, sm, streams, on_usb: bool, settings: tuple[bool, bool], usb_present: bool) -> None:
    """Run until the outputs need reopening (USB speaker plugged/unplugged/failed, or the setting changed)."""
    rk = Ratekeeper(20)
    frame = 0
    self.narration_volume = int(self.params.get("DriveNarrationVolume", return_default=True) or 0)
    while True:
      sm.update(0)
      frame += 1

      self.load_param()

      if frame % NARRATION_POLL_FRAMES == 0:
        self.narration.handle_request(self.params.get("DriveNarrationSay"), now=time.monotonic())
      if frame % ROUTING_CHECK_FRAMES == 0:
        self.narration_volume = int(self.params.get("DriveNarrationVolume", return_default=True) or 0)
        if usb_card_present() != usb_present or self.routing_settings() != settings:
          cloudlog.info("soundd: audio outputs changed, reopening")
          return

      # freeze volume during alerts to avoid mic feedback increasing volume
      if sm.updated['soundPressure']:
        self.spl_filter_weighted.update(sm["soundPressure"].soundPressureWeightedDb)
        if self.current_alert == AudibleAlert.none:
          self.current_volume = self.calculate_volume(float(self.spl_filter_weighted.x))

      self.get_audible_alert(sm)

      # Ramp up immediate warning sound over 4s
      if self.current_alert == AudibleAlert.warningImmediate:
        elapsed = time.monotonic() - self.ramp_start_time
        ramp_vol = float(np.interp(elapsed, [0, ALERT_RAMP_TIME], [self.ramp_start_volume, MAX_VOLUME]))
        self.current_volume = max(self.current_volume, ramp_vol)

      rk.keep_time()

      if not all(s.active for s in streams):
        if on_usb:
          cloudlog.error("soundd: USB output stopped, reopening")
          return
        raise AssertionError("soundd stream stopped")


def main():
  s = Soundd()
  s.soundd_thread()


if __name__ == "__main__":
  main()
