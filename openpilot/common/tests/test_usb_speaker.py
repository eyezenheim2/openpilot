from openpilot.common.hardware.usb import usb_speaker_present


def test_usb_speaker_present(tmp_path):
  cards = tmp_path / "cards"
  cards.write_text(" 0 [sdm845tavilsnd]: sdm845-tavil-s - sdm845-tavil-snd-card\n")
  assert not usb_speaker_present(cards)
  cards.write_text(cards.read_text() + " 1 [Device         ]: USB-Audio - USB Audio Device\n")
  assert usb_speaker_present(cards)
  assert not usb_speaker_present(tmp_path / "missing")
