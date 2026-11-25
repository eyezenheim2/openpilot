# Drive narration voices

The spoken drive narration plays pre-rendered clips, so the comma never runs a speech engine:
`soundd` joins a few short WAVs ("Set to" + "65" + "Speeding up.") when a caption appears.

Clips live in `openpilot/selfdrive/assets/sounds/narration/<voice>/`, one per phrase and one per number
(1-160). They are regular git files, not LFS (see `.gitattributes`), so a plain clone has them.

## Voices

| Setting | Directory | Source |
|---|---|---|
| Voice A | `voice_a` | Piper `en-us-libritts-high`, speaker 832 |
| Voice B | `voice_b` | Piper `en-us-libritts-high`, speaker 348 |
| Voice C | `voice_c` | Piper `en-us-libritts-high`, speaker 444 |
| Voice D | `voice_d` | Piper `en-us-libritts-high`, speaker 656 |

Speakers were picked as the LibriTTS voices closest to Piper's `lessac` voice in pitch, pitch range, pace and
brightness.

Credits: voices synthesized with [Piper](https://github.com/rhasspy/piper) (MIT) using its LibriTTS model,
trained on [LibriTTS](http://www.openslr.org/60/) (CC BY 4.0, Zen et al., 2019).

## Regenerating

Edit the wording in `openpilot/selfdrive/ui/narration_text.py`, then:

```
pip install piper-tts
curl -LO https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-en-us-libritts-high.tar.gz
tar xzf voice-en-us-libritts-high.tar.gz
python3 tools/narration/make_voices.py en-us-libritts-high.onnx
```

To add a voice, add a row to `VOICES` in `narration_text.py` (and an option in the sunnylink settings page).
