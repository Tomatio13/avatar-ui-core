"""
SoundManager for terminal app
- Reproduces static/js/sound.js typing sound (square wave + exponential fade)
- Uses settings from .env via settings.py
- Graceful no-op if pygame/numpy are unavailable or audio init fails
"""
from __future__ import annotations

import time

try:
    import numpy as _np  # type: ignore
    import pygame as _pygame  # type: ignore
    _HAVE_DEPS = True
except Exception:
    _np = None
    _pygame = None
    _HAVE_DEPS = False


class SoundManager:
    def __init__(self, settings_module):
        self.settings = settings_module
        self.sample_rate = 44100
        self._sound_cache: bytes | None = None
        self._enabled = False
        self._init_audio()

    def _init_audio(self):
        if not _HAVE_DEPS:
            return
        try:
            if not _pygame.mixer.get_init():
                _pygame.mixer.init(frequency=self.sample_rate, size=-16, channels=1, buffer=512)
                _pygame.mixer.set_num_channels(8)
            # Warm-up: play short silence to wake device
            n = int(self.sample_rate * 0.02)
            silent = _np.zeros(n, dtype=_np.int16).tobytes()
            try:
                _pygame.mixer.Sound(buffer=silent).play()
                _pygame.time.wait(30)
            except Exception:
                pass
            self._enabled = True
        except Exception:
            self._enabled = False

    def is_enabled(self) -> bool:
        return bool(self._enabled)

    def play_type_sound(self):
        if not self._enabled or not _HAVE_DEPS:
            return
        try:
            # Cache-generated waveform for performance and stability
            if self._sound_cache is None:
                freq = float(self.settings.BEEP_FREQUENCY_HZ)
                duration_ms = int(self.settings.BEEP_DURATION_MS)
                vol_start = float(self.settings.BEEP_VOLUME)
                vol_end = float(self.settings.BEEP_VOLUME_END)
                n_samples = int(self.sample_rate * (duration_ms / 1000.0))
                if n_samples <= 0:
                    return
                t = _np.linspace(0, duration_ms / 1000.0, n_samples, endpoint=False)
                square = _np.sign(_np.sin(2 * _np.pi * freq * t))
                if vol_start <= 0:
                    vol_start = 1e-6
                if vol_end <= 0:
                    vol_end = 1e-6
                env = vol_start * _np.power((vol_end / vol_start), t / (duration_ms / 1000.0))
                wave = square * env
                wave_int16 = (wave * 32767.0).astype(_np.int16)
                self._sound_cache = wave_int16.tobytes()
            _pygame.mixer.Sound(buffer=self._sound_cache).play()
        except Exception:
            # On any runtime audio error, disable to avoid spamming
            self._enabled = False

