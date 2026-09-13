"""Modulation GFSK en bande de base (echantillons IQ complexes) pour la RS41.

RS41 : 4800 bauds, 2FSK avec mise en forme gaussienne, bits emis LSB en
premier dans chaque octet. Un preambule de 320 bits alternes precede la
trame (soit 320/4800 = 66.7 ms), suivi de la trame brouillee de 320 octets.

BT par defaut = 4.0 (au lieu de 0.5, spec RS41 reelle) : verifie par
simulation bout-en-bout AU DEBIT REEL DU PONT RT-660 (32 kHz, voir
verify_decode.py et test_absolute.py), le decodeur cote radtel-tools
(LocalSondeDecoder.kt) ne fait qu'un comptage de passages par zero sans
filtre adapte. En dessous de BT=3.0 a 32 kHz, le calage bit derive au fil
de la trame : le marqueur GPS3 peut meme tomber juste par coincidence avec
des donnees ECEF fausses derriere (pas seulement un echec franc, un faux
positif silencieux). BT=0.5 reste plus proche du spectre reel Vaisala si
jamais le decodeur cible change.
"""
from __future__ import annotations

import numpy as np

from .scrambling import PREAMBLE_BITS

BAUD_RATE = 4800.0


def _bytes_to_bits_lsb_first(data: bytes) -> np.ndarray:
    bits = np.zeros(len(data) * 8, dtype=np.float64)
    for i, byte in enumerate(data):
        for b in range(8):
            bits[i * 8 + b] = (byte >> b) & 1
    return bits


def _gaussian_taps(bt: float, sps: int, span_symbols: int = 4) -> np.ndarray:
    """Reponse impulsionnelle d'un filtre gaussien, normalisee (gain unite)."""
    ntaps = span_symbols * sps + 1
    t = (np.arange(ntaps) - (ntaps - 1) / 2) / sps
    alpha = np.sqrt(np.log(2) / 2) / bt
    h = np.exp(-(t ** 2) / (2 * alpha ** 2))
    return h / np.sum(h)


def build_bitstream(frame_bytes: bytes) -> np.ndarray:
    """Preambule (bits alternes) + trame (LSB en premier), valeurs {0,1}."""
    preamble = np.array([i % 2 for i in range(PREAMBLE_BITS)], dtype=np.float64)
    frame_bits = _bytes_to_bits_lsb_first(frame_bytes)
    return np.concatenate([preamble, frame_bits])


def gfsk_modulate(frame_bytes: bytes, sample_rate: float, deviation_hz: float = 2400.0,
                   bt: float = 4.0, amplitude: float = 1.0) -> np.ndarray:
    """Retourne un tableau complex128 d'echantillons IQ en bande de base."""
    bits = build_bitstream(frame_bytes)
    symbols = 2.0 * bits - 1.0  # NRZ {-1, +1}

    sps = max(1, round(sample_rate / BAUD_RATE))
    upsampled = np.repeat(symbols, sps)

    taps = _gaussian_taps(bt, sps)
    shaped = np.convolve(upsampled, taps, mode="same")

    phase_step = 2.0 * np.pi * deviation_hz / sample_rate
    phase = np.cumsum(shaped * phase_step)
    iq = amplitude * np.exp(1j * phase)
    return iq.astype(np.complex64)


def append_silence(iq: np.ndarray, sample_rate: float, seconds: float) -> np.ndarray:
    if seconds <= 0:
        return iq
    pad = np.zeros(round(sample_rate * seconds), dtype=iq.dtype)
    return np.concatenate([iq, pad])
