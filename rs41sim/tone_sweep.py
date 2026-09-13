"""Generation d'un balayage de tons purs, pour caracteriser la reponse en
frequence audio reelle d'un pont FM (ex: le pont Bluetooth du RT-660), au
lieu de deduire cette reponse depuis le spectre d'un seul signal complexe
(RS41). Chaque ton est un signal audio sinusoidal de frequence fixe injecte
dans le modulateur FM (comme un testeur de dechargerait un vrai audio dans
un emetteur voix) : le recepteur, s'il demodule correctement, doit
reproduire ce meme ton en sortie. Comparer l'amplitude recue ton par ton
donne la reponse en frequence reelle de la chaine, contrairement a un
signal RS41 dont le contenu spectral melange plusieurs choses a la fois."""
from __future__ import annotations

import numpy as np


def generate_tone_sweep_iq(
    sample_rate: float,
    tone_freqs_hz: list[float],
    tone_duration_s: float = 3.0,
    deviation_hz: float = 2400.0,
) -> tuple[np.ndarray, list[tuple[float, float, float]]]:
    """Retourne (iq, segments) ou segments est une liste de
    (frequence_hz, debut_s, fin_s) pour reperer chaque ton dans le flux
    resultant (utile pour aligner une capture audio avec le ton emis a un
    instant donne)."""
    segments: list[tuple[float, float, float]] = []
    chunks: list[np.ndarray] = []
    t_cursor = 0.0
    for f_tone in tone_freqs_hz:
        n = int(round(tone_duration_s * sample_rate))
        t = np.arange(n) / sample_rate
        audio = np.sin(2.0 * np.pi * f_tone * t)
        phase_step = 2.0 * np.pi * deviation_hz / sample_rate
        phase = np.cumsum(audio * phase_step)
        chunks.append(np.exp(1j * phase).astype(np.complex64))
        segments.append((f_tone, t_cursor, t_cursor + tone_duration_s))
        t_cursor += tone_duration_s
    return np.concatenate(chunks), segments
