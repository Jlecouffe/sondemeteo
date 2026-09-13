"""Emission de bits pseudo-aleatoires GFSK a un debit donne, pour comparer
comment le pont audio cible degrade un contenu "facon donnees" (imprevisible,
large bande) par rapport a un ton pur a frequence equivalente (voir
tone_sweep.py). Le balayage de tons a deja ecarte l'hypothese d'un filtre
passe-bas trop etroit (des tons jusqu'a 4 kHz passent avec peu de perte) ;
ce test isole si la degradation vient plutot d'un codec avec pertes (SBC)
qui traite differemment un signal previsible (ton) et un signal imprevisible
(bruit) meme dans une bande qu'il reproduit par ailleurs correctement.

La mise en forme (filtre gaussien BT) est la meme que celle du modulateur
RS41 (modulate.py), seul le debit varie d'un test a l'autre — contrairement
au vrai RS41, dont le debit est fixe a 4800 bauds par le protocole (et par
LocalSondeDecoder.kt cote radtel-tools, qui ne sait pas decoder autre chose),
tester plusieurs debits ici sert seulement a caracteriser le pont audio, pas
a produire un signal RS41 valide a un autre debit."""
from __future__ import annotations

import numpy as np

from .modulate import _gaussian_taps


def generate_noise_fsk_chunk_iq(
    sample_rate: float,
    baud_hz: float,
    chunk_duration_s: float,
    deviation_hz: float,
    bt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Un morceau de bits aleatoires GFSK a `baud_hz`. Independant d'un
    morceau a l'autre (bits aleatoires a chaque appel) : pas besoin de
    continuite de phase au-dela de ce que le filtre gaussien lisse deja en
    debut/fin de morceau, contrairement a un ton pur (voir tone_sweep.py)."""
    sps = max(1, round(sample_rate / baud_hz))
    n_symbols = max(1, int(round(chunk_duration_s * baud_hz)))
    bits = rng.integers(0, 2, size=n_symbols).astype(np.float64)
    symbols = 2.0 * bits - 1.0
    upsampled = np.repeat(symbols, sps)
    taps = _gaussian_taps(bt, sps)
    shaped = np.convolve(upsampled, taps, mode="same")
    phase_step = 2.0 * np.pi * deviation_hz / sample_rate
    phase = np.cumsum(shaped * phase_step)
    return np.exp(1j * phase).astype(np.complex64)
