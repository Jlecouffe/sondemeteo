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


def _fft_convolve_same(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Equivalent de np.convolve(x, h, mode='same'), mais par FFT : pour un
    filtre gaussien long (bas debit -> beaucoup d'echantillons/symbole, voir
    _gaussian_taps), la convolution directe est O(N*M) et devient trop lente
    pour suivre le temps reel (mesure : ~11s pour generer un morceau de 5s a
    1200 bauds, soit plus de 2x plus lent que le temps reel - ca cree de
    vrais trous dans l'emission si on genere au fil de l'eau). La version FFT
    est O(N log N), environ 10x plus rapide ici, resultat identique a la
    precision flottante pres."""
    n = len(x) + len(h) - 1
    nfft = 1 << (n - 1).bit_length()
    X = np.fft.rfft(x, nfft)
    H = np.fft.rfft(h, nfft)
    y = np.fft.irfft(X * H, nfft)[:n]
    start = (len(h) - 1) // 2
    return y[start:start + len(x)]


def generate_noise_fsk_chunk_iq(
    sample_rate: float,
    baud_hz: float,
    chunk_duration_s: float,
    deviation_hz: float,
    bt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bits aleatoires GFSK a `baud_hz`, sur `chunk_duration_s` secondes.
    A appeler UNE SEULE FOIS pour generer tout un segment avant de l'emettre
    (voir app.py._run_noise_loop) : meme avec la FFT, generer par petits
    morceaux au fil de l'emission reste plus lent que le temps reel a bas
    debit (le filtre gaussien s'etale sur plusieurs milliers d'echantillons
    par symbole) - mieux vaut un seul calcul avant coup, puis des envois
    successifs de tranches du resultat deja pret."""
    sps = max(1, round(sample_rate / baud_hz))
    n_symbols = max(1, int(round(chunk_duration_s * baud_hz)))
    bits = rng.integers(0, 2, size=n_symbols).astype(np.float64)
    symbols = 2.0 * bits - 1.0
    upsampled = np.repeat(symbols, sps)
    taps = _gaussian_taps(bt, sps)
    shaped = _fft_convolve_same(upsampled, taps)
    phase_step = 2.0 * np.pi * deviation_hz / sample_rate
    phase = np.cumsum(shaped * phase_step)
    return np.exp(1j * phase).astype(np.complex64)
