"""Codage Reed-Solomon (255,231) raccourci utilise par la RS41 pour la FEC.

Parametres GF(256) tires de l'appel `init_rs_char(8, 0x11d, 0, 1, 24, 0)`
retrouve dans les sources de decodage RS41 publiques (rs1729/RS) :
polynome de corps 0x11D, racine de depart alpha^0, pas alpha^1, 24 octets
de parite. Le code utile transmis est raccourci a 132 octets par mot de
code (au lieu des 231 nominaux) ; comme le polynome generateur ne depend
que de ses 24 racines (et non de la longueur totale du mot de code), le
raccourcissement se traite simplement en encodant directement le message
reellement transmis, sans octets de bourrage virtuels.
"""
from __future__ import annotations

GF_EXP = [0] * 512
GF_LOG = [0] * 256


def _init_tables(prim_poly: int = 0x11D) -> None:
    x = 1
    for i in range(255):
        GF_EXP[i] = x
        GF_LOG[x] = i
        x <<= 1
        if x & 0x100:
            x ^= prim_poly
    for i in range(255, 512):
        GF_EXP[i] = GF_EXP[i - 255]


_init_tables()


def gf_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return GF_EXP[GF_LOG[a] + GF_LOG[b]]


def gf_pow(a: int, power: int) -> int:
    return GF_EXP[(GF_LOG[a] * power) % 255]


NROOTS = 24


def _generator_poly(nroots: int = NROOTS, fcr: int = 0, prim: int = 1) -> list[int]:
    g = [1]
    for i in range(nroots):
        root = gf_pow(2, fcr + i * prim)
        new_g = [0] * (len(g) + 1)
        for j, coef in enumerate(g):
            new_g[j] ^= coef
            new_g[j + 1] ^= gf_mul(coef, root)
        g = new_g
    return g


_GENERATOR = _generator_poly()


def rs_parity(message: bytes, nroots: int = NROOTS) -> bytes:
    """Calcule les `nroots` octets de parite systematique pour `message`."""
    generator = _GENERATOR if nroots == NROOTS else _generator_poly(nroots)
    remainder = list(message) + [0] * nroots
    for i in range(len(message)):
        coef = remainder[i]
        if coef != 0:
            for j, gcoef in enumerate(generator):
                remainder[i + j] ^= gf_mul(gcoef, coef)
    return bytes(remainder[len(message):])


def rs41_fec_parity(data_264: bytes) -> bytes:
    """Construit les 48 octets de parite entrelaces pour les 264 octets de blocs.

    Deux mots de code entrelaces sur les octets pairs/impairs, 132 octets
    utiles + 24 octets de parite chacun.
    """
    if len(data_264) != 264:
        raise ValueError("attendu 264 octets de donnees de blocs")
    even = bytes(data_264[0::2])
    odd = bytes(data_264[1::2])
    parity_even = rs_parity(even)
    parity_odd = rs_parity(odd)
    parity = bytearray(48)
    parity[0::2] = parity_even
    parity[1::2] = parity_odd
    return bytes(parity)
