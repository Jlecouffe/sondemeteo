"""Constantes de trame RS41 : mot d'en-tete et masque de brouillage.

Valeurs issues des projets de retro-ingenierie publics (rs1729/RS,
projecthorus/radiosonde_auto_rx, bazjo/RS41_Decoding). Le motif d'en-tete
et le masque sont mutuellement coherents : HEADER[i] ^ MASK[i] == SYNC_WORD[i]
pour i < 8, ce qui a ete verifie a la construction de ce module.

IMPORTANT (corrige le 2026-09-13, cf. radtel-tools LocalSondeDecoder.kt) :
HEADER est le mot reellement emis EN CLAIR sur l'air pour la synchronisation
bit ; le brouillage XOR ne s'applique qu'a partir de l'octet 8 (parite +
blocs), jamais aux 8 octets d'en-tete eux-memes. Les decodeurs (radtel-tools
comme rs1729) correlent directement sur le motif HEADER dans le flux
demodule, avant tout retrait de brouillage. SYNC_WORD (HEADER ^ MASK[:8])
n'est PAS ce qui est transmis : c'etait une erreur de modelisation de
l'ancienne version de frame.build_frame(), qui brouillait la trame entiere
y compris l'en-tete et empechait donc toute synchronisation en reception.
"""
from __future__ import annotations

# Masque XOR de brouillage (64 octets, cyclique sur toute la trame).
MASK = bytes([
    0x96, 0x83, 0x3E, 0x51, 0xB1, 0x49, 0x08, 0x98,
    0x32, 0x05, 0x59, 0x0E, 0xF9, 0x44, 0xC6, 0x26,
    0x21, 0x60, 0xC2, 0xEA, 0x79, 0x5D, 0x6D, 0xA1,
    0x54, 0x69, 0x47, 0x0C, 0xDC, 0xE8, 0x5C, 0xF1,
    0xF7, 0x76, 0x82, 0x7F, 0x07, 0x99, 0xA2, 0x2C,
    0x93, 0x7C, 0x30, 0x63, 0xF5, 0x10, 0x2E, 0x61,
    0xD0, 0xBC, 0xB4, 0xB6, 0x06, 0xAA, 0xF4, 0x23,
    0x78, 0x6E, 0x3B, 0xAE, 0xBF, 0x7B, 0x4C, 0xC1,
])

# En-tete de trame, dans le domaine "non brouille" (avant XOR par MASK).
HEADER = bytes([0x10, 0xB6, 0xCA, 0x11, 0x22, 0x96, 0x12, 0xF8])

# Mot de synchronisation reellement emis sur l'air (HEADER ^ MASK[:8]).
SYNC_WORD = bytes(h ^ m for h, m in zip(HEADER, MASK[:8]))
assert SYNC_WORD == bytes([0x86, 0x35, 0xF4, 0x40, 0x93, 0xDF, 0x1A, 0x60])

NDATA_LEN = 320          # longueur d'une trame standard (sans XDATA)
HEADER_LEN = 8
RS_PARITY_LEN = 48       # 2 x 24 octets de parite Reed-Solomon, entrelaces
BLOCKS_LEN = NDATA_LEN - HEADER_LEN - RS_PARITY_LEN  # 264 octets de blocs

# Nombre de bits du preambule (motif alterne 0/1) avant l'en-tete.
PREAMBLE_BITS = 320


def scramble(data: bytes) -> bytes:
    """Applique/retire le brouillage XOR cyclique sur `data`, index 0 = MASK[0]."""
    return scramble_at(data, 0)


def scramble_at(data: bytes, start_index: int) -> bytes:
    """Applique/retire le brouillage XOR cyclique sur `data`, en continuant le
    cycle de MASK a partir de `start_index` (utile pour brouiller uniquement
    la partie de la trame qui suit l'en-tete, qui lui doit rester en clair)."""
    return bytes(b ^ MASK[(start_index + i) % len(MASK)] for i, b in enumerate(data))
