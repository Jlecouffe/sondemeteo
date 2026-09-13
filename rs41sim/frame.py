"""Assemblage d'une trame RS41 complete (brouillee, avec FEC) a partir de
parametres de simulation.

Disposition des champs calquee EXACTEMENT sur les decalages absolus lus par
LocalSondeDecoder.kt (radtel-tools), qui ne fait pas d'analyse TLV generique
mais lit des champs a position fixe dans la trame de 320 octets :

    0x000-0x007  en-tete (HEADER), transmis EN CLAIR
    0x008-0x037  parite Reed-Solomon (48 octets)
    0x038        octet reserve/non utilise
    0x039-0x03A  marqueur de bloc FRAME (0x79, 0x28) ; seul l'octet 0x039
                 est verifie par le decodeur (== 0x79)
    0x03B-0x03C  compteur de trame (u16 LE)
    0x03D-0x044  indicatif/serial (8 caracteres ASCII)
    0x045-0x111  zone non lue par le decodeur (bourrage)
    0x112-0x113  marqueur de bloc GPS3 (0x7B, 0x15) ; seul 0x112 est verifie
                 (== 0x7B)
    0x114-0x11F  position ECEF X/Y/Z (i32 LE, centimetres)
    0x120-0x125  vitesse ECEF Vx/Vy/Vz (i16 LE, cm/s)
    0x126        nombre de satellites (u8)
    0x127-0x13F  bourrage final

Le decodeur ne lit ni CRC ni PTU/GPS-time : temperature/humidite/pression/
batterie restent des parametres de SondeParams (utilises par l'IHM) mais ne
sont pas actuellement transmis dans la trame RF, faute de champ decode cote
radtel-tools. A ajouter ici le jour ou LocalSondeDecoder.kt saura les lire.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

from . import gps
from .rs_fec import rs41_fec_parity
from .scrambling import BLOCKS_LEN, HEADER, HEADER_LEN, scramble_at

FRAME_ID = bytes([0x79, 0x28])
GPS_POS_ID = bytes([0x7B, 0x15])

# Decalages relatifs au debut de la zone "blocs" (absolu - HEADER_LEN - RS_PARITY_LEN, soit absolu - 56).
REL_FRAME_ID = 0x039 - 56
REL_FRAME_NO = 0x03B - 56
REL_SERIAL = 0x03D - 56
REL_GPS_POS_ID = 0x112 - 56
REL_ECEF = 0x114 - 56
REL_ECEF_V = 0x120 - 56
REL_NUM_SATS = 0x126 - 56


@dataclass
class SondeParams:
    # Identification
    serial: str = "R1234567"
    frame_counter: int = 0

    # Position / trajectoire
    latitude_deg: float = 45.0
    longitude_deg: float = 5.0
    altitude_m: float = 500.0
    climb_rate_ms: float = 5.0       # vitesse ascensionnelle (m/s, + = monte)
    horizontal_speed_ms: float = 0.0  # vitesse horizontale (m/s)
    heading_deg: float = 0.0          # cap (degres, 0 = Nord)
    num_sats: int = 9
    pdop_x10: int = 15

    # Meteo (non transmis dans la trame actuelle, voir docstring du module)
    temperature_c: float = 15.0
    humidity_pct: float = 50.0
    pressure_hpa: float = 1013.0

    # Statut (non transmis dans la trame actuelle, voir docstring du module)
    battery_decivolts: int = 29  # 2.9 V

    def block_bytes(self) -> bytes:
        return _build_blocks(self)


def _build_blocks(p: SondeParams) -> bytes:
    blocks = bytearray(BLOCKS_LEN)

    blocks[REL_FRAME_ID:REL_FRAME_ID + 2] = FRAME_ID
    struct.pack_into("<H", blocks, REL_FRAME_NO, p.frame_counter & 0xFFFF)
    serial8 = p.serial.encode("ascii", "replace")[:8].ljust(8, b" ")
    blocks[REL_SERIAL:REL_SERIAL + 8] = serial8

    blocks[REL_GPS_POS_ID:REL_GPS_POS_ID + 2] = GPS_POS_ID
    import math
    heading_rad = math.radians(p.heading_deg)
    v_east = p.horizontal_speed_ms * math.sin(heading_rad)
    v_north = p.horizontal_speed_ms * math.cos(heading_rad)
    state = gps.make_ecef_state(
        p.latitude_deg, p.longitude_deg, p.altitude_m,
        v_east, v_north, p.climb_rate_ms,
    )
    struct.pack_into("<iii", blocks, REL_ECEF, state.x_cm, state.y_cm, state.z_cm)
    struct.pack_into("<hhh", blocks, REL_ECEF_V, state.vx_cms, state.vy_cms, state.vz_cms)
    blocks[REL_NUM_SATS] = p.num_sats & 0xFF

    return bytes(blocks)


def build_frame(p: SondeParams) -> bytes:
    """Construit la trame RS41 complete de 320 octets, prete a moduler.

    L'en-tete (8 octets) est transmis en clair ; seuls la parite et les
    blocs (312 octets, positions 8..319) sont brouilles, en poursuivant le
    cycle du masque a partir de l'index 8 (voir scrambling.scramble_at)."""
    blocks = p.block_bytes()
    parity = rs41_fec_parity(blocks)
    payload = parity + blocks
    scrambled_payload = scramble_at(payload, HEADER_LEN)
    return HEADER + scrambled_payload
