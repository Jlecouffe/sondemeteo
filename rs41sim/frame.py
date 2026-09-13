"""Assemblage d'une trame RS41 complete (brouillee, avec FEC) a partir de
parametres de simulation.

Portions verifiees par recoupement de plusieurs sources publiques
(rs1729/RS, projecthorus/radiosonde_auto_rx, bazjo/RS41_Decoding) :
mot d'en-tete, masque de brouillage, CRC16, parametres GF(256)/Reed-Solomon.

Portions simplifiees / non garanties bit-a-bit conformes au format Vaisala
reel (a adapter au besoin dans ton propre decodeur, voir docstrings) :
- ordre des octets et disposition exacte des identifiants de bloc,
- bloc PTU : ce module encode directement temperature/humidite/pression
  sous forme de valeurs scalees, au lieu du schema reel Vaisala
  (mesures brutes + polynome de calibration transmis par fragments).
  Adapte cette partie si tu as besoin de reproduire exactement l'algorithme
  Vaisala plutot que de simplement fournir des valeurs lisibles a decoder.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from datetime import datetime, timezone

from . import gps
from .crc import crc16_ccitt
from .rs_fec import rs41_fec_parity
from .scrambling import BLOCKS_LEN, HEADER, scramble

STATUS_ID = 0x7928
GPS_TIME_ID = 0x7C1E
GPS_POS_ID = 0x7B15
PTU_ID = 0x7A2A
ZERO_ID = 0x7611


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

    # Meteo
    temperature_c: float = 15.0
    humidity_pct: float = 50.0
    pressure_hpa: float = 1013.0

    # Statut
    battery_decivolts: int = 29  # 2.9 V

    def block_bytes(self) -> bytes:
        return _build_blocks(self)


def _build_block(block_id: int, data: bytes) -> bytes:
    if len(data) > 255:
        raise ValueError("bloc trop long")
    header = struct.pack(">HB", block_id, len(data))
    crc = crc16_ccitt(header + data)
    return header + data + struct.pack("<H", crc)


def _status_block(p: SondeParams) -> bytes:
    serial8 = p.serial.encode("ascii", "replace")[:8].ljust(8, b" ")
    data = struct.pack("<H", p.frame_counter & 0xFFFF) + serial8 + \
        struct.pack("<BB", p.battery_decivolts & 0xFF, 0x00)
    return _build_block(STATUS_ID, data)


def _gps_time_block(p: SondeParams) -> bytes:
    week, tow_ms = gps.gps_week_tow(datetime.now(timezone.utc))
    data = struct.pack("<HIBB", week & 0xFFFF, tow_ms & 0xFFFFFFFF,
                        p.num_sats & 0xFF, p.pdop_x10 & 0xFF)
    return _build_block(GPS_TIME_ID, data)


def _gps_pos_block(p: SondeParams) -> bytes:
    import math
    heading_rad = math.radians(p.heading_deg)
    v_east = p.horizontal_speed_ms * math.sin(heading_rad)
    v_north = p.horizontal_speed_ms * math.cos(heading_rad)
    state = gps.make_ecef_state(
        p.latitude_deg, p.longitude_deg, p.altitude_m,
        v_east, v_north, p.climb_rate_ms,
    )
    data = struct.pack(
        "<iiihhhBBB",
        state.x_cm, state.y_cm, state.z_cm,
        state.vx_cms, state.vy_cms, state.vz_cms,
        p.num_sats & 0xFF, 0, p.pdop_x10 & 0xFF,
    )
    return _build_block(GPS_POS_ID, data)


def _ptu_block(p: SondeParams) -> bytes:
    """Encodage simplifie (non standard Vaisala) : valeurs directes scalees."""
    temp_x100 = round(p.temperature_c * 100)
    hum_x10 = max(0, min(1000, round(p.humidity_pct * 10)))
    press_x10 = max(0, min(65535, round(p.pressure_hpa * 10)))
    data = struct.pack("<hHH", temp_x100, hum_x10, press_x10)
    return _build_block(PTU_ID, data)


def _zero_block(remaining_total_len: int) -> bytes:
    data_len = remaining_total_len - 5
    if data_len < 0:
        raise ValueError("plus de place pour le bloc de bourrage")
    return _build_block(ZERO_ID, bytes(data_len))


def _build_blocks(p: SondeParams) -> bytes:
    blocks = [
        _status_block(p),
        _gps_time_block(p),
        _gps_pos_block(p),
        _ptu_block(p),
    ]
    used = sum(len(b) for b in blocks)
    blocks.append(_zero_block(BLOCKS_LEN - used))
    out = b"".join(blocks)
    assert len(out) == BLOCKS_LEN, f"attendu {BLOCKS_LEN} octets de blocs, obtenu {len(out)}"
    return out


def build_frame(p: SondeParams) -> bytes:
    """Construit la trame RS41 complete de 320 octets, brouillee et prete a moduler."""
    blocks = p.block_bytes()
    parity = rs41_fec_parity(blocks)
    raw = HEADER + parity + blocks
    return scramble(raw)
