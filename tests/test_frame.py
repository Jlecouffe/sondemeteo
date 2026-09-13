import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rs41sim.crc import crc16_ccitt
from rs41sim.frame import SondeParams, build_frame
from rs41sim.modulate import build_bitstream, gfsk_modulate
from rs41sim.rs_fec import rs41_fec_parity
from rs41sim.scrambling import HEADER, MASK, NDATA_LEN, SYNC_WORD, scramble


def decode_blocks(blocks: bytes) -> list[tuple[int, bytes]]:
    out = []
    pos = 0
    while pos < len(blocks):
        block_id, length = struct.unpack_from(">HB", blocks, pos)
        data = blocks[pos + 3: pos + 3 + length]
        (crc_read,) = struct.unpack_from("<H", blocks, pos + 3 + length)
        crc_calc = crc16_ccitt(blocks[pos: pos + 3 + length])
        assert crc_read == crc_calc, f"CRC invalide pour le bloc 0x{block_id:04X}"
        out.append((block_id, data))
        pos += 3 + length + 2
    return out


def test_sync_word_matches_known_rs41_value():
    assert SYNC_WORD == bytes([0x86, 0x35, 0xF4, 0x40, 0x93, 0xDF, 0x1A, 0x60])


def test_scramble_is_involutive():
    data = bytes(range(256)) * 2
    assert scramble(scramble(data)) == data


def test_build_frame_length_and_header():
    p = SondeParams()
    frame = build_frame(p)
    assert len(frame) == NDATA_LEN
    raw = scramble(frame)  # scramble() est sa propre inverse (XOR)
    assert raw[:8] == HEADER


def test_all_blocks_have_valid_crc_and_reach_expected_length():
    p = SondeParams(serial="TESTSN01", temperature_c=-12.3, humidity_pct=87.5,
                     pressure_hpa=234.5, latitude_deg=48.85, longitude_deg=2.35,
                     altitude_m=8500.0)
    frame = build_frame(p)
    raw = scramble(frame)
    blocks = raw[56:]
    assert len(blocks) == 264
    parsed = decode_blocks(blocks)
    ids = [bid for bid, _ in parsed]
    assert 0x7928 in ids and 0x7A2A in ids and 0x7B15 in ids and 0x7C1E in ids and 0x7611 in ids


def test_rs_parity_is_reproducible_from_blocks():
    p = SondeParams()
    frame = build_frame(p)
    raw = scramble(frame)
    blocks = raw[56:]
    parity_in_frame = raw[8:56]
    assert rs41_fec_parity(blocks) == parity_in_frame


def test_frame_counter_round_trips_through_status_block():
    p = SondeParams(frame_counter=4242)
    frame = build_frame(p)
    raw = scramble(frame)
    blocks = decode_blocks(raw[56:])
    status_data = dict(blocks)[0x7928]
    (fc,) = struct.unpack_from("<H", status_data, 0)
    assert fc == 4242


def test_bitstream_uses_lsb_first_and_alternating_preamble():
    bits = build_bitstream(bytes([0b10110000]))
    # 320 bits de preambule alterne, puis les 8 bits du seul octet, LSB en premier
    assert list(bits[318:320]) == [0.0, 1.0]
    assert list(bits[320:328]) == [0, 0, 0, 0, 1, 1, 0, 1]


def test_gfsk_modulate_produces_unit_amplitude_iq():
    p = SondeParams()
    frame = build_frame(p)
    iq = gfsk_modulate(frame, sample_rate=1_000_000.0)
    assert iq.dtype == np.complex64
    mags = np.abs(iq)
    assert np.allclose(mags, 1.0, atol=1e-3)
