import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rs41sim.frame import SondeParams, build_frame
from rs41sim.modulate import build_bitstream, gfsk_modulate
from rs41sim.rs_fec import rs41_fec_parity
from rs41sim.scrambling import HEADER, MASK, NDATA_LEN, scramble_at


def test_build_frame_length_and_clear_header():
    """L'en-tete est transmis EN CLAIR : LocalSondeDecoder.kt (radtel-tools)
    correle sur ce motif directement dans le flux demodule, avant tout
    retrait de brouillage."""
    p = SondeParams()
    frame = build_frame(p)
    assert len(frame) == NDATA_LEN
    assert frame[:8] == HEADER


def test_scramble_at_is_involutive():
    data = bytes(range(256)) * 2
    assert scramble_at(scramble_at(data, 8), 8) == data


def test_payload_after_header_matches_scrambled_parity_and_blocks():
    p = SondeParams(serial="TESTSN01", frame_counter=4242,
                     latitude_deg=48.85, longitude_deg=2.35, altitude_m=8500.0)
    frame = build_frame(p)
    blocks = p.block_bytes()
    parity = rs41_fec_parity(blocks)
    expected_payload = scramble_at(parity + blocks, 8)
    assert frame[8:] == expected_payload


def test_fixed_offsets_match_local_sonde_decoder():
    """Verifie que les octets tombent exactement aux decalages absolus lus
    par LocalSondeDecoder.kt une fois le brouillage retire (voir docstring
    de rs41sim.frame)."""
    p = SondeParams(serial="TESTSN01", frame_counter=4242,
                     latitude_deg=48.85, longitude_deg=2.35, altitude_m=8500.0,
                     num_sats=11)
    frame = build_frame(p)
    unscrambled_payload = scramble_at(frame[8:], 8)
    raw = frame[:8] + unscrambled_payload

    assert raw[0x039] == 0x79
    assert raw[0x03A] == 0x28
    (frame_no,) = struct.unpack_from("<H", raw, 0x03B)
    assert frame_no == 4242
    assert raw[0x03D:0x045] == b"TESTSN01"

    assert raw[0x112] == 0x7B
    assert raw[0x113] == 0x15
    assert raw[0x126] == 11


def test_frame_counter_and_serial_round_trip():
    p = SondeParams(serial="ABCD1234", frame_counter=17)
    frame = build_frame(p)
    unscrambled_payload = scramble_at(frame[8:], 8)
    raw = frame[:8] + unscrambled_payload
    (fc,) = struct.unpack_from("<H", raw, 0x03B)
    assert fc == 17
    assert raw[0x03D:0x045] == b"ABCD1234"


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
