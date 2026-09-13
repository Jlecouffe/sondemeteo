"""Verification independante : simule ce que fait LocalSondeDecoder.kt
(radtel-tools) sur le signal produit par rs41sim, pour confirmer que la
trame corrigee se decode correctement de bout en bout (RF -> FM -> bits ->
octets -> champs), sans dependre du build Android."""
import math
import sys

import numpy as np

from rs41sim.frame import SondeParams, build_frame
from rs41sim.modulate import gfsk_modulate
from rs41sim.scrambling import MASK

# 32 kHz = debit reel du pont audio Bluetooth RT-660 (Rt660AudioFraming.SAMPLE_RATE
# cote radtel-tools), pas 44.1 kHz : c'est ce qui compte pour ce test, un signal
# qui decode bien a 44.1 kHz peut echouer a 32 kHz (calage bit moins precis, voir
# app.py / modulate.py pour le choix de BT).
SAMPLE_RATE = 32000.0

RS41_HEADER_BITS = "0000100001101101010100111000100001000100011010010100100000011111"
RS41_HEADER = [int(c) for c in RS41_HEADER_BITS]


def fm_discriminate(iq: np.ndarray) -> np.ndarray:
    """Discriminateur FM simple : derivee de phase -> signal audio (comme la sortie demodulee d'un recepteur)."""
    phase = np.angle(iq)
    dphase = np.diff(phase)
    dphase = (dphase + np.pi) % (2 * np.pi) - np.pi
    audio = dphase * (32000.0 / np.max(np.abs(dphase)))
    return audio.astype(np.int16)


def zero_crossing_bits(samples: np.ndarray, baud: float, sample_rate: float):
    spb = sample_rate / baud
    out = []
    prev = 1 if samples[0] >= 0 else 0
    run = 1
    prev_sample = float(samples[0])
    prev_frac = 0.0
    for i in range(1, len(samples)):
        cur = 1 if samples[i] >= 0 else 0
        if cur == prev:
            run += 1
        else:
            a = prev_sample
            b = float(samples[i])
            frac = (b / (b - a)) if b != a else 0.0
            frac = max(-1.0, min(1.0, frac))
            bit_len = round((run + prev_frac - frac) / spb)
            bit_len = max(1, min(64, bit_len))
            out.extend([prev] * bit_len)
            prev_frac = frac
            prev = cur
            run = 1
        prev_sample = float(samples[i])
    last_len = max(1, min(64, round(run / spb)))
    out.extend([prev] * last_len)
    return out


def find_header(bits, header, max_errors=6):
    best_start, best_err, inverted = -1, 10 ** 9, False
    for inv in (0, 1):
        n = len(bits) - len(header)
        for i in range(n + 1):
            err = 0
            for j, h in enumerate(header):
                b = bits[i + j] if inv == 0 else 1 - bits[i + j]
                if b != h:
                    err += 1
                    if err >= best_err or err > 6:
                        break
            if err < best_err:
                best_err, best_start, inverted = err, i, (inv == 1)
                if err == 0:
                    break
        if best_err == 0:
            break
    if best_start < 0 or best_err > max_errors:
        return None
    return best_start, best_err, inverted


def bits_to_frame_bytes(bits, start, inverted, nbytes=320):
    raw = bytearray(nbytes)
    for byte_pos in range(nbytes):
        v = 0
        for b in range(8):
            bit = bits[start + byte_pos * 8 + b]
            if inverted:
                bit = 1 - bit
            if bit:
                v |= (1 << b)
        raw[byte_pos] = v
    return bytes(raw)


def ecef_to_geodetic(x, y, z):
    a = 6378137.0
    b = 6356752.31424518
    a2b2 = a * a - b * b
    e2 = a2b2 / (a * a)
    ep2 = a2b2 / (b * b)
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    if p < 1.0:
        return None
    t = math.atan2(z * a, p * b)
    lat = math.atan2(z + ep2 * b * math.sin(t) ** 3, p - e2 * a * math.cos(t) ** 3)
    r = a / math.sqrt(1.0 - e2 * math.sin(lat) ** 2)
    alt = p / math.cos(lat) - r
    return math.degrees(lat), math.degrees(lon), alt


def i32le(b, p):
    return int.from_bytes(b[p:p + 4], "little", signed=True)


def i16le(b, p):
    return int.from_bytes(b[p:p + 2], "little", signed=True)


def decode_rs41(pcm: np.ndarray):
    bits = zero_crossing_bits(pcm, 4800.0, SAMPLE_RATE)
    found = find_header(bits, RS41_HEADER)
    if not found:
        return None, "sync entete introuvable"
    start, err, inverted = found
    raw = bits_to_frame_bytes(bits, start, inverted)
    frame = bytes(raw[i] ^ MASK[i % len(MASK)] for i in range(len(raw)))

    if frame[0x039] != 0x79:
        return None, f"marqueur FRAME absent (lu 0x{frame[0x039]:02X}), err_sync={err}"
    if frame[0x112] != 0x7B:
        return None, f"marqueur GPS3 absent (lu 0x{frame[0x112]:02X}), err_sync={err}"

    frame_no = int.from_bytes(frame[0x03B:0x03D], "little")
    serial = frame[0x03D:0x045].decode("ascii", "replace").strip()

    x = i32le(frame, 0x114) / 100.0
    y = i32le(frame, 0x118) / 100.0
    z = i32le(frame, 0x11C) / 100.0
    llh = ecef_to_geodetic(x, y, z)
    sats = frame[0x126]

    return {
        "frame_no": frame_no, "serial": serial,
        "lat": llh[0], "lon": llh[1], "alt": llh[2], "sats": sats,
        "sync_errors": err,
    }, None


def main():
    p = SondeParams(serial="TESTSN01", frame_counter=42,
                     latitude_deg=48.85341, longitude_deg=2.3488, altitude_m=8542.0,
                     climb_rate_ms=5.2, horizontal_speed_ms=12.0, heading_deg=90.0,
                     num_sats=11)
    frame_bytes = build_frame(p)
    iq = gfsk_modulate(frame_bytes, sample_rate=SAMPLE_RATE)  # bt par defaut du module
    pcm = fm_discriminate(iq)

    result, error = decode_rs41(pcm)
    if error:
        print("ECHEC:", error)
        sys.exit(1)
    print("OK, trame decodee :", result)
    ok = (result["serial"] == p.serial and result["frame_no"] == p.frame_counter
          and abs(result["lat"] - p.latitude_deg) < 0.01
          and abs(result["lon"] - p.longitude_deg) < 0.01
          and abs(result["alt"] - p.altitude_m) < 5.0
          and result["sats"] == p.num_sats)
    print("Valeurs conformes aux parametres d'entree :", ok)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
