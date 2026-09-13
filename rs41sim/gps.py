"""Conversions position geodesique WGS84 <-> ECEF, et temps GPS (semaine/TOW)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)

GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
LEAP_SECONDS = 18  # ecart GPS-UTC courant (2017-)


@dataclass
class ECEFState:
    x_cm: int
    y_cm: int
    z_cm: int
    vx_cms: int
    vy_cms: int
    vz_cms: int


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> tuple[float, float, float]:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    n = WGS84_A / math.sqrt(1 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + alt_m) * cos_lat * cos_lon
    y = (n + alt_m) * cos_lat * sin_lon
    z = (n * (1 - WGS84_E2) + alt_m) * sin_lat
    return x, y, z


def velocity_enu_to_ecef(lat_deg: float, lon_deg: float, v_east: float, v_north: float, v_up: float) -> tuple[float, float, float]:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    vx = -sin_lon * v_east - sin_lat * cos_lon * v_north + cos_lat * cos_lon * v_up
    vy = cos_lon * v_east - sin_lat * sin_lon * v_north + cos_lat * sin_lon * v_up
    vz = cos_lat * v_north + sin_lat * v_up
    return vx, vy, vz


def make_ecef_state(lat_deg: float, lon_deg: float, alt_m: float,
                     v_east: float, v_north: float, v_up: float) -> ECEFState:
    x, y, z = geodetic_to_ecef(lat_deg, lon_deg, alt_m)
    vx, vy, vz = velocity_enu_to_ecef(lat_deg, lon_deg, v_east, v_north, v_up)
    return ECEFState(
        x_cm=round(x * 100), y_cm=round(y * 100), z_cm=round(z * 100),
        vx_cms=round(vx * 100), vy_cms=round(vy * 100), vz_cms=round(vz * 100),
    )


def gps_week_tow(dt: datetime | None = None) -> tuple[int, int]:
    """Retourne (semaine_gps, temps_dans_la_semaine_en_ms) pour l'instant `dt` (UTC)."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    gps_time = dt + timedelta(seconds=LEAP_SECONDS)
    delta = gps_time - GPS_EPOCH
    week = delta.days // 7
    tow_seconds = (delta - timedelta(weeks=week)).total_seconds()
    tow_ms = round(tow_seconds * 1000)
    return week, tow_ms
