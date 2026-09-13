"""Avancement simple de la trajectoire simulee entre deux trames."""
from __future__ import annotations

import math

from .frame import SondeParams

EARTH_RADIUS_M = 6371000.0


def advance(p: SondeParams, dt_seconds: float) -> None:
    """Met a jour altitude/latitude/longitude en place selon vitesse/cap."""
    if dt_seconds <= 0:
        return
    p.altitude_m += p.climb_rate_ms * dt_seconds
    if p.horizontal_speed_ms:
        heading_rad = math.radians(p.heading_deg)
        v_north = p.horizontal_speed_ms * math.cos(heading_rad)
        v_east = p.horizontal_speed_ms * math.sin(heading_rad)
        dlat = (v_north * dt_seconds) / EARTH_RADIUS_M
        dlon = (v_east * dt_seconds) / (EARTH_RADIUS_M * math.cos(math.radians(p.latitude_deg)))
        p.latitude_deg += math.degrees(dlat)
        p.longitude_deg += math.degrees(dlon)
