import re
from typing import Tuple
import math

EARTH_RADIUS = 6371000
DMS_REGEX = re.compile(
    r"""
    (?P<sign>[-+]?)
    \s*
    (?P<deg>\d+(?:\.\d+)?)
    [°\s]+
    (?P<min>\d+(?:\.\d+)?)
    [′'\s]+
    (?P<sec>\d+(?:\.\d+)?)
    [″"\s]*
    (?P<hem>[NSEW]?)
    """,
    re.IGNORECASE | re.VERBOSE
)


def dms2dd(text: str) -> float:
    m = DMS_REGEX.fullmatch(text.strip())
    if not m:
        raise ValueError(f"Cannot parse DMS string: {text}")

    deg = float(m["deg"])
    min_ = float(m["min"])
    sec = float(m["sec"])
    hem = (m["hem"] or "").upper()

    dd = deg + min_ / 60.0 + sec / 3600.0

    if m["sign"] == "-" or hem in ("S", "W"):
        dd = -dd
    return dd


if __name__ == "__main__":
    samples = [
        "39°54'45\"N",
        "116°18'11\"E",
        "39°56'06\"N",
        "116°20'53\"E",
    ]
    for s in samples:
        print(f"{s:20} -> {dms2dd(s)}")

    lat1, lon1 = 39.9125, 116.30305555555556
    lat2, lon2 = 39.934999999999995, 116.34805555555555

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    delta_lat_deg = lat2 - lat1
    lat_diff_meters = delta_lat_deg * 111320

    delta_lon_deg = lon2 - lon1
    lon_diff_meters = delta_lon_deg * 111320 * math.cos(lat1_rad)

    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance_meters = EARTH_RADIUS * c

    print(f"Lat diff: {delta_lat_deg:.6f}° ~ {lat_diff_meters:.2f} m")
    print(f"Lon diff: {delta_lon_deg:.6f}° ~ {lon_diff_meters:.2f} m")
    print(lat_diff_meters / 23625)
    print(lon_diff_meters / 47244)
