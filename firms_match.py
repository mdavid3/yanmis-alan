import os
import json
import csv
import datetime as dt
import urllib.request
import ee
from shapely.geometry import shape, Point

SERVICE_ACCOUNT = os.environ.get("GEE_SERVICE_ACCOUNT")
KEY_FILE = os.environ.get("GEE_KEY_FILE", "gee-key.json")
FIRMS_KEY = os.environ.get("FIRMS_MAP_KEY")

EVENTS_FILE = "events.geojson"
REPORT = "firms_report.csv"
VIIRS_SOURCE = "VIIRS_SNPP_SP"
DAY_PAD_BEFORE = 2
DAY_PAD_AFTER = 1
POINT_BUFFER_DEG = 0.008


def init():
    credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, KEY_FILE)
    ee.Initialize(credentials)


def modis_hotspot(geom_geojson, start, end):
    region = ee.Geometry(geom_geojson).buffer(1000)
    col = ee.ImageCollection("FIRMS").filterDate(start, (end + dt.timedelta(days=1)).isoformat() if isinstance(end, dt.date) else end)
    col = ee.ImageCollection("FIRMS").filterDate(start.isoformat(), (end + dt.timedelta(days=1)).isoformat())
    if col.size().getInfo() == 0:
        return 0
    val = col.select("T21").max().reduceRegion(
        reducer=ee.Reducer.max(), geometry=region, scale=1000,
        maxPixels=1e9, bestEffort=True,
    ).get("T21").getInfo()
    return 1 if val is not None else 0


def viirs_points(footprint, bbox, start, end):
    if not FIRMS_KEY:
        return None
    buffered = footprint.buffer(POINT_BUFFER_DEG)
    total = 0
    cursor = start
    while cursor <= end:
        span = min(5, (end - cursor).days + 1)
        url = (
            f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{FIRMS_KEY}/"
            f"{VIIRS_SOURCE}/{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}/{span}/{cursor.isoformat()}"
        )
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                text = resp.read().decode("utf-8")
        except Exception:
            return -1
        lines = text.strip().splitlines()
        if len(lines) > 1:
            header = lines[0].split(",")
            try:
                lat_i = header.index("latitude")
                lon_i = header.index("longitude")
            except ValueError:
                return -1
            for line in lines[1:]:
                parts = line.split(",")
                try:
                    pt = Point(float(parts[lon_i]), float(parts[lat_i]))
                except (ValueError, IndexError):
                    continue
                if buffered.contains(pt):
                    total += 1
        cursor += dt.timedelta(days=span)
    return total


def main():
    init()
    with open(EVENTS_FILE, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    rows = []
    for f in data.get("features", []):
        p = f["properties"]
        ev_id = p["event_id"]
        first = dt.date.fromisoformat(p["first_date"])
        last = dt.date.fromisoformat(p["last_date"])
        start = first - dt.timedelta(days=DAY_PAD_BEFORE)
        end = last + dt.timedelta(days=DAY_PAD_AFTER)

        footprint = shape(f["geometry"])
        if not footprint.is_valid:
            footprint = footprint.buffer(0)
        minx, miny, maxx, maxy = footprint.buffer(POINT_BUFFER_DEG).bounds
        bbox = (round(minx, 4), round(miny, 4), round(maxx, 4), round(maxy, 4))

        modis = modis_hotspot(f["geometry"], start, end)
        viirs = viirs_points(footprint, bbox, start, end)

        rows.append([
            ev_id, p.get("status"), p["first_date"], p["last_date"],
            p.get("max_scene_ha"), modis,
            "no_key" if viirs is None else ("api_error" if viirs == -1 else viirs),
        ])
        print(ev_id, "modis:", modis, "viirs:", rows[-1][-1])

    with open(REPORT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["event_id", "status", "first_date", "last_date",
                    "max_scene_ha", "modis_hotspot", "viirs_points"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
