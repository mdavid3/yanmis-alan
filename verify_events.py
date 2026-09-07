import os
import json
import csv
import datetime as dt
import urllib.request
import ee

SERVICE_ACCOUNT = os.environ.get("GEE_SERVICE_ACCOUNT")
KEY_FILE = os.environ.get("GEE_KEY_FILE", "gee-key.json")

EVENTS_FILE = "events.geojson"
OUT_DIR = "verify_events"
STATUSES = {"confirmed", "pending"}
BUFFER_M = 1000
DIM = 640
RECOVERY_WAIT = 25
RECOVERY_WINDOW = 20


def init():
    credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, KEY_FILE)
    ee.Initialize(credentials)


def mask_clouds(img):
    scl = img.select("SCL")
    valid = (
        scl.neq(3).And(scl.neq(8)).And(scl.neq(9))
        .And(scl.neq(10)).And(scl.neq(11))
        .And(scl.neq(0)).And(scl.neq(1))
    )
    return img.updateMask(valid)


def composite(region, start, end):
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
        .map(mask_clouds)
    )
    return col.median()


def chip(img, geom, region, path):
    vis = img.visualize(bands=["B12", "B8", "B4"], min=200, max=4500)
    outline = (
        ee.Image().byte()
        .paint(ee.FeatureCollection([ee.Feature(geom)]), 1, 3)
        .visualize(palette=["ff0000"])
    )
    url = vis.blend(outline).getThumbURL({"region": region, "dimensions": DIM, "format": "png"})
    urllib.request.urlretrieve(url, path)


def kml_geom(geometry):
    def ring(coords):
        pts = " ".join(f"{x},{y},0" for x, y in coords)
        return f"<outerBoundaryIs><LinearRing><coordinates>{pts}</coordinates></LinearRing></outerBoundaryIs>"

    if geometry["type"] == "Polygon":
        return "<Polygon><tessellate>1</tessellate>" + ring(geometry["coordinates"][0]) + "</Polygon>"
    parts = []
    for poly in geometry["coordinates"]:
        parts.append("<Polygon><tessellate>1</tessellate>" + ring(poly[0]) + "</Polygon>")
    return "<MultiGeometry>" + "".join(parts) + "</MultiGeometry>"


def main():
    init()
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(EVENTS_FILE, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    rows = []
    kml_parts = []

    for f in data.get("features", []):
        p = f["properties"]
        if p.get("status") not in STATUSES:
            continue

        ev_id = p["event_id"]
        first = dt.date.fromisoformat(p["first_date"])
        last = dt.date.fromisoformat(p["last_date"])
        geom = ee.Geometry(f["geometry"])
        region = geom.buffer(BUFFER_M).bounds()

        windows = {
            "pre": ((first - dt.timedelta(days=35)).isoformat(), (first - dt.timedelta(days=5)).isoformat()),
            "post": ((first - dt.timedelta(days=2)).isoformat(), (first + dt.timedelta(days=8)).isoformat()),
            "late": ((last + dt.timedelta(days=RECOVERY_WAIT)).isoformat(),
                     (last + dt.timedelta(days=RECOVERY_WAIT + RECOVERY_WINDOW)).isoformat()),
        }

        status = "ok"
        for label, (s, e) in windows.items():
            path = os.path.join(OUT_DIR, f"{ev_id}_{label}.png")
            try:
                chip(composite(region, s, e), geom, region, path)
            except Exception as ex:
                status = f"{label} failed: {ex}"

        centroid = geom.centroid(10).coordinates().getInfo()
        rows.append([
            ev_id, p["first_date"], p["last_date"], p["status"],
            p.get("max_scene_ha"), p.get("mean_dnbr"), p.get("recovery_delta"),
            round(centroid[1], 5), round(centroid[0], 5), status,
        ])
        kml_parts.append(
            f"<Placemark><name>{ev_id}</name>"
            f"<description>{p['first_date']} -> {p['last_date']} | {p.get('max_scene_ha')} ha | "
            f"dnbr={p.get('mean_dnbr')} | {p['status']}</description>"
            f"<Style><LineStyle><color>ff0000ff</color><width>3</width></LineStyle>"
            f"<PolyStyle><fill>0</fill></PolyStyle></Style>{kml_geom(f['geometry'])}</Placemark>"
        )
        print(ev_id, status)

    kml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
        + "".join(kml_parts) + "</Document></kml>"
    )
    with open(os.path.join(OUT_DIR, "events.kml"), "w", encoding="utf-8") as fh:
        fh.write(kml)

    with open(os.path.join(OUT_DIR, "verify_index.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["event_id", "first_date", "last_date", "status", "max_scene_ha",
                    "mean_dnbr", "recovery_delta", "lat", "lon", "chip_status"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
