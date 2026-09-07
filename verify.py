import os
import json
import glob
import csv
import datetime as dt
import urllib.request
import ee

SERVICE_ACCOUNT = os.environ.get("GEE_SERVICE_ACCOUNT")
KEY_FILE = os.environ.get("GEE_KEY_FILE", "gee-key.json")

IN_DIR = "candidates_filtered"
OUT_DIR = "verify"
AOI = "konya"
BUFFER_M = 800
DIM = 640


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


def chip_url(img, feat_geom, region):
    vis = img.visualize(bands=["B12", "B8", "B4"], min=200, max=4500)
    outline = (
        ee.Image().byte()
        .paint(ee.FeatureCollection([ee.Feature(feat_geom)]), 1, 3)
        .visualize(palette=["ff0000"])
    )
    blended = vis.blend(outline)
    return blended.getThumbURL({"region": region, "dimensions": DIM, "format": "png"})


def kml_polygon(coords):
    rings = []
    outer = coords[0]
    pts = " ".join(f"{x},{y},0" for x, y in outer)
    rings.append(
        f"<outerBoundaryIs><LinearRing><coordinates>{pts}</coordinates></LinearRing></outerBoundaryIs>"
    )
    return "<Polygon><tessellate>1</tessellate>" + "".join(rings) + "</Polygon>"


def main():
    init()
    os.makedirs(OUT_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(IN_DIR, f"{AOI}_*_filtered.geojson")))
    rows = []
    kml_parts = []

    for path in files:
        date_str = os.path.basename(path)[len(AOI) + 1:-17]
        d = dt.date.fromisoformat(date_str)
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        for i, f in enumerate(data.get("features", [])):
            props = f["properties"]
            geom = ee.Geometry(f["geometry"])
            region = geom.buffer(BUFFER_M).bounds()

            pre = composite(region, (d - dt.timedelta(days=35)).isoformat(), (d - dt.timedelta(days=5)).isoformat())
            post = composite(region, (d - dt.timedelta(days=2)).isoformat(), (d + dt.timedelta(days=8)).isoformat())

            tag = f"{AOI}_{date_str}_p{i:02d}"
            try:
                pre_url = chip_url(pre, geom, region)
                post_url = chip_url(post, geom, region)
                urllib.request.urlretrieve(pre_url, os.path.join(OUT_DIR, f"{tag}_pre.png"))
                urllib.request.urlretrieve(post_url, os.path.join(OUT_DIR, f"{tag}_post.png"))
                status = "ok"
            except Exception as e:
                status = f"failed: {e}"

            c = f["geometry"]["coordinates"]
            centroid = ee.Geometry(f["geometry"]).centroid(10).coordinates().getInfo()
            rows.append([
                tag, date_str, props.get("area_ha"), props.get("mean_dnbr"),
                round(centroid[1], 5), round(centroid[0], 5),
                props.get("cropland_frac"), props.get("rectangularity"), status,
            ])

            if f["geometry"]["type"] == "Polygon":
                poly = kml_polygon(c)
            else:
                poly = "".join(kml_polygon(p) for p in c)
            kml_parts.append(
                f"<Placemark><name>{tag}</name>"
                f"<description>date={date_str} area={props.get('area_ha')} dnbr={props.get('mean_dnbr')}</description>"
                f"<Style><LineStyle><color>ff0000ff</color><width>3</width></LineStyle>"
                f"<PolyStyle><fill>0</fill></PolyStyle></Style>{poly}</Placemark>"
            )
            print(tag, status)

    kml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
        + "".join(kml_parts) + "</Document></kml>"
    )
    with open(os.path.join(OUT_DIR, f"{AOI}_survivors.kml"), "w", encoding="utf-8") as fh:
        fh.write(kml)

    with open(os.path.join(OUT_DIR, "verify_index.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["tag", "date", "area_ha", "mean_dnbr", "lat", "lon", "cropland_frac", "rectangularity", "status"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
