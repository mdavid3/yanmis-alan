import os
import json
import glob
import csv
import datetime as dt
import ee

SERVICE_ACCOUNT = os.environ.get("GEE_SERVICE_ACCOUNT")
KEY_FILE = os.environ.get("GEE_KEY_FILE", "gee-key.json")

IN_DIR = "candidates_filtered"
OUT_DIR = "candidates_final"
REPORT = "brightness_report.csv"
DETAIL = "brightness_detail.csv"
DONE_FILE = "brightness_done.json"

BANDS = ["B4", "B8", "B12"]
PRE_START, PRE_END = 35, 5
POST_START, POST_END = 2, 8
BRIGHTEN_LIMIT = 0.05
CHUNK = 400


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


def brightness(region, start, end):
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
        .map(mask_clouds)
    )
    comp = col.median().select(BANDS)
    return comp.reduce(ee.Reducer.mean()).rename("bright")


def polygon_values(img, feats):
    out = {}
    for start in range(0, len(feats), CHUNK):
        batch = feats[start:start + CHUNK]
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry(f["geometry"]), {"fid": i + start})
            for i, f in enumerate(batch)
        ])
        reduced = img.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=20)
        info = reduced.getInfo()
        for feat in info["features"]:
            p = feat["properties"]
            out[p["fid"]] = p.get("mean")
    return out


def main():
    init()
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    detail = []

    files = sorted(glob.glob(os.path.join(IN_DIR, "*_filtered.geojson")))
    done = {}
    if os.path.exists(DONE_FILE):
        with open(DONE_FILE, "r", encoding="utf-8") as fh:
            done = json.load(fh)
    by_prefix = {}
    for path in files:
        b = os.path.basename(path)[:-17]
        p_aoi, p_date = b.rsplit("_", 1)
        by_prefix.setdefault(p_aoi, []).append(p_date)
    tails = {k: set(sorted(v)[-2:]) for k, v in by_prefix.items()}

    for path in files:
        base = os.path.basename(path)[:-17]
        aoi, date_str = base.rsplit("_", 1)
        if date_str in done.get(aoi, []) and date_str not in tails.get(aoi, set()):
            continue
        d = dt.date.fromisoformat(date_str)

        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        feats = data.get("features", [])
        if not feats:
            continue

        region = ee.FeatureCollection([
            ee.Feature(ee.Geometry(f["geometry"])) for f in feats
        ]).geometry().bounds().buffer(1000)

        pre_img = brightness(region, (d - dt.timedelta(days=PRE_START)).isoformat(),
                             (d - dt.timedelta(days=PRE_END)).isoformat())
        post_img = brightness(region, (d - dt.timedelta(days=POST_START)).isoformat(),
                              (d + dt.timedelta(days=POST_END)).isoformat())

        pre_vals = polygon_values(pre_img, feats)
        post_vals = polygon_values(post_img, feats)

        survivors = []
        elim = 0
        for i, f in enumerate(feats):
            pre = pre_vals.get(i)
            post = post_vals.get(i)
            if pre is None or post is None or pre <= 0:
                delta = None
                keep = True
            else:
                delta = (post - pre) / pre
                keep = delta <= BRIGHTEN_LIMIT

            detail.append([
                aoi, date_str, i,
                round(f["properties"].get("area_ha", 0), 1),
                f["properties"].get("mean_dnbr"),
                round(pre, 1) if pre else "",
                round(post, 1) if post else "",
                round(delta, 3) if delta is not None else "",
                "keep" if keep else "eliminate",
            ])

            if keep:
                f["properties"]["brightness_delta"] = round(delta, 3) if delta is not None else None
                survivors.append(f)
            else:
                elim += 1

        total_ha = sum(x["properties"].get("area_ha", 0) for x in survivors)
        rows.append([aoi, date_str, len(feats), elim, len(survivors), round(total_ha, 1)])
        print(aoi, date_str, len(feats), "->", len(survivors), flush=True)

        seen = set(done.get(aoi, []))
        seen.add(date_str)
        done[aoi] = sorted(seen)
        with open(DONE_FILE, "w", encoding="utf-8") as fh:
            json.dump(done, fh)

        if survivors:
            out = {
                "type": "FeatureCollection",
                "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
                "features": survivors,
            }
            name = f"{aoi}_{date_str}_final.geojson"
            with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as fh:
                json.dump(out, fh)

    with open(REPORT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["aoi", "date", "in", "elim_brightening", "out", "out_ha"])
        w.writerows(rows)

    with open(DETAIL, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["aoi", "date", "fid", "area_ha", "mean_dnbr", "pre_bright", "post_bright", "delta", "decision"])
        w.writerows(detail)


if __name__ == "__main__":
    main()
