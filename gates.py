import os
import json
import glob
import math
import csv
import datetime as dt
import ee
from shapely.geometry import shape
from shapely.strtree import STRtree

SERVICE_ACCOUNT = os.environ.get("GEE_SERVICE_ACCOUNT")
KEY_FILE = os.environ.get("GEE_KEY_FILE", "gee-key.json")

IN_DIR = "candidates"
OUT_DIR = "candidates_filtered"
REPORT = "gates_report.csv"

CROP_FRAC_LIMIT = 0.5
RECT_LIMIT = 0.80
RECT_AREA_LIMIT = 50.0
BURST_COUNT = 50
BURST_AREA_LIMIT = 20.0
PERSIST_OVERLAP = 0.05
PERSIST_LOOKAHEAD = 2
CHUNK = 400


def init():
    credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, KEY_FILE)
    ee.Initialize(credentials)


def load_scenes(aoi):
    files = sorted(glob.glob(os.path.join(IN_DIR, f"{aoi}_*.geojson")))
    scenes = []
    for path in files:
        date_str = os.path.basename(path)[len(aoi) + 1:-8]
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        feats = []
        for i, f in enumerate(data.get("features", [])):
            geom = shape(f["geometry"])
            if not geom.is_valid:
                geom = geom.buffer(0)
            props = f.get("properties", {})
            feats.append({
                "id": i,
                "geom": geom,
                "area_ha": props.get("area_ha", 0),
                "mean_dnbr": props.get("mean_dnbr"),
                "geojson_geom": f["geometry"],
            })
        scenes.append({"date": dt.date.fromisoformat(date_str), "path": path, "feats": feats})
    return scenes


def local_metric(geom):
    lat = geom.centroid.y
    k = math.cos(math.radians(lat))

    def proj(g):
        from shapely.ops import transform
        return transform(lambda x, y, z=None: (x * k * 111320.0, y * 111320.0), g)

    m = proj(geom)
    area = m.area
    per = m.length
    compact = 4 * math.pi * area / (per * per) if per > 0 else 0
    mrr = m.minimum_rotated_rectangle
    rect = area / mrr.area if mrr.area > 0 else 0
    return compact, rect


def cropland_fractions(feats):
    wc = ee.Image("ESA/WorldCover/v200/2021").select("Map")
    crop = wc.eq(40).rename("crop")
    out = {}
    for start in range(0, len(feats), CHUNK):
        batch = feats[start:start + CHUNK]
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry(f["geojson_geom"]), {"fid": f["id"]}) for f in batch
        ])
        reduced = crop.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.mean(),
            scale=10,
        )
        info = reduced.getInfo()
        for feat in info["features"]:
            p = feat["properties"]
            out[p["fid"]] = p.get("mean", 0) or 0
    return out


def persistence_flags(scenes):
    trees = []
    for s in scenes:
        geoms = [f["geom"] for f in s["feats"]]
        trees.append(STRtree(geoms) if geoms else None)

    for i, s in enumerate(scenes):
        future = []
        for j in range(i + 1, min(i + 1 + PERSIST_LOOKAHEAD, len(scenes))):
            future.append(j)
        for f in s["feats"]:
            if not future:
                f["persistent"] = None
                continue
            hit = False
            for j in future:
                tree = trees[j]
                if tree is None:
                    continue
                cand_idx = tree.query(f["geom"])
                for ci in cand_idx:
                    other = scenes[j]["feats"][int(ci)]["geom"]
                    inter = f["geom"].intersection(other).area
                    if f["geom"].area > 0 and inter / f["geom"].area >= PERSIST_OVERLAP:
                        hit = True
                        break
                if hit:
                    break
            f["persistent"] = hit


def apply_gates(aoi, scenes):
    rows = []
    os.makedirs(OUT_DIR, exist_ok=True)

    for s in scenes:
        feats = s["feats"]
        n = len(feats)
        if n == 0:
            rows.append([aoi, s["date"].isoformat(), 0, 0, 0, 0, 0, 0, 0.0])
            continue

        crop = cropland_fractions(feats)
        burst_scene = n > BURST_COUNT

        elim_lc = elim_geo = elim_burst = elim_trans = 0
        survivors = []

        for f in feats:
            compact, rect = local_metric(f["geom"])
            flags = []
            if crop.get(f["id"], 0) > CROP_FRAC_LIMIT:
                flags.append("landcover")
            if rect > RECT_LIMIT and f["area_ha"] < RECT_AREA_LIMIT:
                flags.append("geometry")
            if burst_scene and f["area_ha"] < BURST_AREA_LIMIT:
                flags.append("burst")
            if f["persistent"] is False:
                flags.append("transient")

            elim_lc += "landcover" in flags
            elim_geo += "geometry" in flags
            elim_burst += "burst" in flags
            elim_trans += "transient" in flags

            if not flags:
                survivors.append({
                    "type": "Feature",
                    "geometry": f["geojson_geom"],
                    "properties": {
                        "area_ha": f["area_ha"],
                        "mean_dnbr": f["mean_dnbr"],
                        "compactness": round(compact, 3),
                        "rectangularity": round(rect, 3),
                        "cropland_frac": round(crop.get(f["id"], 0), 3),
                        "persistent": f["persistent"],
                    },
                })

        total_ha = sum(x["properties"]["area_ha"] for x in survivors)
        rows.append([
            aoi, s["date"].isoformat(), n,
            elim_lc, elim_geo, elim_burst, elim_trans,
            len(survivors), round(total_ha, 1),
        ])

        if survivors:
            out = {
                "type": "FeatureCollection",
                "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
                "features": survivors,
            }
            name = f"{aoi}_{s['date'].isoformat()}_filtered.geojson"
            with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as fh:
                json.dump(out, fh)

        print(aoi, s["date"].isoformat(), n, "->", len(survivors))

    return rows


def main():
    init()
    all_rows = []
    files = glob.glob(os.path.join(IN_DIR, "*.geojson"))
    prefixes = sorted({os.path.basename(p)[:-19] for p in files})
    for aoi in prefixes:
        scenes = load_scenes(aoi)
        if not scenes:
            continue
        persistence_flags(scenes)
        all_rows.extend(apply_gates(aoi, scenes))

    with open(REPORT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "aoi", "date", "candidates",
            "elim_landcover", "elim_geometry", "elim_burst", "elim_transient",
            "survivors", "survivor_ha",
        ])
        w.writerows(all_rows)


if __name__ == "__main__":
    main()
