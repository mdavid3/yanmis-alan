import os
import sys
import json
import datetime as dt
import ee
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

SERVICE_ACCOUNT = os.environ.get("GEE_SERVICE_ACCOUNT")
KEY_FILE = os.environ.get("GEE_KEY_FILE", "gee-key.json")

BELT = [
    [26.0, 40.2], [26.0, 36.1], [30.5, 35.9], [36.4, 35.9], [36.6, 37.1],
    [34.0, 37.6], [32.6, 38.2], [30.8, 38.8], [28.8, 39.8], [27.0, 40.4],
]

TILES_FILE = "tiles.json"
OUT_DIR = "candidates"
EXCLUSION_FILE = "exclusion.geojson"

SCAN_SCALE = 60
DETAIL_SCALE = 20
CLOUD_LIMIT = 30
REF_WINDOW_DAYS = 30
REF_WINDOW_MAX_DAYS = 60
REF_WINDOW_STEP = 15
MIN_REF_SCENES = 3
DNBR_THRESHOLD = 0.15
DNBR2_THRESHOLD = 0.05
MIN_AREA_HA = 3.0
VALID_MIN = 0.4
WORKERS = 6

write_lock = Lock()


def init():
    credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, KEY_FILE)
    ee.Initialize(credentials)
    ee.data.setDeadline(180000)


def mask_clouds(img):
    scl = img.select("SCL")
    valid = (
        scl.neq(3).And(scl.neq(8)).And(scl.neq(9))
        .And(scl.neq(10)).And(scl.neq(11))
        .And(scl.neq(0)).And(scl.neq(1))
    )
    return img.updateMask(valid).divide(10000).copyProperties(img, ["system:time_start"])


def add_indices(img):
    nbr = img.normalizedDifference(["B8", "B12"]).rename("NBR")
    nbr2 = img.normalizedDifference(["B11", "B12"]).rename("NBR2")
    return img.addBands([nbr, nbr2])


def tile_collection(tile, start, end):
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filter(ee.Filter.eq("MGRS_TILE", tile))
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_LIMIT))
        .map(mask_clouds)
        .map(add_indices)
    )


def discover_tiles():
    aoi = ee.Geometry.Polygon([BELT])
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate("2025-06-01", "2025-09-01")
    )
    tiles = sorted(col.aggregate_array("MGRS_TILE").distinct().getInfo())
    with open(TILES_FILE, "w", encoding="utf-8") as fh:
        json.dump(tiles, fh, indent=2)
    print("karo sayisi:", len(tiles), flush=True)
    return tiles


def load_tiles():
    if not os.path.exists(TILES_FILE):
        return discover_tiles()
    with open(TILES_FILE, "r", encoding="utf-8") as fh:
        return json.load(fh)


def water_mask():
    wc = ee.Image("ESA/WorldCover/v200/2021").select("Map")
    return wc.neq(80).And(wc.neq(50))


def exclusion_mask(scene_date):
    if not os.path.exists(EXCLUSION_FILE):
        return None
    with open(EXCLUSION_FILE, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    feats = [
        f for f in data.get("features", [])
        if f["properties"].get("exclude_until", "") >= scene_date.isoformat()
    ]
    if not feats:
        return None
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry(f["geometry"])) for f in feats])
    return ee.Image().byte().paint(fc, 1).unmask(0).eq(0)


def scene_dates(tile, start, end):
    col = tile_collection(tile, start, end)
    stamps = col.aggregate_array("system:time_start").getInfo()
    return sorted({dt.datetime.utcfromtimestamp(s / 1000).date() for s in stamps})


def reference_composite(tile, target_date):
    window = REF_WINDOW_DAYS
    while window <= REF_WINDOW_MAX_DAYS:
        ref_start = (target_date - dt.timedelta(days=window)).isoformat()
        ref_end = (target_date - dt.timedelta(days=1)).isoformat()
        col = tile_collection(tile, ref_start, ref_end)
        n = col.size().getInfo()
        if n >= MIN_REF_SCENES:
            return col.select(["NBR", "NBR2"]).median(), n, window
        window += REF_WINDOW_STEP
    return None, 0, window


def process_scene(tile, d):
    day_col = tile_collection(tile, d.isoformat(), (d + dt.timedelta(days=1)).isoformat())
    footprint = day_col.first().geometry()
    target = day_col.select(["NBR", "NBR2"]).median()

    valid = target.select("NBR").mask().reduceRegion(
        reducer=ee.Reducer.mean(), geometry=footprint, scale=300,
        maxPixels=1e9, bestEffort=True,
    ).get("NBR").getInfo() or 0.0
    if valid < VALID_MIN:
        return {"date": d.isoformat(), "status": "low_valid", "valid": round(valid, 2)}

    reference, n_ref, window = reference_composite(tile, d)
    if reference is None:
        return {"date": d.isoformat(), "status": "insufficient_reference"}

    delta = reference.subtract(target)
    hit = (
        delta.select("NBR").gt(DNBR_THRESHOLD)
        .And(delta.select("NBR2").gt(DNBR2_THRESHOLD))
        .And(water_mask())
        .rename("burn")
    )
    excl = exclusion_mask(d)
    if excl is not None:
        hit = hit.And(excl)

    cleaned = hit.selfMask()
    vectors = cleaned.reduceToVectors(
        geometry=footprint, scale=SCAN_SCALE, geometryType="polygon",
        eightConnected=True, maxPixels=1e10, bestEffort=True,
    )

    def with_area(f):
        area_ha = f.geometry().area(maxError=30).divide(10000)
        mean_dnbr = delta.select("NBR").reduceRegion(
            reducer=ee.Reducer.mean(), geometry=f.geometry(),
            scale=DETAIL_SCALE, maxPixels=1e9, bestEffort=True,
        ).get("NBR")
        return f.set({"area_ha": area_ha, "mean_dnbr": mean_dnbr})

    fc = vectors.map(with_area).filter(ee.Filter.gte("area_ha", MIN_AREA_HA))
    geo = fc.getInfo()

    count = len(geo.get("features", []))
    total_ha = sum(f["properties"]["area_ha"] for f in geo.get("features", []))
    result = {
        "date": d.isoformat(), "status": "ok", "valid": round(valid, 2),
        "ref_scenes": n_ref, "ref_window_days": window,
        "candidates": count, "total_ha": round(total_ha, 1),
    }
    if count:
        result["geojson"] = geo
    return result


def run_tile(tile, start, end):
    log_path = os.path.join(OUT_DIR, f"{tile}_log.json")
    log = []
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as fh:
            log = json.load(fh)
    done = {x["date"] for x in log if x.get("status") in ("ok", "low_valid", "insufficient_reference")}

    try:
        dates = scene_dates(tile, start, end)
    except Exception as e:
        return tile, [{"date": start, "status": "date_query_failed", "error": str(e)}]

    new_entries = []
    for d in dates:
        if d.isoformat() in done:
            continue
        try:
            result = process_scene(tile, d)
        except Exception as e:
            result = {"date": d.isoformat(), "status": "failed", "error": str(e)[:200]}
        geo = result.pop("geojson", None)
        with write_lock:
            if geo:
                path = os.path.join(OUT_DIR, f"{tile}_{d.isoformat()}.geojson")
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(geo, fh)
            log.append(result)
            with open(log_path, "w", encoding="utf-8") as fh:
                json.dump(log, fh, indent=2)
        new_entries.append(result)
        print(tile, result["date"], result["status"],
              result.get("candidates", ""), result.get("total_ha", ""), flush=True)
    return tile, new_entries


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "discover":
        init()
        discover_tiles()
        return

    if len(sys.argv) >= 3:
        start, end = sys.argv[1], sys.argv[2]
    else:
        today = dt.date.today()
        start = (today - dt.timedelta(days=10)).isoformat()
        end = today.isoformat()

    init()
    os.makedirs(OUT_DIR, exist_ok=True)
    tiles = load_tiles()
    print("islenecek karo:", len(tiles), "aralik:", start, "->", end, flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(run_tile, t, start, end): t for t in tiles}
        for fut in as_completed(futures):
            tile, entries = fut.result()
            ok = sum(1 for e in entries if e["status"] == "ok")
            print("bitti:", tile, len(entries), "sahne,", ok, "islendi", flush=True)


if __name__ == "__main__":
    main()
