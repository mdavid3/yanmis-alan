import os
import json
import datetime as dt
import ee

SERVICE_ACCOUNT = os.environ.get("GEE_SERVICE_ACCOUNT")
KEY_FILE = os.environ.get("GEE_KEY_FILE", "gee-key.json")

AOIS = {
    "mugla": {
        "geom": [27.90, 36.70, 28.40, 37.05],
        "start": "2021-07-15",
        "end": "2021-09-15",
    },
    "konya": {
        "geom": [32.30, 37.60, 32.90, 38.05],
        "start": "2021-06-15",
        "end": "2021-08-15",
    },
}

SCALE = 20
CLOUD_LIMIT = 20
REF_WINDOW_DAYS = 30
REF_WINDOW_MAX_DAYS = 60
REF_WINDOW_STEP = 15
MIN_REF_SCENES = 3
DNBR_THRESHOLD = 0.15
DNBR2_THRESHOLD = 0.05
MIN_AREA_HA = 3.0
OUT_DIR = "candidates"


def init():
    credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, KEY_FILE)
    ee.Initialize(credentials)


def mask_clouds(img):
    scl = img.select("SCL")
    valid = (
        scl.neq(3)
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
        .And(scl.neq(0))
        .And(scl.neq(1))
    )
    return img.updateMask(valid).divide(10000).copyProperties(img, ["system:time_start"])


def add_indices(img):
    nbr = img.normalizedDifference(["B8", "B12"]).rename("NBR")
    nbr2 = img.normalizedDifference(["B11", "B12"]).rename("NBR2")
    return img.addBands([nbr, nbr2])


def collection(aoi, start, end):
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_LIMIT))
        .map(mask_clouds)
        .map(add_indices)
    )


def water_mask(aoi):
    wc = ee.Image("ESA/WorldCover/v200/2021").select("Map").clip(aoi)
    return wc.neq(80).And(wc.neq(50))


def scene_dates(aoi, start, end):
    col = collection(aoi, start, end)
    stamps = col.aggregate_array("system:time_start").getInfo()
    dates = sorted({dt.datetime.utcfromtimestamp(s / 1000).date() for s in stamps})
    return dates


def reference_composite(aoi, target_date):
    window = REF_WINDOW_DAYS
    while window <= REF_WINDOW_MAX_DAYS:
        ref_start = (target_date - dt.timedelta(days=window)).isoformat()
        ref_end = (target_date - dt.timedelta(days=1)).isoformat()
        col = collection(aoi, ref_start, ref_end)
        n = col.size().getInfo()
        if n >= MIN_REF_SCENES:
            return col.select(["NBR", "NBR2"]).median(), n, window
        window += REF_WINDOW_STEP
    return None, 0, window


def target_image(aoi, target_date):
    start = target_date.isoformat()
    end = (target_date + dt.timedelta(days=1)).isoformat()
    col = collection(aoi, start, end)
    return col.select(["NBR", "NBR2"]).median()


def candidates(aoi, reference, target):
    delta = reference.subtract(target)
    hit = (
        delta.select("NBR")
        .gt(DNBR_THRESHOLD)
        .And(delta.select("NBR2").gt(DNBR2_THRESHOLD))
        .And(water_mask(aoi))
        .rename("burn")
    )
    cleaned = (
        hit.focal_min(radius=1, units="pixels")
        .focal_max(radius=1, units="pixels")
        .selfMask()
    )
    vectors = cleaned.reduceToVectors(
        geometry=aoi,
        scale=SCALE,
        geometryType="polygon",
        eightConnected=True,
        maxPixels=1e10,
        bestEffort=True,
    )

    def with_area(f):
        area_ha = f.geometry().area(maxError=10).divide(10000)
        mean_dnbr = delta.select("NBR").reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=f.geometry(),
            scale=SCALE,
            maxPixels=1e9,
            bestEffort=True,
        ).get("NBR")
        return f.set({"area_ha": area_ha, "mean_dnbr": mean_dnbr})

    return vectors.map(with_area).filter(ee.Filter.gte("area_ha", MIN_AREA_HA))


def run_aoi(name, cfg):
    aoi = ee.Geometry.Rectangle(cfg["geom"])
    dates = scene_dates(aoi, cfg["start"], cfg["end"])
    os.makedirs(OUT_DIR, exist_ok=True)
    log = []

    for d in dates:
        reference, n_ref, window = reference_composite(aoi, d)
        if reference is None:
            log.append({"date": d.isoformat(), "status": "insufficient_reference"})
            continue

        target = target_image(aoi, d)
        fc = candidates(aoi, reference, target)

        try:
            geo = fc.getInfo()
        except Exception as e:
            log.append({"date": d.isoformat(), "status": "failed", "error": str(e)})
            continue

        count = len(geo.get("features", []))
        total_ha = sum(f["properties"]["area_ha"] for f in geo.get("features", []))

        if count:
            path = os.path.join(OUT_DIR, f"{name}_{d.isoformat()}.geojson")
            with open(path, "w") as fh:
                json.dump(geo, fh)

        log.append(
            {
                "date": d.isoformat(),
                "status": "ok",
                "ref_scenes": n_ref,
                "ref_window_days": window,
                "candidates": count,
                "total_ha": round(total_ha, 1),
            }
        )
        print(name, d.isoformat(), count, round(total_ha, 1))

    with open(os.path.join(OUT_DIR, f"{name}_log.json"), "w") as fh:
        json.dump(log, fh, indent=2)


def main():
    init()
    for name, cfg in AOIS.items():
        run_aoi(name, cfg)


if __name__ == "__main__":
    main()
