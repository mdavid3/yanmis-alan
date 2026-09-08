import os
import json
import glob
import csv
import datetime as dt
import urllib.request
import ee
from shapely.geometry import shape, mapping, Point
from shapely.ops import unary_union

SERVICE_ACCOUNT = os.environ.get("GEE_SERVICE_ACCOUNT")
KEY_FILE = os.environ.get("GEE_KEY_FILE", "gee-key.json")

IN_DIR = "candidates_final"
OUT_EVENTS = "events.geojson"
OUT_CSV = "events.csv"
OUT_EXCLUSION = "exclusion.geojson"

MATCH_OVERLAP = 0.05
CLOSE_DAYS = 21
CONFIRM_SCENES = 2
AUTO_CONFIRM_HA = 100.0
RECOVERY_WAIT = 25
RECOVERY_WINDOW = 20
RECOVERY_MARGIN = 0.05
EXCLUDE_DAYS = 45
FIRMS_KEY = os.environ.get("FIRMS_MAP_KEY")
VIIRS_SOURCE = "VIIRS_SNPP_SP"
CROP_LIMIT = 0.35
POINT_BUFFER_DEG = 0.008


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


def nbr_mean(geom_geojson, start, end):
    region = ee.Geometry(geom_geojson)
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
        .map(mask_clouds)
    )
    if col.size().getInfo() == 0:
        return None
    nbr = col.median().normalizedDifference(["B8", "B12"])
    val = nbr.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=region, scale=20,
        maxPixels=1e9, bestEffort=True,
    ).get("nd").getInfo()
    return val


def cropland_fraction(geom_geojson):
    region = ee.Geometry(geom_geojson)
    crop = ee.Image("ESA/WorldCover/v200/2021").select("Map").eq(40)
    val = crop.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=region, scale=10,
        maxPixels=1e9, bestEffort=True,
    ).get("Map").getInfo()
    return val or 0.0


def viirs_count(footprint, first, last):
    if not FIRMS_KEY:
        return None
    buffered = footprint.buffer(POINT_BUFFER_DEG)
    minx, miny, maxx, maxy = buffered.bounds
    start = first - dt.timedelta(days=2)
    end = last + dt.timedelta(days=1)
    total = 0
    cursor = start
    while cursor <= end:
        span = min(5, (end - cursor).days + 1)
        url = (
            f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{FIRMS_KEY}/"
            f"{VIIRS_SOURCE}/{round(minx, 4)},{round(miny, 4)},{round(maxx, 4)},{round(maxy, 4)}/"
            f"{span}/{cursor.isoformat()}"
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


def load_detections(aoi):
    files = sorted(glob.glob(os.path.join(IN_DIR, f"{aoi}_*_final.geojson")))
    detections = []
    for path in files:
        date_str = os.path.basename(path)[len(aoi) + 1:-14]
        d = dt.date.fromisoformat(date_str)
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for f in data.get("features", []):
            geom = shape(f["geometry"])
            if not geom.is_valid:
                geom = geom.buffer(0)
            detections.append({
                "date": d,
                "geom": geom,
                "area_ha": f["properties"].get("area_ha", 0),
                "mean_dnbr": f["properties"].get("mean_dnbr"),
            })
    return detections, sorted({x["date"] for x in detections})


def overlaps(a, b):
    inter = a.intersection(b).area
    if inter <= 0:
        return False
    return inter / min(a.area, b.area) >= MATCH_OVERLAP


def range_gap(a, b):
    if a["first_date"] <= b["last_date"] and b["first_date"] <= a["last_date"]:
        return 0
    return min(abs((a["first_date"] - b["last_date"]).days),
               abs((b["first_date"] - a["last_date"]).days))


def merge_events(events):
    changed = True
    while changed:
        changed = False
        for i in range(len(events)):
            for j in range(i + 1, len(events)):
                a, b = events[i], events[j]
                if range_gap(a, b) > CLOSE_DAYS:
                    continue
                if not overlaps(a["footprint"], b["footprint"]):
                    continue
                a["footprint"] = unary_union([a["footprint"], b["footprint"]])
                a["first_date"] = min(a["first_date"], b["first_date"])
                a["last_date"] = max(a["last_date"], b["last_date"])
                a["scene_dates"] |= b["scene_dates"]
                a["detections"].extend(b["detections"])
                del events[j]
                changed = True
                break
            if changed:
                break
    for k, ev in enumerate(events):
        ev["id"] = k + 1
    return events


def build_events(detections, scene_dates, series_end):
    events = []
    for d in sorted(detections, key=lambda x: x["date"]):
        target = None
        best = 0.0
        for ev in events:
            if (d["date"] - ev["last_date"]).days > CLOSE_DAYS:
                continue
            inter = d["geom"].intersection(ev["footprint"]).area
            if inter <= 0:
                continue
            if inter / min(d["geom"].area, ev["footprint"].area) < MATCH_OVERLAP:
                continue
            if inter > best:
                best = inter
                target = ev
        if target is None:
            events.append({
                "id": len(events) + 1,
                "footprint": d["geom"],
                "first_date": d["date"],
                "last_date": d["date"],
                "scene_dates": {d["date"]},
                "detections": [d],
            })
        else:
            target["footprint"] = unary_union([target["footprint"], d["geom"]])
            target["last_date"] = max(target["last_date"], d["date"])
            target["scene_dates"].add(d["date"])
            target["detections"].append(d)

    events = merge_events(events)

    for ev in events:
        n_scenes = len(ev["scene_dates"])
        max_ha = max(x["area_ha"] for x in ev["detections"])
        dnbrs = [x["mean_dnbr"] for x in ev["detections"] if x["mean_dnbr"] is not None]
        ev["n_scenes"] = n_scenes
        ev["max_ha"] = max_ha
        ev["mean_dnbr"] = sum(dnbrs) / len(dnbrs) if dnbrs else None

        later_scenes = [s for s in scene_dates if s > ev["last_date"]]
        expired = any((s - ev["last_date"]).days > CLOSE_DAYS for s in later_scenes) or \
                  (series_end - ev["last_date"]).days > CLOSE_DAYS

        if n_scenes >= CONFIRM_SCENES or max_ha >= AUTO_CONFIRM_HA:
            ev["status"] = "confirmed"
        elif expired:
            ev["status"] = "rejected_transient"
        else:
            ev["status"] = "pending"
    return events


def recovery_test(ev):
    geo = mapping(ev["footprint"])
    f = ev["first_date"]
    pre = nbr_mean(geo, (f - dt.timedelta(days=35)).isoformat(), (f - dt.timedelta(days=5)).isoformat())
    late_start = ev["last_date"] + dt.timedelta(days=RECOVERY_WAIT)
    late = nbr_mean(geo, late_start.isoformat(), (late_start + dt.timedelta(days=RECOVERY_WINDOW)).isoformat())
    if pre is None or late is None:
        return None, None
    return round(late - pre, 3), pre


def main():
    init()
    all_features = []
    all_rows = []
    exclusion_features = []

    prefixes = sorted({
        os.path.basename(p)[:-25]
        for p in glob.glob(os.path.join(IN_DIR, "*_final.geojson"))
    })
    for aoi in prefixes:
        detections, scene_dates = load_detections(aoi)
        if not detections:
            continue
        series_end = max(scene_dates)
        events = build_events(detections, scene_dates, series_end)

        for ev in events:
            recovery_delta = None
            if ev["status"] == "confirmed":
                recovery_delta, _ = recovery_test(ev)
                if recovery_delta is not None and recovery_delta >= -RECOVERY_MARGIN:
                    ev["status"] = "rejected_recovered"

            crop = None
            viirs = None
            fire_type = None
            verification = None
            if ev["status"] in ("confirmed", "pending"):
                crop = round(cropland_fraction(mapping(ev["footprint"])), 3)
                fire_type = "agricultural" if crop >= CROP_LIMIT else "wildland"
                viirs = viirs_count(ev["footprint"], ev["first_date"], ev["last_date"])
                if viirs is not None and viirs > 0:
                    verification = "thermal"
                elif ev["status"] == "confirmed":
                    verification = "unverified"

            props = {
                "aoi": aoi,
                "event_id": f"{aoi}_{ev['id']:03d}",
                "first_date": ev["first_date"].isoformat(),
                "last_date": ev["last_date"].isoformat(),
                "n_detections": len(ev["detections"]),
                "n_scenes": ev["n_scenes"],
                "max_scene_ha": round(ev["max_ha"], 1),
                "mean_dnbr": round(ev["mean_dnbr"], 3) if ev["mean_dnbr"] else None,
                "status": ev["status"],
                "recovery_delta": recovery_delta,
                "fire_type": fire_type,
                "cropland_frac": crop,
                "viirs_points": viirs,
                "verification": verification,
            }
            all_features.append({
                "type": "Feature",
                "geometry": mapping(ev["footprint"]),
                "properties": props,
            })
            all_rows.append([
                aoi, props["event_id"], props["first_date"], props["last_date"],
                props["n_detections"], props["n_scenes"], props["max_scene_ha"],
                props["mean_dnbr"], props["status"], recovery_delta,
                fire_type, crop, viirs, verification,
            ])
            print(props["event_id"], props["first_date"], "->", props["last_date"],
                  props["n_scenes"], "sahne", props["max_scene_ha"], "ha", props["status"])

            if ev["status"] == "confirmed":
                exclusion_features.append({
                    "type": "Feature",
                    "geometry": mapping(ev["footprint"]),
                    "properties": {
                        "event_id": props["event_id"],
                        "exclude_until": (ev["last_date"] + dt.timedelta(days=EXCLUDE_DAYS)).isoformat(),
                    },
                })

    crs = {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}}
    with open(OUT_EVENTS, "w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "crs": crs, "features": all_features}, fh)
    with open(OUT_EXCLUSION, "w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "crs": crs, "features": exclusion_features}, fh)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["aoi", "event_id", "first_date", "last_date", "n_detections",
                    "n_scenes", "max_scene_ha", "mean_dnbr", "status", "recovery_delta",
                    "fire_type", "cropland_frac", "viirs_points", "verification"])
        w.writerows(all_rows)


if __name__ == "__main__":
    main()
