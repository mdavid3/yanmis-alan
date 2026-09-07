import os
import json
import glob

IN_DIR = "candidates"
OUT_DIR = "candidates_clean"


def clean_geometry(geom):
    if geom is None:
        return None
    out = {"type": geom["type"], "coordinates": geom["coordinates"]}
    return out


def clean_feature(f, fid):
    geom = clean_geometry(f.get("geometry"))
    if geom is None:
        return None
    props = {}
    for k, v in (f.get("properties") or {}).items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            props[k] = v
    return {"type": "Feature", "id": fid, "geometry": geom, "properties": props}


def clean_file(path):
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    features = []
    for i, f in enumerate(data.get("features", [])):
        cf = clean_feature(f, i)
        if cf is not None:
            features.append(cf)

    out = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features,
    }
    return out, len(features)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = glob.glob(os.path.join(IN_DIR, "*.geojson"))
    if not files:
        print("dosya bulunamadi")
        return

    for path in files:
        out, n = clean_file(path)
        name = os.path.basename(path)
        target = os.path.join(OUT_DIR, name)
        with open(target, "w", encoding="utf-8") as fh:
            json.dump(out, fh)
        print(name, n)


if __name__ == "__main__":
    main()
