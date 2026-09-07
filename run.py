import sys
import subprocess
import csv
import os

STEPS = [
    ["python", "scan.py"],
    ["python", "gates.py"],
    ["python", "brightness_gate.py"],
    ["python", "events.py"],
]


def main():
    extra = sys.argv[1:]
    for i, step in enumerate(STEPS):
        cmd = step + (extra if i == 0 else [])
        print("==>", " ".join(cmd))
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("adim basarisiz:", step[1])
            sys.exit(result.returncode)

    if os.path.exists("events.csv"):
        with open("events.csv", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        confirmed = [r for r in rows if r["status"] == "confirmed"]
        pending = [r for r in rows if r["status"] == "pending"]
        print()
        print("onayli olay:", len(confirmed))
        for r in confirmed:
            print(" ", r["event_id"], r["first_date"], "->", r["last_date"],
                  r["max_scene_ha"], "ha")
        print("beklemede:", len(pending))


if __name__ == "__main__":
    main()
