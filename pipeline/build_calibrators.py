# pipeline/build_calibrators.py
import json
from pathlib import Path

LEAGUES = {"I1": "Serie A", "E0": "Premier League", "SP1": "La Liga", "D1": "Bundesliga"}
OUT = Path("app/public/data/calibrators.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

def identity_curve():
    # curva monotona y=x (21 punti per interpolazione fluida)
    xs = [i/20 for i in range(21)]
    ys = xs[:]
    return {"x": xs, "y": ys}

def main():
    # soglie intere usate dall'app (i .5 vengono convertiti a interi con +1)
    shots_ks = list(range(4, 31))          # tiri H/A: 4..30
    corners_ks = list(range(3, 21))        # angoli H/A: 3..20
    corners_tot_ks = list(range(5, 25))    # angoli totali: 5..24

    out = {}
    base_curve = identity_curve()
    for code in LEAGUES.keys():
        out[code] = {
            "shots": {"home": {}, "away": {}},
            "corners": {"home": {}, "away": {}},
            "corners_total": {"total": {}},
        }
        for k in shots_ks:
            out[code]["shots"]["home"][str(k)] = base_curve
            out[code]["shots"]["away"][str(k)] = base_curve
        for k in corners_ks:
            out[code]["corners"]["home"][str(k)] = base_curve
            out[code]["corners"]["away"][str(k)] = base_curve
        for k in corners_tot_ks:
            out[code]["corners_total"]["total"][str(k)] = base_curve

    OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT}")

if __name__ == "__main__":
    main()
