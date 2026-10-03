"""Build docs/figures/*.html from the templates and the chart data. Render the PNGs with e2e/render_charts.mjs.

  uv run python scripts/chart_data.py     # recompute the data (CPU, a few minutes)
  uv run python scripts/build_charts.py   # fill the templates
  cd e2e && node render_charts.mjs        # PNGs for the slides
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"


def main() -> None:
    d = json.loads((FIG / "chart_data.json").read_text(encoding="utf-8"))
    laya = json.loads((FIG / "laya_gate_data.json").read_text(encoding="utf-8"))
    roc_mean = next(r["mean"] for r in d["ranges"] if r["label"] == "ROC-AUC") / 100
    pages = {
        "lightgbm_charts.html": d,
        "slide_charts.html": {**laya, "roc": d["roc"], "ranges": d["ranges"], "load": d["load"],
                              "live_vs_offline": d["live_vs_offline"]},
    }
    for name, data in pages.items():
        t = (FIG / "templates" / name).read_text(encoding="utf-8")
        t = t.replace("__ROC_MEAN__", f"{roc_mean:.3f}").replace("__DATA__", json.dumps(data, separators=(",", ":")))
        (FIG / name).write_text(t, encoding="utf-8")
        print("wrote", name)


if __name__ == "__main__":
    main()
