"""Draw the benchmark figures from the CSVs the benchmarks write.

    python benchmarks/plot_benchmarks.py

Adapted from dpd-stream's. Reads benchmarks/results/scale.csv and
effects.csv and writes figures of wall time, peak memory and peak disk
against the number of units, one line per library, in a light and a dark
version:

    docs/figures/{scale_nn,scale_other,effects}_{time,memory,disk}.{light,dark}.svg

scale_nn holds the nearest-neighbor scenarios with exact strata (A, B, E,
F, H), scale_other CEM, kernel matching and the unstratified Mahalanobis match
(C, D, G); see _scenarios.py. match-stream is magenta (dashed: by stratum);
every library has its own color and marker, so no line is identified by
color alone. Points that failed, ran out of memory or timed out are left out
and listed under the figure.

Only this file needs changing to restyle them; the numbers stay in the CSVs.
"""

from __future__ import annotations

import csv
import os
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

HERE = Path(__file__).resolve().parent
# MS_BENCH_RESULTS and MS_BENCH_FIGURES: elsewhere, for a trial run
RESULTS = Path(os.environ.get("MS_BENCH_RESULTS", HERE / "results"))
FIGURES = Path(os.environ.get("MS_BENCH_FIGURES", HERE.parent / "docs" / "figures"))

SLOTS = {
    "blue": ("#2a78d6", "#3987e5"), "orange": ("#eb6834", "#d95926"),
    "aqua": ("#1baf7a", "#199e70"), "yellow": ("#eda100", "#c98500"),
    "magenta": ("#e87ba4", "#d55181"), "green": ("#008300", "#008300"),
    "violet": ("#6250d6", "#9085e9"), "red": ("#e34948", "#e66767"),
}
THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e",
              "grid": "#e4e3de", "axis": "#b9b8b1", "slot": 0},
    "dark": {"surface": "#1a1a19", "text": "#ffffff", "muted": "#c3c2b7",
             "grid": "#34342f", "axis": "#5b5a55", "slot": 1},
}

# Each figure: its CSV, its series -- configuration -> (color slot, marker,
# line style) -- its panels (title, model) and x label.
SERIES = {
    "match-stream": ("magenta", "v", "-"),
    "match-stream (by_stratum)": ("magenta", "v", "--"),
    "MatchIt (R)": ("blue", "o", "-"),
    "Matching (R)": ("yellow", "s", "-"),
    "Matching Matchby (R)": ("yellow", "s", "--"),
    "pymatchit": ("green", "D", "-"),
    "Polars join_asof": ("aqua", "^", "-"),
    "fastmatch (faiss)": ("violet", "P", "-"),
    "fastmatch (scikit-learn)": ("violet", "P", "--"),
    "fastmatch (voyager, approximate)": ("red", "X", ":"),
}
EFFECTS = {
    "match-stream if": ("magenta", "v", "-"),
    "match-stream ai": ("magenta", "o", "--"),
    "match-stream ai n_variance_matches=1 bias_adjustment": ("magenta", "s", ":"),
    "match-stream fitted score ai n_variance_matches=1 score_correction": ("orange", "D", "-"),
    "Matching AI (R)": ("yellow", "o", "--"),
    "Matching AI n_variance_matches=1 BiasAdjust (R)": ("yellow", "s", ":"),
    "match-stream G ai n_variance_matches=1 bias_adjustment": ("magenta", "^", "-."),
    "fastmatch G bias-corrected (faiss)": ("violet", "P", "-"),
}
FIGS = {
    "scale_nn": {
        "csv": "scale", "xlabel": "Units (500 exact strata)",
        "panels": [("A: 1:1 on the score, caliper", "A"),
                   ("B: 1:1 Mahalanobis", "B"),
                   ("E: 3:1, three calipers", "E"),
                   ("F: 1:1 without replacement, calipers", "F"),
                   ("H: 1:1 Mahalanobis without replacement, calipers", "H")],
        "series": SERIES,
    },
    "scale_other": {
        "csv": "scale", "xlabel": "Units",
        "panels": [("C: coarsened exact matching", "C"),
                   ("D: kernel matching on the score", "D"),
                   ("G: 1:1 Mahalanobis, no strata", "G")],
        "series": SERIES,
    },
    "effects": {
        "csv": "effects", "xlabel": "Units",
        "panels": [("Match, then the effect and its standard error", "effects")],
        "series": EFFECTS,
    },
}
METRICS = {
    "time": ("wall_s", "Wall time (seconds)", 1.0),
    "memory": ("peak_memory_mb", "Peak memory (GB)", 1e-3),
    "disk": ("peak_disk_mb", "Peak disk (GB)", 1e-3),
}


def read(name):
    path = RESULTS / f"{name}.csv"
    if not path.exists():
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _count(v, _pos=None):
    if v >= 1e6:
        return f"{v / 1e6:g}M"
    if v >= 1e3:
        return f"{v / 1e3:g}k"
    return f"{v:g}"


def _value(v, _pos=None):
    if v >= 100:
        return f"{v:,.0f}"
    if v >= 1:
        return f"{v:g}"
    return f"{v:.2g}"


def figure(name, metric, theme):
    spec = FIGS[name]
    rows = read(spec["csv"])
    if not rows:
        return None
    col, ylabel, scale = METRICS[metric]
    t = THEMES[theme]
    panels, series = spec["panels"], spec["series"]
    width = max(4.2 * len(panels) + 0.6, 7.0)
    # panels of one model on one scale (the mixed logit's two)
    fig, axes = plt.subplots(1, len(panels), figsize=(width, 4.0), squeeze=False,
                             sharey=bool(spec.get("sharey")))
    fig.patch.set_facecolor(t["surface"])
    failed, handles = [], {}
    per_panel = spec.get("panel_series") or [list(series)] * len(panels)
    for ax, (title, model), labels in zip(axes[0], panels, per_panel):
        ax.set_facecolor(t["surface"])
        mrows = [r for r in rows if r.get("model", model) == model]
        local = {}
        for label in labels:
            slot, marker, style = series[label]
            pts = []
            for r in mrows:
                if r["configuration"] != label:
                    continue
                if r["status"] != "ok":
                    where = f" ({model})" if len(panels) > 1 else ""
                    failed.append(f"{label}{where} at {_count(float(r['n']))}: "
                                  f"{r['status'].split(':')[0]}")
                    continue
                v = float(r.get(col) or 0) * scale
                if metric == "disk" and v <= 0:
                    continue
                pts.append((float(r["n"]), max(v, 1e-3)))
            if not pts:
                continue
            pts.sort()
            color = SLOTS[slot][t["slot"]]
            (h,) = ax.plot(*zip(*pts), linestyle=style, marker=marker, color=color,
                           linewidth=2, markersize=7, markeredgecolor=t["surface"],
                           markeredgewidth=1.2, label=label)
            handles.setdefault(label, h)
            local[label] = h
        if spec.get("legend") == "panel" and local:
            ax.legend(list(local.values()), list(local), loc="upper left", frameon=False,
                      fontsize=7.5, labelcolor=t["text"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, color=t["text"], fontsize=11, loc="left")
        ax.set_xlabel(spec["xlabel"], color=t["muted"], fontsize=9)
        if spec.get("xticks"):
            ax.set_xticks(spec["xticks"])
        ax.xaxis.set_major_formatter(FuncFormatter(_count))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(FuncFormatter(_value))
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 3.0)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="major", color=t["grid"], linewidth=0.8)
        ax.tick_params(colors=t["muted"], labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(t["axis"])
    axes[0][0].set_ylabel(ylabel, color=t["muted"], fontsize=9)
    order = [lb for lb in series if lb in handles]
    # the note of what is not shown at the bottom, the legend above it
    note = ""
    if failed and metric != "disk":          # only match-stream writes to disk
        note = "Not shown: " + "; ".join(dict.fromkeys(failed)) + "."
    if spec.get("note") and metric != "disk":
        note = (note + " " + spec["note"]).strip()
    if note:
        note = "\n".join(textwrap.wrap(note, int(width * 19)))
        fig.text(0.01, 0.01, note, color=t["muted"], fontsize=7, va="bottom")
    note_h = (0.03 * (note.count("\n") + 1) + 0.01) if note else 0.0
    if spec.get("legend") == "panel":
        order = []
    ncol = min(len(order), 4 if len(panels) > 1 else 3)
    nrow = -(-len(order) // max(ncol, 1)) if order else 0
    if order:
        fig.legend([handles[lb] for lb in order], order, loc="lower center",
                   ncol=ncol, frameon=False, fontsize=8.5,
                   labelcolor=t["text"], bbox_to_anchor=(0.5, note_h))
    fig.tight_layout(rect=(0, note_h + 0.06 + 0.05 * nrow, 1, 1))
    return fig


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    for name, spec in FIGS.items():
        for metric in spec.get("metrics", METRICS):
            for theme in THEMES:
                fig = figure(name, metric, theme)
                if fig is None:
                    continue
                path = FIGURES / f"{name}_{metric}.{theme}.svg"
                fig.savefig(path, facecolor=fig.get_facecolor())
                plt.close(fig)
                print(f"wrote {path}")


if __name__ == "__main__":
    main()
