"""README figure: OOT difference to B0 with 95% CI for every branch of the three domains.

  python scripts/figures.py

Reads artifacts/{,kion/,x5/}compare_oot_eval.json (written by compare.py) and writes docs/figures/oot_deltas.png.
No model is scored here: the figure only redraws the bootstrap results of the single OOT run.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
OUT = ROOT / "docs" / "figures" / "oot_deltas.png"

# light chart surface, text and line tokens; categorical slots 1-3 checked for colour-vision pairs, muted grey = reference
SURFACE, TEXT, TEXT2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
FAMILIES = {
    "auto": ("Автопоиск признаков", "#2a78d6"),
    "signal": ("Новые сигналы в бустинге", "#eb6834"),
    "ctx": ("Время запроса (другая постановка)", "#1baf7a"),
    "short": ("Короткая история (Perseus, обрезка B0)", MUTED),
}
PANELS = [
    ("Yambda (музыка)", ART / "compare_oot_eval.json", [
        ("r1_decay", "R1, затухание (96)", "auto"),
        ("r1_window", "R1, окна (96)", "auto"),
        ("r1_window_k56", "R1, окна, 56 = как B0", "auto"),
        ("r1_decay_k56", "R1, затухание, 56", "auto"),
        ("r1_r5", "R1 + гипотеза агента", "auto"),
        ("r1_decay_stable", "R1, отбор по устойчивости", "auto"),
        ("b0_ials", "B0 + iALS", "signal"),
        ("b0_audio", "B0 + аудио", "signal"),
        ("b0_perseus", "B0 + скор Perseus", "signal"),
        ("full", "полный стек", "signal"),
        ("b0_ctx", "B0 + ручной контекст", "ctx"),
        ("r1_ctx", "R1 + найденный контекст", "ctx"),
        ("b0_ctx_mood", "B0 + контекст + состояние", "ctx"),
        ("r3_perseus", "Perseus сам по себе", "short"),
        ("b0_trunc256", "B0 на последних 256 событиях", "short"),
    ]),
    ("KION (кино)", ART / "kion" / "compare_oot_eval.json", [
        ("r1_decay", "R1, затухание (64)", "auto"),
        ("r1_window", "R1, окна (32)", "auto"),
        ("r1_decay_k33", "R1, затухание, 33 = как B0", "auto"),
        ("r1_window_k33", "R1, окна, 33", "auto"),
    ]),
    ("X5 (продукты)", ART / "x5" / "compare_oot_eval.json", [
        ("r1_decay", "R1, затухание (64)", "auto"),
        ("r1_window", "R1, окна (24)", "auto"),
        ("r1_decay_k25", "R1, затухание, 25 = как B0", "auto"),
        ("r1_window_k25", "R1, окна, 25", "auto"),
    ]),
]
METRICS = [("d_gauc", "Δ GAUC"), ("d_ndcg10", "Δ NDCG@10")]
# selective direct labels: the headline rows only
LABELS = {("Yambda (музыка)", "d_gauc"): {"r1_decay", "full", "b0_ctx", "r3_perseus"},
          ("KION (кино)", "d_ndcg10"): {"r1_decay"}}

def fmt(v: float, digits: int) -> str:
    return "0" if abs(v) < 1e-12 else f"{v:+.{digits}f}".replace(".", ",").replace("-", "−")


# axis ticks: 4 digits with a trailing zero dropped, so 0,0015 and 0,002 both read right
ru = FuncFormatter(lambda v, _: fmt(v, 4)[:-1] if len(fmt(v, 4)) > 1 and fmt(v, 4).endswith("0") else fmt(v, 4))


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": TEXT,
                         "axes.edgecolor": BASE, "xtick.color": TEXT2, "ytick.color": TEXT})
    data = [(title, json.loads(path.read_text(encoding="utf-8"))["bootstrap"], rows) for title, path, rows in PANELS]
    fig, axes = plt.subplots(len(PANELS), len(METRICS), figsize=(11, 9.6), facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [len(r) + 1.2 for *_, r in PANELS], "hspace": 0.38, "wspace": 0.08})
    for i, (title, boot, rows) in enumerate(data):
        for j, (key, mname) in enumerate(METRICS):
            ax = axes[i, j]
            ax.set_facecolor(SURFACE)
            ax.axvline(0, color=BASE, lw=1.2, zorder=1)
            for y, (branch, _, fam) in enumerate(rows):
                r, color = boot[branch], FAMILIES[fam][1]
                lo, hi = r[f"{key}_ci"]
                ax.plot([lo, hi], [y, y], color=color, lw=2, solid_capstyle="round", zorder=2)
                ax.plot(r[key], y, "o", ms=7.5, color=color, mec=SURFACE, mew=1.5, zorder=3)
                if branch in LABELS.get((title, key), ()):
                    ax.annotate(fmt(r[key], 4), (hi, y), xytext=(5, 0), textcoords="offset points", va="center",
                                fontsize=8.5, color=TEXT2)
            ax.set_ylim(len(rows) - 0.4, -0.6)
            ax.set_yticks(range(len(rows)))
            ax.set_yticklabels([lab for _, lab, _ in rows] if j == 0 else [])
            ax.tick_params(axis="y", length=0)
            ax.tick_params(axis="x", length=0, labelsize=8.5)
            ax.xaxis.set_major_locator(MaxNLocator(6))
            ax.xaxis.set_major_formatter(ru)
            ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            lo_all = min(boot[b][f"{key}_ci"][0] for b, _, _ in rows)
            hi_all = max(boot[b][f"{key}_ci"][1] for b, _, _ in rows)
            pad = 0.12 * (hi_all - lo_all)
            ax.set_xlim(min(lo_all, 0) - pad, max(hi_all, 0) + 2.6 * pad)
            ax.set_title(f"{title} · {mname}", loc="left", fontsize=10, color=TEXT, pad=6)
    fig.legend(handles=[Line2D([], [], marker="o", color=c, lw=2, ms=7, mec=SURFACE, label=n) for n, c in FAMILIES.values()],
               loc="upper left", bbox_to_anchor=(0.01, 0.985), ncol=4, frameon=False, fontsize=9, handlelength=1.6)
    fig.text(0.01, 0.945, "Разница с B0 на OOT (точка) и 95% ДИ парного бутстрэпа по пользователям. Вправо от нуля — лучше B0. "
             "Шкалы у панелей свои.", fontsize=8.5, color=TEXT2)
    fig.subplots_adjust(left=0.2, right=0.965, top=0.9, bottom=0.04)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170, facecolor=SURFACE)
    print(OUT)


if __name__ == "__main__":
    main()
