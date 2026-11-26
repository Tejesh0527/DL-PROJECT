"""Create result figures and markdown tables from outputs/results/*.json."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "outputs", "results")
FIG = os.path.join(ROOT, "outputs", "figures")
CONDS = ["clear", "rain", "fog", "dark"]


def bar(df, metric, title, path, methods):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    w = 0.8 / len(methods)
    for i, m in enumerate(methods):
        v = [df[(df.condition == c) & (df.method == m)][metric].values[0] for c in CONDS]
        ax.bar(np.arange(4) + i * w, v, w, label=m)
    ax.set_xticks(np.arange(4) + w * (len(methods) - 1) / 2); ax.set_xticklabels(CONDS)
    ax.set_ylabel(metric); ax.set_title(title); ax.legend(fontsize=7); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    d = pd.DataFrame(json.load(open(os.path.join(RES, "ua_detrac_results.json"))))
    b = pd.DataFrame(json.load(open(os.path.join(RES, "bdd100k_results.json"))))
    methods = list(dict.fromkeys(d.method))
    for m in ("AP50", "F1", "MOTA", "IDF1", "ECE", "FPS"):
        bar(d, m, f"UA-DETRAC test: {m}", os.path.join(FIG, f"detrac_{m}.png"), methods)
    bar(b, "AP50", "BDD100K (cross-domain, real weather): AP50", os.path.join(FIG, "bdd_AP50.png"),
        list(dict.fromkeys(b.method)))
    # training curves
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for k in ("baseline", "fyolo"):
        p = os.path.join(ROOT, "outputs", "training", f"{k}_training_curve.csv")
        if os.path.exists(p):
            c = pd.read_csv(p); c.columns = [x.strip() for x in c.columns]
            ax[0].plot(c["epoch"], c["train/box_loss"], marker="o", label=k)
            ax[1].plot(c["epoch"], c["metrics/mAP50(B)"], marker="o", label=k)
    ax[0].set_title("train box loss"); ax[1].set_title("val mAP50"); [a.legend() for a in ax]
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "training_curves.png"), dpi=150); plt.close(fig)
    # gate trace
    logs = json.load(open(os.path.join(ROOT, "outputs", "parameter_reports", "ratfa_gate_logs.json")))
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for c in CONDS:
        key = next(k for k in logs if k.startswith(f"F-YOLO + RATFA|{c}|"))
        ax.plot([x["gamma"] for x in logs[key]], label=f"{c} ({key.split('|')[2]})")
    ax.set_xlabel("frame"); ax.set_ylabel("gamma = q*r*tau"); ax.set_title("RATFA confidence gate per frame")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(FIG, "gate_trace.png"), dpi=150)
    plt.close(fig)
    cols = ["condition", "method", "AP50", "precision", "recall", "F1", "ECE", "MOTA", "IDF1",
            "ID_switches", "FPS", "latency_ms"]
    with open(os.path.join(RES, "results_tables.md"), "w") as f:
        f.write("## UA-DETRAC test (4 sequences x 150 frames)\n\n")
        f.write(d[cols].round(3).to_markdown(index=False))
        f.write("\n\n## BDD100K val subset (real weather labels, cross-domain)\n\n")
        f.write(b[["condition", "method", "n_images", "AP50", "precision", "recall", "F1", "ECE"]]
                .round(3).to_markdown(index=False))
        f.write("\n")
    print(open(os.path.join(RES, "results_tables.md")).read())


if __name__ == "__main__":
    main()
