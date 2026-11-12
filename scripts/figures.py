"""Figures for the paper, from data/results.json (python collect.py first). Light-mode static PNGs (Kaggle writeups take images).
Palette and marks follow the dataviz reference: surface #fcfcfb, ink #0b0b0b / #52514e / #898781, hairline grid #e1e0d9, categorical slots
1-3 (blue, orange, aqua; the three that validate all-pairs for scatter), one axis per panel, selective direct labels.
  python figures.py DATA_JSON OUT_DIR"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

D = json.load(open(sys.argv[1])); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
S1, S2, S3, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#c3c2b7"; S4 = "#eda100"   # categorical slot 4 (validated with slots 1-3)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 9.5,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK, "axes.labelcolor": INK2,
                     "axes.edgecolor": GRID, "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
                     "ytick.labelcolor": INK2, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False})
NAME = {"sft": "SFT on teacher text", "sft_s": "SFT, shortest correct", "sft_c_s": "SFT, shortest + self-anchor", "opd_c": "On-policy (weak teacher)",
        "mix_c": "SFT + on-policy (weak teacher)", "opd_cf": "On-policy (refit teacher)", "mix_cf": "SFT + on-policy (refit teacher)",
        "rw_cf": "Rewrites via corrector", "rb": "Runbook rewrites", "rb_c": "Runbook rewrites + self-anchor", "sft_self_s": "SFT, shortest + KL to original", "sft_b_s": "SFT, shortest + cross-model anchor",
        "sft_mt_s": "SFT, shortest + the corrector's fitting text", "sft_self_mt_s": "SFT, shortest + fitting text + KL to original"}   # 190 (figure 6)
NAME174 = {"sft": "SFT (second run)", "sft_b": "SFT + cross-model anchor", "sft_c": "SFT + self-anchor", "kl_b": "Cross-model anchor only", "sdft": "SDFT (in-context self-teacher)"}
GROUP = {"sft": S1, "sft_s": S1, "sft_c_s": S1, "sft_b": S1, "sft_c": S1, "mix_c": S1, "mix_cf": S1, "opd_c": S2, "opd_cf": S2, "kl_b": S2, "sdft": S2, "rw_cf": S3, "rb": S3, "rb_c": S3, "sft_self_s": S1}


def style(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.7); ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", pad_inches=0.15); plt.close(fig); print("wrote", name)


def legend_groups(ax, loc="upper right", extra=None):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", ms=7, color=c, label=l) for c, l in ((S1, "Trains on the teacher's text"), (S2, "No teacher text (on-policy / KL only)"), (S3, "Trains on student-written text"))]
    ax.legend(handles=h + (extra or []), loc=loc, fontsize=8.5, handletextpad=0.3, labelcolor=INK2)


# Figure 1: the corrector (schematic)
def fig_method():
    """Figure 1: how the corrector connects the teacher and the student (xbridge.py XTr / translator.py gall): it reads source states at 50% and
    65% depth (the teacher's for the cross-model corrector, aligned to the student's tokens; the student's own for the self-corrector) and, in
    both variants, the student's own state at the write layer (35% depth); in a second pass it adds a gated correction at the write layer and
    every layer above; the corrected student's predictions are then the training target (depths are nominal, rounded down to a layer)"""
    from matplotlib.patches import Circle, Rectangle
    fig, ax = plt.subplots(figsize=(11.8, 6.45)); ax.set_xlim(0, 12.4); ax.set_ylim(0, 6.78); ax.axis("off")
    def box(x, y, w, h, text, fc, ec=None, color=INK, size=9, weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec or fc, lw=1.0))
        if text:
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=color, weight=weight)
    def arrow(a, b, color=INK2, rad=0.0, lw=1.4):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, lw=lw, color=color, connectionstyle=f"arc3,rad={rad}", zorder=3))
    def badge(x, y, n, color):
        ax.add_patch(Circle((x, y), 0.15, fc=color, ec=SURF, lw=1.2, zorder=5)); ax.text(x, y, str(n), ha="center", va="center", fontsize=8.5, color="white", weight="bold", zorder=6)
    Y = lambda k: 1.42 + k * 0.47; H = 0.37; yc = lambda k: Y(k) + H / 2
    TX, TW, CX, CW, SX, SW, UX, UW = 0.2, 2.2, 3.15, 2.35, 6.3, 2.3, 9.3, 2.95; CY, CH = 3.07, 1.2
    # the two phases
    for x0, x1, h, sub in ((TX, SX + SW, "A   Fit the corrector", "teacher and student stay frozen"), (UX, UX + UW, "B   Train the student", "")):
        ax.text(x0, 6.56, h, fontsize=11, weight="bold", color=INK); ax.text(x0 + 2.42, 6.56, sub, fontsize=8.6, color=INK2, va="baseline")
        ax.plot([x0, x1], [6.38, 6.38], color=GRID, lw=1.2, solid_capstyle="round")
    teacher = ["embeddings", "layers", "layers", "layers", "50% depth", "layers", "65% depth", "layers", "next-token head"]
    student = ["embeddings", "layers", "35% depth", "layers", "50% depth", "layers", "65% depth", "layers", "next-token head"]
    for x, h, sub in ((TX, "Teacher", "Gemma 4 E4B"), (SX, "Student", "Qwen3-1.7B or Gemma 4 E2B")):
        ax.text(x, 6.08, h, fontsize=10.5, weight="bold", color=INK); ax.text(x, 5.84, sub, fontsize=8.3, color=INK2)
    for k, t in enumerate(teacher):
        hl = "%" in t; box(TX, Y(k), TW, H, t, "#fdeee7" if hl else "#f0efec", color=INK if hl else INK2, size=8.3)
    for k, t in enumerate(student):
        box(SX, Y(k), SW, H, t, "#eef4fc" if t[:2] in ("50", "65") else "#e8f6f0" if t[:2] == "35" else "#f0efec", color=INK if "%" in t else INK2, size=8.3)
        if 2 <= k <= 7:                                                  # the layers that receive the correction
            ax.add_patch(Rectangle((SX - 0.02, Y(k) + 0.03), 0.09, H - 0.06, fc=S3, ec="none", zorder=4))
    box(CX, CY, CW, CH, "", "#e8f6f0", ec=S3); ax.text(CX + CW / 2, CY + 0.83, "Corrector", ha="center", fontsize=10, weight="bold")
    ax.text(CX + CW / 2, CY + 0.38, "a small gated MLP\nstarts as “no change”", ha="center", va="center", fontsize=8.2, color=INK2)
    # 1: what it reads
    for k, yy in ((4, CY + 0.67), (6, CY + 0.90)):
        arrow((TX + TW, yc(k)), (CX, yy), S2, rad=-0.1)                  # cross-model corrector: the teacher's states
        arrow((SX, yc(k)), (CX + CW, yy), S1, rad=0.1)                    # self-corrector: the student's own states, first pass
    arrow((SX, yc(2) + 0.10), (CX + CW, CY + 0.36), INK2, rad=0.22, lw=1.2)   # both: the student's own state at the write layer
    badge(CX + 0.30, 5.52, 1, INK2); ax.text(CX + 0.55, 5.52, "reads hidden states", fontsize=8.8, color=INK, weight="bold", va="center")
    ax.text(CX + CW / 2, 5.20, "cross-model corrector: the teacher's", ha="center", fontsize=8.2, color=S2, weight="bold")
    ax.text(CX + CW / 2, 4.94, "self-corrector: the student's own", ha="center", fontsize=8.2, color=S1, weight="bold")
    ax.text(CX + CW / 2, 4.68, "both: the student's state at 35% depth", ha="center", fontsize=8.2, color=INK2, weight="bold")
    # 2: what it writes
    arrow((CX + CW - 0.30, CY), (SX - 0.04, yc(2) - 0.09), S3, rad=0.32, lw=2.0)   # second pass: the gated correction
    badge(CX - 0.08, 2.66, 2, S3); ax.text(CX + 0.17, 2.66, "adds a gated correction", fontsize=8.8, color=S3, weight="bold", va="center")
    ax.text(CX + 0.17, 2.28, "at 35% depth and every layer\nabove, in a second pass", ha="left", va="center", fontsize=8.2, color=S3, weight="bold")
    # 3: how it is used
    ax.text(UX, 5.96, "The corrected student's predictions\nare the target; only a LoRA is kept.", fontsize=8.6, color=INK2, va="center", linespacing=1.35)
    uses = [("Telepathy-like teaching", "online, item by item (Sections 4.2-4.4)"), ("Distilled LoRA", "a LoRA learns to match them (4.5)"),
            ("On-policy teacher", "they score the student's samples (4.5)"), ("Anchor", "SFT plus a KL pull toward them (4.6)")]
    for k, (h, t) in enumerate(uses):
        y = 4.66 - k * 0.92; box(UX, y, UW, 0.78, "", "#f7f7f5", ec=GRID)
        ax.text(UX + 0.14, y + 0.53, h, fontsize=9, weight="bold", va="center"); ax.text(UX + 0.14, y + 0.24, t, fontsize=7.7, color=INK2, va="center")
    ax.plot([UX - 0.14] * 2, [1.98, 5.38], color=MUTED, lw=1.4, solid_capstyle="round")   # the uses, as one group
    arrow((SX + SW, yc(8)), (UX - 0.16, 5.0), INK2, rad=-0.2); badge(SX + SW + 0.42, yc(8) + 0.02, 3, INK2)
    notes = ["1  Both models read the same text. The cross-model corrector reads the teacher's hidden states (across tokenizers, each student token takes the state of the",
             "    last teacher token ending at or before it). The self-corrector, the control, reads only the student's own. Both also read the student's state where they write.",
             "2  The student runs again with the correction added from 35% depth up (green edges). The corrector is fitted so that this corrected student predicts the teacher's replies.",
             "3  The corrected student's predictions then train a LoRA on the student. Afterwards only the LoRA is kept: the student runs without the teacher or the corrector."]
    for k, t in enumerate(notes):
        ax.text(0.2, 0.98 - k * 0.25, t, fontsize=8.2, color=INK2, va="center")
    save(fig, "fig1_corrector.png")


def fig_wide():
    """Figure 2: the agent-turn LoRAs on all easy and medium LiveCodeBench problems (178), as problems gained and lost against the untrained
    student (the paired counts: form only / student only), three seeds pooled"""
    W = D["wide"]; keys = ["lora_sd", "lora_c", "lora_b", "lora_self"]
    lab = {"lora_self": "SFT + KL to the original student", "lora_b": "Cross-model corrector, distilled", "lora_c": "Self-corrector, distilled", "lora_sd": "Plain SFT on teacher text"}
    pair = lambda s, k: W[s]["pairs"][f"{k} vs qwen"]
    gain = {k: sum(pair(s, k)[0] for s in W) for k in keys}; lost = {k: sum(pair(s, k)[1] for s in W) for k in keys}
    tot = {k: sum(W[s]["passed"][k] for s in W) for k in keys + ["qwen"]}; n = sum(W[s]["problems"] for s in W)
    fig, ax = plt.subplots(figsize=(8.8, 3.7)); style(ax, "x")
    for i, k in enumerate(keys):
        ax.barh(i, gain[k], height=0.5, color=S1); ax.barh(i, -lost[k], height=0.5, color=S2)
        ax.text(gain[k] + 4, i, f"+{gain[k]}", va="center", fontsize=8.8, color=INK)
        ax.text(-lost[k] - 4, i, f"\u2212{lost[k]}", va="center", ha="right", fontsize=8.8, color=INK)
        ax.text(330, i, f"+{gain[k] - lost[k]}", va="center", ha="right", fontsize=8.8, color=INK, weight="bold"); ax.text(392, i, f"{tot[k]}", va="center", ha="right", fontsize=8.8, color=INK2)
    hy = -0.78   # a header row in place of a legend: the two series named where they start, the two derived columns named above their numbers
    ax.scatter([-10, 10], [hy, hy], marker="s", s=42, color=[S2, S1], clip_on=False, zorder=3)
    ax.text(-17, hy, "lost (solved before, now failed)", ha="right", va="center", fontsize=8.6, color=INK2); ax.text(17, hy, "newly solved", ha="left", va="center", fontsize=8.6, color=INK2)
    ax.text(330, hy, "net", ha="right", va="center", fontsize=8.6, color=INK2, weight="bold"); ax.text(392, hy, "now solved", ha="right", va="center", fontsize=8.6, color=INK2)
    ax.axvline(0, color=INK2, lw=1); ax.set_xlim(-130, 400); ax.set_xticks([-100, -50, 0, 50, 100, 150, 200, 250])
    ax.set_yticks(range(len(keys)), [lab[k] for k in keys]); ax.set_ylim(len(keys) - 0.5, -1.15)
    ax.set_xlabel(f"Problems, three seeds pooled ({n:,} problem-runs; the untrained student solved {tot['qwen']}). Farther right is better; a shorter orange bar is better", fontsize=8.3)
    ax.set_title("Trained on agent turns, tested on contest problems: plain SFT gains the most and loses fewer than the corrector forms", fontsize=10, pad=14)
    save(fig, "fig2_wide_coding.png")


GROUP_NAME = ((S1, "Trains on the teacher's text"), (S2, "No teacher text (on-policy / KL only)"), (S3, "Trains on student-written text"))


# Figure 5: teaching vs forgetting map
def fig_map():
    """Figure 5: every Qwen recipe on both teaching measures against problems lost. Points are numbered by held-out rank (1 = most held
    problems solved) and named once in a key under the panels, so both panels carry the same short labels and no leader lines"""
    P = D["main"]["pooled"]
    order = sorted(P, key=lambda a: -P[a]["held solved"]); num = {a: i + 1 for i, a in enumerate(order)}
    fig, axs = plt.subplots(1, 2, figsize=(10.6, 6.3), sharey=True); fig.subplots_adjust(top=0.86, bottom=0.30, left=0.08, right=0.98, wspace=0.08)
    # number offsets in points per panel; the cluster of six near (150-160, 40-58) and (382-406, 37-58) is spread by hand
    offs = [{"sft_s": (7, 5), "sft_self_s": (7, -4), "mix_cf": (7, 6), "sft_c_s": (-7, -6), "sft": (7, 5), "opd_cf": (7, 5), "rb": (7, -5), "rw_cf": (-7, 5), "rb_c": (-7, -5), "mix_c": (7, 5), "opd_c": (7, -5)},
            {"sft_s": (7, 5), "sft_self_s": (-7, 5), "mix_cf": (-6, -7), "sft_c_s": (-7, 5), "sft": (-7, 5), "opd_cf": (7, 5), "rb": (7, -5), "rw_cf": (7, 5), "rb_c": (-7, -6), "mix_c": (7, 5), "opd_c": (7, -5)}]
    for ax, xk, xl, off in ((axs[0], "held solved", "Held-out problems solved (of 564; higher is better)", offs[0]), (axs[1], "fixed", "Trained problems newly solved (of 600; higher is better)", offs[1])):
        style(ax, "both")
        for a, v in P.items():
            y = v["lost held"] + v["lost train pool"]; o = off[a]
            ax.scatter(v[xk], y, s=64, color=GROUP[a], edgecolors=SURF, linewidths=1.6, zorder=3)
            ax.annotate(str(num[a]), (v[xk], y), xytext=o, textcoords="offset points", fontsize=8.6, weight="bold", color=INK2, va="center", ha="right" if o[0] < 0 else "left", zorder=4)
        ax.set_xlabel(xl)
        ax.annotate("", xy=(0.985, 0.025), xytext=(0.905, 0.125), xycoords="axes fraction", textcoords="axes fraction", arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2, mutation_scale=12))
        ax.text(0.90, 0.135, "better", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.6, color=MUTED, style="italic")
    axs[0].set_xlim(105, 178); axs[1].set_xlim(190, 480); axs[0].set_ylim(30, 121)
    axs[0].set_ylabel("Problems lost that the student solved before (lower is better)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", ms=7, color=c, label=l) for c, l in GROUP_NAME]
    fig.legend(handles=h, loc="upper left", ncol=3, bbox_to_anchor=(0.065, 0.945), fontsize=8.8, handletextpad=0.3, columnspacing=1.6, labelcolor=INK2)
    fig.suptitle("Teaching vs. forgetting, Qwen3-1.7B on LiveCodeBench (3 seeds pooled): best is bottom right, and the leading group is within noise",
                 x=0.06, y=0.985, ha="left", fontsize=11, weight="bold")
    # the key: one column per group, each recipe with its number and colour as drawn in the panels; numbered by held-out rank
    key = fig.add_axes([0.06, 0.0, 0.92, 0.185]); key.axis("off"); key.set_xlim(0, 1); key.set_ylim(0, 1)
    key.text(0, 1.0, "Recipes, numbered by held-out problems solved (1 = most)", fontsize=8.8, color=INK2, va="top", weight="bold")
    for cx, (c, _) in zip((0.0, 0.40, 0.72), GROUP_NAME):
        for r, a in enumerate([a for a in order if GROUP[a] == c]):
            y = 0.80 - r * 0.145
            key.scatter(cx + 0.012, y, s=52, color=c, edgecolors=SURF, linewidths=1.4, clip_on=False)
            key.text(cx + 0.03, y, f"{num[a]:>2}", fontsize=8.6, weight="bold", color=INK2, va="center", ha="left")
            key.text(cx + 0.055, y, NAME[a], fontsize=8.6, color=INK2, va="center", ha="left")
    save(fig, "fig3_teach_vs_forget.png")


# Figure 4: drift from the original student on untaught prose (review D1)
def fig_prose():
    """Figure 3: drift on untaught prose per recipe, one dot per seed with the seed range as a line; rows sorted by the mean change in loss.
    The three Table 1 recipes and the own-text rows named in the title are bold in the row labels"""
    Dd = D["drift"]; arms = [a for a in NAME if f"{a}|1|prose" in Dd]
    rows = sorted(((np.mean([Dd[f"{a}|{s}|prose"]["dce"] for s in (1, 2, 3)]), a) for a in arms))
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.6), sharey=True); fig.subplots_adjust(top=0.80, bottom=0.14, left=0.21, right=0.98, wspace=0.08)
    for ax, m, t, xl in ((axs[0], "kl", "How far it moved", "KL(original ‖ trained) on prose, nats per token (farther right = moved more)"),
                         (axs[1], "dce", "Which way it moved", "Change in cross-entropy on prose, nats per token (0 = unchanged)")):
        style(ax, "x")
        for i, (_, a) in enumerate(rows):
            v = [Dd[f"{a}|{s}|prose"][m] for s in (1, 2, 3)]
            ax.scatter(v, [i] * 3, s=30, color=GROUP[a], edgecolors=SURF, linewidths=1.2, zorder=3); ax.plot([min(v), max(v)], [i, i], color=GROUP[a], lw=1.2, zorder=2)
        ax.set_title(t, pad=8); ax.set_xlabel(xl, fontsize=8.4)
    axs[1].axvline(0, color=INK2, lw=1.0); axs[0].set_xlim(0, None); axs[0].set_ylim(len(rows) - 0.4, -1.1)
    for x0, xa, xb, t, ha in ((-0.006, -0.197, -0.222, "lower loss than the original", "right"), (0.006, 0.086, 0.111, "higher loss", "left")):   # direction hints; drawn arrows (no arrow glyph in the font)
        axs[1].text(x0, -0.55, t, ha=ha, va="center", fontsize=8.2, color=INK2, style="italic")
        axs[1].annotate("", xy=(xb, -0.55), xytext=(xa, -0.55), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9, mutation_scale=9))
    axs[0].set_yticks(range(len(rows)), [NAME[a] for _, a in rows])
    for lab, (_, a) in zip(axs[0].get_yticklabels(), rows):
        if a in ("sft", "sft_self_s", "sft_c_s", "rb"):
            lab.set_fontweight("bold"); lab.set_color(INK)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", ms=7, color=c, label=l) for c, l in GROUP_NAME]
    fig.legend(handles=h, loc="upper left", ncol=3, bbox_to_anchor=(0.2, 0.935), fontsize=8.8, handletextpad=0.3, columnspacing=1.6, labelcolor=INK2)
    fig.suptitle("Untaught prose: KL to the original barely moves the student; the corrector anchor lowers loss; own-text rows raise it", x=0.06, y=0.985, ha="left", fontsize=11, weight="bold")
    save(fig, "fig4_untaught_prose.png")


# Figure 5: on-policy is capped by its teacher
def fig_teacher():
    """Figure 4: each corrector teacher, run live, against the on-policy student it trained (held problems solved, three seeds pooled); the
    teachers' fit to E4B's held solutions in the group labels; SFT and the untrained student as reference lines"""
    P = D["main"]["pooled"]; T = D["teachers"]; L = D["teachers live"]
    fit = {k: np.mean([T[s][i] for s in T]) for k, i in (("old", 1), ("refit", 2))}; live = {k: sum(L[s][k] for s in L) for k in ("old", "refit", "alone")}
    stu = {"old": P["opd_c"]["held solved"], "refit": P["opd_cf"]["held solved"]}; sft = P["sft"]["held solved"]; alone = D["main"]["per seed"]["1"]["sft"]["qwen held"] * 3
    fig, ax = plt.subplots(figsize=(7.6, 4.2)); style(ax, "y")
    for g, k in enumerate(("old", "refit")):
        for dx, v, c in ((-0.16, live[k], S3), (0.16, stu[k], S1)):
            ax.bar(g + dx, v, width=0.28, color=c, zorder=2); ax.text(g + dx, v + 2.5, f"{v}", ha="center", fontsize=9.5, color=INK, weight="bold", zorder=3)
    ax.axhline(sft, color=INK2, lw=1.2, zorder=1); ax.text(-0.57, sft + 3, f"SFT on teacher text: {sft}", fontsize=8.3, color=INK2, ha="left")
    ax.axhline(alone, color=MUTED, lw=1, zorder=1); ax.text(-0.57, alone + 3, f"untrained\nstudent: {alone}", fontsize=8.3, color=MUTED, ha="left", va="bottom", linespacing=1.1)
    ax.set_xticks([0, 1], [f"Weak teacher\n{fit['old']:.2f} nats per token on E4B's solutions", f"Refit teacher\n{fit['refit']:.2f} nats (a closer fit)"], fontsize=8.8)
    ax.set_xlim(-0.6, 1.65); ax.set_ylim(0, 192); ax.set_ylabel("Held-out problems solved (of 564; higher is better)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=S3, label="the corrector teacher, run live (needs both models)"), Patch(color=S1, label="the on-policy student it trained (a stored LoRA)")],
              loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=8.3, labelcolor=INK2, frameon=False, handlelength=1.2, handleheight=0.8)
    ax.set_title("On-policy distillation reaches its teacher's live level and stops there; only the refit teacher reaches SFT", fontsize=10.5, loc="left", pad=10)
    save(fig, "fig5_teacher_cap.png")


# Figure 6: where losses come from
def fig_losses():
    F = D["fragile"]; H = F["histogram (lost in k of 24 runs)"]; L = D["lengths"]
    fig, axs = plt.subplots(1, 2, figsize=(11.6, 3.9), gridspec_kw={"width_ratios": [1.1, 1], "wspace": 0.55})
    ax = axs[0]; style(ax, "y"); ks = list(range(25)); vs = [H[str(k)] for k in ks]
    ax.bar(ks, vs, width=0.72, color=[GRAY if k < 6 else S2 for k in ks]); ax.set_xlabel("Runs (of 24: 8 recipes × 3 seeds) in which the problem was lost"); ax.set_ylabel("Problems the student solved before")
    ax.text(7.0, max(vs) * 0.72, f"{F['fragile (>= 6)']} fragile problems\n(lost in 6 or more runs)", fontsize=8.5, color=INK2)
    ax.set_title(f"Losses concentrate on a fragile set ({F['qwen solved']} problems)")
    ax = axs[1]; style(ax, "x"); order = sorted(L, key=lambda a: L[a])
    for i, a in enumerate(order):
        ax.barh(i, L[a], height=0.55, color=GRAY if a == "qwen" else GROUP[a]); ax.text(L[a] + 25, i, f"{L[a]:,.0f}", va="center", fontsize=8)
    ax.set_yticks(range(len(order)), ["Student alone" if a == "qwen" else NAME[a] for a in order]); ax.set_xlabel("Median answer length (characters; all 495 answers, seed 1)")
    unf = 100 * F["lost unfenced"] / F["lost cases"]; ml, mq = np.median(F["lost output chars"]), np.median(F["qwen own chars on the lost cases"])
    ax.set_title(f"Answers take the training text's length", loc="left")
    ax.text(0, -0.17, f"Lost answers: median {ml:,.0f} characters, {unf:.0f}% with no closing fence (the 768-token cap); the student's own\npassing answers to the same problems: {mq:,.0f}.",
            transform=ax.transAxes, fontsize=8, color=INK2, va="top")
    save(fig, "fig6_losses.png")


# Figure 7: online learning (E4B -> E2B)
def fig_online():
    """Figure 4 (file fig7_online.png): telepathy-like teaching online, every teacher on one GPU type (206). Left: held cross-entropy to E4B over the
    stream, mean of the seeds run; hue = what teaches (E4B's distribution orange; the corrector fed E4B's states green; fed E2B's own states blue),
    dashes = the corrector fitted 4x longer; the hairline marks the plain teacher's final level. Right: paired differences on the gate's 240 held
    rows with repository-bootstrap intervals (206c): what each ingredient adds."""
    G = D["online g4"]; T, P = G["table"], G["pairs"]
    lab = {"C": "E4B's distribution (plain teacher)", "S": "Corrector fed E2B's own states", "D": "Corrector fed E4B's states (telepathy-like)", "S4": "E2B's own states, fitted 4× longer", "D4": "E4B's states, fitted 4× longer"}
    col = {"C": S2, "S": S1, "D": S3, "S4": S1, "D4": S3}; dash = {"C": "-", "S": "-", "D": "-", "S4": (0, (4, 2.2)), "D4": (0, (4, 2.2))}
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.1), gridspec_kw={"width_ratios": [1.25, 1]}); fig.subplots_adjust(top=0.80, bottom=0.25, left=0.075, right=0.985, wspace=0.42)
    ax = axs[0]; style(ax, "y"); ends = []; nseed = {}
    for k in ("C", "S", "D", "S4", "D4"):
        runs = [v["curve"] for v in T.get(k, {}).values()]
        if not runs:
            continue
        x = [e[0] for e in runs[0]]; y = [float(np.mean([r[i][1] for r in runs])) for i in range(len(x))]; nseed[k] = len(runs)
        ax.plot(x, y, color=col[k], lw=2, ls=dash[k], label=lab[k], solid_capstyle="round", dash_capstyle="round"); ends.append([y[-1], y[-1], k])
        if k == "C":
            cf = y[-1]
    ax.axhline(cf, color=MUTED, lw=0.8, ls=(0, (1, 2)), zorder=1)
    reach = [v.get("item reaching C's final") for k in ("S", "D", "S4", "D4") for v in T.get(k, {}).values() if v.get("item reaching C's final") is not None]
    ax.annotate(f"the plain teacher's final level ({cf:.3f}):\nthe corrector students are there by item {max(reach)}", xy=(max(reach), cf), xytext=(215, cf + 0.021), fontsize=8.2, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkA=0, shrinkB=2), va="bottom")
    lo, hi = ax.get_ylim(); gap = 0.052 * (hi - lo); ends.sort()
    for i in range(1, len(ends)):
        ends[i][1] = max(ends[i][1], ends[i - 1][1] + gap)
    for v, yl_, k in ends:
        ax.text(608, yl_, f"{v:.3f}", fontsize=8.2, va="center", color=INK2, bbox=dict(facecolor=SURF, edgecolor="none", pad=1.2))
    ax.set_xlim(0, 660); ax.set_xticks(range(0, 601, 100)); ax.set_xlabel("Stream items learned (each seen once)")
    ax.set_ylabel("Cross-entropy to E4B, held repositories\n(lower is closer)", fontsize=8.5); ax.set_title("Speed: taught through the corrector, E2B gets close early", fontsize=10)
    hs, ls_ = ax.get_legend_handles_labels(); fig.legend(hs, ls_, loc="lower left", ncol=3, bbox_to_anchor=(0.03, 0.0), fontsize=8.1, labelcolor=INK2, columnspacing=1.4, handlelength=2.4)
    ax = axs[1]; style(ax, "x"); K = "rows: mean, lo, hi, repos closer, repos"; rows = []
    def allseeds(name):
        return P[name]["all seeds"][K] if name in P else None
    for name, text, c in (("D vs C", "Through the corrector,\nnot E4B's distribution", S2), ("D4 vs D", "Corrector fitted 4× longer", INK2), ("cross vs self", "Fed E4B's states,\nnot E2B's own", S3)):
        v = P[name][K] if name == "cross vs self" else allseeds(name)
        if v:
            rows.append((text, v, c, True))
    for name, tag in (("D vs S", "1× fit"), ("D4 vs S4", "4× fit")):
        for sd in ("1", "2", "3", "4", "5"):
            if sd in P.get(name, {}):
                rows.append((f"   {tag}, seed {sd}", P[name][sd][K], S3, False))
    ys = list(range(len(rows)))[::-1]
    for y, (text, v, c, big) in zip(ys, rows):
        ax.plot([v[1], v[2]], [y, y], color=c, lw=2.2 if big else 1.3, solid_capstyle="round", alpha=1 if big else 0.75); ax.plot([v[0]], [y], "o", color=c, ms=7 if big else 4.5, mec=SURF, mew=1.2)
        ax.text(v[1] - 0.0007, y, f"{-v[0]:.4f}" if big else f"{-v[0]:.4f}", ha="right", va="center", fontsize=8.2 if big else 7.6, color=INK2)
    ax.axvline(0, color=MUTED, lw=0.9); ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=8.2); ax.set_ylim(-0.6, len(rows) - 0.4)
    for t, (text, v, c, big) in zip(ax.get_yticklabels(), rows):
        t.set_color(INK if big else INK2)
    xmin = min(r[1][1] for r in rows) - 0.006; ax.set_xlim(xmin, 0.0015); ax.set_xlabel("Nats per token closer to E4B on 240 held rows (paired; bar: 95% interval over repositories)", fontsize=8.2)
    ax.set_xticks([-0.02, -0.015, -0.01, -0.005, 0]); ax.set_xticklabels(["0.020", "0.015", "0.010", "0.005", "0"])
    ax.set_title("What gives the gain", fontsize=10)
    ns = sorted(set(nseed.values())); fig.suptitle("Telepathy-like teaching online, E4B teaching E2B: the corrector's students reach the plain teacher's final closeness in a sixth of the stream;\nmost of the gain comes with the corrector's form, a small part from E4B's states"
                 + (f"  (mean of {ns[0]} seeds)" if len(ns) == 1 else f"  (mean of the seeds run: {', '.join(f'{k} {v}' for k, v in nseed.items())})"), x=0.03, y=0.985, ha="left", fontsize=10.2, weight="bold")
    save(fig, "fig7_online.png")


# Figure 8: agent replies (169d)
def fig_agent():
    A = D["agent169d"]; forms = [("alone", "Student alone"), ("control", "Self-corrected student (live)"), ("bridge", "Cross-model corrected (live)"), ("graft_bridge_w0", "Corrector, distilled into a graft"),
                                 ("step2 seq_d1000 LoRA", "Plain SFT, dose-matched"), ("step2 seqdense_d1000 LoRA", "Dense SFT (whole span)")]
    fig, axs = plt.subplots(1, 2, figsize=(10.2, 3.4), sharey=True)
    for ax, st, t, xl in ((axs[0], "held agent", "Teacher's held agent replies", "Cross-entropy (lower = closer to the teacher)"),
                          (axs[1], "held prose (untaught)", "Untaught human prose", "Cross-entropy (lower = lower loss)")):
        style(ax, "x")
        for i, (k, n) in enumerate(forms):
            v = [A[s][st][k] for s in A if k in A[s][st]]
            if not v:
                continue
            c = S1 if k.startswith("step2") else (S2 if k in ("bridge", "control", "graft_bridge_w0") else GRAY)
            ax.barh(i, np.mean(v), height=0.5, color=c); ax.text(np.mean(v) + 0.03, i, f"{np.mean(v):.2f}", va="center", fontsize=8.5)
        ax.set_title(t); ax.set_xlabel(xl)
    axs[0].set_yticks(range(len(forms)), [f[1] for f in forms]); axs[0].invert_yaxis()
    fig.suptitle("Agent turns, E4B to Qwen3-1.7B: SFT gets closer to the teacher; corrector forms lower loss on untaught prose", x=0.06, ha="left", fontsize=11, weight="bold")
    fig.tight_layout(); save(fig, "fig8_agent_turns.png")


# Figure 9: the same recipes with Gemma 4 E2B as the student (186), next to Qwen3-1.7B; gains and loss shares, since the two students start apart
def fig_e2b():
    if "e2b" not in D:
        print("no E2B data yet: fig9 skipped"); return
    Q, QD = D["main"], D["drift"]   # Qwen: 179-185; E2B: 186 (sft, sft_s, rb), 187 (the anchors, freshly fitted Gemma correctors) and later families
    fams = [k for k in D if (k == "e2b" or k.endswith(" e2b")) and isinstance(D[k], dict) and "per seed" in D[k]]   # "e2b", "two e2b", any "<name> e2b" added later
    PREF = {"sft": "e2b", "sft_s": "e2b", "rb": "e2b", "sft_self_s": "two e2b", "sft_c_s": "two e2b", "sft_b_s": "two e2b"}
    def src(a):
        for k in ([PREF[a]] if a in PREF and PREF[a] in D else []) + fams:
            if any(a in v for v in D[k]["per seed"].values()):
                return D[k], D.get(f"{k} drift", {})
        return None
    # rows in sections; a section shows only once an arm of it has data (the fitting-text arms are still running)
    SECTIONS = [("Copying the teacher's text", ["sft", "sft_s", "rb"]), ("With an anchor", ["sft_self_s", "sft_c_s", "sft_b_s"]), ("With the corrector's fitting text", ["sft_mt_s", "sft_self_mt_s"])]
    sections = [(t, [a for a in arms_ if src(a)]) for t, arms_ in SECTIONS]; sections = [(t, arms_) for t, arms_ in sections if arms_]
    ypos, dividers, headers, y = {}, [], [], 0.0
    for t, arms_ in sections:
        if ypos:
            dividers.append(y - 0.55); y += 0.75
        headers.append((t, y - 0.55))
        for a in arms_:
            ypos[a] = y; y += 1.0
    arms = list(ypos)
    def gain(fam, s, a):
        c = fam["per seed"][s][a]; return c["held solved"] - c["qwen held"]
    def lost(fam, s, a):
        c = fam["per seed"][s][a]; return 100 * (c["lost held"] + c["lost train pool"]) / (c["qwen held"] + c["qwen train pool"])
    def dce(fam_drift, s, a):
        return fam_drift[f"{a}|{s}|prose"]["dce"]
    H = 2.2 + 0.46 * y   # the figure grows with the rows; the title and legend keep a fixed band above the panels
    fig, axs = plt.subplots(1, 3, figsize=(11.6, H), gridspec_kw={"wspace": 0.12}, sharey=True); fig.subplots_adjust(top=1 - 1.05 / H, bottom=0.72 / H, left=0.235, right=0.985)
    panels = ((axs[0], lambda fam, s, a: gain(fam[0], s, a), "Held-out gain over the student alone", "problems per seed (higher is better)"),
              (axs[1], lambda fam, s, a: lost(fam[0], s, a), "Previously solved problems lost", "% of those solved alone, per seed (lower is better)"),
              (axs[2], lambda fam, s, a: dce(fam[1], s, a), "Untaught prose, change in loss", "nats per token (0 = unchanged; right of 0 = worse)"))
    for ax, f, t, xl in panels:
        style(ax, "x")
        for a in arms:
            for j, (fam, c) in enumerate((((Q, QD), GRAY), (src(a), S1))):
                v = []
                for s in fam[0]["per seed"]:
                    try:
                        v.append(f(fam, s, a))
                    except KeyError:
                        pass
                yy = ypos[a] + (0.17 if j else -0.17)
                ax.scatter(v, [yy] * len(v), s=26, color=c, edgecolors=SURF, linewidths=1.0, zorder=3)
                if v:
                    ax.plot([min(v), max(v)], [yy, yy], color=c, lw=1.1, zorder=2)
        for d in dividers:
            ax.axhline(d, color=GRID, lw=1.2, zorder=1)
        ax.set_title(t, fontsize=10); ax.set_xlabel(xl, fontsize=8.5)
    axs[2].axvline(0, color=INK2, lw=1)
    axs[0].set_yticks([ypos[a] for a in arms], [NAME[a] for a in arms]); axs[0].set_ylim(y - 0.5, -0.95)
    import matplotlib.transforms as mtrans
    for t, hy in headers:   # section names over the row labels
        axs[0].text(-0.02, hy, t, transform=mtrans.blended_transform_factory(axs[0].transAxes, axs[0].transData), ha="right", va="center", fontsize=8.4, color=INK, weight="bold")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="o", ls="", ms=7, color=GRAY, label="Qwen3-1.7B student (for reference)"), Line2D([], [], marker="o", ls="", ms=7, color=S1, label="Gemma 4 E2B student, one dot per seed")],
               loc="upper left", ncol=2, bbox_to_anchor=(0.045, 0.945), fontsize=8.8, labelcolor=INK2)
    fig.suptitle("Gemma 4 E2B as the student: copying raises loss on untaught text; anchors, or the corrector's own text, remove the damage at a similar held-out gain", x=0.04, y=0.985, ha="left", fontsize=11, weight="bold")
    save(fig, "fig9_e2b.png")


ONLY = sys.argv[3:]                                                      # optional: the figure functions to draw (all by default)
for f in (fig_method, fig_wide, fig_map, fig_prose, fig_teacher, fig_losses, fig_online, fig_agent, fig_e2b):
    if not ONLY or f.__name__ in ONLY:
        f()
