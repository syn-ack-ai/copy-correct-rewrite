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
    """Figure 1: how the corrector connects the teacher and the student (xbridge.py: source states at 50% and 65% depth, from the teacher for the
    cross-model variant, aligned to the student's tokens, or from the student's own first pass for the self variant; a gated correction written
    at 35% depth and every layer above in a second pass; the corrected student's predictions used as a training target)"""
    from matplotlib.patches import Circle, Rectangle
    fig, ax = plt.subplots(figsize=(11.8, 5.9)); ax.set_xlim(0, 12.4); ax.set_ylim(0, 6.2); ax.axis("off")
    def box(x, y, w, h, text, fc, ec=None, color=INK, size=9, weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec or fc, lw=1.0))
        if text:
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=color, weight=weight)
    def arrow(a, b, color=INK2, rad=0.0, lw=1.4):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, lw=lw, color=color, connectionstyle=f"arc3,rad={rad}"))
    def badge(x, y, n, color):
        ax.add_patch(Circle((x, y), 0.15, fc=color, ec=SURF, lw=1.2, zorder=5)); ax.text(x, y, str(n), ha="center", va="center", fontsize=8.5, color="white", weight="bold", zorder=6)
    Y = lambda k: 1.3 + k * 0.47; H = 0.37; yc = lambda k: Y(k) + H / 2
    TX, TW, CX, CW, SX, SW, UX, UW = 0.2, 2.2, 3.15, 2.35, 6.3, 2.3, 9.3, 2.95
    teacher = ["embeddings", "layers", "layers", "layers", "50% depth", "layers", "65% depth", "layers", "next-token head"]
    student = ["embeddings", "layers", "35% depth", "layers", "50% depth", "layers", "65% depth", "layers", "next-token head"]
    for x, h, sub in ((TX, "Teacher", "Gemma 4 E4B, frozen"), (SX, "Student", "Qwen3-1.7B or Gemma 4 E2B, frozen"), (UX, "Use in training", "only a LoRA on the student is kept")):
        ax.text(x, 5.95, h, fontsize=10.5, weight="bold", color=INK); ax.text(x, 5.7, sub, fontsize=8.3, color=INK2)
    for k, t in enumerate(teacher):
        hl = "%" in t; box(TX, Y(k), TW, H, t, "#fdeee7" if hl else "#f0efec", color=INK if hl else INK2, size=8.3)
    for k, t in enumerate(student):
        box(SX, Y(k), SW, H, t, "#eef4fc" if t[:2] in ("50", "65") else "#f0efec", color=INK if "%" in t else INK2, size=8.3)
        if 2 <= k <= 7:                                                  # the layers that receive the correction
            ax.add_patch(Rectangle((SX - 0.02, Y(k) + 0.03), 0.09, H - 0.06, fc=S3, ec="none", zorder=4))
    box(CX, 2.95, CW, 1.2, "", "#e8f6f0", ec=S3); ax.text(CX + CW / 2, 3.78, "Corrector", ha="center", fontsize=10, weight="bold")
    ax.text(CX + CW / 2, 3.33, "a small gated MLP\none source per variant", ha="center", va="center", fontsize=8.2, color=INK2)
    for k, yy in ((4, 3.62), (6, 3.85)):
        arrow((TX + TW, yc(k)), (CX, yy), S2, rad=-0.1)                  # cross-model variant: the teacher's states
        arrow((SX, yc(k)), (CX + CW, yy), S1, rad=0.1)                    # self variant: the student's own states, first pass
    ax.text(CX + CW / 2, 4.95, "from the teacher (cross-model variant)", ha="center", fontsize=8.3, color=S2, weight="bold")
    ax.text(CX + CW / 2, 4.62, "or from the student itself (self variant)", ha="center", fontsize=8.3, color=S1, weight="bold")
    badge(CX + CW / 2, 5.32, 1, INK2)
    arrow((CX + CW, 3.05), (SX - 0.04, yc(2)), S3, rad=-0.15, lw=2.0)    # second pass: the gated correction
    ax.text(CX + CW / 2, 2.45, "adds a gated correction\nat 35% depth and every\nlayer above (second pass)", ha="center", va="center", fontsize=8.3, color=S3, weight="bold")
    badge(CX + 0.12, 2.45, 2, S3)
    ax.text(UX, 5.25, "The corrected student's predictions are", fontsize=8.6, color=INK2)
    uses = [("Anchor", "a KL pull toward them during SFT"), ("On-policy or online teacher", "they score the student's own samples"),
            ("Rewriter", "solutions rewritten through them"), ("Distilled LoRA", "the LoRA learns to match them")]
    for k, (h, t) in enumerate(uses):
        y = 4.3 - k * 0.92; box(UX, y, UW, 0.78, "", "#f7f7f5", ec=GRID)
        ax.text(UX + 0.14, y + 0.53, h, fontsize=9, weight="bold", va="center"); ax.text(UX + 0.14, y + 0.24, t, fontsize=7.9, color=INK2, va="center")
    ax.plot([UX - 0.14] * 2, [1.62, 4.92], color=MUTED, lw=1.4, solid_capstyle="round")   # the uses, as one group
    arrow((SX + SW, yc(8)), (UX - 0.16, 4.5), INK2, rad=-0.2); badge(SX + SW + 0.42, yc(8) - 0.02, 3, INK2)
    notes = ["1  The corrector reads hidden states at 50% and 65% depth: the teacher's for the cross-model variant (both models read the same text; each student",
             "    token takes the state of the last teacher token that ends at or before it; Gemma students share E4B's tokens), or the student's own for the self variant.",
             "2  The frozen student runs again, and the corrector adds a gated change to its hidden states at 35% depth and every layer above (green edges).",
             "3  The corrected student's predictions become a training target; after training only a LoRA on the student remains, so the corrector is not needed to run it."]
    for k, t in enumerate(notes):
        ax.text(0.2, 0.92 - k * 0.24, t, fontsize=8.2, color=INK2, va="center")
    save(fig, "fig1_corrector.png")


# Figure 2: wide coding (178)
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
    """Figure 7: the four online setups as a 2x2: hue = who teaches (teacher's own distribution, or the corrector), line style = plain or
    keep + replay; the letters match Table 6 and label each line's end"""
    O = D["online"]; lab = {"direct": "A  Teacher", "bridge": "B  Corrector as teacher", "direct_keep_replay": "C  Teacher + keep + replay", "bridge_keep_replay": "D  Corrector + keep + replay"}
    col = {"direct": S2, "bridge": S3, "direct_keep_replay": S2, "bridge_keep_replay": S3}   # as figures 1 and 4: the teacher orange, the corrector green
    dash = {"direct": "-", "bridge": "-", "direct_keep_replay": (0, (4, 2.2)), "bridge_keep_replay": (0, (4, 2.2))}
    fig, axs = plt.subplots(1, 2, figsize=(10.2, 3.9)); fig.subplots_adjust(top=0.80, bottom=0.27, left=0.08, right=0.98, wspace=0.3)
    for ax, k, yl, t in ((axs[0], "held ce (on)", "Cross-entropy to the teacher, held repos\n(lower is better)", "Learning: every corrector teacher (B, D, control) gets closer"),
                         (axs[1], "KL from base on untaught prose / code / story", "KL from the original student, untaught text\n(lower is better)", "Drift: keep + replay (C, D) cuts it and ends lowest")):
        style(ax, "y")
        ends = []
        for a in [a for a in ("direct", "bridge", "direct_keep_replay", "bridge_keep_replay") if a in O]:
            c = O[a]["run"]["held curve (bridge down)"]; x = [e["item"] for e in c]; y = [e[k] for e in c]
            ax.plot(x, y, color=col[a], lw=2, ls=dash[a], label=lab[a], solid_capstyle="round", dash_capstyle="round"); ends.append([y[-1], y[-1], x[-1], lab[a][0]])
        if "control_keep_replay" in O:   # 189: the self-corrector teacher (reads E2B's own states) + keep + replay, gray dashed
            c = O["control_keep_replay"]["run"]["held curve (bridge down)"]; x = [e["item"] for e in c]; y = [e[k] for e in c]
            ax.plot(x, y, color="#8a8a86", lw=1.6, ls=(0, (4, 2.2)), label="Control: self-corrector + keep + replay", dash_capstyle="round"); ends.append([y[-1], y[-1], x[-1], "Ctl"])
        if "bridge_keep_replay_s2" in O:   # 189: D's second seed, same hue, thin dotted, so the replication is visible without a fifth colour
            c = O["bridge_keep_replay_s2"]["run"]["held curve (bridge down)"]; x = [e["item"] for e in c]; y = [e[k] for e in c]
            ax.plot(x, y, color=S3, lw=1.3, ls=(0, (1, 1.6)), label="D, seed 2", dash_capstyle="round"); ends.append([y[-1], y[-1], x[-1], "D2"])
        if "blend_keep_replay" in O:   # 189: the teacher blended with the original E2B to the corrector's pull (w 0.433) + keep + replay; teacher hue, thin dotted
            c = O["blend_keep_replay"]["run"]["held curve (bridge down)"]; x = [e["item"] for e in c]; y = [e[k] for e in c]
            ax.plot(x, y, color=S2, lw=1.3, ls=(0, (1, 1.6)), label="Control: pull-matched blend + keep + replay", dash_capstyle="round"); ends.append([y[-1], y[-1], x[-1], "Bl"])
        lo, hi = ax.get_ylim(); gap = 0.05 * (hi - lo); ends.sort()   # end labels pushed apart so close values stay readable
        for i in range(1, len(ends)):
            ends[i][1] = max(ends[i][1], ends[i - 1][1] + gap)
        for v, yl_, x_, letter in ends:
            ax.text(x_ + 8, yl_, f"{letter}  {v:.3f}" if len(letter) == 1 else f"{letter} {v:.3f}", fontsize=8.2, va="center", color=INK2)
        ax.set_xlabel("Stream items learned (each seen once)"); ax.set_ylabel(yl, fontsize=8.5); ax.set_title(t, fontsize=10); ax.set_xlim(0, 700)
    axs[0].set_xticks(range(0, 601, 100)); axs[1].set_xticks(range(0, 601, 100))
    hs, ls_ = axs[0].get_legend_handles_labels()
    if len(hs) > 5:   # 189: seven lines need two legend rows
        fig.subplots_adjust(bottom=0.31)
    fig.legend(hs, ls_, loc="lower center", ncol=len(hs) if len(hs) <= 5 else 4, bbox_to_anchor=(0.5, 0.0), fontsize=8.4 if len(hs) < 5 else 7.9, labelcolor=INK2, columnspacing=1.8 if len(hs) < 5 else 1.2, handlelength=2.4)
    fig.suptitle("Online learning on agent decisions, E4B teaching E2B (one seed each, D two): corrector teachers get closest; keep + replay cuts the drift", x=0.04, y=0.985, ha="left", fontsize=10.5, weight="bold")
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


for f in (fig_method, fig_wide, fig_map, fig_prose, fig_teacher, fig_losses, fig_online, fig_agent, fig_e2b):
    f()
