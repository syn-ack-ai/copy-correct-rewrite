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
S1, S2, S3, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#c3c2b7"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 9.5,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK, "axes.labelcolor": INK2,
                     "axes.edgecolor": GRID, "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
                     "ytick.labelcolor": INK2, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False})
NAME = {"sft": "SFT on teacher text", "sft_s": "SFT, shortest correct", "sft_c_s": "SFT, shortest + self-anchor", "opd_c": "On-policy (weak teacher)",
        "mix_c": "SFT + on-policy (weak teacher)", "opd_cf": "On-policy (refit teacher)", "mix_cf": "SFT + on-policy (refit teacher)",
        "rw_cf": "Rewrites via corrector", "rb": "Runbook rewrites", "rb_c": "Runbook rewrites + self-anchor", "sft_self_s": "SFT, shortest + KL to original"}
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
    fig, ax = plt.subplots(figsize=(8.6, 3.9)); ax.set_xlim(0, 10); ax.set_ylim(0, 4.6); ax.axis("off")
    def box(x, y, w, h, text, fc, ec=None, color=INK, size=9, weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec or fc, lw=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=color, weight=weight)
    def arrow(a, b, color=INK2, rad=0.0):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10, lw=1.2, color=color, connectionstyle=f"arc3,rad={rad}"))
    ax.text(0.15, 4.35, "Student (frozen), layer stack", fontsize=9, color=INK2)
    labels = ["embeddings", "layers 1-9", "write layer (35% depth)", "layers 11-13", "read layer (50%)", "layers 15-17", "read layer (65%)", "layers 19-27", "next-token head"]
    for k, t in enumerate(labels):
        y = 0.25 + k * 0.44; hl = t.startswith(("read", "write"))
        box(0.15, y, 2.6, 0.36, t, "#eef4fc" if t.startswith("read") else ("#fdeee7" if t.startswith("write") else "#f0efec"), color=INK if hl else INK2, size=8.5)
    box(4.0, 1.85, 2.3, 1.0, "Corrector\n(small gated MLP,\n~78M parameters)", "#e8f6f0", ec=S3, size=9, weight="bold")
    arrow((2.75, 0.25 + 4 * 0.44 + 0.18), (4.0, 2.45), S1, rad=-0.15); arrow((2.75, 0.25 + 6 * 0.44 + 0.18), (4.0, 2.65), S1, rad=-0.2)
    ax.text(2.85, 3.2, "reads its own states (first pass)", fontsize=8, color=S1)
    arrow((4.0, 2.05), (2.75, 0.25 + 2 * 0.44 + 0.18), S2, rad=-0.15); ax.text(2.95, 0.85, "second pass: writes a gated correction\n(write layer and every layer above)", fontsize=8, color=S2)
    box(4.95, 3.45, 1.7, 0.8, "Cross-model variant:\nteacher states (E4B),\naligned across\ntokenizers", "#f0efec", size=7.6)
    arrow((5.8, 3.45), (5.8, 2.85), MUTED)
    ax.text(6.75, 4.2, "The corrected student is then used as", fontsize=9, color=INK2)
    uses = [("Anchor", "KL toward the corrected student\nwhile SFT trains a LoRA"), ("On-policy teacher", "scores every token the\nstudent samples"),
            ("Rewriter", "writes solutions in the\nstudent's own style")]
    for k, (h, t) in enumerate(uses):
        y = 3.2 - k * 1.15; box(6.75, y, 3.1, 0.85, "", "#f7f7f5", ec=GRID)
        ax.text(6.9, y + 0.62, h, fontsize=9, weight="bold", va="center"); ax.text(6.9, y + 0.28, t, fontsize=8, color=INK2, va="center")
    arrow((6.3, 2.35), (6.75, 2.35), INK2)
    save(fig, "fig1_corrector.png")


# Figure 2: wide coding (178)
def fig_wide():
    W = D["wide"]; keys = ["qwen", "lora_self", "lora_b", "lora_c", "lora_sd"]
    lab = {"qwen": "Qwen3-1.7B alone", "lora_self": "SFT + KL to the original student", "lora_b": "Cross-model corrector, distilled", "lora_c": "Self-corrector, distilled", "lora_sd": "Plain SFT on teacher text"}
    tot = {k: sum(W[s]["passed"][k] for s in W) for k in keys}; n = sum(W[s]["problems"] for s in W)
    fig, ax = plt.subplots(figsize=(7.6, 2.9)); style(ax, "x")
    for i, k in enumerate(keys):
        ax.barh(i, tot[k], height=0.55, color=S1 if k == "lora_sd" else GRAY)
        ax.text(tot[k] + 6, i, f"{tot[k]}", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(keys)), [lab[k] for k in keys]); ax.set_xlabel(f"Contest problems solved, 3 seeds pooled (of {n:,} problem-runs)"); ax.set_xlim(0, max(tot.values()) * 1.12)
    ax.set_title("Trained on agent turns, tested on contest problems: plain SFT transfers most")
    save(fig, "fig2_wide_coding.png")


CLUSTER = {"sft", "sft_s", "sft_c_s", "mix_cf", "rw_cf", "sft_self_s"}   # labelled once in the right panel of figure 3


# Figure 3: teaching vs forgetting map
def fig_map():
    P = D["main"]["pooled"]
    fig, axs = plt.subplots(1, 2, figsize=(10.6, 4.6), sharey=True)
    offs = [{"sft": (-14, 22), "sft_s": (16, -22), "sft_c_s": (-30, -26), "mix_cf": (18, 16), "opd_c": (10, -4), "mix_c": (10, 2), "opd_cf": (10, 4), "rw_cf": (-30, 18), "rb": (10, -6), "rb_c": (-12, -14), "sft_self_s": (14, -16)},
            {"sft": (-12, 24), "sft_s": (24, 6), "sft_c_s": (-40, -10), "mix_cf": (-26, -28), "opd_c": (10, -4), "mix_c": (10, 2), "opd_cf": (-12, 14), "rw_cf": (16, 16), "rb": (12, -12), "rb_c": (-14, -14), "sft_self_s": (-40, 2)}]
    for ax, xk, xl, off in ((axs[0], "held solved", "Held-out problems solved (of 564; higher is better)", offs[0]), (axs[1], "fixed", "Trained problems newly solved (of 600; higher is better)", offs[1])):
        style(ax, "both")
        for a, v in P.items():
            y = v["lost held"] + v["lost train pool"]; ax.scatter(v[xk], y, s=64, color=GROUP[a], edgecolors=SURF, linewidths=1.6, zorder=3)
            if xk == "fixed" and a in CLUSTER:
                continue
            o = off.get(a, (8, 4)); ax.annotate(NAME.get(a, a), (v[xk], y), xytext=o, textcoords="offset points", fontsize=7.8, color=INK2, va="center",
                                                ha="right" if o[0] < 0 else "left", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=0, shrinkB=4))
        ax.set_xlabel(xl)
    axs[0].set_xlim(105, 178); axs[1].set_xlim(190, 470)
    axs[1].annotate("Unlabelled points: each sits at the\nsame height as its label on the left", (392, 49), xytext=(255, 60), textcoords="data", fontsize=7.8, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkB=6))
    axs[0].set_ylabel("Problems lost that the student solved before (lower is better)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", ms=7, color=c, label=l) for c, l in ((S1, "Trains on the teacher's text"), (S2, "No teacher text (on-policy / KL only)"), (S3, "Trains on student-written text"))]
    fig.legend(handles=h, loc="upper left", ncol=3, bbox_to_anchor=(0.055, 0.955), fontsize=8.8, handletextpad=0.3, columnspacing=1.6, labelcolor=INK2); axs[0].set_ylim(30, 121)
    fig.suptitle("Teaching vs. forgetting, Qwen3-1.7B on LiveCodeBench (3 seeds pooled; best is bottom right)", x=0.06, y=1.02, ha="left", fontsize=11, weight="bold")
    save(fig, "fig3_teach_vs_forget.png")


# Figure 4: drift from the original student on untaught prose (review D1)
def fig_prose():
    Dd = D["drift"]; arms = [a for a in NAME if f"{a}|1|prose" in Dd]
    rows = sorted(((np.mean([Dd[f"{a}|{s}|prose"]["dce"] for s in (1, 2, 3)]), a) for a in arms))
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.3), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, m, t, xl in ((axs[0], "kl", "How far it moved", "KL(original || trained), nats per token"), (axs[1], "dce", "Which way it moved", "Change in cross-entropy (below 0 = lower loss than the original)")):
        style(ax, "x")
        for i, (_, a) in enumerate(rows):
            v = [Dd[f"{a}|{s}|prose"][m] for s in (1, 2, 3)]
            ax.scatter(v, [i] * 3, s=30, color=GROUP[a], edgecolors=SURF, linewidths=1.2, zorder=3); ax.plot([min(v), max(v)], [i, i], color=GROUP[a], lw=1.2, zorder=2)
        ax.set_title(t, pad=8); ax.set_xlabel(xl, fontsize=8.6)
    axs[1].axvline(0, color=INK2, lw=1.0); axs[0].set_xlim(0, None)
    axs[0].set_yticks(range(len(rows)), [NAME[a] for _, a in rows]); axs[0].invert_yaxis()
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", ms=7, color=c, label=l) for c, l in ((S1, "Trains on the teacher's text"), (S2, "No teacher text (on-policy / KL only)"), (S3, "Trains on student-written text"))]
    fig.legend(handles=h, loc="upper left", ncol=3, bbox_to_anchor=(0.2, 1.005), fontsize=8.8, handletextpad=0.3, columnspacing=1.6, labelcolor=INK2)
    fig.suptitle("Untaught prose: KL to the original barely moves the student; the corrector anchor lowers loss; own-text rows raise it", x=0.06, y=1.07, ha="left", fontsize=11, weight="bold")
    save(fig, "fig4_untaught_prose.png")


# Figure 5: on-policy is capped by its teacher
def fig_teacher():
    P = D["main"]["pooled"]; T = D["teachers"]; old = np.mean([T[s][1] for s in T]); new = np.mean([T[s][2] for s in T]); alone = np.mean([T[s][0] for s in T])
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw={"width_ratios": [1, 1.35]})
    ax = axs[0]; style(ax, "x")
    vals = [("Student alone", alone, GRAY), ("Weak corrector teacher", old, S2), ("Refit corrector teacher", new, S2)]
    for i, (n, v, c) in enumerate(vals):
        ax.barh(i, v, height=0.5, color=c if i else GRAY); ax.text(v + 0.015, i, f"{v:.2f}", va="center", fontsize=9)
    ax.set_yticks(range(3), [v[0] for v in vals]); ax.invert_yaxis(); ax.set_xlim(0, 1.15); ax.set_xlabel("Teacher's cross-entropy on E4B's held solutions (lower is better)")
    ax.set_title("Teacher quality")
    ax = axs[1]; style(ax, "y")
    for a, c, lab in (("opd", S2, "On-policy only"), ("mix", S1, "SFT + on-policy")):
        y0, y1 = P[f"{a}_c"]["held solved"], P[f"{a}_cf"]["held solved"]; ax.plot([0, 1], [y0, y1], color=c, lw=2, marker="o", ms=7, mec=SURF, mew=1.5)
        ax.text(1.05, y1, f"{lab}: {y0} to {y1}", va="center", fontsize=8.5, color=INK2)
    s = P["sft"]["held solved"]; ax.axhline(s, color=INK2, lw=1); ax.text(0.02, s + 1.2, f"SFT on teacher text ({s})", fontsize=8, color=INK2)
    ax.set_xticks([0, 1], ["weak teacher", "refit teacher"]); ax.set_xlim(-0.15, 1.85); ax.set_ylabel("Held-out problems solved (of 564)")
    ax.set_title("Student result")
    fig.suptitle("On-policy distillation climbs to its teacher, not past it", x=0.06, ha="left", fontsize=11, weight="bold"); fig.tight_layout()
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
    O = D["online"]; lab = {"direct": "Teacher distribution", "bridge": "Cross-model corrector as teacher", "direct_keep_replay": "Teacher + keep + replay"}; col = {"direct": S1, "bridge": S2, "direct_keep_replay": S3}
    fig, axs = plt.subplots(1, 2, figsize=(10.2, 3.6))
    for ax, k, yl, t in ((axs[0], "held ce (on)", "Cross-entropy to the teacher, held repos (lower is better)", "Learning: held-out agent decisions"),
                         (axs[1], "KL from base on untaught prose / code / story", "KL from the original student, untaught text (lower is better)", "Drift on untaught text")):
        style(ax, "y")
        for a in ("direct", "bridge", "direct_keep_replay"):
            c = O[a]["run"]["held curve (bridge down)"]; x = [e["item"] for e in c]; y = [e[k] for e in c]
            ax.plot(x, y, color=col[a], lw=2, label=lab[a]); ax.text(x[-1] + 8, y[-1], f"{y[-1]:.3f}", fontsize=8, va="center", color=INK2)
        ax.set_xlabel("Stream items learned (each seen once)"); ax.set_ylabel(yl, fontsize=8.5); ax.set_title(t); ax.set_xlim(0, 660)
    axs[0].legend(loc="upper right", fontsize=8.2, labelcolor=INK2)
    fig.suptitle("Online learning, Gemma 4 E4B to E2B (one seed per setup): keep + replay cuts drift 6.6×", x=0.06, ha="left", fontsize=11, weight="bold")
    fig.tight_layout(); save(fig, "fig7_online.png")


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


# Figure 9: the same recipes with Gemma 4 E2B as the student (186), next to Qwen3-1.7B
def fig_e2b():
    if "e2b" not in D:
        print("no E2B data yet: fig9 skipped"); return
    E, Q = D["e2b"], D["main"]; arms = [a for a in ("sft", "sft_s", "rb") if a in E["pooled"]]
    fig, axs = plt.subplots(1, 3, figsize=(11.6, 3.6), gridspec_kw={"wspace": 0.45})
    for ax, (k, t) in zip(axs[:2], (("held solved", "Held-out problems solved"), ("lost", "Problems lost that the student solved before"))):
        style(ax, "x")
        for i, a in enumerate(arms):
            for j, (fam, lab, c) in enumerate(((Q, "Qwen3-1.7B", GRAY), (E, "Gemma 4 E2B", GROUP[a]))):
                v = [fam["per seed"][s][a]["held solved"] if k == "held solved" else fam["per seed"][s][a]["lost held"] + fam["per seed"][s][a]["lost train pool"] for s in fam["per seed"] if a in fam["per seed"][s]]
                y = i + (0.18 if j else -0.18); ax.scatter(v, [y] * len(v), s=26, color=c, edgecolors=SURF, linewidths=1.0, zorder=3)
                if v:
                    ax.plot([min(v), max(v)], [y, y], color=c, lw=1.1, zorder=2)
        ax.set_yticks(range(len(arms)), [NAME[a] for a in arms]); ax.invert_yaxis(); ax.set_title(t, fontsize=10); ax.set_xlabel("per seed (gray: Qwen3-1.7B)", fontsize=8.5)
    ax = axs[2]; style(ax, "x"); Dd = D.get("e2b drift", {})
    for i, a in enumerate(arms):
        v = [Dd[f"{a}|{s}|prose"]["dce"] for s in (1, 2, 3) if f"{a}|{s}|prose" in Dd]; vq = [D["drift"][f"{a}|{s}|prose"]["dce"] for s in (1, 2, 3) if f"{a}|{s}|prose" in D["drift"]]
        ax.scatter(vq, [i - 0.18] * len(vq), s=26, color=GRAY, edgecolors=SURF, linewidths=1.0, zorder=3); ax.scatter(v, [i + 0.18] * len(v), s=26, color=GROUP[a], edgecolors=SURF, linewidths=1.0, zorder=3)
    ax.axvline(0, color=INK2, lw=1); ax.set_yticks(range(len(arms)), ["" for _ in arms]); ax.invert_yaxis(); ax.set_title("Untaught prose, change in CE", fontsize=10); ax.set_xlabel("below 0 = lower loss than the original", fontsize=8.5)
    fig.suptitle("The same recipes with a Gemma student: Gemma 4 E4B teaching Gemma 4 E2B (colour) vs Qwen3-1.7B (gray)", x=0.06, y=1.03, ha="left", fontsize=11, weight="bold")
    save(fig, "fig9_e2b.png")


for f in (fig_method, fig_wide, fig_map, fig_prose, fig_teacher, fig_losses, fig_online, fig_agent, fig_e2b):
    f()
