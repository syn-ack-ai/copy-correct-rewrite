# Copy, Correct, or Rewrite? What Teaches a Small Model and What Keeps It From Forgetting

*Lessons for training Gemma 4 E2B/E4B developer agents from a stronger teacher: copying verified text teaches best, a learned state corrector lowers loss on general text but does not teach more, and on-policy signals stop at their teacher.*

## Abstract

A small model can learn from a stronger one by copying its verified answers (SFT), by being pulled toward a corrected version of itself (an *anchor*), by on-policy distillation, or by retraining on solutions it writes from the stronger model's plan. These routes were compared under one pre-registered protocol, with Gemma 4 E4B as teacher and Qwen3-1.7B (contest coding, three seeds) and Gemma 4 E2B (online agent decisions) as students. Copying verified answers taught as much as anything tested. A learned *state corrector* did not teach more in any stored form. To keep what the student knew, a plain KL pull toward the original student worked best. On-policy distillation stopped at its teacher's level. Self-written solutions fixed the most trained problems but degraded untaught text.

## Contributions

- **A controlled negative result.** A learned hidden-state corrector, in four stored forms and even when fed the teacher's states, did not teach the student more than plain SFT on the teacher's verified answers. The comparisons were pre-registered, ran on three seeds, and crossed two tokenizers.
- **What keeps prior knowledge.** A KL pull toward the original student matched the best coding results and left untaught text almost unchanged. The corrector anchor did something else: it lowered loss on untaught text, carrying over the corrector's own fit.
- **A dissociation in self-training.** Plan-guided, self-written solutions moved the student least, yet they were the only recipe that sharpened it and raised its loss on untaught text. An anchor reverses this.
- **A teacher-bound control for on-policy distillation.** The same method was run with a weak and then a refit cross-tokenizer teacher. The student rose to the refit teacher's level and stopped there.
- **A public release.** Code, per-problem results for every run, about 24,000 model answers and the pre-registration log: https://github.com/syn-ack-ai/copy-correct-rewrite.

## 1. Question and setup

Submissions to this competition can be built on Gemma 4 E2B or E4B, as a base model plus LoRA adapters (rank ≤ 128), so any gain from a stronger teacher has to survive being stored in an adapter. Distillation from a stronger model is the main lever, but every distillation route risks erasing what the small model already does well. This study asks **which route teaches the most and forgets the least**. A developer agent built on a small model needs both: the coding skill it is trained for, and the reading, instruction following and decision making it already has. Single-shot coding measures the first. Drift on untaught text and the online agent-decision test measure the second.

**Coding testbed.** The teacher is Gemma 4 E4B (the 4-bit QAT checkpoint, dequantised to bf16); the student is Qwen3-1.7B, whose different tokenizer tests whether the corrector carries knowledge across vocabularies. Training rows: 299 older LiveCodeBench problems (May 2023 to Sep 2024), each with E4B's own solution passing every hidden test. Every recipe trains a rank-64 LoRA (alpha 128) on the MLPs of layers 9-27, lr 3e-4 with warm-up and cosine decay, 299 steps of two rows, the same rows in the same order, cross-entropy on the reply; anchored recipes add 1.0 × KL(anchor ‖ student) over the written span.

**Measures** (per seed; single greedy answers, 768-token cap):
- **Fixed:** of the 200 training problems the untrained student fails, how many it now solves.
- **New:** the same for the 163 of 188 newer held-out problems it fails (Sep 2024 to Apr 2025, easy and medium).
- **Held solved:** all 188 held-out problems.
- **Lost:** problems it solved before (25 held-out, 103 in the training pool) that it now fails.
- **Drift:** KL(original ‖ trained) and the change in cross-entropy on held-out human prose, Python code and stories that no recipe or corrector trained on.

**Protocol.** Three seeds; exact sign tests on discordant problems, pooled over seeds (this treats seeds as independent; per-seed values are in the repository). Every comparison and its predictions were written down before it ran (scorecard in Section 6). Re-evaluating a stored adapter in a new session reproduced its per-problem results exactly (0 of 495 differed), so arms are paired across sessions. Retraining is not exact: two runs of plain SFT differed by 6 held solves.

## 2. The state corrector

![The corrector reads the student's own mid-layer states in a first pass and writes a gated correction in a second pass; the corrected student is then used as an anchor, an on-policy teacher or a rewriter](figures/fig1_corrector.png)

*Figure 1.* The device, called a *translator* during development, is named here for what it does: it **corrects the student's states**.

**Structure.** A gated MLP (78M parameters, about 5% of the student; 83M for the cross-model variant). It reads the student's hidden states at 50% and 65% depth from a first pass. In a second pass it writes a gated, additive correction at 35% depth and every layer above. The **self-corrector** reads the student's own states; the **cross-model corrector** reads E4B's, aligned token by token across the two tokenizers.

**Fit.** Cross-entropy of the corrected student on the target text, plus 0.03 × the size of the change, plus 0.5 × a do-no-harm term that keeps the student's own top-32 distribution wherever it already ranks the true token first. Rows: E4B's agent replies (50%), human prose (15%), Python code (15%), multiple-choice questions (10%) and stories (10%). 600 steps at lr 3e-4.

**Use.** Run live, with both models at inference, the corrected student is far closer to the teacher on agent replies than the student alone: 0.67 nats against 1.39 (Figure 3); the self-corrector alone reaches 0.73. Kaggle accepts only adapters, so four *stored* forms were tested: an anchor during SFT, an on-policy teacher, a rewriter, and a LoRA distilled from it with no teacher text.

## 3. Results

### 3.1 Copying the teacher's verified text teaches the most

![Plain SFT solves 519 of 1,739 contest problem-runs, ahead of the corrector's distilled forms](figures/fig2_wide_coding.png)

*Figure 2.* LoRAs trained on E4B's **agent turns**, not on contest code, then tested on LiveCodeBench: seed 1 on all 705 easy and medium problems, seeds 2-3 on 517.
- Every trained form beats the untrained student (p ≤ 5e-7).
- Plain SFT leads the corrector's distilled self form (161 vs 125, p = 0.04) and the cross-model form (181 vs 114, p = 0.0001).
- The cross-model form is no better than the self form (113 vs 144, p = 0.06): E4B's states added nothing a stored adapter kept. Their live advantage over the self-corrector was already small, 0.03-0.06 nats on agent replies, and stored forms kept about a quarter of it.
- SFT with a KL pull toward the *original* student, trained on agent turns, is the weakest form here (414; vs plain SFT 90 vs 195, p = 5e-10). On the coding rows themselves the same pull did well (Section 3.2).

**Agent replies.** On E4B's held agent replies, SFT at the same dose gets closer to the teacher (0.66 nats) than the corrector's distilled graft (0.71), and even the live cross-model corrector (0.67).

![On E4B's held agent replies, SFT gets closer to the teacher; on untaught human prose the corrector forms reach lower loss](figures/fig8_agent_turns.png)

*Figure 3.* Agent replies, E4B to Qwen3-1.7B, three-seed means. Left: cross-entropy on E4B's held agent replies, which measures closeness to the teacher's writing, not retention. Right: untaught human prose. Dense SFT also trained on story text inside its question contexts, so it is not a clean dose match.

**Coding with anchors** (a separate training run, three seeds): held solved 143 for plain SFT, 146 with the cross-model anchor, 152 with the self-anchor. E4B's states added nothing measurable (cross-model vs self, 20 vs 26 held). In the same run SDFT, a 2026 in-context self-distillation method (Shenfeld et al.), lost clearly: 111 held solved, 148 lost (vs SFT 25 vs 57, p = 0.0005).

**Shortest rows.** Keeping the shortest of E4B's passing solutions instead of the first cut the median row from 355 to 295 tokens. It raised held solved from 149 to 160 and cut losses from 58 to 48. Only one seed shows a clear held difference (16-6, then 12-13 and 15-13).

### 3.2 What keeps the student where it was

![Drift on untaught human prose, one dot per seed: how far each recipe moved the student, and whether its loss went down or up](figures/fig4_untaught_prose.png)

*Figure 4.* Two anchors were compared on the shortest rows, at the same dose:
- **KL to the original student:** a pull toward the untrained student itself.
- **The corrector anchor:** a pull toward the self-corrected student.

They do opposite things to untaught text:

| Recipe | KL from the original (prose) | Change in loss: prose / code / stories |
|---|---|---|
| Plain SFT | 0.074 | −0.06 / −0.08 / −0.10 |
| Shortest + KL to the original | **0.010** | +0.01 / −0.10 / −0.02 |
| Shortest + corrector anchor | 0.089 | −0.24 / −0.21 / −0.51 |

(Cross-entropy in nats per token; lower is better.)

- **KL to the original keeps the student where it was.** It moved 6-9× less than the other copying recipes, and prose loss is unchanged.
- **The corrector anchor moves the student as far as SFT, toward much lower loss.** That arrives through the corrector's soft targets on the coding rows alone: the anchor never sees prose, but the corrector was fitted on it. The anchored student also becomes less confident on untaught prose (entropy +0.66 nats, against +0.29 for SFT): lower loss with softer predictions, so better calibrated.

On coding, neither anchor costs teaching (three seeds):

| Recipe | Held solved | Lost (of 384) | Fresh-question accuracy |
|---|---|---|---|
| Shortest rows | 160 | 48 | 0.527 |
| Shortest + KL to the original | 159 | **40** | 0.521 |
| Shortest + corrector anchor | 157 | 52 | 0.554 |
| Plain SFT | 149 | 58 | 0.557 |
| Untrained student | — | — | 0.513 |

KL to the original ties shortest rows on held problems (34 vs 35) and loses the fewest of the copying recipes. The corrector anchor's distinct value appears where training does real damage (Section 3.4).

### 3.3 On-policy distillation stops at its teacher

![The refit teacher lowers cross-entropy on E4B's held solutions from 0.70 to 0.49; the on-policy student rises from 116 to 148 held solves, level with SFT](figures/fig5_teacher_cap.png)

*Figure 5.* On-policy distillation through the self-corrector first reached 116 held solves, against 149 for SFT. Its teacher assigned E4B's solutions 0.70 nats per token, against 1.00 for the student alone. Refitting the corrector on the 299 solutions lowered this to 0.49 on held solutions. Run live, the refit teacher solves 151 held problems.

| Recipe | Weak teacher | Refit teacher |
|---|---|---|
| On-policy only | 116 | 148 |
| SFT + on-policy | 117 | 158 |

The refit arm beat its weak-teacher version in every seed (19-9, 18-9, 19-6) but only tied SFT (38 vs 39) and lost more training problems (59 vs 42). The refit also memorised its rows (0.04 nats): rewrites generated through it were E4B's text with small edits (median similarity 0.98).

### 3.4 The student's own words: best on trained problems, and the anchor's real job

This route adapts the self-rewrite step of Terminal-RSI (arXiv 2610.22192). There, one model writes and consumes its own runbook; here the stronger model writes it.
1. E4B turns each verified solution into a numbered plan in plain English. A screen rejects plans that contain code or share more than 10% of the solution's code 4-grams.
2. The student solves the problem with the plan in its prompt.
3. Passing solutions are trained on *without* the plan.

298 of 299 plans passed the screen. With a plan the student's greedy answer solved 166 problems, against 95 without. 228 rows were kept, covering 137 of the 204 problems the student failed alone. They are the student's own text: median similarity to E4B's solution 0.20, cross-entropy 0.32 nats under the untrained student (1.05 for E4B's rows), median length 110 tokens (350).

| Recipe | Fixed | New | Held solved | Lost (of 384) | Prose ΔCE (per seed) |
|---|---|---|---|---|---|
| Plain SFT | 389 | 90 | 149 | 58 | −0.05 to −0.07 |
| Runbook rows | **421** | 83 | 143 | 39 | **+0.10 to +0.16** |
| Runbook rows + self-anchor | 382 | 71 | 133 | 37 | −0.23 to −0.25 |

Runbook rows generalised less (83 new, against 90-95), but fixed the most training problems of any recipe (vs SFT 133 vs 84; 51-30, 37-30, 45-24 per seed) and lost the fewest training-pool problems (24, against 29-43 for the SFT recipes). They are the only recipe that **raised** loss on untaught text, in every seed and on prose, code and stories, while barely moving the student (KL 0.022). They are also the only one that lowered its entropy there (−0.23 nats on prose): training on its own confident text sharpens it.

The corrector anchor reversed that damage in every seed, to lower loss than the original (−0.23 to −0.25 on prose; entropy +0.64), and raised fresh-question accuracy by 0.027 in every seed. The cost: 39 fewer fixed problems (29 vs 71, p = 3e-5) and fewer held solves (133 vs 143).

![Two panels: held-out solved vs problems lost, and trained problems fixed vs problems lost, for every coding recipe](figures/fig3_teach_vs_forget.png)

*Figure 6.* Every coding recipe on the same three seeds; best is bottom right. Lost counts problems the student solved before (of 384, held plus training pool). The leading group on held-out problems is within noise. Two independent training runs of plain SFT differed by 6 held solves.

### 3.5 What is lost

![Left: how often each of the 128 problems the student solved before was lost across 24 runs; right: median answer length per recipe](figures/fig6_losses.png)

*Figure 7.* Across 24 runs (8 recipes × 3 seeds), 37 of the 128 problems the student solved before were never lost and 37 were lost in six or more. Lost answers have a median of 1,456 characters, against 535 for the student's own passing answers to the same problems; 30% have no closing fence, consistent with the 768-token cap. Answers take on the training text's length, and among text-trained recipes shorter rows lose less, though the weak-teacher on-policy arms write short answers and lose the most (94 and 115). Trained-on problems are not over-represented among losses (SFT lost 14% of solved training problems, 21% of solved held-out ones).

### 3.6 Online learning on agent decisions (E4B to E2B)

![Held cross-entropy to the teacher and drift on untaught text over 600 stream items for three teacher setups](figures/fig7_online.png)

*Figure 8.* E2B with a rank-16 LoRA met 600 unseen agent decision points from a software-engineering harness, each once, scored before it was learned. The target was the teacher's distribution, the cross-model corrector (fitted on the stream's items, so read on held repositories only), or the teacher plus keep + replay (KL to the original E2B on one untaught row per item, and an earlier item again every fourth step).

| Setup | Held CE (from 0.825) | Drift on untaught text | Held-question drift | Gate decisions fixed (of 52) |
|---|---|---|---|---|
| Teacher distribution | 0.787 | 0.089 | 0.030 | 25 |
| Corrector as teacher | **0.753** | 0.055 | 0.043 | 16 |
| Teacher + keep + replay | 0.775 | **0.014** | 0.034 | 14 |

Learning came mostly in the first 150-200 items. Keep + replay cut drift on untaught text 6.6×; held questions, which it never touched, drifted as in the other setups. Agreement on the 55 held decision points ended where it began (0.818) in all three. One seed per setup; two of five predictions failed.

## 4. Related work

**On-policy distillation.** GKD trains a student on its own samples under a same-tokenizer teacher's distribution (Agarwal et al., arXiv 2306.13649). Here a cross-tokenizer teacher is built from the corrector, and the teacher's own fit to the target text bounds the gain.

**Training on own solutions.** ReST (2308.08998), ReST-EM (2312.06585), rejection-sampling fine-tuning (2308.01825) and STaR (2203.14465) retrain on verified own samples. Terminal-RSI (2610.22192) adds plan-guided self-rewriting. The degradation of untaught text reported here is rarely measured.

**Forgetting.** Self-distillation fine-tuning rewrites targets in the model's own words (Yang et al., 2402.13669). In-context self-distillation (Shenfeld et al., 2601.19897) is the SDFT arm here. Related studies: continual fine-tuning (2308.08747), LoRA forgetting (2405.09673), RL's KL-proximity account (2509.04259), and learning without forgetting, the KL-to-original arm.

**Cross-tokenizer transfer.** Logit-space methods (ULD, 2402.12030; DSKD, 2406.17328), model fusion (2401.10491) and model stitching (1411.5908, 2106.07682) align vocabularies or join representations. The corrector instead writes a learned correction into the student's residual stream.

## 5. A recipe for small Gemma models

1. **Train on the teacher's verified answers, using the shortest correct version.**
2. **Anchor to the original student** (a KL pull) to keep what it knows; it matched the best coding results.
3. **Check the teacher before on-policy distillation:** on-policy training reached its teacher at best.
4. **Pair self-written rows with an anchor:** they teach trained problems best but sharpen the model.
5. **Measure forgetting against the original student on human-written text** (KL and change in loss); loss on teacher-written text rewards imitation.

## 6. Limitations and pre-registration scorecard

- **Students.** The coding student is Qwen3-1.7B, chosen to test transfer across tokenizers. E2B appears only in the online test.
- **Noise.** Pooled p-values are descriptive; differences under about 10 held solves are within the observed noise.
- **Corrector.** Its anchor gain on untaught text is a carry-over of its own fit, which included prose; a multi-task baseline with the same text was not run. Anchors used each seed's 600-step corrector; in an earlier one-seed test, LoRAs from longer-trained correctors solved fewer problems and lost more on prose. The live corrector needs both models at inference, which adapter-only rules exclude.

| Addendum (question) | Predictions held | Pre-set bar |
|---|---|---|
| 169d (corrector forms vs dose-matched SFT, agent replies) | 0 of 4 | failed |
| 170 (online learning) | 3 of 5 | — |
| 174 (SFT with anchors, SDFT) | 3 of 5 | E4B's states add nothing |
| 178 (wide coding) | 0 of 1 (reversed) | — |
| 179 (on-policy, shortest rows) | 2 of 3 | — |
| 181 (refit teacher, rewrites) | 9 of 12 | not met |
| 182 (runbook rows) | 8 of 11 | missed by one |
| 184c (runbook rows + self-anchor) | 4 of 6 | not met |
| 185 (SFT + KL to the original) | 5 of 6 | — |

## Reproducibility

Code, per-problem results, model answers and the full pre-registration log (addenda 165-186) are public at https://github.com/syn-ack-ai/copy-correct-rewrite (Apache-2.0): `code/transfer/xbridge.py` (the corrector), `xlearn.py` (coding recipes and evaluation), `online.py` (online test), and `scripts/` (`collect.py`, `drift.py`, `figures.py`). Not included: text corpora, competition data, problem statements and weights.
