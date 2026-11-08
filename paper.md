# Copy, Correct, or Rewrite? What Teaches a Small Model and What Keeps It From Forgetting

Benjamin Babeshkin

*Lessons for training Gemma 4 E2B/E4B developer agents from a stronger teacher.*

## Abstract

A small agent model trained by a stronger one must gain the teacher's skill without losing its own. Under one pre-registered protocol we compare copying the teacher's verified answers (SFT), anchoring the student, on-policy distillation, and retraining on the student's own plan-guided solutions. A learned *state corrector*, writing into the student's residual stream from hidden states, serves as anchor, on-policy teacher and rewriter. Gemma 4 E4B teaches Qwen3-1.7B and Gemma 4 E2B on contest coding (three seeds) and E2B on online agent decisions. Copying verified answers taught as much as anything, and no stored form of the corrector clearly beat it on coding. A KL pull toward the original student kept untaught text almost unchanged at no cost in teaching. On-policy distillation stopped at its teacher's level, and self-written solutions sharpened the student. Copying made the Gemma student forget far more, and every anchor removed that damage; the corrector anchor turned it into a gain. As an online teacher with keep + replay, the corrector gave the best net result on agent decisions in one seed.

## 1 Introduction

Submissions to the Gemma 4 Developer Agent competition (Kaggle, 2026) are Gemma 4 E2B or E4B (Google, 2026) plus LoRA adapters (Hu et al., 2021) of rank at most 128, so any gain from a stronger teacher must survive being stored in an adapter. Distillation (Hinton et al., 2015) is the main lever, but every route risks erasing what the small model already does well: a developer agent needs new coding skill and its existing reading, instruction following and decision making. We ask which route teaches the most and forgets the least.

Our starting point was Swarm Telepathy (Alabdulkareem et al., 2026), which maps one model's KV cache into a frozen receiver of another family. We built a related *state corrector* that maps hidden states into a student and writes a correction into its residual stream, and asked whether its effect can be stored in an adapter. Our contributions:

- A controlled negative result: no stored form of the corrector taught coding better than SFT.
- A KL pull toward the original student kept prior knowledge at no cost in teaching.
- Self-written solutions moved the student least yet sharpened it; an anchor reverses this.
- On-policy distillation rose to its teacher's level and stopped there.
- A same-family student (Gemma 4 E2B) forgot far more under copying, and any anchor prevented it.
- A first stored gain for the corrector, as an online teacher (one seed).
- Code, results, over 40,000 graded answers and the pre-registration log (Appendix A).

## 2 Background

A teacher row is a prompt x and a verified reply y. SFT minimizes the cross-entropy of y; an anchored recipe adds a pull toward a reference distribution q over the reply:

L = CE(y | x) + λ · KL(q ‖ p_θ), with λ = 1.  (1)

With q the untrained student, eq. (1) is learning without forgetting (Li and Hoiem, 2016); with q the corrected student, it is the *corrector anchor*. On-policy distillation (Agarwal et al., 2023) samples ŷ from the student and minimizes the reverse KL to a teacher t on those tokens:

L = Σ_i KL(p_θ ‖ t) over the positions i of ŷ.  (2)

## 3 Method

![How the corrector connects teacher and student](figures/fig1_corrector.png)

Figure 1. How the corrector connects teacher and student, and how its output trains the student.

**Structure.** A gated MLP (78M parameters for Qwen, 83M cross-model) reads the student's hidden states at 50% and 65% depth in a first pass. In a second pass it writes a gated, additive correction at 35% depth and every layer above. The *self-corrector* reads the student's own states; the *cross-model corrector* reads E4B's, aligned token by token across tokenizers.

**Fit.** We fit it for 600 steps at lr 3e-4 on

L_fit = CE_corrected + 0.03 · ‖Δ‖ + 0.5 · H,  (3)

with ‖Δ‖ the size of the change and H a do-no-harm term that keeps the student's own top-32 distribution wherever it already ranks the true token first. Rows: E4B's agent replies (50%), human prose (15%), Python code (15%), multiple-choice questions (10%) and stories (10%).

**Use.** Run live, with both models at inference, the corrected student is far closer to the teacher on agent replies than the student alone (0.67 nats against 1.39; self-corrector 0.73). We test four stored forms: an anchor in eq. (1), an on-policy teacher in eq. (2), a rewriter, and a LoRA distilled from it with no teacher text.

## 4 Experiments

### 4.1 Setup

**Models.** The teacher is Gemma 4 E4B (4-bit QAT, dequantised to bf16). The coding students are Qwen3-1.7B (Yang et al., 2025), whose different tokenizer tests transfer across vocabularies, and Gemma 4 E2B.

**Training data.** 299 older LiveCodeBench problems (Jain et al., 2024; May 2023 to Sep 2024), each with E4B's own solution passing every hidden test.

**Optimization.** A rank-64 LoRA (alpha 128) on the MLPs of layers 9-27, lr 3e-4 with warm-up and cosine decay, 299 steps of two rows, the same rows and order for every recipe.

**Evaluation.** Greedy answers with a 768-token cap. *Fixed*: of the 200 training problems the untrained student fails, how many it now solves. *New*: the same for the 163 of 188 newer held-out problems (Sep 2024 to Apr 2025, easy and medium) it fails. *Held solved*: all 188. *Lost*: previously solved problems (25 held-out, 103 in the training pool) now failed. *Drift*: KL(original ‖ trained) and the change in cross-entropy on held-out human prose, Python code and stories that nothing trained on. Counts are pooled over three seeds, with exact sign tests on discordant problems, and every comparison was pre-registered with predictions (Appendix C). Re-evaluation reproduced per-problem results exactly (0 of 495 differed), so arms pair across sessions; retraining is not exact (two plain-SFT runs differed by 6 held solves).

### 4.2 Copying the teacher's verified text teaches the most

![Agent-turn LoRAs on LiveCodeBench](figures/fig2_wide_coding.png)

Figure 2. LoRAs trained on E4B's agent turns, tested on LiveCodeBench; plain SFT solves 519 of 1,739 problem-runs (705 problems in seed 1, 517 in seeds 2-3).

Every trained form beats the untrained student (p < 1e-6), and plain SFT leads the corrector's distilled self and cross-model forms (161 vs 125 problems solved by one form only, p = 0.04; 181 vs 114, p = 0.0001). E4B's states added nothing a stored adapter kept (cross-model vs self, 113 vs 144); their live advantage was only 0.05-0.06 nats on agent replies. SFT with a KL pull toward the original is weakest here (vs SFT 90 vs 195, p = 5e-10), though it did well on the coding rows (Section 4.3).

In a separate coding run the anchors did not lose to SFT (held solved: SFT 143, cross-model anchor 146, self-anchor 152; cross-model vs self 20 vs 26), and SDFT, in-context self-distillation (Shenfeld et al., 2026), lost clearly (111 held solved, 148 lost; vs SFT 25 vs 57, p = 0.0005). Keeping the shortest passing solutions (median 295 tokens, not 355) raised held solved from 149 to 160 and cut losses from 58 to 48 (clear in one seed).

### 4.3 What keeps the student where it was

We compare two anchors on the shortest rows at the same dose (Table 1, Figure 3). KL to the original moved the student 6-9× less than the other copying recipes and left prose loss unchanged. The corrector anchor moved it as far as SFT, toward much lower loss and softer predictions (entropy +0.66 nats on prose, against +0.29 for SFT). It never sees prose, so the gain comes through the corrector, which was fitted on prose.

Table 1. Drift from the original Qwen on untaught text, three-seed means.

| Recipe | KL from the original (prose) | Change in loss (nats): prose / code / stories |
|---|---|---|
| Plain SFT | 0.074 | −0.06 / −0.08 / −0.10 |
| Shortest + KL to the original | **0.010** | +0.01 / −0.10 / −0.02 |
| Shortest + corrector anchor | 0.089 | −0.24 / −0.21 / −0.51 |

![Drift on untaught prose per seed](figures/fig4_untaught_prose.png)

Figure 3. Drift on untaught human prose, one point per seed.

Neither anchor costs teaching (Table 2); the corrector anchor's value shows where training does real damage (Sections 4.5 and 4.6).

Table 2. Coding with the two anchors, three seeds pooled (held of 564).

| Recipe | Held solved | Lost (of 384) | Fresh-question accuracy |
|---|---|---|---|
| Shortest rows | 160 | 48 | 0.527 |
| Shortest + KL to the original | 159 | **40** | 0.521 |
| Shortest + corrector anchor | 157 | 52 | 0.554 |
| Plain SFT | 149 | 58 | 0.557 |
| Untrained student | — | — | 0.513 |

### 4.4 On-policy distillation stops at its teacher

On-policy distillation through the self-corrector first reached 116 held solves (SFT 149). Its teacher assigned E4B's solutions 0.70 nats per token (student alone 1.00); refit on the 299 solutions, 0.49, and live it solves 151 held problems. The student then rose to that level (Table 3, Figure 4): the refit arm beat its weak-teacher version in every seed but only tied SFT (38 vs 39) and lost more training problems (59 vs 42). Rewrites through the refit corrector were E4B's text (similarity 0.98).

Table 3. Held solved, three seeds pooled.

| Recipe | Weak teacher | Refit teacher |
|---|---|---|
| On-policy only | 116 | 148 |
| SFT + on-policy | 117 | 158 |

![Teacher fit against on-policy held solves](figures/fig5_teacher_cap.png)

Figure 4. The on-policy student against its teacher's fit.

### 4.5 The student's own words

Adapting the self-rewrite step of Terminal-RSI (Yang et al., 2026), E4B turns each verified solution into a numbered plain-English plan (a screen rejects plans with code or over 10% of the solution's code 4-grams), the student solves the problem with the plan in its prompt, and passing solutions are trained on without the plan. 298 of 299 plans passed, and with a plan the student solved 166 problems greedily (95 without). The 228 rows kept are the student's own text: similarity to E4B's solution 0.20, 0.32 nats under the untrained student (E4B's rows 1.05), 110 tokens (350).

Table 4. Runbook rows, three seeds pooled; prose ΔCE per seed.

| Recipe | Fixed | New | Held solved | Lost (of 384) | Prose ΔCE (per seed) |
|---|---|---|---|---|---|
| Plain SFT | 389 | 90 | 149 | 58 | −0.05 to −0.07 |
| Runbook rows | **421** | 83 | 143 | 39 | **+0.10 to +0.16** |
| Runbook rows + self-anchor | 382 | 71 | 133 | 37 | −0.23 to −0.25 |

Runbook rows generalised less (83 new, against 90-95) but fixed the most training problems (vs SFT 133 vs 84, in every seed) and lost the fewest training-pool problems (24, as did its anchored version; others 29-43). They are the only Qwen recipe that raised loss on untaught text, in every seed and kind, while barely moving the student (KL 0.022), and lowered its entropy the most (−0.23 nats on prose): training on its own confident text sharpens the student. The corrector anchor reversed that in every seed, to lower loss than the original (−0.23 to −0.25 on prose; entropy +0.64), and raised fresh-question accuracy by 0.028, at the cost of 39 fewer fixed problems (29 vs 71, p = 3e-5) and fewer held solves (133 vs 143).

![Teaching against forgetting per recipe](figures/fig3_teach_vs_forget.png)

Figure 5. Every coding recipe, three seeds; best is bottom right. The leading group on held problems is within noise.

Across 24 Qwen runs, lost answers ran long (median 1,456 characters against 535), 30% cut at the token cap.

### 4.6 The same recipes with a Gemma student

Gemma 4 E2B replaced Qwen, with everything else the same. E2B starts stronger (57 held problems solved alone, against 25) and finds E4B's solutions nearly its own (0.27 nats per token, against 1.05). Gains were smaller, about 11 held problems per seed against 19-30 for Qwen. Runbook rows lost their advantage (fixed 61 vs 111 for SFT, p = 0.0002), since E2B's rewrites were no closer to it than E4B's text. Every copying recipe damaged untaught text (Table 5, Figure 6): 4-13× Qwen's KL, sharper predictions (entropy −0.3 to −0.8) and higher loss on prose, code and stories, though a smaller share of solved coding problems was lost.

Table 5. Gemma 4 E2B as the student, means per seed over three seeds (E2B alone solves 57 held problems).

| Recipe | Held solved | Fixed (of 151) | Lost (of 205) | Prose KL | Prose Δ loss |
|---|---|---|---|---|---|
| SFT | 68.0 | 119.7 | 15.7 | 0.29 | +0.69 |
| Shortest rows | 68.0 | 123.0 | 15.7 | 0.29 | +0.57 |
| Runbook rows | 67.0 | 106.0 | 19.3 | 0.29 | +0.85 |
| Shortest + KL to the original | 69.0 | 90.7 | **13.3** | **0.016** | −0.04 |
| Shortest + self-corrector anchor | **72.0** | 95.3 | 16.0 | 0.33 | **−0.60** |
| Shortest + cross-model anchor | 69.7 | 99.3 | 15.3 | 0.32 | −0.58 |

The likely cause is over-fitting familiar text. E2B's training loss on E4B's rows started low (0.28 nats), barely fell in the first epoch and reached 0.07 in the second. Qwen went from 0.62 to 0.24 and softened; its one sharpening recipe, runbook rows, also started familiar (0.25 to 0.03). Yet E2B moved about 13× more than that arm at a similar final loss, so part of the effect is the student.

With fresh Gemma correctors per seed, every anchor removed the damage in every seed and held E2B's training loss at 0.11-0.12, not 0.08. KL to the original kept E2B in place (prose KL 0.016, Δ loss −0.04); both corrector anchors moved it as far as SFT but toward lower loss and softer predictions (Δ loss −0.58 to −0.60, entropy +0.8). Held solves rose by at most 12 over three seeds (self-corrector anchor 216 vs SFT 204, p = 0.22), at about 25 fewer fixes per seed. On E4B's coding rows neither corrector brought E2B closer to E4B's text (0.27-0.30 nats, as E2B alone), though each cut loss on its own fitting mix by 27-29% (Qwen's by 34-37%); E2B's states where the corrector writes are also more uneven (spread about 45, against Qwen's 18-19). Reversed, with E2B's 148 passing answers teaching E4B, SFT barely helped (224 held solved against 207 alone) and damaged E4B's prose most of all (+0.81). Every anchor did significantly better on held problems (244-250; self-corrector vs SFT 52 vs 26, p = 0.004), lost fewer (11-13 per seed against 20) and removed the damage. The cross-model and self anchors tied in both directions (209 vs 216; 249 vs 250). A matched-loss run and a multi-task control were still running.

![Qwen and E2B per recipe](figures/fig9_e2b.png)

Figure 6. Qwen3-1.7B (gray) and Gemma 4 E2B (colour) per recipe; anchors below the line.

### 4.7 Online learning on agent decisions

E2B with a rank-16 LoRA met 600 unseen agent decision points from the competition's harness, each once, scored before learning it. The target was the teacher's distribution (A) or the cross-model corrector (B; fitted on the stream, so read on held repositories only), each also with keep + replay (C, D: KL to the original E2B on one untaught row per item, and an earlier item every fourth step). The result is a plain LoRA; the corrector is needed only during training.

Table 6. Online learning, one seed per setup. Gate (220 held decisions): fixed of the 52 where E2B alone differs from the teacher, broken of the other 168.

| Setup | Held CE (from 0.825) | Untaught drift | Held-question drift | Gate: fixed / broken |
|---|---|---|---|---|
| A Teacher | 0.787 | 0.089 | 0.030 | 25 / 11 |
| B Corrector | 0.753 | 0.055 | 0.043 | 16 / 4 |
| C Teacher + keep + replay | 0.775 | **0.014** | 0.034 | 14 / 6 |
| D Corrector + keep + replay | **0.751** | 0.017 | 0.041 | 23 / **4** |

Keep + replay hurt the plain teacher (A to C: 11 fewer fixed) but helped the corrector teacher (B to D: 7 more). D had the best net gate result (+19, against +14 for A) and beat C on paired decisions (17 vs 6, exact sign test p = 0.035); its untaught drift peaked at 0.05 near item 450 before falling. Seeds 2-3 of C and D were still running, with two controls: a self-corrector teacher and a teacher-student blend matched to the corrector's pull, akin to a teacher assistant (Mirzadeh et al., 2019).

![Online learning curves](figures/fig7_online.png)

Figure 7. Held cross-entropy to the teacher and drift on untaught text over the stream.

## 5 Discussion

**Where the corrector helps.** Its value showed up where it shapes targets during learning: as an anchor that removed E2B's damage, as an online teacher, and live, while E4B's states added nothing stored. Eight held problems that neither E4B nor the untrained student solved were solved by a trained student in all three seeds; for six, E4B's answer was cut at the token cap, and two look genuine. This points to live two-way composition, a generalist and a small specialist correcting each other's states, extending CALM (Bansal et al., 2024).

**Limitations.** Two coding students and one teacher; the online test is one seed so far. Pooled p-values are descriptive, and differences under about 10 held solves are within the observed noise. The corrector anchor's gain on untaught text carries over its own fit, which included prose; a multi-task baseline with the same text was still running. Anchors used 600-step correctors; in an earlier one-seed test, LoRAs from longer-trained correctors mostly did worse. The live corrector needs both models at inference, which adapter-only rules exclude.

**Ethical considerations.** The prose and stories used for fitting and drift are from locally held books and are not released. Model-written code ran only in resource-limited, sandboxed interpreters.

## 6 Related Work

**On-policy distillation.** GKD (Agarwal et al., 2023) trains a student on its own samples under a same-tokenizer teacher; our teacher is built from the corrector, whose fit bounds the gain.

**Training on own solutions.** ReST (Gulcehre et al., 2023), ReST-EM (Singh et al., 2023), rejection-sampling fine-tuning (Yuan et al., 2023) and STaR (Zelikman et al., 2022) retrain on verified own samples; Terminal-RSI (Yang et al., 2026) adds plan-guided self-rewriting. The damage to untaught text we report is rarely measured.

**Forgetting.** Self-distillation fine-tuning (Yang et al., 2024) rewrites targets in the model's own words; in-context self-distillation (Shenfeld et al., 2026) is our SDFT arm. See also LoRA forgetting (Biderman et al., 2024) and RL's KL-proximity account (Shenfeld et al., 2025).

**Cross-model transfer.** Logit-space methods align vocabularies (Boizard et al., 2024; Zhang et al., 2024), model fusion merges distributions (Wan et al., 2024), and model stitching joins representations (Lenc and Vedaldi, 2014; Bansal et al., 2021). Swarm Telepathy (Alabdulkareem et al., 2026) moves a KV cache into a frozen receiver across families.

## 7 Conclusion

For small Gemma models we recommend the teacher's shortest verified answers with an anchor, a KL pull toward the original student or a corrector anchor, which for E2B removed copying's damage; checking the teacher's fit before on-policy distillation; and measuring forgetting against the original student on human-written text. The teacher's states added nothing stored, but the corrector helped as an anchor and as an online teacher; more seeds and matched controls will settle how much.

## References

R. Agarwal, N. Vieillard, Y. Zhou, P. Stanczyk, S. Ramos, et al. (2023). On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes. arXiv:2306.13649. https://arxiv.org/abs/2306.13649

A. Alabdulkareem, I. Tenison, Y. Zhang, Y. Zhu, B. Katz, and D. Gutfreund (2026). Swarm Telepathy: Prefill-Free KV-Cache Transfer Across Families. arXiv:2610.15095. https://arxiv.org/abs/2610.15095

Y. Bansal, P. Nakkiran, and B. Barak (2021). Revisiting Model Stitching to Compare Neural Representations. arXiv:2106.07682. https://arxiv.org/abs/2106.07682

R. Bansal, B. Samanta, S. Dalmia, N. Gupta, S. Vashishth, et al. (2024). LLM Augmented LLMs: Expanding Capabilities through Composition. arXiv:2401.02412. https://arxiv.org/abs/2401.02412

D. Biderman, J. Portes, J. J. Gonzalez Ortiz, M. Paul, P. Greengard, et al. (2024). LoRA Learns Less and Forgets Less. arXiv:2405.09673. https://arxiv.org/abs/2405.09673

N. Boizard, K. El Haddad, C. Hudelot, and P. Colombo (2024). Towards Cross-Tokenizer Distillation: the Universal Logit Distillation Loss for LLMs. arXiv:2402.12030. https://arxiv.org/abs/2402.12030

Google (2026). Gemma 4 E2B and E4B, instruction-tuned QAT checkpoints (w4a16). Hugging Face model cards google/gemma-4-E2B-it-qat-w4a16-ct (revision 971342c) and google/gemma-4-E4B-it-qat-w4a16-ct (revision 6cd26aa).

C. Gulcehre, T. L. Paine, S. Srinivasan, K. Konyushkova, L. Weerts, et al. (2023). Reinforced Self-Training (ReST) for Language Modeling. arXiv:2308.08998. https://arxiv.org/abs/2308.08998

G. Hinton, O. Vinyals, and J. Dean (2015). Distilling the Knowledge in a Neural Network. arXiv:1503.02531. https://arxiv.org/abs/1503.02531

E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685. https://arxiv.org/abs/2106.09685

N. Jain, K. Han, A. Gu, W.-D. Li, F. Yan, et al. (2024). LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code. arXiv:2403.07974. https://arxiv.org/abs/2403.07974

Kaggle (2026). Gemma 4 Developer Agent competition. https://www.kaggle.com/competitions/gemma-4-developer-agent

K. Lenc and A. Vedaldi (2014). Understanding image representations by measuring their equivariance and equivalence. arXiv:1411.5908. https://arxiv.org/abs/1411.5908

Z. Li and D. Hoiem (2016). Learning without Forgetting. arXiv:1606.09282. https://arxiv.org/abs/1606.09282

S.-I. Mirzadeh, M. Farajtabar, A. Li, N. Levine, A. Matsukawa, and H. Ghasemzadeh (2019). Improved Knowledge Distillation via Teacher Assistant. arXiv:1902.03393. https://arxiv.org/abs/1902.03393

I. Shenfeld, J. Pari, and P. Agrawal (2025). RL's Razor: Why Online Reinforcement Learning Forgets Less. arXiv:2509.04259. https://arxiv.org/abs/2509.04259

I. Shenfeld, M. Damani, J. Hübotter, and P. Agrawal (2026). Self-Distillation Enables Continual Learning. arXiv:2601.19897. https://arxiv.org/abs/2601.19897

A. Singh, J. D. Co-Reyes, R. Agarwal, A. Anand, P. Patil, et al. (2023). Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models. arXiv:2312.06585. https://arxiv.org/abs/2312.06585

F. Wan, X. Huang, D. Cai, X. Quan, W. Bi, and S. Shi (2024). Knowledge Fusion of Large Language Models. arXiv:2401.10491. https://arxiv.org/abs/2401.10491

Z. Yang, T. Pang, H. Feng, H. Wang, W. Chen, et al. (2024). Self-Distillation Bridges Distribution Gap in Language Model Fine-Tuning. arXiv:2402.13669. https://arxiv.org/abs/2402.13669

A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, et al. (2025). Qwen3 Technical Report. arXiv:2505.09388. https://arxiv.org/abs/2505.09388

J. Yang, Z. Li, Y. Shi, Z. Li, R. Wang, et al. (2026). Terminal-RSI: Recursive Self-Improving Terminal Agents via Recursive Self-Rewrite and Policy-Refine. arXiv:2610.22192. https://arxiv.org/abs/2610.22192

Z. Yuan, H. Yuan, C. Li, G. Dong, K. Lu, et al. (2023). Scaling Relationship on Learning Mathematical Reasoning with Large Language Models. arXiv:2308.01825. https://arxiv.org/abs/2308.01825

E. Zelikman, Y. Wu, J. Mu, and N. D. Goodman (2022). STaR: Bootstrapping Reasoning With Reasoning. arXiv:2203.14465. https://arxiv.org/abs/2203.14465

S. Zhang, X. Zhang, Z. Sun, Y. Chen, and J. Xu (2024). Dual-Space Knowledge Distillation for Large Language Models. arXiv:2406.17328. https://arxiv.org/abs/2406.17328

## Appendix A Code

Code, per-problem results, over 40,000 graded model answers and the pre-registration log (addenda 165-190) are public at https://github.com/syn-ack-ai/copy-correct-rewrite (Apache-2.0); the corrector is `code/transfer/xbridge.py`. Not included: text corpora, competition data, problem statements and weights.

## Appendix B AI use

The author carried out this study with Claude Opus 5.5 (Anthropic) as an AI research assistant. The author set the research direction, chose the questions and approved each pre-registered experiment; Claude Opus 5.5 designed and ran the experiments, wrote the code and drafted the text. Separate Claude instances reviewed new code and drafts, and verified the numbers.

## Appendix C Pre-registration scorecard

Table 7. Predictions that held, per pre-registered addendum.

| Addendum (question) | Predictions held | Pre-set bar |
|---|---|---|
| 169d (corrector forms vs dose-matched SFT, agent replies) | 0 of 4 | failed |
| 170 (online learning) | 3 of 5 | — |
| 174 (SFT with anchors, SDFT) | 3 of 5 | not met; states added nothing on held |
| 178 (wide coding) | 0 of 1 (reversed) | — |
| 179 (on-policy, shortest rows) | 2 of 3 | — |
| 181 (refit teacher, rewrites) | 9 of 12 | not met |
| 182 (runbook rows) | 8 of 11 | missed by one |
| 184c (runbook rows + self-anchor) | 4 of 6 | not met |
| 185 (SFT + KL to the original) | 5 of 6 | — |
| 186 (Gemma 4 E2B as the coding student) | 5 of 7 | — |
| 187 (two-way, fresh Gemma correctors) | 5 of 7 | not met; cross-model = self |
| 189 (online setup D: re-gates reproduce) | 1 of 1 so far | running |
