# Copy, Correct, or Rewrite? What Teaches a Small Model and What Keeps It From Forgetting

Benjamin Babeshkin

*Lessons for training Gemma 4 E2B/E4B developer agents from a stronger teacher.*

## Abstract

A small agent model trained by a stronger one must gain the teacher's skill without losing its own. Under one pre-registered protocol, Gemma 4 E4B teaches Qwen3-1.7B and Gemma 4 E2B on contest coding, and E2B on a stream of agent decisions. We compare copying the teacher's verified answers (SFT), SFT with a pull toward a reference model (anchoring), on-policy distillation and retraining on the student's own solutions. Copying taught as much as any route, best with the shortest answers. With E2B as the student, copying also caused forgetting: E4B's text is familiar to E2B, and training on it made E2B more confident but worse at predicting text it never trained on. A KL pull toward the original student removed that damage without costing held-out solves. A state corrector, a small network that corrects the student's hidden states, did not store the teacher's coding knowledge in an adapter: fed E4B's states, it taught no better than fed the student's own. As an anchor, its gain came from the text it was fitted on; as an online teacher with keep and replay, it fixed nearly as many held decisions as the plain teacher (23 vs 25) and broke fewer (4-6 vs 11), in two seeds; neither a corrector reading the student's own states nor a pull-matched teacher blend did.

## 1 Introduction

Submissions to the Gemma 4 Developer Agent competition (Kaggle, 2026) are Gemma 4 E2B or E4B (Google, 2026) plus LoRA adapters (Hu et al., 2021) of rank at most 128, so any gain from a stronger teacher must be stored in an adapter. Distillation (Hinton et al., 2015) is the main lever; we ask which route teaches the most and forgets the least.

To test whether a teacher's internal states can be stored, we built a *state corrector*, inspired by Swarm Telepathy (Alabdulkareem et al., 2026), which moves one model's KV cache into a frozen receiver of another family. Ours writes a correction into the student's hidden states. Our contributions:

- Copying verified answers teaches as much as any route, best with the shortest.
- Same-family copying causes forgetting; a KL pull toward the original student removes it at no cost.
- Self-written solutions move the student least yet make it overconfident; on-policy distillation stops at its teacher.
- Teacher states did not help in coding: the corrector fed E4B's states taught no better than fed the student's own. Online, on agent decisions with keep + replay, they did (one seed per control).
- Code, results, 40,000+ graded answers and pre-registrations: https://github.com/syn-ack-ai/copy-correct-rewrite.

## 2 Background

A teacher row is a prompt x and a verified reply y. SFT minimizes the cross-entropy of y; an anchored recipe adds a pull toward a reference distribution q over the reply:

L = CE(y | x) + λ · KL(q ‖ p_θ), with λ = 1.  (1)

With q the untrained student, eq. (1) is learning without forgetting (Li and Hoiem, 2016); with q the corrected student, it is the *corrector anchor*. On-policy distillation (Agarwal et al., 2023) samples ŷ from the student and minimizes the reverse KL to a teacher t on those tokens:

L = Σ_i KL(p_θ ‖ t) over the positions i of ŷ.  (2)

## 3 Method

![The corrector](figures/fig1_corrector.png)

Figure 1. How the corrector connects teacher and student.

**Structure.** A gated MLP reads hidden states at 50% and 65% depth in a first pass. In a second pass it writes a gated, additive correction into the student at 35% depth and above. The *self-corrector* reads the student's own states; the *cross-model corrector* reads the teacher's, aligned token by token across tokenizers. Correctors have 65-121M parameters (Table D1).

**Fit.** We fit it for 600 steps at lr 3e-4 on

L_fit = CE_corrected + 0.03 · ‖Δ‖ + 0.5 · H,  (3)

with ‖Δ‖ the size of the change and H a do-no-harm term that keeps the student's own top-32 distribution wherever it already ranks the true token first. Fitting rows mix agent replies (40-50%) with prose, code, questions, stories and, online, E2B's own samples (Table D1).

**Use.** Live, the corrected Qwen is far closer to the teacher on agent replies (0.67 nats vs 1.39 alone; self-corrector 0.73). We test it as an anchor (eq. 1), an on-policy teacher (eq. 2), a distilled LoRA and an online teacher.

## 4 Experiments

### 4.1 Setup

**Models.** The teacher is Gemma 4 E4B (4-bit QAT, dequantised to bf16). The coding students are Qwen3-1.7B (Yang et al., 2025), with a different tokenizer, and Gemma 4 E2B.

**Training data.** 299 older LiveCodeBench problems (Jain et al., 2024; May 2023 to Sep 2024), each with E4B's own solution passing every hidden test.

**Optimization.** A rank-64 LoRA (alpha 128) on the MLPs of layers 9-27 (Qwen; E2B 12-34), lr 3e-4 with warm-up and cosine decay, 299 steps of two rows, the same rows and order for every recipe.

**Evaluation.** Greedy answers with a 768-token cap. *Fixed*: of the 200 training problems the untrained student fails, how many it now solves. *New*: the same for the 163 of 188 newer held-out problems (Sep 2024 to Apr 2025, easy and medium) it fails. *Held solved*: all 188. *Lost*: previously solved problems (25 held-out, 103 in the training pool) now failed. *Drift*: KL(original ‖ trained) and the change in cross-entropy on held-out human prose, Python code and stories, every seventh text of the corrector's sources, never trained on. *Fresh-question p(true)*: mean probability of the true answer on held-out multiple-choice questions. Counts are pooled over three seeds with exact sign tests on discordant problems; every comparison was pre-registered (Appendices A, C). Re-evaluation is exact (0 of 495 answers differed), so arms pair across sessions; retraining is not (two plain-SFT runs differed by 6 held solves).

### 4.2 Copying the teacher's verified text teaches the most

LoRAs trained on E4B's agent turns and scored on LiveCodeBench (1,739 problem-runs; Figure D1) all beat the untrained student (p < 1e-6). Plain SFT led the corrector's distilled self and cross-model forms (161 vs 125 problems solved by one form only, p = 0.04; 181 vs 114, p = 0.0001) while losing fewer solved problems (76, against 82-89). E4B's states added nothing a stored adapter kept (cross-model vs self, 113 vs 144); their live advantage was only 0.05-0.06 nats on agent replies. SFT with a KL pull toward the original lost the fewest (66) but gained the least (vs SFT 90 vs 195, p = 5e-10); on coding rows it did well (Section 4.5).

On coding rows, the shortest passing solutions (median 295 tokens, not 355) raised held solved from 149 to 160 and cut losses from 58 to 48 (clear in one seed).

### 4.3 Copying within the family forgets, and an anchor fixes it

E2B replaced Qwen, all else equal; it starts stronger (57 held problems solved alone, against 25) and finds E4B's solutions nearly its own (0.27 nats per token, against 1.05). Gains were smaller, about 11 held problems per seed against 19-30 for Qwen. Every copying recipe damaged untaught text (Table 1, Figure 2): 1.6-13× Qwen's KL, sharper predictions (entropy −0.3 to −0.8) and higher loss on prose, code and stories. Low-perplexity training is reported to forget less (Wu et al., 2025), measured on tasks; E2B too lost a smaller share of solved problems (8% vs Qwen's 15%), its damage showing in untaught-text loss and entropy.

Table 1. Gemma 4 E2B student, three-seed means (alone: 57 held).

| Recipe | Held solved | Fixed (of 151) | Lost (of 205) | Prose KL | Prose Δ loss |
|---|---|---|---|---|---|
| SFT | 68.0 | 119.7 | 15.7 | 0.29 | +0.69 |
| Shortest rows | 68.0 | **123.0** | 15.7 | 0.29 | +0.57 |
| Shortest + KL to the original, λ 0.3 | 70.0 | 108.0 | 15.0 | 0.031 | +0.01 |
| Shortest + KL to the original, λ 1 | 69.0 | 90.7 | **13.3** | 0.016 | −0.04 |
| Shortest + self-corrector anchor | **72.0** | 95.3 | 16.0 | 0.33 | −0.60 |
| Shortest + the corrector's fitting text | 71.3 | 105.0 | 16.3 | 0.83 | −0.75 |

The likely cause is over-fitting familiar text: E2B's training loss on E4B's rows started at 0.28 nats and reached 0.07 in the second epoch (Qwen's ended at 0.24). Less training is no fix: stopped at Qwen's loss, E2B's prose damage fell from +0.69 to +0.09, but it kept no held gain (Appendix D).

Every anchor removed the damage in every seed. KL to the original kept E2B in place (prose KL 0.016, Δ loss −0.04) without costing held solves (207 vs 204). A lighter pull (λ = 0.3) also removed the damage and fixed 52 more (p = 0.0001); λ = 3 cost 25 held solves. Anchors to fresh Gemma correctors moved it as far as SFT, toward lower loss and softer predictions (Δ loss −0.58 to −0.60); held solves rose by at most 12 (216 vs 204, p = 0.22), at about 25 fewer fixes per seed. Training directly on the corrector's fitting text matched that anchor (held 214 vs 216, p = 0.90; prose Δ loss −0.75) but moved E2B much further (prose KL 0.83 vs 0.33): the anchor's gain is its fitting text, delivered with less change.

![E2B per recipe](figures/fig9_e2b.png)

Figure 2. Qwen3-1.7B (gray) and Gemma 4 E2B (colour) per recipe.

### 4.4 Reversed: E2B teaching E4B

With E2B's 148 passing answers teaching E4B (weak-to-strong; Burns et al., 2023), SFT barely helped (224 held solved against 207 alone) and damaged E4B's prose most of all (+0.81). Every anchor did significantly better (244-250) and removed the damage; cross-model and self anchors tied (Appendix D).

### 4.5 What keeps the student where it was

Back on Qwen, where copying left untaught text unharmed, the anchors show their own effect (Table 2, Figure D2). On the shortest rows, KL to the original moved the student 6-9× less than the other copying recipes; the corrector anchor moved it as far as SFT, toward lower loss and softer predictions, from the corrector's prose fit.

Table 2. Qwen's drift on untaught text, three-seed means.

| Recipe | KL from the original (prose) | Change in loss (nats): prose / code / stories |
|---|---|---|
| Plain SFT | 0.074 | −0.06 / −0.08 / −0.10 |
| Shortest + KL to the original | **0.010** | +0.01 / −0.10 / −0.02 |
| Shortest + corrector anchor | 0.089 | −0.24 / −0.21 / −0.51 |

Neither anchor costs teaching (Table 3); they matter where training does damage (Sections 4.3 and 4.6).

Table 3. Qwen with the two anchors, three seeds pooled.

| Recipe | Held solved (of 564) | Lost (of 384) | Fresh-question p(true) |
|---|---|---|---|
| Shortest rows | 160 | 48 | 0.527 |
| Shortest + KL to the original | 159 | **40** | 0.521 |
| Shortest + corrector anchor | 157 | 52 | 0.554 |
| Plain SFT | 149 | 58 | 0.557 |
| Untrained student | — | — | 0.513 |

### 4.6 The student's own words sharpen it

Adapting the self-rewrite step of Terminal-RSI (Yang et al., 2026), E4B turns each verified solution into a numbered plain-English plan, the student solves the problem with the plan in its prompt, and passing solutions are trained on without the plan. The 228 rows kept are the student's own text: similarity to E4B's solution 0.20, 0.32 nats under the untrained student (E4B's rows 1.05), 110 tokens (350).

These runbook rows (Table D3) generalised less (83 new, against 90-95) but fixed the most training problems (421 vs SFT's 389; 133 vs 84 by one only) and lost the fewest training-pool problems. They are the only Qwen recipe that raised loss on untaught text, in every seed and kind (prose +0.10 to +0.16), while barely moving the student (KL 0.022), and lowered its entropy the most (−0.23 nats on prose). A small KL is no proof against forgetting (compare Shenfeld et al., 2025). The corrector anchor reversed that in every seed (prose −0.23 to −0.25; fresh-question p(true) +0.028), at the cost of 39 fewer fixed problems (p = 3e-5) and 10 fewer held solves. With E2B they lost their advantage (fixed 61 vs 111 for SFT, p = 0.0002): its rewrites were no closer to it than E4B's text (0.38 vs 0.27 nats).

![Teaching and forgetting](figures/fig3_teach_vs_forget.png)

Figure 3. Coding recipes by held-out rank; best is bottom right.

### 4.7 On-policy distillation stops at its teacher

On-policy distillation through the self-corrector reached 116 held solves, its teacher's own live level (117; it fit E4B's solutions at 0.70 nats, the student 1.00). Refit on the 299 solutions (0.49 nats), the teacher solves 151 live and the student 148 (Table D4, Figure D3), beating the weak-teacher version in every seed but only tying SFT (38 vs 39) and losing more training problems (59 vs 42).

### 4.8 Online learning on agent decisions

E2B with a rank-16 LoRA met 600 unseen decision points from the competition's harness, each once, scored before learning it. The target was the teacher's distribution (A) or the cross-model corrector (B), each also with keep + replay (C, D: KL to the original E2B on one untaught row per item, and an earlier item every fourth step). The corrector was fitted on the training repositories, which the stream draws from; the held measures and gate use repositories neither saw. The gate has 220 held decisions: *fixed* counts the 52 where E2B alone differs from the teacher, *broken* the other 168. The result is a plain LoRA; the corrector is needed only during training. Two controls replace D's corrector, with keep + replay: a self-corrector, which reads E2B's own states, and a pull-matched blend, the teacher mixed with the original E2B (weight 0.43) so that it pulls E2B as far as the corrector does.

Table 4. Online learning on agent decisions; one seed per row.

| Setup | Held CE (from 0.825) | Untaught drift | Held-question drift | Gate: fixed / broken |
|---|---|---|---|---|
| A Teacher | 0.787 | 0.089 | 0.030 | 25 / 11 |
| B Corrector | 0.753 | 0.055 | 0.043 | 16 / 4 |
| C Teacher + keep + replay | 0.775 | **0.014** | 0.034 | 14 / 6 |
| D Corrector + keep + replay | **0.751** | 0.017 | 0.041 | 23 / **4** |
| D, seed 2 | 0.749 | 0.017 | 0.021 | 23 / 6 |
| Self-corrector + keep + replay (control) | 0.755 | 0.025 | 0.073 | 13 / 5 |
| Pull-matched blend + keep + replay (control) | 0.781 | 0.014 | 0.022 | 14 / 4 |

Keep + replay hurt the plain teacher (A to C: 11 fewer fixed) but helped the corrector teacher (B to D: 7 more), and D matched B's closeness at a third of its drift. D had the best net gate result (+19; A +14, within noise, p = 0.40) and beat C on paired decisions (17 vs 6 and 17 vs 8 for its two seeds, p = 0.035 and 0.11). D's second seed reproduced it, differing on 6 of 220 decisions.

Both controls fell back to C's level. The self-corrector came as close to the teacher as D and moved held-question answers most, but fixed no more than C (net +8; D beat it 12 vs 1 and 12 vs 3, p = 0.003 and 0.035): the gain needs E4B's states. The blend also behaved like C (net +10; D beat it 11 vs 2 and 11 vs 4, p = 0.02 and 0.12): D is not just a softer target. Further seeds of C and D, a disagreement-gated teacher and a live test of D's LoRA were still running.

![Online curves](figures/fig7_online.png)

Figure 4. Held cross-entropy and untaught drift over the stream.

## 5 Discussion

**For training small agents.** Train on the teacher's shortest verified answers with a light KL pull toward the original student, which within the family removed copying's damage at no cost; check the teacher's fit before on-policy distillation; and measure forgetting on held-out human text, with entropy, since loss on teacher text rewards imitation.

**Where the corrector helps.** As an anchor it delivers its fitting text with less change; as an online teacher with keep + replay it broke few decisions while fixing nearly as many as the plain teacher, but not when it read the student's own states or was replaced by a pull-matched blend.

**Limitations.** Two coding students; online, one seed per setup (D two). Pooled p-values are descriptive; differences under about 10 held solves are within noise. The KL sweep and the fitting-text control ran only for E2B. About 9% of losses were answers cut at the 768-token cap that passed given more tokens (Appendix D). The online correctors, unlike the coding ones, had no size term and saw E2B's own samples, confounding the coding-online contrast.

**Ethical considerations.** Prose and stories for fitting and drift come from locally held books, not released. Model-written code ran only in sandboxed interpreters. AI use: under the author's direction, Claude Opus 5.5 (Anthropic) designed and ran the experiments, wrote the code and drafted the text (Appendix B).

## 6 Related Work

**Distillation.** GKD (Agarwal et al., 2023), MiniLLM (Gu et al., 2023) and DistiLLM (Ko et al., 2024) distil on the student's own samples with reverse or skewed KL. Reachable targets drive ProKT (Shi et al., 2021), DIST (Huang et al., 2022) and PACED (Xu et al., 2026), which makes expert solutions attainable, like our runbook rows.

**Training on own solutions.** ReST (Gulcehre et al., 2023), ReST-EM (Singh et al., 2023), rejection-sampling fine-tuning (Yuan et al., 2023) and STaR (Zelikman et al., 2022) retrain on verified own samples; Terminal-RSI (Yang et al., 2026) adds plan-guided self-rewriting. The damage to untaught text we report resembles entropy collapse in reasoning RL (Cui et al., 2025).

**Forgetting.** Learning without forgetting (Li and Hoiem, 2016) and EWC (Kirkpatrick et al., 2016) protect old skills with output and weight penalties, and LoRA learns less and forgets less (Biderman et al., 2024); RL's Razor (Shenfeld et al., 2025) ties forgetting to KL from the original. On-policy distillation from the original model restores lost skills (Lu, 2025), continual variants add replay (Chen et al., 2026), and EAFT gates loss by entropy (Diao et al., 2026).

**Cross-model transfer.** Logit-space methods align vocabularies (Boizard et al., 2024; Zhang et al., 2024), model fusion merges distributions (Wan et al., 2024), and model stitching joins representations (Lenc and Vedaldi, 2014; Bansal et al., 2021). Residual-learning distillation projects teacher states for the student (On et al., 2026); steering-vector distillation stores a training-time intervention in a LoRA (Blank et al., 2026).

## 7 Conclusion

Copying verified answers teaches best; within the family it also causes forgetting, and a KL pull toward the original student is the cheapest fix. On-policy distillation stops at its teacher's level.

## References

R. Agarwal, N. Vieillard, Y. Zhou, P. Stanczyk, S. Ramos, et al. (2023). On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes. arXiv:2306.13649. https://arxiv.org/abs/2306.13649

A. Alabdulkareem, I. Tenison, Y. Zhang, Y. Zhu, B. Katz, and D. Gutfreund (2026). Swarm Telepathy: Prefill-Free KV-Cache Transfer Across Families. arXiv:2610.15095. https://arxiv.org/abs/2610.15095

Y. Bansal, P. Nakkiran, and B. Barak (2021). Revisiting Model Stitching to Compare Neural Representations. arXiv:2106.07682. https://arxiv.org/abs/2106.07682

D. Biderman, J. Portes, J. J. Gonzalez Ortiz, M. Paul, P. Greengard, et al. (2024). LoRA Learns Less and Forgets Less. arXiv:2405.09673. https://arxiv.org/abs/2405.09673

C. Blank, A. Bhatia, S. Rajamanoharan, A. Conmy, and N. Nanda (2026). Subliminal Learning Is Steering Vector Distillation. arXiv:2606.00995. https://arxiv.org/abs/2606.00995

N. Boizard, K. El Haddad, C. Hudelot, and P. Colombo (2024). Towards Cross-Tokenizer Distillation: the Universal Logit Distillation Loss for LLMs. arXiv:2402.12030. https://arxiv.org/abs/2402.12030

C. Burns, P. Izmailov, J. H. Kirchner, B. Baker, L. Gao, et al. (2023). Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision. arXiv:2312.09390. https://arxiv.org/abs/2312.09390

Y. Chen, T. Zhu, M. Zhang, X. Chen, J. Huang, et al. (2026). On-Policy Replay for Continual Supervised Fine-Tuning. arXiv:2605.29495. https://arxiv.org/abs/2605.29495

G. Cui, Y. Zhang, J. Chen, L. Yuan, Z. Wang, et al. (2025). The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models. arXiv:2505.22617. https://arxiv.org/abs/2505.22617

M. Diao, L. Yang, W. Gong, Y. Zhang, Z. Yan, et al. (2026). Entropy-Adaptive Fine-Tuning: Resolving Confident Conflicts to Mitigate Forgetting. arXiv:2601.02151. https://arxiv.org/abs/2601.02151

Google (2026). Gemma 4 E2B and E4B, instruction-tuned QAT checkpoints (w4a16). Hugging Face model cards google/gemma-4-E2B-it-qat-w4a16-ct (revision 971342c) and google/gemma-4-E4B-it-qat-w4a16-ct (revision 6cd26aa).

Y. Gu, L. Dong, F. Wei, and M. Huang (2023). MiniLLM: On-Policy Distillation of Large Language Models. arXiv:2306.08543. https://arxiv.org/abs/2306.08543

C. Gulcehre, T. L. Paine, S. Srinivasan, K. Konyushkova, L. Weerts, et al. (2023). Reinforced Self-Training (ReST) for Language Modeling. arXiv:2308.08998. https://arxiv.org/abs/2308.08998

G. Hinton, O. Vinyals, and J. Dean (2015). Distilling the Knowledge in a Neural Network. arXiv:1503.02531. https://arxiv.org/abs/1503.02531

E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685. https://arxiv.org/abs/2106.09685

T. Huang, S. You, F. Wang, C. Qian, and C. Xu (2022). Knowledge Distillation from A Stronger Teacher. NeurIPS 2022. arXiv:2205.10536. https://arxiv.org/abs/2205.10536

N. Jain, K. Han, A. Gu, W.-D. Li, F. Yan, et al. (2024). LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code. arXiv:2403.07974. https://arxiv.org/abs/2403.07974

Kaggle (2026). Gemma 4 Developer Agent competition. https://www.kaggle.com/competitions/gemma-4-developer-agent

J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, et al. (2016). Overcoming catastrophic forgetting in neural networks. arXiv:1612.00796. https://arxiv.org/abs/1612.00796

J. Ko, S. Kim, T. Chen, and S.-Y. Yun (2024). DistiLLM: Towards Streamlined Distillation for Large Language Models. arXiv:2402.03898. https://arxiv.org/abs/2402.03898

K. Lenc and A. Vedaldi (2014). Understanding image representations by measuring their equivariance and equivalence. arXiv:1411.5908. https://arxiv.org/abs/1411.5908

Z. Li and D. Hoiem (2016). Learning without Forgetting. arXiv:1606.09282. https://arxiv.org/abs/1606.09282

K. Lu and Thinking Machines Lab (2025). On-Policy Distillation. Connectionism (Thinking Machines Lab blog). https://thinkingmachines.ai/blog/on-policy-distillation/

T. On, H. Pei, L. Lausen, and G. Karypis (2026). Knowledge Distillation for Large Language Models through Residual Learning. ICLR 2026. https://openreview.net/forum?id=Dh6KxUxG20

I. Shenfeld, J. Pari, and P. Agrawal (2025). RL's Razor: Why Online Reinforcement Learning Forgets Less. arXiv:2509.04259. https://arxiv.org/abs/2509.04259

W. Shi, Y. Song, H. Zhou, B. Li, and L. Li (2021). Follow Your Path: a Progressive Method for Knowledge Distillation. ECML-PKDD 2021. arXiv:2107.09305. https://arxiv.org/abs/2107.09305

A. Singh, J. D. Co-Reyes, R. Agarwal, A. Anand, P. Patil, et al. (2023). Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models. arXiv:2312.06585. https://arxiv.org/abs/2312.06585

F. Wan, X. Huang, D. Cai, X. Quan, W. Bi, and S. Shi (2024). Knowledge Fusion of Large Language Models. arXiv:2401.10491. https://arxiv.org/abs/2401.10491

C.-C. Wu, Z. R. Tam, C.-Y. Lin, Y.-N. Chen, S.-H. Sun, et al. (2025). Mitigating Forgetting in LLM Fine-Tuning via Low-Perplexity Token Learning. NeurIPS 2025. arXiv:2501.14315. https://arxiv.org/abs/2501.14315

Y. Xu, H. Sang, Z. Zhou, R. He, and Z. Wang (2026). PACED: Distillation and On-Policy Self-Distillation at the Frontier of Student Competence. arXiv:2603.11178. https://arxiv.org/abs/2603.11178

A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, et al. (2025). Qwen3 Technical Report. arXiv:2505.09388. https://arxiv.org/abs/2505.09388

J. Yang, Z. Li, Y. Shi, Z. Li, R. Wang, et al. (2026). Terminal-RSI: Recursive Self-Improving Terminal Agents via Recursive Self-Rewrite and Policy-Refine. arXiv:2610.22192. https://arxiv.org/abs/2610.22192

Z. Yuan, H. Yuan, C. Li, G. Dong, K. Lu, et al. (2023). Scaling Relationship on Learning Mathematical Reasoning with Large Language Models. arXiv:2308.01825. https://arxiv.org/abs/2308.01825

E. Zelikman, Y. Wu, J. Mu, and N. D. Goodman (2022). STaR: Bootstrapping Reasoning With Reasoning. arXiv:2203.14465. https://arxiv.org/abs/2203.14465

S. Zhang, X. Zhang, Z. Sun, Y. Chen, and J. Xu (2024). Dual-Space Knowledge Distillation for Large Language Models. arXiv:2406.17328. https://arxiv.org/abs/2406.17328

## Appendix A Code

Code, per-problem results, over 40,000 graded answers and the pre-registration log (addenda 165-197) are public at https://github.com/syn-ack-ai/copy-correct-rewrite (Apache-2.0). Not included: text corpora, competition data, problem statements and weights.

## Appendix B AI use

The author carried out this study with Claude Opus 5.5 (Anthropic) as an AI research assistant. The author set the research direction, chose the questions and approved each pre-registered experiment; Claude Opus 5.5 designed and ran the experiments, wrote the code and drafted the text. Separate Claude instances reviewed new code and drafts, and verified the numbers.

## Appendix C Pre-registration scorecard

Each addendum in the public log states its predictions and, for some, a bar the result had to clear before running. Rows still running were updated as results landed.

Table C1. Predictions that held, per pre-registered addendum.

| Addendum (question) | Predictions held | Pre-set bar |
|---|---|---|
| 169d (corrector forms vs dose-matched SFT, agent replies) | 0 of 4 | failed |
| 170 (online learning) | 3 of 5 | — |
| 174 (SFT with anchors) | 3 of 5 | not met; states added nothing on held |
| 178 (wide coding) | 0 of 1 (reversed) | — |
| 179 (on-policy, shortest rows) | 2 of 3 | — |
| 181 (refit teacher, rewrites) | 9 of 12 | not met |
| 182 (runbook rows) | 8 of 11 | missed by one |
| 184c (runbook rows + self-anchor) | 4 of 6 | not met |
| 185 (SFT + KL to the original) | 5 of 6 | — |
| 186 (Gemma 4 E2B as the coding student) | 5 of 7 | — |
| 187 (two-way, fresh Gemma correctors) | 5 of 7 | not met; cross-model = self |
| 188 (matched-loss run, E2B) | 5 of 7 | not met: stopping early cut the damage but kept no held gain |
| 189 (online setup D: replication, controls) | 6 of 6 so far (seed 2, both controls) | running |
| 190 (fitting-text control for the corrector anchor) | 4 of 5 | met: the anchor's gain is its fitting text |
| 192 (online vs offline SFT teachers) | — | queued |
| 193 (KL strength sweep) | 2 of 4 | not met: λ 0.3 holds, λ 3 costs held solves |
| 194a (longer-cap re-grade of capped losses) | 6 of 6 | — (9% of losses recovered) |
| 196 (online setup D live on 80 held-out agent tasks) | — | queued |
| 197 (disagreement-gated online teacher) | — | queued |

Addenda 191 and 195 (per-user and lottery-ticket grafts) belong to a separate study.

## Appendix D Further results

**Correctors.** Sizes and fitting rows are in Table D1. Correctors fitted for longer than 600 steps gave mostly worse LoRAs (one seed).

Table D1. Corrector parameters and fitting rows by student.

| Student | Self-corrector | Cross-model corrector (reads) | Fitting rows |
|---|---|---|---|
| Qwen3-1.7B, coding | 78M | 83M (E4B) | agent replies 50%, human prose 15%, Python code 15%, multiple-choice questions 10%, stories 10% |
| Gemma 4 E2B, coding | 65M | 75M (E4B) | as above |
| Gemma 4 E4B, coding (Section 4.4) | 121M | 110M (E2B) | as above |
| Gemma 4 E2B, online (Section 4.8) | 69M | 83M (E4B) | agent replies 40%, E2B's own samples 20%, prose 15%, code 10%, questions 10%, stories 5% |

**Agent-turn LoRAs (Section 4.2).** Figure D1 shows the problems each form gained and lost. In a separate coding run the anchors did not lose to SFT (held solved: SFT 143, anchors 146-152).

![Gains and losses](figures/fig2_wide_coding.png)

Figure D1. LoRAs trained on E4B's agent turns, on LiveCodeBench (1,739 problem-runs): problems gained and lost against the untrained student.

**Gemma 4 E2B, every recipe (Section 4.3).** Table D2 adds the runbook rows, the less-training arms (addendum 188), λ = 3, the cross-model anchor and the fitting text with the KL pull. E2B's training loss on E4B's rows started low (0.28 nats), barely fell in the first epoch and reached 0.07 in the second; Qwen's ended at 0.24. Stopped there after one epoch, E2B's prose damage fell from +0.69 to +0.09, yet it moved twice as far as Qwen (KL 0.16 against 0.074), so part of the effect is the student. Stopping early kept no held gain and lost 122 solved problems (SFT 47). Both partly trained arms lost mostly trained problems (75 and 49, against 18), which SFT's second epoch turns into E4B's verified answers. The KL pull did not reduce the fitting text's drift (prose KL 0.81): it covers only the coding replies. The Gemma correctors left E2B's coding loss unchanged.

Table D2. Gemma 4 E2B as the student, per-seed means over three seeds (alone: 57 held solved).

| Recipe | Held solved | Fixed (of 151) | Lost (of 205) | Prose KL | Prose Δ loss |
|---|---|---|---|---|---|
| *Copying* | | | | | |
| SFT | 68.0 | 119.7 | 15.7 | 0.29 | +0.69 |
| Shortest rows | 68.0 | 123.0 | 15.7 | 0.29 | +0.57 |
| Runbook rows | 67.0 | 106.0 | 19.3 | 0.29 | +0.85 |
| *Less training (188)* | | | | | |
| SFT, stopped at loss 0.24 | 56.7 | 79.0 | 40.7 | 0.16 | +0.09 |
| SFT, a tenth of the learning rate | 68.7 | 77.7 | 23.7 | 0.12 | +0.18 |
| *KL to the original (185, 193)* | | | | | |
| Shortest + KL to the original, λ 0.3 | 70.0 | 108.0 | 15.0 | 0.031 | +0.01 |
| Shortest + KL to the original, λ 1 | 69.0 | 90.7 | **13.3** | 0.016 | −0.04 |
| Shortest + KL to the original, λ 3 | 60.7 | 59.7 | 24.3 | **0.014** | −0.05 |
| *Corrector anchors (187)* | | | | | |
| Shortest + self-corrector anchor | **72.0** | 95.3 | 16.0 | 0.33 | −0.60 |
| Shortest + cross-model anchor | 69.7 | 99.3 | 15.3 | 0.32 | −0.58 |
| *The corrector's fitting text (190)* | | | | | |
| Shortest + the corrector's fitting text | 71.3 | 105.0 | 16.3 | 0.83 | −0.75 |
| Shortest + fitting text + KL to the original | 68.7 | 83.3 | 17.3 | 0.81 | −0.76 |

**E2B teaching E4B (Section 4.4).** The self-corrector anchor beat SFT 52 vs 26 on problems solved by one only (p = 0.004), and every anchor lost fewer solved problems than SFT. Cross-model and self anchors tied in both directions (E2B student: 209 vs 216 held solved; E4B student: 249 vs 250).

**Qwen drift (Section 4.5).** Figure D2 shows each seed.

![Prose drift](figures/fig4_untaught_prose.png)

Figure D2. Drift on untaught human prose, one point per seed.

**Runbook rows (Section 4.6).** A screen rejected plans with code or over 10% of the solution's code 4-grams.

Table D3. Runbook rows, three seeds pooled; prose ΔCE per seed.

| Recipe | Fixed | New | Held solved | Lost (of 384) | Prose ΔCE (per seed) |
|---|---|---|---|---|---|
| Plain SFT | 389 | 90 | 149 | 58 | −0.05 to −0.07 |
| Runbook rows | **421** | 83 | 143 | 39 | **+0.10 to +0.16** |
| Runbook rows + self-anchor | 382 | 71 | 133 | 37 | −0.23 to −0.25 |

**On-policy distillation (Section 4.7).** As a rewriter, the refit corrector copied E4B's text (similarity 0.98).

Table D4. Held solved, three seeds pooled.

| Recipe | Weak teacher | Refit teacher |
|---|---|---|
| On-policy only | 116 | 148 |
| SFT + on-policy | 117 | 158 |

![Teachers and students](figures/fig5_teacher_cap.png)

Figure D3. Each corrector teacher run live, and the on-policy student it trained.

**Answers cut at the cap.** Given 768 more tokens, 59 of the 176 lost answers cut at the cap passed (9% of all losses; 18-25% for refit-teacher recipes).

**Online (Section 4.8).** C sits closer to the teacher than A yet flips fewer of E2B's top choices.
