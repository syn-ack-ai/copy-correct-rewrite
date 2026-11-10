# Copy, Correct, or Rewrite? Telepathy-Like Teaching of a Small Agent Model and What Keeps It From Forgetting

Benjamin Babeshkin

*Lessons for training Gemma 4 E2B/E4B developer agents from a stronger teacher.*

## Abstract

A small agent model trained by a stronger one must gain the teacher's skill without losing its own. We study *telepathy-like teaching*: a state corrector, a small network fed Gemma 4 E4B's hidden states, corrects Gemma 4 E2B's hidden states, and the corrected E2B, not E4B's output, teaches a LoRA. Online, on a stream of agent decisions, students taught this way were within a sixth of the stream as close to E4B as students copying E4B's distribution were at its end, and finished closer in every seed. On 80 live agent tasks the LoRA beat untrained E2B served the same way (higher solve rate on 12 tasks, lower on none; p = 0.0005). Most of the gain comes with the corrector's form: a corrector fed E2B's own states taught almost as well, and E4B's states added a small measured remainder. On contest coding, under the same pre-registered protocol, copying the teacher's shortest verified answers taught as much as any route and the corrector stored nothing more. Within the family, copying also caused forgetting, which a KL pull toward the original student removed without costing held-out solves.

## 1 Introduction

A submission to the Gemma 4 Developer Agent competition (Kaggle, 2026) is one Gemma 4 model, here the small E2B or E4B (Google, 2026), plus LoRA adapters (Hu et al., 2021) of rank at most 128, so any gain from a stronger teacher must be stored in an adapter. Distillation (Hinton et al., 2015) is the main lever; we ask which route teaches the most and forgets the least.

Can a teacher's internal states teach better than its outputs? Swarm Telepathy (Alabdulkareem et al., 2026) moves one model's KV cache into a frozen receiver of another family. We train instead: a *state corrector* turns the teacher's hidden states into a correction of the student's, and the corrected student teaches a LoRA. In this *telepathy-like teaching* the teacher reaches the student as states, not tokens, and is gone at inference. Our contributions:

- Online, telepathy-like teaching beats copying the teacher's distribution: the same closeness in a sixth of the stream, a closer student in every seed, and a LoRA that beats untrained E2B live (p = 0.0005).
- Most of that gain comes with the corrector's form, shared by a corrector reading only the student's states; E4B's states add about a ninth.
- In coding, copying verified answers teaches as much as any route, best with the shortest; the corrector stores nothing more.
- Same-family copying causes forgetting; a KL pull toward the original student removes it without costing held-out solves.
- Self-written solutions make the student overconfident; on-policy distillation stops at its teacher.
- Code, results, 77,000+ graded answers and pre-registrations: https://github.com/syn-ack-ai/copy-correct-rewrite.

## 2 Background

A teacher row is a prompt x and a verified reply y. SFT minimizes the cross-entropy of y; an anchored recipe adds a pull toward a reference distribution q over the reply:

L = CE(y | x) + λ · KL(q ‖ p_θ), with λ = 1.  (1)

With q the untrained student, eq. (1) is learning without forgetting (Li and Hoiem, 2016); with q the corrected student, it is the *corrector anchor*. On-policy distillation (Agarwal et al., 2023) samples ŷ from the student and minimizes the reverse KL to a teacher t on those tokens:

L = Σ_i KL(p_θ ‖ t) over the positions i of ŷ.  (2)

Telepathy-like teaching is eq. (2) with the corrected student as t.

## 3 Method

![The corrector](figures/fig1_corrector.png)

Figure 1. How the corrector connects teacher and student.

**Structure.** A gated MLP reads hidden states at 50% and 65% depth and the student's state at its write layer (35% depth); in a second pass it adds a gated correction there and at every layer above. The *cross-model corrector* reads the teacher's states, aligned token by token across tokenizers; the *self-corrector*, our control, reads the student's own. Correctors have 65-121M parameters (Table D1).

**Fit.** With teacher and student frozen, we fit the corrector for 600 steps at lr 3e-4 on

L_fit = CE_corrected + 0.03 · ‖Δ‖ + 0.5 · H,  (3)

with CE_corrected the corrected student's cross-entropy on the teacher's replies, ‖Δ‖ the size of the change and H a do-no-harm term that keeps the student's own top-32 distribution wherever it already ranks the true token first. Fitting rows mix agent replies (40-50%) with prose, code, questions, stories and, online, E2B's own samples (Table D1).

**Telepathy-like teaching.** The student, now with a LoRA, meets each item once. It samples two continuations of the item's context, and eq. (2) pulls the LoRA toward a teacher's distribution on those tokens. The plain teacher is E4B's own distribution. The telepathy-like teacher is the original student, corrected on the same tokens by the corrector fed E4B's states; the control feeds the corrector the student's own states. *Keep + replay* adds a KL to the original student on one untaught row per item and relearns an earlier item every fourth step. Afterwards only the LoRA is kept: no teacher, no corrector.

**Other uses.** We also test the corrected student as an anchor (eq. 1), an offline on-policy teacher and a distilled LoRA (Sections 4.5-4.6).

## 4 Experiments

### 4.1 Setup

**Models.** The teacher is Gemma 4 E4B (4-bit QAT, dequantised to bf16). The online student is Gemma 4 E2B; the coding students are E2B and Qwen3-1.7B (Yang et al., 2025), which has a different tokenizer.

**Online stream.** E2B with a rank-16 LoRA meets 600 unseen decision points from the competition's harness (the context before one of E4B's tool calls). Correctors are fitted on the stream's training repositories; held measures use 17 repositories neither saw. The teachers: E4B's distribution (C); the cross-model corrector (D, telepathy-like); the self-corrector (S); D4 and S4, with correctors fitted four times longer; and DI, a cross-model corrector penalised for writes E2B's own states cannot predict. All use keep + replay.

**Online measures.** *Held CE*: cross-entropy to E4B on held repositories (untrained E2B: 0.825). *Prequential gain*: nats per token gained over the original E2B on each item before learning it. *Gate*: 220 held decisions; *fixed* counts the 52 where E2B alone differs from E4B, *broken* the other 168, and *net* is their difference. Hardware alone moved C by 8 net decisions, so every teacher ran on one GPU type.

**Coding.** Training rows are 299 older LiveCodeBench problems (Jain et al., 2024; May 2023 to Sep 2024), each with E4B's own solution passing every hidden test. Every recipe trains a rank-64 LoRA (alpha 128) on the MLPs of layers 9-27 (Qwen; E2B 12-34) at lr 3e-4 for 299 steps of two rows, in the same order. Answers are greedy with a 768-token cap. *Fixed*: of the 200 training problems the untrained student fails, how many it now solves. *Held solved*: of 188 newer problems (Sep 2024 to Apr 2025). *Lost*: previously solved problems now failed. *Drift*: KL(original ‖ trained) and the change in cross-entropy on held-out human prose, Python code and stories, never trained on. Counts are pooled over three seeds, with exact sign tests on discordant problems; comparisons were pre-registered (Appendices A, C) unless marked. Re-evaluation is exact (0 of 495 answers differed); retraining is not (two plain-SFT runs differed by 6 held solves).

### 4.2 Telepathy-like teaching is faster and closer than copying

Table 1. Online teachers on one GPU type, per seed.

| Teacher | Held CE (from 0.825) | Prequential gain | Gate net |
|---|---|---|---|
| C E4B's distribution | 0.767 | 0.039 | +13 |
| S Self-corrector | 0.756, 0.754 | 0.063, 0.060 | +10, +15 |
| D Cross-model corrector | 0.754, 0.752 | 0.065, 0.063 | +19, +16 |
| DI Cross-model, imitable | 0.754, 0.751 | 0.064, 0.061 | +19, +20 |
| S4 Self-corrector, 4× | 0.744, 0.745 | 0.070, 0.066 | +17, +15 |
| D4 Cross-model, 4× | **0.741, 0.741** | 0.070, 0.067 | **+20, +21** |

Every corrector student learned faster than C's, and by item 100 of 600 (DI: 150) was as close to E4B on held text as C's student at the end (Table 1; Figure 2, left). On two other GPU types D reached C's final level by item 50-100 in every seed and ended 0.017-0.024 nats closer (Appendix D). On decisions the two tie: over all runs D netted +15 to +19, C +8 to +27.

Fitting the corrector four times longer brought its student another 0.008 nats closer; D4's was the closest and steadiest, with no more drift than D's. DI matched D with writes a quarter the size.

![Online curves](figures/fig7_online.png)

Figure 2. Left: closeness to E4B over the stream. Right: what each ingredient adds.

### 4.3 What the teacher's states add

Is this telepathy, or would any corrector do? S never sees E4B's states, yet has most of the gain: on the gate's 240 held rows its student is 0.014 nats closer to E4B than C's, and D's 0.017 (Figure 2, right). Over the first 100 items S's students gained as much as D's. Speed comes with the corrector's form.

The states add a small, consistent remainder. Paired on those rows, the cross-model student beat its self-corrector twin in all four pairs (two seeds, two fits) by 0.0019 nats (95% interval over repositories 0.0012-0.0026), a ninth of D's lead over C, and matched E4B on more decisions (19 vs 7, p = 0.029). We added this analysis after two seeds; our pre-registered bar, 0.005 in every seed, was not met.

The LoRA, not the teacher, is the limit: run live, the cross-model corrector is 0.03-0.05 nats closer to E4B than the self-corrector, and the students keep 0.002.

### 4.4 Live agent tasks

We ran each LoRA in the competition's harness on 80 held-out tasks that E4B failed. D's LoRA solved 19 of 320 attempts, D4's 13 of 160, C's 5 of 160, plain E2B 7 of 320, and a null adapter (zeroed weights, the same serving path) 2 of 160. By task, D beat the null adapter 12 vs 0 (p = 0.0005) and plain E2B 9 vs 1 (p = 0.021), and D4 beat the null adapter 8 vs 1 (p = 0.039); D against C's LoRA was 9 vs 3 (p = 0.15). D's LoRA ended more episodes in errors, mostly time-outs (37 per 160; D4 17, null 9): its episodes run longer.

### 4.5 In coding, the corrector stores nothing more than copying

LoRAs trained on E4B's agent turns and scored on LiveCodeBench (1,739 problem-runs; Figure D1) all beat the untrained Qwen (p < 1e-6), but plain SFT led the corrector's distilled self and cross-model forms (161 vs 125 problems solved by one form only, p = 0.04; 181 vs 114, p = 0.0001) and lost fewer solved problems (76, against 82-89). E4B's states added nothing a stored adapter kept (cross-model vs self, 113 vs 144). The shortest passing solutions (median 295 tokens, not 355) raised held solved from 149 to 160 and cut losses from 58 to 48 (clear in one seed).

Offline on-policy distillation stops at its teacher. Through the self-corrector it reached 116 held solves, the corrected student's own live level (117). With the corrector refit on the 299 solutions the teacher solved 151 live and the student 148 (Table D4, Figure D3), only tying SFT (38 vs 39).

### 4.6 Copying within the family forgets, and an anchor fixes it

With E2B in place of Qwen, the student starts stronger (57 held problems solved alone, against 25) and finds E4B's solutions familiar (0.27 nats per token, against 1.05). Gains were smaller, about 11 held problems per seed against 19-30. Every copying recipe damaged untaught text (Table 2, Figure 3): 2-13× Qwen's KL, sharper predictions (entropy −0.3 to −0.8) and higher loss on prose, code and stories. As with low-perplexity training (Wu et al., 2025), SFT cost E2B few solved problems (8% vs Qwen's 15%).

Table 2. Gemma 4 E2B student, three-seed means (alone: 57 held).

| Recipe | Held solved | Fixed (of 151) | Lost (of 205) | Prose KL | Prose Δ loss |
|---|---|---|---|---|---|
| SFT | 68.0 | **119.7** | 15.7 | 0.29 | +0.69 |
| Shortest + KL to the original, λ 0.3 | 70.0 | 108.0 | 15.0 | 0.031 | +0.01 |
| Shortest + KL to the original, λ 1 | 69.0 | 90.7 | **13.3** | 0.016 | −0.04 |
| Shortest + self-corrector anchor | **72.0** | 95.3 | 16.0 | 0.33 | −0.60 |
| Shortest + the corrector's fitting text | 71.3 | 105.0 | 16.3 | 0.83 | −0.75 |

The likely cause is over-fitting familiar text: E2B's training loss on E4B's rows began near 0.28 nats and reached 0.07 (Qwen's ended at 0.24). Less training is no fix: stopped at Qwen's loss, prose damage fell from +0.69 to +0.09 but no held gain remained (Appendix D).

Every anchor at λ = 1 removed the damage in every seed. KL to the original kept E2B in place at no cost in held solves (207 vs 204) but a quarter fewer fixes (272 vs 369). A lighter pull (λ = 0.3) removed over 90% of the damage and fixed 52 more (p = 0.0001 on trained problems); λ = 3 cost 25 held solves. The corrector anchor moved E2B as far as SFT, toward lower loss; training directly on the corrector's fitting text matched it (held 214 vs 216, p = 0.90) while moving E2B much further (prose KL 0.83 vs 0.33): the anchor's gain is its fitting text, delivered with less change.

The same held in other pairings (Appendix D). With E2B's answers teaching E4B (weak-to-strong; Burns et al., 2023), SFT barely helped (224 held solved against 207 alone) and every anchor did better (244-250; p ≤ 0.03, uncorrected). On Qwen, where copying left untaught text unharmed, anchors cost no held-out solves. Rows the student rewrote from E4B's plans (runbook rows, after Terminal-RSI; Yang et al., 2026) fixed the most training problems (421 vs SFT's 389; p = 0.001 on trained problems) but were the only Qwen recipe to raise loss on untaught text in every seed (prose +0.10 to +0.16) and sharpened it most; the corrector anchor reversed that.

![E2B per recipe](figures/fig9_e2b.png)

Figure 3. Qwen3-1.7B (gray) and Gemma 4 E2B (colour) per recipe.

## 5 Discussion

**Telepathy-like teaching.** The corrected student teaches in the student's own states: E2B reaches in a sixth of the stream what copying E4B's distribution gives at its end, and its LoRA wins live. Most of this needs no teacher at teaching time: the self-corrector, fitted once on E4B's replies, carries it, and E4B's live states add about a ninth. We do not know why: a blend of E4B's and E2B's distributions matched to the corrector's average pull did not reproduce it (Appendix D). In coding, where a LoRA had to store skill from 299 rows, the corrector added nothing over copying.

**For training small agents.** Teach online through a corrector fitted longer, with keep + replay. For stored coding skill, copy the teacher's shortest verified answers with a light KL pull toward the original student, and measure forgetting on held-out human text, with entropy, since loss on teacher text rewards imitation.

**Limitations.** One online pair with a shared tokenizer; on the single GPU type, two seeds per teacher and one for C. The paired cross-against-self analysis was added after two seeds, beside a bar it failed. Closeness to E4B is a proxy: decisions tie with the plain teacher. Live tasks are ones E4B failed, with 2-19 solves per arm and one adapter each. In coding, two students; pooled p-values are descriptive; differences under about 10 held solves are within noise; about 9% of losses were answers cut at the 768-token cap (Appendix D). The online correctors, unlike the coding ones, had no size term and saw E2B's own samples. Across tokenizers, copying the teacher's distribution needs token alignment while the corrector writes into the student's own states (as for Qwen); online this is untested.

**Ethical considerations.** Prose and stories for fitting and drift come from locally held books, not released. Model-written code ran only in sandboxes. AI use: under the author's direction, Claude Opus 5.5 and Claude Fable 5.2 (Anthropic) designed and ran the experiments, wrote the code and drafted the text (Appendix B).

## 6 Related Work

**Distillation.** GKD (Agarwal et al., 2023), MiniLLM (Gu et al., 2023) and DistiLLM (Ko et al., 2024) distil on the student's own samples with reverse or skewed KL. Reachable targets drive ProKT (Shi et al., 2021), DIST (Huang et al., 2022) and PACED (Xu et al., 2026).

**Training on own solutions.** ReST (Gulcehre et al., 2023), ReST-EM (Singh et al., 2023), rejection-sampling fine-tuning (Yuan et al., 2023) and STaR (Zelikman et al., 2022) retrain on verified own samples; Terminal-RSI (Yang et al., 2026) adds plan-guided self-rewriting. The damage we report resembles entropy collapse in reasoning RL (Cui et al., 2025).

**Forgetting.** Learning without forgetting (Li and Hoiem, 2016) and EWC (Kirkpatrick et al., 2016) protect old skills with output and weight penalties, and LoRA learns less and forgets less (Biderman et al., 2024); RL's Razor (Shenfeld et al., 2025) ties forgetting to KL from the original. On-policy distillation from the original model restores lost skills (Lu, 2025), continual variants add replay (Chen et al., 2026), and EAFT gates loss by entropy (Diao et al., 2026).

**Cross-model transfer.** Logit-space methods align vocabularies (Boizard et al., 2024; Zhang et al., 2024), model fusion merges distributions (Wan et al., 2024), and model stitching joins representations (Lenc and Vedaldi, 2014; Bansal et al., 2021). Residual-learning distillation projects teacher states for the student (On et al., 2026); steering-vector distillation stores a training-time intervention in a LoRA (Blank et al., 2026).

## 7 Conclusion

Teaching a small model through a state corrector, telepathy-like, beats copying its teacher online: the same closeness in a sixth of the items, a closer student and a LoRA that wins live. Most of the gain is the corrector's form, a small part the teacher's states. In coding, copying verified answers teaches best; within the family it also forgets, and a KL pull toward the original student is the cheapest fix.

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

Code, per-problem results, over 77,000 graded answers, a notebook that rebuilds Table 1, Figure 2 and the live table, and the pre-registration log (addenda 165-208) are public at https://github.com/syn-ack-ai/copy-correct-rewrite (Apache-2.0). Not included: text corpora, competition data, problem statements and weights.

## Appendix B AI use

The author carried out this study with Claude Opus 5.5 (Anthropic) as an AI research assistant, joined in the final days by Claude Fable 5.2 for an audit of the draft against the results, the reruns it prompted and the revision. The author set the research direction, chose the questions and approved each pre-registered experiment; the models designed and ran the experiments, wrote the code and drafted the text. Separate Claude instances reviewed new code and drafts, and verified the numbers.

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
| 189 (online setup D: seeds, controls) | 6 of 6 for seed 2 and the controls (on the 4090, D beat the self-corrector 12 vs 1 and the blend 11 vs 2; reruns in 205); C seeds 2-3 broke the D-over-C reading | not met: D and C tie on decisions in both same-GPU pairs (discordant decisions 16 vs 18) |
| 190 (fitting-text control for the corrector anchor) | 4 of 5 | met: the anchor's gain is its fitting text |
| 192 (online vs offline SFT teachers) | 2 of 3 scored | copying breaks more and ends further from E4B: online SFT +6 (13 broken; D beats it 18 vs 6, p = 0.023), offline SFT +14 (10 broken) |
| 193 (KL strength sweep) | 2 of 4 | not met: λ 0.3 holds, λ 3 costs held solves |
| 194a (longer-cap re-grade of capped losses) | 6 of 6 | — (9% of losses recovered) |
| 196 (online setup D live on 80 held-out agent tasks) | 4 of 5 | not met: 11 vs 5 of 160 (p = 0.34); timeouts failed the loop gate |
| 197 (disagreement-gated online teacher) | 2 of 4; 197c: 2 of 2 | gating explains D's decision gains (D ahead by 2, 0 and 3 net at seeds 1-3); D's students are closer and faster in every seed |
| 198 (where the corrector teacher pulls) | 5 of 7 | neither selective nor directional |
| 199 (corrector fitted 4× longer) | 4 of 5 | met: E4B's states add more with more fitting (cross lead over self 0.029 → 0.054 held) |
| 200 (realizability; imitable corrector) | 3 of 4; imitable fits 1 of 2 | E4B-specific write about half predictable from E2B's states at 1×, less at 4× (R² 0.53, 0.37); a penalty on the unpredictable part keeps the lead over the self-corrector with writes a quarter the size (R² 0.73) |
| 201 (the 4× corrector as online teacher) | 3 of 4 | met: its student ends 0.010 closer to E4B (0.742 vs 0.752) at the same net (+17 vs +18) |
| 202 (the 4× D LoRA live on the 80 tasks) | 4 of 4 | solve bar met exactly (13 of 160 vs 5 plain and 5 C; by task 9 vs 2, p = 0.065), loop gate missed narrowly (error endings 17 vs a bar of 16; 1× D 37) |
| 206 (the online teachers on one GPU type; seeds 1-2 so far) | so far: 1 held, 2 failed, 6 open | not met at two seeds: the cross-model corrector's students lead the self-corrector's by less than the bars (held CE 0.002-0.004 against 0.005; net 4.5-5 on average against 6) |
| 206c (the same runs paired by row and by decision; added after two seeds) | set before seed 3 | so far: cross-model closer in 4 of 4 pairs, 0.0019 nats [0.0012, 0.0026]; 19 vs 7 decisions (p = 0.029) |
| 207 (the live test with a null adapter; the second adapter pair still running) | so far: 4 of 5 scored | met on solves: D over the null adapter by task 12 vs 0 (p = 0.0005) with 4.75 times its share of solved episodes, and over plain E2B 9 vs 1 (p = 0.021); loop gate failed (error endings 37 per 160 against the null adapter's 9 plus 6) |
| 203 (dense rank-64 LoRAs) | — | queued |

Addenda 191 and 195 (per-user and lottery-ticket grafts) belong to a separate study.

## Appendix D Further results

**Correctors.** Sizes and fitting rows are in Table D1. For Qwen, LoRAs distilled from correctors fitted 2-8× longer solved no more contest problems (5 of 6 fewer, none significantly; one seed) but came closer to E4B on held agent replies (0.72 to 0.65), with more drift on untaught prose. The online correctors (Section 4.1) applied the do-no-harm term wherever student and teacher picked the same token. Depths are nominal, rounded down to a layer: with E2B as the student, the corrector reads E4B's layers 20 and 27 of 42, or E2B's own 17 and 22 of 35, and writes from E2B's layer 12 (Qwen: 9 of 28; E4B: 14 of 42). Every corrector, cross-model or self, also reads the student's own state at the write layer.

Table D1. Corrector parameters and fitting rows by student.

| Student | Self-corrector | Cross-model corrector (reads) | Fitting rows |
|---|---|---|---|
| Qwen3-1.7B, coding | 78M | 83M (E4B) | agent replies 50%, human prose 15%, Python code 15%, multiple-choice questions 10%, stories 10% |
| Gemma 4 E2B, coding | 65M | 75M (E4B) | as above |
| Gemma 4 E4B, coding (Section 4.6) | 121M | 110M (E2B) | as above |
| Gemma 4 E2B, online (Sections 4.2-4.4) | 69M | 83M (E4B) | agent replies 40%, E2B's own samples 20%, prose 15%, code 10%, questions 10%, stories 5% |

**Agent-turn LoRAs (Section 4.5).** Figure D1 shows the problems each form gained and lost. SFT with a KL pull toward the original lost the fewest solved problems (66) but gained the least (against SFT 90 vs 195, p = 5e-10). The cross-model form's live advantage over the self form was only 0.05-0.06 nats on agent replies. In a separate coding run the anchors did not lose to SFT (held solved: SFT 143, anchors 146-152).

![Gains and losses](figures/fig2_wide_coding.png)

Figure D1. LoRAs trained on E4B's agent turns, on LiveCodeBench (1,739 problem-runs): problems gained and lost against the untrained student.

**Gemma 4 E2B, every recipe (Section 4.6).** Anchors to fresh Gemma correctors moved E2B toward lower loss (prose Δ −0.58 to −0.60), raising held solves by at most 12 (p = 0.22) at about 25 fewer fixes per seed. Table D2 adds the shortest rows, the runbook rows, the less-training arms (addendum 188), λ = 3, the cross-model anchor and the fitting text with the KL pull. E2B's training loss on E4B's rows started low (0.28 nats over its first 50 steps), barely fell in the first epoch and reached 0.07 in the second; Qwen's ended at 0.24. Tests marked "on trained problems" count the trained problems solved by one recipe only: λ 0.3 against λ 1, 94 vs 48 (p = 0.0001). The KL pull at λ 0.3 left small rises in untaught loss (prose −0.01 to +0.02 per seed, stories +0.04 to +0.06), 92-99% below SFT's. Stopped there after one epoch, E2B's prose damage fell from +0.69 to +0.09, yet it moved twice as far as Qwen (KL 0.16 against 0.074), so part of the effect is the student. Stopping early kept no held gain and lost 122 solved problems (SFT 47). Both partly trained arms lost mostly trained problems (75 and 49, against 18), which SFT's second epoch turns into E4B's verified answers. The KL pull did not reduce the fitting text's drift (prose KL 0.81): it covers only the coding replies. The Gemma correctors left E2B's coding loss unchanged.

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
| *KL to the original (187, 193)* | | | | | |
| Shortest + KL to the original, λ 0.3 | 70.0 | 108.0 | 15.0 | 0.031 | +0.01 |
| Shortest + KL to the original, λ 1 | 69.0 | 90.7 | **13.3** | 0.016 | −0.04 |
| Shortest + KL to the original, λ 3 | 60.7 | 59.7 | 24.3 | **0.014** | −0.05 |
| *Corrector anchors (187)* | | | | | |
| Shortest + self-corrector anchor | **72.0** | 95.3 | 16.0 | 0.33 | −0.60 |
| Shortest + cross-model anchor | 69.7 | 99.3 | 15.3 | 0.32 | −0.58 |
| *The corrector's fitting text (190)* | | | | | |
| Shortest + the corrector's fitting text | 71.3 | 105.0 | 16.3 | 0.83 | −0.75 |
| Shortest + fitting text + KL to the original | 68.7 | 83.3 | 17.3 | 0.81 | −0.76 |

**E2B teaching E4B (Section 4.6).** E2B's 148 passing answers were the training rows. SFT raised E4B's prose loss by 0.81; every anchor removed the damage. The self-corrector anchor beat SFT 52 vs 26 on problems solved by one only (p = 0.004), and every anchor lost fewer solved problems than SFT. Cross-model and self anchors tied in both directions (E2B student: 209 vs 216 held solved; E4B student: 249 vs 250).

**Qwen drift (Section 4.6).** Table D5 gives the three-seed means and Figure D2 each seed. On the shortest rows, KL to the original moved Qwen 6-9× less than the other copying recipes on prose and stories (3× on code); the corrector anchor moved it as far as SFT, toward lower loss and softer predictions, from the corrector's prose fit. Neither anchor cost held-out solves (Table D6), though both fixed about 6% fewer training problems.

Table D5. Qwen's drift on untaught text, three-seed means.

| Recipe | KL from the original (prose) | Change in loss (nats): prose / code / stories |
|---|---|---|
| Plain SFT | 0.074 | −0.06 / −0.08 / −0.10 |
| Shortest + KL to the original | **0.011** | +0.01 / −0.10 / −0.02 |
| Shortest + corrector anchor | 0.089 | −0.24 / −0.21 / −0.51 |

Table D6. Qwen with the two anchors, three seeds pooled (fresh-question p(true): mean probability of the true answer on held-out multiple-choice questions).

| Recipe | Held solved (of 564) | Lost (of 384) | Fresh-question p(true) (untrained 0.513) |
|---|---|---|---|
| Shortest rows | 160 | 48 | 0.527 |
| Shortest + KL to the original | 159 | **40** | 0.521 |
| Shortest + corrector anchor | 157 | 52 | 0.554 |
| Plain SFT | 149 | 58 | 0.557 |

![Prose drift](figures/fig4_untaught_prose.png)

Figure D2. Drift on untaught human prose, one point per seed.

**Runbook rows (Section 4.6).** E4B turns each verified solution into a numbered plain-English plan, the student solves the problem with the plan in its prompt, and passing solutions are trained on without the plan. The 228 rows kept are the student's own text: similarity to E4B's solution 0.20, 0.32 nats under the untrained student (E4B's rows 1.05), 110 tokens (350). They solved slightly fewer new held-out problems (83 against 90-95, within noise), lost the fewest training-pool problems, barely moved the student (prose KL 0.022) and lowered its entropy the most (−0.23 nats on prose). The corrector anchor reversed the rise in untaught loss in every seed (prose −0.23 to −0.25; fresh-question p(true) +0.028) at the cost of 39 fewer fixed problems and 10 fewer held solves. With E2B the rows lost their advantage: its rewrites were no closer to it than E4B's text (0.38 vs 0.27 nats). A screen rejected plans with code or over 10% of the solution's code 4-grams. On trained problems solved by one recipe only: runbook rows against SFT 133 vs 84 (p = 0.001; on the fixed problems alone 105 vs 73, p = 0.02); with the self-anchor against without, 29 vs 71 (p = 3e-5); for E2B, runbook rows against SFT 61 vs 111 (p = 0.0002). The two anchors of Table D6 fixed 382 and 385 training problems against 406 for the shortest rows.

Table D3. Runbook rows, three seeds pooled; prose ΔCE per seed.

| Recipe | Fixed | New | Held solved | Lost (of 384) | Prose ΔCE (per seed) |
|---|---|---|---|---|---|
| Plain SFT | 389 | 90 | 149 | 58 | −0.05 to −0.07 |
| Runbook rows | **421** | 83 | 143 | 39 | **+0.10 to +0.16** |
| Runbook rows + self-anchor | 382 | 71 | 133 | 37 | −0.23 to −0.25 |

**On-policy distillation (Section 4.5).** The first self-corrector teacher fit E4B's solutions at 0.70 nats (the student alone 1.00), the refit one at 0.49. The refit teacher's student beat the weak teacher's in every seed but lost more training problems than SFT (59 vs 42). As a rewriter, the refit corrector copied E4B's text (similarity 0.98-0.99). The teachers' live counts come from the rewrite sessions, whose batching differs from the evaluation sessions' (the untrained student solves 69 there, 75 here).

Table D4. Held solved, three seeds pooled.

| Recipe | Weak teacher | Refit teacher |
|---|---|---|
| On-policy only | 116 | 148 |
| SFT + on-policy | 117 | 158 |

![Teachers and students](figures/fig5_teacher_cap.png)

Figure D3. Each corrector teacher run live, and the on-policy student it trained.

![Teaching and forgetting](figures/fig3_teach_vs_forget.png)

Figure D4. Coding recipes on Qwen by held-out rank; best is bottom right.

**Answers cut at the cap.** Given 768 more tokens, 59 of the 176 lost answers cut at the cap passed (9% of all losses; 18-25% for refit-teacher recipes).

**Online (Sections 4.2-4.3): earlier runs on three GPU types.** Before the one-GPU table the setups ran on a 4090, on A100s and on a G4 (Table D7; A and B have no keep + replay). Keep + replay cut A's drift six-fold. C sits closer to the teacher than A yet flips fewer of E2B's top choices. On decisions, seeds and hardware decided: D's seed-1 lead over C on the 4090 (17 vs 6, p = 0.035) vanished in a same-stream A100 rerun (9 vs 7), and C led at seeds 3 (7 vs 11) and 2 (4 vs 14; D on the 4090, C on the A100). Closeness and speed reproduced: in every seed D's students ended 0.017-0.024 nats closer to E4B on held text than C's, predicted each new item better before learning it (prequential gain 0.062-0.065 against 0.035-0.040) and reached C's final held cross-entropy by item 50-100 of 600. The pull-matched blend (E4B's distribution mixed with E2B's to the corrector's average pull) and the gated teacher (E4B's distribution only where its top token differs from E2B's) have no learned module. The gated teacher netted +16, +17 and +12, within 3 of D on average, so gating on disagreements explains D's decision gains over C; its students stayed near C's closeness (0.764-0.766) and speed (0.044-0.047). On the 4090 at seed 1, D beat the self-corrector 12 vs 1 (p = 0.003) and the blend 11 vs 2 (p = 0.022) on the 220 decisions.

Table D7. Online setups across three GPU types; seeds listed per row.

| Setup | Seeds | Held CE (from 0.825) | Untaught drift | Gate net (fixed / broken) |
|---|---|---|---|---|
| A Teacher | 1 | 0.787 | 0.089 | +14 (25 / 11) |
| B Corrector | 1 | 0.753 | 0.055 | +12 (16 / 4) |
| C Teacher + keep + replay | 3 | 0.768-0.775 | 0.012-0.014 | +8 (A100 +16), +27, +19 |
| D Corrector + keep + replay | 3 | 0.749-0.752 | 0.014-0.042 | +19 (A100 +18), +17, +15 |
| Self-corrector + keep + replay | 1 | 0.755 | 0.025 | +8 (13 / 5) |
| Pull-matched blend + keep + replay | 1 | 0.781 | 0.014 | +10 (14 / 4) |
| Gated teacher + keep + replay | 3 | 0.764-0.766 | 0.011-0.015 | +16, +17, +12 (G4) |

**Online: one GPU type (Table 1).** Every run of Table 1 is a G4 session with the same inputs and stream order per seed, scored on the 4090's partition of the 220 decisions (on a G4, E2B alone differs from it on 3). Fixed / broken per run: C 21 / 8; S 13 / 3 and 21 / 6; D 23 / 4 and 19 / 3; DI 20 / 1; S4 22 / 5 and 22 / 7; D4 24 / 4 and 25 / 4. Untaught drift (KL from the original E2B, mean of the last six checkpoints; single checkpoints move between 0.010 and 0.047 within a run): C 0.013; S 0.024, 0.021; D 0.022, 0.013; DI 0.021; S4 0.022, 0.017; D4 0.017, 0.013.

**Online: cross-model against self.** The pre-registered rule (addendum 206) asked, at matched fitting, for a held cross-entropy at least 0.005 lower in every seed or 6 more net decisions on average with a lead in each seed. At two seeds neither is met: the held cross-entropy differences are 0.003 and 0.002 (1× fit), 0.003 and 0.004 (4×); the net differences 9, 1, 3 and 6. After two seeds we added a paired comparison (amendment 206c; its three-seed rule and predictions were fixed before any seed-3 result was read). Every student is scored on the gate's same 240 held rows, from 118 episodes in 17 repositories; intervals are bootstraps over repositories. D minus S: −0.0027 [−0.0037, −0.0016] and −0.0010 [−0.0018, −0.0001]. D4 minus S4: −0.0011 [−0.0021, 0.0000] and −0.0028 [−0.0040, −0.0015]. The four pairs together: −0.0019 [−0.0026, −0.0012], the cross-model student closer in 15 of the 17 repositories. The intervals describe these students; for a claim about the teacher the pair is the unit, and 4 of 4 pairs has p = 0.125. For scale on the same rows: D minus C −0.0170 [−0.0202, −0.0131]; S minus C −0.0144 [−0.0177, −0.0102]; the 4× fit minus the 1× fit −0.0081 (D) and −0.0080 (S); DI minus D +0.0008 [−0.0003, +0.0019]. Per decision over the four pairs, more cross-model than self-corrector students matched E4B on 19 decisions and fewer on 7 (p = 0.029); pair by pair, D over S 11 vs 2 (p = 0.022) and 3 vs 2, D4 over S4 8 vs 5 and 8 vs 2. The mean prequential gain is higher for the cross-model student in all four pairs, by 0.0001-0.0035, but not over the first 100 items (0.047 against 0.046, 0.044 against 0.049, 0.050 against 0.049, 0.049 against 0.052).

**Online: the correctors as teachers.** Run live on held rows, the cross-model corrected E2B is 0.029 nats closer to E4B than the self-corrected E2B (0.754 against 0.783), and 0.054 closer when both are fitted four times longer (2,400 steps, more code and story rows; 0.714 against 0.768). The part of the cross-model corrector's write that is specific to E4B is about half predictable from E2B's own states (R² 0.53; 0.37 at 4×). DI's corrector is fitted with a penalty on the unpredictable part: R² 0.73, writes a quarter the size, and the same lead over the self-corrector on held rows (0.027). On its own fitting points, D's teacher put less of its pull where E4B and E2B disagree than the blend (49% vs 57%) and moved E2B's top token to E4B's less often (26% vs 40%).

**Online: copying.** Copying online, SFT on E4B's logged replies with keep + replay (seed 1, A100), fixed 19 decisions, broke 13 and ended further from E4B on held text than the original E2B (0.900 against 0.824); D beat it on 18 decisions against 6 (p = 0.023) and C on 15 against 5 (p = 0.041). Offline SFT on the same 600 items (two passes) netted +14 (24 fixed, 10 broken), but on the gate's replies both SFT students sat further from E4B than the original E2B (1.10 offline, 0.91 online, against 0.83), where C, the gated teacher and D brought it closer (0.78, 0.77, 0.76).

**Live (Section 4.4).** Each attempt runs the 80 tasks once in the competition's harness; E4B solved none of them. The null adapter is D's adapter with its output matrices set to zero: untrained E2B through the adapter serving path. That path took 1.8 times plain E2B's wall time (85-100 minutes per attempt against 43-55) but ended no more episodes in errors and solved no more, so the LoRAs' time-outs come from their own longer episodes (Table D8). By task on the solve rate: D over the null adapter 12 vs 0 (p = 0.0005), over plain E2B 9 vs 1 (p = 0.021), over C's LoRA 9 vs 3 (p = 0.15); D4's LoRA over the null adapter 8 vs 1 (p = 0.039), against D's 6 vs 5. The pre-registered rule (addendum 207) reads D over four attempts and asks for p < 0.05 against the null adapter and twice its share of solved episodes; D has 4.75 times (5.9% against 1.25%). The loop gate (error endings per 160 at most the null adapter's plus 6) fails for D (37) and narrowly for D4's LoRA (17).

Table D8. The live test: 80 held-out tasks per attempt.

| Arm | Attempts | Solved | Tasks solved at least once | Error endings (time-outs) | Model calls per task |
|---|---|---|---|---|---|
| Plain E2B | 4 | 7 of 320 | 5 | 19 (1) | 13-15 |
| Null adapter | 2 | 2 of 160 | 2 | 9 (4) | 15 |
| C's LoRA | 2 | 5 of 160 | 4 | 25 (20) | 22-23 |
| D's LoRA | 4 | 19 of 320 | 12 | 73 (48) | 20-23 |
| D4's LoRA | 2 | 13 of 160 | 9 | 17 (11) | 16 |
