# Copy, Correct, or Rewrite? Telepathy-Like Teaching of a Small Agent Model and What Keeps It From Forgetting

Benjamin Babeshkin

*Lessons for training Gemma 4 E2B/E4B developer agents from a stronger teacher.*

## Abstract

A small agent model taught by a larger one should gain the teacher's skill without losing its own. We study *telepathy-like teaching*: the teacher reaches the student through hidden states, not outputs. A small network, the *state corrector*, reads the teacher's hidden states (Gemma 4 E4B) and corrects the student's (Gemma 4 E2B). The corrected student then teaches a LoRA; only the LoRA is kept.

On a stream of agent decisions this beat copying the teacher's output distribution: students were as close to E4B after a sixth of the stream as copying students at its end, and finished closer in every seed. Most of that gain does not need the teacher's states: a corrector that sees only E2B's own states taught almost as well; E4B's states added a small, consistent extra. On 80 live agent tasks the LoRA beat untrained E2B (ahead on 12 tasks, behind on none; p = 0.0005) but not LoRAs taught by copying.

On contest coding, taught on verified answers, the corrector's students matched copying (76 and 75 of 188 held problems against 76 and 69; untrained 57) and left fewer answers unfinished. Offline, copying from a same-family teacher damaged untaught text; a KL penalty toward the original student removed the damage at no cost in held-out solves.

## 1 Introduction

A submission to the Gemma 4 Developer Agent competition (Kaggle, 2026) is one Gemma 4 model, here the small E2B or E4B (Google, 2026), plus LoRA adapters (Hu et al., 2021) of rank at most 128, so any gain from a stronger teacher must be stored in an adapter. Distillation (Hinton et al., 2015) is the main lever; we ask which route teaches the most and forgets the least.

Can a teacher's internal states teach better than its outputs? Swarm Telepathy (Alabdulkareem et al., 2026) moves one model's KV cache into a frozen receiver of another family. We train instead: a *state corrector* turns the teacher's hidden states into a correction of the student's, and the corrected student teaches a LoRA. In this *telepathy-like teaching* the teacher reaches the student as states, not tokens, and is gone at inference. Our contributions:

- Online, telepathy-like teaching beats copying the teacher's distribution on speed and closeness: the same closeness in a sixth of the stream, and a closer student in every seed. Live, its LoRA beats untrained E2B (p = 0.0005) but not the copying teacher's.
- Most of that gain comes with the corrector's form, shared by a corrector reading only the student's states; E4B's states add under a tenth.
- In coding, the corrector's students match copying on held problems and leave fewer answers unfinished; they are not better.
- Same-family copying damages untaught text; a KL pull toward the original student removes the damage without costing held-out solves.
- Code, results, 100,000+ graded answers and pre-registrations: https://github.com/syn-ack-ai/copy-correct-rewrite.

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

## 4 Experiments

### 4.1 Setup

**Models.** The teacher is Gemma 4 E4B (4-bit QAT, dequantised to bf16). The online student is Gemma 4 E2B; the coding students are E2B and Qwen3-1.7B (Yang et al., 2025), which has a different tokenizer.

**Online stream.** E2B with a rank-16 LoRA meets 600 unseen decision points from the competition's harness (the context before a tool call in E4B's failed runs). Correctors are fitted on the stream's training repositories; held measures use 17 repositories neither saw. The teachers: E4B's distribution (C); the cross-model corrector (D, telepathy-like); the self-corrector (S); D4 and S4, with correctors fitted four times longer; and DI, a cross-model corrector penalised for writes E2B's own states cannot predict. All use keep + replay.

**Online measures.** *Held CE*: cross-entropy to E4B on held repositories (untrained E2B: 0.825). *Prequential gain*: nats per token gained over the original E2B on each item before learning it. *Gate*: 240 held rows, 220 with a tool decision; *fixed* counts the 52 decisions where E2B alone differs from E4B, *broken* the other 168, and *net* is their difference. One teacher and seed on two GPU types differed by 8 net decisions, so every teacher ran on one GPU type (Appendix D).

**Coding.** Training rows are 299 older LiveCodeBench problems (Jain et al., 2024; May 2023 to Sep 2024), each with E4B's own solution passing every hidden test. Every offline recipe trains a rank-64 LoRA (alpha 128) on the MLPs of layers 9-27 (Qwen; E2B 12-34) at lr 3e-4 for 299 steps of two rows, in the same order. Answers are greedy with a 768-token cap. *Fixed*: of the 200 training problems untrained Qwen fails (E2B: 151), how many it now solves. *Held solved*: of 188 newer problems (Sep 2024 to Apr 2025). *Lost*: previously solved problems now failed. *Drift*: KL(original ‖ trained) and the change in cross-entropy on held-out human prose, Python code and stories, never trained on. Counts are pooled over three seeds, with exact sign tests on discordant problems; comparisons were pre-registered (Appendices A, C) unless marked.

### 4.2 Telepathy-like teaching is faster and closer than copying

Table 1. Online teachers on one GPU type: mean of three seeds (range).

| Teacher | Held CE (from 0.825) | Prequential gain | Gate net |
|---|---|---|---|
| C Copying E4B's distribution | 0.771 (0.767-0.774) | 0.037 | +16.7 (+13 to +19) |
| S Self-corrector | 0.755 (0.754-0.756) | 0.061 | +14.0 (+10 to +17) |
| D Cross-model corrector | 0.753 (0.752-0.754) | 0.063 | +18.3 (+16 to +20) |
| DI Cross-model, imitable | 0.753 (0.751-0.754) | 0.062 | +16.0 (+9 to +20) |
| S4 Self-corrector, 4× | 0.745 (0.744-0.746) | 0.068 | +15.7 (+15 to +17) |
| D4 Cross-model, 4× | **0.741 (0.741-0.742)** | 0.068 | **+21.3 (+20 to +23)** |

Within the first 50-100 of the 600 items (the imitable corrector once 150) every corrector student was as close to E4B on held text as the copying student at the end, and it finished 0.011-0.033 nats closer in every seed (Table 1; Figure 2, left). On decisions the standard-fit correctors tie with copying (Table 1); the copying student broke the most (8 to 12 of 168; the corrector students 1 to 7).

Fitting the corrector four times longer brought its student another 0.009-0.013 nats closer; the cross-model one's was the closest in every seed and the only student ahead of the copying one on decisions in every seed (by 3 to 7, never significant). The imitable corrector (DI), with writes a quarter the size, matched the standard cross-model corrector on closeness.

![Online curves](figures/fig7_online.png)

Figure 2. Left: closeness to E4B over the stream. Right: what each ingredient adds.

### 4.3 What the teacher's states add

Is this telepathy, or would any corrector do? Mostly the latter, though E4B's states appear to add a slight edge.

The self-corrector never sees E4B's states, yet its student is as fast and has most of the gain: on the gate's 240 held rows it is 0.019 nats closer to E4B than the copying student, against the cross-model student's 0.020 (Figure 2, right).

The edge is under a tenth of that lead, but consistent. Paired on those rows, the cross-model student was closer than its self-corrector twin in all six pairs (three seeds, two fits; p = 0.031) by 0.0017 nats (95% interval over repositories 0.0009-0.0025), and matched E4B on more decisions (22 vs 6, p = 0.004). We registered this six-pair test after two seeds; our first bar, 0.005 nats in every seed, was not met.

The teachers differ more: with the corrector connected, the cross-model corrected E2B is 0.03 nats closer to E4B than the self-corrected E2B (0.05 when fitted longer), yet their students keep 0.002 either way. Whether a student can keep more of this edge needs further investigation.

### 4.4 Live agent tasks

We ran each LoRA in the competition's harness on 80 held-out tasks E4B first failed (Table D8). The five LoRAs taught by copying or by the cross-model corrector solved 65 of 960 task attempts (6.8%), plain E2B 7 of 320 and a null adapter (zeroed weights) 2 of 160; by task they beat the null adapter 16 vs 1 (p = 0.0003) and plain E2B 16 vs 2 (p = 0.001), and the cross-model corrector's first LoRA alone beat the null adapter 12 vs 0 (p = 0.0005). The teachers are level: corrector-taught LoRAs solved 6.7% and copying-taught 6.9%, and two adapters of one teacher differed more than the teachers did (5 and 17 of 160 for copying). The self-corrector's students solved fewer, 9 of 320, against 32 of 480 for their cross-model twins (by task 11 vs 3, p = 0.057; our registered bar, p < 0.05, was not met).

### 4.5 In coding, the corrector matches copying

**Online.** E2B met the 299 training problems as a stream (598 items) and learned only from its answers that pass the tests, or E4B's verified solution when none did. Correctors were fitted on verified solutions a reviewer model judged direct or their author model rewrote. Untrained E2B solves 57 of 188 held problems; E4B, 74. Over two seeds the cross-model corrector's students solved 76 and 75, students of E4B's distribution (passing answers only) 79 and 67, of its verified text 76 and 69 (Table D9); the self-corrector's, 73 and 75. No pair separates (corrector against distribution: 25 vs 20, p = 0.55). The correctors' students left fewer answers unfinished (79-101 of 487, against 106-157) but solved fewer training problems than copying the text (43 vs 84, p = 0.0003). A hybrid closes that gap (86 vs 86 over three seeds) by copying E4B's verified text where no answer passes; it solved 80, 69 and 77 held problems, level with copying (43 vs 37, p = 0.58). A shorter reviewer-picked solution instead left fewer answers unfinished (59-72 against 75-116) without solving more.

**Offline.** Earlier (Figure D1, Table D6), Qwen LoRAs distilled through the corrector trailed plain SFT (p = 0.04 and 0.0001), E4B's states added nothing a stored adapter kept, the shortest passing solutions taught best (held 160 vs 149), and on-policy distillation stopped at its teacher (Figure D3).


### 4.6 Copying within the family forgets, and an anchor fixes it

With E2B in place of Qwen, the student starts stronger (57 held solves alone, against 25) and finds E4B's solutions familiar (0.27 nats per token, against 1.05). Every copying recipe damaged untaught text (Table 2, Figure 3): 2-13× Qwen's KL, sharper predictions (entropy −0.3 to −0.8) and higher loss on prose, code and stories. SFT cost E2B few solved problems (8% vs Qwen's 15%; cf. Wu et al., 2025).

Table 2. Gemma 4 E2B student, three-seed means (alone: 57 held).

| Recipe | Held solved | Fixed (of 151) | Lost (of 205) | Prose KL | Prose Δ loss |
|---|---|---|---|---|---|
| SFT | 68.0 | **119.7** | 15.7 | 0.29 | +0.69 |
| Shortest + KL to the original, λ 0.3 | 70.0 | 108.0 | 15.0 | 0.031 | +0.01 |
| Shortest + KL to the original, λ 1 | 69.0 | 90.7 | **13.3** | 0.016 | −0.04 |
| Shortest + self-corrector anchor | **72.0** | 95.3 | 16.0 | 0.33 | −0.60 |
| Shortest + the corrector's fitting text | 71.3 | 105.0 | 16.3 | 0.83 | −0.75 |

The likely cause is over-fitting familiar text: E2B's training loss fell from 0.28 nats to 0.07 (Qwen's ended at 0.24).

Every anchor at λ = 1 removed the damage in every seed. KL to the original kept E2B in place at no cost in held solves (207 vs 204) but a quarter fewer fixes (272 vs 359). A lighter pull (λ = 0.3) removed over 90% of the damage and fixed 52 more (p < 0.0001). The corrector anchor moved E2B as far as SFT, toward lower loss; training directly on the corrector's fitting text matched it (held 214 vs 216, p = 0.90) while moving E2B much further (prose KL 0.83 vs 0.33): the anchor's gain is its fitting text, delivered with less change.

With E2B's answers teaching E4B (weak-to-strong; Burns et al., 2023), SFT barely helped and every anchor did better (Appendix D). On Qwen, anchors cost no held-out solves. Rows the student rewrote from E4B's plans (runbook rows, after Terminal-RSI; Yang et al., 2026) fixed the most training problems (421 vs 389; p = 0.02) but alone raised loss on untaught text in every seed; the corrector anchor reversed that.

![E2B per recipe](figures/fig9_e2b.png)

Figure 3. Qwen3-1.7B (gray) and Gemma 4 E2B (colour) per recipe.

## 5 Discussion

**Telepathy-like teaching.** The corrected student teaches in the student's own states: E2B reaches in a sixth of the stream what copying E4B's distribution gives at its end, and its LoRA beats untrained E2B live. Most of the speed and closeness needs no teacher at teaching time: the self-corrector, fitted once on E4B's replies, carries it, and E4B's states add under a tenth. Live, the self-corrector's students did not beat untrained E2B. Why the form helps is open (Appendix D).

**What the student keeps.** The student seems to absorb only what it can compute. Longer fitting nearly doubled what E4B's states add to the corrected model and made that part less predictable from E2B's own states (R² 0.53 to 0.37); the student kept no more. In coding, the cross-model corrector's students did not separate from the self-corrector's. If the limit is the student's, a larger student should keep more, and the rest needs a live link to the teacher.

**For training small agents.** Teach online through a corrector fitted longer on verified, direct answers, with keep + replay; where the student's own answer fails its tests, copy the verified text. For stored coding skill, copying the teacher's shortest verified answers with a light KL pull toward the original student does as well; measure forgetting on held-out human text with entropy, since loss on teacher text rewards imitation.

**Limitations.** One online pair with a shared tokenizer; on one GPU type, three seeds per teacher. Closeness to E4B is a proxy: decisions tie with the plain teacher. Live tasks are ones E4B first failed, with 2-19 solves per arm. In coding, two students; online, 2-3 seeds; pooled p-values are descriptive; differences under about 13 held solves are noise; 9% of losses passed with the 768-token cap doubled (Appendix D). The online correctors, unlike the coding ones, were fitted without eq. (3)'s size term and saw E2B's own samples. Online teaching across tokenizers (as for Qwen) is untested.

**Ethical considerations.** Prose and stories for fitting and drift come from locally held books, not released. Model-written code ran only in sandboxes. AI use: under the author's direction, Claude Opus 5.5 and Fable 5.2 (Anthropic) designed and ran the experiments, wrote the code and drafted the text (Appendix B).

## 6 Related Work

**Distillation.** GKD (Agarwal et al., 2023), MiniLLM (Gu et al., 2023) and DistiLLM (Ko et al., 2024) distil on the student's own samples with reverse or skewed KL. Reachable targets drive ProKT (Shi et al., 2021), DIST (Huang et al., 2022) and PACED (Xu et al., 2026).

**Training on own solutions.** ReST (Gulcehre et al., 2023), ReST-EM (Singh et al., 2023), rejection-sampling fine-tuning (Yuan et al., 2023) and STaR (Zelikman et al., 2022) retrain on verified own samples; Terminal-RSI (Yang et al., 2026) adds plan-guided self-rewriting. The damage we report resembles entropy collapse in reasoning RL (Cui et al., 2025).

**Forgetting.** Learning without forgetting (Li and Hoiem, 2016) and EWC (Kirkpatrick et al., 2016) protect old skills with output and weight penalties, and LoRA learns less and forgets less (Biderman et al., 2024); RL's Razor (Shenfeld et al., 2025) ties forgetting to KL from the original. On-policy distillation from the original model restores lost skills (Lu, 2025), continual variants add replay (Chen et al., 2026), and EAFT gates loss by entropy (Diao et al., 2026).

**Cross-model transfer.** Logit-space methods align vocabularies (Boizard et al., 2024; Zhang et al., 2024), model fusion merges distributions (Wan et al., 2024), and model stitching joins representations (Lenc and Vedaldi, 2014; Bansal et al., 2021). Residual-learning distillation projects teacher states for the student (On et al., 2026); steering-vector distillation stores a training-time intervention in a LoRA (Blank et al., 2026).

## 7 Conclusion

Teaching a small model through a state corrector, telepathy-like, beats copying its teacher online: the same closeness in a sixth of the items and a closer student; live, LoRAs from either teacher beat the untrained student. Most of the gain is the corrector's form, a small part the teacher's states. In coding it matches copying; copying within the family also forgets, and a KL pull toward the original student is the cheapest fix.

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

Code, per-problem results, over 100,000 graded answers, a notebook that rebuilds Table 1, Figure 2 and the live table, and the pre-registration log (addenda 165-209) are public at https://github.com/syn-ack-ai/copy-correct-rewrite (Apache-2.0). A Kaggle notebook runs the rebuild from that repository at a pinned commit: https://www.kaggle.com/code/skypanther0/telepathy-like-teaching-of-a-small-agent-model. Not included: text corpora, competition data, problem statements and weights.

## Appendix B AI use

The author carried out this study with Claude Opus 5.5 (Anthropic) as an AI research assistant, joined in the final days by Claude Fable 5.2 for an audit of the draft against the results, the reruns it prompted and the revision. The author set the research direction, chose the questions and approved each pre-registered experiment; the models designed and ran the experiments, wrote the code and drafted the text. Separate Claude instances reviewed new code and drafts, and verified the numbers. DeepSeek V4.1 Flash read the coding correctors' fitting solutions as a reviewer (Appendix D); every solution was written by E4B or E2B.

## Appendix C Pre-registration scorecard

Each addendum in the public log states its predictions and, for some, a bar the result had to clear before running. A prediction is each statement registered with its own probability; "not scored" means not run, not computed, or without an outcome in the log; a prediction that held only in part counts as failed.

Table C1. Predictions that held, per pre-registered addendum.

| Addendum (question) | Predictions held | Pre-set bar |
|---|---|---|
| 169d (corrector forms vs dose-matched SFT, agent replies) | 0 of 4 | failed |
| 170 (online learning) | 4 of 6 held | — |
| 174 (SFT with anchors) | 4 of 6 held, 1 failed, 1 not scored (counted from the log's tables; it lists no outcomes); 174c: 2 of 3 held | no bar was set; the log's reading is on held problems solved, where states added nothing (cross-model anchor 146, self-anchor 152; 20 vs 26, p = 0.46); on new held problems, which the registered rule names, 89 against 87 |
| 176 (Qwen correctors fitted 2-8× longer; one seed, the second cancelled) | 1 of 5 held, 3 failed, 1 not scored | — (5 of 6 longer fits solved fewer contest problems, none significantly; closer on agent replies, 0.72 to 0.65, with more drift on untaught prose) |
| 178 (wide coding) | 0 of 2 held, 1 failed (reversed), 1 not scored | — |
| 179 (on-policy, shortest rows) | 2 of 6 held, 2 failed, 2 not scored | — |
| 181 (refit teacher, rewrites) | 9 of 12 | not met |
| 182 (runbook rows) | 9 of 12 held | missed by one |
| 183 (online setup D, first run: the corrector as teacher with keep + replay; one seed) | 6 of 6 held (read from 189's tables; the log lists no outcomes) | — |
| 184c (runbook rows + self-anchor) | 4 of 7 held | not met |
| 185 (SFT + KL to the original) | 5 of 6 | — |
| 186 (Gemma 4 E2B as the coding student) | 5 of 7 | — |
| 187 (two-way, fresh Gemma correctors) | 5 of 8 held | not met; cross-model = self |
| 188 (matched-loss run, E2B) | 6 of 9 held | not met: stopping early cut the damage but kept no held gain |
| 189 (online setup D: seeds, controls) | 7 of 9 held; failed at seed 3: 18 decisions fixed (at least 19 registered) and drift 0.042 (at most 0.035 registered). Amendments with their own predictions: 189b 1 of 3 held; 189c 1 of 1 held; 189f 1 of 4 held, 2 failed, 1 not scored; 189g 2 of 2 held (its other two are 197c's). On the 4090, D beat the self-corrector 12 vs 1 and the blend 11 vs 2; 205, which registered reruns of both, was not run: the self-corrector ran again as 206's S arm and the blend rerun was dropped; C seeds 2-3 broke the D-over-C reading | not met: D and C tie on decisions in both same-GPU pairs (discordant decisions 16 vs 18) |
| 190 (fitting-text control for the corrector anchor) | 4 of 5 | met: the anchor's gain is its fitting text |
| 192 (online vs offline SFT teachers) | 4 of 6 held, 1 failed, 1 not scored | not met: D's net leads both SFT arms by at least 4, but the paired test against offline SFT is 12 vs 8 (p = 0.50; the bar asks p < 0.1); copying breaks more and ends further from E4B: online SFT +6 (13 broken; D beats it 18 vs 6, p = 0.023), offline SFT +14 (10 broken) |
| 193 (KL strength sweep) | 2 of 4 | not met: λ 0.3 holds, λ 3 costs held solves |
| 194a (longer-cap re-grade of capped losses) | 6 of 6 | — (9% of losses recovered) |
| 196 (online setup D live on 80 held-out agent tasks) | 4 of 5 held | not met: D solved 11 of 160 attempts, plain E2B 5 and C 5; by task D over plain 7 vs 3 (p = 0.34); timeouts failed the loop gate |
| 197 (disagreement-gated online teacher) | 2 of 4; 197c: 2 of 2 | gating explains D's decision gains (D ahead by 2, 0 and 3 net at seeds 1-3); D's students are closer and faster in every seed |
| 198 (where the corrector teacher pulls) | 5 of 7 | neither selective nor directional |
| 199 (corrector fitted 4× longer) | 4 of 5 | met: E4B's states add more with more fitting (cross lead over self 0.029 → 0.054 held) |
| 200 (realizability; imitable corrector) | 4 of 8 held, 2 failed, 2 not scored (the imitable corrector's online student was not run as registered) | E4B-specific write about half predictable from E2B's states at 1×, less at 4× (R² 0.53, 0.37); a penalty on the unpredictable part keeps the lead over the self-corrector with writes a quarter the size (R² 0.73) |
| 201 (the 4× corrector as online teacher) | 3 of 4 | met: its student ends 0.010 closer to E4B (0.742 vs 0.752) at the same net (+17 vs +18) |
| 202 (the 4× D LoRA live on the 80 tasks) | 4 of 4 held | solve bar met exactly (13 of 160 vs 5 plain and 5 C; by task 9 vs 2, p = 0.065), loop gate missed narrowly on two of its three counts (error endings 17 vs a bar of 16, multi-submit episodes 13 vs 10; 1× D 37 error endings) |
| 206 (the online teachers on one GPU type, three seeds) | 8 of 11 held | not met: the cross-model corrector's students lead the self-corrector's by less than the bars (held CE 0.002-0.004 against 0.005; net 4.3 and 5.7 on average against 6) |
| 206c (the same runs paired by row and by decision; registered after two seeds, before seed 3) | 8 of 9 held | met: cross-model closer in 6 of 6 pairs (p = 0.031), 0.0017 nats [0.0009, 0.0025]; 22 vs 6 decisions (p = 0.004) |
| 207 (the live test with a null adapter, two adapters per teacher) | 7 of 9 held; rules: 1 of 3 met | met on solves: D over the null adapter by task 12 vs 0 (p = 0.0005) with 4.75 times its share of solved episodes, and over plain E2B 9 vs 1 (p = 0.021); D against C not met (9 vs 3 and 4 vs 10 on the two adapter pairs); loop gate failed (error endings 36.5 per 160 against the null adapter's 9 plus 6) |
| 208 (the self-corrector students live) | 4 of 9 held; neither rule met; 208b (control attempts after a driver upgrade): 2 of 3 held, 1 not scored | not met: the self-corrector adapters solved 9 of 320 and do not beat the null adapter (5 vs 2 by task); the cross-model adapters lead them 11 vs 3 (p = 0.057, above the bar) |
| 209 (the online teachers on contest coding, correctors as fitted for the agent stream) | 4 of 15 held, 3 failed, 8 not read for both seeds | no rule read for the cross-model corrector (one seed; its second was stopped): 63 of 188 held problems against 75 for E4B's distribution and 76 for its verified text (untrained 57); the two copying forms tie over two seeds (75, 69 and 76, 69; 25 vs 26) |
| 209a (the correctors fitted with the contest rows) | 1 of 13 held, 2 failed, 10 not read | no rule read (seed 1 stopped before its evaluation): the fit brought the teacher from 0.336 to 0.310 nats on held coding rows, but its student solved 61 held problems at seed 2 (copying 69) and wrote a placeholder answer on 72 and 67 of 487 problems; the fit's draws were 30% verified solutions |
| 209b-209d, 209f (stopped sessions, restarts, a probe and a filter by word lists) | 209d: 0 of 6 held, 6 not scored; 209f: 2 of 3 held, 1 not scored (read on 209g's fit) | not run or stopped; superseded by 209e and 209g |
| 209e (correctors fitted on verified solutions only; teaching only on answers that pass) | 6 of 11 held, 4 failed, 1 not scored (the teaching predictions are read on 209g's sessions, which reran them) | the data's smoke test passed (0 of 934 rows failed); rule 1 not met (21 placeholder answers at seed 1 of the rerun; fewer than 10 in each seed registered); rule 4 not met: the filter adds nothing measurable for E4B's distribution (24 vs 22 pooled, p = 0.88); the teaching sessions were stopped and rerun as 209g |
| 209g (a reviewer model reads every fitting solution; both 4× correctors fitted on the reviewed rows) | 9 of 10 held, 1 not scored | reviewer check met (12 of 12 planted rows flagged); held solved 79 and 67 (E4B's distribution, passing answers only), 73 and 63 (self-corrector), 75 (cross-model corrector, seed 1; seed 2 stopped) with 21 placeholder answers, so rule 1 (fewer than 10 in each seed) is not met; rules 2, 3 and 7 not read |
| 209h (why the cross-model corrector's student writes the placeholder; token targets) | 1 of 11 held, 1 failed, 9 not scored (the registered fit and its teaching were not run) | the reading did not hold: on the 21 placeholder problems E4B favours the placeholder more than untrained E2B on 5 (predicted at least 14); by its wording the placeholder that is left is E2B's own |
| 209i (rambling and overlong solutions rewritten by their author model; both correctors refitted with token targets) | 15 of 22 held, 6 failed, 1 not scored | "matches" met, "the best" not met: the cross-model corrector's students solved 76 and 75 held problems against 79 and 67 (pooled 25 vs 20, p = 0.55); placeholder rule met (7 and 7 of 487); cross-model against self-corrector 24 vs 21 pooled (p = 0.77; self-corrector 73 and 75): not met |
| 209k (does the cross-model corrector's student write more like E4B; stored answers) | 0 of 3 | not met: a style score tells E4B from E2B with accuracy 0.63; the cross-model student scores above the self-corrector's on 84 against 103 held problems (p = 0.19), the copying student above it on 121 against 67 (p = 0.0001) |
| 209l (a "placeholder next" direction erased from what the corrector reads) | 1 of 10 held, 3 failed, 6 not run | gate not passed, nothing fitted: the direction separates held problems with AUC 0.61 against the bar of 0.80; erased in E4B itself, 9 of 17 held placeholder problems still continue with it (a random direction: all 17) |
| 209m (what follows the placeholder) | 4 of 6 | met for E4B only: a second code block follows all 29 of its placeholder answers and 14 pass; untrained E2B 12 of 26 with 3 passing |
| 209n (the hybrid: the verified text as the target where no sample passes) | 4 of 7 held, 3 failed | rules 1 and 2 met in both seeds (0 and 5 placeholder answers, the plain corrector 7 and 7; 99 and 100 of 151 training problems fixed against 92 and 86); held problems solved 80 and 69 against 76 and 75 (pooled 20 vs 22, p = 0.88); "ahead of copying" not met (69 against 69 at seed 2); third seed 77: the highest three-seed total (226, against 220 for the verified text and 212 for E4B's distribution on passing answers), no pair apart (43 vs 37, p = 0.58; 46 vs 32, p = 0.14) |
| 209o (the teacher's own corrector, fitted on the rewritten solutions) | 3 of 3 | met: on the 17 held problems where E4B starts the placeholder, its opening is less likely under the corrected E4B on 16 (p = 0.0003) and is its first choice on 3 |
| 209p (the student's own placeholder answers made less likely) | 2 of 3 at seed 1 | rule met at seed 1: no placeholder answer (the plain corrector 7) at 73 held problems solved (76; 7 vs 10, p = 0.63); only 3 of the student's 1,494 samples were placeholders and were penalised, so one seed does not show the penalty did it |
| 209q (what the corrected teacher writes) | 4 of 5 | rules met: with its own corrector on, E4B solves 13 of 57 held problems it had failed and loses 2 of 20 (p = 0.007); 2 of its 17 placeholder answers remain; the same loop with the corrector off solves 1 of the 57 and 18 of the 20 (by problem, corrector on against off: 14 vs 2, p = 0.004) |
| 209r (the hybrid copies the reviewed solution where no sample passes) | 3 of 4 | rule met at seed 1: 71 of 487 answers unfinished (the hybrid 116, 113, 75), no placeholder answer, 75 held problems solved (the hybrid 80 at that seed; 11 vs 16, p = 0.44); 109 of 151 training problems fixed |
| 209r, seeds 2 and 3 (the same against the hybrid at its three seeds) | 10 of 12 held, 2 failed | rule 1 met: fewer answers unfinished than the hybrid in every seed pair (71, 72, 59 against 116, 113, 75; by answer 53 vs 155, p < 0.0001); held problems level (75, 70, 71 against 80, 69, 77; 37 vs 47, p = 0.33); the per-seed rule met at seed 2 and failed at seed 3 (3 placeholder answers against a limit of 2) |
| 203 (dense rank-64 LoRAs) | 0 of 4 held, 4 not scored | not run: its runner was restarted past its start cutoff and ended at once |

Not in the table, though they registered predictions in the public log, where their outcomes are recorded: corrector building and screens before the paper's runs (165 to 169, 171, 172, 174q, 175, 180); addenda registered and not run (173 and 174b, whose conditions did not occur; 204 and 205, replaced by 206; 206d, seeds 4 and 5; the 4090 run of 184); loop pilots on the 4090 outside this paper (177, 177b, 177c); and a separate study (191 and 195: per-user and lottery-ticket grafts).

## Appendix D Further results

**Correctors.** Sizes and fitting rows are in Table D1. For Qwen, LoRAs distilled from correctors fitted 2-8× longer solved no more contest problems (5 of 6 fewer, none significantly; one seed) but came closer to E4B on held agent replies (0.72 to 0.65 for the cross-model corrector's, 0.73 to 0.66 for the self-corrector's), with more drift on untaught prose (KL from the original 0.19 to 0.26 and 0.20 to 0.25). The online correctors (Section 4.1) applied the do-no-harm term wherever student and teacher picked the same token. Depths are nominal, rounded to the nearest layer: with E2B as the student, the corrector reads E4B's layers 20 and 27 of 42, or E2B's own 17 and 22 of 35, and writes from E2B's layer 12 (Qwen: 9 of 28; E4B: 14 of 42). Every corrector, cross-model or self, also reads the student's own state at the write layer.

Table D1. Corrector parameters (those the forward pass uses) and fitting rows by student. The online correctors' stored files also hold a linear map of 4.7M and 7.9M parameters that their forward pass does not use.

| Student | Self-corrector | Cross-model corrector (reads) | Fitting rows |
|---|---|---|---|
| Qwen3-1.7B, coding | 78M | 83M (E4B) | agent replies 50%, human prose 15%, Python code 15%, multiple-choice questions 10%, stories 10% |
| Gemma 4 E2B, coding | 65M | 75M (E4B) | as above |
| Gemma 4 E4B, coding (Section 4.6) | 121M | 110M (E2B) | as above |
| Gemma 4 E2B, online (Sections 4.2-4.4) | 65M | 75M (E4B) | agent replies 40%, E2B's own samples 20%, prose 15%, code 10%, questions 10%, stories 5% |

**Agent-turn LoRAs (Section 4.5).** Figure D1 shows the problems each form gained and lost. SFT with a KL pull toward the original lost the fewest solved problems (66) but gained the least (against SFT 90 vs 195, p = 5e-10). The cross-model form's live advantage over the self form was only 0.05-0.06 nats on agent replies. In a separate coding run the anchors did not lose to SFT (held solved: SFT 143, anchors 146-152).

![Gains and losses](figures/fig2_wide_coding.png)

Figure D1. LoRAs trained on E4B's agent turns, on LiveCodeBench (1,739 problem-runs): problems gained and lost against the untrained student.

**Gemma 4 E2B, every recipe (Section 4.6).** Anchors to fresh Gemma correctors moved E2B toward lower loss (prose Δ −0.58 to −0.60), raising held solves by at most 12 (p = 0.22) at 24-28 fewer fixes per seed than the shortest rows alone. Table D2 adds the shortest rows, the runbook rows, the less-training arms (addendum 188), λ = 3, the cross-model anchor and the fitting text with the KL pull. E2B's training loss on E4B's rows started low (0.28 nats over its first 50 steps), barely fell in the first epoch and reached 0.07 in the second; Qwen's ended at 0.24. Tests marked "on trained problems" count the trained problems solved by one recipe only: λ 0.3 against λ 1, 94 vs 48 (p = 0.0001). The KL pull at λ 0.3 left small rises in untaught loss (prose −0.01 to +0.02 per seed, stories +0.04 to +0.06), 92-99% below SFT's. Stopped there after one epoch, E2B's prose damage fell from +0.69 to +0.09, yet it moved twice as far as Qwen (KL 0.16 against 0.074), so part of the effect is the student. Stopping early kept no held gain and lost 122 solved problems (SFT 47). Both partly trained arms lost mostly trained problems (75 and 49, against 18), which SFT's second epoch turns into E4B's verified answers. The KL pull did not reduce the fitting text's drift (prose KL 0.81): it covers only the coding replies. The Gemma correctors left E2B's coding loss unchanged.

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

**Qwen drift (Section 4.6).** Table D3 gives the three-seed means and Figure D2 each seed. On the shortest rows, KL to the original moved Qwen 6-9× less than the other copying recipes on prose and stories (3× on code); the corrector anchor moved it as far as SFT, toward lower loss and softer predictions, from the corrector's prose fit. Neither anchor cost held-out solves (Table D4), though they fixed 5-6% fewer training problems.

Table D3. Qwen's drift on untaught text, three-seed means.

| Recipe | KL from the original (prose) | Change in loss (nats): prose / code / stories |
|---|---|---|
| Plain SFT | 0.074 | −0.06 / −0.08 / −0.10 |
| Shortest + KL to the original | **0.011** | +0.01 / −0.10 / −0.02 |
| Shortest + corrector anchor | 0.089 | −0.24 / −0.21 / −0.51 |

Table D4. Qwen with the two anchors, three seeds pooled (fresh-question p(true): mean probability of the true answer on held-out multiple-choice questions).

| Recipe | Held solved (of 564) | Lost (of 384) | Fresh-question p(true) (untrained 0.513) |
|---|---|---|---|
| Shortest rows | 160 | 48 | 0.527 |
| Shortest + KL to the original | 159 | **40** | 0.521 |
| Shortest + corrector anchor | 157 | 52 | 0.554 |
| Plain SFT | 149 | 58 | 0.557 |

![Prose drift](figures/fig4_untaught_prose.png)

Figure D2. Drift on untaught human prose, one point per seed.

**Runbook rows (Section 4.6; Table D5).** E4B turns each verified solution into a numbered plain-English plan, the student solves the problem with the plan in its prompt, and passing solutions are trained on without the plan. The 228 rows kept are the student's own text: similarity to E4B's solution 0.20, 0.32 nats under the untrained student (E4B's rows 1.05), 110 tokens (350). They solved slightly fewer new held-out problems (83 against 90-95, within noise), lost the fewest training-pool problems, barely moved the student (prose KL 0.022) and lowered its entropy the most (−0.23 nats on prose). The corrector anchor reversed the rise in untaught loss in every seed (prose −0.23 to −0.25; fresh-question p(true) +0.028) at the cost of 39 fewer fixed problems and 10 fewer held solves. With E2B the rows lost their advantage: its rewrites were no closer to it than E4B's text (0.38 vs 0.27 nats). A screen rejected plans with code or over 10% of the solution's code 4-grams. On trained problems solved by one recipe only: runbook rows against SFT 133 vs 84 (p = 0.001; on the fixed problems alone 105 vs 73, p = 0.02); with the self-anchor against without, 29 vs 71 (p = 3e-5); for E2B, runbook rows against SFT 61 vs 111 (p = 0.0002). The two anchors of Table D4 fixed 382 and 385 training problems against 406 for the shortest rows.

Table D5. Runbook rows, three seeds pooled; prose ΔCE per seed.

| Recipe | Fixed | New | Held solved | Lost (of 384) | Prose ΔCE (per seed) |
|---|---|---|---|---|---|
| Plain SFT | 389 | 90 | 149 | 58 | −0.05 to −0.07 |
| Runbook rows | **421** | 83 | 143 | 39 | **+0.10 to +0.16** |
| Runbook rows + self-anchor | 382 | 71 | 133 | 37 | −0.23 to −0.25 |

**On-policy distillation (Section 4.5).** The first self-corrector teacher fit E4B's solutions at 0.70 nats (the student alone 1.00), the refit one at 0.49. The refit teacher's student beat the weak teacher's in every seed but lost more previously solved train-pool problems than SFT (59 vs 42). As a rewriter, the refit corrector copied E4B's text (similarity 0.98-0.99). The teachers' live counts come from the rewrite sessions, whose batching differs from the evaluation sessions' (the untrained student solves 69 there, 75 here).

Table D6. Held solved, three seeds pooled.

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

Table D7. Online setups across three GPU types; seeds listed per row. Untaught drift is read at the last checkpoint (item 600).

| Setup | Seeds | Held CE (from 0.825) | Untaught drift | Gate net (fixed / broken) |
|---|---|---|---|---|
| A Teacher | 1 | 0.787 | 0.089 | +14 (25 / 11) |
| B Corrector | 1 | 0.754 | 0.055 | +12 (16 / 4) |
| C Teacher + keep + replay | 3 | 0.768-0.775 | 0.012-0.014 | +8 (A100 +16), +27, +19 |
| D Corrector + keep + replay | 3 | 0.749-0.752 | 0.014-0.042 | +19 (A100 +18), +17, +15 |
| Self-corrector + keep + replay | 1 | 0.755 | 0.025 | +8 (13 / 5) |
| Pull-matched blend + keep + replay | 1 | 0.781 | 0.014 | +10 (14 / 4) |
| Gated teacher + keep + replay | 3 | 0.764-0.766 | 0.011-0.015 | +16, +17, +12 (G4) |

**Online: one GPU type (Table 1).** Every run of Table 1 is a G4 session with the same inputs and stream order per seed, scored on the 4090's partition of the 220 decisions (on a G4, E2B alone differs from it on 3). Fixed / broken per run (seeds 1, 2, 3): C 21 / 8, 30 / 12, 29 / 10; S 13 / 3, 21 / 6, 20 / 3; D 23 / 4, 19 / 3, 22 / 2; DI 20 / 1, 22 / 2, 13 / 4; S4 22 / 5, 22 / 7, 20 / 5; D4 24 / 4, 25 / 4, 24 / 1. The first checkpoint at or below C's final held cross-entropy: item 100 at seed 1 (DI 150), item 50 at seeds 2 and 3 (D4 100 at seed 3). Untaught drift (KL from the original E2B, mean of the last six checkpoints, to four places; single checkpoints move between 0.010 and 0.047 within a run): C 0.0130, 0.0120, 0.0155; S 0.0236, 0.0211, 0.0230; D 0.0218, 0.0132, 0.0155; DI 0.0208, 0.0172, 0.0179; S4 0.0220, 0.0169, 0.0239; D4 0.0168, 0.0131, 0.0148.

**Online: cross-model against self.** The pre-registered rule (addendum 206) asked, at matched fitting, for a held cross-entropy at least 0.005 lower in every seed or 6 more net decisions on average with a lead in each seed. At three seeds neither is met: the held cross-entropy differences are 0.003, 0.002 and 0.003 (1× fit) and 0.003, 0.004 and 0.004 (4×); the mean net differences 4.3 and 5.7. After two seeds we added a paired comparison (amendment 206c; its six-pair rule and predictions were fixed before any seed-3 result was read). Every student is scored on the gate's same 240 held rows, from 118 episodes in 17 repositories; intervals are bootstraps over repositories. D minus S at seeds 1 to 3: −0.0027 [−0.0037, −0.0016], −0.0010 [−0.0018, −0.0001], −0.0020 [−0.0034, −0.0008]. D4 minus S4: −0.0011 [−0.0021, 0.0000], −0.0028 [−0.0040, −0.0015], −0.0008 [−0.0025, +0.0012]. The six pairs together: −0.0017 [−0.0025, −0.0009], the cross-model student closer in 14 of the 17 repositories; by fit, −0.0019 [−0.0026, −0.0012] and −0.0015 [−0.0026, −0.0004]. The intervals describe these students; for a claim about the teacher the pair is the unit, and 6 of 6 pairs has p = 0.031. For scale on the same rows, over three seeds: D minus C −0.0204 [−0.0240, −0.0162]; S minus C −0.0185 [−0.0222, −0.0145]; the 4× fit minus the 1× fit −0.0076 (D) and −0.0080 (S); DI minus D −0.0001 [−0.0009, +0.0006]. Per decision over the six pairs, more cross-model than self-corrector students matched E4B on 22 decisions and fewer on 6 (p = 0.004); pair by pair, D over S 11 vs 2 (p = 0.022), 3 vs 2 and 4 vs 1, D4 over S4 8 vs 5, 8 vs 2 and 10 vs 2 (p = 0.039). DI against D: 4 vs 4, 6 vs 2 and, at seed 3, 0 vs 11 (p = 0.001). The mean prequential gain is higher for the cross-model student in five of the six pairs, but not over the first 100 items (1×: 0.047 against 0.046, 0.050 against 0.049, 0.045 against 0.046; 4×: 0.044 against 0.049, 0.049 against 0.052, 0.041 against 0.047).

**Online: the correctors as teachers.** Run live on held rows, the cross-model corrected E2B is 0.029 nats closer to E4B than the self-corrected E2B (0.754 against 0.783), and 0.054 closer when both are fitted four times longer (2,400 steps, more code and story rows; 0.714 against 0.768). The part of the cross-model corrector's write that is specific to E4B is about half predictable from E2B's own states (R² 0.53; 0.37 at 4×). Of these gaps between the teachers, the students kept 0.0019 nats at 1× and 0.0015 at 4× (the paired comparison above). DI's corrector is fitted with a penalty on the unpredictable part: R² 0.73, writes a quarter the size, and the same lead over the self-corrector on held rows (0.027). On the blend's 24 calibration points, D's teacher put less of its pull where E4B and E2B disagree than the blend (49% vs 57%) and moved E2B's top token to E4B's less often (26% vs 40%).

**Online: copying.** Copying online, SFT on E4B's logged replies with keep + replay (seed 1, A100), fixed 19 decisions, broke 13 and ended further from E4B on held text than the original E2B (0.900 against 0.824); D beat it on 18 decisions against 6 (p = 0.023) and C on 15 against 5 (p = 0.041). Offline SFT on the same 600 items (two passes) netted +14 (24 fixed, 10 broken), but on the gate's replies both SFT students sat further from E4B than the original E2B (1.10 offline, 0.91 online, against 0.83), where C, the gated teacher and D brought it closer (0.78, 0.77, 0.76).

**Live (Section 4.4).** Each attempt runs the 80 tasks once in the competition's harness. The tasks are ones E4B failed in the run that chose them; in later attempts of its own it solved 5 to 10 of the first 40 and 5 and 9 of the other 40. The null adapter is D's adapter with its output matrices set to zero: untrained E2B through the adapter serving path. That path took about 1.9 times plain E2B's wall time (85-100 minutes per attempt against 43-55) but ended no more episodes in errors and solved no more, so the LoRAs' time-outs come from their own longer episodes (Table D8). C and D each have two adapters of seed 1, trained on different GPU types. By task on the solve rate: D's first adapter over the null adapter 12 vs 0 (p = 0.0005), over plain E2B 9 vs 1 (p = 0.021), over C's first 9 vs 3 (p = 0.15); D's second over the null adapter 9 vs 1 (p = 0.021), against C's second 4 vs 10 (p = 0.18); D4's over the null adapter 8 vs 1 (p = 0.039). Pooled, the corrector-taught adapters against C's: 8 vs 7. By teacher (not registered): the corrector-taught adapters over the null adapter 13 vs 0 (p = 0.0002) and over plain E2B 13 vs 2 (p = 0.007); C's two adapters 10 vs 1 (p = 0.012) and 10 vs 2 (p = 0.039), though C's first adapter alone does not (3 vs 1). The pre-registered rule (addendum 207) reads D over four attempts and asks for p < 0.05 against the null adapter and twice its share of solved episodes; D has 4.75 times (5.9% against 1.25%). D against C is not claimed. The loop gate (error endings per 160 at most the null adapter's plus 6) fails for every taught adapter but C's second (11): D's 36.5 and 16, D4's 17, C's first 25. The self-corrector's students (addendum 208, two attempts each): against the null adapter 2 vs 2 (S) and 4 vs 1 (S4); D over S 11 vs 2 (p = 0.022), D4 over S4 7 vs 1 (p = 0.070), pooled 11 vs 3 (p = 0.057). Neither registered reading is met: the self-corrector adapters do not beat the null adapter, and the cross-model lead misses p < 0.05. After a driver upgrade on the serving GPU, D's first adapter again solved 5 of 80 (before: 5, 6, 3, 5).

Table D8. The live test: 80 held-out tasks per attempt.

| Arm | Attempts | Solved | Tasks solved at least once | Error endings (time-outs) | Model calls per task |
|---|---|---|---|---|---|
| Plain E2B | 4 | 7 of 320 | 5 | 19 (1) | 14.0 |
| Null adapter | 2 | 2 of 160 | 2 | 9 (4) | 14.6 |
| C's LoRA, first adapter | 2 | 5 of 160 | 4 | 25 (20) | 22.2 |
| C's LoRA, second adapter | 2 | 17 of 160 | 12 | 11 (9) | 16.6 |
| D's LoRA, first adapter | 4 | 19 of 320 | 12 | 73 (48) | 21.1 |
| D's LoRA, second adapter | 2 | 11 of 160 | 10 | 16 (14) | 18.2 |
| D4's LoRA | 2 | 13 of 160 | 9 | 17 (11) | 15.9 |
| S's LoRA | 2 | 3 of 160 | 2 | 21 (17) | 18.7 |
| S4's LoRA | 2 | 6 of 160 | 5 | 21 (11) | 19.1 |

**Online coding (Section 4.5).** The stream is the 299 training problems, each met twice in a seeded order (598 items), with the rank-16 LoRA, keep + replay and schedule of the online runs (Sections 3 and 4.1). At each item the student samples two answers of up to 512 tokens and each is run on the problem's tests; it learns on those that pass and, when none passes, on E4B's verified solution of the problem, toward the same teacher (39-42% of the learned items for the corrector students, 46-48% before the rewriting, 38-39% for E4B's distribution). Copying E4B's verified text needs no samples. The correctors are fitted for 2,400 steps on 602 solutions of the training problems written by E4B or E2B themselves, each passing its tests, and on the general text of Table D1. A reviewer model (DeepSeek V4.1 Flash, two votes per reading) read every solution for placeholders and thinking aloud, for rambling comments and for needless length; 392 of the 602 are versions by their own author model, each tested and read again: 281 rewritten after being told the kind of fault and the line where it starts, never a way to solve the problem, and 111 continued from before the fault or sampled again. Where E4B's top token is not the solution's next token (6.1% of the coding positions) the fit's target is that token; elsewhere it is E4B's distribution. On held coding rows the cross-model corrected E2B is at 0.345 nats from E4B and the self-corrected one at 0.359 (E2B alone 0.381).

Table D9. Online coding: E2B's students by teacher, seeds 1 and 2, and seed 3 for four rows (487 problems: 299 training, 188 held; answers cut at the first closing fence).

| Teacher | Held solved (of 188) | Fixed (of 151) | Lost | Unfinished (of 487) | Placeholder answers |
|---|---|---|---|---|---|
| None (untrained E2B) | 57 | | | 207 | 26 |
| E4B's distribution | 75, 69 | 96, 80 | 14, 18 | 132, 159 | 1, 0 |
| E4B's distribution, passing answers only | 79, 67, 66 | 108, 81, 80 | 8, 27, 33 | 115, 157, 104 | 0, 0, 51 |
| E4B's verified text | 76, 69, 75 | 106, 99, 104 | 12, 22, 14 | 106, 109, 101 | 0, 0, 0 |
| Self-corrector | 73, 75 | 88, 82 | 20, 20 | 101, 79 | 6, 9 |
| Cross-model corrector | 76, 75 | 92, 86 | 26, 13 | 96, 100 | 7, 7 |
| Hybrid: cross-model corrector on passing answers, verified text where none passes | 80, 69, 77 | 99, 100, 110 | 12, 14, 14 | 116, 113, 75 | 0, 5, 0 |
| Hybrid copying the reviewed solution where none passes | 75, 70, 71 | 109, 101, 106 | 15, 21, 24 | 71, 72, 59 | 0, 1, 3 |
| Self-corrector, fitted before the rewriting | 73, 63 | 82, 81 | 20, 24 | 141, 131 | 0, 10 |
| Cross-model corrector, fitted before the rewriting (seed 1) | 75 | 78 | 21 | 138 | 21 |

By problem, pooled over seeds 1 and 2, the cross-model corrector's students against those of E4B's distribution on passing answers: 25 vs 20 held problems (p = 0.55); against E4B's verified text 25 vs 19 (p = 0.45); against the self-corrector's 24 vs 21 (p = 0.77); against untrained E2B 46 vs 9. On the 299 training problems the verified text solves more (by problem 84 vs 43, p = 0.0003; on the 151 that untrained E2B fails, 58 vs 31, p = 0.006). Seeds of one recipe differ by up to 13 held problems (79 and 66; 12 within seeds 1 and 2). Read by problem (not registered), the cross-model corrector's two students agree most: 68 held problems are solved by both and 15 by one only, against 60 and 28 for the self-corrector's (problems where only one pair differs: 8 vs 21, p = 0.024) and 59 to 63 and 18 to 28 for the copying students. The last two rows are the same correctors fitted before the rewriting (608 solutions: 518 as first written, 90 continued by their author from before the first flagged line), with E4B's distribution as the only target: the rewriting and the token targets shortened the students' answers (held median 1,819 and 1,793 characters against 2,091), cut the unfinished ones from 138 to 96 and the placeholder answers from 21 to 7 at seed 1, and raised the held answers that finish and are wrong from 18 to 35. The self-corrector's student changed the same way in its unfinished and its wrong answers (its placeholder answers went from 0 and 10 to 6 and 9), so this comes from the fit's text and targets, not from E4B's states. Untaught text is unharmed for every teacher: KL from the original E2B 0.011-0.019, prose cross-entropy within 0.04 of the untrained 4.473.

**The hybrid.** Copying shows a whole correct answer, which is what teaches a problem the student cannot yet do; the corrected student teaches in the student's own text, which refines an answer it can already write. The hybrid uses each where the tests say it applies: on the items where a sample of the student passes (65% and 63% at seeds 1 and 2) the teacher is the corrected student as before, and on the others the target is E4B's verified solution itself (cross-entropy on its tokens), where the plain corrector run pulls toward the corrected student's distribution on that text and so keeps E2B's own habits there. On the training problems it closes the plain corrector's gap to copying: it fixed 99 and 100 of the 151 (the plain corrector 92 and 86); by problem over all 299, pooled over the two seeds, 87 vs 53 against the plain corrector (p = 0.005) and 53 vs 60 against the verified text (p = 0.57). On held problems it is level with the others: 80 and 69 solved; pooled by problem 20 vs 22 against the plain corrector (p = 0.88), 26 vs 23 against E4B's distribution on passing answers (p = 0.78) and 30 vs 26 against the verified text (p = 0.69). The 80 of seed 1, the most of any run, was not repeated, and the registered rule for "ahead of copying" (more held problems than both copying forms in every seed) failed at seed 2 (69 against 67 and 69). It wrote 0 and 5 placeholder answers (the plain corrector 7 and 7), and its answers are as long and as often unfinished as the copying students' (116 and 113 unfinished; held median 2,028 and 2,173 characters): the text it copies is E4B's solution as E4B wrote it, not a rewritten one. Third seeds were run for the hybrid and the two copying forms it was registered against. The hybrid solved 77 held problems (110 of the 151 fixed, no placeholder answer, 75 answers unfinished), the verified text 75, and E4B's distribution on passing answers 66 with a placeholder answer on 51 of the 487 problems, 15 of them problems untrained E2B solves: copying E4B's distribution can bring the placeholder back in one seed where two had none. Over the three seeds the hybrid's total is the highest (226 held problems, against 220 and 212) and no pair separates: by problem 43 vs 37 against the verified text (p = 0.58) and 46 vs 32 against E4B's distribution on passing answers (p = 0.14); over the 299 training problems 86 vs 86 and 136 vs 75 (p < 0.0001). A variant copies, where no sample passes, the shortest solution the reviewer model accepted for the problem (285 of the 299 problems have one; median 713 characters against 1,164 for E4B's solution as written). Over three seeds its students left 71, 72 and 59 answers unfinished, the fewest in Table D9 at each seed and fewer than the hybrid's at each (116, 113, 75; by answer, pooled, 53 unfinished only for the variant against 155 only for the hybrid, p < 0.0001): the rule registered before its seeds 2 and 3 (fewer in every seed pair and pooled p < 0.05) was met. Their held answers are shorter (median 1,615, 1,557 and 1,398 characters against 2,028, 2,173 and 1,936). They solved no more: 75, 70 and 71 held problems against 80, 69 and 77 (by problem 37 vs 47, p = 0.33), and 109, 101 and 106 of the 151 fixed against 99, 100 and 110 (over the 299 training problems 97 vs 104, p = 0.67). Of the 155 answers unfinished for the hybrid and not for the variant, 56 pass the tests, 94 are wrong and 5 do not parse or are placeholders; more of the variant's held answers finish and are wrong (54, 58 and 65 against 25, 32 and 52). It wrote 0, 1 and 3 placeholder answers (the hybrid 0, 5 and 0), so the per-seed rule (at most 105 unfinished, at least 70 held, at most 2 placeholder answers) was met at seeds 1 and 2 and failed at seed 3.

A placeholder answer is the starter signature, a comment such as "# ... implementation" and the closing fence. Untrained E2B writes it on 26 of the 487 problems and E4B on 29. For E4B it is a skeleton before the answer: made to continue past the fence, it opens a second code block after all 29, and 14 of those pass the tests (4 of its 17 on held problems, so it would solve 78, not 74). E2B mostly continues with prose: a second block follows 12 of its 26, and 3 pass.

**The teacher's own corrector.** A corrector of the same form fitted for E4B itself on the 602 solutions (its source is E4B's own states) changes what E4B writes. On the 17 held problems where E4B, given the prompt and the signature, starts the placeholder, its opening is less likely under the corrected E4B on 16 (p = 0.0003) and is its first choice on 3. We then let E4B answer 77 held problems again: the 17 where its stored greedy answer is a placeholder, 40 of the 86 where it is unfinished and 20 of the 74 it solves. With this corrector on it solves 13 of the 57 failed ones (2 of the 17 and 11 of the 40) and 18 of the 20; the same generation loop with the corrector off solves 1 of the 57 and 18 of the 20 (by problem 14 vs 2, p = 0.004). Most of the gain is answers that now finish within the 768 new tokens (16 of the 40 unfinished ones finish, against 2 with the corrector off). We did not test the corrected E4B as a teacher.
