# Pre-registration scorecard

Which pre-registered predictions held, per addendum, for the paper "Copy, Correct, or Rewrite?" (paper.md). The predictions, designs and results are in PREREGISTRATION.md. Rows marked running were updated as results landed. The paper's Appendix C carries the rows for this paper (all but 191 and 195).

Predictions that held, per pre-registered addendum.

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