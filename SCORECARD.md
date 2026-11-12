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
| 206 (the online teachers on one GPU type, three seeds) | 6 of 9 held, 1 in part, 2 failed | not met: the cross-model corrector's students lead the self-corrector's by less than the bars (held CE 0.002-0.004 against 0.005; net 4.3 and 5.7 on average against 6) |
| 206c (the same runs paired by row and by decision; registered after two seeds, before seed 3) | 5 of 6 | met: cross-model closer in 6 of 6 pairs (p = 0.031), 0.0017 nats [0.0009, 0.0025]; 22 vs 6 decisions (p = 0.004) |
| 207 (the live test with a null adapter, two adapters per teacher) | rule 1 of 3 | met on solves: D over the null adapter by task 12 vs 0 (p = 0.0005) with 4.75 times its share of solved episodes, and over plain E2B 9 vs 1 (p = 0.021); D against C not met (9 vs 3 and 4 vs 10 on the two adapter pairs); loop gate failed (error endings 36.5 per 160 against the null adapter's 9 plus 6) |
| 208 (the self-corrector students live) | neither rule; predictions 1 of 5, 1 in part | not met: the self-corrector adapters solved 9 of 320 and do not beat the null adapter (5 vs 2 by task); the cross-model adapters lead them 11 vs 3 (p = 0.057, above the bar) |
| 209 (the online teachers on contest coding, correctors as fitted for the agent stream) | 4 of 15 held, 3 failed, 8 not read for both seeds | no rule read for the cross-model corrector (one seed; its second was stopped): 63 of 188 held problems against 75 for E4B's distribution and 76 for its verified text (untrained 57); the two copying forms tie over two seeds (75, 69 and 76, 69; 25 vs 26) |
| 209a (the correctors fitted with the contest rows) | 1 of 13 held, 2 failed, 10 not read | no rule read (seed 1 stopped before its evaluation): the fit brought the teacher from 0.336 to 0.310 nats on held coding rows, but its student solved 61 held problems at seed 2 (copying 69) and wrote a placeholder answer on 72 and 67 of 487 problems; the fit's draws were 30% verified solutions |
| 209b-209d, 209f (stopped sessions, restarts, a probe and a filter by word lists) | no prediction read | not run or stopped; superseded by 209e and 209g |
| 209e (correctors fitted on verified solutions only; teaching only on answers that pass) | 4 of 11 held, 2 failed, 5 not read | the data's smoke test passed (0 of 934 rows failed); rule 4 not met: the filter adds nothing measurable for E4B's distribution (24 vs 22 pooled, p = 0.88); the teaching sessions were stopped and rerun as 209g |
| 209g (a reviewer model reads every fitting solution; both 4× correctors fitted on the reviewed rows) | 8 of 11 held, 3 not scored | reviewer check met (12 of 12 planted rows flagged); held solved 79 and 67 (E4B's distribution, passing answers only), 73 and 63 (self-corrector), 75 (cross-model corrector, seed 1; seed 2 stopped) with 21 placeholder answers; rules 1-3 not read |
| 209h (why the cross-model corrector's student writes the placeholder; token targets) | 1 of 2 on the probe; the registered fit not run | the reading did not hold: on the 21 placeholder problems E4B favours the placeholder more than untrained E2B on 5 (predicted at least 14); by its wording the placeholder that is left is E2B's own |
| 209i (rambling and overlong solutions rewritten by their author model; both correctors refitted with token targets) | 15 of 22 held, 6 failed, 1 not scored | "matches" met, "the best" not met: the cross-model corrector's students solved 76 and 75 held problems against 79 and 67 (pooled 25 vs 20, p = 0.55); placeholder rule met (7 and 7 of 487); cross-model against self-corrector 24 vs 21 pooled (p = 0.77; self-corrector 73 and 75): not met |
| 209k (does the cross-model corrector's student write more like E4B; stored answers) | 0 of 3 | not met: a style score tells E4B from E2B with accuracy 0.63; the cross-model student scores above the self-corrector's on 84 against 103 held problems (p = 0.19), the copying student above it on 121 against 67 (p = 0.0001) |
| 209l (a "placeholder next" direction erased from what the corrector reads) | 1 of 10 held, 3 failed, 6 not run | gate not passed, nothing fitted: the direction separates held problems with AUC 0.61 against the bar of 0.80; erased in E4B itself, 9 of 17 held placeholder problems still continue with it (a random direction: all 17) |
| 209m (what follows the placeholder) | 4 of 6 | met for E4B only: a second code block follows all 29 of its placeholder answers and 14 pass; untrained E2B 12 of 26 with 3 passing |
| 209n (the hybrid: the verified text as the target where no sample passes) | 4 of 7 held, 3 failed | rules 1 and 2 met in both seeds (0 and 5 placeholder answers, the plain corrector 7 and 7; 99 and 100 of 151 training problems fixed against 92 and 86); held problems solved 80 and 69 against 76 and 75 (pooled 20 vs 22, p = 0.88); "ahead of copying" not met (69 against 69 at seed 2); third seed 77: the highest three-seed total (226, against 220 for the verified text and 212 for E4B's distribution on passing answers), no pair apart (43 vs 37, p = 0.58; 46 vs 32, p = 0.14) |
| 209o (the teacher's own corrector, fitted on the rewritten solutions) | 3 of 3 | met: on the 17 held problems where E4B starts the placeholder, its opening is less likely under the corrected E4B on 16 (p = 0.0003) and is its first choice on 3 |
| 209p (the student's own placeholder answers made less likely) | 2 of 3 at seed 1 | rule met at seed 1: no placeholder answer (the plain corrector 7) at 73 held problems solved (76; 7 vs 10, p = 0.63); only 3 of the student's 1,494 samples were placeholders and were penalised, so one seed does not show the penalty did it |
| 209q (what the corrected teacher writes) | 4 of 5 | rules met: with its own corrector on, E4B solves 13 of 57 held problems it had failed and loses 2 of 20 (p = 0.007); 2 of its 17 placeholder answers remain; the same loop with the corrector off solves 1 of the 57 and 18 of the 20 (by problem, corrector on against off: 14 vs 2, p = 0.004) |
| 209r (the hybrid copies the reviewed solution where no sample passes) | 3 of 4 | rule met at seed 1: 71 of 487 answers unfinished (the hybrid 116, 113, 75), no placeholder answer, 75 held problems solved (the hybrid 80 at that seed; 11 vs 16, p = 0.44); 109 of 151 training problems fixed |
| 203 (dense rank-64 LoRAs) | — | queued |
