# Manifest

623 files, 49.3 MB in total (MANIFEST.md not counted). Sizes are on-disk sizes; `.gz` files are gzip-compressed JSON Lines.
Run labels give the addendum and seed (see README.md, "Run folders").

| File | Size | Purpose |
|---|---|---|
| `LICENSE` | 11.1 KB | Apache License 2.0 |
| `MANIFEST.md` | (this file) | This file: every file with its size and purpose |
| `PREREGISTRATION.md` | 534.3 KB | Pre-registration log, addenda 165-209 and their amendments, verbatim (one phrase and one bullet redacted) |
| `README.md` | 17.7 KB | Overview: the study, findings, layout, how the runs worked, what is excluded, rebuild steps, licence |
| `SCORECARD.md` | 9.2 KB | Which pre-registered predictions held, per addendum (the table of the paper's Appendix C) |
| `code/colab/xq_bridge.sh` | 3.0 KB | Colab session: fit the bridge and control correctors and run the suite (166) |
| `code/colab/xq_curve.sh` | 5.2 KB | Colab session: corrector training-length curve (176) |
| `code/colab/xq_dosebase.sh` | 3.2 KB | Colab session: dose-matched SFT baselines and their evaluation (169d) |
| `code/colab/xq_erase.sh` | 5.9 KB | Colab job: the placeholder direction (209l) |
| `code/colab/xq_gbase.sh` | 2.6 KB | Colab session: graft and LoRA forms scored together on identical rows (169) |
| `code/colab/xq_lcb.sh` | 13.4 KB | Colab job: online coding, teachers as fitted for the agent stream (209, 209a) |
| `code/colab/xq_lcbv.sh` | 17.0 KB | Colab job: online coding with verified fit rows and teaching only on passing answers (209e-209r) |
| `code/colab/xq_learn.sh` | 23.5 KB | Colab session for the coding recipes: build, arms, short rows, refit corrector, runbook rows, E2B student, two-way, multi-task (174-193) |
| `code/colab/xq_online.sh` | 7.3 KB | Colab session: one online run (600 items) and its 240-point gate for one teacher and seed, with pinned inputs (189c, 197, 201, 206) |
| `code/colab/xq_rsi.sh` | 7.7 KB | Colab session: the attempt-verify-update loop pilot (177) |
| `code/colab/xq_selftest.sh` | 3.9 KB | Colab session: SFT + KL to the original student on agent rows, then LiveCodeBench (174q) |
| `code/colab/xq_stubpast.sh` | 3.8 KB | Colab job: continuing past the placeholder (209m) |
| `code/colab/xq_tfix.sh` | 8.9 KB | Colab job: the teacher's own corrector (209o) |
| `code/colab/xq_tfixgen.sh` | 5.1 KB | Colab job: the corrected teacher's answers and the control (209q) |
| `code/colab/xq_wide.sh` | 4.6 KB | Colab session: agent-turn LoRAs on all easy and medium LiveCodeBench problems (178) |
| `code/colab/xq_x1.sh` | 2.3 KB | Colab session: retention accounting on stored correctors and LoRAs (169, X1) |
| `code/dreamer/dense203c_4090.sh` | 5.2 KB | 4090 runner: dense rank-64 LoRAs toward the 4x corrector, E4B and the 4x self-corrector (203) |
| `code/dreamer/imit200c_4090.sh` | 4.2 KB | 4090 runner: the imitable corrector fits (200c) |
| `code/dreamer/judge_rows.py` | 17.6 KB | Reviewer-model votes on fit rows through OpenRouter (the key is read from the environment, never stored) |
| `code/dreamer/lcb_best16.py` | 7.1 KB | Best of 16: groups and picks for flagged fit rows (209i) |
| `code/dreamer/lcb_bestall.py` | 12.2 KB | Per-problem pick among verified solutions by a judge model (after the paper's runs) |
| `code/dreamer/lcb_fit_rows.py` | 25.6 KB | Fit rows of the coding correctors: pool, rules of form, final file (209g-209i) |
| `code/dreamer/lcb_regen_judge.sh` | 4.8 KB | Judging a resampling batch of fit rows (209i) |
| `code/dreamer/lcb_style.py` | 4.0 KB | Style score: does a student write like E4B (209k) |
| `code/dreamer/live_export.py` | 2.0 KB | Exports the live test's per-task outcomes to data/live/live80_attempts.json |
| `code/dreamer/live_loops.py` | 1.5 KB | Live test loop gate: error endings and repeated submits per attempt |
| `code/dreamer/online_4090.sh` | 3.0 KB | 4090 runner: the online test, three setups and their gates (170) |
| `code/dreamer/online_bkr_4090.sh` | 2.5 KB | 4090 runner: online test arm D, corrector teacher + keep + replay (183) |
| `code/dreamer/online_rep2_4090.sh` | 2.6 KB | 4090 runner: online C seeds 2-3 (189b) |
| `code/dreamer/online_rep3_4090.sh` | 2.9 KB | 4090 runner: online against offline SFT teachers on agent decisions (192, running at the time of release) |
| `code/dreamer/online_rep_4090.sh` | 3.4 KB | 4090 runner: online D seeds 2-3, self-corrector and blend teachers, re-gates (189) |
| `code/dreamer/real200_4090.sh` | 1.5 KB | 4090 runner: realizability of the corrector's write (200) |
| `code/dreamer/resdec_4090.sh` | 2.6 KB | 4090 runner: resdec control and replication seeds (180) |
| `code/dreamer/rsi4090_prep.sh` | 2.3 KB | 4090 runner: inputs for the loop pilot (177) |
| `code/dreamer/rsi4090_run.sh` | 3.6 KB | 4090 runner: the loop pilot arms and summary (177) |
| `code/dreamer/s4_plain.sh` | 4.1 KB | Live test: serves QAT E2B (optionally with a LoRA) on the 4090 and runs one attempt of the task list on the rollout host |
| `code/dreamer/scale199_4090.sh` | 5.9 KB | 4090 runner: the correctors fitted four times longer and their held rung (199) |
| `code/dreamer/x184_4090.sh` | 5.4 KB | 4090 runner: keep rows, selective corrector and arms (184) |
| `code/dreamer/x184_prep_4090.sh` | 3.9 KB | 4090 runner: inputs for addendum 184 |
| `code/dreamer/x196_live.sh` | 3.5 KB | Live test runner: D's LoRA, C's LoRA and plain E2B on 80 held-out tasks (196) |
| `code/dreamer/x202_live.sh` | 5.4 KB | Live test runner: the 4x D LoRA (202) |
| `code/dreamer/x202_pertask.py` | 1.3 KB | Live test analysis by task (202) |
| `code/dreamer/x207_null.py` | 1.5 KB | Builds the null adapter: D's adapter with every lora_B set to zero (207) |
| `code/dreamer/x207_pertask.py` | 2.5 KB | Live test analysis by task (207) |
| `code/dreamer/x207b_live.sh` | 8.0 KB | Live test runner: the null adapter, more D and plain attempts, the A100 adapter pair (207) |
| `code/dreamer/x208_live.sh` | 6.7 KB | Live test runner: the self-corrector students' LoRAs (208) |
| `code/dreamer/x208_pertask.py` | 2.6 KB | Live test analysis by task (208) |
| `code/dreamer/xq_onl11_launch.sh` | 6.5 KB | Local launcher: the online teachers on one GPU type, seeds 1-3 (206) |
| `code/dreamer/xq_onl12_launch.sh` | 6.4 KB | Local launcher: seeds 4-5 of the cross-against-self pairs (206d) |
| `code/dreamer/xq_side14_launch.sh` | 2.8 KB | Local launcher: queues Colab sessions for 181 and 178 |
| `code/dreamer/xq_side15_launch.sh` | 3.8 KB | Local launcher: queues Colab sessions for 182 and 178 seed 1, with a balance guard |
| `code/dreamer/xq_side16_launch.sh` | 2.4 KB | Local launcher: queues Colab sessions for 184c (rb_c) |
| `code/dreamer/xq_side17_launch.sh` | 2.5 KB | Local launcher: queues Colab sessions for 185 (sft_self_s) |
| `code/dreamer/xq_side18_launch.sh` | 2.3 KB | Local launcher: queues Colab sessions for 186 (Gemma 4 E2B student) |
| `code/dreamer/xq_side19_launch.sh` | 4.6 KB | Local launcher: queues Colab sessions for 187 (two-way E4B and E2B, running at the time of release) |
| `code/dreamer/xq_side20_launch.sh` | 3.0 KB | Local launcher: queues Colab sessions for 188 (matched-loss E2B, running at the time of release) |
| `code/dreamer/xq_side21_launch.sh` | 2.6 KB | Local launcher: queues Colab sessions for 190 (multi-task control) |
| `code/dreamer/xq_side22_launch.sh` | 3.0 KB | Local launcher: queues Colab sessions for 193 (KL strength sweep) and 194a (re-grade) |
| `code/transfer/build_mix.py` | 4.8 KB | Builds the mixed rows (prose, stories, code, questions) the corrector was fitted on (165) |
| `code/transfer/dequant.py` | 3.0 KB | Dequantises the 4-bit QAT Gemma checkpoint to bf16 |
| `code/transfer/graft_agg.py` | 1.8 KB | Per-seed summary of corrector forms against dose-matched SFT (169d) |
| `code/transfer/lcb_fix.py` | 6.6 KB | Continuations, resamples and rewrites of flagged fit rows by their author model (209g-209i) |
| `code/transfer/lcb_own.py` | 11.5 KB | The models' own verified solutions as corrector fit rows, with the smoke test of the data (209e) |
| `code/transfer/lcb_rows.py` | 4.6 KB | Contest-coding stream rows for the online runs (209) |
| `code/transfer/mix_merge.py` | 1.2 KB | Appends the mixed rows to the agent rows in one data folder (165) |
| `code/transfer/online.py` | 157.8 KB | Online test E4B to E2B (170, 183, 189: seeds, self-corrector and blend teachers) and earlier same-tokenizer experiments |
| `code/transfer/rb_pair.py` | 3.2 KB | Pairs runbook arms with stored arms across sessions (182, 184c) |
| `code/transfer/realize.py` | 9.1 KB | How much of the cross-model corrector's write can be predicted from the student's own states (200) |
| `code/transfer/rsi.py` | 65.7 KB | Attempt-verify-update loop pilot (177) |
| `code/transfer/stub_dir.py` | 12.2 KB | The placeholder direction: probe and erasure (209l, 209o) |
| `code/transfer/stub_past.py` | 4.8 KB | What follows a placeholder answer (209m) |
| `code/transfer/tfix_gen.py` | 6.2 KB | The teacher's answers with its own corrector on or off (209q) |
| `code/transfer/translator.py` | 33.6 KB | Translator and corrector modules used by online.py |
| `code/transfer/xbridge.py` | 67.1 KB | The state corrector across tokenizers (E4B to Qwen3-1.7B): alignment, fitting, suite, graft and LoRA forms |
| `code/transfer/xcode.py` | 22.1 KB | LiveCodeBench loading, generation, the sandboxed hidden-test scorer, the wide coding test |
| `code/transfer/xlearn.py` | 95.0 KB | The coding recipes (SFT, anchors, on-policy, rewrites, runbook rows) and their evaluation |
| `data/drift.json` | 460.3 KB | Drift from the original student on untaught prose, code and stories, per arm, seed and row (KL, CE change, entropy) |
| `data/drift_ss.json` | 46.4 KB | Drift from the original student for arm sft_self_s (185) |
| `data/generations/xqcC1/xl_gens_onl_c.jsonl.gz` | 208.3 KB | Greedy answers per problem with pass flag (209, E4B's distribution, seed 1): student of E4B's distribution |
| `data/generations/xqcC1/xl_gens_qwen.jsonl.gz` | 211.2 KB | Greedy answers per problem with pass flag (209, E4B's distribution, seed 1): the untrained student |
| `data/generations/xqcC2/xl_gens_onl_c.jsonl.gz` | 214.6 KB | Greedy answers per problem with pass flag (209, E4B's distribution, seed 2): student of E4B's distribution |
| `data/generations/xqcC2/xl_gens_qwen.jsonl.gz` | 211.2 KB | Greedy answers per problem with pass flag (209, E4B's distribution, seed 2): the untrained student |
| `data/generations/xqcD1/xl_gens_onl_d4.jsonl.gz` | 212.3 KB | Greedy answers per problem with pass flag (209, cross-model corrector as fitted for the agent stream, seed 1): student of the cross-model corrector (agent-stream fit) |
| `data/generations/xqcD1/xl_gens_qwen.jsonl.gz` | 211.2 KB | Greedy answers per problem with pass flag (209, cross-model corrector as fitted for the agent stream, seed 1): the untrained student |
| `data/generations/xqcDf2/xl_gens_onl_d4f.jsonl.gz` | 177.7 KB | Greedy answers per problem with pass flag (209a, cross-model corrector fitted with the contest rows, seed 2): student of the cross-model corrector fitted with the contest rows |
| `data/generations/xqcDf2/xl_gens_qwen.jsonl.gz` | 211.2 KB | Greedy answers per problem with pass flag (209a, cross-model corrector fitted with the contest rows, seed 2): the untrained student |
| `data/generations/xqcF1/xl_gens_onl_sft.jsonl.gz` | 191.7 KB | Greedy answers per problem with pass flag (209, E4B's verified text, seed 1): student of E4B's verified text (online SFT) |
| `data/generations/xqcF1/xl_gens_qwen.jsonl.gz` | 211.2 KB | Greedy answers per problem with pass flag (209, E4B's verified text, seed 1): the untrained student |
| `data/generations/xqcF1/xl_gens_sft.jsonl.gz` | 206.7 KB | Greedy answers per problem with pass flag (209, E4B's verified text, seed 1): SFT on teacher text |
| `data/generations/xqcF2/xl_gens_onl_sft.jsonl.gz` | 189.7 KB | Greedy answers per problem with pass flag (209, E4B's verified text, seed 2): student of E4B's verified text (online SFT) |
| `data/generations/xqcF2/xl_gens_qwen.jsonl.gz` | 211.2 KB | Greedy answers per problem with pass flag (209, E4B's verified text, seed 2): the untrained student |
| `data/generations/xqcF2/xl_gens_sft.jsonl.gz` | 204.1 KB | Greedy answers per problem with pass flag (209, E4B's verified text, seed 2): SFT on teacher text |
| `data/generations/xqlCF1/xl_gens_mix_c.jsonl.gz` | 108.3 KB | Greedy answers per problem with pass flag (181, seed 1): SFT + on-policy, weak teacher |
| `data/generations/xqlCF1/xl_gens_mix_cf.jsonl.gz` | 214.1 KB | Greedy answers per problem with pass flag (181, seed 1): SFT + on-policy, refit teacher |
| `data/generations/xqlCF1/xl_gens_opd_c.jsonl.gz` | 81.2 KB | Greedy answers per problem with pass flag (181, seed 1): on-policy, weak teacher |
| `data/generations/xqlCF1/xl_gens_opd_cf.jsonl.gz` | 219.0 KB | Greedy answers per problem with pass flag (181, seed 1): on-policy, refit teacher |
| `data/generations/xqlCF1/xl_gens_qwen.jsonl.gz` | 61.9 KB | Greedy answers per problem with pass flag (181, seed 1): the untrained student |
| `data/generations/xqlCF1/xl_gens_rw_cf.jsonl.gz` | 205.6 KB | Greedy answers per problem with pass flag (181, seed 1): rewrites via corrector |
| `data/generations/xqlCF1/xl_gens_sft.jsonl.gz` | 214.3 KB | Greedy answers per problem with pass flag (181, seed 1): SFT on teacher text |
| `data/generations/xqlCF2/xl_gens_mix_c.jsonl.gz` | 129.5 KB | Greedy answers per problem with pass flag (181, seed 2): SFT + on-policy, weak teacher |
| `data/generations/xqlCF2/xl_gens_mix_cf.jsonl.gz` | 212.4 KB | Greedy answers per problem with pass flag (181, seed 2): SFT + on-policy, refit teacher |
| `data/generations/xqlCF2/xl_gens_opd_c.jsonl.gz` | 98.2 KB | Greedy answers per problem with pass flag (181, seed 2): on-policy, weak teacher |
| `data/generations/xqlCF2/xl_gens_opd_cf.jsonl.gz` | 229.4 KB | Greedy answers per problem with pass flag (181, seed 2): on-policy, refit teacher |
| `data/generations/xqlCF2/xl_gens_qwen.jsonl.gz` | 61.9 KB | Greedy answers per problem with pass flag (181, seed 2): the untrained student |
| `data/generations/xqlCF2/xl_gens_rw_cf.jsonl.gz` | 211.9 KB | Greedy answers per problem with pass flag (181, seed 2): rewrites via corrector |
| `data/generations/xqlCF2/xl_gens_sft.jsonl.gz` | 201.3 KB | Greedy answers per problem with pass flag (181, seed 2): SFT on teacher text |
| `data/generations/xqlCF3/xl_gens_mix_c.jsonl.gz` | 84.8 KB | Greedy answers per problem with pass flag (181, seed 3): SFT + on-policy, weak teacher |
| `data/generations/xqlCF3/xl_gens_mix_cf.jsonl.gz` | 213.6 KB | Greedy answers per problem with pass flag (181, seed 3): SFT + on-policy, refit teacher |
| `data/generations/xqlCF3/xl_gens_opd_c.jsonl.gz` | 89.3 KB | Greedy answers per problem with pass flag (181, seed 3): on-policy, weak teacher |
| `data/generations/xqlCF3/xl_gens_opd_cf.jsonl.gz` | 219.3 KB | Greedy answers per problem with pass flag (181, seed 3): on-policy, refit teacher |
| `data/generations/xqlCF3/xl_gens_qwen.jsonl.gz` | 61.9 KB | Greedy answers per problem with pass flag (181, seed 3): the untrained student |
| `data/generations/xqlCF3/xl_gens_rw_cf.jsonl.gz` | 209.6 KB | Greedy answers per problem with pass flag (181, seed 3): rewrites via corrector |
| `data/generations/xqlCF3/xl_gens_sft.jsonl.gz` | 210.7 KB | Greedy answers per problem with pass flag (181, seed 3): SFT on teacher text |
| `data/generations/xqlE2B1/xl_gens_qwen.jsonl.gz` | 211.2 KB | Greedy answers per problem with pass flag (186, Gemma 4 E2B student, seed 1): the untrained student |
| `data/generations/xqlE2B1/xl_gens_rb.jsonl.gz` | 215.5 KB | Greedy answers per problem with pass flag (186, Gemma 4 E2B student, seed 1): runbook rows |
| `data/generations/xqlE2B1/xl_gens_sft.jsonl.gz` | 206.7 KB | Greedy answers per problem with pass flag (186, Gemma 4 E2B student, seed 1): SFT on teacher text |
| `data/generations/xqlE2B1/xl_gens_sft_s.jsonl.gz` | 180.9 KB | Greedy answers per problem with pass flag (186, Gemma 4 E2B student, seed 1): SFT, shortest correct |
| `data/generations/xqlE2B1/xl_rb_gens.jsonl.gz` | 347.9 KB | Runbook build generations: E4B's plans with screen results, the student's solutions with pass flags (no prompts) |
| `data/generations/xqlE2B2/xl_gens_qwen.jsonl.gz` | 211.2 KB | Greedy answers per problem with pass flag (186, Gemma 4 E2B student, seed 2): the untrained student |
| `data/generations/xqlE2B2/xl_gens_rb.jsonl.gz` | 209.6 KB | Greedy answers per problem with pass flag (186, Gemma 4 E2B student, seed 2): runbook rows |
| `data/generations/xqlE2B2/xl_gens_sft.jsonl.gz` | 204.1 KB | Greedy answers per problem with pass flag (186, Gemma 4 E2B student, seed 2): SFT on teacher text |
| `data/generations/xqlE2B2/xl_gens_sft_s.jsonl.gz` | 181.8 KB | Greedy answers per problem with pass flag (186, Gemma 4 E2B student, seed 2): SFT, shortest correct |
| `data/generations/xqlE2B2/xl_rb_gens.jsonl.gz` | 347.9 KB | Runbook build generations: E4B's plans with screen results, the student's solutions with pass flags (no prompts) |
| `data/generations/xqlE2B3/xl_gens_qwen.jsonl.gz` | 211.2 KB | Greedy answers per problem with pass flag (186, Gemma 4 E2B student, seed 3): the untrained student |
| `data/generations/xqlE2B3/xl_gens_rb.jsonl.gz` | 209.3 KB | Greedy answers per problem with pass flag (186, Gemma 4 E2B student, seed 3): runbook rows |
| `data/generations/xqlE2B3/xl_gens_sft.jsonl.gz` | 202.4 KB | Greedy answers per problem with pass flag (186, Gemma 4 E2B student, seed 3): SFT on teacher text |
| `data/generations/xqlE2B3/xl_gens_sft_s.jsonl.gz` | 183.9 KB | Greedy answers per problem with pass flag (186, Gemma 4 E2B student, seed 3): SFT, shortest correct |
| `data/generations/xqlE2B3/xl_rb_gens.jsonl.gz` | 347.9 KB | Runbook build generations: E4B's plans with screen results, the student's solutions with pass flags (no prompts) |
| `data/generations/xqlKL1/xl_gens_qwen.jsonl.gz` | 211.2 KB | Greedy answers per problem with pass flag (193, KL strength sweep (E2B), seed 1): the untrained student |
| `data/generations/xqlKL1/xl_gens_sft.jsonl.gz` | 206.7 KB | Greedy answers per problem with pass flag (193, KL strength sweep (E2B), seed 1): SFT on teacher text |
| `data/generations/xqlKL1/xl_gens_sft_self_s_l03.jsonl.gz` | 183.3 KB | Greedy answers per problem with pass flag (193, KL strength sweep (E2B), seed 1): SFT, shortest + KL to the original student at lambda 0.3 |
| `data/generations/xqlKL1/xl_gens_sft_self_s_l3.jsonl.gz` | 198.4 KB | Greedy answers per problem with pass flag (193, KL strength sweep (E2B), seed 1): SFT, shortest + KL to the original student at lambda 3 |
| `data/generations/xqlKL2/xl_gens_qwen.jsonl.gz` | 211.2 KB | Greedy answers per problem with pass flag (193, KL strength sweep (E2B), seed 2): the untrained student |
| `data/generations/xqlKL2/xl_gens_sft.jsonl.gz` | 204.1 KB | Greedy answers per problem with pass flag (193, KL strength sweep (E2B), seed 2): SFT on teacher text |
| `data/generations/xqlKL2/xl_gens_sft_self_s_l03.jsonl.gz` | 182.4 KB | Greedy answers per problem with pass flag (193, KL strength sweep (E2B), seed 2): SFT, shortest + KL to the original student at lambda 0.3 |
| `data/generations/xqlKL2/xl_gens_sft_self_s_l3.jsonl.gz` | 196.9 KB | Greedy answers per problem with pass flag (193, KL strength sweep (E2B), seed 2): SFT, shortest + KL to the original student at lambda 3 |
| `data/generations/xqlKL3/xl_gens_qwen.jsonl.gz` | 211.2 KB | Greedy answers per problem with pass flag (193, KL strength sweep (E2B), seed 3): the untrained student |
| `data/generations/xqlKL3/xl_gens_sft.jsonl.gz` | 202.4 KB | Greedy answers per problem with pass flag (193, KL strength sweep (E2B), seed 3): SFT on teacher text |
| `data/generations/xqlKL3/xl_gens_sft_self_s_l03.jsonl.gz` | 180.8 KB | Greedy answers per problem with pass flag (193, KL strength sweep (E2B), seed 3): SFT, shortest + KL to the original student at lambda 0.3 |
| `data/generations/xqlKL3/xl_gens_sft_self_s_l3.jsonl.gz` | 197.8 KB | Greedy answers per problem with pass flag (193, KL strength sweep (E2B), seed 3): SFT, shortest + KL to the original student at lambda 3 |
| `data/generations/xqlML1/xl_gens_qwen.jsonl.gz` | 211.2 KB | Greedy answers per problem with pass flag (188, matched-loss test (E2B), seed 1): the untrained student |
| `data/generations/xqlML1/xl_gens_sft.jsonl.gz` | 206.7 KB | Greedy answers per problem with pass flag (188, matched-loss test (E2B), seed 1): SFT on teacher text |
| `data/generations/xqlML1/xl_gens_sft_es.jsonl.gz` | 220.3 KB | Greedy answers per problem with pass flag (188, matched-loss test (E2B), seed 1): SFT stopped at training loss 0.24 |
| `data/generations/xqlML1/xl_gens_sft_lr.jsonl.gz` | 201.3 KB | Greedy answers per problem with pass flag (188, matched-loss test (E2B), seed 1): SFT at a tenth of the learning rate |
| `data/generations/xqlML2/xl_gens_qwen.jsonl.gz` | 211.2 KB | Greedy answers per problem with pass flag (188, matched-loss test (E2B), seed 2): the untrained student |
| `data/generations/xqlML2/xl_gens_sft.jsonl.gz` | 204.1 KB | Greedy answers per problem with pass flag (188, matched-loss test (E2B), seed 2): SFT on teacher text |
| `data/generations/xqlML2/xl_gens_sft_es.jsonl.gz` | 213.0 KB | Greedy answers per problem with pass flag (188, matched-loss test (E2B), seed 2): SFT stopped at training loss 0.24 |
| `data/generations/xqlML2/xl_gens_sft_lr.jsonl.gz` | 202.6 KB | Greedy answers per problem with pass flag (188, matched-loss test (E2B), seed 2): SFT at a tenth of the learning rate |
| `data/generations/xqlML3/xl_gens_qwen.jsonl.gz` | 211.2 KB | Greedy answers per problem with pass flag (188, matched-loss test (E2B), seed 3): the untrained student |
| `data/generations/xqlML3/xl_gens_sft.jsonl.gz` | 202.4 KB | Greedy answers per problem with pass flag (188, matched-loss test (E2B), seed 3): SFT on teacher text |
| `data/generations/xqlML3/xl_gens_sft_es.jsonl.gz` | 190.5 KB | Greedy answers per problem with pass flag (188, matched-loss test (E2B), seed 3): SFT stopped at training loss 0.24 |
| `data/generations/xqlML3/xl_gens_sft_lr.jsonl.gz` | 206.3 KB | Greedy answers per problem with pass flag (188, matched-loss test (E2B), seed 3): SFT at a tenth of the learning rate |
| `data/generations/xqlMT1/xl_gens_qwen.jsonl.gz` | 211.2 KB | Greedy answers per problem with pass flag (190, multi-task control (E2B), seed 1): the untrained student |
| `data/generations/xqlMT1/xl_gens_sft.jsonl.gz` | 206.7 KB | Greedy answers per problem with pass flag (190, multi-task control (E2B), seed 1): SFT on teacher text |
| `data/generations/xqlMT1/xl_gens_sft_mt_s.jsonl.gz` | 183.0 KB | Greedy answers per problem with pass flag (190, multi-task control (E2B), seed 1): SFT, shortest + the corrector's fitting text |
| `data/generations/xqlMT1/xl_gens_sft_self_mt_s.jsonl.gz` | 193.0 KB | Greedy answers per problem with pass flag (190, multi-task control (E2B), seed 1): SFT, shortest + fitting text + KL to the original student |
| `data/generations/xqlMT2/xl_gens_qwen.jsonl.gz` | 211.2 KB | Greedy answers per problem with pass flag (190, multi-task control (E2B), seed 2): the untrained student |
| `data/generations/xqlMT2/xl_gens_sft.jsonl.gz` | 204.1 KB | Greedy answers per problem with pass flag (190, multi-task control (E2B), seed 2): SFT on teacher text |
| `data/generations/xqlMT2/xl_gens_sft_mt_s.jsonl.gz` | 178.1 KB | Greedy answers per problem with pass flag (190, multi-task control (E2B), seed 2): SFT, shortest + the corrector's fitting text |
| `data/generations/xqlMT2/xl_gens_sft_self_mt_s.jsonl.gz` | 189.6 KB | Greedy answers per problem with pass flag (190, multi-task control (E2B), seed 2): SFT, shortest + fitting text + KL to the original student |
| `data/generations/xqlMT3/xl_gens_qwen.jsonl.gz` | 211.2 KB | Greedy answers per problem with pass flag (190, multi-task control (E2B), seed 3): the untrained student |
| `data/generations/xqlMT3/xl_gens_sft.jsonl.gz` | 202.4 KB | Greedy answers per problem with pass flag (190, multi-task control (E2B), seed 3): SFT on teacher text |
| `data/generations/xqlMT3/xl_gens_sft_mt_s.jsonl.gz` | 180.3 KB | Greedy answers per problem with pass flag (190, multi-task control (E2B), seed 3): SFT, shortest + the corrector's fitting text |
| `data/generations/xqlMT3/xl_gens_sft_self_mt_s.jsonl.gz` | 186.0 KB | Greedy answers per problem with pass flag (190, multi-task control (E2B), seed 3): SFT, shortest + fitting text + KL to the original student |
| `data/generations/xqlO1/xl_gens_mix_c.jsonl.gz` | 108.3 KB | Greedy answers per problem with pass flag (179, seed 1): SFT + on-policy, weak teacher |
| `data/generations/xqlO1/xl_gens_opd_c.jsonl.gz` | 81.2 KB | Greedy answers per problem with pass flag (179, seed 1): on-policy, weak teacher |
| `data/generations/xqlO1/xl_gens_qwen.jsonl.gz` | 61.9 KB | Greedy answers per problem with pass flag (179, seed 1): the untrained student |
| `data/generations/xqlO1/xl_gens_sft.jsonl.gz` | 214.3 KB | Greedy answers per problem with pass flag (179, seed 1): SFT on teacher text |
| `data/generations/xqlO1/xl_gens_sft_c_s.jsonl.gz` | 155.8 KB | Greedy answers per problem with pass flag (179, seed 1): SFT, shortest + self-anchor |
| `data/generations/xqlO1/xl_gens_sft_s.jsonl.gz` | 186.1 KB | Greedy answers per problem with pass flag (179, seed 1): SFT, shortest correct |
| `data/generations/xqlO2/xl_gens_mix_c.jsonl.gz` | 129.5 KB | Greedy answers per problem with pass flag (179, seed 2): SFT + on-policy, weak teacher |
| `data/generations/xqlO2/xl_gens_opd_c.jsonl.gz` | 98.2 KB | Greedy answers per problem with pass flag (179, seed 2): on-policy, weak teacher |
| `data/generations/xqlO2/xl_gens_qwen.jsonl.gz` | 61.9 KB | Greedy answers per problem with pass flag (179, seed 2): the untrained student |
| `data/generations/xqlO2/xl_gens_sft.jsonl.gz` | 201.3 KB | Greedy answers per problem with pass flag (179, seed 2): SFT on teacher text |
| `data/generations/xqlO2/xl_gens_sft_c_s.jsonl.gz` | 155.5 KB | Greedy answers per problem with pass flag (179, seed 2): SFT, shortest + self-anchor |
| `data/generations/xqlO2/xl_gens_sft_s.jsonl.gz` | 172.9 KB | Greedy answers per problem with pass flag (179, seed 2): SFT, shortest correct |
| `data/generations/xqlO3/xl_gens_mix_c.jsonl.gz` | 84.8 KB | Greedy answers per problem with pass flag (179, seed 3): SFT + on-policy, weak teacher |
| `data/generations/xqlO3/xl_gens_opd_c.jsonl.gz` | 89.3 KB | Greedy answers per problem with pass flag (179, seed 3): on-policy, weak teacher |
| `data/generations/xqlO3/xl_gens_qwen.jsonl.gz` | 61.9 KB | Greedy answers per problem with pass flag (179, seed 3): the untrained student |
| `data/generations/xqlO3/xl_gens_sft.jsonl.gz` | 210.7 KB | Greedy answers per problem with pass flag (179, seed 3): SFT on teacher text |
| `data/generations/xqlO3/xl_gens_sft_c_s.jsonl.gz` | 161.2 KB | Greedy answers per problem with pass flag (179, seed 3): SFT, shortest + self-anchor |
| `data/generations/xqlO3/xl_gens_sft_s.jsonl.gz` | 185.1 KB | Greedy answers per problem with pass flag (179, seed 3): SFT, shortest correct |
| `data/generations/xqlRB0/xl_rb_gens.jsonl.gz` | 170.5 KB | Runbook build generations: E4B's plans with screen results, the student's solutions with pass flags (no prompts) |
| `data/generations/xqlRB1/xl_gens_qwen.jsonl.gz` | 61.9 KB | Greedy answers per problem with pass flag (182, seed 1): the untrained student |
| `data/generations/xqlRB1/xl_gens_rb.jsonl.gz` | 94.6 KB | Greedy answers per problem with pass flag (182, seed 1): runbook rows |
| `data/generations/xqlRB1/xl_gens_sft.jsonl.gz` | 214.3 KB | Greedy answers per problem with pass flag (182, seed 1): SFT on teacher text |
| `data/generations/xqlRB2/xl_gens_qwen.jsonl.gz` | 61.9 KB | Greedy answers per problem with pass flag (182, seed 2): the untrained student |
| `data/generations/xqlRB2/xl_gens_rb.jsonl.gz` | 92.8 KB | Greedy answers per problem with pass flag (182, seed 2): runbook rows |
| `data/generations/xqlRB2/xl_gens_sft.jsonl.gz` | 201.3 KB | Greedy answers per problem with pass flag (182, seed 2): SFT on teacher text |
| `data/generations/xqlRB3/xl_gens_qwen.jsonl.gz` | 61.9 KB | Greedy answers per problem with pass flag (182, seed 3): the untrained student |
| `data/generations/xqlRB3/xl_gens_rb.jsonl.gz` | 86.9 KB | Greedy answers per problem with pass flag (182, seed 3): runbook rows |
| `data/generations/xqlRB3/xl_gens_sft.jsonl.gz` | 210.7 KB | Greedy answers per problem with pass flag (182, seed 3): SFT on teacher text |
| `data/generations/xqlRBC1/xl_gens_qwen.jsonl.gz` | 61.9 KB | Greedy answers per problem with pass flag (184c, seed 1): the untrained student |
| `data/generations/xqlRBC1/xl_gens_rb_c.jsonl.gz` | 82.4 KB | Greedy answers per problem with pass flag (184c, seed 1): runbook rows + self-anchor |
| `data/generations/xqlRBC1/xl_gens_sft.jsonl.gz` | 214.3 KB | Greedy answers per problem with pass flag (184c, seed 1): SFT on teacher text |
| `data/generations/xqlRBC2/xl_gens_qwen.jsonl.gz` | 61.9 KB | Greedy answers per problem with pass flag (184c, seed 2): the untrained student |
| `data/generations/xqlRBC2/xl_gens_rb_c.jsonl.gz` | 91.6 KB | Greedy answers per problem with pass flag (184c, seed 2): runbook rows + self-anchor |
| `data/generations/xqlRBC2/xl_gens_sft.jsonl.gz` | 201.3 KB | Greedy answers per problem with pass flag (184c, seed 2): SFT on teacher text |
| `data/generations/xqlRBC3/xl_gens_qwen.jsonl.gz` | 61.9 KB | Greedy answers per problem with pass flag (184c, seed 3): the untrained student |
| `data/generations/xqlRBC3/xl_gens_rb_c.jsonl.gz` | 84.6 KB | Greedy answers per problem with pass flag (184c, seed 3): runbook rows + self-anchor |
| `data/generations/xqlRBC3/xl_gens_sft.jsonl.gz` | 210.7 KB | Greedy answers per problem with pass flag (184c, seed 3): SFT on teacher text |
| `data/generations/xqlS1/xl_gens_kl_b.jsonl.gz` | 80.9 KB | Greedy answers per problem with pass flag (174, seed 1): cross-model anchor only |
| `data/generations/xqlS1/xl_gens_qwen.jsonl.gz` | 61.9 KB | Greedy answers per problem with pass flag (174, seed 1): the untrained student |
| `data/generations/xqlS1/xl_gens_sdft.jsonl.gz` | 101.2 KB | Greedy answers per problem with pass flag (174, seed 1): SDFT |
| `data/generations/xqlS1/xl_gens_sft.jsonl.gz` | 209.0 KB | Greedy answers per problem with pass flag (174, seed 1): SFT on teacher text |
| `data/generations/xqlS1/xl_gens_sft_b.jsonl.gz` | 180.4 KB | Greedy answers per problem with pass flag (174, seed 1): SFT + cross-model anchor |
| `data/generations/xqlS1/xl_gens_sft_c.jsonl.gz` | 174.0 KB | Greedy answers per problem with pass flag (174, seed 1): SFT + self-anchor |
| `data/generations/xqlS2/xl_gens_kl_b.jsonl.gz` | 101.3 KB | Greedy answers per problem with pass flag (174, seed 2): cross-model anchor only |
| `data/generations/xqlS2/xl_gens_qwen.jsonl.gz` | 61.9 KB | Greedy answers per problem with pass flag (174, seed 2): the untrained student |
| `data/generations/xqlS2/xl_gens_sdft.jsonl.gz` | 108.7 KB | Greedy answers per problem with pass flag (174, seed 2): SDFT |
| `data/generations/xqlS2/xl_gens_sft.jsonl.gz` | 198.6 KB | Greedy answers per problem with pass flag (174, seed 2): SFT on teacher text |
| `data/generations/xqlS2/xl_gens_sft_b.jsonl.gz` | 178.8 KB | Greedy answers per problem with pass flag (174, seed 2): SFT + cross-model anchor |
| `data/generations/xqlS2/xl_gens_sft_c.jsonl.gz` | 174.0 KB | Greedy answers per problem with pass flag (174, seed 2): SFT + self-anchor |
| `data/generations/xqlS3/xl_gens_kl_b.jsonl.gz` | 74.4 KB | Greedy answers per problem with pass flag (174, seed 3): cross-model anchor only |
| `data/generations/xqlS3/xl_gens_qwen.jsonl.gz` | 61.9 KB | Greedy answers per problem with pass flag (174, seed 3): the untrained student |
| `data/generations/xqlS3/xl_gens_sdft.jsonl.gz` | 90.8 KB | Greedy answers per problem with pass flag (174, seed 3): SDFT |
| `data/generations/xqlS3/xl_gens_sft.jsonl.gz` | 212.3 KB | Greedy answers per problem with pass flag (174, seed 3): SFT on teacher text |
| `data/generations/xqlS3/xl_gens_sft_b.jsonl.gz` | 180.2 KB | Greedy answers per problem with pass flag (174, seed 3): SFT + cross-model anchor |
| `data/generations/xqlS3/xl_gens_sft_c.jsonl.gz` | 184.2 KB | Greedy answers per problem with pass flag (174, seed 3): SFT + self-anchor |
| `data/generations/xqlSS1/xl_gens_qwen.jsonl.gz` | 61.9 KB | Greedy answers per problem with pass flag (185, seed 1): the untrained student |
| `data/generations/xqlSS1/xl_gens_sft.jsonl.gz` | 214.3 KB | Greedy answers per problem with pass flag (185, seed 1): SFT on teacher text |
| `data/generations/xqlSS1/xl_gens_sft_self_s.jsonl.gz` | 166.8 KB | Greedy answers per problem with pass flag (185, seed 1): SFT, shortest + KL to the original student |
| `data/generations/xqlSS2/xl_gens_qwen.jsonl.gz` | 61.9 KB | Greedy answers per problem with pass flag (185, seed 2): the untrained student |
| `data/generations/xqlSS2/xl_gens_sft.jsonl.gz` | 201.3 KB | Greedy answers per problem with pass flag (185, seed 2): SFT on teacher text |
| `data/generations/xqlSS2/xl_gens_sft_self_s.jsonl.gz` | 151.3 KB | Greedy answers per problem with pass flag (185, seed 2): SFT, shortest + KL to the original student |
| `data/generations/xqlSS3/xl_gens_qwen.jsonl.gz` | 61.9 KB | Greedy answers per problem with pass flag (185, seed 3): the untrained student |
| `data/generations/xqlSS3/xl_gens_sft.jsonl.gz` | 210.7 KB | Greedy answers per problem with pass flag (185, seed 3): SFT on teacher text |
| `data/generations/xqlSS3/xl_gens_sft_self_s.jsonl.gz` | 167.9 KB | Greedy answers per problem with pass flag (185, seed 3): SFT, shortest + KL to the original student |
| `data/generations/xqlTW2b1/xl_gens_qwen.jsonl.gz` | 211.2 KB | Greedy answers per problem with pass flag (187, E4B teaches E2B, seed 1): the untrained student |
| `data/generations/xqlTW2b1/xl_gens_sft.jsonl.gz` | 206.7 KB | Greedy answers per problem with pass flag (187, E4B teaches E2B, seed 1): SFT on teacher text |
| `data/generations/xqlTW2b1/xl_gens_sft_b_s.jsonl.gz` | 173.7 KB | Greedy answers per problem with pass flag (187, E4B teaches E2B, seed 1): SFT, shortest + cross-model anchor |
| `data/generations/xqlTW2b1/xl_gens_sft_c_s.jsonl.gz` | 173.0 KB | Greedy answers per problem with pass flag (187, E4B teaches E2B, seed 1): SFT, shortest + self-anchor |
| `data/generations/xqlTW2b1/xl_gens_sft_self_s.jsonl.gz` | 187.2 KB | Greedy answers per problem with pass flag (187, E4B teaches E2B, seed 1): SFT, shortest + KL to the original student |
| `data/generations/xqlTW2b2/xl_gens_qwen.jsonl.gz` | 211.2 KB | Greedy answers per problem with pass flag (187, E4B teaches E2B, seed 2): the untrained student |
| `data/generations/xqlTW2b2/xl_gens_sft.jsonl.gz` | 204.1 KB | Greedy answers per problem with pass flag (187, E4B teaches E2B, seed 2): SFT on teacher text |
| `data/generations/xqlTW2b2/xl_gens_sft_b_s.jsonl.gz` | 171.3 KB | Greedy answers per problem with pass flag (187, E4B teaches E2B, seed 2): SFT, shortest + cross-model anchor |
| `data/generations/xqlTW2b2/xl_gens_sft_c_s.jsonl.gz` | 173.1 KB | Greedy answers per problem with pass flag (187, E4B teaches E2B, seed 2): SFT, shortest + self-anchor |
| `data/generations/xqlTW2b2/xl_gens_sft_self_s.jsonl.gz` | 185.3 KB | Greedy answers per problem with pass flag (187, E4B teaches E2B, seed 2): SFT, shortest + KL to the original student |
| `data/generations/xqlTW2b3/xl_gens_qwen.jsonl.gz` | 211.2 KB | Greedy answers per problem with pass flag (187, E4B teaches E2B, seed 3): the untrained student |
| `data/generations/xqlTW2b3/xl_gens_sft.jsonl.gz` | 202.4 KB | Greedy answers per problem with pass flag (187, E4B teaches E2B, seed 3): SFT on teacher text |
| `data/generations/xqlTW2b3/xl_gens_sft_b_s.jsonl.gz` | 181.6 KB | Greedy answers per problem with pass flag (187, E4B teaches E2B, seed 3): SFT, shortest + cross-model anchor |
| `data/generations/xqlTW2b3/xl_gens_sft_c_s.jsonl.gz` | 177.5 KB | Greedy answers per problem with pass flag (187, E4B teaches E2B, seed 3): SFT, shortest + self-anchor |
| `data/generations/xqlTW2b3/xl_gens_sft_self_s.jsonl.gz` | 183.4 KB | Greedy answers per problem with pass flag (187, E4B teaches E2B, seed 3): SFT, shortest + KL to the original student |
| `data/generations/xqlTW4b1/xl_gens_qwen.jsonl.gz` | 135.5 KB | Greedy answers per problem with pass flag (187, E2B teaches E4B, seed 1): the untrained student |
| `data/generations/xqlTW4b1/xl_gens_sft.jsonl.gz` | 131.4 KB | Greedy answers per problem with pass flag (187, E2B teaches E4B, seed 1): SFT on teacher text |
| `data/generations/xqlTW4b1/xl_gens_sft_b.jsonl.gz` | 123.9 KB | Greedy answers per problem with pass flag (187, E2B teaches E4B, seed 1): SFT + cross-model anchor |
| `data/generations/xqlTW4b1/xl_gens_sft_c.jsonl.gz` | 119.8 KB | Greedy answers per problem with pass flag (187, E2B teaches E4B, seed 1): SFT + self-anchor |
| `data/generations/xqlTW4b1/xl_gens_sft_self.jsonl.gz` | 126.4 KB | Greedy answers per problem with pass flag (187, E2B teaches E4B, seed 1): SFT + KL to the original student |
| `data/generations/xqlTW4b2/xl_gens_qwen.jsonl.gz` | 135.5 KB | Greedy answers per problem with pass flag (187, E2B teaches E4B, seed 2): the untrained student |
| `data/generations/xqlTW4b2/xl_gens_sft.jsonl.gz` | 131.8 KB | Greedy answers per problem with pass flag (187, E2B teaches E4B, seed 2): SFT on teacher text |
| `data/generations/xqlTW4b2/xl_gens_sft_b.jsonl.gz` | 128.2 KB | Greedy answers per problem with pass flag (187, E2B teaches E4B, seed 2): SFT + cross-model anchor |
| `data/generations/xqlTW4b2/xl_gens_sft_c.jsonl.gz` | 127.4 KB | Greedy answers per problem with pass flag (187, E2B teaches E4B, seed 2): SFT + self-anchor |
| `data/generations/xqlTW4b2/xl_gens_sft_self.jsonl.gz` | 131.4 KB | Greedy answers per problem with pass flag (187, E2B teaches E4B, seed 2): SFT + KL to the original student |
| `data/generations/xqlTW4b3/xl_gens_qwen.jsonl.gz` | 135.5 KB | Greedy answers per problem with pass flag (187, E2B teaches E4B, seed 3): the untrained student |
| `data/generations/xqlTW4b3/xl_gens_sft.jsonl.gz` | 136.4 KB | Greedy answers per problem with pass flag (187, E2B teaches E4B, seed 3): SFT on teacher text |
| `data/generations/xqlTW4b3/xl_gens_sft_b.jsonl.gz` | 129.9 KB | Greedy answers per problem with pass flag (187, E2B teaches E4B, seed 3): SFT + cross-model anchor |
| `data/generations/xqlTW4b3/xl_gens_sft_c.jsonl.gz` | 128.9 KB | Greedy answers per problem with pass flag (187, E2B teaches E4B, seed 3): SFT + self-anchor |
| `data/generations/xqlTW4b3/xl_gens_sft_self.jsonl.gz` | 129.0 KB | Greedy answers per problem with pass flag (187, E2B teaches E4B, seed 3): SFT + KL to the original student |
| `data/generations/xqxC1/xl_gens_onl_cv.jsonl.gz` | 198.9 KB | Greedy answers per problem with pass flag (209g, E4B's distribution on passing answers, seed 1): student of E4B's distribution, passing answers only |
| `data/generations/xqxC1/xl_gens_qwen.jsonl.gz` | 211.2 KB | Greedy answers per problem with pass flag (209g, E4B's distribution on passing answers, seed 1): the untrained student |
| `data/generations/xqxC2/xl_gens_onl_cv.jsonl.gz` | 211.6 KB | Greedy answers per problem with pass flag (209g, E4B's distribution on passing answers, seed 2): student of E4B's distribution, passing answers only |
| `data/generations/xqxC2/xl_gens_qwen.jsonl.gz` | 211.2 KB | Greedy answers per problem with pass flag (209g, E4B's distribution on passing answers, seed 2): the untrained student |
| `data/generations/xqxD1/xl_gens_onl_d4v.jsonl.gz` | 195.8 KB | Greedy answers per problem with pass flag (209g, cross-model corrector fitted before the rewriting, seed 1): student of the cross-model corrector fitted on verified solutions |
| `data/generations/xqxD1/xl_gens_qwen.jsonl.gz` | 211.2 KB | Greedy answers per problem with pass flag (209g, cross-model corrector fitted before the rewriting, seed 1): the untrained student |
| `data/generations/xqxS1/xl_gens_onl_s4v.jsonl.gz` | 204.6 KB | Greedy answers per problem with pass flag (209g, self-corrector fitted before the rewriting, seed 1): student of the self-corrector fitted on verified solutions |
| `data/generations/xqxS1/xl_gens_qwen.jsonl.gz` | 211.2 KB | Greedy answers per problem with pass flag (209g, self-corrector fitted before the rewriting, seed 1): the untrained student |
| `data/generations/xqxS2/xl_gens_onl_s4v.jsonl.gz` | 196.5 KB | Greedy answers per problem with pass flag (209g, self-corrector fitted before the rewriting, seed 2): student of the self-corrector fitted on verified solutions |
| `data/generations/xqxS2/xl_gens_qwen.jsonl.gz` | 211.2 KB | Greedy answers per problem with pass flag (209g, self-corrector fitted before the rewriting, seed 2): the untrained student |
| `data/generations/xqyTgen/gen_scored.jsonl.gz` | 39.4 KB | E4B's answers on 77 held problems (209q, E4B with its own corrector on): id, kind of its stored answer, text, pass |
| `data/generations/xqyTgenA/gen_scored_alone.jsonl.gz` | 36.4 KB | E4B's answers on 77 held problems (209q control, the corrector off): id, kind of its stored answer, text, pass |
| `data/generations/xqzC3/xl_gens_onl_cv.jsonl.gz` | 168.0 KB | Greedy answers per problem with pass flag (209g, E4B's distribution on passing answers, seed 3): student of E4B's distribution, passing answers only |
| `data/generations/xqzC3/xl_gens_qwen.jsonl.gz` | 211.2 KB | Greedy answers per problem with pass flag (209g, E4B's distribution on passing answers, seed 3): the untrained student |
| `data/generations/xqzD1/xl_gens_onl_d4w.jsonl.gz` | 166.1 KB | Greedy answers per problem with pass flag (209i, cross-model corrector, seed 1): student of the cross-model corrector fitted with token targets (209i; the hybrids 209n, 209r; the penalty 209p) |
| `data/generations/xqzD1/xl_gens_qwen.jsonl.gz` | 211.2 KB | Greedy answers per problem with pass flag (209i, cross-model corrector, seed 1): the untrained student |
| `data/generations/xqzD2/xl_gens_onl_d4w.jsonl.gz` | 165.9 KB | Greedy answers per problem with pass flag (209i, cross-model corrector, seed 2): student of the cross-model corrector fitted with token targets (209i; the hybrids 209n, 209r; the penalty 209p) |
| `data/generations/xqzD2/xl_gens_qwen.jsonl.gz` | 211.2 KB | Greedy answers per problem with pass flag (209i, cross-model corrector, seed 2): the untrained student |
| `data/generations/xqzF3/xl_gens_onl_sft.jsonl.gz` | 186.5 KB | Greedy answers per problem with pass flag (209, E4B's verified text, seed 3): student of E4B's verified text (online SFT) |
| `data/generations/xqzF3/xl_gens_qwen.jsonl.gz` | 211.2 KB | Greedy answers per problem with pass flag (209, E4B's verified text, seed 3): the untrained student |
| `data/generations/xqzF3/xl_gens_sft.jsonl.gz` | 202.4 KB | Greedy answers per problem with pass flag (209, E4B's verified text, seed 3): SFT on teacher text |
| `data/generations/xqzN1/xl_gens_onl_d4w.jsonl.gz` | 185.1 KB | Greedy answers per problem with pass flag (209n, the hybrid, seed 1): student of the cross-model corrector fitted with token targets (209i; the hybrids 209n, 209r; the penalty 209p) |
| `data/generations/xqzN1/xl_gens_qwen.jsonl.gz` | 211.2 KB | Greedy answers per problem with pass flag (209n, the hybrid, seed 1): the untrained student |
| `data/generations/xqzN2/xl_gens_onl_d4w.jsonl.gz` | 186.2 KB | Greedy answers per problem with pass flag (209n, the hybrid, seed 2): student of the cross-model corrector fitted with token targets (209i; the hybrids 209n, 209r; the penalty 209p) |
| `data/generations/xqzN2/xl_gens_qwen.jsonl.gz` | 211.2 KB | Greedy answers per problem with pass flag (209n, the hybrid, seed 2): the untrained student |
| `data/generations/xqzN3/xl_gens_onl_d4w.jsonl.gz` | 165.6 KB | Greedy answers per problem with pass flag (209n, the hybrid, seed 3): student of the cross-model corrector fitted with token targets (209i; the hybrids 209n, 209r; the penalty 209p) |
| `data/generations/xqzN3/xl_gens_qwen.jsonl.gz` | 211.2 KB | Greedy answers per problem with pass flag (209n, the hybrid, seed 3): the untrained student |
| `data/generations/xqzP1/xl_gens_onl_d4w.jsonl.gz` | 170.0 KB | Greedy answers per problem with pass flag (209p, placeholder penalty, seed 1): student of the cross-model corrector fitted with token targets (209i; the hybrids 209n, 209r; the penalty 209p) |
| `data/generations/xqzP1/xl_gens_qwen.jsonl.gz` | 211.2 KB | Greedy answers per problem with pass flag (209p, placeholder penalty, seed 1): the untrained student |
| `data/generations/xqzR1/xl_gens_onl_d4w.jsonl.gz` | 152.1 KB | Greedy answers per problem with pass flag (209r, the hybrid copying the reviewed solution, seed 1): student of the cross-model corrector fitted with token targets (209i; the hybrids 209n, 209r; the penalty 209p) |
| `data/generations/xqzR1/xl_gens_qwen.jsonl.gz` | 211.2 KB | Greedy answers per problem with pass flag (209r, the hybrid copying the reviewed solution, seed 1): the untrained student |
| `data/generations/xqzS1/xl_gens_onl_s4w.jsonl.gz` | 164.1 KB | Greedy answers per problem with pass flag (209i, self-corrector, seed 1): student of the self-corrector fitted with token targets (209i) |
| `data/generations/xqzS1/xl_gens_qwen.jsonl.gz` | 211.2 KB | Greedy answers per problem with pass flag (209i, self-corrector, seed 1): the untrained student |
| `data/generations/xqzS2/xl_gens_onl_s4w.jsonl.gz` | 153.3 KB | Greedy answers per problem with pass flag (209i, self-corrector, seed 2): student of the self-corrector fitted with token targets (209i) |
| `data/generations/xqzS2/xl_gens_qwen.jsonl.gz` | 211.2 KB | Greedy answers per problem with pass flag (209i, self-corrector, seed 2): the untrained student |
| `data/lcb/fit/gate_lora_onl_bridge_keep_replay_lcb_s2.json` | 1.2 KB | Agent gate of the coding-taught LoRA [209a's correctors] (onl_bridge_keep_replay_lcb_s2) |
| `data/lcb/fit/gate_lora_onl_bridge_keep_replay_lcb_s2_points.json` | 23.3 KB | Agent gate of the coding-taught LoRA [209a's correctors] (onl_bridge_keep_replay_lcb_s2_points) |
| `data/lcb/fit/online_bridge_keep_replay_lcb_s2.json` | 4.6 KB | Online coding run [209a's correctors]: stream curve and counts of passing samples (bridge_keep_replay_lcb_s2) |
| `data/lcb/gate_lora_onl_bridge_keep_replay_lcb.json` | 1.2 KB | Agent gate of the coding-taught LoRA (onl_bridge_keep_replay_lcb) |
| `data/lcb/gate_lora_onl_bridge_keep_replay_lcb_points.json` | 23.3 KB | Agent gate of the coding-taught LoRA (onl_bridge_keep_replay_lcb_points) |
| `data/lcb/gate_lora_onl_bridge_keep_replay_lcbv.json` | 1.2 KB | Agent gate of the coding-taught LoRA (onl_bridge_keep_replay_lcbv) |
| `data/lcb/gate_lora_onl_bridge_keep_replay_lcbv_points.json` | 23.3 KB | Agent gate of the coding-taught LoRA (onl_bridge_keep_replay_lcbv_points) |
| `data/lcb/gate_lora_onl_control_keep_replay_lcbv.json` | 1.2 KB | Agent gate of the coding-taught LoRA (onl_control_keep_replay_lcbv) |
| `data/lcb/gate_lora_onl_control_keep_replay_lcbv_points.json` | 23.2 KB | Agent gate of the coding-taught LoRA (onl_control_keep_replay_lcbv_points) |
| `data/lcb/gate_lora_onl_control_keep_replay_lcbv_s2.json` | 1.2 KB | Agent gate of the coding-taught LoRA (onl_control_keep_replay_lcbv_s2) |
| `data/lcb/gate_lora_onl_control_keep_replay_lcbv_s2_points.json` | 23.3 KB | Agent gate of the coding-taught LoRA (onl_control_keep_replay_lcbv_s2_points) |
| `data/lcb/gate_lora_onl_direct_keep_replay_lcb.json` | 1.2 KB | Agent gate of the coding-taught LoRA (onl_direct_keep_replay_lcb) |
| `data/lcb/gate_lora_onl_direct_keep_replay_lcb_points.json` | 23.3 KB | Agent gate of the coding-taught LoRA (onl_direct_keep_replay_lcb_points) |
| `data/lcb/gate_lora_onl_direct_keep_replay_lcb_s2.json` | 1.2 KB | Agent gate of the coding-taught LoRA (onl_direct_keep_replay_lcb_s2) |
| `data/lcb/gate_lora_onl_direct_keep_replay_lcb_s2_points.json` | 23.2 KB | Agent gate of the coding-taught LoRA (onl_direct_keep_replay_lcb_s2_points) |
| `data/lcb/gate_lora_onl_direct_keep_replay_lcbv.json` | 1.2 KB | Agent gate of the coding-taught LoRA (onl_direct_keep_replay_lcbv) |
| `data/lcb/gate_lora_onl_direct_keep_replay_lcbv_points.json` | 23.3 KB | Agent gate of the coding-taught LoRA (onl_direct_keep_replay_lcbv_points) |
| `data/lcb/gate_lora_onl_direct_keep_replay_lcbv_s2.json` | 1.2 KB | Agent gate of the coding-taught LoRA (onl_direct_keep_replay_lcbv_s2) |
| `data/lcb/gate_lora_onl_direct_keep_replay_lcbv_s2_points.json` | 23.3 KB | Agent gate of the coding-taught LoRA (onl_direct_keep_replay_lcbv_s2_points) |
| `data/lcb/gate_lora_onl_direct_keep_replay_lcbv_s3.json` | 1.2 KB | Agent gate of the coding-taught LoRA (onl_direct_keep_replay_lcbv_s3) |
| `data/lcb/gate_lora_onl_direct_keep_replay_lcbv_s3_points.json` | 23.3 KB | Agent gate of the coding-taught LoRA (onl_direct_keep_replay_lcbv_s3_points) |
| `data/lcb/gate_lora_onl_sft_keep_replay_lcb.json` | 1.2 KB | Agent gate of the coding-taught LoRA (onl_sft_keep_replay_lcb) |
| `data/lcb/gate_lora_onl_sft_keep_replay_lcb_points.json` | 23.3 KB | Agent gate of the coding-taught LoRA (onl_sft_keep_replay_lcb_points) |
| `data/lcb/gate_lora_onl_sft_keep_replay_lcb_s2.json` | 1.2 KB | Agent gate of the coding-taught LoRA (onl_sft_keep_replay_lcb_s2) |
| `data/lcb/gate_lora_onl_sft_keep_replay_lcb_s2_points.json` | 23.3 KB | Agent gate of the coding-taught LoRA (onl_sft_keep_replay_lcb_s2_points) |
| `data/lcb/gate_lora_onl_sft_keep_replay_lcb_s3.json` | 1.2 KB | Agent gate of the coding-taught LoRA (onl_sft_keep_replay_lcb_s3) |
| `data/lcb/gate_lora_onl_sft_keep_replay_lcb_s3_points.json` | 23.3 KB | Agent gate of the coding-taught LoRA (onl_sft_keep_replay_lcb_s3_points) |
| `data/lcb/i/gate_lora_onl_bridge_keep_replay_lcbv.json` | 1.2 KB | Agent gate of the coding-taught LoRA [209i] (onl_bridge_keep_replay_lcbv) |
| `data/lcb/i/gate_lora_onl_bridge_keep_replay_lcbv_points.json` | 23.3 KB | Agent gate of the coding-taught LoRA [209i] (onl_bridge_keep_replay_lcbv_points) |
| `data/lcb/i/gate_lora_onl_bridge_keep_replay_lcbv_s2.json` | 1.2 KB | Agent gate of the coding-taught LoRA [209i] (onl_bridge_keep_replay_lcbv_s2) |
| `data/lcb/i/gate_lora_onl_bridge_keep_replay_lcbv_s2_points.json` | 23.3 KB | Agent gate of the coding-taught LoRA [209i] (onl_bridge_keep_replay_lcbv_s2_points) |
| `data/lcb/i/gate_lora_onl_control_keep_replay_lcbv.json` | 1.2 KB | Agent gate of the coding-taught LoRA [209i] (onl_control_keep_replay_lcbv) |
| `data/lcb/i/gate_lora_onl_control_keep_replay_lcbv_points.json` | 23.3 KB | Agent gate of the coding-taught LoRA [209i] (onl_control_keep_replay_lcbv_points) |
| `data/lcb/i/gate_lora_onl_control_keep_replay_lcbv_s2.json` | 1.2 KB | Agent gate of the coding-taught LoRA [209i] (onl_control_keep_replay_lcbv_s2) |
| `data/lcb/i/gate_lora_onl_control_keep_replay_lcbv_s2_points.json` | 23.3 KB | Agent gate of the coding-taught LoRA [209i] (onl_control_keep_replay_lcbv_s2_points) |
| `data/lcb/i/online_bridge_keep_replay_lcbv.json` | 4.9 KB | Online coding run [209i]: stream curve and counts of passing samples (bridge_keep_replay_lcbv) |
| `data/lcb/i/online_bridge_keep_replay_lcbv_s2.json` | 4.9 KB | Online coding run [209i]: stream curve and counts of passing samples (bridge_keep_replay_lcbv_s2) |
| `data/lcb/i/online_control_keep_replay_lcbv.json` | 4.9 KB | Online coding run [209i]: stream curve and counts of passing samples (control_keep_replay_lcbv) |
| `data/lcb/i/online_control_keep_replay_lcbv_s2.json` | 4.9 KB | Online coding run [209i]: stream curve and counts of passing samples (control_keep_replay_lcbv_s2) |
| `data/lcb/i/xl_eval_lcb_onl_d4w_s1.json` | 33.9 KB | Online coding evaluation [209i]: per-problem pass for untrained E2B and the arm (onl_d4w_s1), held CE, off-task CE |
| `data/lcb/i/xl_eval_lcb_onl_d4w_s2.json` | 33.9 KB | Online coding evaluation [209i]: per-problem pass for untrained E2B and the arm (onl_d4w_s2), held CE, off-task CE |
| `data/lcb/i/xl_eval_lcb_onl_s4w_s1.json` | 33.9 KB | Online coding evaluation [209i]: per-problem pass for untrained E2B and the arm (onl_s4w_s1), held CE, off-task CE |
| `data/lcb/i/xl_eval_lcb_onl_s4w_s2.json` | 33.9 KB | Online coding evaluation [209i]: per-problem pass for untrained E2B and the arm (onl_s4w_s2), held CE, off-task CE |
| `data/lcb/n/gate_lora_onl_bridge_keep_replay_lcbv.json` | 1.2 KB | Agent gate of the coding-taught LoRA [209n, the hybrid] (onl_bridge_keep_replay_lcbv) |
| `data/lcb/n/gate_lora_onl_bridge_keep_replay_lcbv_points.json` | 23.3 KB | Agent gate of the coding-taught LoRA [209n, the hybrid] (onl_bridge_keep_replay_lcbv_points) |
| `data/lcb/n/gate_lora_onl_bridge_keep_replay_lcbv_s2.json` | 1.2 KB | Agent gate of the coding-taught LoRA [209n, the hybrid] (onl_bridge_keep_replay_lcbv_s2) |
| `data/lcb/n/gate_lora_onl_bridge_keep_replay_lcbv_s2_points.json` | 23.3 KB | Agent gate of the coding-taught LoRA [209n, the hybrid] (onl_bridge_keep_replay_lcbv_s2_points) |
| `data/lcb/n/gate_lora_onl_bridge_keep_replay_lcbv_s3.json` | 1.2 KB | Agent gate of the coding-taught LoRA [209n, the hybrid] (onl_bridge_keep_replay_lcbv_s3) |
| `data/lcb/n/gate_lora_onl_bridge_keep_replay_lcbv_s3_points.json` | 23.3 KB | Agent gate of the coding-taught LoRA [209n, the hybrid] (onl_bridge_keep_replay_lcbv_s3_points) |
| `data/lcb/n/online_bridge_keep_replay_lcbv.json` | 5.0 KB | Online coding run [209n, the hybrid]: stream curve and counts of passing samples (bridge_keep_replay_lcbv) |
| `data/lcb/n/online_bridge_keep_replay_lcbv_s2.json` | 5.0 KB | Online coding run [209n, the hybrid]: stream curve and counts of passing samples (bridge_keep_replay_lcbv_s2) |
| `data/lcb/n/online_bridge_keep_replay_lcbv_s3.json` | 5.0 KB | Online coding run [209n, the hybrid]: stream curve and counts of passing samples (bridge_keep_replay_lcbv_s3) |
| `data/lcb/n/xl_eval_lcb_onl_d4w_s1.json` | 33.9 KB | Online coding evaluation [209n, the hybrid]: per-problem pass for untrained E2B and the arm (onl_d4w_s1), held CE, off-task CE |
| `data/lcb/n/xl_eval_lcb_onl_d4w_s2.json` | 33.9 KB | Online coding evaluation [209n, the hybrid]: per-problem pass for untrained E2B and the arm (onl_d4w_s2), held CE, off-task CE |
| `data/lcb/n/xl_eval_lcb_onl_d4w_s3.json` | 33.9 KB | Online coding evaluation [209n, the hybrid]: per-problem pass for untrained E2B and the arm (onl_d4w_s3), held CE, off-task CE |
| `data/lcb/online_bridge_keep_replay_lcb.json` | 4.6 KB | Online coding run: stream curve and counts of passing samples (bridge_keep_replay_lcb) |
| `data/lcb/online_bridge_keep_replay_lcbv.json` | 4.9 KB | Online coding run: stream curve and counts of passing samples (bridge_keep_replay_lcbv) |
| `data/lcb/online_control_keep_replay_lcbv.json` | 4.9 KB | Online coding run: stream curve and counts of passing samples (control_keep_replay_lcbv) |
| `data/lcb/online_control_keep_replay_lcbv_s2.json` | 4.9 KB | Online coding run: stream curve and counts of passing samples (control_keep_replay_lcbv_s2) |
| `data/lcb/online_direct_keep_replay_lcb.json` | 4.6 KB | Online coding run: stream curve and counts of passing samples (direct_keep_replay_lcb) |
| `data/lcb/online_direct_keep_replay_lcb_s2.json` | 4.6 KB | Online coding run: stream curve and counts of passing samples (direct_keep_replay_lcb_s2) |
| `data/lcb/online_direct_keep_replay_lcbv.json` | 4.9 KB | Online coding run: stream curve and counts of passing samples (direct_keep_replay_lcbv) |
| `data/lcb/online_direct_keep_replay_lcbv_s2.json` | 4.9 KB | Online coding run: stream curve and counts of passing samples (direct_keep_replay_lcbv_s2) |
| `data/lcb/online_direct_keep_replay_lcbv_s3.json` | 4.9 KB | Online coding run: stream curve and counts of passing samples (direct_keep_replay_lcbv_s3) |
| `data/lcb/online_sft_keep_replay_lcb.json` | 4.6 KB | Online coding run: stream curve and counts of passing samples (sft_keep_replay_lcb) |
| `data/lcb/online_sft_keep_replay_lcb_s2.json` | 4.6 KB | Online coding run: stream curve and counts of passing samples (sft_keep_replay_lcb_s2) |
| `data/lcb/online_sft_keep_replay_lcb_s3.json` | 4.6 KB | Online coding run: stream curve and counts of passing samples (sft_keep_replay_lcb_s3) |
| `data/lcb/p/gate_lora_onl_bridge_keep_replay_lcbv.json` | 1.3 KB | Agent gate of the coding-taught LoRA [209p] (onl_bridge_keep_replay_lcbv) |
| `data/lcb/p/gate_lora_onl_bridge_keep_replay_lcbv_points.json` | 23.2 KB | Agent gate of the coding-taught LoRA [209p] (onl_bridge_keep_replay_lcbv_points) |
| `data/lcb/p/online_bridge_keep_replay_lcbv.json` | 5.0 KB | Online coding run [209p]: stream curve and counts of passing samples (bridge_keep_replay_lcbv) |
| `data/lcb/p/xl_eval_lcb_onl_d4w_s1.json` | 33.9 KB | Online coding evaluation [209p]: per-problem pass for untrained E2B and the arm (onl_d4w_s1), held CE, off-task CE |
| `data/lcb/r/gate_lora_onl_bridge_keep_replay_lcbv.json` | 1.3 KB | Agent gate of the coding-taught LoRA [209r] (onl_bridge_keep_replay_lcbv) |
| `data/lcb/r/gate_lora_onl_bridge_keep_replay_lcbv_points.json` | 23.2 KB | Agent gate of the coding-taught LoRA [209r] (onl_bridge_keep_replay_lcbv_points) |
| `data/lcb/r/online_bridge_keep_replay_lcbv.json` | 5.1 KB | Online coding run [209r]: stream curve and counts of passing samples (bridge_keep_replay_lcbv) |
| `data/lcb/r/xl_eval_lcb_onl_d4w_s1.json` | 33.9 KB | Online coding evaluation [209r]: per-problem pass for untrained E2B and the arm (onl_d4w_s1), held CE, off-task CE |
| `data/lcb/xl_base_ids.json` | 4.9 KB | Online coding: ids of the 299 training and 188 held problems |
| `data/lcb/xl_eval_lcb_onl_c_s1.json` | 33.9 KB | Online coding evaluation: per-problem pass for untrained E2B and the arm (onl_c_s1), held CE, off-task CE |
| `data/lcb/xl_eval_lcb_onl_c_s2.json` | 33.9 KB | Online coding evaluation: per-problem pass for untrained E2B and the arm (onl_c_s2), held CE, off-task CE |
| `data/lcb/xl_eval_lcb_onl_cv_s1.json` | 33.9 KB | Online coding evaluation: per-problem pass for untrained E2B and the arm (onl_cv_s1), held CE, off-task CE |
| `data/lcb/xl_eval_lcb_onl_cv_s2.json` | 33.9 KB | Online coding evaluation: per-problem pass for untrained E2B and the arm (onl_cv_s2), held CE, off-task CE |
| `data/lcb/xl_eval_lcb_onl_cv_s3.json` | 34.0 KB | Online coding evaluation: per-problem pass for untrained E2B and the arm (onl_cv_s3), held CE, off-task CE |
| `data/lcb/xl_eval_lcb_onl_d4_s1.json` | 34.0 KB | Online coding evaluation: per-problem pass for untrained E2B and the arm (onl_d4_s1), held CE, off-task CE |
| `data/lcb/xl_eval_lcb_onl_d4f_s2.json` | 34.0 KB | Online coding evaluation: per-problem pass for untrained E2B and the arm (onl_d4f_s2), held CE, off-task CE |
| `data/lcb/xl_eval_lcb_onl_d4v_s1.json` | 34.0 KB | Online coding evaluation: per-problem pass for untrained E2B and the arm (onl_d4v_s1), held CE, off-task CE |
| `data/lcb/xl_eval_lcb_onl_s4v_s1.json` | 33.9 KB | Online coding evaluation: per-problem pass for untrained E2B and the arm (onl_s4v_s1), held CE, off-task CE |
| `data/lcb/xl_eval_lcb_onl_s4v_s2.json` | 33.9 KB | Online coding evaluation: per-problem pass for untrained E2B and the arm (onl_s4v_s2), held CE, off-task CE |
| `data/lcb/xl_eval_lcb_onl_sft_s1.json` | 50.8 KB | Online coding evaluation: per-problem pass for untrained E2B and the arm (onl_sft_s1), held CE, off-task CE |
| `data/lcb/xl_eval_lcb_onl_sft_s2.json` | 50.8 KB | Online coding evaluation: per-problem pass for untrained E2B and the arm (onl_sft_s2), held CE, off-task CE |
| `data/lcb/xl_eval_lcb_onl_sft_s3.json` | 50.8 KB | Online coding evaluation: per-problem pass for untrained E2B and the arm (onl_sft_s3), held CE, off-task CE |
| `data/live/live80_attempts.json` | 180.1 KB | The live test (196, 202, 207, 208): per attempt and task, whether the hidden tests pass, how the episode ended, and the number of model calls |
| `data/online/a100_4x_gate_lora_onl_bridge_keep_replay.json` | 1.2 KB | Online test gate on 240 held decision points: cross-model corrector + keep + replay (D), corrector fitted 4x longer, seed 1, Colab A100 |
| `data/online/a100_4x_gate_lora_onl_bridge_keep_replay_points.json` | 23.2 KB | Online test gate, per decision and per row: cross-model corrector + keep + replay (D), corrector fitted 4x longer, seed 1, Colab A100 |
| `data/online/a100_4x_online_bridge_keep_replay.json` | 4.6 KB | Online test run: prequential windows and held curve: cross-model corrector + keep + replay (D), corrector fitted 4x longer, seed 1, Colab A100 |
| `data/online/a100_gate_lora_onl_bridge_keep_replay.json` | 1.2 KB | Online test gate on 240 held decision points: cross-model corrector + keep + replay (D), seed 1, Colab A100 |
| `data/online/a100_gate_lora_onl_bridge_keep_replay_points.json` | 23.2 KB | Online test gate, per decision and per row: cross-model corrector + keep + replay (D), seed 1, Colab A100 |
| `data/online/a100_gate_lora_onl_bridge_keep_replay_s3.json` | 1.2 KB | Online test gate on 240 held decision points: cross-model corrector + keep + replay (D), seed 3, Colab A100 |
| `data/online/a100_gate_lora_onl_bridge_keep_replay_s3_points.json` | 23.2 KB | Online test gate, per decision and per row: cross-model corrector + keep + replay (D), seed 3, Colab A100 |
| `data/online/a100_gate_lora_onl_direct_keep_replay.json` | 1.2 KB | Online test gate on 240 held decision points: E4B's distribution + keep + replay (C), seed 1, Colab A100 |
| `data/online/a100_gate_lora_onl_direct_keep_replay_points.json` | 23.2 KB | Online test gate, per decision and per row: E4B's distribution + keep + replay (C), seed 1, Colab A100 |
| `data/online/a100_gate_lora_onl_direct_keep_replay_s2.json` | 1.2 KB | Online test gate on 240 held decision points: E4B's distribution + keep + replay (C), seed 2, Colab A100 |
| `data/online/a100_gate_lora_onl_direct_keep_replay_s2_points.json` | 23.2 KB | Online test gate, per decision and per row: E4B's distribution + keep + replay (C), seed 2, Colab A100 |
| `data/online/a100_gate_lora_onl_direct_keep_replay_s3.json` | 1.2 KB | Online test gate on 240 held decision points: E4B's distribution + keep + replay (C), seed 3, Colab A100 |
| `data/online/a100_gate_lora_onl_direct_keep_replay_s3_points.json` | 23.2 KB | Online test gate, per decision and per row: E4B's distribution + keep + replay (C), seed 3, Colab A100 |
| `data/online/a100_gate_lora_onl_gated_keep_replay.json` | 1.2 KB | Online test gate on 240 held decision points: disagreement-gated teacher + keep + replay, seed 1, Colab A100 |
| `data/online/a100_gate_lora_onl_gated_keep_replay_points.json` | 23.2 KB | Online test gate, per decision and per row: disagreement-gated teacher + keep + replay, seed 1, Colab A100 |
| `data/online/a100_gate_lora_onl_gated_keep_replay_s2.json` | 1.2 KB | Online test gate on 240 held decision points: disagreement-gated teacher + keep + replay, seed 2, Colab A100 |
| `data/online/a100_gate_lora_onl_gated_keep_replay_s2_points.json` | 23.2 KB | Online test gate, per decision and per row: disagreement-gated teacher + keep + replay, seed 2, Colab A100 |
| `data/online/a100_gate_lora_onl_sft_keep_replay.json` | 1.2 KB | Online test gate on 240 held decision points: online SFT on E4B's logged replies + keep + replay, seed 1, Colab A100 |
| `data/online/a100_gate_lora_onl_sft_keep_replay_points.json` | 23.2 KB | Online test gate, per decision and per row: online SFT on E4B's logged replies + keep + replay, seed 1, Colab A100 |
| `data/online/a100_gate_lora_sft_ce.json` | 1.1 KB | Online test gate on 240 held decision points: offline SFT on the stream's items, seed 1, Colab A100 |
| `data/online/a100_gate_lora_sft_ce_points.json` | 23.2 KB | Online test gate, per decision and per row: offline SFT on the stream's items, seed 1, Colab A100 |
| `data/online/a100_online_bridge_keep_replay.json` | 4.6 KB | Online test run: prequential windows and held curve: cross-model corrector + keep + replay (D), seed 1, Colab A100 |
| `data/online/a100_online_bridge_keep_replay_s3.json` | 4.6 KB | Online test run: prequential windows and held curve: cross-model corrector + keep + replay (D), seed 3, Colab A100 |
| `data/online/a100_online_direct_keep_replay.json` | 4.6 KB | Online test run: prequential windows and held curve: E4B's distribution + keep + replay (C), seed 1, Colab A100 |
| `data/online/a100_online_direct_keep_replay_s2.json` | 4.6 KB | Online test run: prequential windows and held curve: E4B's distribution + keep + replay (C), seed 2, Colab A100 |
| `data/online/a100_online_direct_keep_replay_s3.json` | 4.6 KB | Online test run: prequential windows and held curve: E4B's distribution + keep + replay (C), seed 3, Colab A100 |
| `data/online/a100_online_gated_keep_replay.json` | 4.7 KB | Online test run: prequential windows and held curve: disagreement-gated teacher + keep + replay, seed 1, Colab A100 |
| `data/online/a100_online_gated_keep_replay_s2.json` | 4.7 KB | Online test run: prequential windows and held curve: disagreement-gated teacher + keep + replay, seed 2, Colab A100 |
| `data/online/a100_online_sft_keep_replay.json` | 4.6 KB | Online test run: prequential windows and held curve: online SFT on E4B's logged replies + keep + replay, seed 1, Colab A100 |
| `data/online/g4_4x_gate_lora_onl_bridge_keep_replay.json` | 1.2 KB | Online test gate on 240 held decision points: cross-model corrector + keep + replay (D), corrector fitted 4x longer, seed 1, Colab G4 |
| `data/online/g4_4x_gate_lora_onl_bridge_keep_replay_points.json` | 23.2 KB | Online test gate, per decision and per row: cross-model corrector + keep + replay (D), corrector fitted 4x longer, seed 1, Colab G4 |
| `data/online/g4_4x_gate_lora_onl_bridge_keep_replay_s2.json` | 1.2 KB | Online test gate on 240 held decision points: cross-model corrector + keep + replay (D), corrector fitted 4x longer, seed 2, Colab G4 |
| `data/online/g4_4x_gate_lora_onl_bridge_keep_replay_s2_points.json` | 23.3 KB | Online test gate, per decision and per row: cross-model corrector + keep + replay (D), corrector fitted 4x longer, seed 2, Colab G4 |
| `data/online/g4_4x_gate_lora_onl_bridge_keep_replay_s3.json` | 1.2 KB | Online test gate on 240 held decision points: cross-model corrector + keep + replay (D), corrector fitted 4x longer, seed 3, Colab G4 |
| `data/online/g4_4x_gate_lora_onl_bridge_keep_replay_s3_points.json` | 23.3 KB | Online test gate, per decision and per row: cross-model corrector + keep + replay (D), corrector fitted 4x longer, seed 3, Colab G4 |
| `data/online/g4_4x_gate_lora_onl_control_keep_replay.json` | 1.2 KB | Online test gate on 240 held decision points: self-corrector + keep + replay (S), corrector fitted 4x longer, seed 1, Colab G4 |
| `data/online/g4_4x_gate_lora_onl_control_keep_replay_points.json` | 23.2 KB | Online test gate, per decision and per row: self-corrector + keep + replay (S), corrector fitted 4x longer, seed 1, Colab G4 |
| `data/online/g4_4x_gate_lora_onl_control_keep_replay_s2.json` | 1.2 KB | Online test gate on 240 held decision points: self-corrector + keep + replay (S), corrector fitted 4x longer, seed 2, Colab G4 |
| `data/online/g4_4x_gate_lora_onl_control_keep_replay_s2_points.json` | 23.3 KB | Online test gate, per decision and per row: self-corrector + keep + replay (S), corrector fitted 4x longer, seed 2, Colab G4 |
| `data/online/g4_4x_gate_lora_onl_control_keep_replay_s3.json` | 1.2 KB | Online test gate on 240 held decision points: self-corrector + keep + replay (S), corrector fitted 4x longer, seed 3, Colab G4 |
| `data/online/g4_4x_gate_lora_onl_control_keep_replay_s3_points.json` | 23.2 KB | Online test gate, per decision and per row: self-corrector + keep + replay (S), corrector fitted 4x longer, seed 3, Colab G4 |
| `data/online/g4_4x_online_bridge_keep_replay.json` | 4.6 KB | Online test run: prequential windows and held curve: cross-model corrector + keep + replay (D), corrector fitted 4x longer, seed 1, Colab G4 |
| `data/online/g4_4x_online_bridge_keep_replay_s2.json` | 4.6 KB | Online test run: prequential windows and held curve: cross-model corrector + keep + replay (D), corrector fitted 4x longer, seed 2, Colab G4 |
| `data/online/g4_4x_online_bridge_keep_replay_s3.json` | 4.6 KB | Online test run: prequential windows and held curve: cross-model corrector + keep + replay (D), corrector fitted 4x longer, seed 3, Colab G4 |
| `data/online/g4_4x_online_control_keep_replay.json` | 4.6 KB | Online test run: prequential windows and held curve: self-corrector + keep + replay (S), corrector fitted 4x longer, seed 1, Colab G4 |
| `data/online/g4_4x_online_control_keep_replay_s2.json` | 4.6 KB | Online test run: prequential windows and held curve: self-corrector + keep + replay (S), corrector fitted 4x longer, seed 2, Colab G4 |
| `data/online/g4_4x_online_control_keep_replay_s3.json` | 4.6 KB | Online test run: prequential windows and held curve: self-corrector + keep + replay (S), corrector fitted 4x longer, seed 3, Colab G4 |
| `data/online/g4_gate_lora_onl_bridge_keep_replay.json` | 1.2 KB | Online test gate on 240 held decision points: cross-model corrector + keep + replay (D), seed 1, Colab G4 |
| `data/online/g4_gate_lora_onl_bridge_keep_replay_points.json` | 23.2 KB | Online test gate, per decision and per row: cross-model corrector + keep + replay (D), seed 1, Colab G4 |
| `data/online/g4_gate_lora_onl_bridge_keep_replay_s2.json` | 1.2 KB | Online test gate on 240 held decision points: cross-model corrector + keep + replay (D), seed 2, Colab G4 |
| `data/online/g4_gate_lora_onl_bridge_keep_replay_s2_points.json` | 23.2 KB | Online test gate, per decision and per row: cross-model corrector + keep + replay (D), seed 2, Colab G4 |
| `data/online/g4_gate_lora_onl_bridge_keep_replay_s3.json` | 1.2 KB | Online test gate on 240 held decision points: cross-model corrector + keep + replay (D), seed 3, Colab G4 |
| `data/online/g4_gate_lora_onl_bridge_keep_replay_s3_points.json` | 23.3 KB | Online test gate, per decision and per row: cross-model corrector + keep + replay (D), seed 3, Colab G4 |
| `data/online/g4_gate_lora_onl_control_keep_replay.json` | 1.2 KB | Online test gate on 240 held decision points: self-corrector + keep + replay (S), seed 1, Colab G4 |
| `data/online/g4_gate_lora_onl_control_keep_replay_points.json` | 23.2 KB | Online test gate, per decision and per row: self-corrector + keep + replay (S), seed 1, Colab G4 |
| `data/online/g4_gate_lora_onl_control_keep_replay_s2.json` | 1.2 KB | Online test gate on 240 held decision points: self-corrector + keep + replay (S), seed 2, Colab G4 |
| `data/online/g4_gate_lora_onl_control_keep_replay_s2_points.json` | 23.3 KB | Online test gate, per decision and per row: self-corrector + keep + replay (S), seed 2, Colab G4 |
| `data/online/g4_gate_lora_onl_control_keep_replay_s3.json` | 1.2 KB | Online test gate on 240 held decision points: self-corrector + keep + replay (S), seed 3, Colab G4 |
| `data/online/g4_gate_lora_onl_control_keep_replay_s3_points.json` | 23.2 KB | Online test gate, per decision and per row: self-corrector + keep + replay (S), seed 3, Colab G4 |
| `data/online/g4_gate_lora_onl_direct_keep_replay.json` | 1.2 KB | Online test gate on 240 held decision points: E4B's distribution + keep + replay (C), seed 1, Colab G4 |
| `data/online/g4_gate_lora_onl_direct_keep_replay_points.json` | 23.3 KB | Online test gate, per decision and per row: E4B's distribution + keep + replay (C), seed 1, Colab G4 |
| `data/online/g4_gate_lora_onl_direct_keep_replay_s2.json` | 1.2 KB | Online test gate on 240 held decision points: E4B's distribution + keep + replay (C), seed 2, Colab G4 |
| `data/online/g4_gate_lora_onl_direct_keep_replay_s2_points.json` | 23.2 KB | Online test gate, per decision and per row: E4B's distribution + keep + replay (C), seed 2, Colab G4 |
| `data/online/g4_gate_lora_onl_direct_keep_replay_s3.json` | 1.2 KB | Online test gate on 240 held decision points: E4B's distribution + keep + replay (C), seed 3, Colab G4 |
| `data/online/g4_gate_lora_onl_direct_keep_replay_s3_points.json` | 23.3 KB | Online test gate, per decision and per row: E4B's distribution + keep + replay (C), seed 3, Colab G4 |
| `data/online/g4_gate_lora_onl_gated_keep_replay_s3.json` | 1.2 KB | Online test gate on 240 held decision points: disagreement-gated teacher + keep + replay, seed 3, Colab G4 |
| `data/online/g4_gate_lora_onl_gated_keep_replay_s3_points.json` | 23.3 KB | Online test gate, per decision and per row: disagreement-gated teacher + keep + replay, seed 3, Colab G4 |
| `data/online/g4_imit1_gate_lora_onl_bridge_keep_replay.json` | 1.2 KB | Online test gate on 240 held decision points: cross-model corrector + keep + replay (D), imitable corrector (beta 1), seed 1, Colab G4 |
| `data/online/g4_imit1_gate_lora_onl_bridge_keep_replay_points.json` | 23.2 KB | Online test gate, per decision and per row: cross-model corrector + keep + replay (D), imitable corrector (beta 1), seed 1, Colab G4 |
| `data/online/g4_imit1_gate_lora_onl_bridge_keep_replay_s2.json` | 1.2 KB | Online test gate on 240 held decision points: cross-model corrector + keep + replay (D), imitable corrector (beta 1), seed 2, Colab G4 |
| `data/online/g4_imit1_gate_lora_onl_bridge_keep_replay_s2_points.json` | 23.2 KB | Online test gate, per decision and per row: cross-model corrector + keep + replay (D), imitable corrector (beta 1), seed 2, Colab G4 |
| `data/online/g4_imit1_gate_lora_onl_bridge_keep_replay_s3.json` | 1.2 KB | Online test gate on 240 held decision points: cross-model corrector + keep + replay (D), imitable corrector (beta 1), seed 3, Colab G4 |
| `data/online/g4_imit1_gate_lora_onl_bridge_keep_replay_s3_points.json` | 23.3 KB | Online test gate, per decision and per row: cross-model corrector + keep + replay (D), imitable corrector (beta 1), seed 3, Colab G4 |
| `data/online/g4_imit1_online_bridge_keep_replay.json` | 4.6 KB | Online test run: prequential windows and held curve: cross-model corrector + keep + replay (D), imitable corrector (beta 1), seed 1, Colab G4 |
| `data/online/g4_imit1_online_bridge_keep_replay_s2.json` | 4.6 KB | Online test run: prequential windows and held curve: cross-model corrector + keep + replay (D), imitable corrector (beta 1), seed 2, Colab G4 |
| `data/online/g4_imit1_online_bridge_keep_replay_s3.json` | 4.6 KB | Online test run: prequential windows and held curve: cross-model corrector + keep + replay (D), imitable corrector (beta 1), seed 3, Colab G4 |
| `data/online/g4_online_bridge_keep_replay.json` | 4.6 KB | Online test run: prequential windows and held curve: cross-model corrector + keep + replay (D), seed 1, Colab G4 |
| `data/online/g4_online_bridge_keep_replay_s2.json` | 4.6 KB | Online test run: prequential windows and held curve: cross-model corrector + keep + replay (D), seed 2, Colab G4 |
| `data/online/g4_online_bridge_keep_replay_s3.json` | 4.6 KB | Online test run: prequential windows and held curve: cross-model corrector + keep + replay (D), seed 3, Colab G4 |
| `data/online/g4_online_control_keep_replay.json` | 4.6 KB | Online test run: prequential windows and held curve: self-corrector + keep + replay (S), seed 1, Colab G4 |
| `data/online/g4_online_control_keep_replay_s2.json` | 4.6 KB | Online test run: prequential windows and held curve: self-corrector + keep + replay (S), seed 2, Colab G4 |
| `data/online/g4_online_control_keep_replay_s3.json` | 4.6 KB | Online test run: prequential windows and held curve: self-corrector + keep + replay (S), seed 3, Colab G4 |
| `data/online/g4_online_direct_keep_replay.json` | 4.6 KB | Online test run: prequential windows and held curve: E4B's distribution + keep + replay (C), seed 1, Colab G4 |
| `data/online/g4_online_direct_keep_replay_s2.json` | 4.6 KB | Online test run: prequential windows and held curve: E4B's distribution + keep + replay (C), seed 2, Colab G4 |
| `data/online/g4_online_direct_keep_replay_s3.json` | 4.6 KB | Online test run: prequential windows and held curve: E4B's distribution + keep + replay (C), seed 3, Colab G4 |
| `data/online/g4_online_gated_keep_replay_s3.json` | 4.7 KB | Online test run: prequential windows and held curve: disagreement-gated teacher + keep + replay, seed 3, Colab G4 |
| `data/online/gate_lora_onl_blend_keep_replay.json` | 1.2 KB | Online test gate on 240 held decision points, the pull-matched blend control (189) |
| `data/online/gate_lora_onl_blend_keep_replay_points.json` | 23.2 KB | Per-decision gate output, the pull-matched blend control (189) |
| `data/online/gate_lora_onl_bridge.json` | 1.1 KB | Online test gate on 240 held decision points, setup 'corrector as teacher' |
| `data/online/gate_lora_onl_bridge_keep_replay.json` | 1.1 KB | Online test gate on 240 held decision points: cross-model corrector + keep + replay (D), seed 1, RTX 4090 |
| `data/online/gate_lora_onl_bridge_keep_replay_points.json` | 23.2 KB | Online test gate, per decision and per row: cross-model corrector + keep + replay (D), seed 1, RTX 4090 |
| `data/online/gate_lora_onl_bridge_keep_replay_s2.json` | 1.2 KB | Online test gate on 240 held decision points, setup D, seed 2 (189) |
| `data/online/gate_lora_onl_bridge_keep_replay_s2_points.json` | 23.2 KB | Per-decision gate output, setup D, seed 2 (189): match, probability, whether E2B alone differs |
| `data/online/gate_lora_onl_bridge_points.json` | 23.2 KB | Online test gate, per decision and per row: cross-model corrector, no keep + replay (B), seed 1, RTX 4090 |
| `data/online/gate_lora_onl_control_keep_replay.json` | 1.2 KB | Online test gate on 240 held decision points, the self-corrector control (189) |
| `data/online/gate_lora_onl_control_keep_replay_points.json` | 23.2 KB | Per-decision gate output, the self-corrector control (189) |
| `data/online/gate_lora_onl_direct.json` | 1.1 KB | Online test gate on 240 held decision points, setup 'teacher distribution' |
| `data/online/gate_lora_onl_direct_keep_replay.json` | 1.1 KB | Online test gate on 240 held decision points, setup 'teacher + keep + replay' |
| `data/online/gate_lora_onl_direct_keep_replay_points.json` | 23.2 KB | Online test gate, per decision and per row: E4B's distribution + keep + replay (C), seed 1, RTX 4090 |
| `data/online/gate_lora_onl_direct_points.json` | 23.2 KB | Online test gate, per decision and per row: E4B's distribution, no keep + replay (A), seed 1, RTX 4090 |
| `data/online/gate_rows_groups.json` | 1.9 KB | The gate's 240 held rows in order: episode and repository as integer indices, for the clustered bootstraps (206c) |
| `data/online/online_blend_keep_replay.json` | 5.0 KB | Online test, the teacher blended with the original E2B to the corrector's pull (w 0.433) + keep + replay, seed 1 (189 control): calibration, prequential windows and held curve |
| `data/online/online_bridge.json` | 4.5 KB | Online test, setup 'corrector as teacher': prequential windows and held curve |
| `data/online/online_bridge_keep_replay.json` | 4.5 KB | Online test run: prequential windows and held curve: cross-model corrector + keep + replay (D), seed 1, RTX 4090 |
| `data/online/online_bridge_keep_replay_s2.json` | 4.6 KB | Online test, setup D (corrector + keep + replay), seed 2 (189): prequential windows and held curve |
| `data/online/online_control_keep_replay.json` | 4.6 KB | Online test, the self-corrector teacher (reads E2B's own states) + keep + replay, seed 1 (189 control): prequential windows and held curve |
| `data/online/online_direct.json` | 4.5 KB | Online test, setup 'teacher distribution': prequential windows and held curve |
| `data/online/online_direct_keep_replay.json` | 4.5 KB | Online test, setup 'teacher + keep + replay': prequential windows and held curve |
| `data/results.json` | 205.2 KB | Every number the paper uses (output of scripts/collect.py) |
| `data/rows/xqlB/xl_rows.jsonl` | 480.5 KB | Training rows: E4B's verified solutions, 299 problems (no problem statements) |
| `data/rows/xqlE2B1/xl_rows_rb.jsonl` | 460.1 KB | Training rows: the student's runbook-guided solutions, 280 problems (no problem statements) |
| `data/rows/xqlE2B2/xl_rows_rb.jsonl` | 460.1 KB | Training rows: the student's runbook-guided solutions, 280 problems (no problem statements) |
| `data/rows/xqlE2B3/xl_rows_rb.jsonl` | 460.1 KB | Training rows: the student's runbook-guided solutions, 280 problems (no problem statements) |
| `data/rows/xqlRB0/xl_rows_rb.jsonl` | 136.8 KB | Training rows: the student's runbook-guided solutions, 228 problems (no problem statements) |
| `data/rows/xqlShort/xl_rows_short.jsonl` | 410.3 KB | Training rows: E4B's shortest verified solutions (no problem statements) |
| `data/rows/xqlTW4b1/xl_rows_from_e2b.jsonl` | 204.8 KB | Training rows for E4B: E2B's own passing answers, 148 problems (no problem statements) |
| `data/runs/xqd1/xdb_eval_s1.json` | 18.6 KB | Corrector forms vs dose-matched SFT (169d, seed 1): held agent CE, retention, KL on untaught text |
| `data/runs/xqd2/xdb_eval_s2.json` | 14.5 KB | Corrector forms vs dose-matched SFT (169d, seed 2): held agent CE, retention, KL on untaught text |
| `data/runs/xqd3/xdb_eval_s3.json` | 14.5 KB | Corrector forms vs dose-matched SFT (169d, seed 3): held agent CE, retention, KL on untaught text |
| `data/runs/xqlB/xl_base.json` | 156.5 KB | Build baselines (174 build): per-problem pass for E4B and Qwen, row and held ids, E4B's held solutions |
| `data/runs/xqlB/xl_precheck.json` | 443 B | Gate before the arms (174 build): corrector vs control on E4B's solutions |
| `data/runs/xqlCF1/xl_cf_gate.json` | 783 B | Refit-corrector gate (181, seed 1): CE of E4B's solutions, alone / old / refit |
| `data/runs/xqlCF1/xl_eval_cf_s1.json` | 120.5 KB | Evaluation (181, seed 1): fixed / new / kept counts, per-problem pass, held CE, off-task CE, paired tests |
| `data/runs/xqlCF1/xl_rw.json` | 646 B | Rewrite-row build summary (181, seed 1) |
| `data/runs/xqlCF2/xl_cf_gate.json` | 782 B | Refit-corrector gate (181, seed 2): CE of E4B's solutions, alone / old / refit |
| `data/runs/xqlCF2/xl_eval_cf_s2.json` | 120.5 KB | Evaluation (181, seed 2): fixed / new / kept counts, per-problem pass, held CE, off-task CE, paired tests |
| `data/runs/xqlCF2/xl_rw.json` | 645 B | Rewrite-row build summary (181, seed 2) |
| `data/runs/xqlCF3/xl_cf_gate.json` | 781 B | Refit-corrector gate (181, seed 3): CE of E4B's solutions, alone / old / refit |
| `data/runs/xqlCF3/xl_eval_cf_s3.json` | 120.5 KB | Evaluation (181, seed 3): fixed / new / kept counts, per-problem pass, held CE, off-task CE, paired tests |
| `data/runs/xqlCF3/xl_rw.json` | 646 B | Rewrite-row build summary (181, seed 3) |
| `data/runs/xqlE2B1/drift_e2b_s1.json` | 45.2 KB | Drift from the original student (186, Gemma 4 E2B student, seed 1): KL, CE change and entropy on untaught prose, code and stories, per arm and row |
| `data/runs/xqlE2B1/xl_eval_e2b_s1.json` | 68.0 KB | Evaluation (186, Gemma 4 E2B student, seed 1): fixed / new / kept counts, per-problem pass, held CE, off-task CE, paired tests |
| `data/runs/xqlE2B1/xl_rb.json` | 1.1 KB | Runbook-row build summary (186, Gemma 4 E2B student, seed 1) |
| `data/runs/xqlE2B2/drift_e2b_s2.json` | 45.2 KB | Drift from the original student (186, Gemma 4 E2B student, seed 2): KL, CE change and entropy on untaught prose, code and stories, per arm and row |
| `data/runs/xqlE2B2/xl_eval_e2b_s2.json` | 68.0 KB | Evaluation (186, Gemma 4 E2B student, seed 2): fixed / new / kept counts, per-problem pass, held CE, off-task CE, paired tests |
| `data/runs/xqlE2B2/xl_rb.json` | 1.1 KB | Runbook-row build summary (186, Gemma 4 E2B student, seed 2) |
| `data/runs/xqlE2B3/drift_e2b_s3.json` | 45.2 KB | Drift from the original student (186, Gemma 4 E2B student, seed 3): KL, CE change and entropy on untaught prose, code and stories, per arm and row |
| `data/runs/xqlE2B3/xl_eval_e2b_s3.json` | 67.9 KB | Evaluation (186, Gemma 4 E2B student, seed 3): fixed / new / kept counts, per-problem pass, held CE, off-task CE, paired tests |
| `data/runs/xqlE2B3/xl_rb.json` | 1.1 KB | Runbook-row build summary (186, Gemma 4 E2B student, seed 3) |
| `data/runs/xqlKL1/drift_e2b_s1.json` | 46.0 KB | Drift from the original student (193, KL strength sweep (E2B), seed 1): KL, CE change and entropy on untaught prose, code and stories, per arm and row |
| `data/runs/xqlKL1/xl_eval_e2b_s1.json` | 68.1 KB | Evaluation (193, KL strength sweep (E2B), seed 1): fixed / new / kept counts, per-problem pass, held CE, off-task CE, paired tests |
| `data/runs/xqlKL2/drift_e2b_s2.json` | 46.0 KB | Drift from the original student (193, KL strength sweep (E2B), seed 2): KL, CE change and entropy on untaught prose, code and stories, per arm and row |
| `data/runs/xqlKL2/xl_eval_e2b_s2.json` | 68.0 KB | Evaluation (193, KL strength sweep (E2B), seed 2): fixed / new / kept counts, per-problem pass, held CE, off-task CE, paired tests |
| `data/runs/xqlKL3/drift_e2b_s3.json` | 46.1 KB | Drift from the original student (193, KL strength sweep (E2B), seed 3): KL, CE change and entropy on untaught prose, code and stories, per arm and row |
| `data/runs/xqlKL3/xl_eval_e2b_s3.json` | 68.0 KB | Evaluation (193, KL strength sweep (E2B), seed 3): fixed / new / kept counts, per-problem pass, held CE, off-task CE, paired tests |
| `data/runs/xqlML1/drift_e2b_s1.json` | 45.6 KB | Drift from the original student (188, matched-loss test (E2B), seed 1): KL, CE change and entropy on untaught prose, code and stories, per arm and row |
| `data/runs/xqlML1/xl_eval_e2b_s1.json` | 67.7 KB | Evaluation (188, matched-loss test (E2B), seed 1): fixed / new / kept counts, per-problem pass, held CE, off-task CE, paired tests |
| `data/runs/xqlML2/drift_e2b_s2.json` | 45.5 KB | Drift from the original student (188, matched-loss test (E2B), seed 2): KL, CE change and entropy on untaught prose, code and stories, per arm and row |
| `data/runs/xqlML2/xl_eval_e2b_s2.json` | 67.7 KB | Evaluation (188, matched-loss test (E2B), seed 2): fixed / new / kept counts, per-problem pass, held CE, off-task CE, paired tests |
| `data/runs/xqlML3/drift_e2b_s3.json` | 45.5 KB | Drift from the original student (188, matched-loss test (E2B), seed 3): KL, CE change and entropy on untaught prose, code and stories, per arm and row |
| `data/runs/xqlML3/xl_eval_e2b_s3.json` | 67.7 KB | Evaluation (188, matched-loss test (E2B), seed 3): fixed / new / kept counts, per-problem pass, held CE, off-task CE, paired tests |
| `data/runs/xqlMT1/drift_e2b_s1.json` | 45.4 KB | Drift from the original student (190, multi-task control (E2B), seed 1): KL, CE change and entropy on untaught prose, code and stories, per arm and row |
| `data/runs/xqlMT1/xl_eval_e2b_s1.json` | 68.4 KB | Evaluation (190, multi-task control (E2B), seed 1): fixed / new / kept counts, per-problem pass, held CE, off-task CE, paired tests |
| `data/runs/xqlMT2/drift_e2b_s2.json` | 45.4 KB | Drift from the original student (190, multi-task control (E2B), seed 2): KL, CE change and entropy on untaught prose, code and stories, per arm and row |
| `data/runs/xqlMT2/xl_eval_e2b_s2.json` | 68.4 KB | Evaluation (190, multi-task control (E2B), seed 2): fixed / new / kept counts, per-problem pass, held CE, off-task CE, paired tests |
| `data/runs/xqlMT3/drift_e2b_s3.json` | 45.5 KB | Drift from the original student (190, multi-task control (E2B), seed 3): KL, CE change and entropy on untaught prose, code and stories, per arm and row |
| `data/runs/xqlMT3/xl_eval_e2b_s3.json` | 68.4 KB | Evaluation (190, multi-task control (E2B), seed 3): fixed / new / kept counts, per-problem pass, held CE, off-task CE, paired tests |
| `data/runs/xqlO1/xl_eval_s1.json` | 102.5 KB | Evaluation (179, seed 1): fixed / new / kept counts, per-problem pass, held CE, off-task CE, paired tests |
| `data/runs/xqlO2/xl_eval_s2.json` | 102.5 KB | Evaluation (179, seed 2): fixed / new / kept counts, per-problem pass, held CE, off-task CE, paired tests |
| `data/runs/xqlO3/xl_eval_s3.json` | 102.6 KB | Evaluation (179, seed 3): fixed / new / kept counts, per-problem pass, held CE, off-task CE, paired tests |
| `data/runs/xqlRB0/xl_rb.json` | 1.1 KB | Runbook-row build summary (182 build) |
| `data/runs/xqlRB0/xl_rb_probe.json` | 4.9 KB | Runbook plan probe (182 build): yield on 24 rows and 6 example plans |
| `data/runs/xqlRB1/xl_eval_rb_s1.json` | 51.3 KB | Evaluation (182, seed 1): fixed / new / kept counts, per-problem pass, held CE, off-task CE, paired tests |
| `data/runs/xqlRB2/xl_eval_rb_s2.json` | 51.3 KB | Evaluation (182, seed 2): fixed / new / kept counts, per-problem pass, held CE, off-task CE, paired tests |
| `data/runs/xqlRB3/xl_eval_rb_s3.json` | 51.3 KB | Evaluation (182, seed 3): fixed / new / kept counts, per-problem pass, held CE, off-task CE, paired tests |
| `data/runs/xqlRBC1/xl_eval_rbc_s1.json` | 51.3 KB | Evaluation (184c, seed 1): fixed / new / kept counts, per-problem pass, held CE, off-task CE, paired tests |
| `data/runs/xqlRBC2/xl_eval_rbc_s2.json` | 51.3 KB | Evaluation (184c, seed 2): fixed / new / kept counts, per-problem pass, held CE, off-task CE, paired tests |
| `data/runs/xqlRBC3/xl_eval_rbc_s3.json` | 51.3 KB | Evaluation (184c, seed 3): fixed / new / kept counts, per-problem pass, held CE, off-task CE, paired tests |
| `data/runs/xqlRG1/xl_regrade_s1.json` | 3.8 KB | (194a, longer-cap re-grade, seed 1) |
| `data/runs/xqlRG2/xl_regrade_s2.json` | 3.4 KB | (194a, longer-cap re-grade, seed 2) |
| `data/runs/xqlRG3/xl_regrade_s3.json` | 3.4 KB | (194a, longer-cap re-grade, seed 3) |
| `data/runs/xqlS1/xl_eval_s1.json` | 102.6 KB | Evaluation (174, seed 1): fixed / new / kept counts, per-problem pass, held CE, off-task CE, paired tests |
| `data/runs/xqlS2/xl_eval_s2.json` | 102.7 KB | Evaluation (174, seed 2): fixed / new / kept counts, per-problem pass, held CE, off-task CE, paired tests |
| `data/runs/xqlS3/xl_eval_s3.json` | 102.7 KB | Evaluation (174, seed 3): fixed / new / kept counts, per-problem pass, held CE, off-task CE, paired tests |
| `data/runs/xqlSS1/xl_eval_sfself_s1.json` | 51.3 KB | Evaluation (185, seed 1): fixed / new / kept counts, per-problem pass, held CE, off-task CE, paired tests |
| `data/runs/xqlSS2/xl_eval_sfself_s2.json` | 51.3 KB | Evaluation (185, seed 2): fixed / new / kept counts, per-problem pass, held CE, off-task CE, paired tests |
| `data/runs/xqlSS3/xl_eval_sfself_s3.json` | 51.2 KB | Evaluation (185, seed 3): fixed / new / kept counts, per-problem pass, held CE, off-task CE, paired tests |
| `data/runs/xqlShort/xl_short.json` | 283 B | Shortest-rows build summary (179a) |
| `data/runs/xqlTW2b1/drift_two_e2b_s1.json` | 61.0 KB | Drift from the original student (187, E4B teaches E2B, seed 1): KL, CE change and entropy on untaught prose, code and stories, per arm and row |
| `data/runs/xqlTW2b1/xl_eval_two_e2b_s1.json` | 84.8 KB | Evaluation (187, E4B teaches E2B, seed 1): fixed / new / kept counts, per-problem pass, held CE, off-task CE, paired tests |
| `data/runs/xqlTW2b1/xl_translator_bridge.log` | 5.7 KB | Cross-model corrector fit log (187, E4B teaches E2B, seed 1) |
| `data/runs/xqlTW2b1/xl_translator_control.log` | 5.8 KB | Self-corrector fit log (187, E4B teaches E2B, seed 1) |
| `data/runs/xqlTW2b2/drift_two_e2b_s2.json` | 61.0 KB | Drift from the original student (187, E4B teaches E2B, seed 2): KL, CE change and entropy on untaught prose, code and stories, per arm and row |
| `data/runs/xqlTW2b2/xl_eval_two_e2b_s2.json` | 84.8 KB | Evaluation (187, E4B teaches E2B, seed 2): fixed / new / kept counts, per-problem pass, held CE, off-task CE, paired tests |
| `data/runs/xqlTW2b2/xl_translator_bridge.log` | 5.8 KB | Cross-model corrector fit log (187, E4B teaches E2B, seed 2) |
| `data/runs/xqlTW2b2/xl_translator_control.log` | 5.8 KB | Self-corrector fit log (187, E4B teaches E2B, seed 2) |
| `data/runs/xqlTW2b3/drift_two_e2b_s3.json` | 61.0 KB | Drift from the original student (187, E4B teaches E2B, seed 3): KL, CE change and entropy on untaught prose, code and stories, per arm and row |
| `data/runs/xqlTW2b3/xl_eval_two_e2b_s3.json` | 84.7 KB | Evaluation (187, E4B teaches E2B, seed 3): fixed / new / kept counts, per-problem pass, held CE, off-task CE, paired tests |
| `data/runs/xqlTW2b3/xl_translator_bridge.log` | 5.8 KB | Cross-model corrector fit log (187, E4B teaches E2B, seed 3) |
| `data/runs/xqlTW2b3/xl_translator_control.log` | 5.8 KB | Self-corrector fit log (187, E4B teaches E2B, seed 3) |
| `data/runs/xqlTW4b1/drift_two_e4b_s1.json` | 61.2 KB | Drift from the original student (187, E2B teaches E4B, seed 1): KL, CE change and entropy on untaught prose, code and stories, per arm and row |
| `data/runs/xqlTW4b1/xl_base_from_e2b.json` | 152.2 KB | Baselines for E4B trained on E2B's passing answers (187, E2B teaches E4B, seed 1) |
| `data/runs/xqlTW4b1/xl_eval_two_e4b_s1.json` | 73.2 KB | Evaluation (187, E2B teaches E4B, seed 1): fixed / new / kept counts, per-problem pass, held CE, off-task CE, paired tests |
| `data/runs/xqlTW4b1/xl_translator_bridge.log` | 5.8 KB | Cross-model corrector fit log (187, E2B teaches E4B, seed 1) |
| `data/runs/xqlTW4b1/xl_translator_control.log` | 5.8 KB | Self-corrector fit log (187, E2B teaches E4B, seed 1) |
| `data/runs/xqlTW4b2/drift_two_e4b_s2.json` | 61.2 KB | Drift from the original student (187, E2B teaches E4B, seed 2): KL, CE change and entropy on untaught prose, code and stories, per arm and row |
| `data/runs/xqlTW4b2/xl_base_from_e2b.json` | 152.2 KB | Baselines for E4B trained on E2B's passing answers (187, E2B teaches E4B, seed 2) |
| `data/runs/xqlTW4b2/xl_eval_two_e4b_s2.json` | 73.2 KB | Evaluation (187, E2B teaches E4B, seed 2): fixed / new / kept counts, per-problem pass, held CE, off-task CE, paired tests |
| `data/runs/xqlTW4b2/xl_translator_bridge.log` | 5.8 KB | Cross-model corrector fit log (187, E2B teaches E4B, seed 2) |
| `data/runs/xqlTW4b2/xl_translator_control.log` | 5.8 KB | Self-corrector fit log (187, E2B teaches E4B, seed 2) |
| `data/runs/xqlTW4b3/drift_two_e4b_s3.json` | 61.2 KB | Drift from the original student (187, E2B teaches E4B, seed 3): KL, CE change and entropy on untaught prose, code and stories, per arm and row |
| `data/runs/xqlTW4b3/xl_base_from_e2b.json` | 152.2 KB | Baselines for E4B trained on E2B's passing answers (187, E2B teaches E4B, seed 3) |
| `data/runs/xqlTW4b3/xl_eval_two_e4b_s3.json` | 73.2 KB | Evaluation (187, E2B teaches E4B, seed 3): fixed / new / kept counts, per-problem pass, held CE, off-task CE, paired tests |
| `data/runs/xqlTW4b3/xl_translator_bridge.log` | 5.8 KB | Cross-model corrector fit log (187, E2B teaches E4B, seed 3) |
| `data/runs/xqlTW4b3/xl_translator_control.log` | 5.8 KB | Self-corrector fit log (187, E2B teaches E4B, seed 3) |
| `data/runs/xqr2/xs_train_bridge_s0.03_h0.5_r1.log` | 5.7 KB | Corrector fit log (Qwen correctors, seeds 1-2 (the paper's recipe)) |
| `data/runs/xqr2/xs_train_bridge_s0.03_h0.5_r2.log` | 5.7 KB | Corrector fit log (Qwen correctors, seeds 1-2 (the paper's recipe)) |
| `data/runs/xqr2/xs_train_control_s0.03_h0.5_r1.log` | 5.8 KB | Corrector fit log (Qwen correctors, seeds 1-2 (the paper's recipe)) |
| `data/runs/xqr2/xs_train_control_s0.03_h0.5_r2.log` | 5.7 KB | Corrector fit log (Qwen correctors, seeds 1-2 (the paper's recipe)) |
| `data/runs/xqr3/xs_train_bridge_s0.03_h0.5_r3.log` | 5.7 KB | Corrector fit log (Qwen correctors, seed 3) |
| `data/runs/xqr3/xs_train_control_s0.03_h0.5_r3.log` | 5.7 KB | Corrector fit log (Qwen correctors, seed 3) |
| `data/runs/xqw1/xw_results_s1.json` | 2.1 KB | Wide coding results (178, seed 1): passes per form, by platform and difficulty, paired tests |
| `data/runs/xqw2/xw_results_s2.json` | 2.1 KB | Wide coding results (178, seed 2): passes per form, by platform and difficulty, paired tests |
| `data/runs/xqw3/xw_results_s3.json` | 2.1 KB | Wide coding results (178, seed 3): passes per form, by platform and difficulty, paired tests |
| `figures/fig1_corrector.png` | 261.8 KB | Figure 1: how the corrector connects teacher and student, and its uses (schematic) |
| `figures/fig2_wide_coding.png` | 90.4 KB | Figure D1: agent-turn LoRAs on all easy and medium LiveCodeBench problems, problems gained and lost (178) |
| `figures/fig3_teach_vs_forget.png` | 189.8 KB | Figure D4: held solved and trained fixed vs problems lost, every coding recipe |
| `figures/fig4_untaught_prose.png` | 154.2 KB | Figure D2: drift on untaught human prose per recipe and seed |
| `figures/fig5_teacher_cap.png` | 86.9 KB | Figure D3: each corrector teacher run live beside the on-policy student it trained |
| `figures/fig6_losses.png` | 169.9 KB | Extra figure (cut from the paper): how often each solved problem was lost, and answer length per recipe |
| `figures/fig7_online.png` | 236.3 KB | Figure 2: telepathy-like teaching online, E4B to E2B on one GPU type: closeness over the stream, and what each ingredient adds (206, 206c) |
| `figures/fig8_agent_turns.png` | 103.7 KB | Extra figure (cut from the paper): agent replies, E4B to Qwen3-1.7B (169d) |
| `figures/fig9_e2b.png` | 160.0 KB | Figure 3: Gemma 4 E2B as the coding student against Qwen3-1.7B, with the anchors and the fitting-text control (186, 187, 190) |
| `notebooks/kaggle_telepathy_like_teaching.ipynb` | 9.1 KB | Source of the Kaggle companion notebook: clones this repository at a pinned commit and runs the same rebuild |
| `notebooks/telepathy_like_teaching.ipynb` | 332.1 KB | Notebook: rebuilds Table 1, the cross-against-self analysis, Figure 2 and the live table from the files in data/ (no GPU) |
| `paper.md` | 61.3 KB | The paper: Copy, Correct, or Rewrite? Telepathy-Like Teaching of a Small Agent Model and What Keeps It From Forgetting |
| `scripts/collect.py` | 19.4 KB | Collects every number the paper uses from the run folders into data/results.json (standard library) |
| `scripts/drift.py` | 3.7 KB | Measures drift from the original student on untaught text (needs adapters and text rows not included) |
| `scripts/figures.py` | 33.6 KB | Draws the nine figures from data/results.json (matplotlib) |
| `scripts/g4_rows.py` | 4.9 KB | The paired row-level analysis (206c): cross-model against self-corrector students on the gate's 240 held rows, repository bootstrap, speed (standard library) |
| `scripts/g4_table.py` | 3.6 KB | The one-GPU online table (206): fixed / broken / net, held CE, prequential gain per run, paired sign tests on the 220 decisions (standard library) |
| `scripts/lcb_failkinds.py` | 4.3 KB | How the online coding answers fail (unfinished, placeholder, wrong), from data/generations |
| `scripts/lcb_table.py` | 9.1 KB | Online coding table (Table D9): held solved, fixed, lost and paired sign tests per arm and seed, from data/lcb |
| `scripts/live_table.py` | 3.4 KB | The live test's table and by-task sign tests from data/live/live80_attempts.json (standard library) |
| `scripts/to_html.py` | 9.8 KB | Renders paper.md into one self-contained HTML page |
