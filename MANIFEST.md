# Manifest

352 files, 36.6 MB in total (MANIFEST.md not counted). Sizes are on-disk sizes; `.gz` files are gzip-compressed JSON Lines.
Run labels give the addendum and seed (see README.md, "Run folders").

| File | Size | Purpose |
|---|---|---|
| `LICENSE` | 11.1 KB | Apache License 2.0 |
| `MANIFEST.md` | (this file) | This file: every file with its size and purpose |
| `PREREGISTRATION.md` | 205.4 KB | Pre-registration log, addenda 165-197, verbatim (one phrase redacted) |
| `README.md` | 14.3 KB | Overview: the study, findings, layout, how the runs worked, what is excluded, rebuild steps, licence |
| `SCORECARD.md` | 1.9 KB | Which pre-registered predictions held, per addendum (moved out of the paper) |
| `code/colab/xq_bridge.sh` | 3.0 KB | Colab session: fit the bridge and control correctors and run the suite (166) |
| `code/colab/xq_curve.sh` | 5.2 KB | Colab session: corrector training-length curve (176) |
| `code/colab/xq_dosebase.sh` | 3.2 KB | Colab session: dose-matched SFT baselines and their evaluation (169d) |
| `code/colab/xq_gbase.sh` | 2.6 KB | Colab session: graft and LoRA forms scored together on identical rows (169) |
| `code/colab/xq_learn.sh` | 23.5 KB | Colab session for the coding recipes: build, arms, short rows, refit corrector, runbook rows, E2B student, two-way, multi-task (174-193) |
| `code/colab/xq_rsi.sh` | 7.7 KB | Colab session: the attempt-verify-update loop pilot (177) |
| `code/colab/xq_selftest.sh` | 3.9 KB | Colab session: SFT + KL to the original student on agent rows, then LiveCodeBench (174q) |
| `code/colab/xq_wide.sh` | 4.6 KB | Colab session: agent-turn LoRAs on all easy and medium LiveCodeBench problems (178) |
| `code/colab/xq_x1.sh` | 2.3 KB | Colab session: retention accounting on stored correctors and LoRAs (169, X1) |
| `code/dreamer/online_4090.sh` | 3.0 KB | 4090 runner: the online test, three setups and their gates (170) |
| `code/dreamer/online_bkr_4090.sh` | 2.5 KB | 4090 runner: online test arm D, corrector teacher + keep + replay (183) |
| `code/dreamer/online_rep2_4090.sh` | 2.6 KB | 4090 runner: online C seeds 2-3 (189b) |
| `code/dreamer/online_rep3_4090.sh` | 2.9 KB | 4090 runner: online against offline SFT teachers on agent decisions (192, running at the time of release) |
| `code/dreamer/online_rep_4090.sh` | 3.4 KB | 4090 runner: online D seeds 2-3, self-corrector and blend teachers, re-gates (189) |
| `code/dreamer/resdec_4090.sh` | 2.6 KB | 4090 runner: resdec control and replication seeds (180) |
| `code/dreamer/rsi4090_prep.sh` | 2.3 KB | 4090 runner: inputs for the loop pilot (177) |
| `code/dreamer/rsi4090_run.sh` | 3.6 KB | 4090 runner: the loop pilot arms and summary (177) |
| `code/dreamer/x184_4090.sh` | 5.4 KB | 4090 runner: keep rows, selective corrector and arms (184) |
| `code/dreamer/x184_prep_4090.sh` | 3.9 KB | 4090 runner: inputs for addendum 184 |
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
| `code/transfer/mix_merge.py` | 1.2 KB | Appends the mixed rows to the agent rows in one data folder (165) |
| `code/transfer/online.py` | 134.5 KB | Online test E4B to E2B (170, 183, 189: seeds, self-corrector and blend teachers) and earlier same-tokenizer experiments |
| `code/transfer/rb_pair.py` | 3.2 KB | Pairs runbook arms with stored arms across sessions (182, 184c) |
| `code/transfer/rsi.py` | 46.8 KB | Attempt-verify-update loop pilot (177) |
| `code/transfer/xbridge.py` | 67.1 KB | The state corrector across tokenizers (E4B to Qwen3-1.7B): alignment, fitting, suite, graft and LoRA forms |
| `code/transfer/xcode.py` | 22.1 KB | LiveCodeBench loading, generation, the sandboxed hidden-test scorer, the wide coding test |
| `code/transfer/xlearn.py` | 94.8 KB | The coding recipes (SFT, anchors, on-policy, rewrites, runbook rows) and their evaluation |
| `data/drift.json` | 460.3 KB | Drift from the original student on untaught prose, code and stories, per arm, seed and row (KL, CE change, entropy) |
| `data/drift_ss.json` | 46.4 KB | Drift from the original student for arm sft_self_s (185) |
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
| `data/online/gate_lora_onl_blend_keep_replay.json` | 1.2 KB | Online test gate on 240 held decision points, the pull-matched blend control (189) |
| `data/online/gate_lora_onl_blend_keep_replay_points.json` | 23.2 KB | Per-decision gate output, the pull-matched blend control (189) |
| `data/online/gate_lora_onl_bridge.json` | 1.1 KB | Online test gate on 240 held decision points, setup 'corrector as teacher' |
| `data/online/gate_lora_onl_bridge_keep_replay.json` | 1.1 KB | Online test gate on 240 held decision points, setup 'corrector + keep + replay (D)' |
| `data/online/gate_lora_onl_bridge_keep_replay_points.json` | 23.2 KB | Online test gate, per decision point (re-gated, identical totals), setup 'corrector + keep + replay (D)' |
| `data/online/gate_lora_onl_bridge_keep_replay_s2.json` | 1.2 KB | Online test gate on 240 held decision points, setup D, seed 2 (189) |
| `data/online/gate_lora_onl_bridge_keep_replay_s2_points.json` | 23.2 KB | Per-decision gate output, setup D, seed 2 (189): match, probability, whether E2B alone differs |
| `data/online/gate_lora_onl_bridge_points.json` | 23.2 KB | Online test gate, per decision point (re-gated, identical totals), setup 'corrector as teacher (B)' |
| `data/online/gate_lora_onl_control_keep_replay.json` | 1.2 KB | Online test gate on 240 held decision points, the self-corrector control (189) |
| `data/online/gate_lora_onl_control_keep_replay_points.json` | 23.2 KB | Per-decision gate output, the self-corrector control (189) |
| `data/online/gate_lora_onl_direct.json` | 1.1 KB | Online test gate on 240 held decision points, setup 'teacher distribution' |
| `data/online/gate_lora_onl_direct_keep_replay.json` | 1.1 KB | Online test gate on 240 held decision points, setup 'teacher + keep + replay' |
| `data/online/gate_lora_onl_direct_keep_replay_points.json` | 23.2 KB | Online test gate, per decision point (re-gated, identical totals), setup 'teacher + keep + replay (C)' |
| `data/online/gate_lora_onl_direct_points.json` | 23.2 KB | Online test gate, per decision point (re-gated, identical totals), setup 'teacher distribution (A)' |
| `data/online/online_blend_keep_replay.json` | 5.0 KB | Online test, the teacher blended with the original E2B to the corrector's pull (w 0.433) + keep + replay, seed 1 (189 control): calibration, prequential windows and held curve |
| `data/online/online_bridge.json` | 4.5 KB | Online test, setup 'corrector as teacher': prequential windows and held curve |
| `data/online/online_bridge_keep_replay.json` | 4.5 KB | Online test, setup 'corrector + keep + replay (D)': prequential windows and held curve |
| `data/online/online_bridge_keep_replay_s2.json` | 4.6 KB | Online test, setup D (corrector + keep + replay), seed 2 (189): prequential windows and held curve |
| `data/online/online_control_keep_replay.json` | 4.6 KB | Online test, the self-corrector teacher (reads E2B's own states) + keep + replay, seed 1 (189 control): prequential windows and held curve |
| `data/online/online_direct.json` | 4.5 KB | Online test, setup 'teacher distribution': prequential windows and held curve |
| `data/online/online_direct_keep_replay.json` | 4.5 KB | Online test, setup 'teacher + keep + replay': prequential windows and held curve |
| `data/results.json` | 179.2 KB | Every number the paper uses (output of scripts/collect.py) |
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
| `figures/fig1_corrector.png` | 231.4 KB | Figure 1: how the corrector connects teacher and student, and its uses (schematic) |
| `figures/fig2_wide_coding.png` | 90.4 KB | Figure D1: agent-turn LoRAs on all easy and medium LiveCodeBench problems, problems gained and lost (178) |
| `figures/fig3_teach_vs_forget.png` | 189.8 KB | Figure 3: held solved and trained fixed vs problems lost, every coding recipe |
| `figures/fig4_untaught_prose.png` | 154.2 KB | Figure D2: drift on untaught human prose per recipe and seed |
| `figures/fig5_teacher_cap.png` | 86.9 KB | Figure D3: each corrector teacher run live beside the on-policy student it trained |
| `figures/fig6_losses.png` | 169.9 KB | Extra figure (cut from the paper): how often each solved problem was lost, and answer length per recipe |
| `figures/fig7_online.png` | 264.6 KB | Figure 4: online learning, E4B to E2B, the setups and the self-corrector control, held cross-entropy and drift |
| `figures/fig8_agent_turns.png` | 103.7 KB | Extra figure (cut from the paper): agent replies, E4B to Qwen3-1.7B (169d) |
| `figures/fig9_e2b.png` | 160.0 KB | Figure 2: Gemma 4 E2B as the coding student against Qwen3-1.7B, with the anchors and the fitting-text control (186, 187, 190) |
| `paper.md` | 31.3 KB | The paper: Copy, Correct, or Rewrite? |
| `scripts/collect.py` | 14.0 KB | Collects every number the paper uses from the run folders into data/results.json (standard library) |
| `scripts/drift.py` | 3.7 KB | Measures drift from the original student on untaught text (needs adapters and text rows not included) |
| `scripts/figures.py` | 31.3 KB | Draws the nine figures from data/results.json (matplotlib) |
| `scripts/to_html.py` | 9.8 KB | Renders paper.md into one self-contained HTML page |
