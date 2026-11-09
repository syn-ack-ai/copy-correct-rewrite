# Copy, Correct, or Rewrite? What Teaches a Small Model and What Keeps It From Forgetting

Code, results and pre-registrations for a paper-track writeup in the Kaggle "Gemma 4 Developer Agent" competition. The paper is
[`paper.md`](paper.md).

Author: Benjamin Babeshkin. The study was carried out with Claude Opus 5.5 (Anthropic) as an AI research assistant; the paper's Appendix B
says who did what.

The study compares ways for a small model to learn from a stronger one under one pre-registered protocol:

- supervised fine-tuning (SFT) on the teacher's verified answers;
- a pull toward a *state corrector*, a learned network that rewrites the student's hidden states (called "translator" during development;
  the code names it `bridge` when it reads the teacher's states and `control` when it reads the student's own states);
- on-policy distillation;
- retraining on solutions the student writes itself from the teacher's plan ("runbook rows").

Teacher: Gemma 4 E4B. Students: Qwen3-1.7B and Gemma 4 E2B on LiveCodeBench contest coding (three seeds each), E4B and E2B teaching
each other with freshly fitted Gemma correctors (three seeds each way), and Gemma 4 E2B on online agent decisions.

## Key findings

From the paper's abstract and results:

- **Copying the teacher's verified answers taught as much as anything tested.** The shortest correct version did best.
- **Within the Gemma family, copying also forgot.** E4B's text is familiar to E2B; training on it sharpened E2B and raised its loss on
  untaught text far more than Qwen's. The likely cause is over-fitting familiar text.
- **A KL pull toward the original student removed that damage without costing held-out solves** (prose KL 0.016; held 207 vs 204 for SFT).
  Reversed, with E2B's answers teaching E4B, SFT barely helped and every anchor did significantly better (244-250 held vs 224).
- **Self-written solutions ("runbook rows") moved the student least yet sharpened it.** A small KL is no proof against forgetting.
- **On-policy distillation reached its teacher's level and stopped there.**
- **The state corrector did not store the teacher's knowledge in an adapter:** fed E4B's states, it taught no better than fed the student's
  own. As an anchor it matched training on its own fitting text (held 214 vs 216, p = 0.90; addendum 190), delivered with less change.
  As an online teacher with keep + replay it gave the best net result on agent decisions in two seeds; a corrector
  reading the student's own states did not.

The paper's limitations (Section 5) and the pre-registration scorecard (Appendix C) apply to all of these.

**Status (2026-11-08, evening).** In the paper and in this repository: 185 (`data/runs/xqlSS*`), 186 (`xqlE2B*`), 187 (`xqlTW2b*`: E4B
teaches E2B; `xqlTW4b*`: E2B teaches E4B), 188 (`xqlML*`: the matched-loss test), 193 (`xqlKL*`: the KL strength sweep), 194a (`xqlRG*`: the longer-cap re-grade), 190 (`xqlMT*`: the multi-task control for the corrector anchor) and the online test's setup D,
seeds 1-2 with the self-corrector control (183, 189; `data/online/`; seed 2 reproduced seed 1, differing on 6 of 220 gate decisions). Running or queued at the time of
this snapshot, with designs and predictions in `PREREGISTRATION.md`: online D seed 3, the blend control and setup C seeds 2-3 (189), D's LoRA live on 80
held-out agent tasks (196), a disagreement-gated teacher (197) and online against offline SFT teachers (192).
Addenda 191 and 195 (per-user and lottery-ticket grafts) are planned for a separate paper.

## Repository layout

```
paper.md                 the paper
figures/                 the figures (PNG), made by scripts/figures.py: four in the body, three in Appendix D, two extra (MANIFEST.md)
scripts/                 collect.py, figures.py, drift.py, to_html.py (paper data and rendering)
data/
  results.json           every number the paper uses (scripts/collect.py)
  drift.json             drift from the original student on untaught text, per arm and seed (scripts/drift.py)
  drift_ss.json          the same for arm sft_self_s (185)
  online/                online test (E4B to E2B): run curves and the 240-point gate, per setup
  runs/<run>/            evaluation and build summaries, laid out as the original run folders (see "Run folders")
  rows/<run>/            training rows: id, reply, source, difficulty (problem statements removed)
  generations/<run>/     model answers per LiveCodeBench question id, gzip-compressed
code/
  transfer/              experiment code (Python)
  colab/                 session scripts for Colab G4 runs
  dreamer/               launchers and runners for the local RTX 4090
PREREGISTRATION.md       the pre-registration log, addenda 165-197
SCORECARD.md             which pre-registered predictions held, per addendum (designs and predictions written before each run)
MANIFEST.md              every file, its size and its purpose
LICENSE                  Apache License 2.0
```

`paper.md` keeps the paths of the original repository in its "Reproducibility" section. In this repository:
`bench/archetypes/transfer/` is `code/transfer/`, `kaggle_g4agent/colab/` is `code/colab/`, `kaggle_g4agent/dreamer/` is `code/dreamer/`,
`papers/kaggle_bridge/` is the repository root, and `bench/archetypes/README.md` is `PREREGISTRATION.md`.

### Code

| File | Purpose |
|---|---|
| `code/transfer/xbridge.py` | The state corrector across tokenizers (E4B to Qwen3-1.7B): alignment, fitting, the suite, graft and LoRA forms (addenda 166-176) |
| `code/transfer/xlearn.py` | The coding recipes and their evaluation: build, shortest rows, refit corrector, rewrites, runbook rows, keep rows, the E2B student, all arms (174-188) |
| `code/transfer/xcode.py` | LiveCodeBench loading, generation and the sandboxed hidden-test scorer; the wide coding test (172, 178) |
| `code/transfer/online.py` | The online test, E4B to E2B (170, 183), and earlier same-tokenizer experiments in the same file |
| `code/transfer/rsi.py` | The attempt-verify-update loop pilot (177) |
| `code/transfer/rb_pair.py` | Pairs the runbook arms with stored arms across sessions (182, 184c) |
| `code/transfer/graft_agg.py` | Per-seed summary of the dose-matched comparison (169d) |
| `code/transfer/dequant.py` | Dequantises the 4-bit QAT checkpoint to bf16 |
| `code/transfer/build_mix.py`, `mix_merge.py` | Build and merge the mixed rows (prose, code, stories, questions) the corrector was fitted on (165) |

`online.py` is the current version. It adds addendum 189's seed offset (ON_SEED), the self-corrector and blend teachers, and a per-point
gate output; with ON_SEED unset and the original teachers it trains exactly as the version that ran 170 and 183.

The code is a snapshot of 2026-11-08. It also holds paths that the paper does not report: arms `sft_k`, `sft_ck`, `xk` (addendum 184),
`sft_es`, `sft_lr` (188) and `sft_mt_s`, `sft_self_mt_s` (190) in `xlearn.py`, and the online test's control teachers (189) in `online.py`;
188, 189 and 190 were running at the time of the snapshot.

## How the experiments ran

- **Colab G4 sessions.** Each `code/colab/*.sh` script is one session. A local launcher (`colab_run.sh`, not included) started it on a
  Colab G4 GPU, and the `code/dreamer/xq_side*_launch.sh` scripts queued sessions (at most three at a time, with a compute-unit balance
  guard in the later launchers). Most sessions first run a small pass of every step, then the full run.
- **The local RTX 4090** (24 GB) ran `code/dreamer/online_4090.sh` and `online_bkr_4090.sh` (the online test), `resdec_4090.sh`,
  `rsi4090_prep.sh` and `rsi4090_run.sh` (the loop pilot), and `x184_prep_4090.sh` and `x184_4090.sh` (addendum 184).
- **Inputs.** Sessions downloaded their inputs from the Hugging Face dataset `SkyPanther/g4agent-job`. That dataset is **private**: it holds
  copies of the code, the agent and text rows, the training rows, and trained correctors and adapters, so the scripts cannot be rerun as
  they are. Public inputs were pinned:
  LiveCodeBench `code_generation_lite` at revision `0fe84c3912ea0c4d4a78037083943e8f0c4dd505`, `google/gemma-4-E4B-it-qat-w4a16-ct` at
  revision `6cd26aaa2357fb2bad8c51699a7558a4d1a965bb` and `google/gemma-4-E2B-it-qat-w4a16-ct` at revision
  `971342c08f607aa7779983f6b5289778b5d271a7` (both dequantised with `dequant.py`), and `Qwen/Qwen3-1.7B`.
- **Scoring.** Every generated program ran against all hidden tests in a fresh interpreter with CPU, memory and time limits (`xcode.py`;
  on the 4090 inside a bubblewrap sandbox).

### Run folders

`data/runs/` and `data/generations/` keep the original run-folder names:

| Folder | Experiment (addendum) |
|---|---|
| `xqlB` | The build: E4B's verified rows on 299 older problems, baselines on both pools (174) |
| `xqlS1`-`xqlS3` | SFT, SFT with corrector anchors, the anchor alone, SDFT; seeds 1-3 (174) |
| `xqlShort` | The shortest-correct rows (179a) |
| `xqlO1`-`xqlO3` | SFT, shortest rows, shortest rows + self-anchor, on-policy with the weak teacher, SFT + on-policy (179) |
| `xqlCF1`-`xqlCF3` | The refit corrector: on-policy, SFT + on-policy, rewrites (181) |
| `xqlRB0` | The runbook-row build: E4B's plans and the student's solutions (182) |
| `xqlRB1`-`xqlRB3` | Runbook rows (182) |
| `xqlRBC1`-`xqlRBC3` | Runbook rows + self-anchor (184c) |
| `xqlSS1`-`xqlSS3` | Shortest rows + KL to the original student (185) |
| `xqlTW2b1`-`xqlTW2b3` | E4B teaches E2B with freshly fitted Gemma correctors: shortest rows + KL to the original, + self-anchor, + cross-model anchor (187) |
| `xqlTW4b1`-`xqlTW4b3` | E2B teaches E4B on E2B's 148 passing answers: SFT, KL to the original, self-anchor, cross-model anchor (187) |
| `xqr2`, `xqr3` | Fit logs of Qwen's per-seed correctors (the paper's recipe), read by `collect.py` |
| `xqlKL1`-`xqlKL3` | The KL strength sweep on E2B: shortest rows + KL to the original at lambda 0.3 and 3 (193) |
| `xqlRG1`-`xqlRG3` | The longer-cap re-grade: each Qwen recipe's lost answers cut at 768 tokens, continued by 768 more (194a) |
| `xqlML1`-`xqlML3` | The matched-loss test on E2B: SFT stopped at training loss 0.24, and SFT at a tenth of the learning rate (188) |
| `xqlMT1`-`xqlMT3` | The multi-task control on E2B: shortest rows + the corrector's fitting text, with and without KL to the original (190) |
| `xqlE2B1`-`xqlE2B3` | Gemma 4 E2B as the student: SFT, shortest rows, runbook rows (E2B re-solves with the plans from `xqlRB0`), drift (186) |
| `xqw1`-`xqw3` | LoRAs trained on agent turns, tested on all easy and medium LiveCodeBench problems (178) |
| `xqd1`-`xqd3` | Corrector forms against dose-matched SFT on agent replies (169d) |

Arm keys in the data and their names in the figures (`scripts/figures.py`):

| Key | Name |
|---|---|
| `qwen` | the untrained student alone (Qwen3-1.7B; Gemma 4 E2B in `xqlE2B*`, `xqlTW2b*` and `xqlMT*`; Gemma 4 E4B in `xqlTW4b*`) |
| `sft` | SFT on teacher text |
| `sft_s` | SFT, shortest correct |
| `sft_c_s` | SFT, shortest + self-anchor |
| `sft_self_s` | SFT, shortest + KL to the original student (185) |
| `sft_self_s_l03` / `sft_self_s_l3` | SFT, shortest + KL to the original student at lambda 0.3 / 3 (193) |
| `sft_es` / `sft_lr` | SFT stopped at training loss 0.24 / at a tenth of the learning rate (188) |
| `sft_mt_s` / `sft_self_mt_s` | SFT, shortest + the corrector's fitting text / + KL to the original student (190) |
| `opd_c` / `opd_cf` | On-policy (weak teacher) / (refit teacher) |
| `mix_c` / `mix_cf` | SFT + on-policy (weak teacher) / (refit teacher) |
| `rw_cf` | Rewrites via corrector |
| `rb` / `rb_c` | Runbook rewrites / + self-anchor |
| `sft_b` / `sft_c` | SFT + cross-model anchor / + self-anchor (174) |
| `kl_b` | Cross-model anchor only (174) |
| `sdft` | SDFT (in-context self-teacher) (174) |
| `lora_sd`, `lora_c`, `lora_b`, `lora_self` | Plain SFT on teacher text; self-corrector, distilled; cross-model corrector, distilled; SFT + KL to the original student (178) |

## Rebuilding `results.json` and the figures

`scripts/collect.py` reads the run folders and writes `results.json`. It needs only the Python standard library. It expects the original
layout, with the generations uncompressed next to the evaluation files:

```bash
mkdir -p rebuild && cp -r data/runs/* rebuild/
for f in data/generations/*/xl_gens_*.jsonl.gz; do
  r=$(basename "$(dirname "$f")"); gunzip -c "$f" > "rebuild/$r/$(basename "$f" .gz)"
done
python scripts/collect.py rebuild data/results.json   # also reads data/online/ and data/drift*.json
python scripts/figures.py data/results.json figures/  # needs matplotlib and numpy
python scripts/to_html.py paper.md paper.html         # optional: one self-contained HTML page
```

`scripts/drift.py` documents how `drift.json` was measured. It needs the trained adapters and the untaught text rows, which are not
included, so it cannot be rerun from this repository.

## What is not included, and why

- **Text corpora.** The prose rows (passages from copyrighted novels), the story rows, and the agent rows (contexts and replies recorded in
  the Kaggle competition harness, whose data may not be redistributed). This covers the `s3` data, the mixed rows, `xq_questions.jsonl`
  and every `.npy` / `.npz` array.
- **Competition material.** The competition harness, its task data and its documentation.
- **Problem statements.** The prompts (LiveCodeBench problem statements) are removed: the training rows keep only `id`, `reply`,
  `source` and `difficulty`, and the generation files hold no prompts. Model answers can quote short pieces of a statement in their
  comments. The statements can be fetched from LiveCodeBench by question id at the pinned revision. The `sha` fields in
  `xl_base.json`, `xl_short.json` and `xl_rb.json` refer to the full rows files, so they do not match the rows released here.
- **Model weights.** No adapters, corrector checkpoints or base-model weights.
- **Some code.** Modules that the released code imports but that belong to earlier experiments: `translator.py`, `pair.py` and `store.py`
  (imported by `online.py`); `shape_store.py`, `shape_translator.py` and `fresh.py` (imported by `build_mix.py`). Also the Colab launcher
  `colab_run.sh` and the environment file `colab_env.sh` it writes, and `rep2_summary.py` (called by `resdec_4090.sh`).
- **Logs** and run artefacts other than the files listed in `MANIFEST.md`.

## Licence

Apache License 2.0; see [`LICENSE`](LICENSE). The licence covers the code and the documents written for this study. The model outputs in
`data/` were produced by Gemma 4 E4B, Gemma 4 E2B and Qwen3-1.7B; their use may also be subject to those models' own terms.
