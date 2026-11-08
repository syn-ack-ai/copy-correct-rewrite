# Copy, Correct, or Rewrite? What Teaches a Small Model and What Keeps It From Forgetting

Code, results and pre-registrations for a paper-track writeup in the Kaggle "Gemma 4 Developer Agent" competition. The paper is
[`paper.md`](paper.md).

The study compares ways for a small model to learn from a stronger one under one pre-registered protocol:

- supervised fine-tuning (SFT) on the teacher's verified answers;
- a pull toward a *state corrector*, a learned network that rewrites the student's hidden states (called "translator" during development;
  the code names it `bridge` when it reads the teacher's states and `control` when it reads the student's own states);
- on-policy distillation;
- retraining on solutions the student writes itself from the teacher's plan ("runbook rows").

Teacher: Gemma 4 E4B. Students: Qwen3-1.7B on LiveCodeBench contest coding (three seeds) and Gemma 4 E2B on online agent decisions (one
seed).

## Key findings

As stated in the paper's abstract:

- **Copying the teacher's verified answers taught as much as anything tested,** and more than any route through the corrector.
- **The corrector did not teach more than SFT.** The corrector is a learned network that rewrites the student's hidden states. Used as an
  anchor, it moved the student about as far as SFT, but toward lower loss on untaught text, and it reversed real damage. It did not protect
  solved coding problems better than simply training on the shortest correct answers.
- **On-policy distillation reached its teacher's level and stopped there.**
- **Student-written solutions fixed the most trained problems, and degraded untaught text in every seed.**

And in the results sections:

- Section 3.2: "The anchor does **not** keep the student closer to its original: on prose its KL is as large as SFT's (0.089 vs 0.074). It
  moves the student toward much **lower** loss on untaught text".
- Section 3.6 (online learning, E4B to E2B): "Keep + replay cut drift on untaught text 6.6×." The same section notes: "One seed per setup,
  and two of the five predictions failed."

The paper's limitations (Section 6) and the pre-registration scorecard apply to all of these.

**Status (2026-11-08).** Two runs were still in progress when this repository was first published: the SFT + KL-to-the-original-student
baseline on the coding task (addendum 185), and the coding recipes repeated with Gemma 4 E2B as the student (addendum 186, three seeds).
Their results and the paper's final version will be added here.

## Repository layout

```
paper.md                 the paper
figures/                 the eight figures (PNG), made by scripts/figures.py
scripts/                 collect.py, figures.py, drift.py, to_html.py (paper data and rendering)
data/
  results.json           every number the paper uses (scripts/collect.py)
  drift.json             drift from the original student on untaught text, per arm and seed (scripts/drift.py)
  drift_rbc.json         the same for arm rb_c
  online/                online test (E4B to E2B): run curves and the 240-point gate, per setup
  runs/<run>/            evaluation and build summaries, laid out as the original run folders (see "Run folders")
  rows/<run>/            training rows: id, reply, source, difficulty (problem statements removed)
  generations/<run>/     model answers per LiveCodeBench question id, gzip-compressed
code/
  transfer/              experiment code (Python)
  colab/                 session scripts for Colab G4 runs
  dreamer/               launchers and runners for the local RTX 4090
PREREGISTRATION.md       the pre-registration log, addenda 165-185 (designs and predictions written before each run)
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
| `code/transfer/xlearn.py` | The coding recipes and their evaluation: build, shortest rows, refit corrector, rewrites, runbook rows, keep rows, all arms (174-185) |
| `code/transfer/xcode.py` | LiveCodeBench loading, generation and the sandboxed hidden-test scorer; the wide coding test (172, 178) |
| `code/transfer/online.py` | The online test, E4B to E2B (170, 183), and earlier same-tokenizer experiments in the same file |
| `code/transfer/rsi.py` | The attempt-verify-update loop pilot (177) |
| `code/transfer/rb_pair.py` | Pairs the runbook arms with stored arms across sessions (182, 184c) |
| `code/transfer/graft_agg.py` | Per-seed summary of the dose-matched comparison (169d) |
| `code/transfer/dequant.py` | Dequantises the 4-bit QAT checkpoint to bf16 |
| `code/transfer/build_mix.py`, `mix_merge.py` | Build and merge the mixed rows (prose, code, stories, questions) the corrector was fitted on (165) |

`online.py` is the version that ran the online test on the 4090. The copy in the original repository later gained a replication-seed
offset and a per-point gate output (a diff of 4 added and 7 removed lines).

The code is a snapshot of 2026-11-08. It also holds paths that the paper does not report: arms `sft_k`, `sft_ck`, `xk` (addendum 184) and
`sft_self_s` (185) in `xlearn.py`, and `STAGE=e2b` in `xq_learn.sh` (addendum 186, Gemma 4 E2B as the coding student, not run at the time
of the snapshot).

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
  revision `6cd26aaa2357fb2bad8c51699a7558a4d1a965bb` (dequantised with `dequant.py`), and `Qwen/Qwen3-1.7B`.
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
| `xqw1`-`xqw3` | LoRAs trained on agent turns, tested on all easy and medium LiveCodeBench problems (178) |
| `xqd1`-`xqd3` | Corrector forms against dose-matched SFT on agent replies (169d) |

Arm keys in the data and their names in the figures (`scripts/figures.py`):

| Key | Name |
|---|---|
| `qwen` | the untrained student (Qwen3-1.7B alone) |
| `sft` | SFT on teacher text |
| `sft_s` | SFT, shortest correct |
| `sft_c_s` | SFT, shortest + self-anchor |
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
