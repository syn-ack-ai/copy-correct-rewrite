# Pre-registration log: addenda 165-208

This file is an excerpt of the project's running pre-registration log. It holds the addenda from "Addendum 165" through the end of
"Addendum 208", copied verbatim and in their original order (addendum 180 sits between 177 and 178 in the source). Each addendum states its
design and its predictions before the run; results and readings were appended after the run. Addenda without a result were still
running when this excerpt was taken.

Redactions: one phrase in addendum 165 is replaced by "[redacted]". Nothing else is changed.

Paths in the log refer to the original repository layout. In this repository: `transfer/` and `bench/archetypes/transfer/` are
`code/transfer/`; `kaggle_g4agent/colab/` is `code/colab/`; `kaggle_g4agent/dreamer/` is `code/dreamer/`; `papers/kaggle_bridge/` is the
repository root. Files the log names that are not in this repository (for example `translator.py`, `store.py`, `shape_store.py`, the
`reports/` notes, other launchers) were not released.

---

## Addendum 165 (2026-11-07, 05:50): a generic, robust translator, then the arms again with it (user: "do all 5 fixes first then restart tye arms on the 4090 with that new translator after we do a smaller smoke test"; the five fixes proposed at 05:30)

Why the translators are narrow (R161, R165, 163 basis): trained for one task they lose others (trivia → chat questions 0.425 vs 0.742 alone; question-trained → story geometry 0.691 vs 0.764); they replace E2B's layer-12 state, so the control (E2B's own states) changes E2B's state as much as the bridge (31% vs 29% of its norm) and half the bridge's main directions are shared with it; the late depths pass the answer through; ~1,800 agent points along E4B's replies, 300-600 steps.
1. Design (translator.py arch gall): from E4B's source states AND E2B's own layer-12 state, a gated change at layer 12 and gated additions at layers 13-34; no replacement, no ridge term in the forward; heads start at zero, so the start is E2B unchanged. Loss: cross-entropy to E4B's top 32 + 0.1 × the change's size (whitened, write layer and additions) + 0.5 × do-no-harm (where E4B and E2B alone pick the same token, cross-entropy to E2B alone's top 32). Training uses E2B's own layer-12 state from the pass without the bridge (exact: nothing before layer 12 is changed); every other use computes it inside the forward.
2. Data (transfer/build_mix.py, mix_merge.py; local only, [redacted]): draws per sequence agent decision points 40%, E2B's own sample at an agent point (generated during training) 20%, prose 15% (300 passages), Python standard-library code 10% (395 chunks), pattern questions on the training items 10% (840, every 5th item held), stories 5% (224). Held out: every 7th prose/story/code text, the held question items, and the asked test's 200 fresh items (never in any translator's training).
3. Mid layers only: E4B 50/65% (layers 20, 27); control E2B 17, 22.
4. Regularisation: Gaussian noise 0.1 on the whitened source, each source depth dropped with probability 0.15 per sequence, 600 steps (twice the ladder's).
5. The genericity suite (online.py mixgate): one translator pair unchanged on held rows of every kind (agent along E4B's reply 40, agent along E2B's own sample 30, prose / code / story / question 40 each, fresh questions 150): cross-entropy to E4B and top-1, harm (KL from E2B alone where E4B and E2B alone agree), state change against E2B's own at layers 12, 17, 22, p(true) and answer mass for questions, for alone / bridge / control / sham (the bridge fed E4B's states shifted by half the span). Run for gall, the hint pair and the R3 pair.
Order (dreamer/translator2_4090.sh, code4): wait for the running res gate (old hint translator), stop the old queue; smoke test (both gall translators 20 steps, the suite on 3 rows per kind; halts on failure); both gall translators; the agent rung and the suite; alldirect (the baseline every arm needs); if the gall bridge beats its control on the agent rung by ≥ 0.05 nats over the whole reply: its range check, both bases, then flip, res, resctl, resdec with gall (tag _g); otherwise resctl and resdec with the old hint translator; then alldirect at rank 64.
Predictions:
- The gall control's state change at layer 12 is under a third of the hint control's on the agent rows (70%).
- On the agent rung the gall bridge beats its control by ≥ 0.05 nats (60%), by less than the hint pair's 0.126 (70%).
- Harm (KL from E2B alone where E4B and E2B agree) is lower for the gall bridge than the hint bridge on every kind (65%).
- The gall bridge is closer to E4B than E2B alone on at least 3 of prose, code, story, question (60%); the hint or R3 bridge is worse than alone on at least one of them (55%).
- Fresh questions: answer mass ≥ 0.95 through the gall bridge (70%), and above the hint bridge's (60%); p(true) through the gall bridge above E2B alone's (50%).
- Sham: through gall within 0.02 of alone on agent cross-entropy (55%); through the hint bridge more than 0.02 worse than alone (60%).
- With gall, res_g moves at least as many of the 52 disagreeing decisions as res with the old translator (45%).
Amendment 165a (05:55, before any run): an independent full review of the translator code found no error in the gall maths or the hook paths; its nine points were all applied: the smoke test now takes the longest reply among the longest rows (peak memory; also forced into the suite's smoke); the suite's harm KL is computed in chunks; the do-no-harm term is the tail cross-entropy (top 32 plus the rest), which has no gradient when the bridge changes nothing (the top-32 renormalised form pulled E2B toward a sharper distribution); bridge and control are seeded alike (torch seed 7, and each on-policy sample seeded from the draw), so they train on the same E2B samples; the in-forward path computes the translator on the written span only; look() frees the teacher's full logits before the student pass; the ridge is reported only for gall (its forward has no ridge term) and its targets are freed; one-kind training (old architectures) keeps its old row order; stories from held question items were removed from the story kind (213 stories now; the held question split is clean).
Amendment 165b (08:03, before the gall agent rung; user: "if it falls short remove the penalty?"). Training curves on identical draws: the gall bridge led its control by about +0.01 nats (steps 75-375) and +0.02 (400-500) on the mixed rows, far below the hint pair's 0.126 agent-rung margin; the size penalty (0.1) is the main suspect (the old translators changed E2B's state by ~30%). dreamer/translator3_4090.sh watches translator2: if the gall bridge's agent-rung margin over its control is ≥ 0.05 nats, nothing changes; otherwise, after translator2's three suites, it stops translator2 and trains gall with no size penalty (ON_TR_SIZE=0; do-no-harm, mix, mid layers, noise, depth dropout unchanged; data3, same rows), its rung gate and suite, alldirect, then the arms with it if its margin is ≥ 0.05 (tag _g0), else resctl and resdec with the old hint translator; then rank 64.
Predictions: gall (penalty 0.1) agent-rung margin below 0.05 (65%). With no size penalty: margin ≥ 0.05 (50%); its control's state change at layer 12 still under half the hint control's (60%); harm on fresh questions no worse than with the penalty (55%).
Results so far (08:12-08:29), recorded before any no-penalty result:
- Agent rung (120 points), whole-reply cross-entropy to E4B: E2B alone 0.861; gall bridge 0.758, gall control 0.783 (margin 0.026); hint bridge 0.746, hint control 0.872 (margin 0.126). Disagreements moved (of 23): gall 8 / control 7; hint 12 / control 9.
- Genericity suite (held rows; cross-entropy to E4B, alone → bridge / control / sham):
  | Kind | alone | gall b / c / sham | hint b / c / sham | R3 b / c / sham |
  |---|---|---|---|---|
  | agent (E4B's reply) | 0.871 | 0.763 / 0.788 / 0.917 | 0.755 / 0.891 / 14.28 | 0.720 / 0.860 / 16.44 |
  | agent (E2B's own sample) | 0.99-1.08 | 0.883 / 0.915 / 1.042 (alone 0.989) | 0.913 / 1.098 / 14.09 (alone 1.027) | 0.918 / 1.096 / 16.48 (alone 1.084) |
  | prose | 2.776 | 2.415 / 2.477 / 2.531 | 3.830 / 4.536 / 8.59 | 3.878 / 4.281 / 9.67 |
  | code | 1.375 | 1.181 / 1.220 / 1.284 | 1.726 / 1.956 / 10.21 | 1.502 / 1.764 / 12.45 |
  | story | 2.483 | 2.127 / 2.182 / 2.241 | 3.267 / 4.222 / 8.76 | 3.192 / 4.084 / 9.79 |
  | question (held items) | 0.869 | 0.541 / 0.619 / 0.728 | 0.986 / 5.285 / 9.77 | 1.397 / 3.113 / 10.90 |
  | fresh questions (asked test items) | 0.738 | 0.420 / 0.516 / 0.577 | 1.142 / 5.179 / 9.79 | 1.299 / 3.150 / 10.31 |
  Fresh questions p(true): alone 0.678; gall bridge 0.797, control 0.756, sham 0.672, answer mass 0.992; hint bridge 0.444 (mass 0.873); R3 bridge 0.439 (mass 0.759). State change at layer 12 (norm ratio to E2B's own): gall 0.07-0.10 (bridge and control alike); hint 0.71-0.77 / 0.78-0.87; R3 0.77-0.84 / 0.80-0.87. Harm (KL from E2B alone where E4B and E2B agree), agent: gall 0.04, hint 0.14, R3 0.16; prose 0.18 / 1.52 / 1.77.
- Reading: gall is the only translator that helps E2B on every kind (prose, code, stories, held and fresh questions, agent turns along either model's reply), and its bridge beats its own control on every kind (by 0.025 agent to 0.096 fresh questions). The old translators help on agent turns only, harm everything else (fresh questions p(true) 0.678 → 0.44), and collapse when fed misaligned states (cross-entropy 10-16); gall's sham costs at most +0.05. Its control is no longer handicapped: the old controls had to rebuild E2B's whole layer-12 state from its own later layers (ridge R² 0.33) and were worse than E2B alone, which inflated the old bridge-over-control margins (0.126 hint, and R167's 0.163 at 31B → E4B used the same design). With a control that can also learn to improve E2B from its own state, the E4B-specific part on agent turns is ~0.026 nats.
- Predictions (addendum 165): gall control's layer-12 change under a third of the hint control's: yes (0.09 vs 0.78). Agent-rung margin ≥ 0.05: no (0.026); under 0.126: yes. Harm lower than the hint bridge's on every kind: yes. Closer to E4B than alone on ≥ 3 of prose/code/story/question: yes (4 of 4); hint or R3 worse than alone on at least one: yes (on all four). Fresh questions: mass ≥ 0.95 (yes, 0.992), above the hint's (yes), p(true) above alone's (yes, 0.797 vs 0.678). Sham within 0.02 of alone on agent: no (+0.046); hint sham more than 0.02 worse: yes (14.3). Amendment 165b: penalty-0.1 margin below 0.05: yes.
Amendment 165c (08:35, before any no-penalty result): the 0.05 agent-rung rule was set when every control was a replacement translator; it no longer picks the arms' translator. dreamer/translator4_4090.sh lets translator3 finish the no-penalty translators, their gates and alldirect, then runs the arms with the gall version whose agent-rung margin over its control is larger (tag _g or _g0), with no fallback to the old hint translator; then rank 64.
Amendment 165d (08:55, before any no-penalty result; user: "this could either be good or bad... change may make it better or worse"). At step 200 the no-penalty bridge translator changes E2B about three times as much as the penalised one (size 0.093 vs 0.033) for ~0.002-0.04 lower training loss on the same rows. The arms' translator is now chosen by dreamer/translator5_4090.sh (replacing translator4, which was still only watching): the no-penalty version only if all hold: its agent-rung margin over its control is larger than the penalised one's (0.026); its mean harm over the suite's kinds is at most 1.25 times the penalised one's; its sham cost on agent turns (sham minus alone) is at most 0.05 above the penalised one's; its fresh-question p(true) through the bridge is at least the penalised one's minus 0.02. Otherwise the penalised version (data2, tag _g).
Predictions: the no-penalty bridge's agent margin is within ±0.02 of 0.026 (55%); the no-penalty version fails at least one robustness check (55%); the penalised version is chosen (60%).

## Addendum 166 (2026-11-07, 09:20): the cross-tokenizer bridge, Gemma 4 E4B → Qwen3-1.7B on Colab (user: "right now we are teaching models with the same tokenizer, and in this case maybe pure distilation is easier better, but when we try doing it with Qwen and Gemma the bridge may be better with the translator... can we try the smaller qwen models with the e4b as the teacher and a translator that uses qwens tokenizer to bridge the gemma 4 one?"; chose Qwen3-1.7B and Colab now)

Why it may matter more here: with one tokenizer, output distillation gets the teacher's full distribution as a target; across tokenizers (Gemma 262k, Qwen 151k) it cannot, and falls back to sequence-level SFT on the teacher's text or approximations (ULD, ALM). A state bridge needs only position alignment, and bridged Qwen speaks Qwen's own vocabulary, so it can supply full soft targets that cross-tokenizer distillation otherwise lacks.
Step 1 (transfer/xbridge.py, kaggle_g4agent/colab/xq_bridge.sh): both models read the same text; each Qwen position gets E4B's state from the last Gemma token ending at or before the Qwen token's end (character offsets; checked on real text: Qwen's pieces of "<|turn>" take the state before it until the Gemma token completes). The gall translator (addendum 165) sized for Qwen: E4B's layers 20, 27 and Qwen's write-layer state (layer 9 of 28) → gated change at layer 9, gated additions at 10-27; control: Qwen's own layers 14, 18. Loss: bridged Qwen's next-token cross-entropy on the reply part (E4B's own replies on agent rows; the text itself on prose, code, stories; the letter on plain-format questions) + 0.1 × change size + 0.5 × do-no-harm (where Qwen alone already puts the true token on top). Mixed rows (agent 50%, prose 15%, code 15%, questions 10%, stories 10%; agent contexts cut to 3,072 Gemma tokens), noise 0.1, depth dropout 0.15, 600 steps each; small pass first. Suite: held rows of every kind, Qwen alone / bridge / control / sham; cross-entropy and top-1 on the reply tokens, harm, state change, p(true) on held and fresh plain-format questions, E4B's own p(true) for reference.
Step 2 (designed after step 1, if the bridge carries E4B's information): a Qwen LoRA by sequence-level distillation (SFT on E4B's replies, the cross-tokenizer baseline) against distillation from bridged Qwen's full own-vocabulary distributions.
Predictions:
- The bridge beats its control on agent-reply cross-entropy by more than the same-tokenizer gall margin (0.026) (60%), by ≥ 0.05 (45%).
- The bridge is closer to the text than Qwen alone on every kind (60%).
- Fresh plain questions: E4B's p(true) above Qwen alone's (70%); the bridge's above Qwen alone's (55%) and above its control's (50%).
- Sham within 0.1 nats of Qwen alone on agent replies (60%).
Amendment 166a (09:30, before launch): an independent review found no crash bug (alignment, off-by-one, letter tokens, hooks in transformers 5.20.2, gradients, memory and the job's inputs checked on the real data and tokenizers); applied: the written span never includes Qwen's first token (Qwen has no BOS, so its first position is likely an attention sink whose outliers would inflate the whitening spreads) nor positions aligned only to E4B's BOS (s0 = max(1, first position aligned past BOS, reply start − 1,024)); the spread ratios are logged; the control skips E4B's pass; the suite reuses the alone pass's states.
Small pass (Colab, 20 steps each, 3 rows per kind): runs; calibration spread ratios (max/median) source 10.6, write layer 6.4, sites 18.3 (no sink outliers); 20 steps take ~10 s. On the plain pattern questions Qwen3-1.7B alone is near chance (p(true) 0.29-0.40) where E4B gets 0.82-0.84: a large gap to bridge.

## Addendum 167 (2026-11-07, 09:15): the penalty-vs-prediction loop on the Qwen bridge (user: "can we have some kind of eval loop where penalty and prediction are optimized?"; chose the Qwen bridge, Colab, 3 parallel)

Settings (size penalty : do-no-harm weight): the reference 0.1:0.5 (addendum 166's run) and 0:0.5, 0.01:0.5, 0.03:0.5, 0.3:0.5, 0.1:0.25, 0.1:1.0; each a bridge and a control translator at 600 steps (a pair takes ~10 minutes on a G4, so no halving) and the full suite. dreamer/xq_sweep_launch.sh starts three sessions (two settings each, kaggle_g4agent/colab/xq_sweep.sh) only if the reference bridge leads its control on average over the kinds. Pick (transfer/sweep_pick.py): prediction = the bridge's lead over its own control (control minus bridge cross-entropy), on agent replies and averaged over kinds; admissible if mean harm ≤ 1.25 × the reference's, sham cost on agent replies ≤ the reference's + 0.05, fresh-question p(true) ≥ the reference's − 0.02; score 0.5 × agent lead + 0.5 × mean lead; the full table is reported.
Predictions: the lead rises as the size penalty falls from 0.3 to 0.01 and flattens below (55%); the winner has a size penalty ≤ 0.03 (50%); no setting's lead exceeds the reference's by more than 0.03 nats (55%); a do-no-harm weight of 1.0 fails the fresh p(true) check or loses lead (45%).
Addendum 166 result (reference, size 0.1, harm 0.5; Colab, 600 steps each, 4 min bridge / 2 min control): held rows, next-token cross-entropy on the reply tokens (Qwen tokens), alone → bridge / control / sham:
| Kind | alone | bridge | control | sham | bridge lead over control |
|---|---|---|---|---|---|
| agent (E4B's replies) | 1.505 | 0.723 | 0.769 | 1.006 | 0.046 |
| prose | 3.645 | 3.294 | 3.305 | 3.321 | 0.011 |
| code | 1.919 | 1.655 | 1.659 | 1.678 | 0.004 |
| story | 3.753 | 2.938 | 2.966 | 3.042 | 0.028 |
| question (held items) | 1.547 | 0.268 | 0.472 | 1.261 | 0.204 |
| fresh questions | 1.625 | 0.285 | 0.434 | 1.476 | 0.149 |
Fresh plain questions p(true): Qwen alone 0.506, bridge 0.914, control 0.850, sham 0.569, E4B itself 0.854 (answer mass through the bridge 0.998); held items: 0.512 / 0.890 / 0.782 / 0.619 / E4B 0.826. State change at the write layer 0.09-0.13 (bridge and control alike).
Reading: across tokenizers the bridge leads its own (fair, gall) control on every kind, mean lead 0.074 nats, about three times the same-tokenizer gall margin (0.026); on the pattern questions bridged Qwen3-1.7B passes E4B itself (0.914 vs 0.854 on items no translator saw), and the gain needs E4B's aligned states (sham 0.569, near Qwen alone). The control's own large gain (0.506 → 0.850) is question-format learning from the 10% question rows: a translator that sees the student's state is also an adapter, which is why the control comparison is needed.
Predictions (166): agent lead above 0.026: yes (0.046); ≥ 0.05: no. Bridge closer than alone on every kind: yes. Fresh p(true): E4B above Qwen alone: yes; bridge above alone: yes; above its control: yes. Sham within 0.1 of alone on agent replies: no (1.006 vs 1.505: the sham keeps the translator's self-correction, so it is better than alone, not equal). The loop (addendum 167) launched at 09:24 (reference mean lead 0.074 > 0).

## Addendum 168 (2026-11-07, 09:35): step 2, a Qwen LoRA from the bridge against cross-tokenizer distillation (user: "yes, set up step 2 after the sweep")

After the loop's pick (the reference 0.1:0.5 if no setting is admissible): kaggle_g4agent/colab/xq_step2.sh (one G4) retrains that setting's bridge and control translators (600 steps), then three Qwen3-1.7B LoRAs (rank 16, MLP at the graft depths, lr 3e-4, 300 steps of 2 rows; 70% agent rows with E4B's replies, 30% training questions with E4B's own answer letter as the reply, right or wrong) on the same rows, teacher-forced: seq (cross-entropy on E4B's tokens: sequence-level distillation, the usual cross-tokenizer route); bridge (forward KL to the bridged Qwen's full own-vocabulary distribution at every reply position, adapter off); control (the same from the control-bridged Qwen). Scored with no bridge, with Qwen alone, on the suite's held rows: cross-entropy on E4B's agent replies, p(true) on held and fresh plain questions, cross-entropy and KL from Qwen alone on prose, code, stories. Small pass first; launched by dreamer/xq_step2_launch.sh after the loop and an independent code review.
Predictions: on fresh questions without the bridge, bridge-LoRA p(true) above seq-LoRA's (55%) and above control-LoRA's (55%); bridge-LoRA within 0.05 nats of seq-LoRA on E4B's agent replies (seq trains on exactly those tokens) (55%); every LoRA above Qwen alone on fresh p(true) (70%); no LoRA moves prose/code/story KL from Qwen alone above 0.1 (60%).
Amendment 167a (09:45, the loop half done; user: "we want to optimize on the eval step against the loss and penalty, etc"). The first four settings (size:harm → agent lead / mean lead / harm / fresh p(true) through the bridge): 0.1:0.5 0.046 / 0.074 / 0.287 / 0.914; 0:0.5 0.054 / 0.066 / 0.322 / 0.904; 0.01:0.5 0.047 / 0.038 / 0.297 / 0.783; 0.1:0.25 0.052 / 0.073 / 0.390 / 0.902. Size 0.01 falls between 0 and 0.1 on every input yet drops to 0.783 on fresh questions: single-seed runs scored on 40 rows per kind are too noisy to rank near-ties. Added before step 2 (dreamer/xq_rep_launch.sh, transfer/rep_pick.py; XB_SEED in xbridge.py, SIZE:HARM:SEED in xq_sweep.sh): the loop's top two admissible settings (filled from the reference) retrained with seeds 1-3 on three sessions, scored on the larger suite (120 rows per kind, all 200 fresh questions); pick by the mean score over seeds (0.5 × agent lead + 0.5 × mean lead), unless its mean harm exceeds 1.25 × the other's or its mean fresh p(true) is more than 0.02 lower; step 2 uses that pick. Prediction: the two settings' mean scores differ by less than the seed spread (60%).
Amendment 168a (09:50): an independent review of step 2 found no bug (PEFT 0.21.2 keeps the decoder-layer hooks; the student pass includes the LoRA; disable_adapter gives base Qwen for the targets; no bridge in the student's pass; letter tokens and the answer position checked on all 1,040 question rows). Applied: the small pass trains on the longest agent row first (peak memory); both launchers accept only a SIZE:HARM pick line; the retrained translators are kept. Design notes: each row's loss is a mean over its positions, so a one-token question row weighs as much as a long agent reply (all arms alike); on question rows only seq uses E4B's letter, bridge and control learn the bridged (control-bridged) Qwen's distribution at the answer position.
Addendum 167 result (09:43; 40 rows per kind, one seed each): score (0.5 × agent lead + 0.5 × mean lead) by size:harm: 0.3:0.5 0.090 (agent lead 0.044, mean lead 0.136); 0.03:0.5 0.089 (0.049, 0.128); 0.1:0.25 0.063 (inadmissible: harm 0.390); 0.1:1.0 0.061 (0.043, 0.079); 0.1:0.5 (reference) 0.060 (0.046, 0.074); 0:0.5 0.060 (0.054, 0.066); 0.01:0.5 0.042 (inadmissible: fresh p(true) 0.783). Harm 0.245-0.390, sham cost about −0.5 everywhere, fresh p(true) 0.90-0.91 except 0.01. The mean lead zigzags with the size penalty (0.3 → 0.136, 0.1 → 0.074, 0.03 → 0.128, 0.01 → 0.038, 0 → 0.066), so the order is mostly run noise; the replication (167a) decides between 0.3:0.5 and 0.03:0.5. Agent leads are flat (0.043-0.054) in every setting.
Predictions (167): lead rises as the penalty falls and flattens below 0.01: no (zigzag). Winner's size ≤ 0.03: pending replication. No setting beats the reference's lead by more than 0.03: no on the mean lead (0.136 vs 0.074), yes on the agent lead. Harm 1.0 fails fresh p(true) or loses lead: no (0.901; score ≈ the reference's).
Amendment 167a result (10:03; 3 seeds each, 120 rows per kind, 200 fresh questions): size 0.03 harm 0.5: agent lead 0.051 ± 0.004, mean lead 0.068 ± 0.026, harm 0.267 ± 0.032, fresh p(true) through the bridge 0.846 ± 0.054 (control 0.817 ± 0.033), score 0.060; size 0.3 harm 0.5: 0.050 ± 0.002, 0.045 ± 0.034, 0.269 ± 0.046, 0.840 ± 0.038 (control 0.815 ± 0.040), score 0.047. Pick 0.03:0.5 (higher mean; the guards pass), but the two are a tie within the seed spread: the single-run mean leads of 0.136 and 0.128 were lucky draws (replicated 0.045-0.068, about the reference's 0.074). The one stable quantity is the agent lead, 0.050 ± 0.003 nats in every setting and seed: E4B's specific contribution on agent replies across tokenizers; on fresh questions the bridge's edge over its control is about +0.03 p(true) on 200 items. Prediction (167a): the mean scores differ by less than the seed spread: yes. Step 2 runs with 0.03:0.5.
Addendum 168 result (10:23; setting 0.03:0.5; one seed; Qwen3-1.7B + LoRA, no bridge at test; 40 rows per kind, 150 fresh questions):
| | E4B's agent replies (CE) | held questions p(true) / CE | fresh questions p(true) / CE | KL from Qwen alone: prose / code / story |
|---|---|---|---|---|
| Qwen alone | 1.505 | 0.512 / 1.547 | 0.506 / 1.625 | - |
| seq (SFT on E4B's tokens and answers) | 0.826 | **0.787** / 1.068 | 0.796 / 0.854 | 0.055 / 0.059 / 0.109 |
| bridge (bridged Qwen's distributions) | 0.830 | 0.770 / **0.601** | **0.839** / **0.441** | 0.043 / 0.048 / 0.061 |
| control (control-bridged Qwen's) | 0.830 | 0.685 / 0.818 | 0.804 / 0.514 | 0.037 / 0.046 / 0.051 |
(E4B itself on the fresh plain questions: 0.854.) LoRA training: E4B's own answers were right on 90% of the training questions.
Reading: stored in an ordinary LoRA and used with no bridge, distillation from the bridged Qwen matches sequence-level distillation on E4B's agent replies (0.830 vs 0.826, the tokens seq trains on), beats it on fresh questions (0.839 vs 0.796 p(true), nearly E4B's 0.854; cross-entropy 0.441 vs 0.854: seq is confidently wrong more often) and moves Qwen less where it was not taught (story KL 0.061 vs 0.109). Of the 0.043 edge over seq on fresh questions, 0.035 is over the control: E4B's information, not soft targets as such (control 0.804 ≈ seq 0.796). On the held question items seq is slightly ahead on p(true) (0.787 vs 0.770) but far worse on cross-entropy. The first stored result where the bridge beats plain distillation; one seed, so a replication comes next.
Predictions (168): fresh p(true) bridge above seq: yes; above control: yes; bridge within 0.05 of seq on agent replies: yes; every LoRA above alone on fresh: yes; no LoRA's KL above 0.1 on prose/code/story: no (seq on stories 0.109).
Amendment 168b (10:45; user: "yes, replicate with the ULD baseline too"): step 2 replicated with seeds 1-3 on three Colab sessions (dreamer/xq_step2_rep_launch.sh; each seed its own translator pair at 0.03:0.5, XB_SEED, and its own LoRAs), the larger suite (120 rows per kind, 200 fresh questions), four arms: seq, uld, bridge, control. uld = seq + the Wasserstein-1 distance between E4B's and Qwen's sorted next-token probabilities (top 4,096; weight 1.0), E4B's taken at the aligned Gemma position (Universal Logit Distillation, Boizard et al. 2024, the standard cross-tokenizer logit method, here with our character alignment instead of index truncation). Aggregated by transfer/step2_agg.py (means, spreads, per-seed paired fresh-question differences). Also found and stopped: two tunnel loops from last night's live tests (tunnel_e4b.sh live3/live4) that kept reattaching to the stopped sessions.
Predictions: bridge above seq on fresh p(true) by ≥ 0.02 in the mean (55%) and in at least 2 of 3 seeds (60%); bridge above uld (50%); uld at least seq (55%); bridge's off-task KL below seq's on prose, code and stories (65%); all four within 0.05 on E4B's agent replies (65%).
Amendment 168b result (11:02; 3 seeds, each with its own translators; 120 rows per kind, 200 fresh questions; no bridge at test): mean ± spread over seeds:
| | E4B's agent replies CE | held p(true) / CE | fresh p(true) / CE | KL from Qwen alone: prose / code / story |
|---|---|---|---|---|
| Qwen alone | 1.419 | 0.519 / 1.469 | 0.513 / 1.498 | - |
| seq | **0.826** ± 0.004 | **0.794** ± 0.031 / 0.668 | **0.826** ± 0.003 / 0.518 | 0.056 / 0.059 / 0.104 |
| uld | 0.838 ± 0.005 | 0.637 ± 0.016 / 0.635 | 0.667 ± 0.031 / 0.544 | 0.057 / 0.046 / 0.112 |
| bridge | 0.839 ± 0.004 | 0.770 ± 0.004 / 0.515 | 0.799 ± 0.008 / 0.431 | 0.044 / 0.052 / 0.073 |
| control | 0.843 ± 0.003 | 0.750 ± 0.023 / 0.532 | 0.796 ± 0.030 / **0.394** | 0.047 / 0.049 / 0.082 |
Per-seed fresh p(true): bridge − seq −0.012, −0.037, −0.031 (mean −0.027); bridge − control +0.049, −0.042, +0.004 (mean +0.004); bridge − uld +0.132; uld − seq −0.159.
Reading (corrects the single-seed addendum 168 result, which was a lucky draw): stored in an ordinary LoRA, the bridged Qwen's distributions do not beat sequence-level distillation; seq is ahead on fresh and held question accuracy in every seed, and the bridge adds nothing over its own control (+0.004). What the soft targets do give, bridge and control alike: lower cross-entropy on questions (better calibrated) and less drift where Qwen was not taught (story KL 0.073-0.082 vs 0.104). So across tokenizers as with one tokenizer (R166), E4B's information is there live (agent lead 0.050 ± 0.003, fresh p(true) +0.03 over the control) but what survives into a LoRA is what output distillation already gives. ULD with weight 1.0 harms accuracy (−0.16 vs seq): matching sorted probabilities pulls Qwen toward E4B's sharpness without saying which token, and at a one-token answer that is the wrong signal; a smaller weight might tie seq, so this is ULD as configured, not ULD at its best.
Predictions (168b): bridge ≥ seq + 0.02 on fresh p(true): no (−0.027); in ≥ 2 of 3 seeds: no (0 of 3); bridge above uld: yes; uld ≥ seq: no; bridge's off-task KL below seq's on all three: yes; all within 0.05 on agent replies: yes.
Amendment 162b / addendum 163 result (alldirect, 11:32; 240 held-out points, 52 where base E2B's decision differs from E4B's): alldirect (every token, reverse KL to E4B, no state term) cross-entropy 0.791, tool choice 0.95, no-thinking decisions matching E4B 0.832, probability of E4B's choice 0.795, disagreeing decisions moved 20 of 52; res (the same + the faded state pull) 0.790, 0.96, 0.805, 0.764, 14 of 52; gapdirect 0.803, 0.94, 0.800, 0.696, 9 of 52. The state pull added nothing (res within noise of alldirect, slightly behind on decisions); res's lead over the selection arms was all-token distillation. Plain all-token on-policy distillation is the best behaviour arm at E4B → E2B, as S4c is at 31B → E4B; token selection (gapdirect, flip) loses to it.
Predictions: 162b: alldirect below gapdirect and flip on cross-entropy: yes; moves at least as many decisions as gapdirect: yes (20 vs 9); randdirect: not run. 163: res moves at least as many as alldirect: no (14 vs 20); by ≥ 4 more: no; within 0.01 on cross-entropy: yes (0.790 vs 0.791); res above resctl: pending.
Correction to the addendum 166 reading (11:50): "bridged Qwen3-1.7B passes E4B itself on the pattern questions" was a single run (0.914 vs 0.854); replicated over 3 translator seeds on all 200 fresh questions the live bridge scores 0.846 ± 0.054, a tie with E4B (0.854). Research report reports/Why LoRA loses bridge gains.md (11:48): stored LoRAs keep ~80-93% of the gain Qwen can make from its own states (what the control also gives) and ~8% (0.004 of 0.050 nats on agent replies) of the E4B-specific part; ranked causes: missing computation (imitation gap; strong), redundancy with SFT on E4B's text (strong), a metric that rewards sharpness (strong), thin off-policy dose (moderate, untested), dilution by the shared format (moderate, untested), rank/placement (weak); experiments X1-X7; the agent-reply lead is the primary storage metric (fresh p(true) has a standard error close to the whole live edge).

## Addendum 169 (2026-11-07, 12:30): X1 and the emulator graft (user: "yes, start with X1 and the emulator graft"; "what if the lora is running at recording while both models are being asked a question, we kinda did this by creating a graft ... we would have to figure out how to convert the graft in such a way that it updates the smaller model in a way that it preserves what it carried over"; "can you do a code review of the graft part to make sure we are doing it right ... use a subagent")

X1 (report experiment; transfer/xbridge.py x1, kaggle_g4agent/colab/xq_x1.sh; one session, seeds 1-3 in turn, the saved 168b translators and LoRAs uploaded to the HF dataset xq/x1/): per row, cross-entropy on the reply for Qwen alone, live bridge, live control, bridge LoRA, control LoRA; retention f = stored lead / live lead on the rows the LoRAs trained on (by kind, all unique) and on held rows, with paired bootstrap intervals; rows the translators trained on flagged; question cross-entropy against E4B's letter and the true letter; where the live lead sits per token; live bridge vs E4B p(true), paired.
Emulator graft, record stage (xbridge.py graft / geval, kaggle_g4agent/colab/xq_graft.sh; seed 1 first, a screen): with no E4B at its input, a student form learns to reproduce the bridged Qwen or the control-bridged Qwen: dense forward KL over the written span + XB_GSTATE_W × the per-layer state error on that span (whitened by the translator's spreads, fixed per-layer normaliser from 16 calibration rows); forms: graft (gated residual MLP of width 1,024 after each of layers 9-27, ~80M parameters, one forward pass) at state weight 1 and 0, and a PEFT LoRA (rank 64, MLP at layers 9-27) at weight 1 (if it matches the graft, no compile step is needed); 1,000 steps of 2 rows (70% E4B's agent replies, 30% training questions with E4B's letter), teacher-forced (own samples left for the compile stage: generating them would cost ~1 h per arm). Retention = (control form − bridge form) / (live control − live bridge) and each arm's fidelity (alone − form) / (alone − live target), on X1's held rows and on the rows each pair trained on, with bootstrap intervals. Two independent reviews: the first found no crash bug but six problems with what the numbers would mean (state target before the written span contradicting the one after it, no fidelity ceiling, X1's mixed train set and wrong letter, rounded ratios without intervals, an unbalanced loss, the compile's LoRA not covering layers 9 and 23-27); all were applied and a second review checks the rewrite.
Predictions: X1: f on held agent rows ≤ 0.2 in every seed (75%); f on the trained agent rows < 0.5 (65%); the top 10% of tokens carry at least half the net live lead (60%); live bridge within ±0.03 of E4B on fresh p(true), paired (65%). Graft screen: graft retention on held agent rows ≤ 0.3 (65%); control-form fidelity ≥ 0.7 (60%); the rank-64 LoRA within ±0.1 of the graft's retention (60%); state weight 0 vs 1 within ±0.1 retention (55%). If the graft keeps ≥ 0.25 of the lead on held rows, compile it (and replicate with seeds 2-3); otherwise the E4B-specific part is recorded as not computable from Qwen's own states at this scale.
Amendment 169a (12:55, before launch): the second review ran the whole pipeline end to end on tiny random models with the real tokenizers and data (no crash); applied: one shared state normaliser for both arms (the mean of the bridge's and the control's whitened deltas from Qwen alone on the same calibration rows) and the bridge translator's spreads for both (the per-arm normalisers differed ~1.5x); the KL over exactly the state span (positions s0..n-1); questions keep the true letter in the graft rows (the KL arms' target is the distribution before the letter; no extra E4B pass); the small pass takes the longest written span in Qwen tokens; the calibration loop always advances; graft_eval takes the training rows from any form's record and the saved graft width, adds a do-no-harm check (KL from Qwen alone on held prose, code, stories, 30 each) and plain gains next to the ratios; X1's E4B-letter cross-entropy is labelled as seq's target. Libraries synced (user: "install everything you need and sync up the libraries"): torch 2.15.0, transformers 5.22.2, peft 0.21.2, accelerate 1.15.0, trl 1.17.1 on the Mac and the Colab environment (these runs are the first on it); the 4090 after its arms queue (dreamer/sync_libs_4090.sh). Launched: X1 (session xqx1) and the graft screen seed 1 (xqg1).
Amendment 169b (12:42; user: "if the graft and smaller model work independently of the bridge, that in itself is what we publish in the kaggle paper"; "then we will have to figure out how to convert that working graft to a lora that we can use on kaggle"). The claim's bars, all on the same held rows with the bridge down: the graft alone beats Qwen alone; the bridge-graft beats the control-graft (retention ≥ 0.25 of the live E4B lead); the graft beats plain distillation on the same data (step-2 seq LoRA: SFT on E4B's text; and ULD); no harm on untaught prose, code, stories; 3 seeds. graft_eval now also scores addendum 168b's step-2 LoRAs (seq, uld, bridge, control) when present; dreamer/xq_gbase_launch.sh uploads the screen's forms after it ends and runs kaggle_g4agent/colab/xq_gbase.sh (the forms and the baselines together, seed 1). Conversion for Kaggle: Kaggle's vLLM accepts LoRA rank ≤ 128, up to 8 adapters; routes: the rank-64 LoRA form trained directly on the graft's objective (in the screen), or a compile (per-layer least-squares fit of the graft's output onto a low-rank MLP weight change, then chained state + output matching on 10-20x more rows, no E4B needed, Qwen's own samples included), or a weight-delta record form converted by PEFT's convert_to_lora. Then 31B → E4B with gall translators.
X1 result (12:49; 3 seeds; the saved 168b translators and LoRAs; no training). Agent replies (live lead control − bridge, cross-entropy; retention f = stored / live with 90% bootstrap intervals): seed 1 live 0.058, f on the LoRAs' training rows 0.21 [0.19, 0.24], on held rows 0.10 [0.07, 0.14]; seed 2 0.049, 0.16 [0.14, 0.19], 0.07 [0.02, 0.11]; seed 3 0.050, 0.18 [0.15, 0.21], 0.10 [0.06, 0.14]; Qwen's own gain (alone − control) kept by the control LoRA 0.86-0.88 on training rows, 0.91 on held rows; retention on training rows the translators never saw equals the rest (0.21, 0.17, 0.17). Held agent tokens: the top 10% by per-token lead carry 1.48-1.61 × the net lead and the bottom 10% −0.86 to −1.01 ×; the top 10% hold ~0.6 of KL(bridged ‖ control). Questions: the live lead swings −0.12 to +0.38 across translator seeds (unusable for storage, as the report warned). Live bridge − E4B p(true), fresh questions, paired: −0.034, +0.064, −0.030 (mean 0.000).
Reading: the LoRAs store only ~18% of the E4B-specific lead even on the rows they trained on (and ~9% held): not mainly a generalisation (lookup) failure but a failure to absorb E4B's part at this dose; the lead is a small net of large per-token swings, which a smooth adapter averages away; Qwen's own part is stored at ~90%. Predictions (169, X1): f held ≤ 0.2 every seed: yes; f train < 0.5: yes; top 10% carry ≥ half the net lead: yes (≈1.5×); live bridge within ±0.03 of E4B on fresh p(true): yes on the mean (per seed −0.034 to +0.064).

## Addendum 170 (2026-11-07, 13:05): online learning on the 4090 (user: "maybe also we are looking at this in the wrong way... cant this be seen as an online learning problem? ie, the smaller model, if it had the ability to learn while its running, could inheret the calculation and knowledge from the other model using this telepathy method... but we would need to figure out how to do online learning without catastrophic forgetting"; "yes, set up the online learning test on the 4090")

Why: X1 showed the stored LoRAs absorb only ~18% of E4B's specific lead even on their own training rows after 300 steps; the report's untested causes were thin, off-policy dose and dilution. Online learning answers both with the student's own samples on the inputs it meets, accumulated over a stream; our forgetting evidence (memory: on-policy data loses ~0-3% elsewhere vs 5-38% for SFT on a teacher's text; replay must be the model's own outputs) says bridge-driven online learning starts in the low-forgetting regime.
Design (online.py online, dreamer/online_4090.sh, code5; after the arms queue and the library sync): E2B with a rank-16 LoRA meets 600 unseen agent decision points once each in a fixed order. Per item: scored before learning it (prequential; the gate's measures: cross-entropy to E4B's top 32 along E4B's logged reply and the no-thinking decision vs E4B's; base E2B on the same item, paired), then learned (2 samples × 256 tokens, the teacher scores every token, reverse KL, one update, lr 3e-4 cosine over the stream). Every 150 items, the bridge down: the gate's measures on 60 held points and KL(base ‖ student) on 15 held untaught rows (prose, code, story). Arms: A direct (E4B's own distribution); B bridge (base E2B bridged by the no-penalty gall translator); C direct + KL(student ‖ base) on an untaught row each item (weight 1) + an earlier item again every 4th item. Each final LoRA gated on the 240 points.
Predictions: A's prequential gain over base on unseen items (cross-entropy) in the last 100 items ≥ 2 × the first 100 (60%); A at the final gate moves ≥ 20 of 52 disagreeing decisions (alldirect's) (50%); B within ±4 decisions of A (65%); C's final KL from base on untaught rows ≤ half of A's (60%) with held decisions within 0.03 of A's (60%); A's KL from base on untaught rows rises at every held checkpoint (60%).
Amendment 170a (13:15, before the run; an independent review found no crash bug): the stream draws from 231 runs in 31 repos, so by the last window ~94% of items come from an already-learned run: the prequential windows now also report items from first-visit runs, and the held curve (60 points from repos the stream never touches) is scored every 50 items and is the main learning curve; arm B's translator was trained on the stream's items, so A vs B is read on the held curve and the gate only; the forgetting check adds held question rows, which the keep term never touches; flat lr after warm-up (no late damping in an online stream); the small pass starts with the longest row; results saved at every held checkpoint; the runner waits for no python online.py process and a free GPU (twice), and checks the library sync. Prediction 1 is restated on first-visit-run items: their gain over base in the last 100 items ≥ 2 × the first 100's (55%).
Graft screen result (seed 1, 13:40; held agent rows 120, the bridge down; live lead control − bridge 0.058): retention (stored lead / live lead, 90% bootstrap) graft outputs-only (state weight 0) 0.275 [0.231, 0.317]; graft outputs + states 0.195 [0.144, 0.241]; LoRA rank 64 outputs + states 0.208 [0.155, 0.254]; step-2 LoRA on the same rows (X1, seed 1) 0.10 [0.07, 0.14]. On the agent rows the forms trained on: 0.322 [0.279, 0.364], 0.150 [0.093, 0.200], 0.253 [0.199, 0.300] (X1's step-2 LoRA 0.21): held ≈ trained, so what is learned generalises rather than being memorised per item. Fidelity (share of the target's gain over Qwen alone reproduced with no E4B): bridge forms 0.93-0.94, control forms ~1.0 on held agent rows. Untaught held text: every form lowers cross-entropy (prose −0.25 to −0.29 nats, code −0.24, stories −0.67 to −0.77 vs Qwen alone; KL from alone 0.13-0.17 on prose, below the live translators' 0.21-0.22), so no harm in likelihood. Questions: seed 1's translator has a negative live lead on questions (−0.12 held, −0.05 fresh), unusable for retention, as X1 showed.
Reading: the dense objective (KL over the whole written span, 1,000 steps) stores 2-2.75x the E4B-specific share of the step-2 output-only LoRA on the same held rows, with non-overlapping intervals; the layer-state term does not help (outputs-only is best); a rank-64 LoRA is as good as the 80M graft at the same objective. Predictions (169): graft retention ≤ 0.3: yes; control-form fidelity ≥ 0.7: yes; LoRA-64 within ±0.1 of the graft: yes; state weight 0 vs 1 within ±0.1: yes (0.08). Decision rule: the outputs-only graft's point estimate 0.275 passes 0.25 (interval lower bound 0.23): replicate with seeds 2-3, with a rank-64 LoRA at state weight 0 (the Kaggle-ready form) added.
Amendment 169c (13:50; the pre-registered rule): replicate with seeds 2 and 3 (sessions xqg2, xqg3; each seed's own 168b translators): the outputs-only graft and a rank-64 LoRA at state weight 0 (the Kaggle-ready form; not in the screen), each toward the bridge and the control, then the same eval. Seed 1's same-rows comparison with the step-2 baselines (xqgb) runs alongside. Predictions: the outputs-only graft's held agent retention ≥ 0.2 in both seeds (60%); the rank-64 LoRA at weight 0 within ±0.07 of the graft in every seed (65%); the 3-seed mean of the graft ≥ 0.25 (50%).
Same-rows comparison (seed 1, xqgb, 13:50; the bridge down; cross-entropy, lower is better; held agent / train agent / held prose / held code / held story): Qwen alone 1.385 / 1.566 / 3.680 / 1.729 / 3.753; live bridge 0.666 / 0.703 / 3.328 / 1.459 / 2.961; graft toward bridge (outputs only) 0.707 / 0.777 / 3.412 / 1.494 / 3.028; graft toward control 0.723 / 0.795 / 3.434 / 1.499 / 3.079; LoRA-64 toward bridge (+states) 0.710 / 0.774 / 3.400 / 1.493 / 2.986; step-2 seq LoRA (SFT on E4B's text, 300 steps, reply only) 0.774 / 0.856 / 3.549 / 1.578 / 3.596; step-2 bridge / control / uld LoRAs 0.780 / 0.786 / 0.788 on held agent. Questions (p(true) held / fresh): seq 0.849 / 0.822 (best on held), graft toward bridge 0.798 / 0.779, graft toward control 0.827 / 0.812, LoRA-64 0.802 / 0.773.
Reading: on agent replies the dense recipe beats plain distillation by 0.067 nats (graft 0.707 vs seq 0.774), but most of that (~0.05) is also reached by the control graft, which never sees E4B: it comes from the recipe (KL over the whole written span, soft targets from a translator-corrected Qwen, 1,000 steps), with E4B's specific part the 0.016 on top; the dense forms also lower cross-entropy on untaught prose, code and stories well beyond the step-2 LoRAs (story 3.03 vs 3.60), changing Qwen more (KL from alone 0.13-0.17 vs ~0.05) but towards lower loss: no harm in likelihood. Plain SFT keeps the best question accuracy on seed 1. Caveat for bar 3: the step-2 baselines had 300 steps on reply tokens; a dose-matched seq (1,000 steps) and a dense SFT baseline (next-token loss over the whole span) are needed before claiming the recipe beats plain distillation.
Amendment 169d (13:55; user: "yes, add the dose-matched baselines after the replication"): for bar 3, two plain-distillation LoRAs in the same form and dose as the graft-objective LoRA (rank 64, MLP at layers 9-27, 1,000 steps × 2 rows of the same 70/30 agent/question kinds): seq_d1000 (SFT on E4B's reply tokens) and seqdense_d1000 (next-token loss over the whole written span, no teacher: the dense recipe without any translator). dreamer/xq_dosebase_launch.sh waits for the replication, uploads seeds 2-3's forms, runs kaggle_g4agent/colab/xq_dosebase.sh for seeds 1-3 (three sessions; small pass first), every form and baseline on identical rows, and summarises with transfer/graft_agg.py.
Predictions: on held agent replies, the bridge-graft (outputs only) beats seq_d1000 by ≥ 0.02 nats in the 3-seed mean (55%) and beats seqdense_d1000 (50%); seqdense_d1000 is within ±0.02 of the control graft (50%: the dense recipe's gain over seq comes mostly from the span, not the translator); seq_d1000 keeps the best question accuracy (55%).
Result (169d, 14:50; 3 seeds, held agent replies, cross-entropy, lower is better): plain SFT on E4B's text at the same dose (seq_d1000) 0.662 / 0.659 / 0.662 and dense SFT over the written span (seqdense_d1000) 0.632 / 0.630 / 0.634, against the outputs-only bridge graft 0.707 / 0.718 / 0.714, the rank-64 LoRA 0.716 / 0.710 (seeds 2-3), the live bridge 0.666 / 0.679 / 0.674 and the 300-step seq LoRA 0.774 / 0.767 / 0.765. At the same dose both plain-distillation baselines beat the stored bridge forms, by 0.052 (seq) and 0.081 (seqdense) nats in every seed, and even the live bridge. The 300-step comparison earlier (graft 0.05 better than seq) was a dose effect. Where the bridge forms win: untaught prose (graft 3.41, LoRA-64 3.40-3.42, seq_d1000 3.55-3.59, seqdense_d1000 4.43-4.55 against 3.68 alone: dense SFT harms by ~0.8 nats) and fresh-question stability (p(true): graft 0.78 / 0.84 / 0.80, seq_d1000 0.84 / 0.39 / 0.87, seqdense_d1000 0.74 / 0.78 / 0.74). Untaught stories are not usable for the dense arms (seqdense 1.30-1.38 against 3.75 alone: the stories appear in question contexts, which dense SFT trains on). Predictions: graft beats seq_d1000 by ≥ 0.02: no (worse by 0.052); graft beats seqdense_d1000: no (worse by 0.081); seqdense_d1000 within ±0.02 of the control graft: no (0.09 better); seq_d1000 keeps the best question accuracy: no on the 3-seed mean (seed 2 collapses to 0.39), yes in seeds 1 and 3. Reading: on the measure that matches SFT's own objective (E4B's reply tokens), more plain distillation beats the bridge-trained forms, whose target, the bridged Qwen, sits at 0.67; bar 3 fails on held agent replies. What the bridge forms add is robustness: no harm off-task and steadier question accuracy. Task-level tests (addendum 172, coding with hidden tests) decide whether either difference matters.

**Online result (170, 2026-11-08 02:50; one seed per arm; E2B rank-16 LoRA, 600 stream items, E4B teacher).**

| arm | held CE, start → item 150 → item 600 | gate CE (E2B alone 0.826) | gate decisions fixed (of 52) | P(E4B's choice) (alone 0.655) | KL from base on untaught rows | KL on held questions |
|---|---|---|---|---|---|---|
| A direct | 0.825 → 0.786 → 0.787 | 0.791 | 25 | 0.774 | 0.089 | 0.030 |
| B bridge | 0.825 → 0.763 → 0.754 | 0.761 | 16 | 0.683 | 0.055 | 0.043 |
| C direct + keep + replay | 0.825 → 0.788 → 0.775 | 0.783 | 14 | 0.739 | 0.014 | 0.034 |

Prequential CE gain over base, first 100 → last 100 items: A 0.015 → 0.042, B 0.044 → 0.067, C 0.016 → 0.050. On first-visit-run items: A 0.015 → 0.061 (the last window has only 6 such items). Held decisions (55 points) peak at 0.87-0.89 mid-stream and end at 0.818 in all arms, where they started. The gate's 240 points are the better read.

Predictions:
- A's prequential gain in the last 100 ≥ 2 × the first 100: yes (2.9×; first-visit items 4×, on 6 items).
- A fixes ≥ 20 of 52 at the gate: yes (25).
- B within ±4 decisions of A: no (16 vs 25).
- C's KL on untaught rows ≤ half of A's: yes (0.014 vs 0.089, 6.6× less), with held decisions within 0.03 of A's: yes (equal).
- A's KL rises at every checkpoint: no (small dips at 400, 500 and 600; overall 0.027 → 0.089).

Reading:
- Online learning from one pass moves E2B toward E4B on repos it never saw. Most of the held gain comes in the first 150-200 items, then flattens.
- Direct fixes the most decisions.
- The bridge teacher gives the lowest CE and drifts less than direct, but fixes fewer decisions. Its translator was fitted on the stream's items.
- Keep + replay cuts drift on untaught rows 6.6× and still ends with better held CE than direct (0.775 vs 0.787). It fixes fewer gate decisions (14 vs 25).
- The protection covers only what is kept: drift on held questions, which the keep term never touches, is the same in all arms (0.03-0.04).
- This matches the coding result (174-181): a pull toward the base or a corrected self is what limits forgetting, at some cost to the decision changes.
- One seed per arm.

## Addendum 171 (2026-11-07, 14:30): the reconciliation loop (user: "so the graft cant reproduce it, but e4b could, couldnt we try to see if the graft from the smaller model, can be attached to the larger model and then we can try to reconsile the differenece and then re-attach it to the smaller model and see what happens?"; "I've added another 100 to colab... Lets try this loop and see what happens")

The graft lives in Qwen's coordinates, so it cannot run inside E4B; the E4B-side half of the system is the translator, so "reconcile on the larger model's side" = fine-tune the translator toward corrections the graft can reproduce (an imitable teacher: A2D, Warrington et al. 2021; LEAP's realisable corrections; report X7). Seed 1 (translators of 168b at 0.03:0.5; the screen's outputs-only grafts), kaggle_g4agent/colab/xq_recon.sh, transfer/xbridge.py (train reconcile mode, train_graft warm start, reval). Per round k = 1, 2: (a) reconcile: the bridge translator continued from round k-1 for 300 steps (lr 1e-4) with its own loss (cross-entropy on the reply, size, do-no-harm) + β × KL(bridged Qwen ‖ grafted Qwen with graft k-1, no E4B) over the written span; (b) re-attach: graft k-1 continued 500 steps (outputs-only dense KL) toward the reconciled translator; (c) dose control (session A): the round-0 graft continued 500 steps per round toward the unchanged translator. β = 5 (session A, with the dose control) and β = 20 (session B). Then every translator live and every graft with the bridge down on X1's held agent rows, 200 fresh questions and 30 held prose rows, with paired bootstrap differences. The measure that matters is Qwen's quality with the bridge down (cross-entropy on held agent replies) of the reconciled graft against the dose-control graft, not the retention ratio (the live lead itself is expected to shrink).
Predictions: the round-2 reconciled translator's live lead over the control on held agent rows is ≤ half of round 0's at β = 20 (60%) and ≥ 0.7 of it at β = 5 (55%); the round-2 reconciled graft is better than the round-2 dose-control graft on held agent replies by ≥ 0.005 nats (40%); the stored lead (control graft − graft) grows across rounds in the reconciled arm (50%).
Amendment 171a (15:20, after a code review, before any run): the continued grafts train on fixed rows like the round-0 grafts (XB_GONP=0; the default 0.3 own samples would have changed the recipe and split the re-attach and dose arms' rows); each round draws fresh rows (XB_ROUND) and the re-attach and dose arms of a round see the same ones; the pull acts only on agent and question rows (the graft's kinds) and there with the clean bridge (no noise, no depth dropout), over positions s0..n-1 as the graft trains; session A also continues the control graft 500 steps per round toward the unchanged control translator (the step-matched control for the stored lead); a third session runs β = 0 (the same loop with no pull: the extra-training control). The evaluation now reports the decisive paired difference, re-attached minus dose-control graft per round, and saves per-row values.
Result (16:30; seed 1; held agent replies, cross-entropy, lower is better; per round 300 translator steps + 500 graft steps): re-attached graft after round 2: β = 0 (no pull) 0.675, β = 5 0.681, β = 20 0.698; round 0 0.707; the dose control (the round-0 graft continued toward the unchanged translator, same steps) 0.699; the step-matched control graft 0.721. At β = 5 the re-attached graft beats the dose control by 0.012 [0.010, 0.014] (round 1) and 0.018 [0.015, 0.021] (round 2), and the stored E4B-specific lead (step-matched control graft − graft) grows 0.016 → 0.031 → 0.039 (dose control 0.019, 0.021). But the live translator's own lead grows too (control − live: β = 0: 0.058 → 0.085 → 0.101; β = 5: 0.073 → 0.082; β = 20: 0.056 → 0.054), and the no-pull session (β = 0) gives the best grafts: the gain comes from training the translator further (the 600-step translators were undertrained), not from pulling it toward what the graft can reproduce; the pull lowers the gain in proportion to β. What the pull does buy is less drift on untaught prose: round-2 graft − round-0 graft on held prose +0.071 at β = 0, +0.016 at β = 5, +0.012 at β = 20 (the live round-2 translators: 3.444, 3.396, 3.373 vs the control's 3.344). Fresh questions: intervals too wide to read. Predictions: live lead ≤ half at β = 20: no (0.054 vs 0.058); ≥ 0.7 at β = 5: yes (it grew); re-attached ≥ 0.005 better than the dose control: yes (0.018), but explained by translator training (β = 0 better still); the stored lead grows: yes. Reading: a better live bridge gives a better stored graft (best 0.675, against plain SFT's 0.66 and dense SFT's 0.63 at the same 1,000-step graft dose in 169d), so the bridge graft still does not beat plain distillation on this measure. One seed; session B's files were lost to a failed download (its logged means are used).
Replication result (169c, 14:37; held agent rows, the bridge down; retention = stored E4B-specific lead / live lead, 90% bootstrap): outputs-only graft: seed 1 0.275 [0.231, 0.317], seed 2 0.259 [0.210, 0.308], seed 3 0.211 [0.157, 0.259], mean 0.248; rank-64 LoRA outputs-only (the Kaggle-ready form): seed 2 0.240 [0.189, 0.289], seed 3 0.233 [0.185, 0.277] (seed 1 had the outputs + states LoRA, 0.208); the step-2 output-only LoRAs on the same rows (X1): 0.10, 0.07, 0.10. On the trained agent rows: graft 0.32, 0.37, 0.24; LoRA-64 0.47 (seed 2), 0.34 (seed 3). Live leads 0.058, 0.049, 0.050; bridge-form fidelity 0.94-0.95 in every seed. Untaught prose: every form lowers cross-entropy by 0.25-0.28 nats (no harm). Fresh questions: the live lead swings by seed (−0.05, +0.14, +0.11), retention there unusable.
Reading: in every seed the dense objective stores 2-3x the E4B-specific share of output-only distillation (≈ 0.24-0.25 vs ≈ 0.09, non-overlapping intervals), and the rank-64 LoRA does as well as the 80M graft: no compile step is needed; a Kaggle-legal LoRA trained directly on the bridged Qwen's dense outputs carries about a quarter of E4B's specific contribution with the bridge down. Predictions (169c): graft ≥ 0.2 in seeds 2-3: yes (0.259, 0.211); LoRA-64 within ±0.07 of the graft in every seed: yes; 3-seed graft mean ≥ 0.25: no (0.248, just under).

## Addendum 172 (2026-11-07, 14:50): can stored bridge forms make Qwen solve harder coding problems? (user: "can you try our best recipe wwith the graft and lora 64, on the qwen and e4b pair and see if qwen can do tougher coding tasks?"; earlier: "Did it solve any that it could only solve with the larger model bridge up?"; benchmark chosen: a LiveCodeBench subset)

Every earlier measure of the bridge was next-token cross-entropy or p(true). This is the first task-level test: whole programs checked by hidden tests. Data: LiveCodeBench code_generation_lite test5 + test6 (problems released Sep 2024 - Apr 2025), every easy and medium problem (test6 alone: 95; AtCoder stdin and LeetCode functional), up to 30 tests each (public first). One greedy attempt, ≤ 768 new tokens, the same plain prompt for every model (statement, I/O rule or the LeetCode starter code, then an open ```python block; no chat template, because the bridge needs one shared text). Own checker (transfer/xcode.py): each test in a fresh interpreter, CPU 8 s, memory 2 GB, no environment secrets; stdin answers compared token by token (float tolerance 1e-6), functional answers as JSON. Arms, all seed 2: Qwen3-1.7B alone; Gemma 4 E4B alone; Qwen + the outputs-only graft toward the bridged Qwen (graft_b) and toward the control-bridged Qwen (graft_c); Qwen + the rank-64 outputs-only LoRAs (lora_b, lora_c; the Kaggle-ready form); Qwen + the step-2 plain SFT LoRA on E4B's text (lora_seq; a reference, not dose-matched). Gap set = problems E4B passes and Qwen alone fails (fixed by the first evaluation). On the gap set only: the live bridge (Qwen with the seed-2 translator fed E4B's states, everything re-tokenized and re-aligned at each generated token, no cache, exactly as in training) and the live control (the same with Qwen's own states). Paired exact sign tests on discordant problems. kaggle_g4agent/colab/xq_code.sh; small pass first.
Caveats fixed in advance: the plain prompt gives lower absolute scores than published chat-template numbers for both models; the problems may be in either model's pre-training data (contamination raises absolute scores, and it is shared by every Qwen arm); none of the stored forms saw contest problems in training (agent turns 0.7, questions 0.3).
Predictions: E4B passes ≥ 15 problems Qwen alone fails (70%); graft_b and lora_b each within ±4 problems of Qwen alone, sign-test p > 0.1 (70%); bridge form vs its own control form (graft_b vs graft_c, lora_b vs lora_c) not significant (80%); the live bridge solves ≥ 3 gap problems the live control does not (45%); of those, at least one is also solved with the bridge down by graft_b or lora_b (30%); lora_seq within ±4 of Qwen alone (65%).
Amendment 172a (15:00, after 169d's result and a code review; the only coding output seen is a 4-problem smoke pass at 48 tokens): two more arms, the dose-matched plain-distillation LoRAs of 169d (seed 2: lora_sd = seq_d1000, lora_dd = seqdense_d1000), which replace lora_seq as the fair baseline; on the gap set a third live arm, Qwen alone on the same token-by-token path (alone_live), the matched baseline for the live bridge. Scorer fixes from the review: functional problems run the solution as a module with empty stdin and captured prints (only the returned value counts, as in LiveCodeBench); integers compare exactly (the float tolerance only for tokens with '.', 'e' or 'E'); LiveCodeBench's own import string; every test (no cap at 30); the gap set fixed once and reused; a scorer self-check before each evaluation. Added prediction: lora_sd and lora_dd each within ±4 problems of lora_b (60%).
Interim result (fast arms, 16:00; 188 problems, one greedy attempt, the bridge down for every Qwen form; seed 2): Qwen alone 23, E4B 74; graft_c 46, lora_c 43, lora_sd 38, lora_b 37, graft_b 35, lora_dd 32, lora_seq 32. New (Qwen failed, the form passes) / lost (Qwen passed, the form fails): graft_c +23 / 0, lora_c +22 / 2, lora_b +20 / 6, lora_sd +17 / 2, graft_b +16 / 4, lora_dd +14 / 5, lora_seq +12 / 3. Of the 55 problems only E4B solves: graft_c 19, lora_c 18, lora_b 16, graft_b 14, lora_sd 14, lora_dd 12, lora_seq 11. Paired: every form beats Qwen alone (p ≤ 0.04) except lora_dd (p = 0.06); graft_c beats graft_b (15 vs 4, p = 0.02); lora_c vs lora_b 13 vs 7 (p = 0.26); lora_b vs lora_sd 10 vs 11. Qwen alone scores 0 of 88 LeetCode problems: it often returns the starter code with `pass` (checked: the scorer is right); the trained forms write the body, part of their gain. Predictions: E4B ≥ 15 Qwen fails: yes (55); graft_b / lora_b within ±4 of Qwen: no (+12, +14); bridge form vs its control not significant: no for the graft (the control is better), yes for the LoRA; lora_seq within ±4: no (+9); lora_sd / lora_dd within ±4 of lora_b: yes / no (+1, −5). Reading: on whole programs with hidden tests, training toward the control-bridged Qwen (Qwen's own states, a learned gated correction fitted to E4B's text, then distilled densely) gives the most new solves and loses nothing; E4B's states do not help and the bridge forms do worse; plain SFT at the same dose sits between. One seed; the live stage is pending.

## Addendum 173 (2026-11-07, 14:50): replicate res_g0 if its control separates (user: "res_g0 looks promising"; "yes, replicate first if the control separates")

res_g0 (behaviour loop with the state pull toward the no-penalty gall translator's bridged states; E4B → E2B; 300 steps, rank 16) fixed 22 of the 52 gate decisions where E2B alone differs from E4B (cross-entropy to E4B along E4B's replies 0.788), against alldirect's 20 (0.791), the old-translator res arm's 14 (0.790), flip_g0's 4 and rand's 3. One seed each. Rule fixed now, before resctl_g0's gate: the control separates when resctl_g0 (the same arm, state pull toward the control translator's states, E2B's own) fixes ≤ 17 of 52 (≥ 5 below res_g0). If it does: the seed-1 adapters of alldirect, res_g0 and resctl_g0 are re-gated with per-point output, then seeds 2 and 3 of all three arms (~/toyq/behave/code6 = code4 + ON_SEED, seed offset 1000 × seed on every training seed, + per-point gate output; the same environment as seed 1; alldirect seed 1 ran on agent-only rows, as code6 does), then paired exact sign tests per seed and pooled (rep_summary.py). Either way the library sync and the online test (addendum 170) run after it. kaggle_g4agent/dreamer/replicate_4090.sh (runs on the 4090 from ~/toyq/behave).
Predictions (if run): res_g0 beats resctl_g0 in fixed decisions in all 3 seeds (55%); res_g0's 3-seed mean is ≥ 2 above alldirect's (40%); pooled sign test res_g0 vs alldirect p < 0.1 (25%); pooled res_g0 vs resctl_g0 p < 0.1 (45%).
Result (15:18): resctl_g0 fixes 20 of the 52 decisions (cross-entropy to E4B 0.787, probability of E4B's choice 0.790, top-token match 0.841), against res_g0's 22 (0.788, 0.791, 0.827) and alldirect's 20 (0.791, 0.795, 0.832). The control does not separate (20 > 17), so by the rule fixed in advance there is no replication; the library sync and the online test run next. Reading: the state pull toward E4B's bridged states adds nothing over the same pull toward E2B's own (control) states, and both match plain on-policy distillation; res_g0's 22 was within noise. The gain in these arms is the on-policy distillation, not E4B's states, in line with the cross-tokenizer results (169d).


## Addendum 174 (2026-11-07, 15:25): does the bridge add anything on top of plain SFT? Counted as fixed / new / kept (user: "we should test on 3 things really... how many problems it could not solve before, but now can, how many new problems it solves and not losing any current knowledge"; training pool: older LiveCodeBench)

169d showed plain SFT on E4B's text beating the bridge-trained forms on held cross-entropy, a measure that rewards copying E4B's text. This tests task success instead, and separates learning, generalising and forgetting. Training pool: LiveCodeBench test.jsonl-test4.jsonl (May 2023 - Sep 2024, every difficulty); rows = E4B's own solutions that pass every hidden test (greedy; for easy and medium failures the first of 4 samples at T 0.8 that passes), the code up to its closing fence. Held pool: test5 + test6 easy and medium (Sep 2024 - Apr 2025; the problems of addendum 172), never trained on. Arms, seeds 1-3 (each seed's translators of 168b), rank-64 LoRA on the MLPs of Qwen layers 9-27, 2 epochs, the same rows and order in every arm of a seed: sft (cross-entropy on E4B's solution), sft_b (+ 1.0 × KL(bridged Qwen ‖ student) over the written span), sft_c (+ the same toward the control-bridged Qwen), kl_b (the bridge KL alone). Counts against Qwen alone (greedy, the same session): (1) fixed: trained problems Qwen failed that the arm passes; (2) new: held problems Qwen failed that the arm passes (and on the gap set, E4B passes); (3) kept: problems Qwen passed that the arm still passes; plus off-task cross-entropy on X1's held agent rows, fresh-question p(true) and untaught prose. Exact paired sign tests per seed, pooled over seeds. transfer/xlearn.py, kaggle_g4agent/colab/xq_learn.sh (build once, then one session per seed).
Predictions (3-seed pooled): sft fixes the most trained problems (55%: copying the exact solutions helps there); sft_b solves more new held problems than sft_c (35%) and than sft (35%); sft_b keeps more of what Qwen solved than sft (50%); kl_b keeps the most and fixes the fewest (55%); off-task, kl_b and sft_b keep prose and fresh-question accuracy closer to Qwen alone than sft (60%). If sft_b does not beat sft_c on new problems, the bridge adds nothing measurable beyond the recipe at this scale.
Amendment 174a (15:55, after a code review, before any run): a gate before the arms. The build session also measures, with seed 1's translators, the cross-entropy of E4B's verified solution tokens for Qwen alone, the live bridge and the live control, on the rows and on E4B's passing held solutions. The three seed sessions run only if the bridge beats its control on the rows by ≥ 0.02 nats with the 90% bootstrap lower bound above 0. If it does not, sft_b and sft_c cannot differ beyond noise, and the ~60 CU are not spent. Review fixes: every generation is kept; samples are scored in parallel; a row needs E4B's own closing fence (none added by hand); rows and baselines are checked against each other by hash; the evaluation generates only where a count can move (the trained problems, the held ones, and the untrained train-pool problems Qwen passed), re-scores every arm-vs-Qwen disagreement one at a time, splits fixed by row source and difficulty, and reports kept separately for held and train-pool problems. Added measures: the held cross-entropy of E4B's passing solutions per arm, the KL split into prompt and reply positions, and the target's own cross-entropy. LiveCodeBench is pinned to revision 0fe84c39. Note: kl_b has no control arm, so its result cannot be credited to E4B; sft_b vs sft_c is the decisive pair.
Build and gate (17:20): the train pool has 713 problems; E4B solves 244 greedily and 55 more by sampling (299 rows, fenced, verified); Qwen alone solves 103 of the train pool and 23 of the 188 held, E4B 74 held. Gate (seed 1's translators): cross-entropy of E4B's solution tokens, Qwen alone 1.025, live control 0.713, live bridge 0.668 on the 299 rows, lead 0.045 [0.043, 0.048]; on E4B's 74 passing held solutions 1.000, 0.694, 0.646, lead 0.048 [0.043, 0.053]: GATE_PASS, the arms sft, sft_b, sft_c, kl_b, sdft run for seeds 1-3. On contest code E4B's states carry about as much beyond the control as on agent rows (0.05 nats).

## Addendum 174b (2026-11-07, 16:10): do we still need the translator for the control? (user: "do we still need the translator for the control?"; "yes, make that swap if the gate fails")

The control recipe of 172 has three parts: an anchor to Qwen itself over the whole span (soft targets that equal Qwen's own distribution where E4B's text does not pull), do-no-harm and size limits inside the translator, and two-stage smoothing (fit a corrected model, then distill it). The first two can be had in one stage with no translator. If 174's gate fails, the seed sessions run sft, sft_self (SFT + 1.0 × KL(original Qwen ‖ student) over the written span, the standard KL-to-self regulariser of Learning without Forgetting) and sft_c instead of sft_b and kl_b; the gate decision is written into xl_base.json and each session chooses its arms from it. Predictions: sft_c solves more new held problems than sft_self (40%); sft_self and sft_c keep about the same (within 2 problems, 55%); both keep more than sft (65%).
Related work (search 11-07, four queries, not exhaustive): the principle that near-self targets learn as well as SFT with much less forgetting is known. Yang et al. 2024 (SDFT, arXiv 2402.13669) let the model rewrite the targets in its own words before fine-tuning. Shenfeld et al. 2026 (SDFT, arXiv 2601.19897, ICML 2026) use the model itself, conditioned on the demonstration in context, as an on-policy teacher with reverse KL, and report higher new-task accuracy than SFT with much less forgetting. Learning without Forgetting (Li and Hoiem 2016) adds a KL to the original model's outputs. The control translator resembles a ReFT intervention (Wu et al. 2024, arXiv 2404.03592: a learned edit of hidden states with the base frozen); Deep Context Distillation (arXiv 2503.08727) trains plug-in knowledge modules on a teacher's outputs and hidden states. No source found that builds the near-self teacher as the student plus a gated hidden-state correction fitted to the teacher's text and then distils it densely into a LoRA. The construction looks new, the principle does not; before any claim it must beat Shenfeld et al.'s in-context self-teacher and KL-to-self SFT (sft_self) at the same dose.
Amendment 174c (16:15; user: "add the in-context self-teacher arm too"): a fifth arm, sdft, in both arm sets (gate pass or fail), following Shenfeld et al. 2026: each step the student samples its own answer to the row's problem (temperature 1.0, at most 512 tokens, stopping at the closing fence); the teacher is an EMA of the student's LoRA (α = 0.02, from their sweep {0.01, 0.02, 0.05}; their frozen-base teacher underperforms, the live student diverges) reading the same problem with E4B's verified solution in context (their template adapted to our code prompt: "This is an example for a response to the question: ... Now answer with a response of your own."); the loss is the per-token KL(teacher ‖ student) over the vocabulary on the sampled tokens, the first 3 masked; no cross-entropy on E4B's text; the same LoRA-64, rows, order and steps as the other arms (they use full fine-tuning). Their caveat: small models with weak in-context learning give a poor teacher signal (their 3B model lagged SFT). Predictions: sdft solves fewer new held problems than sft (55%: Qwen3-1.7B is small); sdft keeps the most of what Qwen solved (60%); sft_c solves more new problems than sdft (55%).

## Addendum 174q (2026-11-07, 16:30): is the translator a missing piece? The quick test (user: "can we test if the translator is actually a missing piece we need sooner?")

The control LoRA of 172 (translator path) solved 43 of 188 with 2 lost; plain SFT at the same dose 38. Same rows, form and dose (rank-64 LoRA, MLP at layers 9-27, 1,000 steps × 2 rows, 70/30 agent/question rows), seeds 2 and 3: seqself = SFT on E4B's reply + 1.0 × KL(original Qwen ‖ student) over the written span, the graft-objective LoRA's span, no translator (xbridge.py lora seqself). Scored on the 188 problems of 172 (one greedy attempt, the bridge down) with Qwen alone, the control LoRA, the bridge LoRA and plain SFT re-run in the same session (kaggle_g4agent/colab/xq_selftest.sh). The translator counts as a missing piece if the control LoRA beats seqself on new problems in both seeds and in the pooled paired test (p < 0.1). Prediction: seqself within ±3 problems of the control LoRA in both seeds (50%); seqself loses no more than 2 problems Qwen solves (60%).
Result (17:15; 188 problems, one greedy attempt, the bridge down; re-runs of the earlier forms reproduced their seed-2 scores exactly): solved, seed 2 / seed 3 / pooled: Qwen alone 23 / 23 / 46; control LoRA 43 / 28 / 71; bridge LoRA 37 / 35 / 72; plain SFT 38 / 30 / 68; seqself (SFT + KL to self, no translator) 35 / 30 / 65. Control vs seqself: 14 vs 6 (p = 0.12) in seed 2, 7 vs 9 (p = 0.80) in seed 3, pooled 21 vs 15 (p = 0.41). The rule (the control beats seqself in both seeds and pooled p < 0.1) is not met: the translator is not shown to be a missing piece. Seed 3 also reverses seed 2's control-over-bridge order (bridge 35 vs control 28, 12 vs 5, p = 0.14); pooled the three translator-free and translator forms sit within 7 problems of each other while all beat Qwen alone. Predictions: seqself within ±3 of the control in both seeds: no (−8, +2); seqself loses ≤ 2 Qwen-solved problems: no (3, 3). Reading: the 172 interim reading ("the control recipe is best") was one seed and does not hold; at this scale the stored forms are interchangeable within seed noise, and what generalises is that any of them, trained on E4B's agent turns and answers, makes Qwen a better contest coder.

## Addendum 175 (2026-11-07, 16:45): does a better-trained translator also help the self-bridge? (user: "it would be interesting to see if using the better trained translator also help the self-tuned/self bridged control we were doing... (in the translator is used by it at all)")

171 showed the 600-step translators were undertrained (two more rounds of 300 steps raised the bridge's live lead 0.058 → 0.101 at β = 0). The control translator (Qwen's own states; the self-bridge behind the best coding LoRA of 172) was never trained further. Seed 2: both translators get the β = 0 recipe (two rounds of 300 steps, lr 1e-4, agent and question rows clean, fresh rows per round; XB_CLEAN=1, no graft), then a rank-64 LoRA toward each (the outputs-only graft objective, 1,000 steps, fixed rows, as 172's LoRAs): lora_c_long, lora_b_long. Scored with the original lora_c / lora_b and Qwen alone on the 188 problems of 172 (one greedy attempt, the bridge down) and on X1's held rows (geval). kaggle_g4agent/colab/xq_longtr.sh. If the self-bridge improves with its translator, the translator is doing work beyond anchoring to self (read with 174q). Predictions: lora_c_long solves ≥ 3 more than lora_c (35%); lora_b_long solves ≥ 3 more than lora_b (40%); lora_c_long still ≥ lora_b_long (60%).
Result (17:30; seed 2, 188 problems, one greedy attempt, the bridge down): Qwen alone 23; control LoRA 43 → long-trained control LoRA 36 (5 vs 12, p = 0.14); bridge LoRA 37 → long-trained bridge LoRA 37 (11 vs 11); long control vs long bridge 36 vs 37 (8 vs 9). Both long forms still beat Qwen alone (16 vs 3, 17 vs 3). Predictions: lora_c_long ≥ +3: no (−7); lora_b_long ≥ +3: no (0); lora_c_long ≥ lora_b_long: no (36 vs 37). Reading: two more rounds of translator training (which raised the live bridge's lead on agent rows in 171) do not give better coding LoRAs in this seed; the control's 43 looks like seed 2's high draw (seed 3: 28). 176 (both translators, seeds 2-3, up to 4,800 steps) runs anyway (amendment 176a).

## Addendum 176 (2026-11-07, 17:00; queued, runs only if 175's long-trained control LoRA solves ≥ 3 more problems than the original control LoRA; user: "if the translator actually does add something, then a better trained translator might work a lot better"; "yes, queue the curve if 175 shows a gain")

A training-length curve for the control translator (the self-bridge). Seeds 2 and 3: the 600-step translator continued in rounds of 300 steps with 175's recipe (lr 1e-4, agent and question rows clean, fresh rows per round, no graft); a rank-64 LoRA toward it at 1,200, 2,400 and 4,800 total steps (the outputs-only graft objective, 1,000 steps, as 172's LoRAs); scored with Qwen alone and the original control LoRA (600 steps) on the 188 LiveCodeBench problems of 172 and on X1's held rows and untaught prose (geval). kaggle_g4agent/colab/xq_curve.sh, launched by kaggle_g4agent/dreamer/xq_side3_launch.sh. Predictions (if run): solved problems rise from 600 to 2,400 steps in both seeds (55%); 4,800 is no better than 2,400 (60%); untaught prose gets worse with length (65%, as in 171).
Amendment 176a (17:20, before 175's result and before any 176 run; user: "the control bridge still seems to be pretty good though, and maybe with a better trained translator that one might still win... or a better trained bridge translator that one may be better"; "yes, widen 176 with both translators"): no gate, and the bridge translator gets the same curve. Seeds 2 and 3 each: control and bridge translators at 600 (the originals), 1,200, 2,400 and 4,800 steps, a LoRA from each, all scored in one session. Context: 174q seed 3 (early scores of the existing forms) reversed seed 2's order: control LoRA 28 solved (+12 new, 7 lost), bridge LoRA 35 (+18, 6), plain SFT 30 (+10, 3), against seed 2's 43, 37, 38; pooled over the two seeds 71, 72, 68 (Qwen alone 46): no form leads reliably, seed-to-seed swings reach 15 problems. Added predictions: at 2,400 steps the bridge LoRA beats the control LoRA pooled over both seeds (45%); the spread between seeds at a fixed length exceeds the change from 600 to 2,400 steps for at least one translator (60%).
Result, seed 2 control curve (19:20; 188 problems, one greedy attempt, the bridge down; Qwen alone 23): control LoRA from the 600-step translator 43 (+22 new, 2 lost), 1,200 steps 38 (+18, 3), 2,400 steps 40 (+19, 2), 4,800 steps 36 (+16, 3); vs 600 steps 7 vs 12, 10 vs 13, 5 vs 12 (p = 0.14). The translator's training-batch cross-entropy kept falling (1.11 → 0.61 → 0.45; the bridge's alike, 1.07 → 0.60 → 0.44) while the LoRAs found its targets harder to fit (KL left 0.044 → 0.065 → 0.095): longer training memorises the training rows, it does not give better stored forms. Seed 3 cancelled (user: "yes cancel 176, just leave it at 600, taht seems to be the sweet spot"); the 600-step translators stay the default. Bridge curve, seed 2: 37 (600 steps), 32 (1,200), 35 (2,400), 39 (4,800); 4,800 vs 600 13 vs 11 (p = 0.84); at equal length bridge vs control 4 vs 10, 5 vs 10, 11 vs 8. Both curves are flat within noise; 175's 1,200-step bridge LoRA scored 37 against this run's 32 (retrained translators add run-to-run noise). Predictions: solved rises from 600 to 2,400 in both seeds: no (seed 2, both translators); 4,800 no better than 2,400: control yes, bridge no (39 vs 35, ns); bridge beats control at 2,400 pooled: no (seed 2: 35 vs 40).
Correction (19:45, geval of the same LoRAs): on X1's held agent rows (new tasks and repos, the translators' own domain) the LoRAs improve steadily with translator length: bridge 0.716 / 0.686 / 0.664 / 0.651, control 0.728 / 0.701 / 0.675 / 0.660 at 600 / 1,200 / 2,400 / 4,800 steps (live bridge with the 600-step translator 0.679; dose-matched plain SFT 0.66 in 169d). Untaught prose erodes with length (bridge 3.42 / 3.47 / 3.57 / 3.65, control 3.43 / 3.49 / 3.58 / 3.66; Qwen alone 3.68); fresh questions flat. So longer translators are not merely memorising: they learn corrections that generalise within the agent domain, while losing the off-domain protection and not helping contest code. The 600-step choice holds for contest coding; for agentic training (the Kaggle target) longer translators may pay, at a cost to untaught text. E4B-specific part at any length: 0.009-0.015 nats (bridge vs control LoRA).

## Addendum 177 (2026-11-07, 17:10): design of an RSI pilot with the self-bridge as the update step (user: "this could, if it continues to work, be a way to do RSI loops?"; "yes, design the pilot now - I think self improvement in this case could come either from the internet, or a stronger models outputs")

Design only. It starts after tonight's decision on whether the translator matters (174q, 175, 174). If the translator adds nothing, the self-bridge arm is dropped and the pilot compares the remaining update rules.

**Question.** In a loop of attempt → verify → update, does the self-bridge update (a gated self-correction translator on the model's own states, fitted to verified winning text, then distilled densely into a LoRA) keep adding new solved problems round after round without losing old ones, where plain SFT on wins (ReST / expert iteration) plateaus or erodes?

**Setup (reuses addenda 172-174).**
- Student: Qwen3-1.7B. Pools: the 174 train pool (older LiveCodeBench, test1-4) for attempts and wins; the 188 held problems of 172 for new and kept, never trained on.
- Verifier: the hidden tests (xcode.py scorer).
- The model of round r is the round r-1 model with its LoRA merged and saved, then used as the student path. So the translator always reads the current model's own states.

**One round, per loop arm.**
1. **Attempt.** Greedy, plus 4 samples at T 0.8, on the easy and medium train-pool problems.
2. **Wins.** Per problem, the first passing attempt with its own closing fence. The win set depends on the source arm:
   - **S, self:** only the model's own wins.
   - **T, stronger model:** the model's own wins, plus E4B's verified solutions (174 build) for the problems it failed.
   - **I, internet (stage 2, needs the user's OK for downloads):** attempts made with retrieved reference text in the prompt, so the verified win carries outside knowledge; the update trains the model without that text (context distillation).
     - Retrieval index: offline, algorithm and library documentation only (cp-algorithms-style pages, the Python docs). It must never hold editorials or solutions to the problems, and must not use Wikipedia.
     - The retrieval query is written by the model from the problem statement.
     - Leakage check: no indexed page may contain a pool problem's id, title or sample I/O.
3. **Update.** The same number of training steps for every rule:
   - **U1 ReST:** SFT LoRA on the wins.
   - **U2 KL-SFT:** SFT plus 1.0 × KL to the previous round's model over the span.
   - **U3 self-bridge:**
     - Fit the control translator on the current model's own states toward the wins: cross-entropy, size penalty and do-no-harm; 600 steps plus the two clean rounds that 175 tests.
     - Then a rank-64 LoRA trained toward the corrected model (outputs-only graft objective, 1,000 steps).
   - **U4 in-context self-teacher (sdft):** only if 174 shows it is competitive.
4. **Measure** after the update (greedy):
   - **fixed:** train-pool problems that round 0 failed and the model now solves;
   - **new:** held problems that round 0 failed and the model now solves;
   - **kept:** problems round 0 solved that it still solves, held and train separately;
   - **off-task:** held agent replies, fresh questions, untaught prose;
   - **health:** pass@4 diversity on the train pool, stub and `pass` rate, mean length, and runs that hit the token cap without a closing fence.

**Arms, stage 1, one seed.**
- S×U1, S×U3, T×U1, T×U3: three rounds each.
- U2 is added if the budget allows.
- About 45-50 min per loop-round on a G4 (sampling about 25 min, scoring 5, update 10-15, evaluation 10). 12 loop-rounds, about 90 CU over 3 parallel sessions, about 3.5 h.
- If promising: a second seed, then stage 2 (I), then the rsi-loop harness on E4B with the 31B or DeepSeek as T.

**Decision rules (fixed now).**
- **The self-bridge is a better RSI update** if, at round 3, U3 beats U1 on new held problems under the same source, in a paired exact test (p < 0.1), while losing no more kept problems.
- **It compounds** if U3's new count rises from round 1 to round 3 in both sources.
- **Erosion:** kept falling by ≥ 3 problems in any round.

**Predictions.**
- U3 keeps gaining in rounds 2-3 while U1 is flat after round 1: 35%.
- U3 loses ≤ 2 kept problems through round 3 while U1 loses ≥ 4: 40%.
- T beats S on new problems by ≥ 5 at round 3 under both updates: 60%.
- S×U3 still gains in round 3 (pure self-improvement compounding): 25%.

**Amendment 177a (18:30, before any run).** User: "the RSI loop should also try to first translate offpolicy to onpolicy using the translators/self-bridge".
- **New update rule U5, the on-policy self-bridge.**
  - The control translator is fitted on the current model's own states toward the wins, as in U3.
  - The LoRA then trains on the model's own samples of the round: all K samples of every train-pool problem, cut at their own fence. The target is the corrected model on those tokens: per-token KL over the sampled reply, the first 3 tokens masked.
  - The wins, own or stronger-model, only ever fit the translator; nothing is copied. This is 179's opd_c inside the loop.
  - Default arms are now S|T × U1, U3, U5; E4B runs S:U1, S:U3, S:U5.
- **A rewrite rule (U6) is not added.** Sampling with the translator on cannot use the KV cache, which costs hours per round. The loop already does the cheap form of it: model r is the corrected model distilled, and its verified samples are the next round's wins.
- **Review fixes (code review 11-07; no correctness bug found):**
  - The small pass has 12 train problems and fails unless every arm merges a round-1 update.
  - Health gains a "no runnable body" share.
  - Adapters are saved in fp32, so each model can be rebuilt exactly.
  - Logs are named by student.
  - Summary rules are added for U5.
- **Defaults that differ from the design above:** 400 LoRA steps per round for every rule (not 1,000 for U3); no clean translator rounds (175 found none helped).
- **Review notes, kept for the reading:**
  - U3 learns little from its own greedy wins, because there the target is close to the model itself; most of its signal is in sample wins and stronger rows.
  - Fixed steps per round give S arms more epochs than T arms.
  - One seed of about 160 both-failed held problems will rarely reach p < 0.1.
  - E4B hits the 768-token cap on about half its greedy answers at round 0; that is its baseline, not a regression.
- **Small Colab test (19:30-19:47, passed).** Qwen ran S|T × U1, U3, U5 and E4B ran S:U1, S:U3, S:U5, each for 2 rounds, 12 train / 4 held problems, 4 steps per phase.
  - Wins: Qwen 2 own greedy, plus 7 stronger rows under T; E4B 1 greedy and 2 sample.
  - The translator learns on real states in 4 steps: Qwen S 0.19 → 0.15, T 1.13 → 0.88; E4B 0.155 → 0.143.
  - U5 trains on 23-24 own samples per round, and the wins only fit the translator.
  - Every arm merged round 1 and chained into round 2, and the summary ran. The counts move by at most one problem (no signal at this size). The pipeline works for both students.
- **Full pilot, stage 1 (queued 19:50 after 179).** User: "yes, queue the four-arm Qwen pilot after 179".
  - Qwen3-1.7B, seed 1, 3 rounds on the 517 easy and medium train-pool problems.
  - Session xqrsiS1 runs S:U1 (ReST) and S:U5 (on-policy self-bridge). Session xqrsiT1 runs T:U1 and T:U5 (own wins plus E4B's 174 rows, on-policy through the translator for U5).
  - Each finished arm is uploaded to HF xq/rsi/qwen_S and xq/rsi/qwen_T (colab_run -w), per the review.
  - U3 and E4B follow only if U5 shows anything.

## Addendum 180 (2026-11-07, 18:20; queued on the 4090 after the online test): resdec_g0's control and replication (user: "YEs, do this: Should I queue resdec_g0's control and replication seeds on the 4090 for after the online test?")

resdec_g0 fixed 27 of 52 gate decisions. It is the state pull along the decision basis toward the gall-bridged E4B states (E4B → E2B). That is the best of the series: res_g0 22, alldirect 20, alldirect at rank 64 21, resctl_g0 20. It has no control yet, and the 4090's libraries were synced after it ran (transformers 5.22.2, peft 0.21.2).
- **Runs:** seeds 1-3 under the synced libraries, seed 1 rerun too, each arm gated with per-point output (code6):
  - resdec: the decision-basis pull toward the bridged states;
  - resctldec: the same pull toward E2B's own control-translator states.
- **Then:** paired exact sign tests per seed and pooled (rep2_summary.py).
- **Script:** kaggle_g4agent/dreamer/resdec_4090.sh, about 8 h after ONLINE_RUN_DONE.
- **Predictions:**
  - The seed-1 rerun fixes 23-31 (60%).
  - resdec beats resctldec pooled with p < 0.1 (30%).
  - The 3-seed mean of resdec exceeds alldirect's 20 by ≥ 3 (40%).


**Result (2026-11-08, 06:13 on the 4090; resdec.log, rep2_summary.json).**
- **resdec replication:** 22, 25 and 19 of 52 gate decisions fixed in seeds 1-3 (mean 22), against the original run's 27. That is the level of res_g0 (22) and plain on-policy alldirect (20): the original 27 was a high draw.
- **The control resctldec never ran.** All three seeds failed at online.py's `assert not DEC or RES_SRC == "bridge"`: the decision-direction basis (addendum 164) is defined only for the bridge source, so the queued control was invalid as designed, and the resdec-vs-control comparison was not made.
- **Reading:** the decision-basis state pull does not beat the plain on-policy arms once replicated. Testing a control would need a decision basis computed from E2B's own control-translator states (not built).

## Addendum 178 (2026-11-07, 17:35; queued after 175): the same LoRAs on more problems (user: "self-bridge does seem slightly better though I think"; "yes, queue it after 175")

Over 174q's two seeds the forms sit within 7 problems pooled (control 71, bridge 72, plain SFT 68, SFT + KL to self 65, Qwen 46 of 376), and one seed moves a form by 15. The forms trained only on agent rows and questions, never on LiveCodeBench, so every easy and medium problem of every release is a fair test item. Seeds 2 and 3: the same five arms (Qwen alone, control LoRA, bridge LoRA, plain SFT seq_d1000, seqself) on test.jsonl-test4.jsonl easy and medium, pooled with their 174q results (greedy is deterministic: re-runs reproduced exactly); seed 1: the outputs-only control and bridge LoRAs and seqself trained first (169c and 174q recipes), then all six files. Pooled paired sign tests over seeds and problems. kaggle_g4agent/colab/xq_wide.sh, launched by kaggle_g4agent/dreamer/xq_side5_launch.sh. Predictions: pooled over three seeds, the control LoRA beats plain SFT with p < 0.1 (30%); the translator forms (control and bridge) gain more new problems and lose more solved ones than the translator-free forms (60%).

## Addendum 179 (2026-11-07, 17:50; built and CPU-tested, not yet run): on-policy training through the translators (user: "is there anyway we could make offpolicy training on policy using the translators?"; "yes, build the on-policy arms")

Off-policy: the student copies text it would not write (E4B's solutions). On-policy: the student trains on its own samples, with a teacher's judgement on every token. A different-tokenizer teacher cannot score the student's tokens, but the translator can: written into the student, it turns the teacher's states into a distribution in the student's own vocabulary, at every position of the student's own text. Two arms in transfer/xlearn.py, on 174's rows, order and dose (LoRA-64, 2 epochs), sampling as in sdft (T 1.0, at most 512 tokens): opd_b, where the student samples, E4B reads the sample (prefill only, aligned as in xbridge.encode), the bridge translator writes its states into the base student (LoRA off), and that bridged distribution on every sampled token is the target, per-token KL(target ‖ student), the first 3 tokens masked; opd_c, the same with the control translator (the student's own states), an on-policy self-bridge for which E4B's text was only ever used to fit the translator. The teachers are fixed (base + translator); sdft's teacher is an EMA. Run with sft as an in-session anchor (XL_ARMS=sft,opd_b,opd_c on xq_learn.sh STAGE=arms), seeds 1-3, after 174's sdft result. Predictions: opd_b beats opd_c on new held problems pooled over seeds (40%); the better on-policy arm keeps more than sft (65%); opd_b solves at least as many new problems as sft (35%).
Early seed-1 indicator for 174 (18:00, the four finished arms on the 188 held problems, Qwen alone 23): sft 48 (+31 new, 6 lost), sft_b 47 (+29, 5), sft_c 50 (+29, 2), kl_b 35 (+18, 6); sft_b vs sft_c 5 vs 8. Why the losses (regenerated, each failure classified): of 19 lost problems across the four arms, 7 are answers cut at the 768-token cap (sft, sft_b, sft_c: 2,700-3,200 characters where Qwen's own passing answer was 178-986), 7 pass when regenerated (noise), 5 are genuine logic errors (mostly kl_b). Qwen alone writes short code (median 174 characters), E4B long defensive code (median 2,173; the rows 1,475): the copying arms take on E4B's verbosity, the off-policy cost in single-shot form; kl_b keeps Qwen's short style.
Amendment 179a (18:15; user: "yes, add the shortest-correct rows fix too"): xlearn.py build_short gives every row problem 4 more E4B samples (T 0.8) and keeps, among all of E4B's passing, fenced attempts (the row, the build's greedy and samples, these), the shortest in Qwen tokens (xl_rows_short.jsonl; model-written only). Arms sft_s and sft_c_s = sft and sft_c on those rows, run with 179's arms. Prediction: sft_s loses fewer held problems Qwen solved than sft (60%) and solves at least as many new ones (55%).
Short rows built (20:15): about 4.2 passing E4B candidates per problem; 205 of 299 rows changed; median Qwen tokens 355 → 295; rows over 600 tokens 28 → 12 (over 2,400 characters 27 → 13). 179 started 20:16 (seeds 1-2; seed 3 at the next slot).

**174 interim, seeds 1-2 (19:20).** One evaluation pass per seed; Qwen alone solves 99 trained and 25 held in each.

Per arm, pooled over the two seeds:

| Arm | Held solved | New | Lost held | Fixed | Lost trained |
|---|---|---|---|---|---|
| sft | 99 | 58 | 9 | 255 | 26 |
| sft_c | 98 | 54 | 6 | 250 | 32 |
| sft_b | 95 | 57 | 12 | 234 | 21 |
| sdft | 84 | 50 | 16 | 211 | 62 |
| kl_b | 81 | 46 | 15 | 158 | 39 |

Per seed, held solved: sft 42 / 57, sft_c 50 / 48, sft_b 43 / 52, sdft 43 / 41, kl_b 37 / 44.

- The decisive pair sft_b vs sft_c, on held problems: 2 vs 9 in seed 1 and 12 vs 8 in seed 2, pooled 14 vs 17 (p ≈ 0.7). E4B's states add nothing.
- Batch composition alone moves one model by about ±6 held problems (the early test's 48 vs the full pass's 42 for seed-1 sft).

Reading so far:
- The verified text is the main signal: the copying arms fix about 120 per seed, kl_b 69-89.
- Copying costs some of Qwen's own solves.
- sft_c loses the fewest held problems (3 per seed) but more trained-pool ones than sft.
- sdft, the published in-context self-teacher, keeps worst: it lost 62 trained-pool solves, as its authors warned for small models.

Seed 3 (20:40), the first three arms. Held solved: sft 44, sft_b 51, sft_c 54. Lost held: 6, 6, 4. Fixed: 134, 122, 123.

Three seeds pooled, held problems (Qwen alone 25 per seed):

| Arm | Held solved | Lost held (of Qwen's 75) |
|---|---|---|
| sft_c | 152 | 10 |
| sft_b | 146 | 18 |
| sft | 143 | 15 |

Paired, pooled:
- sft_c vs sft: 35 vs 26 (p = 0.31)
- sft_c vs sft_b: 26 vs 20 (p = 0.46)
- sft_b vs sft: 34 vs 31 (p = 0.80)

The self-bridge leads in every seed or nearly so, on both solved and lost, but the edge (9 of 564) is inside the noise. E4B's states add nothing over plain copying.

Seed 3, last two arms (21:30). kl_b: held 37, new 17, lost held 5, fixed 51, lost trained 29. sdft collapsed: held 27 (only 2 above Qwen alone), lost held 14, fixed 72, lost trained 52.

All five arms, three seeds pooled, held problems:

| Arm | Held solved | Lost held (of 75) | Fixed (trained) | Lost trained (of 297) |
|---|---|---|---|---|
| sft_c | 152 | 10 | 373 | 41 |
| sft_b | 146 | 18 | 356 | 32 |
| sft | 143 | 15 | 389 | 41 |
| kl_b | 118 | 20 | 209 | 68 |
| sdft | 111 | 30 | 283 | 114 |

- The copied text carries most of the gain: kl_b, the bridge signal without copying, fixes about half as many problems as any copying arm.
- sdft is unstable at 1.7B. Its seeds are 43, 41 and 27 held. Each seed loses more trained-pool solves than the last (25, 37, 52).

**Amendment 179b (19:10; user: "yes, add the hybrid arm to 179").** Arm mix_c: every row gives both the cross-entropy on E4B's verified solution (the text) and opd_c's on-policy term on the student's own sample toward the control-corrected base, weight 1 each. It combines the strongest signal so far (the text) with the on-policy correction.

Prediction: mix_c fixes as many as sft (within 5 per seed) and loses fewer held and trained problems than sft pooled (50%).
Launch (19:15; user: "yes, I added more CU, go ahead without opd_b"): 179 runs sft, opd_c, mix_c, sft_s, sft_c_s for seeds 1-3 (opd_b dropped: E4B's states added nothing in any test), after the short-rows build, by kaggle_g4agent/dreamer/xq_side10_launch.sh (XL_ARMS on xq_learn.sh STAGE=arms).
## Addendum 181 (2026-11-07, 22:05; built, in review, not yet run): E4B's text through a refit translator, on-policy (user: "I thought we were going to use the qwen + translator + e4b text (off policy) and pass it through the translator to make it onpolicy?"; "yes, build opd_cf and the rewrite arm too")

179's opd_c had the on-policy half of the design, but its control translator was the seed's 600-step one, fitted on agent turns and questions, never on the 299 contest solutions that sft copies. Its teacher assigns E4B's solutions 0.71 nats per token (Qwen alone 1.03), and the opd_c student stopped there (0.70): on-policy distillation brought Qwen to its teacher, and the teacher had never seen the solutions. This addendum gives the teacher the text first. transfer/xlearn.py, xq_learn.sh STAGE=cf, per seed:

- **fit_cf:** the seed's control translator fine-tuned on the 299 rows (E4B's verified solutions): 600 clean steps (no noise or depth dropout) at lr 1e-4, warmup 30, cosine; loss as xbridge.train (cross-entropy on the reply + size + do-no-harm where Qwen alone is right).
- **Gate (CF_PASS):** on the rows, the refit beats the old control by ≥ 0.05 nats (5% bound > 0), and on E4B's 74 passing held solutions (never trained on) it beats Qwen alone (5% bound > 0). On CF_FAIL no arms run.
- **opd_cf:** opd_c with the refit translator (the student samples, the refit-corrected base scores every token, first 3 masked; same rows, order, LoRA and steps).
- **rw_cf (rewrite):** with the refit translator live, Qwen writes each row problem's solution: greedy + 4 samples (T 0.8), and 4 more samples where none passed. A row is the greedy rewrite if it passes every hidden test, else the first passing sample, cut at its own first fence; no row where none passed. Plain cross-entropy on those rows, steps matched to sft's 299 (the rows cycle). Generation is KV-cached (two passes per token: Qwen alone for the source states, Qwen with the translator for the logits), checked against xcode.gen_bridged on 2 problems.
- **Also measured:** the teachers' own greedy solve rates (old and refit control, live) on the row problems and the 188 held; how on-policy the rewrite rows are (Qwen alone's cross-entropy of the rewrite rows against E4B's rows, text similarity, length).
- **Evaluation:** in the same session as this seed's 179 adapters (sft, opd_c, mix_c, uploaded from the finished 179 runs; sft retrained if missing), so opd_cf vs opd_c and rw_cf vs sft are paired on the same problems.

Predictions (pooled over seeds 1-3):
- The gate passes in all three seeds (75%).
- The refit teacher solves more held problems greedily than the old control teacher (55%).
- Passing rewrites exist for ≥ 150 of the 299 row problems (60%); the rewrite rows have lower Qwen-alone cross-entropy than E4B's rows (median; 85%).
- opd_cf fixes more trained problems than opd_c (75%) and solves more held problems (50%).
- opd_cf solves held problems within 5 per seed of sft or better (45%) and loses fewer (held + trained) than sft (60%).
- rw_cf loses fewer (held + trained) than sft (60%) and solves held problems within 5 per seed of sft or better (50%).

Success bar for "off-policy text made on-policy through the translator works": an arm within 10 held solves of sft pooled over three seeds while losing at most two thirds of sft's lost problems (held + trained, pooled).

**Amendment 181a (22:15, before any 181 run; user: "yes, add mix_cf to 181").** 179's mix_c and opd_c both used the old translator; mix_c seed 1 solved 165 trained / 37 held (sft 210 / 44; its outputs are short and stub-like, Qwen's habits). Arm mix_cf = mix_c with the refit translator: E4B's text copied plus the on-policy term toward the refit-corrected base. Prediction: mix_cf solves more held problems than mix_c pooled (75%) and is within 5 held per seed of sft or better (45%).

**179 complete, three seeds (2026-11-08, 03:40).** Pooled over seeds 1-3. Fixed = trained problems Qwen alone failed (600 problem-runs), new = held problems Qwen alone failed (489), held solved of 564, lost = problems Qwen alone passed (held 75, train pool 309).

| arm | fixed | new | held solved | lost held | lost trained |
|---|---|---|---|---|---|
| sft | 389 | 90 | 149 | 16 | 42 |
| sft_s | 406 | 95 | 160 | 10 | 38 |
| sft_c_s | 385 | 91 | 157 | 9 | 43 |
| opd_c | 208 | 60 | 116 | 19 | 75 |
| mix_c | 246 | 59 | 117 | 17 | 98 |

Paired, pooled: opd_c vs sft held 28 vs 61 (p = 0.0006); mix_c vs sft held 25 vs 57 (p = 0.0005); sft_s vs sft held 43 vs 32 (p = 0.25), trained 112 vs 92 (p = 0.18). Seed 1's sft_s result (p = 0.05) keeps its direction but is not significant over three seeds. Off-task, 3-seed means (Qwen alone / sft / sft_s / sft_c_s / opd_c / mix_c): held agent CE 1.385 / 1.162 / 1.159 / 1.045 / 1.022 / 1.053; untaught prose 3.680 / 3.620 / 3.609 / 3.434 / 3.447 / 3.487. Predictions: the better on-policy arm keeps more than sft (65%): no (opd_c loses 19 + 75, sft 16 + 42). The opd_b predictions are untested (opd_b was not among the arms run). 179a: sft_s loses fewer held than sft (60%): yes (10 vs 16); solves at least as many new (55%): yes (95 vs 90).

**181 result (2026-11-08, 03:40; three seeds; each session also re-evaluated that seed's 179 adapters, and sft, opd_c and mix_c reproduced 179's per-seed counts exactly).**
- Gate: CF_PASS in all three seeds. On the 299 rows the refit scores 0.042-0.046 nats per token (old control 0.713-0.718, Qwen alone 1.025): it nearly reproduces the solutions it was fitted on. On E4B's 74 passing held solutions (never fitted): refit 0.480-0.491, old control 0.694-0.698, Qwen alone 1.000.
- Live teachers, greedy (translator on Qwen's own states, no E4B): held 151 of 564 (51 / 53 / 47), old control 117 (37 / 38 / 42), Qwen alone 69. Row problems 799 of 897 against old 446 and alone 285; on rows this is memorisation.
- Rewrite: passing rows for 293 / 289 / 289 of 299 (greedy 262-269). Median similarity to E4B's row 0.98-0.99, identical in 110-122. Qwen-alone CE median 0.95 vs 1.03 for E4B's rows; median length 332-336 vs 350-353 Qwen tokens. The rewrites are E4B's text with small edits, not Qwen-style solutions.

| arm | fixed | new | held solved | lost held | lost trained | held CE of E4B's solutions |
|---|---|---|---|---|---|---|
| sft | 389 | 90 | 149 | 16 | 42 | 0.439 |
| opd_c | 208 | 60 | 116 | 19 | 75 | 0.684 |
| mix_c | 246 | 59 | 117 | 17 | 98 | 0.433 |
| opd_cf | 359 | 89 | 148 | 16 | 59 | 0.439 |
| rw_cf | 405 | 81 | 142 | 14 | 41 | 0.497 |
| mix_cf | 399 | 95 | 158 | 12 | 40 | 0.424 |

Paired, pooled (a only vs b only): opd_cf vs opd_c held 56 vs 24 (p = 0.0005), trained 235 vs 65; mix_cf vs mix_c held 58 vs 17 (p = 2e-6), trained 263 vs 48; opd_cf vs sft held 38 vs 39 (p = 1), trained 89 vs 136 (p = 0.002); mix_cf vs sft held 41 vs 32 (p = 0.35), trained 95 vs 83 (p = 0.41); rw_cf vs sft held 33 vs 40 (p = 0.48), trained 100 vs 83 (p = 0.24). Off-task, 3-seed means (opd_cf / mix_cf / rw_cf; sft 1.162 / 3.620 / 0.557, Qwen alone 1.385 / 3.680 / 0.513): held agent CE 1.077 / 1.108 / 1.274; untaught prose 3.563 / 3.589 / 3.687; fresh-question p(true) 0.573 / 0.560 / 0.571.

Predictions: gate passes in all three seeds (75%): yes. Refit teacher solves more held greedily than the old (55%): yes (151 vs 117). Passing rewrites for ≥ 150 rows (60%): yes; rewrite rows lower Qwen-alone CE (85%): yes (0.95 vs 1.03). opd_cf fixes more trained than opd_c (75%): yes (359 vs 208); solves more held (50%): yes (148 vs 116). opd_cf within 5 held per seed of sft or better (45%): no (seed 2: 49 vs 55); loses fewer than sft (60%): no (75 vs 58). rw_cf loses fewer than sft (60%): yes, barely (55 vs 58); within 5 held per seed (50%): no (seed 2: 47 vs 55). 181a: mix_cf more held than mix_c (75%): yes (158 vs 117); within 5 held per seed of sft or better (45%): yes (+9, −5, +5). Success bar (within 10 held of sft and at most two thirds of sft's 58 losses, i.e. ≤ 38): not met (mix_cf 158 / 52, rw_cf 142 / 55, opd_cf 148 / 75).

Reading: the teacher-ceiling explanation holds. The same on-policy method with the refit teacher goes from 116 to 148 held (sft 149), and its CE on E4B's held solutions from 0.68 to 0.44 (sft's level). On-policy through the translator now matches copying, but it does not beat copying and does not keep more: opd_cf loses more trained problems than sft (59 vs 42). mix_cf has the best point estimates of the 181 arms (158 held, 52 lost), tied with 179's sft_s (160 / 48); neither separates from sft pairwise. The refit memorised its rows (0.04 nats), so on row problems the on-policy target and the rewrites are close to E4B's own solution: what 181 tests is a better teacher, not text that Qwen would write. A rewrite arm that tests on-policy text needs a teacher that has not memorised the problems (e.g. a refit fitted on half the rows, rewrites made on the other half).

**178 interim, seeds 2-3 (2026-11-08, 03:30; seed 1 not run: the Colab balance reached 0 before its session started).** 517 easy and medium problems per seed (test-test4). Passed, seed 2 / seed 3: Qwen alone 106 / 106; bridge LoRA 152 / 129; control LoRA 177 / 142; plain SFT (seq_d1000) 165 / 165; seqself 132 / 137. Paired, pooled: control vs plain SFT 82 vs 93 (p = 0.45); bridge vs plain SFT 64 vs 113 (p = 0.0003); control vs bridge 104 vs 66 (p = 0.004); seqself vs plain SFT 53 vs 114 (p = 3e-6). Prediction (control beats plain SFT with p < 0.1, 30%): not on two seeds (SFT ahead by 11). Reading: on 1,034 problem-runs plain SFT is at least as good as the control LoRA, and better than the bridge and seqself forms. The 174q impression that the self-bridge is slightly better does not hold.

**178 complete (2026-11-08, 05:50).** Seed 1 ran all six files (705 problems), seeds 2-3 test-test4 (517 each).

| Arm | Passed: seed 1 / 2 / 3 | Pooled (of 1,739) |
|---|---|---|
| Qwen alone | 130 / 106 / 106 | 342 |
| Bridge LoRA | 171 / 152 / 129 | 452 |
| Control LoRA | 164 / 177 / 142 | 483 |
| Plain SFT (seq_d1000) | 189 / 165 / 165 | 519 |
| seqself | 145 / 132 / 137 | 414 |

Paired, pooled:
- Plain SFT vs control 161 vs 125 (p = 0.04).
- Plain SFT vs bridge 181 vs 114 (p = 0.0001).
- Control vs bridge 144 vs 113 (p = 0.06).
- Plain SFT vs seqself 195 vs 90 (p = 5e-10).
- Every form beats Qwen alone (p ≤ 5e-7).

Prediction (the control beats plain SFT with p < 0.1, 30%): no; the reverse holds, with p = 0.04. The other prediction (translator forms gain more new and lose more solved) was not computed.

Reading: on wide contest coding, plain SFT on E4B's text at the same dose is the best of these forms. The translator-trained LoRAs teach less, and E4B's states (the bridge) teach less than Qwen's own (the control).

## Addendum 182 (2026-11-08, 03:50; built, in review, not yet run): the runbook arm (user: "also anything here help us https://arxiv.org/abs/2610.22192"; "yes, build the runbook arm")

Why: 181's rewrite arm could not test Qwen-written rows. The refit translator had memorised the 299 solutions (0.04 nats), so its rewrites were E4B's text (median similarity 0.98, 110-122 identical). Terminal-RSI (arXiv 2610.22192, a Qwen3.8-27B terminal agent) turns a verified trajectory into a runbook (a plan, with the solution removed). The same policy re-executes the task with the runbook, the verified re-runs are kept, and they are trained on without the runbook. Its first rewrite stage gives its largest single gain (four-suite mean 20.7 → 34.3). But it has one run and no ablation against plain SFT or ReST. Here the plan carries E4B's know-how and Qwen writes the code, so the rows are in Qwen's own style.

Design (transfer/xlearn.py `runbook` and `train rb`; xq_learn.sh STAGE=rbuild once, STAGE=rb per seed; dreamer/xq_side15_launch.sh):
- **Plans.** E4B (raw completion, as in the build) reads each row problem with its own verified solution (the sft row). It writes at most 8 numbered steps in plain English, no code, in ≤ 320 tokens. Leakage screen: ≥ 2 steps and 60 characters, no fence, no line that reads as code, and ≤ 10% of the solution's code-token 4-grams. Greedy first, then up to 2 samples (T 0.8) for plans that fail.
- **Re-solve.** Qwen alone gets the plan inserted before the closing instruction ("A plan for the solution: … following the plan"). It writes a greedy solution, then 4 samples (T 0.8) where that failed, then 4 more where none passed (182a). A row is the greedy solution if it passes every hidden test, else the first passing sample, cut at its own first fence. Rows are stored with the original prompt, with no plan. There is no row where no solution passed or no plan passed the screen.
- **Built once.** The rows do not depend on the seed (as with 179a's short rows). They are uploaded, and the seed sessions run, only if there are ≥ 100 rows (otherwise RB_FEW).
- **Arm rb.** Cross-entropy on the runbook rows. Rank-64 LoRA with sft's layers, lr, schedule and steps (299 rows × 2 epochs / ACC; the rows cycle). Seeds 1-3; each session has that seed's 179 sft adapter as its in-session anchor.
- **Pairing.** rb is paired with sft in-session. It is paired with sft_s (179) and with rw_cf, mix_cf and opd_cf (181) through their stored per-problem results. That is valid only if the session's sft reproduces its stored per-problem results exactly (181 reproduced 179 exactly); otherwise only rb vs sft is reported.
- **Also measured.**
  - Plans passing the screen.
  - Qwen's greedy solve rate with the plan against Qwen alone, and coverage where Qwen alone fails.
  - Qwen-alone CE of the rows against E4B's rows.
  - Similarity to E4B's row and to Qwen alone's own greedy.
  - Length against the E4B rows and the shortest rows.
- **Scope.** The rows contain no plan or hint text; the plan appears only in the generation prompt. This is the contest test, not Kaggle rows. Qwen3-1.7B is Apache-2.0, and Qwen trains on Qwen-written rows, as in 181's rw_cf.

Predictions (pooled over seeds 1-3 where relevant):
- Plans pass the screen for ≥ 280 of 299 problems (70%).
- Qwen greedy with the plan solves ≥ 150 of the 299 row problems; Qwen alone solves 99 (55%).
- There are runbook rows for ≥ 200 of 299 problems (55%) and for ≥ 100, the run threshold (85%).
- The rows are not E4B's text:
  - median similarity to E4B's row < 0.5 (75%)
  - Qwen-alone CE median ≤ 0.5, against 1.03 for E4B's rows (70%)
  - median length shorter than the shortest E4B rows (295 Qwen tokens) (70%)
- rb fixes fewer trained problems than sft (389): it has fewer rows, and problems Qwen cannot solve even with a plan have none (75%).
- rb loses fewer of the problems Qwen alone solved (held + trained) than sft's 58 (60%), and at most two thirds of them, ≤ 38 (30%).
- rb solves held problems within 10 of sft (149) or better (50%), and within 10 of sft_s (160) or better (40%).

Success bar (as in 181): within 10 held solves of sft, pooled, while losing at most two thirds of sft's lost problems (≤ 38).

Cost: one build session (~1.5 h) and three seed sessions (~1.5 h each), about 60 CU; 178 seed 1 (~15 CU) runs beside the build. Balance guard: no session starts below 20 CU.

**Amendment 182a (04:30, after an independent code review, before any run).** The review found no crash bugs; it found two data problems and some waste.
- **Screen.** 297 of 299 E4B rows have comments, so the 4-gram bound mostly measured English. In a simulation on the real rows, plans made of E4B's own comment lines passed 4 of 299 and plans with code lines inline passed 162. The 4-grams now come from the code with comments and strings removed. Inline code spans over 20 characters containing "=", "(" or "[" are rejected. Re-run: comment-style plans pass 276 of 299, inline code passes 2 of 299, unfenced code passes 8 of 299.
- **Plan parser.** It kept markdown lists only up to the first blank line before a bullet. Indented and bulleted lines after a blank line are now kept, and "**3.**" and "Step 4:" count as steps.
- **Generation order.** Re-solve round 1 is greedy only, then 4 samples where it failed, then 4 more. The selection rule is unchanged.
- **Yield probe.** Before the full build, plans are made for 24 problems at the full length. The build stops (RB_PROBE_FAIL) if fewer than 60% pass the screen.
- **Seed sessions** no longer download translators or unused adapters.
- **Launcher.** A session needs 20 CU for itself and for each session already running; otherwise it waits. The CLI call is time-limited. A failed start no longer leaves a phantom slot.
- **Stats.** Qwen alone is also counted on the planned subset.

**Build result (xqlRB0, 2026-11-08, 04:28; one session of about 20 minutes).**
- Plans pass the screen for 298 of 299 problems: 289 at the greedy attempt, 7 at the second, 2 at the third; one failed on code overlap. Probe: 23 of 24. Median plan: 8 steps, 850 characters, code-token 4-gram overlap 0.0.
- With the plan, Qwen greedy solves 166 of the 299 row problems; alone it solves 95. Where Qwen alone fails, it solves 87 of 204.
- Rows: 228 (166 greedy, 62 samples), covering 137 of the 204 problems Qwen alone fails.
- The rows are Qwen's text, not E4B's:
  - median similarity to E4B's row 0.20, none identical
  - Qwen-alone CE median 0.32 against 1.05 for E4B's rows
  - median length 110 Qwen tokens, against 350 for E4B's rows and 282 for the shortest E4B rows
  - on problems Qwen alone already passes, similarity to its own greedy is 0.58 (2 identical)
- Predictions:
  - plans ≥ 280: yes
  - Qwen with the plan ≥ 150: yes
  - rows ≥ 200: yes; ≥ 100: yes
  - similarity < 0.5: yes
  - CE ≤ 0.5: yes
  - shorter than the shortest E4B rows: yes
- Uploaded to HF xq/learn/. The arm rb seed sessions xqlRB1-3 follow.

**Result (2026-11-08, 05:51; three seeds).** In every seed, the in-session sft and Qwen reproduced their stored per-problem results exactly, so rb is paired with the 179/181 arms (transfer/rb_pair.py).

| Arm | Fixed | New | Held solved | Lost held | Lost trained |
|---|---|---|---|---|---|
| sft | 389 | 90 | 149 | 16 | 42 |
| rb | 421 | 83 | 143 | 15 | 24 |
| sft_s | 406 | 95 | 160 | 10 | 38 |
| sft_c_s | 385 | 91 | 157 | 9 | 43 |
| mix_cf | 399 | 95 | 158 | 12 | 40 |
| rw_cf | 405 | 81 | 142 | 14 | 41 |

Paired, rb only vs other only:
- vs sft: trained 133 vs 84 (p = 0.001), held 37 vs 43 (p = 0.58).
- vs sft_s: trained 109 vs 80 (p = 0.04), held 33 vs 50 (p = 0.08).
- vs mix_cf: trained 117 vs 80 (p = 0.01), held 33 vs 48 (p = 0.12).

Off-task, per seed (rb; sft; Qwen alone):
- Untaught human prose: 3.78 / 3.84 / 3.84; sft 3.61-3.63; Qwen alone 3.68.
- Fresh-question p(true): 0.50 / 0.53 / 0.49; sft 0.55-0.57; Qwen alone 0.513.
- E4B's held agent replies: 1.47-1.50; sft 1.16; Qwen alone 1.385. This one measures closeness to E4B, not forgetting.

Predictions:
- rb fixes fewer trained problems than sft: no (421 vs 389, the most of any arm).
- rb loses fewer than sft: yes (39 vs 58).
- rb loses at most two thirds of sft's (≤ 38): no, by one (39).
- rb's held solved within 10 of sft or better: yes (143 vs 149).
- rb's held solved within 10 of sft_s or better: no (143 vs 160).

Success bar: within 10 held of sft (yes, −6) and ≤ 38 lost (39): not met, by one problem.

Reading:
- Qwen-written rows carrying E4B's plan are the best teacher for the problems trained on: the most fixed and the fewest trained lost. Outputs are short; median 510 characters, half sft's truncations.
- They generalise less (new 83, against 90-95) and change Qwen's general text model for the worse. It is the only arm whose untaught-prose CE rises above Qwen alone's, in every seed, and its question accuracy stays at base.
- Training on its own short, confident text (0.32 nats) sharpens the model. 184b's rb_c tests whether the self-bridge KL removes that drift.

## Addendum 183 (2026-11-08, 04:25; queued on the 4090 after resdec, not yet run): online arm D, bridge teacher + keep + replay (user: "yes, queue the bridge keep-and-replay arm after resdec")

Why: in 170 the bridge teacher (B) reached the lowest CE to E4B (held 0.754) and drifted less than direct, but it fixed 16 of 52 gate decisions. Keep + replay (C) cut drift on untaught rows 6.6× (0.014 vs 0.089) and fixed 14. D combines them: does the bridge's CE survive the keep term, with drift at C's level?

Design (kaggle_g4agent/dreamer/online_bkr_4090.sh): 170's code (code5), stream, settings and gate. ON_ONL_TEACH=bridge, ON_ONL_KEEP=1.0, ON_ONL_REPLAY=4. A small pass first (bridge + keep + replay every 2nd item). It starts after RESDEC_DONE and a free GPU (twice). The RSI pilot (rsi4090_run.sh) now waits for ONLINE_BKR_DONE, so the GPU is never shared. One seed, as in 170, so differences under ~3 gate decisions or ~0.01 nats are not readable.

Predictions:
- D's held CE at item 600 is within 0.01 of B's (≤ 0.764) (55%).
- D's KL from base on untaught rows ≤ 0.02 (C 0.014, B 0.055) (65%).
- D fixes ≥ 16 of 52 gate decisions, B's level (35%), and ≥ 14, C's level (55%).
- D's gate CE ≤ 0.770 (B 0.761, C 0.783) (55%).
- D's KL on held questions, which the keep term never touches, is within ±0.015 of B's 0.043 (65%).

## Addendum 184 (2026-11-08, 05:10; built, in review, not yet run; queued on the 4090 after the RSI pilot): a selective translator and row-level do-no-harm (user: "can we also try creating a translator that was trained on the coding problems this agent is likely to lose? or how would you improve the translator to maximize learning and minimize forgetting?"; "que the translator training on the 4090")

Why (from 179 and 181, 24 runs: 8 arms × 3 seeds):
- Of the 128 problems Qwen alone solves, 37 are never lost, 37 are lost in ≥ 6 runs and 15 in ≥ 12. Within one arm, 3-6 are lost in all three seeds and 15-21 in only one.
- 25 of the 37 fragile problems are trained rows: Qwen was right, its answer was replaced by E4B's in training, and it was lost.
- Lost outputs have a median of 1,456 characters, against 535 for Qwen's own passing answers; 30% have no closing fence (cut at 768 tokens).
- 181's refit translator memorised its rows (0.04 nats): as a KL target it was E4B's text, and it generalised to unseen solutions at 0.48.

Design (transfer/xlearn.py `build_keep`, `fit_xk`, arms sft_k, sft_ck, xk; dreamer/x184_prep_4090.sh, dreamer/x184_4090.sh):
- **Keep rows** (no model). On the 95 row problems the build's Qwen run solved, the row is Qwen's own passing greedy solution (median 355 characters). The other 204 keep the shortest E4B row (median 1,290). The same 299 problems, prompts and dose as sft_s.
- **Cross-fitted selective translator** (per seed). The row problems are split in two folds, stratified. For each fold, the seed's control translator is fine-tuned on that fold only, with 600 clean steps at lr 1e-4 (as fit_cf).
  - Learn rows (E4B's shortest solution where Qwen fails): cross-entropy + size + do-no-harm.
  - Keep rows (Qwen's own solution): KL(Qwen alone ‖ corrected) over the whole reply + size + 0.1 × the mean gate.
  - The fragile train problems are oversampled 3×. These are the 29 Qwen solved that were lost in ≥ 6 of the 24 runs (x184_fragile.json), 21 of them with keep rows. The 8 fragile held problems are only reported, never trained on.
  - Each row's target comes from the translator fitted on the other fold, so the target never memorised that row.
  - Gate, on the unseen fold. XK_PASS: it beats the old control on learn rows by ≥ 0.05 nats (5% bound > 0), and on keep rows its KL from Qwen alone exceeds the old control's by less than 0.02 (95% bound). Arm xk runs only on XK_PASS.
- **Arms** (rank-64 LoRA, sft's layers, lr, schedule and steps; the same order per seed):
  - sft_s and sft_c_s: anchors, retrained on the 4090.
  - sft_k: CE on the keep rows.
  - sft_ck: sft_k + KL to the old control translator (sft_c_s on keep rows).
  - xk: sft_k + KL to the cross-fit translator.
- **Evaluation** as 174-182: fixed / new / kept, off-task, plus the fragile set's losses (train pool and held). Seeds 1-3.
- **Machine.** The 4090 (24 GB): Qwen3-1.7B with no E4B weights (XB_TEACHER_TOK_ONLY, the tokenizer and config only). The training files hold only the problems the evaluation reads (the row problems and the train-pool problems Qwen passed, every difficulty). Every hidden test runs in the bubblewrap sandbox. Numbers from this machine are compared with each other only; the sft_s anchor's per-problem agreement with its Colab results is reported.

Predictions (pooled over seeds 1-3):
- XK_PASS in all three seeds (55%).
- sft_k loses fewer of the problems Qwen solved (held + train pool) than sft_s (65%), and at most half of sft_s's losses on the fragile train set (55%).
- sft_k fixes within 15 of sft_s (70%) and solves held problems within 10 of sft_s (65%).
- sft_ck has lower untaught-prose CE than sft_k in every seed (85%), with losses within ±5 of sft_k (60%).
- xk solves ≥ 5 more held problems than sft_ck (30%), loses ≥ 5 fewer (30%), and loses fewer of the fragile held problems (25%).

Success bars:
- Row-level do-no-harm works: sft_k loses ≥ 10 fewer than sft_s, pooled, with held solved within 5.
- The selective translator adds protection: xk loses ≥ 5 fewer than sft_ck, pooled, with held solved within 5.

**Amendment 184a (05:40, after an independent code review, before any run).** The review found no crash bugs in the Python, which it checked on CPU.
- **Checks that passed.**
  - The teacher's config gives 42 layers (sources [20, 27], as in the Colab logs). No 184 path touches the teacher weights.
  - The folds give every row an unseen teacher (keep 48 / 47 and learn 102 / 102 per seed), with no leakage.
  - Only the fragile "train" list influences training.
  - The prep's held check matches load()/pools().
  - The filtered train files hold exactly the 307 problems evaluate reads (299 trained + 8 untrained that Qwen passed).
  - Estimated GPU peak is 12-14 GB (longest span 1,442 tokens).
- **Fixed.**
  - The code and the fragile file are now copied to the 4090 and checked by the prep and the runner.
  - The runner also stops waiting if the RSI pilot's runner ends without its end marker.
  - Disk: the cross-fit translators and the seeds' translator copies are deleted after use, and the room check is 2 GB per seed.
  - A failed arm no longer cancels the seed's evaluation.
  - The prep log starts fresh.
  - Peak GPU memory is logged per arm.
- **Reading caveat.** For the keep arms, "kept, train pool" and the fragile train losses are in-sample: they train on Qwen's own passing text for 95 of the 103 problems Qwen solves. That is the intended effect of row-level do-no-harm, but whether protection generalises is read on held new / kept and off-task. Only 6 fragile held problems are Qwen-passed, too few to rank arms.
- **Time.** About 4.5 h at Colab speed for the three seeds, likely longer on the 4090.

**Amendment 184b (05:50, before any 184 run; user: "yes, add rb_c to 184").**

182's seeds 1-2 (Colab, pooled):
- **Coding.** rb fixes the most trained problems (274; sft 254, sft_s 264; rb vs sft 88 vs 60, p = 0.03) and loses the fewest trained ones (18; others 27-44). Held solved 96 (sft 99, sft_s 108; rb vs sft_s 23 vs 35, p = 0.15). Its outputs are short (median 511 characters, sft 1,610-1,791) and fewer are cut at the token cap (59-61 of 495, sft 121-148).
- **Untaught human prose** (novel passages, not model-written): rb is worse than Qwen alone (3.78-3.84 against 3.68). Every arm trained on E4B's text is better (sft 3.61, sft_c_s 3.43).
- **Fresh questions.** p(true) stays at base (0.50-0.52) where sft gains (0.55-0.57).
- **Reading.** Training on Qwen's own short, confident text (0.32 nats) sharpens its distribution.

Arms added to 184:
- **rb**, retrained on the 4090 as the anchor.
- **rb_c**: sft_c on the runbook rows (CE + the old control translator's KL over the span, as sft_c_s).

The arm list per seed is now sft_s, sft_c_s, sft_k, sft_ck, xk (on XK_PASS), rb, rb_c. Each seed's adapter weights are deleted after its evaluation, for disk; they are seeded and re-trainable in minutes.

Predictions (pooled over seeds 1-3, on the 4090):
- rb_c's untaught-prose CE is below rb's in every seed (85%) and below Qwen alone's (60%).
- rb_c's fresh-question p(true) is ≥ rb's + 0.02 (55%).
- rb_c fixes within 15 of rb (65%) and solves at least as many held problems as rb (55%).
- rb_c loses no more (held + trained) than rb (50%).
- rb reproduces its Colab direction against sft_s: more fixed, fewer lost trained (70%).

**Amendment 184c (2026-11-08, 06:05; user: "use the remaining CU to run rb_c on colab").**
- **Setup.** rb_c also runs on Colab G4, the machine of every stored arm, before the 4090 run, so it pairs directly with the stored rb, sft_s, sft_c_s and mix_cf. Seeds 1-3, one session each: xq_learn.sh STAGE=rb XL_ARMS=rb_c XL_TAG=rbc, with the seed's control translator and 179's sft as the in-session anchor. Launcher dreamer/xq_side16_launch.sh (balance guard 20 CU per session); 62 CU available, about 20 expected.
- **Pairing.** transfer/rb_pair.py RUNS rb_c xqlRBC rbc. Cross-session pairs are used only where sft and Qwen reproduce their stored per-problem results.
- **Predictions.** As 184b's, now on Colab against the stored rb.
- **Also.** rb_c meets 182's success bar: within 10 held solves of sft (149) and ≤ 38 lost (35%).
- **4090.** The 4090's rb and rb_c in 184 stay as a same-machine replication.

**184c result (2026-11-08, 06:45; Colab, three seeds).** In every seed, sft and Qwen reproduced their stored per-problem results (rb_pair.py rb_c xqlRBC rbc).

| Arm | Fixed | New | Held solved | Lost held | Lost train pool |
|---|---|---|---|---|---|
| rb_c | 382 | 71 | 133 | 13 | 24 |
| rb | 421 | 83 | 143 | 15 | 24 |
| sft | 389 | 90 | 149 | 16 | 42 |
| sft_s | 406 | 95 | 160 | 10 | 38 |

Paired, rb_c only vs other only:
- vs rb: trained 29 vs 71 (p = 3e-5; per seed 10-26, 12-20, 7-25), held 24 vs 34 (p = 0.24).
- vs sft_s: held 23 vs 50 (p = 0.002; per seed 5-15, 11-17, 7-18).

Untaught text, per seed:
- Untaught prose CE: 3.43 / 3.43 / 3.45 (rb 3.78-3.84, Qwen alone 3.68).
- Fresh-question p(true): 0.526 / 0.553 / 0.519, which is rb's + 0.026-0.028 in every seed.

Predictions:
- Prose below rb in every seed: yes.
- Below Qwen alone: yes.
- p(true) ≥ rb + 0.02: yes.
- Fixed within 15 of rb: no (−39).
- Held ≥ rb: no (133 vs 143).
- Lost ≤ rb: yes (37 vs 39).

Success bar: not met (held 16 below sft).

**Drift from the original student (review item D1, 2026-11-08, papers/kaggle_bridge/scripts/drift.py).** Measured on 124 held untaught rows (43 prose, 57 code, 24 story), which no arm and no corrector trained on: KL(original ‖ trained) and the change in CE over the reply tokens, three seeds each. Values are means over seeds, prose / code / story.

| Arm | KL | ΔCE |
|---|---|---|
| sft | 0.074 / 0.096 / 0.107 | −0.06 / −0.08 / −0.10 |
| sft_s | 0.061 / 0.086 / 0.095 | −0.07 / −0.08 / −0.11 |
| sft_c_s | 0.089 / 0.076 / 0.142 | −0.24 / −0.21 / −0.51 |
| opd_c | 0.082 / 0.075 / 0.118 | −0.23 / −0.23 / −0.44 |
| rb | 0.022 / 0.029 / 0.034 | +0.14 / +0.10 / +0.14 |
| rb_c | 0.088 / 0.048 / 0.149 | −0.24 / −0.19 / −0.54 |

Reading:
- The corrector anchor does not keep the student closer to its original. Its KL is as large as SFT's or larger, except on code.
- It moves the student toward much lower loss on untaught text. That improvement comes through the corrector's soft targets on the code rows alone; the corrector itself was fitted on agent rows with prose, code, stories and questions.
- rb barely moves the student (smallest KL) yet raises its loss in every seed and every kind. Training on its own confident text sharpens it.
- rb_c removes that damage and improves on base, at a cost of 39 fixed problems against rb.
- On solved coding problems, the anchor loses no fewer than shortest rows alone (sft_c_s vs sft_s held 34 vs 37).

**Entropy (review item, 2026-11-08, 07:10).** drift.py now records the mean predictive entropy. Change from the original student on the same untaught rows, prose / code / story (original 2.07 / 0.83 / 1.83 nats), three-seed means:

| Arm | Entropy change |
|---|---|
| rb | −0.23 / −0.12 / −0.22 |
| rw_cf | +0.06 / +0.06 / +0.06 |
| sft_s | +0.25 / +0.18 / +0.25 |
| sft | +0.29 / +0.19 / +0.27 |
| mix_cf | +0.32 / +0.23 / +0.35 |
| opd_cf | +0.41 / +0.29 / +0.43 |
| mix_c | +0.55 / +0.38 / +0.61 |
| rb_c | +0.64 / +0.29 / +0.72 |
| sft_c_s | +0.66 / +0.38 / +0.72 |
| opd_c | +0.66 / +0.43 / +0.71 |

- rb is the only arm that makes the student more confident on text it never trained on: sharpening, now measured.
- Every arm trained on E4B's text or toward the corrector softens it. The corrector-target arms soften it most, and they also lower its loss most (better calibrated).
- The anchor turns rb's −0.23 into +0.64.

## Addendum 185 (2026-11-08, 07:10; built, not yet run): SFT + KL to the original student on the shortest rows (user: "yes, run the SFT + KL-to-original baseline on colab")

Why:
- The paper's review found the coding study's simplest missing baseline. sft_c_s anchors toward the corrected student; nothing anchors toward the original one.
- 184c and D1 show the corrector anchor moves the student as far as SFT, but toward lower loss on untaught text, and reverses rb's damage.
- A pull toward the original is the textbook keep term (Learning without Forgetting). It should keep the student close without the corrector's general-text gain.

Design:
- Arm sft_self_s: the shortest rows (as sft_s and sft_c_s), the same dose and order, + 1.0 × KL(original Qwen ‖ student) over the written span. No translator.
- Colab G4, seeds 1-3, one session each: xq_learn.sh STAGE=rb XL_ARMS=sft_self_s XL_TAG=sfself, with 179's sft as the in-session anchor.
- Pairing: rb_pair.py RUNS sft_self_s xqlSS sfself. Drift from the original: papers/kaggle_bridge/scripts/drift.py.
- Launcher: dreamer/xq_side17_launch.sh, MIN_CU 12 per session (42 CU available, about 20 expected).

Predictions (pooled over seeds 1-3):
- On untaught prose, sft_self_s moves the student less than both sft_s and sft_c_s: lower KL from the original (75%).
- sft_self_s improves untaught prose CE less than sft_c_s does: ΔCE above −0.15 (85%).
- sft_self_s fixes fewer trained problems than sft_s (65%).
- sft_self_s solves held problems within 10 of sft_s (60%), and fewer than sft_c_s (55%).
- sft_self_s loses no more (held + train pool) than sft_s (50%).

**Result (2026-11-08, 07:55; Colab, three seeds).** In every seed, sft and Qwen reproduced their stored per-problem results.

| Arm | Fixed | New | Held solved | Lost held | Lost train pool |
|---|---|---|---|---|---|
| sft_self_s | 382 | 95 | 159 | 11 | 29 |
| sft_s | 406 | 95 | 160 | 10 | 38 |
| sft_c_s | 385 | 91 | 157 | 9 | 43 |
| sft | 389 | 90 | 149 | 16 | 42 |

Paired, sft_self_s only vs other only:
- vs sft_s: held 34 vs 35 (p = 1), trained 89 vs 107 (p = 0.23).
- vs sft_c_s: held 32 vs 30.
- vs sft: held 41 vs 31 (p = 0.29).

Drift from the original Qwen (drift.py), prose / code / story:
- KL 0.010 / 0.030 / 0.016 (sft_s 0.061 / 0.086 / 0.095, sft_c_s 0.089 / 0.076 / 0.142).
- ΔCE +0.01 / −0.10 / −0.02.
- Entropy change −0.07 / +0.13 / −0.03.

Off-task: fresh-question p(true) 0.514 / 0.520 / 0.530 (base 0.513, sft 0.557); untaught prose CE 3.69 (base 3.68).

Predictions:
- Moves less than sft_s and sft_c_s: yes (6-9× lower KL on prose).
- Prose ΔCE above −0.15: yes (+0.01).
- Fixes fewer than sft_s: yes (382 vs 406, n.s.).
- Held within 10 of sft_s: yes (159 vs 160).
- Fewer held than sft_c_s: no (159 vs 157).
- Lost ≤ sft_s: yes (40 vs 48).

5 of 6 held.

Reading: the plain pull toward the original student is what keeps the student where it was. It leaves untaught text almost unchanged and loses the fewest of the copying recipes, with teaching equal to shortest rows. The corrector anchor instead moves the student as far as SFT, toward lower loss (its own fit); it is a way to add general-text improvement, not to stay put.

## Addendum 186 (2026-11-08, 07:50; built, in review, not yet run): the coding recipes with Gemma 4 E2B as the student (user: "replicate with e2b"; the paper review's top request: the main evidence was Qwen3-1.7B, and the only Gemma-to-Gemma result was one online seed)

Design:
- **Arms:** sft, sft_s and rb with Gemma 4 E2B (QAT checkpoint, revision 971342c, dequantised to bf16) as the student. The same 299 E4B rows, the shortest rows and the measures as 179-182.
- **rb rows:** E2B re-solves each problem with the same accepted E4B plans as the Qwen build (XL_RB_PLANS; no new plans). The rows are E2B's own passing solutions, trained without the plan.
- **LoRA:** rank 64 on the MLP projections from 35% depth to the top. The same relative depths as Qwen's layers 9-27.
- **No E4B weights:** XB_TEACHER_TOK_ONLY. The student's BOS is in every tokenisation and generation (XB_SBOS="<bos>"; Gemma's raw completions need it).
- **Kept pool:** XL_NO_KEPT_POOL. E2B has no build run on the untrained train-pool problems, so "kept" counts trained and held problems only.
- **Drift:** drift from the original E2B on the same held untaught prose / code / story rows, in-session (drift.py).
- **Run:** seeds 1-3, one Colab session each (xq_learn.sh STAGE=e2b; dreamer/xq_side18_launch.sh), about 45 CU (the user added 100 CU). Amendment 186a (after code review, before any run): the stage requires XB_SEED; the small pass falls back to E4B rows if E2B solves none of its 3 problems; drift failures are not fatal; xl_rb.json's "alone" fields refer to the build's Qwen run, so E2B's own alone results come from the in-session evaluation (key "qwen" = the student alone); E2B's sampling uses its own generation config (top-k 64; Qwen's is 20).
- **Code changes:** _lora_targets for the Gemma layer path, BOS handling in encode / align / generation, plan reuse in runbook. All are no-ops for the Qwen runs unless the new variables are set.

Predictions (three seeds pooled; per seed read as direction only):
- E2B alone solves at least 25 of the 188 held problems, as many as Qwen alone (60%).
- With E4B's plans, E2B solves more row problems greedily than alone (90%), and there are ≥ 150 E2B runbook rows (55%).
- The Qwen ordering repeats on E2B: sft_s solves at least as many held problems as sft − 3 (65%); rb fixes the most trained problems of the three (60%); rb raises untaught-prose CE above the original E2B (55%); sft lowers it (60%).

**Result (2026-11-08, 09:21; Colab, three seeds).** E2B alone solves 148 of the 299 row problems and 57 of 188 held-out problems (Qwen: 99 and 25). The runbook build gave 280 rows; with the plans, E2B's greedy answer solved 236 row problems. E2B's own loss on E4B's rows is 0.27 nats, against 0.38 on its own plan-guided rows (Qwen: 1.05 and 0.32).

| Arm | Fixed (of 453) | New (of 393) | Held solved (of 564) | Lost held (of 171) | Lost trained (of 444) |
|---|---|---|---|---|---|
| E2B alone | — | — | 171 | — | — |
| sft | 359 | 62 | 204 | 29 | 18 |
| sft_s | 369 | 60 | 204 | 27 | 20 |
| rb | 318 | 61 | 201 | 31 | 27 |

Paired:
- sft vs E2B alone: held 62 vs 29 (p = 0.0007).
- sft_s vs sft: held 43 vs 43.
- rb vs sft: trained 61 vs 111 (p = 0.0002), held 44 vs 47.

Drift from the original E2B, means over seeds, prose / code / story:

| Arm | KL | ΔCE | Entropy change |
|---|---|---|---|
| sft | 0.29 / 0.20 / 0.24 | +0.69 / +0.38 / +0.84 | −0.76 / −0.36 / −0.62 |
| sft_s | 0.29 / 0.20 / 0.23 | +0.57 / +0.32 / +0.71 | −0.70 / −0.33 / −0.55 |
| rb | 0.29 / 0.23 / 0.24 | +0.85 / +0.60 / +0.99 | −0.82 / −0.41 / −0.66 |

Off-task, per seed:
- Untaught prose CE rises from 4.47 to 5.17-5.19 (sft), 5.02-5.09 (sft_s) and 5.23-5.46 (rb).
- CE on E4B's held agent replies rises from 0.685 to 0.75 (sft) and 0.81-0.83 (rb): the student moves away from the teacher's agent text.
- Fresh-question p(true) is about unchanged (0.57-0.60, base 0.582).

Predictions:
- E2B alone ≥ 25 held: yes (57).
- More row problems with the plans: yes (236 vs 148).
- ≥ 150 rb rows: yes (280).
- sft_s ≥ sft − 3 held: yes (204 vs 204).
- rb fixes the most: no (318 vs 359 / 369).
- rb raises prose CE: yes.
- sft lowers it: no (+0.69).

5 of 7 held.

Reading:
- **Teaching.** With a Gemma student, the teacher's text is already close to the student's own (0.27 nats). Rewriting in the student's words adds nothing: rb fixes fewer, and shortest rows tie SFT. Every recipe gains about 11 held problems per seed.
- **Forgetting.** Every recipe moves E2B 4× further on untaught text than it moved Qwen, sharpens it (entropy −0.3 to −0.8), and raises its loss by 0.3-1.0 nats. With the same LoRA recipe, forgetting is a much larger problem in the Gemma student. 187's anchors (KL to the original E2B, the self-corrector and the cross-model corrector) test what reverses it.

## Addendum 187 (2026-11-08, 08:45; built, in review, not yet run): two-way E4B ↔ E2B with freshly fitted correctors (user: "for e2b and e4b"; chose "Two-way E4B ↔ E2B"; "I think we should replecate all of that, yeah, including the 600 trained correctors for both")

Why: the corrector's only clear value so far is live. The user's research direction is two-way composition: a larger model and a smaller one each give the other knowledge it lacks. So far the corrector question has only been asked across tokenizers (E4B → Qwen) and online (E4B → E2B, one seed).

Design: xq_learn.sh STAGE=two, xlearn.py build_from_gens and arm sft_b_s, launcher dreamer/xq_side19_launch.sh. Six Colab sessions, seeds 1-3 per direction. Each session fits its own 600-step self-corrector and cross-model corrector with xbridge.py train, using the Qwen correctors' recipe: size 0.03, harm 0.5, lr 3e-4, the same row mix. The correctors are kept.

**Direction 1, E4B teaches E2B (XL_DIR=e2b).**
- Correctors: E2B's self-corrector, and the cross-model corrector from E4B's states into E2B (same tokenizer, so the alignment is exact).
- Arms on the shortest E4B rows: sft_self_s (KL to the original E2B), sft_c_s (self-corrector anchor), sft_b_s (cross-model anchor).
- Pairing: with 186's sft, sft_s and rb through stored per-problem results. 186's sft adapter is the in-session anchor.

**Direction 2, E2B teaches E4B (XL_DIR=e4b).**
- Rows: E2B's own greedy passing answers on the 299 row problems (186 seed 1, the student alone; deterministic).
- Correctors: E4B's self-corrector, and the cross-model corrector from E2B's states into E4B.
- Arms: sft (on E2B's answers), sft_self (KL to the original E4B), sft_c (self-corrector anchor), sft_b (cross-model anchor).
- Baseline: E4B alone, in-session.

Both directions use the same measures: fixed, new, held solved, lost (trained + held), off-task, and drift from the original student. E4B's fixed counts are small by construction (row problems that E2B solves and E4B fails).

Predictions (pooled over seeds 1-3):
- Direction 1:
  - sft_b_s solves held problems within ±10 of sft_c_s (70%).
  - sft_self_s has lower prose KL from the original E2B than sft_s and sft_c_s (80%).
  - sft_c_s and sft_b_s lower untaught-prose CE more than sft_s (65%).
- Direction 2:
  - There are fewer than 200 E2B rows (80%).
  - E4B sft solves at most as many held problems as E4B alone (60%).
  - sft_b solves no more held problems than sft_c (75%).
  - Every arm fixes at most 5 problems per seed (70%).

Success bar for the two-way idea: in direction 2, sft_b beats sft_c by ≥ 5 held solved pooled, with no more losses (15%).

Budget: about 25 CU per E4B session and 15 per E2B session, about 120 CU. The balance guard starts a session only while the balance covers 20 CU for it and for each running session.

**Amendment 187a (09:00, after code review, before any run).** The review found no crash bugs. It dry-ran every new step on CPU with tiny random-weight Gemma 4 models of the real structure (shared KV layers, per-layer embeddings, the real tokenizer and rows).

Confirmed:
- Gemma 4 decoder layers return a plain [B, L, D] tensor, so the corrector hooks are correct.
- E2B is 35 layers of width 1536 (write layer 12, sites 13-34). E4B is 42 layers of width 2560 (write layer 14, sites 15-41).

Changes:
- `encode` uses the identity alignment when both tokenizations are equal.
- E2B's rows take the re-scored pass flags from 186's evaluation (XL_ROWS_PASS).
- The launcher falls back to any finished E2B seed. It skips direction e4b if E2B passes fewer than 30 row problems.
- The guard is 25 CU per session.

Caveats recorded before the run:
- **Direction 2's correctors may barely move E4B.** Half the fitting mix is E4B's own agent replies, already low-loss for an E4B student, so sft_c and sft_b may sit close to sft_self. Each corrector's fitted-versus-alone CE (in its log) is reported with the result.
- **Shared KV layers.** In both Gemma students most sites sit in layers that reuse earlier layers' keys and values. There, an added correction changes only its own position, not what later positions attend to (E2B layers 15-34, E4B 24-41), so the corrector is less expressive than on Qwen.

## Addendum 188 (2026-11-08, 09:45; built, not yet run): the matched-loss test for E2B's damage to untaught text (user: "yes, add it and queue the matched-loss test")

Why: in 186 every E2B recipe damaged untaught text far more than the same recipes did on Qwen. Prose ΔCE was +0.57 to +0.85, entropy fell 0.7 to 0.8, and KL from the original was about 0.29. The proposed explanation is over-fitting of familiar text: E4B's rows are already low-loss for E2B (0.31 at step 0), and two epochs drive them to about 0.07, which sharpens the model. Qwen ends at 0.24 on the same rows and softens. On top of that, E2B moves about 14 times more than Qwen at a similar final loss, which points to a student-specific sensitivity. (Qwen's rb ends at 0.03 with prose KL 0.02; E2B's sft ends at 0.07 with prose KL 0.27.) This test separates the two.

The 186 training curves show where the loss falls. Seed 1 had 0.309 at step 0, 0.254 at step 150 and 0.090 at step 200. Seeds 2 and 3 have the same shape. The first epoch barely lowers the loss, and the second memorizes the rows.

Design: xq_learn.sh STAGE=e2b with XL_ARMS="sft_es sft_lr", seeds 1-3, sessions xqlML1-3, launcher dreamer/xq_side20_launch.sh. The rows, adapter, layers and evaluation are those of 186's sft.
- **sft_es:** sft on all 299 E4B rows that stops when the 50-step mean training CE reaches 0.24 (XL_STOP_CE). That is Qwen's last-50 level on the same rows. From the curves, this is near the end of the first epoch.
- **sft_lr:** sft at a tenth of the learning rate (3e-5) for all 299 steps.
- **Anchor:** each session copies in 186's sft adapter for its seed and re-scores it, both for pairing and as the in-session anchor. Drift from the original E2B is measured for all three adapters.

Caveat recorded before the run: neither arm matches dose. sft_es trains for about half the steps, and sft_lr moves the weights less. If both cut the damage, that fits the over-fit explanation but does not separate the final loss from the size of the move. The comparison that does separate them is damage at the same final training loss as Qwen (sft_es against Qwen's sft).

Predictions (pooled over seeds 1-3, against 186's sft for the same seed):
1. sft_es stops between steps 130 and 200 in every seed (70%).
2. sft_es's prose ΔCE is at most half of sft's (65%).
3. sft_es's prose entropy falls by less than 0.3 (60%).
4. sft_es keeps at least half of sft's held gain over E2B alone (55%).
5. sft_lr's prose ΔCE is at most +0.15 (70%), and its held gain is under half of sft's (65%).
6. The re-scored 186 sft adapter reproduces 186's per-problem results exactly (90%).
7. At the same final loss as Qwen's sft (0.24), sft_es moves E2B more than Qwen's sft moved Qwen. Qwen's sft had prose KL 0.074 and prose ΔCE −0.06. Prediction: sft_es's prose KL is above 0.15 (55%), and its prose ΔCE is above zero (60%). Both would mean part of the move is the student.

Success bar for the over-fit explanation: sft_es keeps at least 70% of sft's pooled held gain and cuts prose ΔCE by at least half. If so, "stop at the student's matched loss" goes into the recipe. If sft_es keeps at least 70% of sft's damage, the explanation is wrong and student sensitivity dominates.

Budget: about 12 CU per session, about 40 CU. The sessions start after 187's six sessions have all started, under the same slot counter and the 25 CU guard.

**188 result (2026-11-08, 16:10; xqlML1-3, all exit 0).** sft_es stopped at steps 154, 156 and 159 (running CE 0.239-0.240). The re-scored 186 sft adapter reproduced 186 exactly (487 of 487 problems, every seed), so all arms pair with 186. Pooled over seeds (E2B alone: 171 held solved):

| Arm | Held solved (of 564) | Fixed (of 453) | Lost (of 615 solved alone) | Prose KL | Prose ΔCE | Prose entropy change |
|---|---|---|---|---|---|---|
| sft (186, re-scored) | 204 | 359 | 47 | 0.29 | +0.69 | −0.77 |
| sft_es (stop at CE 0.24) | 170 | 237 | 122 | 0.16 | +0.09 | −0.29 |
| sft_lr (lr 3e-5) | 206 | 233 | 71 | 0.12 | +0.18 | −0.36 |
| sft_self_s (187, KL to the original), for reference | 207 | 272 | 40 | 0.016 | −0.04 | — |

Paired (pooled, exact sign test): sft_es vs sft, held 27 vs 61 (p = 0.0004); sft_es vs E2B alone, held 46 vs 47 (p = 1.0). sft_lr vs sft, held 47 vs 45 (p = 0.92); trained 30 vs 187. Every seed agrees: sft_es held 54-60 (alone 57) and lost 35-47 (sft 15-17); prose ΔCE 0.04-0.14 (sft 0.68-0.69). Code and stories move the same way (ΔCE: sft +0.38 / +0.84, sft_es +0.04 / +0.10, sft_lr +0.10 / +0.19).

Predictions: 1 held (stops at 154-159). 2 held (ΔCE 0.09, 13% of sft's). 3 held narrowly (pooled entropy fall 0.29; seeds 0.25, 0.31, 0.32). 4 failed: sft_es kept none of sft's held gain (−1 against +33). 5 failed on both parts: sft_lr's ΔCE was +0.18 (bar 0.15) and it kept all of sft's held gain (+35). 6 held (487 of 487). 7 held: at Qwen's final loss, sft_es moved E2B further than Qwen's sft moved Qwen (prose KL 0.16 vs 0.074; ΔCE +0.09 vs −0.06). 5 of 7.

Success bar for the over-fit explanation (sft_es keeps at least 70% of the held gain and halves ΔCE): not met. ΔCE fell by 87%, but the held gain fell to zero. The failure condition (sft_es keeps at least 70% of the damage) is also not met: it kept 13%.

Reading: most of the damage to untaught text comes from the second epoch, as the over-fit account says, but stopping there is no fix. After one epoch E2B has lost the most solved problems (122) and gained nothing on held problems; the second epoch recovers the gain and does the damage. Part of the move is the student: at the same final loss E2B moved twice as far as Qwen. A tenth of the learning rate keeps the gain with a quarter of the damage, and KL to the original does better on held, fixed, lost and drift at once, so the recipe stays shortest rows + KL to the original. "Stop at the matched loss" does not go into the recipe.

## Addendum 189 (2026-11-08, 10:15; built, in review, not yet run): online setup D replicated, and two controls for it (user: "so the corrector does seem useful in this model"; "yes, queue D seeds 2-3 and the controls after the gate")

Why: setup D (183: the corrector teacher + keep + replay, E4B → E2B) is the first place where the corrector beat the plain teacher on several measures at once. Seed 1:

| Setup | Gate fixed (of 52) | Top choice matches teacher (of 220) | Gate CE | Untaught drift |
|---|---|---|---|---|
| A, teacher | 25 | 0.827 | 0.791 | 0.089 |
| B, corrector | 16 | 0.818 | 0.761 | 0.055 |
| C, teacher + keep + replay | 14 | 0.800 | 0.783 | 0.014 |
| D, corrector + keep + replay | 23 | 0.850 | 0.758 | 0.023 (item 550) |

D was also the only setup that changed held-decision agreement during the stream (47-49 of 55 from item 50 on; A, B and C ended at 45). Keep + replay lowered the plain teacher's gate count (25 → 14) but raised the corrector's (16 → 23). All four ran on the same code and libraries (transformers 5.22.2 from 11-07 17:36, before A started). D's result is one seed, and it does not say why D works. Two possibilities: E4B's information reaching the student through the corrector, or just a softer target, closer to the student, which is easier to learn.

Design: dreamer/online_rep_4090.sh, code7 (code5 plus this addendum's online.py). It runs before the RSI pilot, which was paused at 10:10 after 15 minutes of its first arm; 184 waits behind the pilot. Same settings as 170 and 183: rank-16 LoRA, 600 stream items, OPD_K 2, 256 tokens, keep 1.0, replay every 4th item.
0. **Re-gate** the four seed-1 adapters (A, B, C, D) with per-point output, for exact paired tests on the 52 decisions. The earlier gate files are backed up.
1. **D, seeds 2 and 3** (ON_SEED=2, 3). The seed offsets the stream's subset and order, the replay and keep picks, the LoRA init and the samples.
2. **Control teacher, seed 1:** base E2B corrected by the self-corrector gall_e2b. It reads E2B's own states, fitted like gall_e4b on the same data. Does E4B's information matter?
3. **Blend teacher, seed 1:** log_softmax(w × E4B + (1 − w) × base E2B), with w set before training. On base E2B's samples at 24 points outside the stream, the blend's starting pull KL(base ‖ blend) is matched to the corrector teacher's KL(base ‖ bridged E2B). Is the corrector anything more than a softer target? If the corrector pulls at least as far as E4B, the blend would be setup C, so it is not run; the calibration is still reported.

Order: re-gates, D seed 2, control, blend, D seed 3, each with a small pass first. About 15 hours.

Predictions:
1. The re-gates reproduce the stored gate counts exactly (25, 16, 14, 23) (85%).
2. D seeds 2 and 3 each fix at least 19 of 52 gate decisions (60%). The three-seed mean is at least 20 (60%).
3. D's untaught drift stays at or below 0.035 in seeds 2 and 3 (75%).
4. The control teacher fixes fewer gate decisions than D seed 1 (60%), and its gate CE is higher than D's 0.758 (60%).
5. The blend weight w lies between 0.2 and 0.8 (65%).
6. The blend teacher fixes fewer gate decisions than D seed 1 (55%), with drift within 0.01 of D's (55%).

Success bar for "the corrector is a useful online teacher": D's three-seed mean is at least 20 fixed, D seed 1 beats both control and blend by at least 4 fixed decisions, and D's drift is at most twice C's. If the blend matches D, the paper says that a softer target, not the corrector, drives the gain.

**Amendment 189a (10:40, after code review, before any new training run).** The independent review found no crash bugs and no wrong targets. It confirmed:
- With ON_SEED unset, the stream and the samples are identical to before.
- Seeds 2 and 3 vary the stream, the picks, the LoRA init and the samples.
- The control teacher matches res_passes' control path.
- The blend's calibration cannot change the stream's samples.
- Memory stays below the bridge arm's.

Applied, and swapped into code7 before D seed 2 started:
- The blend's small pass no longer stops the arm when its 2-point calibration gives w = 1. Only the full 24-point calibration decides.
- The blend weight is interpolated on √KL, which is close to linear in w. Plain interpolation on KL ran slightly low.
- The full run requires all 24 calibration points.

**Amendment 189b (10:25, before any new training run; user: "yes, queue C seeds 2-3 after D seed 3").** Setup C (E4B's own distribution + keep + replay), seeds 2 and 3, so that D and C both have three seeds. Runner dreamer/online_rep2_4090.sh starts after ONLINE_REP_DONE, with the same code7, settings, small pass and gate. The RSI pilot now waits for ONLINE_REP2_DONE. The waiting pilot and 184 runners were stopped by exact PID and relaunched; the running queue was not touched. The first re-gate (A) reproduced its stored gate exactly: 25 fixed, CE 0.7914.

Predictions:
1. C fixes at most 19 of 52 in each of seeds 2 and 3 (60%).
2. C's three-seed mean is at least 4 below D's (55%).
3. C's untaught drift is at most 0.02 in each seed (75%).

Success bar for "the corrector is a useful online teacher", revised: D's three-seed mean beats C's by at least 4 fixed decisions. On the paired decisions, pooled over seeds, D's wins over C must outnumber its losses with an exact sign test p < 0.1. D seed 1 must also beat both control teachers by at least 4, and D's mean drift must be at most twice C's.

**189, interim (11:05): the re-gates and paired tests at seed 1.** All four re-gates reproduced their stored counts exactly: 25, 16, 14 and 23 (prediction 1 held). Per decision, on the 220 gate decisions:

| Setup | Fixed (of 52 where E2B alone differs) | Broken (of 168 where it agreed) | Net | Matches (of 220) |
|---|---|---|---|---|
| A, teacher | 25 | 11 | +14 | 182 |
| B, corrector | 16 | 4 | +12 | 180 |
| C, teacher + keep + replay | 14 | 6 | +8 | 176 |
| D, corrector + keep + replay | 23 | 4 | **+19** | **187** |

Exact sign tests, D against each setup, D-only against the other's only:
- **vs C:** 13 vs 4 on the 52 (p = 0.049); 17 vs 6 on all 220 (p = 0.035).
- **vs A:** 7 vs 9 on the 52 (p = 0.80); 14 vs 9 on all 220 (p = 0.41).
- **vs B:** 10 vs 3 on the 52 (p = 0.09).

Reading: A fixes about as many decisions as D but breaks almost three times as many. Within this seed, D beats C. These p-values only cover the decisions of each trained adapter, not the spread between training runs; the replicated online arms (resdec) ranged 19-25 across seeds. Seeds 2-3 decide.

**189, interim 2 (2026-11-08, 14:30 gate; read 15:05): D seed 2.** Same code7 and settings as seed 1, ON_SEED=2 (a different stream order, picks, LoRA init and samples). Gate on the same 220 decisions:

| | Fixed (of 52) | Broken (of 168) | Net | Matches (of 220) | Held CE at item 600 | Untaught drift (end / peak) | Held-question drift |
|---|---|---|---|---|---|---|---|
| D seed 1 | 23 | 4 | +19 | 187 | 0.751 | 0.017 / 0.051 | 0.041 |
| D seed 2 | 23 | 6 | +17 | 185 | 0.749 | 0.017 / 0.032 | 0.021 |

- The two seeds differ on only 6 of the 220 decisions (seed 2 only 2, seed 1 only 4, p = 0.69).
- Seed 2 against the seed-1 adapters of the other setups: vs C 13 vs 4 on the 52 (p = 0.049), 17 vs 8 on all 220 (p = 0.11); vs A 7 vs 9 (p = 0.80); vs B 10 vs 3 (p = 0.09). These pair seed 2 of D with seed 1 of the others; the matched-seed test against C waits for C seeds 2-3.
- Prediction 2, seed 2 part (at least 19 fixed): held (23). Prediction 3, seed 2 part (untaught drift at most 0.035): held (0.017 at the end, 0.032 at its peak). Prediction 1 already held. Predictions 4-6 (control and blend) wait for those arms.

Reading: the online result is not one lucky run; a second seed reproduced it nearly decision for decision. The success bar still needs D seed 3, C seeds 2-3 and both controls.

**189, interim 3 (2026-11-08, 17:54 gate; read 18:15): the self-corrector teacher (control), seed 1.** Same code7, settings and gate as D, with the target from the corrector that reads E2B's own states (gall_e2b) instead of E4B's. At item 600: held CE 0.755 (D: 0.751 and 0.749), untaught drift 0.025 (D: 0.017 both seeds), held-question drift 0.073 (D: 0.041 and 0.021). Gate CE 0.762 (D: 0.758 and 0.757). The gate partition (E2B alone) is identical to every earlier gate.

| Setup (keep + replay) | Fixed (of 52) | Broken (of 168) | Net |
|---|---|---|---|
| D, cross-model corrector, seed 1 | 23 | 4 | +19 |
| D, cross-model corrector, seed 2 | 23 | 6 | +17 |
| Control, self-corrector, seed 1 | 13 | 5 | +8 |
| C, teacher's distribution, seed 1 | 14 | 6 | +8 |

Paired over all 220 decisions: D seed 1 vs control 12 vs 1 (p = 0.003); D seed 2 vs control 12 vs 3 (p = 0.035); C vs control 6 vs 6 (p = 1.0).

Prediction 4 held (the control fixes fewer than D seed 1, 13 vs 23; its gate CE is higher, 0.762 vs 0.758). Success-bar part "D seed 1 beats the control by at least 4 fixed": met (10). The blend teacher is running.

Reading: on agent decisions the corrector's gain needs E4B's states. The self-corrector reaches nearly the same held CE but fixes no more decisions than the plain teacher with keep + replay. This differs from coding, where the cross-model and self anchors tied. One control seed. The blend control decides whether a softer target alone, matched to the corrector's pull, does as well.

**Amendment 189c (2026-11-08, 17:38, before either run; user: "yes, move C seeds 2-3 to Colab").** Setup C seeds 2 and 3 move from the 4090 queue (online_rep2_4090.sh) to two Colab G4 sessions (colab/xq_online.sh, sessions xqoC2 and xqoC3, launcher dreamer/xq_onl_launch.sh), so they run in parallel with the 4090's control, blend and D seed 3. Nothing else changes:
- Code: online.py, translator.py, pair.py, store.py, mix_merge.py and dequant.py are the committed repo files, md5-equal to the 4090's code7. Libraries: torch 2.15.0, transformers 5.22.2, peft 0.21.2, accelerate 1.15.0 on both machines.
- Inputs: the s3 rows merged with the 165 mix by mix_merge.py, sha256-equal to the 4090's data2 files (checked in the session before training). The correctors gall_e4b.pt and gall_e2b.pt are sha256-equal to the 4090's data3/out.
- Models: the same QAT revisions, dequantized by dequant.py. The session compares the sha256 of each model file with the 4090's copy and logs the result.
- Settings: online_rep2_4090.sh's environment (TR_PLE_CPU=1 kept), the same small pass, 600-item run and 240-point gate.

What does change is the GPU (RTX PRO 6000 instead of RTX 4090), so bf16 arithmetic can differ. The gate re-runs E2B alone and E4B live in the same session, so near-tie decisions may flip. Analysis rule, set now: fixed / broken / net for the Colab seeds are scored on the 4090's partition (the 52 decisions where E2B alone differs from E4B and the 168 where it agrees, from the stored seed-1 gates). The number of the 220 decisions where the Colab session's E2B-alone or E4B choice differs from the 4090's is reported as the hardware noise level, and so is C seed 1's gate against the Colab seeds. The 4090's online_rep2 runner is replaced by one that only passes ONLINE_REP2_DONE on to 192 once the 189 queue ends.

Predictions 189b 1-3 and the revised success bar are unchanged. One added prediction: at most 6 of the 220 gate decisions differ between the Colab and 4090 runs of E2B alone (70%).

**Amendment 189c, review changes (2026-11-08, 17:52, before either run).** The code review found three problems and two smaller points:
- GPU: Colab G4 was running at about 9.5 CU/h today, and one C run takes about 3.5-4 h with setup and gate (C seed 1 trained for 11,186 s on the 4090), so two G4 sessions would need 70-75 CU against a 72.6 CU balance. The sessions run on A100 40 GB instead (5.3 CU/h; the job fits the 4090's 24 GB). The hardware note above now reads A100, not RTX PRO 6000.
- Models: the 4090's dequantized E4B and E2B were made with an older library stack (compressed-tensors 0.15.0.1, torch 2.10, transformers 5.13.1), so a fresh dequant on Colab may not give the same weights. The sessions instead download the Mac's copies, which are sha256-equal to the 4090's (on/models/ in the private job repo), and a mismatch stops the session.
- Outputs: the adapter and the learning curve are copied out as soon as training ends, and the partial curve is copied if any step fails.
- The correctors are not loaded for teacher = direct (only for bridge, blend and control); they are still copied and checked.
- The per-point gate files store agreement with E4B and P(E4B's choice), not token ids. So the hardware-noise count is the number of the 220 decisions whose E2B-alone agreement bit flips; a change in E4B's own choice is counted in it too.

**Amendment 189d (2026-11-08, 20:47, before the run; user: "added 100 more cu").** D seed 3 moves from the 4090 to a Colab A100 session (xqoD3: colab/xq_online.sh TEACH=bridge ON_SEED=3; the same code7, inputs, correctors, models and settings as 189c, now with the teacher as a parameter). The 4090's 189 runner was replaced during the blend arm by online_rep1b_4090.sh, which finishes the blend exactly as arm() would (waits for the running training, logs its lines, runs the gate) and then writes ONLINE_REP_DONE without bridge seed 3. So 196 (D live) starts on the 4090 right after the blend gate, about 21:45 instead of about 01:00. D seed 3 now runs on different hardware from seeds 1-2, as C seeds 2-3 do. The same analysis rule applies (the 4090's gate partition; hardware noise reported). Predictions unchanged.

**189, interim 4 (2026-11-08, 21:31 gate; read 21:33): the blend teacher, seed 1.** Same code7, settings, stream and gate as D seed 1, with the target log_softmax(w × E4B + (1 − w) × base E2B). Calibration on 24 points outside the stream (10,987 tokens): KL(base ‖ E4B) 0.323, KL(base ‖ corrector teacher) 0.064, so the blend was run, with w = 0.433.

| Setup (keep + replay) | Fixed (of 52) | Broken (of 168) | Net | Gate CE | Held CE (item 600) | Untaught drift | Held-question drift |
|---|---|---|---|---|---|---|---|
| D, corrector, seed 1 | 23 | 4 | +19 | 0.758 | 0.751 | 0.017 | 0.041 |
| Blend, w = 0.433, seed 1 | 14 | 4 | +10 | 0.787 | 0.781 | 0.014 | 0.022 |
| C, teacher's distribution, seed 1 | 14 | 6 | +8 | 0.783 | 0.775 | 0.014 | 0.034 |
| Control, self-corrector, seed 1 | 13 | 5 | +8 | 0.762 | 0.755 | 0.025 | 0.073 |

Paired over all 220 decisions: D seed 1 vs blend 11 vs 2 (p = 0.022; on the 52 differing, 10 vs 1, p = 0.012); D seed 2 vs blend 11 vs 4 (p = 0.12).

Prediction 5 held (w = 0.433, inside 0.2-0.8). Prediction 6 held (the blend fixes fewer than D seed 1, 14 vs 23; untaught drift 0.014, within 0.01 of D's 0.017). Success-bar part "D seed 1 beats both control and blend by at least 4 fixed": met (by 10 and 9). Still open: D's three-seed mean (seed 3 on Colab) and D's drift against C's three seeds.

Reading: a softer target alone does not explain D. Matched to the corrector's pull from base E2B (a fifth of E4B's), the blend behaves like setup C: close drift, few breaks, but it fixes no more decisions than C and sits further from the teacher on held text (CE 0.781 vs D's 0.751). D's target is as close to E2B as the blend's on average but points elsewhere, and the self-corrector control shows that E2B's own states do not give the gain. Not ruled out (review, 21:50): a selective target. The blend matches the corrector's average pull with a uniform mix; the corrector's pull is likely larger where E4B and E2B disagree, and a teacher applied only at disagreements would be near on average and carry E4B's choices with no learned module. Addendum 197 (the disagreement-gated teacher) tests this: if it matches D, the corrector is sufficient but not necessary; if it falls short, the state-informed correction matters. Until then the paper does not call the corrector necessary. One blend seed; hardware as D seed 1 (the 4090).


## Addendum 190 (2026-11-08, 12:05; built, in review, not yet run): the multi-task control for the corrector anchor (user: "run the k on prose also and see what happens... in that case both prose and agentic work/solved should be similar to the correctors")

Why: in 187 (E2B student, seeds 1-2) the self-corrector anchor (sft_c_s) solved the most held problems and turned SFT's damage to untaught prose (ΔCE +0.69) into a gain (−0.61), with softer predictions (entropy +0.85). The corrector was fitted on a text mix: agent replies 0.5, prose 0.15, code 0.15, questions 0.1, stories 0.1, split "train", with the untaught drift rows held out from the same sources. So the anchor's benefit may only be that text arriving by another route. The paper lists the missing multi-task baseline as a limitation, and the Fable review named the confound.

Design: arms sft_mt_s and sft_self_mt_s in xlearn.py.
- They are sft_s and sft_self_s: the same shortest rows, order and dose.
- Each step adds MT_N = 2 rows of the corrector's own fitting text, drawn from split "train" by the corrector's kind weights (XB.W), encoded as in the corrector's fit. The loss is cross-entropy on the reply, weight 1.0, the same per-row weight as the coding rows.
- Mix rows come from their own random stream (seed 190 + s), so the coding order is unchanged.
- Gemma 4 E2B student, seeds 1-3, three Colab sessions: xq_learn.sh STAGE=e2b with XL_ARMS=sft_mt_s,sft_self_mt_s, launcher dreamer/xq_side21_launch.sh.
- Each session re-scores 186's sft adapter, which pairs it with 186 and 187 (the 186-to-187 re-score reproduced 0 differences of 487). Drift runs for every adapter in the session.
- Comparisons: against 187's sft_c_s, sft_b_s, sft_self_s and 186's sft_s, by stored per-problem results.

Queue order (Colab balance 112 CU at 12:00): these sessions start before 188's, which was queued first but had not started. 188 runs if the balance allows.

Predictions (pooled over seeds 1-3; the anchors' 187 values once all seeds are in):
1. sft_mt_s lowers untaught prose loss: prose ΔCE at most −0.3 (65%).
2. sft_mt_s softens prose less than the corrector anchors: its entropy change is below +0.4 (70%).
3. sft_mt_s solves held problems within ±10 of sft_c_s (55%).
4. sft_self_mt_s moves the student less than sft_mt_s: lower prose KL from the original (80%).
5. sft_mt_s reaches lower held agent-reply CE than sft_c_s, because it trains on agent replies directly (60%).

Success bar for "the anchor only carries the corrector's fitting text": sft_mt_s solves held problems within ±10 of sft_c_s, with prose ΔCE within 0.15 of sft_c_s's. If both hold, the paper says the corrector anchor's benefit is reproduced by multi-task SFT on the same text. If sft_c_s beats sft_mt_s by more than 10 held problems, or keeps a clearly larger prose gain, the anchor adds something beyond its text.

**Amendment 190a (12:25, after code review, before any run).** The review found no crash bugs and confirmed that held rows cannot leak in: the mix uses only split "train", and evaluation and drift use only split "held". It ran the mix code on the local data with the real tokenizer: the mix counts match the corrector logs, the reply-span cross-entropy scores exactly the reply tokens, and there is one BOS per row.

Applied:
- **Dose:** MT_N 4 mix rows per step, so 4 × 299 ≈ 1,200 rows. That matches the corrector's own exposure (600 steps × 2). The first design gave half that.
- **Loss:** the mix rows' mean is divided by MT_N, with weight 1.0.
- **Clipping:** the share of steps clipped at norm 1.0 is saved for every arm, because the mix rows share the clip with the coding rows.
- **Pairs:** in-session paired tests for sft_mt_s and sft_self_mt_s against sft and against each other.

xlearn.py was uploaded to HF before the sessions were released.

Caveat recorded: 48 of the 57 held code chunks come from standard-library modules that also have training chunks (different lines), so code drift for the mix arms is a near-domain measure. The corrector saw the same modules. Untaught prose shares no 32-token span with the training text.

**Result 187 (2026-11-08, 12:55; Colab, six sessions, seeds 1-3 each way).** Held solved pooled over three seeds, with exact sign tests on discordant held problems.

Direction 1, E4B teaches E2B (E2B alone 171):

| Arm | Held solved | Fixed per seed (of 151) | Lost per seed (of 205) | Prose KL / ΔCE / Δentropy |
|---|---|---|---|---|
| sft | 204 | 119.7 | 15.7 | 0.290 / +0.69 / −0.77 |
| sft_s (186) | 204 | 123.0 | 15.7 | 0.291 / +0.57 / −0.7 |
| sft_self_s | 207 | 90.7 | 13.3 | 0.016 / −0.04 / −0.01 |
| sft_c_s | 216 | 95.3 | 16.0 | 0.326 / −0.60 / +0.83 |
| sft_b_s | 209 | 99.3 | 15.3 | 0.323 / −0.58 / +0.80 |

Paired held tests: sft_c_s vs sft 46 vs 34 (p = 0.22), vs sft_s 38 vs 26 (p = 0.17), vs sft_b_s 33 vs 26 (p = 0.44); sft_self_s vs sft 41 vs 38.

Direction 2, E2B teaches E4B, on 148 rows (E2B's passing answers; E4B alone 207):

| Arm | Held solved | Lost per seed (of 205) | Prose KL / ΔCE / Δentropy |
|---|---|---|---|
| sft | 224 | 20.0 | 0.292 / +0.81 / −0.93 |
| sft_self | 244 | 11.3 | 0.012 / −0.01 / −0.04 |
| sft_c | 250 | 13.3 | 0.185 / −0.42 / +0.67 |
| sft_b | 249 | 12.3 | 0.184 / −0.44 / +0.67 |

Paired held tests: sft_c vs sft 52 vs 26 (p = 0.004); sft_b vs sft 53 vs 28 (p = 0.007); sft_self vs sft 47 vs 27 (p = 0.03); sft_c vs sft_b 30 vs 29.

Corrector fits: the self and cross-model correctors cut loss on their own fitting mix by 27-29% for E2B and E4B (Qwen's: 34-37%), with self and cross within 0.04 in every seed. On E4B's shortest coding rows, every anchor target, including E2B corrected with E4B's states, scored 0.27-0.30 nats, the same as E2B alone. The anchors held the final training loss at 0.11-0.12 (E2B; sft_s 0.08) and 0.09-0.11 (E4B; sft 0.07). Calibration spread at the sites was 45-46 (E2B) and 31-32 (E4B), against 18-19 for Qwen.

Predictions:
- Direction 1:
  - sft_b_s within ±10 held of sft_c_s: yes (209 vs 216).
  - sft_self_s has the lowest prose KL: yes (0.016).
  - The corrector anchors lower untaught-prose CE more than sft_s: yes (−0.60, −0.58 vs +0.57).
- Direction 2:
  - Fewer than 200 rows: yes (148).
  - E4B sft at most as many held as alone: no (224 vs 207).
  - sft_b no more held than sft_c: yes (249 vs 250).
  - Every arm fixes at most 5 per seed: no (9-10).

5 of 7 held. Success bar (direction 2: sft_b beats sft_c by at least 5 held): not met. The cross-model and self anchors tie in both directions.

Reading:
- In the same family, any anchor removes the damage that plain copying does to untaught text, in both directions and every seed. KL to the original keeps the student in place; the corrector anchors move it toward lower loss.
- When E4B learns from E2B's familiar answers, the anchors also raise held-out coding significantly (+20 to +26 over SFT). For E2B learning from E4B, the held gain is within noise (at most +12).
- E4B's states add nothing over the student's own. The anchor's prose gain must come through the corrector's fitting mix (it does not move E2B on the coding rows): addendum 190 tests this.

## Addendum 191 (2026-11-08, 13:40; design, not yet built; for a separate paper): per-user grafts, a small specialist attached to a large frozen model (user: "each user of a larger model could have their own 'graft' a smaller model attached to the larger one"; "yes, write the graft addendum and queue it after... if we publish it, it will be in a different paper")

Question: can a small model that learned a user's private material act as a graft on a large frozen model? The graft would be the small model's states written into the large model through a corrector. The large model should then answer questions about that material, compared with retrieval and with a per-user LoRA, without harming its general skills.

Why: in 187 the E2B→E4B corrector equalled E4B's self-corrector (within 0.02 nats), because E2B knew nothing E4B did not. The graft idea only matters when the small side knows something the large side lacks. Swarm Telepathy (Alabdulkareem et al., 2026) reports zero-shot transfer of specialized knowledge from a fine-tuned sender through a generic mapper; this is the same question for a residual-stream corrector.

**Material: a private library that neither model's weights contain.**
- E4B writes a fictional Python package, so the text is model-written: about 30 modules of 3-6 pure, stdlib-only, deterministic functions with docstrings and type hints. Each module is seeded by a short domain prompt.
- Code from the 4090 runs only in the bubblewrap sandbox (XC_SANDBOX=bwrap). A function is kept only if it imports and its E4B-written example calls run deterministically. Names that collide with standard-library or common package names are dropped.
- Training text: the module sources with docstrings, plus the "seen" example calls with their executed outputs.

**Questions.** Templated; answers come from the code's syntax tree or from execution:
- (a) which module defines a function;
- (b) a function's parameters in order;
- (c) a parameter's default;
- (d) the return value of a seen call;
- (e) the return value of an unseen call, written by E4B separately, never in the training text and executed for the answer.

**Scoring.** Exact match of the greedy short answer after normalisation, and the cross-entropy of the correct answer given the question (continuous and more sensitive). Reported per question type.

**Arms.** The large model is E4B, answering closed-book unless stated; seeds 1-3:
- **A.** E4B alone.
- **B.** E4B with the defining module's source in the prompt (oracle retrieval; the reference ceiling).
- **C.** E4B with a per-user LoRA (rank 16) trained on the training text.
- **D.** The E2B specialist alone: E2B with a LoRA (rank 16) trained on the same text, the same steps.
- **E.** Zero-shot graft: E4B with 187's generic E2B→E4B corrector for the same seed. That corrector was fitted only on the general mix and never saw the library. It reads the specialist's states, so the specialist feeds the corrector.
- **F.** Graft control: the same corrector reading base E2B's states.

E against F isolates what the specialist's states carry. Phase 2, run only after phase 1 is read: a corrector fitted on the library text reading the specialist (G), against the same fit reading E4B's own states (G0). This separates knowledge carried by the specialist from the corrector memorising the text.

**General skills.** E4B's CE and KL from E4B alone on the held untaught prose, code and stories, and on held agent replies, for C, E and F (grafts measured with the graft on).

**Cost.** Added memory and time per token for C, E and F.

Predictions (pooled over seeds 1-3, the mean over question types unless stated):
1. B answers best on every question type (85%).
2. C beats A on questions (a)-(c) (80%) and gains less on (e) than on (a)-(c) (65%).
3. D beats A on library questions (80%).
4. E beats F on answer CE (55%). On accuracy, E beats F by at least 10 points (35%).
5. E reaches less than C's accuracy gain over A (75%).
6. E and F both lower E4B's CE on untaught prose, the generic corrector's carry-over (70%); C does not (60%).

Success bar for "a zero-shot graft carries a user's knowledge": E beats F by at least 10 accuracy points pooled, with a paired test p < 0.05 on questions, and E's gain over A is at least half of C's. If E is about equal to F, phase 2 decides whether a fitted graft beats the per-user LoRA.

**Compute and queue.**
- Runs on the 4090 after the online queue (ONLINE_REP2_DONE). The RSI pilot and 184 wait behind it.
- The large and small models share the GPU with the per-layer embeddings in CPU memory (TR_PLE_CPU, as online.py); the corrector code (xbridge.py) is to get the same option.
- Steps: library generation (about 1 h), sandbox check, questions, two LoRAs per seed, the evaluation of A-F. About 6 h.
- Before queuing: a small CPU test, and an independent review of the generator, the sandbox check and the scoring.
- Nothing leaves the machine. The library and questions are model-written, from E4B.

**Amendment 191a (13:55, design, before any build; user: "with a graft we had no forgetting because the graft adds the extra layers").** The large model's weights never change. Detached, E4B is exactly the original, so forgetting inside the large model is excluded by construction, and per-user grafts cannot damage the shared model or each other. A per-user LoRA shares this frozen-base property when detached. The open question is forgetting *while attached*: a graft adds to E4B's residual stream on every input, a LoRA changes every forward pass, and either can change behaviour on unrelated inputs. The corrector-style graft is gated and fitted with a do-no-harm term, so it should stay off where E4B is already right. Added to 191:

- **Drift with the module attached:** for C (per-user LoRA), E and F (grafts), E4B's KL from E4B alone and ΔCE on untaught prose, code and stories, on held agent replies, and p(true) on fresh questions.
- **Online phase**, after phase 1 is read: a second release of the library, about 10 further modules written by E4B the same way, arrives as a stream. Each module is seen once and its questions are scored before its update (progressive validation), with keep + replay as in 189. Graft arm: the specialist's LoRA learns and the corrector stays fixed. LoRA arm: E4B's LoRA learns. Measured: accuracy on the new modules after the stream, release-1 accuracy kept, and drift with the module attached.
- **Check:** detached, every arm reproduces E4B alone exactly.

Predictions:
7. With the module attached after phase 1, the graft's prose KL from E4B alone is below the per-user LoRA's (60%).
8. After the online phase, the graft keeps more release-1 accuracy than the per-user LoRA (55%).
9. The detached check passes for every arm (a check, not a finding).

Bar for "always-on learning without forgetting": the graft's attached drift is at most half the per-user LoRA's on untaught prose. After the online phase, it keeps at least 90% of its release-1 accuracy and learns the new modules, gaining at least half of what the per-user LoRA gains on them.

**Amendment 191b (14:10, design, before any build; user: "we chat with the model, and all interaction text, history, etc, is saved to the graft. We then clear the context and ask the model of what we talked about. If it remembers the graft in essence added the extra layer").** Conversation memory becomes the primary test. The library questions become the follow-up.

**Material.**
- 50 conversations about a fictional user's software projects. Each carries 8 randomized facts: a project name, a function name, a port, a version, a deadline, the bug discussed, the decision taken, a teammate's name. Values are drawn at random (random pronounceable names, numbers, dates), so no model can guess them.
- E4B role-plays both the user, who brings the facts up naturally over several turns, and the assistant, so the text is model-written. A conversation is kept only if every fact string appears in it.

**Memory step ("saved to the graft").**
- The conversations arrive one after another as a stream, as in a user's history. After each one, the learning module takes K passes over its transcript, with K in {1, 3, 10} as a dose curve. E4B and the corrector stay frozen.
- Two variants:
  - (i) the transcript alone;
  - (ii) the transcript plus about 10 question-answer pairs about it, written by the specialist itself (self-study, as in Cartridges and synthetic continued pretraining). Facts trained only as running text are often not answerable.

**Test.** A cleared context, then the questions:
- (a) direct: "Earlier we talked about my project: which port does the dev server use?";
- (b) paraphrased;
- (c) open: "What did we decide about the bug?", scored by the facts recalled.

Exact match after normalisation, and the cross-entropy of the right answer. Reported for early against late conversations, which measures forgetting inside the memory as the stream grows.

**Arms.**
- **A.** E4B alone, cleared context. The floor: near chance.
- **B.** E4B with the conversation in its context. The ceiling: in-context memory.
- **C.** E4B with a per-user LoRA trained the same way.
- **D.** The E2B specialist alone, trained the same way.
- **E.** The graft: E4B with 187's generic E2B→E4B corrector reading the specialist.
- **F.** Graft control: the same corrector reading an untrained E2B.

Plus 191a's attached drift on untaught text and the detached check.

Predictions:
10. A is near chance and B near ceiling on direct questions (90%).
11. With the transcript alone, C and D answer direct questions poorly at K = 1 (75%); self-study pairs (variant ii) raise both (70%).
12. E beats F on answer CE in variant ii (55%); on direct-question accuracy E beats F by at least 10 points (35%).
13. Early conversations are recalled less than late ones in C, D and E at the end of the stream (65%).

Bar for "the graft added a memory layer": in variant ii, E beats F by at least 10 points on direct questions (paired test p < 0.05), keeps 191a's drift bar, and recalls at least half of what D (the specialist alone) recalls. A live demo, where the user chats and then asks in a cleared context, follows only if the bar is met.

**Amendment 191c (14:25, design, before any build; user: "ideally we would need to create some kind of system in between the graft and the chat history, where it's automatically updated").** The 191 harness is built as that system, so the experiment runs it with a scripted user and a later live demo runs the same code with a person. Components:

1. **History log:** every turn is appended with a timestamp. Until consolidation, retrieval over the log answers.
2. **Memory writer:** runs after each conversation (cadence C = 1, plus C = 10 as a batch variant). The specialist writes study notes and question-answer pairs about the new turns (variant ii). Its adapter trains K passes on them, with replay of a sample of older memories and keep (KL to base E2B on untaught rows), as in 189. E4B and the corrector are frozen.
3. **Gate before promotion:** recall on the new facts, kept recall on a sample of older ones, and attached drift on untaught text (191a). A failed candidate is logged and the previous version stays live.
4. **Registry:** versioned adapter files, a current pointer and rollback. Deletion rebuilds from the last version without the deleted conversation.
5. **Serving:** E4B, the corrector, and the specialist with the current adapter, swapped between turns.

Added measures: the gate's pass rate and why candidates fail; recall over the stream for each version (the learning curve with the gate on and off); time per consolidation on the 4090. No new predictions: these describe the system. 191b's bar stays the test.

## Addendum 192 (2026-11-08, 14:45; built, in review, not yet run): is online distillation a better teacher than SFT on agent decisions? (user: "is online a better teacher (with less breaking) then SFT?"; "yes, add both and queue them before the graft")

Why: every model scored on the 220-decision gate learned on-policy (the student samples, the teacher scores every token), offline or online. Plain SFT, which copies E4B's logged replies, was never gated, so "online versus SFT" has no answer yet. On coding, plain SFT on verified answers beat on-policy distillation (179-181). The gate so far (seed 1):

| Setup | Fixed (of 52) | Broken (of 168) | Net |
|---|---|---|---|
| Offline on-policy, alldirect | 20 | 5 | +15 |
| Offline best recipe, resdec (4 runs) | 19-27 | 2-9 | +14 to +22 |
| Online A, teacher | 25 | 11 | +14 |
| Online C, teacher + keep + replay | 14 | 6 | +8 |
| Online D, corrector + keep + replay | 23 | 4 | +19 |

Design: online.py additions (additive; every earlier path unchanged), runner dreamer/online_rep3_4090.sh, after ONLINE_REP2_DONE and before the graft build. The RSI pilot and 184 wait behind it. Seed 1, the same 600 stream items as the online setups, rank-16 LoRA on E2B, then gate_lora on the same 240 points.
1. **Offline SFT** (ON_LORA_LOSS=ce, ON_LORA_POOL=stream): plain cross-entropy on E4B's logged reply. 600 steps × 2 rows drawn from the 600 stream items, about two passes, at lr 3e-4 with warm-up and cosine decay. This is the classic batch baseline: the same data, revisited, no keep or replay.
2. **Online SFT** (ON_ONL_TEACH=sft, keep 1.0, replay every 4th): each stream item once, cross-entropy on E4B's logged reply, flat lr, with D's protections.

So copying versus distillation is compared online (2 vs D, both with keep + replay), and online versus batch for copying (2 vs 1).

Caveat: a logged reply is a single sample, while D learns from two of the student's own samples per item scored by the corrector. Neither the token count nor the target type is matched, by design: these are the two recipes as a practitioner would run them.

Predictions (seed 1):
1. Offline SFT fixes at least 20 of the 52 (60%) and breaks more than D's 4 (70%).
2. Online SFT breaks at most 6 (60%) and fixes fewer than D's 23 (55%).
3. D's net (+19) is at least both SFT arms' net (55%).
4. Offline SFT's untaught drift is above online SFT's (75%).

Bar for "online distillation through the corrector is a better teacher than SFT": D's net beats both SFT arms' by at least 4, with paired exact sign tests on the 220 decisions at p < 0.1. One seed, so this reads as a direction until seeds are added.

**Amendment 192a (13:45, after code review, before any run).** The review found no crash bug in online.py. It confirmed:
- The cross-entropy scores exactly E4B's logged reply tokens.
- The offline pool is exactly the online stream (seed 1).
- The online sft branch trains with the adapter on.
- Every earlier path is unchanged.

It found one crash-level bug in the runners. online_rep2_4090.sh and online_rep3_4090.sh waited for their predecessor's marker with a plain grep, but each had just written a log line that named that marker. Both waits passed at once and left only the free-GPU check, so the two runners could have started together on the 24 GB card when the 189 queue ended. Neither had started.

Fixed and deployed at 13:42 (each waiting runner was stopped by exact PID, the files swapped with mv, then relaunched):
- The waits match only the exact marker line, and the log lines no longer name markers.
- Offline SFT does not load E4B (about 9 GB freed) and starts its LoRA from seed 161, as the online arms do.
- Both SFT small passes train first on the longest reply, to test the memory peak.
- train_lora logs peak GPU memory.

**Result 190 (2026-11-08, 15:20; Colab, seeds 1-3; each session re-scored 186's sft identically).** E2B; held solved pooled over three seeds (E2B alone 171); fixed and lost per seed; prose drift as three-seed means; held agent-reply CE (E2B alone 0.685).

| Arm | Held | Fixed | Lost | Prose KL / ΔCE / Δentropy | Agent CE |
|---|---|---|---|---|---|
| sft | 204 | 119.7 | 15.7 | 0.290 / +0.69 / −0.77 | 0.749 |
| sft_self_s (187) | 207 | 90.7 | 13.3 | 0.016 / −0.04 / −0.01 | 0.668 |
| sft_c_s (187) | 216 | 95.3 | 16.0 | 0.326 / −0.60 / +0.83 | 0.650 |
| sft_b_s (187) | 209 | 99.3 | 15.3 | 0.323 / −0.58 / +0.80 | 0.648 |
| sft_mt_s | 214 | 105.0 | 16.3 | 0.829 / −0.75 / +1.18 | 0.504 |
| sft_self_mt_s | 206 | 83.3 | 17.3 | 0.811 / −0.76 / +1.18 | 0.502 |

Paired held tests: sft_mt_s vs sft_c_s 32 vs 34 (p = 0.90); vs sft_s 42 vs 32 (p = 0.30); vs sft 46 vs 36 (p = 0.32). sft_self_mt_s vs sft_self_s 40 vs 41.

Training: coding CE last50 was 0.114-0.131 (sft_mt_s) and 0.135-0.149 (sft_self_mt_s), held up like the anchors' 0.11-0.12. Mix CE fell from 1.6-1.8 to 1.2-1.4. 93-97% of steps were clipped at norm 1.0, so the coding signal was scaled down along with the mix.

Predictions:
1. sft_mt_s prose ΔCE at most −0.3: yes (−0.75).
2. sft_mt_s entropy change below +0.4: no (+1.18).
3. sft_mt_s within ±10 held of sft_c_s: yes (214 vs 216).
4. sft_self_mt_s lower prose KL than sft_mt_s: yes, barely (0.811 vs 0.829).
5. sft_mt_s lower agent CE than sft_c_s: yes (0.504 vs 0.650).

4 of 5 held. Success bar ("the anchor only carries the corrector's fitting text"): met. Held 214 vs 216 is within ±10, and prose ΔCE −0.75 vs −0.60 is within 0.15.

Reading: the corrector anchor's gains on E2B (held solves, lower loss on untaught text) are reproduced by training on the text the corrector was fitted on. Direct training reaches lower agent-reply loss but moves the student about 2.5 times as far (prose KL 0.83 vs 0.33), and the KL pull to the original cannot hold it in place against that much new text. The anchor is a gentler channel for its fitting text, not a source of anything beyond it. The "untaught" prose comes from the same books as the fitting prose (every seventh passage held out), so these prose gains are in-domain for both routes.

## Addendum 193 (2026-11-08, 15:40; built, in review, not yet run): KL strength sweep for the anchor to the original student (user chose it from the Fable review: "λ = 1 only. Even a three-point λ sweep on one seed would make the recommendation robust")

Why: the paper recommends a KL pull toward the original student, which removed E2B's copying damage at no cost in held solves (λ = 1; 187: prose ΔCE −0.04, KL 0.016, held 207 vs SFT 204). Only λ = 1 was run.

Design: arms sft_self_s_l03 and sft_self_s_l3 in xlearn.py: sft_self_s, the shortest rows plus KL to the student with the LoRA off, at λ = 0.3 and 3. E2B student, seeds 1-3, one Colab session per seed (xq_learn.sh STAGE=e2b, XL_ARMS=sft_self_s_l03,sft_self_s_l3, launcher dreamer/xq_side22_launch.sh). Each session re-scores 186's sft; λ = 1 is 187's sft_self_s, paired through stored per-problem results; drift for every adapter.

Predictions (pooled over seeds 1-3):
1. Prose KL from the original falls with λ: λ 0.3 > λ 1 (0.016) > λ 3 (80%).
2. Training problems fixed fall with λ: λ 0.3 > λ 1 (90.7 per seed) > λ 3 (70%).
3. Held solved within ±10 of each other across the three strengths (65%).
4. Prose ΔCE at λ = 0.3 lies between λ 1's −0.04 and SFT's +0.69, above +0.1 (55%).

Bar for "the KL pull removes copying damage across strengths": at λ = 0.3 and λ = 3, held solved within 10 of SFT (204) and prose ΔCE at most +0.15.

**193 result (2026-11-08, 16:55; xqlKL1-3, all exit 0).** Each session's re-scored 186 sft and E2B alone matched the stored results exactly (487 of 487, every seed), so λ = 1 (187's sft_self_s) pairs exactly. Pooled over seeds (E2B alone: 171 held solved):

| Arm | Held solved (of 564) | Fixed (of 453) | Lost (of 615) | Prose KL | Prose ΔCE | Prose entropy change |
|---|---|---|---|---|---|---|
| sft | 204 | 359 | 47 | 0.29 | +0.69 | −0.76 |
| KL to the original, λ 0.3 | 210 | 324 | 45 | 0.031 | +0.006 | −0.09 |
| KL to the original, λ 1 (187) | 207 | 272 | 40 | 0.016 | −0.035 | −0.01 |
| KL to the original, λ 3 | 182 | 179 | 73 | 0.014 | −0.045 | +0.04 |

Paired (exact sign test, pooled): λ 0.3 vs λ 1, held 34 vs 31 (p = 0.80), trained 94 vs 48 (p = 0.0001). λ 3 vs λ 1, held 23 vs 48 (p = 0.004), trained 39 vs 156. λ 3 vs sft, held 34 vs 56 (p = 0.026). Prose ΔCE per seed: λ 0.3 −0.013, +0.023, +0.007; λ 3 −0.056, −0.041, −0.040.

Predictions: 1 held (prose KL 0.031 > 0.016 > 0.014). 2 held (fixed 324 > 272 > 179). 3 failed (held 210, 207, 182: λ 3 is 25 below λ 1). 4 failed (λ 0.3's ΔCE is +0.006, not above +0.1: the light pull removes the damage too). 2 of 4.

Bar ("removes copying damage across strengths": held within 10 of sft's 204 and prose ΔCE at most +0.15 at both λ 0.3 and λ 3): not met. λ 0.3 passes both parts (210; +0.006). λ 3 removes the damage but loses 22 held solves, fixes half as many and loses more solved problems than sft (73 vs 47).

Reading: the protection does not need a strong pull. λ 0.3 keeps the student as close (prose ΔCE about zero, KL a tenth of sft's) and teaches more than λ 1 (52 more fixes, held level). Too strong a pull (λ 3) both blocks learning and breaks solved problems. The recommendation becomes a light pull, λ 0.3 to 1; λ 0.3 is the better tested point (three seeds, E2B only).

## Addendum 194 (2026-11-08, 15:40; built, in review, not yet run): do coding losses mean forgetting or verbosity? A longer-cap re-grade (user chose it from the Fable review: "Greedy + 768-cap conflates 'forgot how to solve' with 'got verbose'. Even a 1,536-cap re-grade of the lost set")

Why: 30% of lost answers had no closing fence at the 768-token cap, and lost answers ran long (median 1,456 characters against 535). A "lost" problem may be one the trained student still knows how to solve but now answers too verbosely to finish.

Design: no training. For each of the 11 Qwen recipes of Figure 5 and seeds 1-3, the stored adapter regenerates every problem it lost (solved by Qwen alone, failed by the recipe, both greedy at 768 new tokens): 665 problems in total, 37-115 per recipe pooled. Generation is greedy with 1,536 new tokens; each answer is scored alone in the sandboxed scorer. xlearn.py regrade, xq_learn.sh STAGE=regrade (Qwen only, E4B tokenizer), lost lists from the stored per-problem results (xq/regrade/lost.json), adapters at xq/prior/sS. One Colab session per seed. Measured per recipe: recovered (now passes), answers longer than 768 tokens, and recovered among those.

Predictions (pooled):
1. At least 25% of the 665 lost problems are recovered (55%).
2. Runbook rows, whose answers are short, recover a smaller share than plain SFT (65%).
3. The weak-teacher on-policy arms (opd_c, mix_c), whose answers were short but lost the most, recover under 20% (60%).

Reading rule, set now: if at least half the lost problems are recovered, the paper calls most coding losses verbosity at the cap, not forgetting. Below a quarter, they are mostly real forgetting. In between, both, and the paper reports the share.

**Amendment 194a (2026-11-08, 16:20, before any re-grade run; user: "go ahead and fix 194").** A code review found that 194 as built regenerated each recipe's lost problems in new batches. Greedy outputs change with batch composition alone: Qwen alone, generated in two different batchings, differed on 144 of 495 texts and 16 pass results. The lost set was chosen for failures, so a regeneration would "recover" some of them by chance. Only 176 of the 665 lost answers (26.5%) actually hit the 768-token cap (no closing fence). The other 489 are complete answers that failed; identical batches reproduce them exactly (495 of 495 across sessions), so more room cannot change them.

Revised design:
- Only the 176 capped answers are re-graded. Each stored adapter continues its own stored answer by up to 768 more tokens (1,536 in all), greedy. The input is the prompt's ids followed by the stored answer's ids, tokenized apart as when it was generated. Then stored answer + continuation is scored, one problem at a time. The 489 complete lost answers stay failed. Input scratchpad regrade/capped.json, uploaded as xq/regrade/capped.json; code xlearn.py regrade() and gen_continue(); sessions xqlRG1-3, unchanged otherwise.
- Fidelity check before the run (CPU, fp32, Qwen3-1.7B alone, two prompts): the re-tokenized prefix equals the original ids, and prefix + continuation equals the uninterrupted greedy text, in both prompts. On GPU in bf16 batches, small numerical differences may still change a continuation.
- What the capped answers look like (inspected before the run; no outcome is known): 136 of 176 stop inside a comment or docstring line, so the student was writing explanations when it ran out of room. 31 (18%) look like loops (a line repeated at least 3 times in the last 12, or the last 300 characters repeated earlier). Capped / lost per recipe, pooled: sft 29/58, sft_s 12/48, sft_c_s 12/52, opd_c 10/94, mix_c 21/115 (11 loop-like), opd_cf 34/75, rw_cf 19/55, mix_cf 26/52, rb 1/39, rb_c 1/37, sft_self_s 11/40.

Predictions (replacing 194's):
1. At least 40% of the 176 capped answers close their code block within the 768 extra tokens (60%).
2. Between 15% and 40% of the 176 pass after the continuation (55%).
3. Of the 31 loop-like answers, at most 2 pass (75%).
4. Pooled over all 665 lost answers, at most 10% are recovered (70%), so the paper's "lost" counts are mostly real failures.
5. The recipe with the highest recovered share of its lost answers is opd_cf or mix_cf (55%).
6. rb and rb_c recover at most 1 answer between them (90%).

Reading rule: if at least 10% of the 665 are recovered, the paper reports lost counts with and without the recovered answers and says that part of "lost" is verbosity at the cap. If under 5%, the Limitations sentence says the cap barely changes the lost counts. In between, the paper gives the number. Recipe rankings in the paper change only if a recipe's corrected lost count moves past another's by more than the noise level stated in the paper (about 10 problems).

**194a result (2026-11-08, 16:55; xqlRG1-3, all exit 0, 4-13 min each).** Every capped answer re-tokenized to exactly 768 prefix tokens. Pooled: 104 of the 176 capped answers (59%) closed their code block within 768 more tokens, 72 ran to the new cap, and 59 (34% of capped) passed. That is 8.9% of all 665 lost answers. Of the 31 loop-like answers, 2 passed.

| Recipe | Lost | Capped | Closed | Recovered | Share of lost | Lost after |
|---|---|---|---|---|---|---|
| mix_cf | 52 | 26 | 21 | 13 | 25.0% | 39 |
| opd_cf | 75 | 34 | 24 | 14 | 18.7% | 61 |
| rw_cf | 55 | 19 | 16 | 10 | 18.2% | 45 |
| sft | 58 | 29 | 15 | 8 | 13.8% | 50 |
| sft_s | 48 | 12 | 9 | 3 | 6.2% | 45 |
| sft_c_s | 52 | 12 | 6 | 3 | 5.8% | 49 |
| sft_self_s | 40 | 11 | 4 | 2 | 5.0% | 38 |
| mix_c | 115 | 21 | 7 | 4 | 3.5% | 111 |
| opd_c | 94 | 10 | 2 | 2 | 2.1% | 92 |
| rb | 39 | 1 | 0 | 0 | 0% | 39 |
| rb_c | 37 | 1 | 0 | 0 | 0% | 37 |

Predictions: 1 held (59% closed). 2 held (34% passed). 3 held (2 of 31). 4 held (8.9%). 5 held (mix_cf, 25%). 6 held (rb + rb_c: 0). 6 of 6.

Reading (pre-registered rule: between 5% and 10%, the paper gives the number): most losses are real failures, not verbosity. Verbosity matters for the refit-teacher recipes (mix_cf, opd_cf, rw_cf: 18-25% of their losses) and plain SFT (14%). Rankings move only for mix_cf, whose losses fall from 52 to 39, level with runbook rows (39, 37) and KL to the original (38). The other orderings in the paper hold within the stated noise.

## Addendum 195 (2026-11-08, 18:16; designed, not yet built; for the per-user graft paper, not the Kaggle one): lottery-ticket grafts of a fine-tuning change (user: "what if we find a small network and extract it into a graft? https://arxiv.org/pdf/1803.03635"; "yes, pre-register 195 and queue it after 192")

Why: copying E4B's answers taught E2B but damaged untaught text (186: prose ΔCE +0.69), and a KL pull fixed it (187, 193). The lottery ticket hypothesis (Frankle and Carbin, 2018, arXiv 1803.03635) says a dense network holds sparse subnetworks that carry its function. Panigrahi et al. (2023, arXiv 2302.06600) found that about 0.01% of a fine-tuned language model's weights carry over 95% of the fine-tuning gain, and that grafting only those onto the pretrained model kept the skill with better calibration and less interference between tasks. Ansell et al. (2022, arXiv 2110.07560) retrain only such a ticket (LT-SFT) and add the sparse diffs together. If the coding skill sits in a small part of the change, a sparse graft could keep the gain while leaving the rest of the model, and so untaught text, alone. It would also give small, stackable per-user grafts. A sparse diff is not a LoRA, so this is for the graft paper (191), not the Kaggle submission.

Design, stage 1 (no training; the 4090 after 192, before the RSI pilot):
- Sources, per seed 1-3: 186's E2B sft adapter (the damaging one) and 187's E2B sft_self_s (shortest rows + KL to the original, λ 1).
- The change: ΔW = (alpha / r) · B·A for every LoRA target matrix, merged in fp32 and cast to bf16 (W = W0 + M ⊙ ΔW).
- The ticket: the top k of |ΔW| entries over all target matrices together (global magnitude, as in the lottery-ticket papers), k ∈ {100%, 10%, 1%, 0.1%} for both sources, plus 0.01% for sft.
- Each graft, the 100% merge (the reference) and E2B alone are scored in the same run: greedy, 768 tokens, the 299 trained and 188 held problems as in 186. Problems where an arm and E2B alone disagree are re-scored one at a time. Drift (KL, ΔCE, entropy over the reply tokens) is measured on the same held untaught rows as drift.py.
- Also recorded: the share of the change's squared norm carried by each ticket, and how the ticket spreads over layers and matrices.
- Code: transfer/sparse_graft.py (new; reviewed and given a CPU test before the run). Runner: dreamer/x195_4090.sh. It waits for the exact ONLINE_REP3_DONE line, then the GPU free twice. The RSI runner is relaunched to wait for 195's end line, and x184 is relaunched unchanged, so nothing competes for the GPU.
- Comparisons are within the run (the 100% merge is the reference), because the 4090's dequantized E2B may differ slightly from the Colab copy the adapters were trained on.

Predictions (pooled over seeds; gain = held solved minus E2B alone):
1. The top 1% carries at least 30% of the sft change's squared norm (70%).
2. The sft 1% graft keeps at least 70% of the 100% merge's held gain (55%).
3. The sft 1% graft's prose ΔCE is at most half of the 100% merge's (60%).
4. The sft 0.01% graft keeps at most 30% of the held gain (65%).
5. The KL-anchored change can be sparsified more safely: its 1% graft keeps at least 70% of its own held gain (60%).
6. Fixed training problems fall faster with sparsity than held gain: at 1%, sft keeps a smaller share of its fixes than of its held gain (55%).

Success bar for "a lottery-ticket graft is a forgetting fix": some k ≤ 1% keeps at least 80% of sft's held gain with prose ΔCE at most a third of the 100% merge's, pooled over the three seeds. If met, stage 2 retrains only the ticket from the original E2B (LT-SFT), with and without the KL pull, against LoRA at the same dose. If not met, the reading is that coding skill learned by copying is spread through the change, which is also worth reporting for per-user grafts.

## Addendum 196 (2026-11-08, 19:24; designed, not yet built): online setup D live on held agent tasks (user: "does D also work for other models we have? as a lora, etc?"; "yes, set up D's live test ahead of 195")

Why: D (the cross-model corrector as online teacher, with keep + replay) gave the best net result on the 220-decision gate in two seeds (+19, +17; C +8; the self-corrector control +8). The gate scores single next-token decisions. Twice before, a LoRA that looked better on such offline gates did worse in live episodes by copying the teacher's habits: S4b looped (addendum 131), and the 31B-bridged C arm resolved 14/40 against the direct teacher's 17/40 (addendum 158). D's output is a plain rank-16 LoRA on E2B, the Kaggle format, so the question is whether its gain survives full agent episodes.

Design (the 4090 after 192's exact ONLINE_REP3_DONE line, with the GPU free twice; before 195, whose runner is relaunched to wait for this one):
- Model: QAT E2B as Kaggle serves it (google/gemma-4-E2B-it-qat-w4a16-ct, revision 971342c) on the 4090's vLLM (port 8000, never 8080), under the competition alias, through dreamer/s4_plain.sh (now taking the model repo and revision as parameters; E4B by default).
- Arms, full strength as trained and gated: plain E2B (submissions/s1t1024); C's LoRA, seed 1 (lora_onl_direct_keep_replay); D's LoRA, seed 1 (lora_onl_bridge_keep_replay). The LoRA arms use submissions/s1t1024_s4b with the adapter served as s4b.
- Tasks: live40 + conf40, 80 held-out-repo tasks of train_s1 from 12 repositories. All 12 are in the online test's held split, none in its training stream or the corrector's fitting data. G4_TIME_SCALE=3 as in every earlier live arm.
- Two attempts per arm, run in the order plain, D, C, plain, D, C, so time-of-day effects spread over the arms.
- Measures: resolved, and the loop gate of addendum 156 (dreamer/live_loops.py: replies cut at the token cap, episodes with more than one submit_patch, episodes ending in an error).
- Context: plain QAT E4B (the teacher) resolved 5-10 of live40 per attempt and 5 of conf40, so E2B's rate will be low and differences of a few tasks are noise.

Predictions (160 task-attempts per arm):
1. Plain E2B resolves at most 20 of 160 (70%).
2. D passes the loop gate: replies cut at the cap, multi-submit episodes and error endings each at most plain's count + 6 (65%).
3. D resolves at least as many as C (60%).
4. D resolves at least as many as plain E2B (55%).
5. C resolves within ±8 of plain E2B (60%).

Success bar for "D's online gain carries to live episodes": D resolves at least plain + 8 and at least C + 8 over the 160 attempts (paired per task with an exact sign test reported), and passes the loop gate. If not met, the paper calls D's gain decision-level and untested or not shown in episodes.

Order afterwards: 195 (its runner waits for this runner's end line, exactly "HH:MM X196_DONE" or "HH:MM X196_HALTED", in ~/x195/logs/x196.log on the 4090), then the RSI pilot, then x184.

**Amendment 196a (2026-11-08, 20:02, before any run; user: "can we move this up so we can test it asap, because this seems like a novel way to do training now of smaller models").** 196 now runs right after the 189 queue (after D seed 3, about 00:30), ahead of 192. The runner waits for the exact ONLINE_REP_DONE line instead of ONLINE_REP3_DONE. The 4090 pass-through (online_rep2b_4090.sh) also waits for 196's end line before it writes ONLINE_REP2_DONE, so 192 starts only after 196. 195 still waits for 192 and 196, and the RSI pilot and x184 follow as before. No other change. Expected: the blend gate about 21:15, C seeds 2-3 on Colab about 23:00, D seed 3 about 00:30, 196 about 06:00-10:00.

## Addendum 197 (2026-11-08, 20:25; built, in review, not yet run): a disagreement-gated teacher, the control Fable 5.1 asked for (user: "yes, fix it", answering "Add the confidence-gated teacher control? ... on the 4090 after the live test")

Why (Fable 5.1's review of draft 10j): the corrector's fit includes a do-no-harm term that keeps E2B's own distribution wherever E2B already ranks the true token first. So D's target may be "E2B where E2B is right, E4B-informed where it is wrong", which by construction breaks fewer agreeing decisions and does not fight keep + replay. The self-corrector control shares that term and fixed only 13, so the term alone does not give the fixes. But nothing yet separates "E4B's states" from "E4B's own output, used only where E2B disagrees". The blend control mixes uniformly, so it cannot separate them either.

Design: setup D with a different target, everything else identical (code8 = code7 plus this teacher; seed 1; keep 1.0, replay every 4; small pass, 600 items, the 240-point gate; 4090). Teacher "gated": at every reply position, E4B's next-token distribution where base E2B's top token differs from E4B's, and base E2B's own distribution elsewhere (online.py ON_ONL_TEACH=gated; the share of positions taken from E4B is recorded). Queue: after 196's end line, before 192 (the 4090 pass-through now waits for 197's end line, exactly "HH:MM X197_DONE" or "HH:MM X197_HALTED" in ~/toyq/behave/logs/x197.log), then 195, the RSI pilot and x184.

Predictions (gate on the same 220 decisions; D seed 1: 23 fixed, 4 broken, +19; seed 2: 23, 6, +17):
1. The gated teacher breaks at most 6 of the 168 agreeing decisions (70%).
2. It fixes at least 4 fewer than D's two-seed mean of 23 (55%).
3. Its net is at most +15 (55%).
4. Its held CE at item 600 is between D's (0.75) and C's (0.775) (55%).

Reading rule: if the gated teacher nets at least +16 (within D's seed range minus 1), the paper says D's gain is explained by gating on E2B's disagreement, a simpler recipe needing no corrector, and the "teacher states" claim is withdrawn. If it nets at most +12, E4B's states through the corrector add something beyond gating, with the caveat of one seed. In between, the paper reports both and draws no mechanism claim.

**Amendment 197a (2026-11-08, 20:47, before the run; user: "added 100 more cu").** The gated teacher runs on a Colab A100 session (xqoG1: colab/xq_online.sh TEACH=gated CODE=code8 ON_SEED=1; code8 = code7 plus the gated teacher, online.py md5 5ad4956e39), starting when a C session frees a slot (about 22:00), instead of on the 4090 after 196. The 4090 runner online_rep4_4090.sh was stopped before it started anything, and the pass-through releases 192 right after 196 again. Seed 1 on Colab draws the same stream as D seed 1 on the 4090. The analysis uses the 4090's gate partition, as in 189c. Predictions and the reading rule are unchanged.


## Addendum 198 (2026-11-08, 22:05; built, in review, not yet run): where does the corrector teacher's pull fall? (user: "yes, queue the disagreement check after the live test")

Why (a review of 189 interim 4): the blend matched the corrector teacher's average pull from base E2B (0.064 nats, a fifth of E4B's 0.323) with a uniform mix, so it rules out "a uniformly softer target", not "a selective one". If the corrector pulls mostly where E4B and base E2B disagree, it does what the gated teacher (197) does by rule. 197 tests that by training; this checks directly where the pull goes, before any training.

Design (transfer/pull_split.py; forward passes only): the blend calibration's own 24 points and base E2B samples, reproduced exactly as online.py's blend_alpha draws them (the 24 points after the 600-item stream in the same permutation; two samples per point, 256 tokens at most, torch seed 189). At every sampled token: the pull from base E2B, KL(base ‖ target), of four targets: the corrector teacher (D's), E4B (A and C), the pull-matched blend (w = 0.433) and the gated teacher (E4B where E4B's top token differs from base E2B's, so its pull is E4B's there and 0 elsewhere). Split by that disagreement (the gated teacher's own mask): each target's share of its pull on disagreement tokens and its pull per disagreement token over per agreement token (selectivity); at disagreement tokens, how often each target's top token becomes E4B's and the probability it gives E4B's top token; at agreement tokens, how often the top token changes; closeness to E4B, KL(E4B ‖ target). 95% intervals from a point bootstrap (the 24 points resampled, 2000 times). Code9 = code8 + pull_split.py (online.py unchanged, md5 checked). Queue: the 4090, after 196's end line, before 192; a small pass on 2 points first; about 10 minutes. The runner online_rep2c_4090.sh replaces the pass-through online_rep2b_4090.sh and writes ONLINE_REP2_DONE after 198, also if 198 fails.

Predictions:
1. The run reproduces the blend calibration: the corrector's and E4B's mean pulls within 2% of 0.0637 and 0.3229 (75%).
2. Disagreement tokens are under 25% of all tokens (70%).
3. At least half of E4B's pull falls on disagreement tokens (65%).
4. The corrector puts a larger share of its pull on disagreement tokens than the blend does (55%).
5. At disagreement tokens, the corrector's top token becomes E4B's more often than the blend's does (60%).
6. The gated teacher's mean pull is at least twice the corrector's, so the gated teacher is not "near on average" (70%).
7. At agreement tokens the corrector changes base E2B's top token on under 3% of tokens (65%).

Reading rule (set before the run):
- Selective: the corrector's share of its pull on disagreement tokens exceeds the blend's by at least 0.10, with the bootstrap interval of the difference above 0. Then the corrector acts partly as a disagreement gate, and 197 decides whether the gate alone is enough.
- Directional: at disagreement tokens, the corrector moves the top token to E4B's more often than the blend, interval above 0. Then the corrector aims at E4B's choices more than a uniform mix with the same average pull.
- If neither holds, the corrector's first-step pull is placed and aimed like the blend's at the token level, and D's gain over the blend is not explained by where the pull falls. The paper then reports that and leaves the mechanism open.
- This is one corrector, one set of 24 points, base E2B only (the student before any training). It describes D's target at the start, not during the run. It cannot say whether the corrector is necessary; only 197 can.

**Amendment 189e / 197b (2026-11-08, 21:51; relaunch, before either run produced anything).** Neither Colab session of 189d/197a ran: xqoD3 got an A100 at 20:50, but the HF token upload failed at 20:54 (colab_run exit 4; the session was gone by 21:50, no compute left running), and xqoG1 was refused an A100 six times from 20:55 to 21:00 (exit 3; probably the account's session limit while the dead D3 session still counted). Found at 21:50. The relaunch script dreamer/xq_onl3_launch.sh starts both again with the same commands, code, seeds and settings (G1 first, at 21:51; D3 when a slot frees, the C2/C3 sessions ending about 22:15), and retries a failed start up to 3 times, 10 minutes apart. The failed attempts' folders are kept as runs/xqoG1_f1 and runs/xqoD3_f1. Predictions and reading rules are unchanged.

**Amendment 198a (2026-11-08, 21:59; review changes, before the run).** An independent review found no bugs in pull_split.py (reproduction of blend_alpha's points, seed and samples; every per-token quantity; memory at most 7 full-vocabulary tensors, as blend_alpha). In the runner it found that a missing code9 would exit before writing ONLINE_REP2_DONE and block 192 forever, and that the old pass-through (online_rep2b_4090.sh) had to be stopped first or 192 and 198 could load at once. Changes: the log is defined first and a missing code9 writes the marker; both pull_split.py calls run under `timeout 2h`; the summary line uses the project's Python; the JSON now also holds the point estimates the reading rule compares (the corrector-minus-blend differences, the gated/corrector pull ratio and the reproduction's % difference). Deployed 21:59: code9 = code8 (online.py md5 5ad4956e39) + pull_split.py (md5 eee1b4ab3c); online_rep2b_4090.sh stopped by its exact PID with no ONLINE_REP2_DONE line written; online_rep2c_4090.sh started and waiting for 196's end line. Predictions unchanged.

**189, interim 5 and 197, result (2026-11-09, 05:25; read before D seed 3 and before 196 ended).** C seeds 2 and 3 (Colab A100, xqoC2/xqoC3, ended 01:09 and 01:19) and the gated teacher (197, Colab A100, xqoG1, seed 1 = D seed 1's stream, ended 04:50), scored on the 4090's partition (52 differing, 168 agreeing) as 189c and 197a set:

| Setup (seed, GPU) | Fixed / 52 | Broken / 168 | Net | Held CE at 600 |
|---|---|---|---|---|
| C, seed 1 (4090) | 14 | 6 | +8 | 0.775 |
| C, seed 2 (A100) | 30 | 3 | +27 | 0.768 |
| C, seed 3 (A100) | 24 | 5 | +19 | 0.769 |
| D, seed 1 (4090) | 23 | 4 | +19 | 0.751 |
| D, seed 2 (4090) | 23 | 6 | +17 | 0.749 |
| Gated teacher, seed 1 (A100) | 21 | 5 | +16 | 0.764 |

Hardware noise: E2B alone's agreement bit differs between the A100 and the 4090 on 3 of the 220 decisions in each of the three sessions (189c's added prediction, at most 6, held). On each session's own partition (51/169) the nets are +26, +18 and +15.

Paired (all 220 decisions; D only right, this only right, exact sign test): C seed 1 against C seed 2 2 vs 21 (p < 0.001) and against C seed 3 4 vs 15 (p = 0.019). D seed 1 against C seed 2 5 vs 13 (p = 0.10), against C seed 3 8 vs 8 (p = 1.0), against the gated teacher 7 vs 4 (p = 0.55). D seed 2 against C seed 2 4 vs 14 (p = 0.031).

189b predictions on C seeds 2-3 and the revised success bar: C's three seeds net +8, +27 and +19 (mean +18), D's two net +19 and +17 (mean +18). D does not beat C across seeds. C seed 1 was the low draw, and the seed-to-seed range of C (19) is larger than any D-C gap at seed 1 (11). Confound, stated now: seeds 2 and 3 of C ran on the A100 and seed 1 on the 4090, so seed (stream order, initialisation, samples) and GPU are not separated. D seed 3 (A100, running since 01:27) gives the same-GPU, same-seed comparison with C seed 3. Held CE still orders D (0.749-0.751) below C (0.768-0.775): D stays closer to the teacher on held text; it does not fix more decisions.

197 predictions: 1 (breaks at most 6) held, 5. 2 (fixes at most 19) failed, 21. 3 (net at most +15) failed, +16. 4 (held CE between 0.75 and 0.775) held, 0.764. Reading rule: the gated teacher nets +16 on the 4090 partition, the "at least +16" branch, so D's gain at seed 1 is explained by gating on E2B's disagreement, a simpler recipe needing no corrector, and the "teacher states" claim is withdrawn. It is at the threshold (+15 on its own partition), one seed, and on the other GPU; with C seeds 2-3 the question it answers (what makes D beat C) has lost its premise.

What the paper must now say (to be written after D seed 3, about 08:30): online, with keep + replay, the corrector teacher matched the plain teacher across seeds, not beat it; at seed 1 it beat C, the self-corrector and the blend, and a disagreement-gated teacher did nearly as well. The corrector's online result is closeness to the teacher on held text (CE), not decisions. No mechanism or "states" claim.

**Amendment 189f (2026-11-09, 05:34, before the run; user: "yes, launch the C seed 1 A100 rerun").** Setup C seed 1 again, on a Colab A100 (xqoC1: colab/xq_online.sh ON_SEED=1, teacher direct, code7; launcher dreamer/xq_onl4_launch.sh = xq_onl3_launch.sh with this one session, same slot cap, lock, balance guard and retries; no new job code). Seed offset 0 draws the 4090 C seed 1's stream: the gated run (A100, offset 0) matched C seed 1's base CE in all six windows within 0.0005, while seed 2's windows differ by up to 0.10. Why: C's three seeds net +8 (4090), +27 and +19 (A100), and seed and GPU are confounded. With the stream fixed, this run measures what the GPU and run-to-run divergence (on-policy samples drift apart once bf16 arithmetic differs) do on their own. Scored on the 4090's partition, as 189c.

Predictions:
1. Base CE per window reproduces the 4090 C seed 1's within 0.002 (90%).
2. The rerun nets between +3 and +13, within 5 of the 4090's +8 (50%).
3. The paired test of the rerun against the 4090 C seed 1 (220 decisions) gives p > 0.05 (55%).
4. The gated teacher (A100, same stream) nets more than the rerun (55%).

Reading rule: if the rerun nets at most +13, the stream explains C seed 1's low result, the GPU does not lift results, and D's seed-1 lead over C, the self-corrector and the blend is a stream-1 result (one stream of three). If it nets at least +19 (D seed 1's level), run-to-run variation on one stream is as large as the D-C gap: the seed-1 contrasts are within run noise, and the paper reports D and C as tied with no seed-1 contrast claim. In between: report both, no claim. Either way, cross-GPU paired tests are labelled as such.

**Amendments 189g / 197c (2026-11-09, 05:45, before any of these runs; user: "shouldnt we also run d on the a100 and see how that looks? and gated teacher should have 2 more seeds").** Three more Colab A100 sessions (launcher dreamer/xq_onl5_launch.sh; xq_online.sh, the same code, inputs, models and settings as 189c/189d/197a): D seed 1 (xqoD1, TEACH=bridge, code7, offset 0, the 4090 D seed 1's stream), then the gated teacher seed 3 (xqoG3, code8), then seed 2 (xqoG2). Then seed 1 has C (189f), D and the gated teacher on one GPU and stream, and so does seed 3 (C seed 3, D seed 3, gated seed 3); seed 2 has C and the gated teacher on the A100 and D on the 4090. Balance guard 50 CU per session (a run is about 7 h at about 6.7 CU/h, about 47 CU), so with D seed 3 and the C seed 1 rerun running, D seed 1 starts only once the balance reaches 150 CU. Scored on the 4090's partition, as 189c.

Predictions:
1. D seed 1 on the A100 nets within 5 of the 4090's +19 (55%).
2. The gated teacher's mean net over its three seeds is within 4 of D's mean net over its seeds (55%).
3. The gated teacher's held CE at item 600 is above D's in every seed with both (65%).
4. On the A100, D's net minus C's is within ±8 at seed 1 and at seed 3 (60%).

Final reading rules (replace 197's one-seed rule once these are in; same-GPU pairs first):
- D vs C: D beats C online only if D nets at least 6 more than C on average over the same-GPU pairs and is ahead in each; otherwise the paper reports them as tied, with the seed-to-seed range.
- D vs gated: the corrector adds something beyond gating only if D nets at least 6 more than the gated teacher on average over the same-GPU pairs and is ahead in each; if the gated teacher is within 3 of D on average, gating explains D's decision gains; in between, no claim.
- Held CE is reported for every run beside the nets; a closeness claim needs D below both others in every same-GPU pair.

## Addendum 199 (2026-11-09, 07:06; built, in review, not yet run): the corrector scaling check (user: "its also possible that maybe training the corrector on a lot more text may make it better overall"; "yes, pre-register and queue the corrector scaling check")

Why: the online correctors (no-penalty gall: gall_e4b reads E4B's states and is D's teacher; gall_e2b reads E2B's own and is the self-corrector's) were fitted for only 600 steps. The self-corrector teacher matched D on held CE (0.755 vs 0.751) and on the prequential gain (0.063 vs 0.064), so E4B's states add almost nothing at this fit. Either the correctors are under-fitted and E4B's states would add more with more fitting, or this is their ceiling. Earlier evidence that teacher quality matters: refitting the coding self-corrector (0.70 → 0.49 nats on E4B's solutions) lifted on-policy distillation from 116 to 148 held solves.

Design (no online training; corrector fits and forward passes only): both correctors fitted again, 4x longer (2,400 steps, the same lr, cosine schedule, noise, depth dropout, do-no-harm 0.5, no size term, draw weights and seed), on data4 = data3's rows unchanged plus extra training rows from transfer/build_mix_extra.py: 443 more Python standard-library code chunks (from files that gave build_mix.py no held chunk, line offsets disjoint from its chunks; code pool 338 → 781) and 79 more story texts (189 → 268); +38% tokens (586,584 → 808,137). Prose (all 300 local passages already used), questions and agent rows (the s3 pool) cannot grow from local text, so the agent text is the same, seen about 4x as often. Every held row is unchanged (checked). The calibration rows are the 1x fit's (online.py ON_CAL_POOL = data3's training-row count; code10 = code9 + this change; the whitening means of the 1x and 4x cross correctors are compared as a check). Then, with code10, the agent rung (held agent points: whole-reply cross-entropy to E4B for E2B alone, cross, self) and the genericity suite (held rows of every kind) for the 4x pair and again for the 1x pair. Stored 1x rung (code4): alone 0.8606, cross 0.754, self 0.7833 (self minus cross 0.029). Runner dreamer/scale199_4090.sh on the 4090: after 198 (ONLINE_REP2_DONE) and before 192 (online_rep3b_4090.sh replaces online_rep3_4090.sh and waits for this runner's exact end line "HH:MM SCALE199_DONE", which every exit path writes); a small pass first; about 7.5 h.

Predictions (rung, whole reply, cross-entropy to E4B; 1x as re-evaluated by code10):
1. The 4x cross corrector is at least 0.01 below the 1x cross (55%).
2. The 4x self corrector is at least 0.01 below the 1x self (50%).
3. The margin (self minus cross) grows by at least 0.01 (35%).
4. Suite: the 4x cross corrector is lower than the 1x cross on agent rows and on at least 3 of prose, code, story, question (50%).
5. Suite: mean harm over kinds (KL from E2B alone where E4B and E2B alone agree) rises by at most 25% (60%).

Reading rule: if the cross corrector improves by less than 0.005, this fitting is saturated, and only more agent text (new E4B rollouts) remains to test. If both improve by at least 0.01 and the margin grows by less than 0.01, the correctors were under-fitted, but E4B's states add no more with more fitting. If the cross improves by at least 0.01 and the margin grows by at least 0.01, E4B's states add more with more fitting, and an online D run with the 4x corrector is the next test. 199 says nothing about decisions; only an online run can.

**Amendment 199a (2026-11-09, 07:13; review changes, before the run).** An independent review found one data bug: 5 of the 79 extra stories were near-copies of held stories (versions of one item share their opening; 36-65% of their 16-token windows occur in a held story row), which would have raised the held stories with a near-copy in the fitting mix from 7 of 24 (1x) to 12 of 24 and tilted the story comparison toward 4x. build_mix_extra.py now drops any extra story with more than 20% of its 16-token windows in a held story row: exactly those 5 (74 extra stories; 517 extra rows, 221,211 tokens). It confirmed: ON_CAL_POOL reproduces the 1x calibration rows and the rng state after them; train(), gate_rung() and gate_mix() are unchanged since the 1x fit's commit; append_extra.py is correct; every runner exit path writes SCALE199_DONE; nothing writes into data3 or the 1x correctors; the old online_rep3_4090.sh was already stopped (07:02). Applied smaller points: a time limit on every step (fits 5 h, others 1 h), an EXIT trap that writes SCALE199_DONE if the runner dies, a fixed-date READY deadline, an assert on ON_CAL_POOL and the pool size saved in the checkpoint. Noted: with fixed draw weights the 4x fit makes about 480 code draws over 781 code rows, so the extra text is only partly seen; 4x steps and more text are not separated (the reading rule says "4x fitting", not "more text"). The runner was restarted (07:13) with these changes, and code10/READY set.

**196, result (2026-11-09, 07:06 end; read 07:20).** Online setup D's LoRA (seed 1), C's LoRA (seed 1) and plain QAT E2B, live on the 80 held-out-repo agent tasks, two attempts each (160 task-attempts per arm):

| Arm | Resolved (a + b) | Tasks solved at least once | Replies at the cap | Episodes submitting 2+ times | Episodes ending in an error (timeouts) |
|---|---|---|---|---|---|
| Plain E2B | 5 (4 + 1) | 4 | 59 / 2,418 | 4 | 10 (1) |
| D | 11 (5 + 6) | 9 | 18 / 3,869 | 3 | 37 (27) |
| C | 5 (3 + 2) | 4 | 33 / 4,072 | 25 | 25 (20) |

Paired by task (more solves of the two attempts; exact sign test): D over plain 7 vs 3 (p = 0.34), D over C 6 vs 1 (p = 0.13), C over plain 3 vs 3. Predictions: 1 (plain at most 20) held; 2 (D passes the loop gate) failed: D's error endings are 37 against plain's 10 (bar 16), mostly the 24-minute session timeout (27), while its cut replies (18 vs 59) and repeat submits (3 vs 4) are fewer; 3 (D at least C) held; 4 (D at least plain) held; 5 (C within 8 of plain) held. Success bar (D at least plain + 8 and C + 8, and the loop gate): not met. Reading: D's decision-level gain may carry to episodes (11 vs 5 and 5; 9 tasks against 4), but the counts are small and not significant, and D's episodes run longer (60% more replies than plain) and time out more. Both LoRA arms time out far more than plain (27 and 20 against 1); part of that may be LoRA serving being slower under vLLM, which this test does not separate. The paper reports D's gain as decision-level, with this live result as suggestive only.

**198, result (2026-11-09, 07:13 end; read 07:25).** On the blend calibration's 24 points and 48 base E2B samples (10,987 tokens; reproduced exactly: corrector pull 0.0637, E4B 0.3229, 0.000% and −0.002% off):

| Target | Mean pull | Share of pull on disagreement tokens | Selectivity | Top token becomes E4B's (disagreement) | Top token changes (agreement) | KL(E4B ‖ target), disagreement / agreement |
|---|---|---|---|---|---|---|
| Base E2B | 0 | | | 0 | 0 | 0.983 / 0.151 |
| Corrector (D) | 0.064 | 0.49 | 3.9 | 0.26 | 0.023 | 0.626 / 0.111 |
| Blend (w 0.43) | 0.064 | 0.57 | 5.3 | 0.40 | 0 | 0.312 / 0.051 |
| Gated | 0.195 | 1.0 | | 1.0 | 0 | 0 / 0.151 |
| E4B | 0.323 | 0.60 | 6.2 | 1.0 | 0 | 0 / 0 |

Disagreement tokens are 19.7% of all. Point-bootstrap 95% intervals: corrector minus blend, share of the pull on disagreement tokens −0.079 [−0.102, −0.054]; top-token adoption −0.135 [−0.162, −0.106]; gated pull / corrector pull 3.06 [2.79, 3.40].

Predictions: 1 (reproduction within 2%) held; 2 (disagreement under 25%) held, 19.7%; 3 (at least half of E4B's pull there) held, 60%; 4 (the corrector's share above the blend's) failed, 0.49 vs 0.57; 5 (the corrector adopts E4B's top token more often than the blend) failed, 0.26 vs 0.40; 6 (gated pull at least 2x the corrector's) held, 3.1x; 7 (the corrector changes base E2B's top token on under 3% of agreement tokens) held, 2.3%.

Reading rule: neither "selective" (the difference is negative, its interval below 0) nor "directional" (likewise). The corrector teacher is less selective than a uniform blend of the same average pull, follows E4B's top choice less often, and sits further from E4B than the blend at disagreement tokens (0.63 vs 0.31). It is also the only target that changes base E2B's top token where E2B and E4B agree (2.3%). D's target is not "E4B where E2B is wrong". It is a broad, mostly-E2B shift learned from the fitting text, consistent with the self-corrector matching D on held CE and on the prequential gain. The gated teacher, which is pure selectivity at 3x the pull, matched D's decisions at seed 1 (197) by a different route. So D's decision gains, where they appear, are not explained by selectivity; its CE gains look like the corrector's text fit, not E4B's states.

## Addendum 200 (2026-11-09, 08:34; built, in review, not yet run): how much of each corrector can E2B compute, and an imitable corrector (user: "run the regression check on the 4090 - lets test the imitable corrector also, and see how well it does")

Why (reports/Corrector layer placement evidence.md, ranks 1 and 6): the E4B-reading corrector beats the self-corrector by 0.01-0.10 nats as a model (0.029 on the rung), but students taught through the two end only 0.004 apart. If the E4B-specific part of its writes is computation E2B cannot do from its own states, no LoRA placement or loss can store it (imitation gap). An earlier bound (amendment 163a, the old hint translator): a ridge from E2B's own states predicted 69%/50% of the bridge's state change at layers 17/22, against 81%/71% of its control's.

Part 1, realizability (transfer/realize.py; forward passes and small fits): for a corrector pair, on held rows (the rung's 120 agent points, 40 each of prose, code, story, question; up to 96 reply positions per row), targets per corrector: the layer-12 change, the additions at 13 and at 14 (the last layers whose writes reach E2B's shared KV), the additions at 15-34 summed, and each target's difference cross minus self (the E4B-specific part). Features: E2B's own states from the pass without a corrector, f12 (layer 12), f22 (12, 17, 22), fall (12, 14, 17, 22, 28, 34). Ridge (lambda on a validation split) and a one-hidden-layer MLP (2048, all targets jointly, early stop), split by row (every 5th test, the next validation); test R² pooled over dimensions, also agent vs the rest. Run on the 1x pair and on 199's 4x pair.

Part 2, the imitable corrector (online.py ON_IMIT_BETA; code11): the 1x cross fit (data3, 600 steps, no size term, same seed) plus beta x the mean squared part of its whitened writes that a ridge from E2B's own states does not predict: the layer-12 change from layer 12, the additions (averaged over sites) from layers 12, 17 and 22. The ridge is refitted every 10 steps from exponentially decayed statistics of the corrector's own detached writes (after 50 warm-up steps), so the penalty pushes the writes toward what E2B's states predict while the cross-entropy keeps pulling toward E4B. Beta 1 and 10 (the report proposed 0.1 and 1; at this scale 0.1 x a residual of about 0.03 would be noise against cross-entropy differences of 0.03). Each with the rung, the suite and realize.py. Then online setup D, seed 1 (the 4090, the same stream, keep 1.0, replay every 4, the 240-point gate), with the larger beta whose rung margin over the self-corrector is at least 0.01 (else beta 1). Runner dreamer/imit200_4090.sh after 199 (SCALE199_DONE) and before 192 (online_rep3c_4090.sh now waits for "HH:MM IMIT200_DONE"); small passes first; about 7 h.

Predictions:
1. The self-corrector's writes: MLP test R² from f22 at least 0.8 for the layer-12 change (its inputs are these states) (70%).
2. The cross corrector's writes: MLP R² from f22 between 0.5 and 0.85 for the layer-12 change (55%).
3. The E4B-specific part (cross minus self), layer-12 change: MLP R² from fall at most 0.3 (55%).
4. 199's 4x pair: the E4B-specific part is less predictable than the 1x pair's (MLP R² from fall lower by at least 0.05) (50%).
5. The beta 1 imitable corrector keeps a rung margin over the self-corrector of at least 0.01 (60%); beta 10 does not (55%).
6. The online student taught by the imitable corrector: held CE at item 600 within 0.005 of D seed 1's 0.751 (55%); gate net within 5 of D seed 1's +19 (60%).

Reading rules: R² of the E4B-specific part at most 0.3 means E2B cannot compute the corrector's E4B lead: stop rewiring for transfer (the report's rank 1). At least 0.5 means it can: LoRA placement and a hidden-state loss are worth a run. The imitable corrector helps transfer if its student ends at least 0.005 closer to E4B on held CE than D seed 1's (0.751; the student measure reproduces within about 0.002 across seeds) and nets at least D's +19 minus 2 on the gate; if held CE is within 0.005, imitability makes no difference at this dose. One seed; decisions noisy.

**189, interim 6 (2026-11-09, 08:35): D seed 3 (Colab A100, xqoD3, ended about 08:20).** Scored on the 4090 partition: fixed 18 of 52, broken 3 of 168, net +15 (own partition 17 / 3; E2B alone's agreement bit differs from the 4090's on 3 of 220, as in every A100 session). Held CE at item 600 0.752 (D seeds 1-2: 0.751, 0.749); untaught drift (KL from base, prose / code / story) 0.042 (seeds 1-2: 0.017, 0.017; C seeds 2-3 on the A100 0.012, 0.014); prequential gain over base 0.062 (seeds 1-2: 0.065, 0.063; C 0.035-0.038).

The same-GPU, same-seed pair (189g's first rule): D seed 3 against C seed 3 (+19), 7 vs 11 on the 220 decisions (p = 0.48), 3 vs 9 on the 52 differing (p = 0.15). D's three seeds net +19, +17, +15 (mean +17); C's +8, +27, +19 (mean +18). Against the gated teacher (seed 1, A100) 4 vs 5. So far: D is not ahead of C on decisions in either same-GPU comparison it has (seed 3: behind by 4); its nets are narrower (range 4 vs C's 19); its held CE and prequential gain lead C's in every seed. The seed-1 A100 pair (C and D reruns, about 13:00) completes 189g's D-vs-C reading.

**Amendment 200a (2026-11-09, 08:45; review changes, before the run).** An independent review found no code bug in the penalty, realize.py or the runner (shapes, gradient only through the writes, nothing changes without ON_IMIT_BETA, memory, the rung's own agent points, every exit path writes IMIT200_DONE, the D commands match). One mismatch with the text above: realize.py drew its positions from the whole written span (context span and reply), not the reply; it now draws reply positions only, as pre-registered. Applied smaller points: the penalty's feature standardisation is now the exact mean and sd over the 50 warm-up steps, then frozen, with no ridge statistics gathered before (it was an EMA that drifted under the accumulated statistics); the decay 0.99 applies per micro-step (two per step), so the ridge remembers about the last 50 steps; realize.py also reports R² with every dimension standardised (Gemma's large activations could dominate the pooled raw R²); the small pass checks that the penalty became active (the last logged unpredicted share above 0); the runner refuses to start over existing beta fits; an assert that the penalty sees E2B's layers 17 and 22. Noted: the penalty covers the layer-12 change and the additions averaged over the 22 sites, so unpredictable parts that cancel across sites go unpenalised, while realize.py scores 13, 14 and 15-34 separately. The beta fits should match the 1x fit's log (tr0_gall_e4b.log) up to step 49, before the penalty starts; that is checked after the run. 192's runner was switched to online_rep3c_4090.sh at 08:35 (online_rep3b stopped first).

**Amendment 200b (2026-11-09, 09:28; logging only, before the run).** The "harm" in corrector fit logs is the do-no-harm term, a cross-entropy from E2B alone's top-32 distribution (tail lumped) to the corrected model at positions where E2B alone and E4B agree. So it contains E2B's own entropy there, which varies with the rows drawn, plus the drift. Fit logs now also show "harm KL" (the term minus E2B's own entropy, which is the KL between the two 33-bucket distributions, checked on random distributions to 1e-7 and 0 when nothing changes). This adds a sixth entry to the fit history and changes no loss, draw or output. It applies to the beta fits of addendum 200 (code11/online.py redeployed; the runner has not started). The 1x and 4x fits (199) log the old term only, so their drift is read from the suite's harm (the KL on fixed held rows). 1x suite harm, cross / self: agent 0.039 / 0.031, E2B's own sample 0.042 / 0.031, prose 0.19 / 0.15, code 0.081 / 0.069, story 0.16 / 0.13, questions 0.23 / 0.11, fresh questions 0.15 / 0.096.

**Amendment 197d (2026-11-09, 09:37; user: "use the g4 if a100 is not aviable").** The gated seeds 3 and 2 kept getting no A100 (exit 3 no runtime at 08:25 and 08:33, "Service Unavailable" on retries) and then waited on the balance guard until 100 CU were added. dreamer/xq_onl6_launch.sh now runs them with a GPU fallback: each start tries an A100, and if Colab has none (or the start fails) a G4 at once; 3 rounds, 10 minutes apart; the same job, code8, slot counter, lock and balance guard. xq_onl5's seed-3 waiter was stopped by PID (it held no session or lock); its seed-2 attempt (on A100) is left to finish, and onl6 takes seed 2 over only if that ends without a runtime. Hardware: a gated seed on a G4 is a third GPU type; it is scored on the 4090's partition like the others, its hardware noise (E2B alone's agreement flips against the 4090) is reported, and its comparisons with the A100 C and D seeds are labelled cross-GPU (197c's reading rules put same-GPU pairs first).

**199, interim (2026-11-09, 10:52): the 4x cross corrector's rung, before the 4x self fit.** Fitted 2,400 steps (10:46; last-25-step means: change size 0.28, gate 0.31; the 1x ended at 0.084, 0.39). Agent rung (120 held points), against E2B alone, with code10 (the 1x pair re-scored with code10 reproduced its stored rung exactly):

| | E2B alone | 1x cross | 4x cross | 1x self |
|---|---|---|---|---|
| Whole reply, cross-entropy to E4B | 0.861 | 0.754 | **0.714** | 0.783 |
| Whole reply, top-1 agrees with E4B | 0.828 | 0.866 | **0.884** | 0.854 |
| Tool choice after thinking matches E4B | 0.937 | 0.937 | **0.958** | 0.937 |
| Decision without thinking, where alone differs: now on E4B's choice (of 23) | 0 | 0.39 | **0.52** | 0.39 |

Prediction 1 (4x cross at least 0.01 below 1x cross) held: 0.040 lower, closing 37% more of E2B alone's gap to E4B (0.147 against 0.107). The 4x cross reaches about what the full-connection R3 translator reached (0.711-0.720; four read depths up to 95%, replacing E2B's layer-12 state), with the 1x design's mid reads and gated additions. Not yet known: whether the 4x self corrector improves as much (predictions 2-3, the margin; fit 10:51 to about 14:10) and whether the larger writes cost generality (the suite's harm and the other kinds, about 15:00; R3's agent gain came with worse-than-alone prose, code, stories and questions).

## Addendum 201 (2026-11-09, 11:02; built, in review, not yet run): online setup D with the 4x corrector (user: "Either way though, this seems like it could help with distilling knowledge cross models, without making them drift too much... can we test the agentic workflow and see how it does with the 4x corrector?")

Why: 199's 4x cross corrector reached 0.714 on the rung (1x 0.754, alone 0.861), as close as the full-connection R3 translator, with the 1x design. Whether a closer teacher gives a better student is the open question (the corrector students so far kept little of the 1x corrector's E4B-specific lead).

Design: online setup D, seed 1 (offset 0: the same stream as every seed-1 run), the 4x cross corrector (data4/out/gall_e4b.pt, 2,400 steps, sha256 80c7ccdf; uploaded to the private job repo as on/out4x/gall_e4b.pt) as the teacher, everything else as D (code7, keep 1.0, replay every 4, the small pass, 600 items, the 240-point gate), on a Colab A100 (G4 if none: amendment 197d's fallback), session xqoD4x, launcher dreamer/xq_onl7_launch.sh. colab/xq_online.sh gains CORR=4x (default 1x; with 1x nothing changes); the session checks the corrector's sha256. Starts after gated seed 3 has a running job, so the gated seeds keep their place; about 7 h on an A100. Compared with the seed-1 A100 reruns: D with the 1x corrector (189g) and C (189f), same GPU type and stream; scored on the 4090's partition.

Predictions (against the D 1x A100 rerun):
1. Held CE at item 600 at least 0.005 lower (60%).
2. Mean prequential gain over base higher (65%).
3. Gate net at least as high (50%).
4. Untaught drift (KL from base on prose / code / story at item 600) at most 1.5x (55%).

Reading rule: the 4x corrector transfers more to the student if its student's held CE is at least 0.005 lower than the D 1x rerun's and its gate net is no more than 3 below. If held CE is within 0.005, a closer teacher does not give a closer student at this dose. One seed; decisions noisy. If it transfers, a live test of its LoRA (as 196) is the next step, on the 4090.

**Amendment 201a (2026-11-09, review, before the run).** An independent review found no bug: with CORR unset the job script behaves exactly as before (gated seed 3 will use it), CORR=4x passes only with TEACH=bridge CODE=code7, the file on the job repo has the expected sha256, the checkpoint loads strictly into code7's translator and passes bridge_setup's config check, the gate does not use the corrector, and the launcher touches only runs/xqoD4x*. xq_onl7_launch.sh started; it waits for gated seed 3's running job.

## Addendum 202 (2026-11-09, 11:37; built, in review, not yet run): the 4x-corrector D LoRA live (user: "ok, so the real test is going to be on the live check"; "yes, queue the live test after the imitable run")

Design: 201's LoRA (online setup D, seed 1, 199's 4x corrector as teacher; Colab session xqoD4x, about 20:30) live on 196's 80 held-out-repo agent tasks, on QAT E2B as Kaggle serves it (s4_plain.sh, vLLM on the 4090, rollouts on box), two attempts (160 task-attempts), then the loop gate. Only the new arm runs; it is compared task by task with 196's arms on the same tasks and harness (plain 5/160, D 11/160, C 5/160), with exact sign tests. Caveats stated now: 196's arms ran yesterday night, the 1x D LoRA was trained on the 4090 and this one on an A100, and the counts are small (196's D over plain, 11 vs 5, was p = 0.34). Runner dreamer/x202_live.sh on the Mac: waits for 201's adapter and its session's end, copies the adapter to the 4090, waits for 200's end line and the GPU free twice; gives up at 2026-11-10 03:00. It writes exactly "HH:MM X202_DONE" or "HH:MM X202_HALTED" to the 4090 queue log (an EXIT trap writes HALTED if it dies), and 192's runner (online_rep3d_4090.sh, replacing online_rep3c) waits for that line. About 4 h.

Predictions (160 task-attempts):
1. The 4x D LoRA resolves at least as many as 196's D (11) (50%).
2. It resolves at least plain + 8 (13) (35%).
3. Its episodes ending in an error are at most D's 37 (55%).
4. Tasks where it solves more than D, against fewer, is at least even (50%).

Reading rule: as 196's success bar, a live gain needs at least plain + 8 and a loop gate within plain's counts + 6. Anything smaller is reported as a count with its sign test, and not as a gain. If 201's online result showed no transfer (held CE within 0.005 of the D 1x rerun), the live test still runs (the user's call) and is read the same way.

**Amendment 202a (2026-11-09, 11:42; review changes, before the run).** An independent review found two bugs and a launch hazard, all fixed before launch. (1) 192's new runner waited only for the X202 line, so an early X202_HALTED (no Colab GPU for 201, a failed copy) could have started 192 beside 200 on the 24 GB card; online_rep3d now waits for both IMIT200_DONE and the X202 line. (2) x202_live.sh would have waited until 03:00 if 201 ended with no adapter in a way the launcher does not call "gave up" (a failed run, a lost session, the launcher done); it now halts at once in those cases. (3) online_rep3c was stopped (PID 597522, args checked) before online_rep3d started. Also applied: the GPU wait has the 03:00 deadline too (before the test starts); a relaunch removes an old adapter copy before copying. Noted: to stop x202_live.sh, kill its process group so s4_plain.sh's trap stops vLLM on port 8000. Checked by the review: the arm command is 196's D arm with only NAME and LORA changed, the marker lines and quoting are right, the per-task script tolerates missing arms. x202_live.sh started at 11:42 (Mac, PGID 40806), online_rep3d at 11:42.

**Amendment 192b (2026-11-09, 11:47; before either run; user: "move 192 to colab").** On the 4090, 192 would have run from about 02:00 to 09:00 on Nov 10, the freeze day. Its two arms now run on Colab (dreamer/xq_onl8_launch.sh; A100, G4 if none): online SFT (xqoS1: colab/xq_online.sh TEACH=sft, ON_ONL_TEACH=sft, keep 1.0, replay every 4; small pass first) and offline SFT (xqoF1: TEACH=sftoff, online.py lora train ON_LORA_LOSS=ce ON_LORA_POOL=stream, about two passes over the 600 stream items; small pass first). Each gets the 240-point gate. Both run seed 1 on code7, the same code and commands as online_rep3_4090.sh, so they pair with the seed-1 A100 reruns of C and D (same GPU type and stream), scored on the 4090's partition. They start after the 4x online run (201) has a running job, so the gated seeds and 201 keep their places: online SFT when gated seed 2's slot frees (about 17:00), offline SFT at the next free slot (about 20:30). On the 4090, online_rep3e_4090.sh replaces online_rep3d and only passes ONLINE_REP3_DONE on to 195 after 200 and 202. 192's predictions and reading rule are unchanged.

**Amendment 192c (2026-11-09, 11:53; review, deployment).** An independent review found no bug: the existing job modes behave exactly as before (simulated for gated seed 3 and the 4x run), the sftoff and sft modes reproduce online_rep3_4090.sh's commands and file names at code7, the early exit still yields JOB_EXIT 0 and the output download, and the 4090 pass-through writes the exact ONLINE_REP3_DONE line only after 200 and 202. Deployed: online_rep3d (PID 600657, args checked) stopped and online_rep3e started on the 4090; xq_onl8_launch.sh started on the Mac. Noted: at 196 CU and about 20 CU/h, the balance guard (50 CU per session) may hold the SFT sessions back past their estimated starts.

## Addendum 203 (2026-11-09, 12:12; built, in review, not yet run): dense rank-64 LoRAs toward the 4x corrector (user: "Yes, go ahead", to "pre-register it now as a conditional run, ready to go after the live test")

Why: in the Qwen pair (169c), a rank-64 LoRA trained with a dense objective (the corrected model's outputs over the whole written span, 1,000 steps) kept about a quarter of E4B's specific lead with the corrector off, 2-3x the output-only forms; rank alone did nothing (164: rank 16 and 64 fixed 20 and 21 of 52). The online students here keep about a seventh. This tests that form on E2B with 199's 4x corrector.

Design (online.py train_lora, ON_LORA_LOSS=dense; code12 = code11 + this mode): rank-64 LoRAs (alpha 128, MLP projections at layers 12-34, the corrector's write zone; lr 3e-4, cosine; 1,000 steps x 2 rows of the agent training pool, seed 7) trained with a forward KL (E4B-style top 32 + a tail bin) over the reply and up to 512 context positions before it, all inside the corrector's written span, along E4B's logged replies. Three teachers, one LoRA each, in this order: the 4x cross corrector (199), E4B itself, and the 4x self-corrector (if 199 finished it). Each gated on the 240 points (the 4090 partition natively). Runner dreamer/dense203_4090.sh on the 4090 after 200 (IMIT200_DONE) and 202 (X202 line), before 195 (online_rep3f_4090.sh now also waits for "HH:MM DENSE203_DONE", which every exit path writes); a small pass first; skipped if not started by 06:00 on Nov 10, and stops after an arm that ends past 09:30. About 5-6 h. It runs unless the user cancels it after tonight's results (the regression check and 201).

Predictions (held CE = the gate's cross-entropy to E4B along E4B's replies):
1. The E4B-target LoRA has the lowest held CE of the three (65%).
2. Retention, (self-target CE minus 4x-target CE) / (4x self minus 4x cross on 199's rung), is at least 0.2 (45%).
3. The 4x-corrector LoRA nets at least as many gate decisions as the E4B-target LoRA (50%).
4. Each of the three nets more than D seed 1's online LoRA (+19) (40%).

Reading rule: retention at least 0.25 means the dense rank-64 form stores the 4x corrector's E4B-specific lead as well as in the Qwen pair; under 0.15, no better than the online students. A decision advantage of one teacher over another needs at least 6 more net and an exact sign test p < 0.05 on the 220 decisions. One seed.

**Amendment 203a (2026-11-09, 12:21; review changes, before the run).** An independent review found an out-of-memory risk that would most likely have failed the small pass (about 265 of the 1,839 agent rows reach about 24.5 GB on the 24 GB card at 512 context positions): the bridge teacher now uses teacher_lp (E4B run with one logit kept, so E4B's full log-softmax is never built), and the student's logits are freed before the backward pass, saving about 3 GB at the peak; if the small pass still fails, the runner retries it once at 256 context positions and then uses 256 for every arm (logged). Second, the three LoRAs did not start from the same weights (only the corrector arms drew random numbers before the LoRA was built); torch.manual_seed(203) now precedes it in the dense mode. Third, online_rep3e was stopped (PID 601052, args checked) before online_rep3f and dense203_4090.sh started; code12/READY set. Checked by the review: the corrector hooks act only in the teacher pass, the student pass has the adapter on and no corrector write, teacher and student positions align inside the written span, the kd and ce modes are unchanged, gate_lora loads a rank-64 LoRA on layers 12-34, and every runner exit writes DENSE203_DONE.

**189, interim 7 (2026-11-09, 13:26): the seed-1 A100 pair (189f, 189g: xqoC1 ended 12:13, xqoD1 12:46).** Same stream as the 4090 seed-1 runs (offset 0); scored on the 4090 partition (E2B alone's agreement bit differs on 3 of 220 in both, as in every A100 session). C seed 1: fixed 23 of 52, broken 7 of 168, net +16 (the 4090 run: +8); held CE 0.774 (4090 0.775), drift 0.014, prequential gain 0.040. D seed 1: fixed 21, broken 3, net +18 (4090: +19); held CE 0.752 (4090 0.751), drift 0.014 (4090 0.017), prequential gain 0.065 (4090 0.065).

189f's question, what the GPU and run divergence do on their own with the stream fixed: C moved 8 net decisions (14 vs 6 discordant against its 4090 run, p = 0.12; 20 of 220 decisions changed), D moved 1 (4 vs 5; 9 changed). The 4090 C seed 1 was a low draw; D's result reproduced.

189g's first rule, D against C on one GPU and stream: seed 1 9 vs 7 on the 220 decisions (p = 0.80; 3 vs 5 on the 52 differing); seed 3 (interim 6) 7 vs 11. Pooled over both same-GPU pairs, C fixes more of the 52 (14 vs 6, p = 0.12) and D breaks fewer of the 168 (4 vs 10, p = 0.18); neither is significant. D's seed-1 lead on the 4090 (17 vs 6, p = 0.035) did not reproduce: it came from C's low draw. Reading: on decisions, C and D tie; D's held CE (0.749-0.752 in four runs, against C's 0.768-0.775) and prequential gain (0.062-0.065, against 0.035-0.040) lead in every run. Against the gated teacher (seed 1, A100, +16): D 4 vs 2, C 9 vs 9; 197's reading (gating explains D at seed 1) stands on one GPU.

**Result 199 (2026-11-09, 14:20; the 4090, 14:17).** Held rung (120 agent points, whole reply, cross-entropy to E4B; E2B alone 0.861): 1x cross 0.754, self 0.783 (margin 0.029); 4x cross 0.714, self 0.768 (margin 0.054). Top-1 agreement 4x cross 0.884, self 0.860 (1x 0.866, 0.854). Tool choice after thinking 0.958 cross, 0.937 self (alone 0.937). Of the 23 decisions where E2B alone differs from E4B, the corrected model now takes E4B's choice on 12 (4x cross) and 10 (4x self), against 9 and 9 at 1x.

Predictions: 1 (4x cross at least 0.01 below 1x cross) held, -0.040; 2 (4x self at least 0.01 below 1x self) held, -0.015; 3 (margin grows by at least 0.01) held, +0.025; 4 (4x cross below 1x cross on agent and at least 3 of prose, code, story, question) held, on all five; 5 (mean suite harm rises by at most 25%) failed: the cross corrector's harm (KL from E2B alone where E4B and E2B alone agree, mean over the seven kinds) rose 30% (0.127 to 0.165), the self-corrector's 50% (0.089 to 0.134). 4 of 5.

Suite, self minus cross cross-entropy (1x then 4x): agent 0.028, 0.055; prose 0.075, 0.181; code 0.049, 0.088; story 0.064, 0.159; questions 0.010, 0.158; fresh questions 0.104, 0.163; agent with E2B's own sample 0.035, 0.069 (these rows are sampled afresh in each run, so compare within a run). The cross corrector's lead over the self-corrector roughly doubled on every kind of held text, including kinds it was not mostly fitted on. The 4x self-corrector gained on agent rows (0.788 to 0.774) but not elsewhere (prose 2.476 to 2.479; questions 0.538 to 0.621, worse). The writes grew: layer-12 change on agent rows 0.22 to 0.34 (cross), 0.16 to 0.22 (self). A sham (the cross corrector fed other rows' E4B states) now costs far more (agent CE 0.93 to 1.43), so the 4x corrector leans harder on the E4B states it reads.

Reading (199's rule): the cross corrector improved by at least 0.01 and the margin grew by at least 0.01, so E4B's states add more with more fitting, on held rows of every kind; the 1x correctors were under-fitted, and the E4B-specific part was the part most under-fitted. An online D run with the 4x corrector is the next test (201, running on Colab). Early in 201 (items 0-75, same items as D seed 1), the student stays further from its 4x teacher (learn KL 0.058-0.085 at items 25-75, against 0.022-0.049 for the 1x teacher; about half to two-thirds of the starting gap left, against a third) and is no closer to E4B (held CE 0.773 at item 50, D seed 1 0.770). Whether E2B can compute the larger E4B-specific part is 200's realizability check (the 4x pair), starting now.

**Amendment 200c (2026-11-09, 14:25; before these runs).** 200's runner ended at 14:20, at step 1: the imitable fit's small pass ran out of memory in the backward pass (21.5 GiB allocated, 1 GiB more needed for the logits' gradient). Step 0 (the longest row, the penalty still in its warm-up, no optimiser state yet) had passed; the penalty, which read each site's write through the padded full-length copy, gave the backward pass 22 full-length gradient buffers to hold while the logits' gradient was built. realize.py's small pass had passed, so nothing of steps 2-4 ran. Changes: (1) step 2 (realizability of the 1x and 4x pairs) runs now, unchanged (dreamer/real200_4090.sh, code11, from 14:22; ends "REAL200_DONE"). (2) code13 = code12 + online.py: the penalty reads the unpadded writes (the same values, no padded copies), and the small pass trains the longest row again once the penalty is on (with the optimiser state present), and logs its peak memory. (3) dreamer/imit200c_4090.sh then runs step 1's imitable small pass and step 3 (beta 1 and 10 cross correctors, data3 rows, 600 steps, no size term, each with the rung, the suite and realize.py) as pre-registered; beta 10 starts only before 18:00, so the GPU is free before 202. (4) Step 4 (online D with the chosen imitable corrector, about 6 h) no longer fits on the 4090 before the Nov 10 freeze (202 and 203 hold it overnight); it runs only if the user moves it to Colab. Predictions and reading rules unchanged. In review.

**Result 200, part 1: realizability (2026-11-09, 14:28; the 4090, real200, 14:22-14:27; 19.7k reply positions, 3.8k test, split by row).** MLP test R² (ridge in brackets) of each write from E2B's own states (fall = layers 12, 14, 17, 22, 28, 34; f22 = 12, 17, 22), pooled over dimensions:

| Pair | Self w12 (f22) | Cross w12 (f22) | E4B-specific w12 (fall) | E4B-specific a13 / a14 / a15+ (fall) | E4B-specific, agent / other rows (a15+) |
|---|---|---|---|---|---|
| 1x | 0.85 (0.73) | 0.62 (0.58) | 0.53 (0.49) | 0.53 / 0.51 / 0.52 | 0.59 / 0.31 |
| 4x | 0.71 (0.58) | 0.48 (0.49) | 0.37 (0.41) | 0.45 / 0.44 / 0.40 | 0.49 / 0.18 |

Predictions: 1 (self w12 at least 0.8) held, 0.85; 2 (cross w12 between 0.5 and 0.85) held, 0.62; 3 (E4B-specific w12 at most 0.3) failed, 0.53; 4 (the 4x pair's E4B-specific part less predictable by at least 0.05) held, 0.37 vs 0.53. With every dimension standardised, the MLP's w12 R² is strongly negative (dimensions with almost no training variance); it is not used.

Sizes (mean squared norm per position, raw): the E4B-specific layer-12 change grew from 48 to 119 with 4x fitting, the summed additions at 15-34 from 348 to 1,027. Split with the R² above, the part E2B's states predict grew about 1.7x (w12 26 to 44) and the part they do not about 3.3x (23 to 75).

Reading (200's rules): at 1x the E4B-specific part is about half predictable from E2B's own states (0.49-0.53, at the 0.5 line: LoRA placement and a hidden-state loss are worth a run); at 4x it is 0.37-0.45, between the two rules. Most of what the longer fit added is the kind E2B's states do not predict, and on non-agent rows the E4B-specific part is mostly unpredictable (0.18-0.31). This fits 201's early items (the student closes a smaller share of the 4x teacher's gap) and says the 4x teacher's extra is mostly information E2B does not compute, not only a capacity gap; 203 (a rank-64 LoRA with a dense target) tests the capacity side. Caveat: these probes read six E2B layers with a one-hidden-layer MLP on 12k positions; a trained E2B could compute more than they find, and R² in state space does not weigh the parts by their effect on the output.

**Amendment 200c, review changes (14:39; before the fits).** An independent review confirmed the diagnosis and the same values, and found that keeping the unpadded writes for the penalty also kept them alive through the backward pass (half the saving, and extra memory for every fit). Now: they are taken out of the shared slot at once and freed before the backward pass; the padded copies, which the hooks re-read when gradient checkpointing recomputes the layers, are stored in bf16, the type the hook casts them to anyway (half their memory, the same values); with no size term (ON_TR_SIZE=0, as here) the size term's average is detached (its gradient is multiplied by zero; this frees the 22 additions it saved). The small pass logs peak allocated and reserved memory. The runner now caps every step's timeout at the time left before 20:25 and skips a step that cannot fit (a fit with under 75 min left, an evaluation with under 10), restores 200's no-vLLM check, and gives up waiting for the GPU at 18:55. code13 deployed (online.py md5 dc241ba8); imit200c started at 14:38.

## Addendum 204 (2026-11-09, 14:47; built, in review, not yet run): one more online teacher, chosen by 200's beta-1 result (user: "yes, set that up", to "use one Colab slot: run the imitable corrector online if beta 1 keeps its lead over the self-corrector, otherwise the 4x self-corrector")

Why: 201's early items and 200's realizability both say the 4x cross teacher's extra is mostly not absorbed, and mostly not computable from E2B's own states. Two teachers would tell whether a learnable teacher transfers better: the beta-1 imitable cross corrector (keeps E4B's signal, penalises the part E2B's states do not predict) and the 4x self-corrector (199's data4/out/gall_e2b: fitted as long as 201's teacher, reads only E2B's own layers 17 and 22, so all it adds is computable by E2B). One Colab slot (about 47 CU) is free for one of them.

Rule, fixed now: when 200c's beta-1 rung is in (data5_b1/out/gate_rung_gall.json on the 4090), if its whole-reply margin over the self-corrector (control minus bridge) is at least 0.01 (200's prediction 5), run online D with the beta-1 corrector (xq_online.sh TEACH=bridge CORR=imit1); otherwise, or if beta 1 did not finish by 19:00, run the self-corrector setup with the 4x self-corrector (TEACH=control CORR=4xself). Seed 1 (offset 0, the stream of every seed-1 run), keep 1.0, replay every 4, 600 items, the 240-point gate, code7 (as 189g, 201), Colab A100, G4 if none. The chosen file goes to the private dataset at on/out204/ with its sha256 checked in the session. Launcher dreamer/xq_onl9_launch.sh, session xqo204; it starts only once both 192b sessions have begun (their launcher does not count this session), within the 3-session cap and the balance guard; no start after 23:30 (so it ends before about 07:00 on Nov 10).

Predictions, imitable arm (200's prediction 6, with the same-GPU reference D seed 1 A100: held CE 0.752, net +18): held CE at item 600 within 0.005 of 0.752 (55%); gate net within 5 of +18 (60%). Reading as 200: imitability helps transfer if the student ends at least 0.005 closer to E4B than 0.752 and nets at least +16; within 0.005, it makes no difference at this dose.

Predictions, 4x self arm: 1. held CE at item 600 at most 0.755 (the 1x self-corrector's student) (55%); 2. held CE within 0.005 of 201's 4x cross student, or below it (50%); 3. gate net within 5 of +18 (55%); 4. untaught drift at item 600 at most 0.03 (60%). Reading: if this student ends within 0.005 of 201's or closer to E4B, the 4x cross teacher's E4B-specific extra did not reach its student (with 200: information E2B does not compute); if it ends at least 0.005 further from E4B than 201's, part of the extra transferred. If it ends at least 0.005 closer than D seed 1 (at most 0.747), a self-corrector fitted longer is a better teacher with no E4B states at all.

One seed; decisions noisy; scored on the 4090 partition with paired sign tests against D seed 1 (A100) and 201.

**Amendment 204a (14:57; review changes, before the launch).** An independent review found two launcher bugs, both fixed: (1) the wait for the 192b sessions accepted any ended status, including the "no runtime" ends their launcher retries (and one session giving up), so 204 could have started while a 192b retry was still to come, and that launcher does not count 204: a fourth session. It now waits until each 192b session has started a job, given up, or ended for good. (2) That wait had no deadline; it now ends at 23:30 like the start. Also: the 19:00 read is retried once after 200c ends and once after 19:00 (a rung written in the last minutes, or one failed read, could have been taken as no rung), only a rung written by 19:00 counts, the margin is checked to be a number, the deadlines carry seconds, and the copy has a connect timeout. xq_online.sh: no findings (both checkpoints pass code7's loaders' checks; the replacement file is the one sha-checked). Launcher started.

**197, interim (2026-11-09, 14:58): gated teacher seed 3 (xqoG3, Colab G4, ended 14:56).** Scored on the 4090 partition (E2B alone's agreement bit differs on 3 of 220, as on the A100): fixed 21 of 52, broken 9 of 168, net +12 (own partition 19 / 10). Held CE at item 600 0.766 (D seed 3 0.752, C seed 3 0.769); untaught drift 0.015; prequential gain 0.044; the share of positions taught from E4B 0.19. Paired on the 220 decisions: against D seed 3 (A100) 8 vs 11 (p = 0.65), C seed 3 11 fewer than C's 12 wins (5 vs 12, p = 0.14), gated seed 1 6 vs 10. Not a same-GPU pair (G4 against A100). D minus gated so far: seed 1 (A100, same GPU) +2, seed 3 (cross-GPU) +3. Prediction 3 (gated held CE above D's in every seed with both) holds so far (0.764, 0.766 against 0.752, 0.752). The final reading (197c: same-GPU pairs first) waits for seed 2 (A100, about 17:00; D seed 2 ran on the 4090).

## Addendum 205 (2026-11-09, 15:19; built, in review, not yet run): the seed-1 controls and D seed 2 again, off the 4090 (user: "yes, update what needs to be updated, and rerun stuff that would be valuable", after a reread of the paper against the results)

Why: the reread found that the paper's online claim ("teacher states did not help online; decisions did not separate") leaves out 189's own control results. At seed 1 on the 4090, D beat the self-corrector 12 vs 1 on the 220 decisions (p = 0.003; interim 3) and the blend 11 vs 2 (p = 0.022; interim 4), and 189's reading then was that the decision gain needs E4B's states. Since then C seed 1, which D also beat on the 4090 (17 vs 6), netted +16 instead of +8 when rerun on an A100 (interim 7), and D tied it there. The self-corrector and the blend have only their 4090 runs (13 and 14 fixed of 52, like C's 14 there; every run off the 4090 fixed 18-30). So their decision results are either real or the same low draw; no run tells. Second, seed 2's D-vs-C pair (4 vs 14, p = 0.03) is D on the 4090 against C on the A100.

Design: three Colab sessions with colab/xq_online.sh, the same code7, inputs, models, correctors and settings as 189c/189g (keep 1.0, replay every 4, 600 items, the 240-point gate; A100, a G4 at once if Colab has none):
- xqoSC1: the self-corrector as the teacher, seed 1 (TEACH=control; offset 0, the stream of every seed-1 run).
- xqoD2: setup D, seed 2 (TEACH=bridge ON_SEED=2), so seed 2 has C, D and the gated teacher on one GPU type.
- xqoBL1: the pull-matched blend, seed 1 (TEACH=blend; the weight is calibrated in the session by 189's procedure and reported; the 4090's was 0.433).
Launcher dreamer/xq_onl10_launch.sh, which also takes over 204 from xq_onl9 (amendment 204b below). Order as slots free: the self-corrector, 204, D seed 2, the blend. 3 sessions at most, lock and balance guard as before; no 205 session starts after 03:00 on Nov 10. Scored on the 4090 partition; hardware flips reported; a session that lands on a G4 is reported as cross-GPU, like gated seed 3.

Predictions:
1. The self-corrector rerun nets at least +13 (5 above its 4090 run) (55%) and fixes at least 18 of 52 (50%).
2. Its held CE at item 600 is within 0.003 of the 4090 run's 0.755 (80%) and above D seed 1's 0.752 on the A100 (75%).
3. D seed 1 (A100) against the self-corrector rerun, paired on the 220 decisions: the two discordant counts differ by at most 5 (50%).
4. D seed 2 on the A100 nets within 4 of its 4090 run's +17 (60%); its held CE is within 0.003 of 0.749 (80%); against C seed 2 (A100, +27) it has the smaller discordant count (70%).
5. The blend rerun nets at least +13 (50%); its weight is within 0.05 of 0.433 (80%); its held CE is within 0.004 of 0.781 (75%).

Reading rules, fixed now (same-GPU pairs; a G4 run counts as supporting evidence only):
- Self-corrector: if the rerun nets at least +15 (within 3 of D seed 1's +18 on the A100) and D seed 1 (A100) does not beat it on the paired test (p > 0.1), the 4090 result was a low draw: the paper says the cross-model and self correctors tie on decisions on one GPU (one seed), and the "states did not help online" claim stands for decisions as for closeness. If it nets at most +10 and D beats it paired (p < 0.05), the deficit reproduced: the paper says the self-corrector's student matches D's on closeness and speed but fixes fewer decisions, so what the corrector reads from E4B matters for decisions (one seed, two GPUs). Otherwise the paper reports both runs and makes no claim either way.
- Blend: the same rule against D seed 1 (A100), for "a softer target alone does as well".
- D seed 2: with seeds 1 and 3 it gives three same-GPU D-vs-C pairs; 189g's rule applies to the three (D beats C only if it nets at least 6 more on average and leads in each; otherwise tied). The cross-GPU seed-2 pair is then dropped from the text.
One seed per control; decisions are noisy (a rerun moved C by 8 net).

**Amendment 204b (15:19; before 204 starts).** 204's launcher xq_onl9 (it had started no session and made no decision) is replaced by xq_onl10, which runs 204 unchanged (the same rule at the same 19:00 read, the same upload and sha check, the same job, no start after 23:30) but after 205's self-corrector session has begun, because that rerun decides a claim in the paper's abstract and 204 is exploratory. One launcher for all four keeps a single session count.

**Amendment 205a (15:31; review changes, before any session).** An independent review found no blocking bug in the launcher or in xq_online.sh (both control and blend follow the 4090 runner's commands exactly; code7; output names match). Changes made: (1) xq_onl9 was stopped first (PID 63977 and its sleep; it held no lock, had made no decision and started no session), or both launchers would have run 204. (2) The next session's waiter now starts only once the session before has a running job (or its waiter has ended, or 45 minutes have passed since its first start attempt): with "starting" alone, a failed A100 start could lose its slot to the next session, as happened to gated seed 3 this morning. (3) 204's 19:00 read is retried up to three times after the loop, and nothing is copied or uploaded for 204 after 23:30. Noted, unchanged: the D seed 2 and blend waiters start only after 204's decision and start, by design (the pre-registered order). xq_onl10 started at 15:31.

**Result 200, part 2, interim (15:39): the beta-1 imitable corrector (200c; the 4090, fit 14:41-15:30, rung 15:37).** The 1x cross fit with beta 1 x the part of its writes that a ridge from E2B's own states does not predict. Fit at step 599 (last 25 steps): cross-entropy 0.976 (the 1x fit 0.967 at the same steps and rows), change size 0.021 (1x: 0.084, a quarter), harm KL 0.055, unpredicted share of its writes 0.32. Held rung (120 agent points, whole reply; E2B alone 0.861; the control is the 1x self-corrector 0.783): 0.756, margin over the self-corrector 0.027 (the 1x cross corrector: 0.754, margin 0.029); top-1 0.866 (1x 0.866); of the 23 decisions where E2B alone differs, 9 move to E4B's choice (1x cross 9, self 9; 4x cross 12, 4x self 10). Prediction 5, first half (beta 1 keeps a rung margin over the self-corrector of at least 0.01): held, 0.027. So nearly all of the 1x corrector's E4B-specific lead on held agent rows survives when its writes are held to a quarter of the size and two thirds predictable from E2B's own states. Under 204's rule the online teacher is the beta-1 corrector. The suite, realize.py on it, and the beta-10 fit follow.

## Addendum 206 (2026-11-09, 15:39; built, in review; runs only on the user's go): the online teachers on one GPU type, three seeds (user: "pattern d (D Corrector + keep + replay) if that works it changes the paper, so we need to research and fix and optimize around this... I want the paper to be more about using telepathy learning")

Why: the online results sit on three GPU types (4090, A100, G4), the controls on one seed and one GPU, and a rerun on another GPU moved the plain teacher by 8 net decisions. Whether the paper can say that the teacher's states teach (the corrector fed E4B's states against the same corrector fed E2B's own) needs every teacher on one GPU type, at three seeds, at both corrector fits. A G4 session takes about 2.7 h for a 600-item run and its gate (gated seed 3), an A100 about 6.8 h.

Design: colab/xq_online.sh, code7, the same inputs, models, stream seeds and settings as 189c/189g/201 (keep 1.0, replay every 4, 600 items, the 240-point gate), every session on a G4 (no other GPU type; a session that gets no G4 waits and retries). Teachers, each at seeds 1, 2, 3:
- C: E4B's own distribution (TEACH=direct).
- D: the 1x cross corrector (TEACH=bridge).
- S: the 1x self-corrector (TEACH=control).
- D4: the 4x cross corrector (199; TEACH=bridge CORR=4x).
- S4: the 4x self-corrector (199's data4/out/gall_e2b.pt, sha256 07a097db; TEACH=control CORR=4x).
- DI: the beta-1 imitable cross corrector (200c; sha256 a6cc43ff; TEACH=bridge CORR=imit1).
18 sessions, about 25 CU each. Order: seed 1 (D4, S4, DI, S, D, C), then seed 2, then seed 3, as slots free. Launcher dreamer/xq_onl11_launch.sh, which replaces xq_onl10: 205's three reruns and 204 are run here on G4 instead (S and D seed 2 are in the list; the blend is dropped for now), with G4 references for every pair, so 205's and 204's questions are answered by same-GPU pairs at three seeds. Scored on the 4090 partition as before; hardware flips reported. The A100 and 4090 runs stay as the second- and third-GPU replication.

Measures per run: held CE at item 600 (closeness to E4B on held repositories), prequential gain (speed), untaught drift, gate fixed / broken / net.

Predictions:
1. D's held CE is below C's in all three pairs (90%), by at least 0.015 in each (80%).
2. D and C tie on decisions under 189g's rule (D not ahead by 6 on average with a lead in each pair) (75%).
3. S's held CE is within 0.006 of D's in every seed (70%).
4. S's net is within 5 of D's on average over the three seeds (55%).
5. D4's held CE is below D's by at least 0.005 in at least two of three seeds (45%).
6. D4's held CE is below S4's by at least 0.005 in every seed (40%).
7. DI's held CE is within 0.004 of D's in every seed (60%); DI's untaught drift is at or below D's in at least two of three seeds (60%).
8. D4's untaught drift is above D's in every seed (65%).
9. E2B alone's agreement bit differs from the 4090's on at most 6 of 220 in every session (85%).

Reading rules, fixed now:
- The teacher's states teach (the paper may call the method's gain a gain from the teacher's states) only if, at matched fitting, the cross corrector's students beat the self-corrector's in the pre-registered form at 4x: held CE lower by at least 0.005 in every seed (D4 vs S4), or net decisions at least 6 higher on average with a lead in each seed; and the same comparison at 1x (D vs S) does not point the other way. If D vs S and D4 vs S4 both tie on both measures (held CE within 0.005 on average, nets within 3 on average), the paper says the corrector form, not the teacher's states, gives the gain. In between, the paper reports the pairs and makes no claim.
- Dose: the corrector fitted longer is the better teacher if D4's held CE is below D's by at least 0.005 in every seed; decisions by 189g's rule.
- D vs C: 189g's rule over the three G4 pairs for decisions; a closeness claim needs D below C in every pair.
- DI: imitability helps if DI's held CE is at least 0.005 below D's in every seed or its net is at least 6 higher on average with a lead in each; within 0.005 and within 3, it makes no difference at this dose.
Three seeds per teacher; no correction for the number of comparisons; decisions are noisy.

**Amendment 206a (15:52; before any 206 session; user: "yes, go on all three").** The user approved (1) the G4 table as pre-registered in 206, (2) moving the experiment freeze from Nov 10 to the morning of Nov 11, and (3) a larger live test with the adapters merged into the weights ahead of 203 on the 4090 (addendum 207, to be pre-registered before it runs). xq_onl10 was stopped at 15:52 before it had made 204's decision or started any session (PID 69257 and its sleep; no lock held). 204 and 205 are therefore not run on the A100: their questions are answered on G4 by 206's same-GPU pairs (S and D at seeds 1-3; S4 and DI as teachers at seeds 1-3; the blend rerun is dropped). 204's and 205's predictions and reading rules are not scored; 206's replace them. The beta-1 rung margin (0.027, at least 0.01) would have picked the imitable corrector under 204's rule; it is 206's DI arm.

## Addendum 207 (2026-11-09, 15:55; built, in review, not yet run): the live test again, larger and with a null adapter (user: "yes, go on all three", to "a bigger live test ... ahead of the rank-64 run")

Why: in 196 the LoRA arms took about 2.2 times the plain arm's wall time per attempt (D 127 and 108 min, C 130 and 104, plain 55 and 48) and ended far more episodes in a time-out (27 and 20 of 160, against 1). The only difference between the plain and adapter submissions is the line "adapter: s4b" (s1t1024 against s1t1024_s4b on box), so the adapter serving path itself may be slow under vLLM, and plain E2B was not a like-for-like control for a LoRA. D's lead (11 of 160 against 5 and 5) also rests on two attempts per arm (p = 0.34 against plain) and on C's 4090 seed-1 adapter, the low draw of interim 7. (Merging an adapter into the weights, the first idea, is not how Kaggle serves a submission and would change the base from the 4-bit QAT checkpoint; the null adapter controls the same thing on the competition's serving path.)

Design: dreamer/x207_live.sh on the Mac, after 202's end line and before 203 (dreamer/dense203b_4090.sh now waits for X207_DONE or X207_HALTED; amendment 203b). The same 80 tasks (box tmp/s4/live80.txt), harness, time scale and QAT E2B checkpoint as 196. Attempts, in this order, one s4_plain.sh run each:
- N: a null adapter, D seed 1's adapter with every lora_B tensor set to zero (dreamer/x207_null.py), served and submitted exactly like a LoRA arm: plain E2B's outputs through the adapter path. Two attempts (x207_N_a, x207_N_b).
- D: the 196 adapter (seed 1, the 4090), two more attempts (x207_D_c, x207_D_d): four with 196's.
- Plain: two more attempts (x207_plain_c, x207_plain_d): four with 196's.
- C and D of the seed-1 A100 reruns (xqoC1, xqoD1: one GPU, one stream), two attempts each (x207_CA_a/b, x207_DA_a/b).
Order: N_a, D_c, plain_c, N_b, D_d, plain_d, CA_a, DA_a, CA_b, DA_b. No start after 12:00 on Nov 10; no new attempt after 19:00 (the later ones are then not run and are reported as such). Analysis (dreamer/x207_pertask.py, only attempts with all 80 task results): episodes solved, tasks solved at least once, error endings by kind, wall time per attempt, and paired comparisons by task on the solve rate (solves / attempts), exact sign tests; live_loops.py for the loop gate.

Predictions:
1. The null adapter's attempts take at least 1.5 times the plain attempts' wall time (70%).
2. The null adapter ends at least 20 of 160 episodes in an error (60%).
3. The null adapter solves at most 5 of 160 (65%).
4. D's two new attempts solve at least 8 of 160 (55%); plain's two new attempts at most 8 of 160 (80%).
5. Pooled over four attempts, D has the higher solve rate on more tasks than the null adapter (75%), with p < 0.05 (35%).
6. The A100 pair: D solves at least 8 of 160 (50%); C within 4 of 196's C, 5 of 160 (60%).

Reading rules, fixed now: the like-for-like control for a LoRA arm is the null adapter. D helps live if, pooled over its four attempts, it beats the null adapter by task (p < 0.05) and solves at least twice the null adapter's share of episodes; the paper may then say so, with the counts. If D beats the null adapter on counts but not at p < 0.05, the paper reports the counts as suggestive, as now. If the null adapter solves within 2 per 160 of D, 196's lead over plain is not evidence for D. Loop gate against the null adapter: D passes if its error endings per 160 are at most the null adapter's plus 6. If prediction 1 holds, 196's time-outs are read as a cost of the adapter path on this harness, and the paper says so. C and D from the A100 pair are reported beside the 4090 adapters; D vs C live is claimed only if D leads in both pairs and the pooled test over both has p < 0.05.

**Amendment 203b (15:55; before 203 starts).** 203's runner on the 4090 is replaced by dreamer/dense203b_4090.sh: the same commands, now also after 207's end line (X207_DONE or X207_HALTED), not started after 23:00 on Nov 10, no new arm after 03:00 on Nov 11 (was 06:00 and 09:30 on Nov 10). The old runner is stopped with SIGKILL after checking its command line, so that its exit trap does not write DENSE203_DONE and release 195.

**Amendment 206b (15:57; review changes, before any 206 session).** An independent review found no blocking bug in the launcher or in xq_online.sh (the session cap holds with the other launchers; every arm picks the intended corrector file, and the three uploaded files have the pinned sha256; seeds 2-3 write the names the script copies). Changes: the order within a seed is S, D, S4, D4, DI, C (the text above said D4, S4, DI, S, D, C; the 1x pair goes first because the paper's current claim rests on it); no session starts after 17:00 on Nov 10; every session runs a frozen copy of the job script (runs/xql/job206/xq_online.sh), so a later edit cannot change seeds 2-3; the slot kept for 192b's offline SFT session is released if its launcher is gone. v2/infra/colab/colab_run.sh now puts a time limit on its sessions, stop and usage calls (they could hang for good and hold a slot; replaced by rename, so the three running sessions keep the old file). xq_onl11 started at 15:57 with at most 3 sessions.

**Amendment 207a (16:10; review changes, before the run).** An independent review found: (1) 203b's wait had no deadline, so a 207 that never wrote its end line would have held 203, 195 and everything after for good: the wait now ends at 23:00 on Nov 10 with "203 not run". (2) x207_live.sh would have halted at once if started before the null adapter existed, and a restart after a halt would have left a stale X207 line that releases 203 beside the restarted 207: it now refuses to start while an X207 line is in the queue log, a failed ssh is no longer read as a missing file, and its marker is written in one ssh (never twice) and only after vLLM on port 8000 is gone. (3) The time-out count in x207_pertask.py matched "time", which also catches context-window errors ("Retried: 5 times"); it now matches "timeout" (on 196's files this gives 27 / 20 / 1, as recorded). Also: the pooled D-vs-C test over both adapter pairs that the reading rule names is computed; a failed attempt no longer stops the test (two failures do), and the analysis always runs; wall time is of the successful try only. The null adapter was built (27 lora_B tensors zeroed, the lora_A tensors and the config unchanged) and the A100 adapters copied with matching checksums. The swap on the 4090: the old 203 runner (PID 602231, in its wait loop, only child a sleep, no X202, X207 or DENSE203 line in the queue log) was stopped with SIGKILL and dense203b_4090.sh started at 16:10 (one process; DENSE203_DONE not written). x207_live.sh started at 16:10.

**Result 197c (16:42): the gated teacher at three seeds (seed 2: xqoG2, Colab A100, ended 16:38).** Seed 2 on the 4090 partition: fixed 21 of 52, broken 4 of 168, net +17; held CE at item 600 0.766, untaught drift 0.011, prequential gain 0.045, share of positions taught from E4B 0.19; hardware flips 3 of 220. All three seeds: nets +16 (A100), +17 (A100), +12 (G4), mean +15.0; fixed 21 in each; held CE 0.764, 0.766, 0.766; prequential gain 0.047, 0.045, 0.044. D: nets +18 (A100), +17 (4090), +15 (A100), mean +16.7; held CE 0.752, 0.749, 0.752. D minus gated: +2 at seed 1 (the one same-GPU pair), 0 at seed 2 (D on the 4090), +3 at seed 3 (the gated run on a G4). Paired on the 220 decisions: seed 1 D 4 vs 2; seed 2 9 vs 9; seed 3 11 vs 8. Against C on the same GPU and seed, the gated teacher at seed 2 is behind 1 vs 11 (p = 0.006; C seed 2 netted +27).

197c's predictions: 2 (the gated teacher's mean net within 4 of D's) held, 15.0 against 16.7; 3 (its held CE above D's in every seed) held. Final reading (197c's rule): the gated teacher is within 3 of D on average and in the same-GPU pair, so gating on E2B's disagreements explains D's decision gains; D adds closeness (held CE lower by 0.012-0.017 in every seed) and speed (prequential gain 0.062-0.065 against 0.044-0.047), not decisions. Two of the three pairs are cross-GPU; 206's G4 table does not include the gated teacher (seed 3 is its only G4 run).

**Result 200, part 2, interim 2 (16:44): the beta-10 imitable corrector (fit 15:46-16:35, rung 16:42).** Fit at step 599: cross-entropy 0.971 (1x 0.967, beta 1 0.976), change size 0.0098 (1x 0.084, beta 1 0.021), harm KL 0.054, unpredicted share 0.28 (beta 1 0.32). Held rung: 0.756, margin over the 1x self-corrector 0.028 (1x cross 0.029, beta 1 0.027), top-1 0.865. Prediction 5, second half (beta 10 does not keep a margin of at least 0.01): failed; it keeps 0.028. So the cross corrector's lead over the self-corrector on held agent rows needs writes only about a ninth the size of the 1x corrector's, of which under a third is not predictable from E2B's own states. Beta 1's realizability (realize.py, MLP from all six E2B layers): the E4B-specific layer-12 write R² 0.73 (agent rows 0.77; the 1x corrector 0.53, the 4x 0.37), the additions at 15-34 0.55 (1x 0.52); its suite matches the 1x corrector's cross-entropy on every kind (agent 0.762 vs 0.760, prose 2.41 vs 2.40, code 1.18 vs 1.17, story 2.12 vs 2.12, questions 0.53 vs 0.53) with less than half the layer-12 change (agent 0.084 vs 0.222). 200's rule for its online run picks the larger beta with a margin of at least 0.01, which is beta 10; 206's DI arm, fixed before this result, is beta 1. Beta 10 online is left for a second round after 206 (to be pre-registered), so that 206's launcher is not restarted.

**192b, interim (16:44): online SFT (xqoS1, Colab A100, seed 1, ended 16:43).** Cross-entropy on E4B's logged reply, each stream item once, keep 1.0, replay every 4 (192's arm 2), on the stream and GPU type of the seed-1 A100 reruns. On the 4090 partition: fixed 19 of 52, broken 13 of 168, net +6 (hardware flips 3 of 220). Held CE to E4B at item 600: 0.900, above the original E2B's 0.824 (C seed 1 on the A100 0.774, D 0.752); untaught drift 0.016 (C and D 0.014); prequential gain over base -0.065 (C +0.040, D +0.065): before learning each new item the SFT student predicts E4B's text worse than the original E2B. Training took 5,418 s (C 22,366 s). Paired on the 220 decisions, same GPU type and stream: D beats online SFT 18 vs 6 (p = 0.023), C 15 vs 5 (p = 0.041), the gated teacher 18 vs 8 (p = 0.076); on the 168 agreeing decisions SFT loses 3 vs 13 to D. Predictions: 2, first half (online SFT breaks at most 6) failed, 13; second half (fixes fewer than D's 23) held, 19. Predictions 1, 3 and 4 wait for the offline SFT session (xqoF1, running). Reading so far: on agent decisions, copying E4B's replies online fixes about as many decisions as distilling its distribution but breaks two to four times as many and moves the student's distribution away from E4B's on held repositories; both distillation teachers beat it on the same GPU and stream.

**Amendment 207b (17:15; before any 207 attempt).** The 4090 has been idle since 200c ended at 16:50, and 202 cannot start before 201's session ends (about 20:00-20:45). dreamer/x207b_live.sh replaces x207_live.sh: an early block runs at once, the null adapter's first attempt (x207_N_a), then plain (x207_plain_c), then the null adapter's second (x207_N_b), with no early adapter attempt started after 18:40 and no early plain attempt after 19:40; the block ends as soon as 201's session has ended (its job.log, out.tgz or adapter is on the Mac), 202 has started or ended, or an attempt has failed. The rest of 207 runs after 202's end line in the pre-registered order, skipping what the early block did. So the null adapter and a plain attempt run in the evening instead of after 202; attempts, analysis, predictions and reading rules are unchanged. A focused review of the change found two ways the early block could have collided with 202 (the GPU looks free for about a minute between two back-to-back attempts, and a blind retry after "vLLM not up"), a dropped wait for the GPU after a failed attempt, and a wait with no deadline; all fixed (the early block watches for 201's session ending, does not retry, and the waits are bounded by 19:00 on Nov 10). The old runner (PID 80723, waiting, no X207 line written) is stopped with SIGKILL so that its exit trap writes no X207_HALTED.

**Result 192b (17:55): offline SFT (xqoF1, Colab A100, seed 1, ended 17:53), with the online SFT of the interim above.** Offline SFT (cross-entropy on E4B's logged replies, the 600 stream items, 600 steps of two rows, about two passes; no keep or replay) on the 4090 partition: fixed 24 of 52, broken 10 of 168, net +14 (hardware flips 3). Cross-entropy to E4B along E4B's replies on the gate's points: 1.096, against 0.826 for the original E2B (online SFT 0.913; C seed 1 on the A100 0.784, the gated teacher 0.774, D 0.759). Paired on the 220 decisions: offline over online SFT 12 vs 4 (p = 0.077); D over offline SFT 12 vs 8 (p = 0.50; on the 168 agreeing decisions 9 vs 2), C 11 vs 9, the gated teacher 12 vs 10. Predictions: 1 (offline SFT fixes at least 20 and breaks more than D's 4) held, 24 and 10; 2 (online SFT breaks at most 6; fixes fewer than D's 23) half, 13 broken and 19 fixed; 3 (D's net at least both SFT arms') held, +18 on the A100 (+19 on the 4090) against +14 and +6; 4 (offline SFT's untaught drift above online SFT's) not measured: the offline arm writes no drift curve. Reading: copying E4B's replies reaches the distillation teachers' net on decisions only offline with two passes, and in both forms it breaks more of the decisions E2B already had right (10 and 13, against 3 for D, 5 for the gated teacher and 7 for C) and leaves E2B's distribution further from E4B's than before training (1.10 and 0.91 against 0.83), where every distillation teacher brings it closer (0.76-0.78). One seed.

**207, interim (18:59): the null adapter's first attempt (x207_N_a, the 4090, 17:18-18:57).** Plain E2B's outputs through the adapter path (D's adapter with lora_B zeroed, submissions/s1t1024_s4b): 2 of 80 resolved; 5 episodes ended in an error, 3 of them time-outs; 1,271 model calls (15.9 per task); 100 min. For comparison, 196's attempts: plain 4 and 1 of 80 resolved, 4 and 6 error endings (1 and 0 time-outs), 1,252 and 1,166 calls, 55 and 48 min; D 5 and 6 resolved, 21 and 16 error endings (14 and 13 time-outs), 2,073 and 1,796 calls, 127 and 108 min. So the adapter path alone costs about 1.9 times plain's wall time at plain's number of calls (prediction 1 holds so far), but brings 3 time-outs per 80, not D's 13-14: D's time-outs come from the slower adapter path together with its longer episodes (about 24 calls per task against 16). Prediction 2 (at least 20 error endings in the null adapter's 160) is unlikely to hold (5 in the first 80); prediction 3 (at most 5 resolved of 160) is on course (2 of 80). x207_plain_c started at 18:57 in the early block; the null adapter's second attempt runs after 202.

**206, interim 1 (19:28): the 1x self-corrector at seed 1 on a G4 (xqgSC1, 16:44-19:27).** On the 4090 partition: fixed 13 of 52, broken 3 of 168, net +10 (own partition 11 / 4; hardware flips 3 of 220). Held CE at item 600 0.756, untaught drift 0.014, prequential gain 0.063. Its 4090 run (189, interim 3): fixed 13, broken 5, net +8; held CE 0.755, drift 0.025, gain 0.063. The two runs of the self-corrector differ on 8 of the 220 decisions (5 vs 3; 10 of the 13 fixed decisions are the same). So the self-corrector's low count of fixed decisions reproduced on a second GPU type; it was not a low draw like C's 4090 seed 1 (14 fixed there, 23 on the A100). Every seed-1 teacher that uses E4B at teaching time fixed more off the 4090: C 23, the gated teacher 21, D 21 (A100), offline SFT 24. Paired against this run on the 220 decisions (cross-GPU, same stream): D seed 1 on the A100 12 vs 4 (p = 0.077; on the 52 differing 11 vs 3), D seed 1 on the 4090 13 vs 4 (p = 0.049), C on the A100 16 vs 10, the gated teacher 13 vs 7. Closeness and speed again match D's within 0.005 (0.756 against 0.752; 0.063 against 0.065). The same-GPU pair (xqgD1, D at seed 1 on a G4) is at item 350; 206's reading waits for it and for seeds 2-3. xqgSCx1 (the 4x self-corrector, seed 1) started at 19:27.

**Result 201 (19:44): online setup D with the 4x corrector (xqoD4x, Colab A100, seed 1, 12:47-19:42).** Against the D 1x rerun on the same GPU type and stream (xqoD1): held CE at item 600 0.742 against 0.752 (0.0096 lower); prequential gain 0.069 against 0.065; gate on the 4090 partition fixed 21 of 52, broken 4 of 168, net +17 (1x: 21 / 3, +18; paired 5 vs 6); untaught drift 0.017 against 0.014 (1.2x), held-question drift 0.034 against 0.025; cross-entropy to E4B along its replies on the gate's points 0.751 against 0.759 (C 0.784, the gated teacher 0.774, the original E2B 0.826); hardware flips 3. Predictions: 1 (held CE at least 0.005 lower) held; 2 (prequential gain higher) held; 3 (gate net at least as high) failed by one (+17 against +18); 4 (drift at most 1.5x) held. Reading (201's rule): the student's held CE is at least 0.005 lower and its net is no more than 3 below, so the corrector fitted 4x longer transfers more to the student: a closer teacher gave a closer student, at the same decisions. Paired against the other seed-1 teachers: C 8 vs 7, the gated teacher 8 vs 7, online SFT 16 vs 5 (p = 0.027), the self-corrector on G4 14 vs 7 (on the 52 differing 12 vs 4; cross-GPU). One seed; 206 runs D4 and S4 at three seeds on G4, and 202 tests this adapter live next.

**Exploratory, not pre-registered (20:34): verify-before-submit habits in the logged episodes (user: "it does not seem like a knowledge issue, but rather a habit one of following a standard bug fix protocol"; "yes, run that check now").** dreamer/live_habits.py on box, from the traces: for every episode that called submit_patch, what ran between its last successful edit and its first submit. E4B's own logged episodes (e4b_base_s1 and e4b_base_pilot, the source of the online stream; 1,473 episodes, 22% resolved, 1,385 submitted): a test-suite run that passed 19% of submits (60% of those resolved); an ad-hoc run only (python -c or a script, exit 0) 12% (43% resolved); a test-suite run with failures, submitted anyway 30% (6% resolved); an ad-hoc run that failed 4% (10%); nothing run after the last edit 32% (15%); no successful edit 2%. The same call three or more times in 18% of episodes; two or more submits in 1%. So the teacher follows "edit, run the tests, submit on green" in about a fifth of its submissions (a third with ad-hoc checks), and that is where most of its solves are.

E2B on the 80 live tasks (196, 202, 207 so far): plain (240 episodes) passed-tests-then-submit 7% of submits, tests failed and submitted anyway 49%, nothing run 18%, no successful edit 19%; the same call 3+ times in 55% of episodes. With D's 1x LoRA (160 episodes): passed tests 4%, ad-hoc run only 13% (plain 2%; E4B 12%), tests failed and submitted anyway 40%, ad-hoc failed 11%, nothing run 15%, no successful edit 17%; same call 3+ times 57%; timed out 17%. With C's LoRA (160): tests failed and submitted anyway 54%, two or more submits in 14% of episodes. The 4x D LoRA after 37 episodes: 33% submitted on failing tests, 24% with no successful edit. In every arm nearly all solves come from the verified submissions (plain 5 of 6, D 8 of 11). E4B's logged run solved none of the 80 live tasks (they were picked from its failures), so the live test is on tasks the teacher failed once.

Reading (exploratory): the students' missing protocol is also the teacher's; a closer copy of E4B will not add it. E4B's data does hold the protocol in a minority of episodes (258 with passing tests before the submit, 156 of them resolved; 170 more with an ad-hoc check), which a filtered stream could teach from; refusing to submit on red would need the student's own failing states (on-policy) or contrast pairs.

**206, interim 2 (20:39): D at seed 1 on a G4 (xqgD1, 17:54-20:39), the same-GPU pair with interim 1's self-corrector.** D: fixed 23 of 52, broken 4 of 168, net +19 (the 4090 run 23 / 4, +19; the A100 run 21 / 3, +18); held CE at item 600 0.754, untaught drift 0.017, prequential gain 0.065; hardware flips against the 4090 partition 3. The two G4 sessions agree with each other on every one of the 220 E2B-alone bits. Paired on the 220 decisions, same GPU type, same stream: D over the self-corrector 11 vs 2 (p = 0.022); on the 52 differing decisions 11 vs 1 (p = 0.006); on the 168 agreeing 0 vs 1. Net +19 against +10. Held CE 0.754 against 0.756 (within 0.005: a tie on closeness), prequential gain 0.065 against 0.063. D's three seed-1 runs (4090, A100, G4) differ from each other on 10-11 decisions with no direction (5 vs 5, 6 vs 5). So at seed 1 the decision gap between the corrector fed E4B's states and the one fed E2B's own is the same on the 4090 (12 vs 1) and on the G4 (11 vs 2): the cross corrector's student fixes 21-23 of the 52 decisions where E2B differs from E4B, the self-corrector's 13, at equal closeness and speed. 206's reading needs seeds 2-3 and the 4x pair. First sight of the 4x self-corrector's student (xqgSCx1, item 250): held CE 0.748, level with the 4x cross student's 0.749 at that item on the A100 and 0.012 below the 1x self-corrector's student, so the closeness gain from the longer fit may not need E4B's states either; its run ends about 22:00.

**202, interim (21:28): the 4x-corrector D LoRA live, first attempt (x202_D4x_a, 19:54-21:24).** 5 of 80 resolved; 11 episodes ended in an error, 6 of them time-outs; 1,427 model calls (17.8 per task); 90 min. 196's D LoRA (1x corrector, the 4090 seed-1 adapter) on the same 80 tasks: 5 and 6 resolved, 21 and 16 error endings, 14 and 13 time-outs, 2,073 and 1,796 calls, 127 and 108 min; C's LoRA 3 and 2 resolved, 11 and 9 time-outs; the null adapter (207) 2 resolved, 3 time-outs, 1,271 calls, 100 min; plain 4, 1 and 1 resolved, 1, 0 and 0 time-outs. Its five solves (two google/textfsm, one jaraco/inflect, two pallets/markupsafe tasks): the 1x D attempts solved 2 and 3 of the same five, C's 1 and 2, the null adapter none, plain 2, 0 and 0. So in one attempt the 4x student solves as many as the 1x student with fewer than half its time-outs and about a fifth fewer calls. The second attempt (x202_D4x_b) is running; 202's analysis and the loop gate follow it.

**206, interim 3 (22:13): the 4x self-corrector at seed 1 on a G4 (xqgSCx1, 19:27-22:11).** Fixed 22 of 52, broken 5 of 168, net +17 (hardware flips 3); held CE at item 600 0.744; untaught drift 0.017, held-question drift 0.018; prequential gain 0.070; cross-entropy to E4B along its replies on the gate's points 0.753. Beside it, at seed 1: the 1x self-corrector on the G4 13 / 3, +10, 0.756, gain 0.063; D with the 1x corrector on the G4 23 / 4, +19, 0.754, 0.065; D with the 4x corrector on the A100 (201) 21 / 4, +17, 0.742, 0.069 (its G4 run, xqgDx1, is at about item 500). Paired on the 220 decisions: the 4x self-corrector over the 1x self-corrector 11 vs 4 (on the 52 differing 10 vs 1, p = 0.012); D 1x over the 4x self-corrector 7 vs 5; D 4x (A100) against the 4x self-corrector 8 vs 8. So the self-corrector fitted four times longer has no decision deficit: its student fixes as many decisions as the cross correctors' students (22 against 21-23) and is as close and as fast as the 4x cross corrector's (0.744 and 0.070 against 0.742 and 0.069, the latter on another GPU type). The 1x decision gap between the cross and self correctors (11 vs 2 on the G4, 12 vs 1 on the 4090) does not carry to the 4x pair at seed 1. Under 206's rule this is the in-between case so far (a decision lead for the cross corrector at 1x, a tie at 4x): no claim that the teacher's states teach; what E4B's states gave the 1x corrector, a longer fit gave the self-corrector without them. Seeds 2-3 and the G4 run of D4 are pending.

**206, interim 4 (22:29): D with the 4x corrector at seed 1 on a G4 (xqgDx1, 19:42-22:26); the four corrector students of seed 1 are now on one GPU type.** D4: fixed 24 of 52, broken 4 of 168, net +20; held CE 0.741; untaught drift 0.014; prequential gain 0.070; gate cross-entropy to E4B 0.752; hardware flips 3. The seed-1 table on G4 (fixed / broken, net; held CE; gain): S 13 / 3, +10; 0.756; 0.063. D 23 / 4, +19; 0.754; 0.065. S4 22 / 5, +17; 0.744; 0.070. D4 24 / 4, +20; 0.741; 0.070. Paired on the 220 decisions: D4 over S4 8 vs 5 (p = 0.58; on the 52 differing 5 vs 3); D over S 11 vs 2 (p = 0.022); S4 over S 11 vs 4; D4 against D 1 vs 0 (the two D students differ on one decision of 220; the two 4x D runs on G4 and A100 differ on 21). At seed 1, by 206's measures: 4x pair, held CE gap 0.003 and net gap 3 (a tie on both); 1x pair, held CE gap 0.003 (tie) and net gap 9 (a lead for the cross corrector); dose, D4 below D by 0.013 on held CE at one more net decision, S4 below S by 0.012 at seven more. So the longer fit, not E4B's states, moved closeness and speed for both correctors, and it removed the self-corrector's decision deficit. Seeds 2 and 3 follow (xqgSC2 started 22:27).

**Result 202 (22:46): the 4x-corrector D LoRA live (x202_D4x_a 19:54-21:24, x202_D4x_b 21:24-22:45; the 4090).** Resolved 5 and 8 of 80: 13 of 160, on 9 tasks. 196's arms on the same tasks: plain 5 of 160 (4 tasks), D with the 1x corrector 11 (9 tasks), C 5 (4 tasks). Per task over two attempts (exact sign tests): the 4x D LoRA over plain, more solves on 9 tasks, fewer on 2 (p = 0.065); over C 7 vs 1 (p = 0.070); against the 1x D LoRA 5 vs 4. Loop gate (live_loops.py; the 4x D LoRA, then plain, 1x D, C, each over 160): episodes ending in an error 17 (11 time-outs), 10, 37, 25; replies cut at the cap 32, 59, 18, 33; episodes with two or more submits 13, 4, 3, 25. Model calls 2,824 against 2,418 for plain and 3,869 for the 1x D LoRA; 90 and 81 min per attempt (1x D 127 and 108).

Predictions: 1 (at least 196's D, 11) held, 13; 2 (at least plain + 8 = 13) held, 13; 3 (error endings at most D's 37) held, 17; 4 (more-than-D tasks at least even) held, 5 vs 4. 4 of 4. Reading (202's rule, 196's bar): the solve bar is met exactly (plain + 8 and C + 8), but the loop gate is missed on two of its three counts, each narrowly: error endings 17 against a bar of 16 (plain's 10 + 6) and multi-submit episodes 13 against 10 (plain's 4 + 6); replies at the cap pass (32 against 65). So by the rule this is reported as counts with their sign tests, not as a live gain: 13 of 160 against 5 for plain E2B and 5 for the plain teacher's LoRA (p = 0.065 and 0.070 by task), with error endings less than half the 1x D LoRA's. Caveats as pre-registered: 196's arms ran the night before; this adapter was trained on an A100, the 1x one on the 4090; and 207's null adapter (2 of 80, 5 error endings in its first attempt) is the like-for-like control still running.

**206, interim 5 (23:30): D with the beta-1 imitable corrector at seed 1 on a G4 (xqgIM1, 20:39-23:26).** DI: fixed 20 of 52, broken 1 of 168, net +19; held CE 0.754; untaught drift 0.015, held-question drift 0.023; prequential gain 0.064; gate cross-entropy to E4B 0.760; hardware flips 3. Against D with the 1x corrector on the same GPU type and stream (23 / 4, +19; 0.754; drift 0.017; gain 0.065): paired 4 vs 4 on the 220 decisions; against the 1x self-corrector 10 vs 1 (p = 0.012). Prediction 7 at seed 1: held CE within 0.004 of D's (0.0002) and drift at or below D's (0.015 against 0.017) both hold. Reading so far (206's rule for DI): within 0.005 on held CE and within 3 on net, so holding the corrector's writes to a quarter of the size and two thirds predictable from E2B's own states makes no difference to what the student keeps at this dose; it broke one decision where D broke four. The seed-1 table on G4 lacks only C (xqgC1, running); seed 2 has started (S, D).

**206, interim 6 (00:51, Nov 10): the plain teacher at seed 1 on a G4 (xqgC1, 22:13-00:47); seed 1 is complete on one GPU type.** C: fixed 21 of 52, broken 8 of 168, net +13; held CE at item 600 0.767 (0.774 at item 550; its A100 run 0.774, its 4090 run 0.775); untaught drift 0.016; prequential gain 0.039; gate cross-entropy to E4B 0.776; hardware flips 3. C's three seed-1 runs: 4090 14 / 6, +8; A100 23 / 7, +16; G4 21 / 8, +13 (G4 against A100 7 vs 10 on the 220 decisions). The seed-1 table on G4 (fixed / broken, net; held CE; gain): C 21 / 8, +13; 0.767; 0.039. S 13 / 3, +10; 0.756; 0.063. D 23 / 4, +19; 0.754; 0.065. DI 20 / 1, +19; 0.754; 0.064. S4 22 / 5, +17; 0.744; 0.070. D4 24 / 4, +20; 0.741; 0.070. Paired against C on the 220 decisions: D 12 vs 6 (p = 0.24), D4 13 vs 6 (p = 0.17), DI 10 vs 4 (p = 0.18), S4 12 vs 8, S 9 vs 12 (on the 52 differing 2 vs 10). Prediction 1 at seed 1: D's held CE is below C's (0.754 against 0.767), by 0.013, short of the 0.015 of its second half. Prediction 2 at seed 1: D leads C by 6 net decisions, the rule's threshold; it needs a lead in each pair and 6 on average. Every corrector student is closer to E4B than C's (by 0.011 to 0.026) and learns faster (0.063-0.070 against 0.039); C breaks the most decisions (8; the correctors 1 to 5). Scoring script for the G4 table: scratchpad g4/g4score.py (the 4090 partition, paired sign tests), to be moved into the repository with the final table. Seed 2 is running (S, D, S4).

**207, interim 2 (00:59, Nov 10): D's third live attempt (x207_D_c, the 4090 seed-1 adapter, 22:49-00:58).** 3 of 80 resolved in 128 min; 19 error endings, 11 of them time-outs; 21.7 model calls per task (the null adapter 14.5, plain 14.8, the 4x D adapter 15.8-16.0). D's three attempts: 5, 6, 3 of 80 (14 of 240, 5.8%; 11 tasks solved at least once; 56 error endings, 38 time-outs). Beside it per 80: plain 2, 3, 1 (6 of 240); the null adapter 2 (one attempt); C 5 of 160; the 4x D adapter 5, 8. By task on the solve rate (x207_pertask.py): D over the null adapter 9 vs 2 (p = 0.065), over plain 7 vs 1 (p = 0.070), over C 8 vs 3 (p = 0.23); the 4x D adapter against D 7 vs 5. Prediction 4's first half (D's two new attempts solve at least 8 of 160) now needs 5 from x207_D_d. D's error endings stay high on the third attempt (19; the null adapter 5), so the 1x adapter is on course to miss the loop gate against the null adapter (at most its error endings per 160 plus 6). Next: x207_N_b, then D_d, plain_d and the A100 pair.

**206, interim 7 (01:12, Nov 10): the 1x self-corrector at seed 2 on a G4 (xqgSC2, 22:27-01:09).** Fixed 21 of 52, broken 6 of 168, net +15; held CE at item 600 0.754; untaught drift 0.016, held-question drift 0.022; prequential gain 0.060; gate cross-entropy to E4B 0.760; hardware flips 3. Its seed-1 run on the G4 fixed 13 and netted +10 (this run over that one 8 vs 3 on the 220 decisions), so the self-corrector's low count of fixed decisions at seed 1, which reproduced on two GPU types, does not repeat at seed 2: with another stream order its student fixes as many as the cross correctors' (21 against 20-24 at seed 1). Beside it at seed 2, on other GPU types: D on the 4090 23 / 6, +17, held CE 0.749, gain 0.063 (this run against it 4 vs 6); the gated teacher on the A100 21 / 4, +17, 0.766; C on the A100 30 / 3, +27, 0.768. The same-GPU pair (xqgD2) ends about 02:10. If D at seed 2 nets within 5 of +15, the 1x decision gap of seed 1 (11 vs 2) is a property of that stream order, not of the teacher, and prediction 4 (S's net within 5 of D's on average) stays open. Closeness again matches D's seed-1 value (0.754).

**206, interim 8 (02:13, Nov 10): D at seed 2 on a G4 (xqgD2, 23:26-02:11); the 1x pair at seed 2 is a tie.** D: fixed 19 of 52, broken 3 of 168, net +16; held CE at item 600 0.752; untaught drift 0.017, held-question drift 0.024; prequential gain 0.063; gate cross-entropy to E4B 0.759; hardware flips 3 (its 4090 run at seed 2: 23 / 6, +17, 0.749; the two differ on 15 decisions, 7 vs 8). Paired with the self-corrector of interim 7 on the 220 decisions, same GPU type and stream: 3 vs 2 (p = 1.0; on the 52 differing 0 vs 2); net +16 against +15; held CE 0.752 against 0.754; gain 0.063 against 0.060. So the decision gap of seed 1 (11 vs 2, net +19 against +10) does not repeat at seed 2: with this stream order both students fix about 20 of the 52 and are equally close. Over the two seeds: D nets +19, +16 (mean 17.5), S +10, +15 (mean 12.5); held CE 0.753 against 0.755; gain 0.064 against 0.061. Predictions so far: 3 (S's held CE within 0.006 of D's in every seed) holds in both seeds (0.003, 0.002); 4 (S's net within 5 of D's on average) stands at exactly 5.0 with one seed to come. By 206's rule the 1x pair is a tie on closeness and, after two seeds, 5 apart on net (a tie needs within 3 on average, a lead for D 6 with a lead in each seed): the in-between case unless seed 3 has the self-corrector ahead. D's own spread across its five runs so far (seeds 1-2, three GPU types) is +16 to +19; the self-corrector's is +8 to +15, so its seed-1 student is the outlier in either reading. The 4x pair, DI and C at seed 2 are running or queued.

**207, interim 3 (02:24, Nov 10): the null adapter's second attempt (x207_N_b, 00:59-02:23).** 0 of 80 resolved in 85 min; 4 error endings, 1 a time-out; 14.8 model calls per task. The null adapter's two attempts: 2 and 0 of 80 (2 of 160; 2 tasks; 9 error endings, 4 time-outs; 100 and 85 min). Predictions: 1 (at least 1.5 times the plain attempts' wall time) holds so far, 92 min on average against 52 (55, 48, 53), 1.8 times; 2 (at least 20 of 160 episodes end in an error) failed, 9; 3 (at most 5 of 160 solved) held, 2. So the adapter path costs wall time (1.8 times) but not episodes: plain E2B through it ends about as many episodes in an error as plain E2B served directly (9 of 160 against 15 of 240) and solves no more (2 of 160 against 6 of 240; by task 1 vs 5, p = 0.22). The time-outs of the LoRA arms are therefore the adapters' own longer episodes, not the serving path (D 21-23 calls per task and 38 time-outs in 240; the null adapter 14.5-14.8 and 4 in 160). By task on the solve rate against the null adapter (x207_pertask.py): D (three attempts, 14 of 240) 11 vs 0, p = 0.001; the 4x D adapter (13 of 160) 8 vs 1, p = 0.039; C (5 of 160) 3 vs 1. D's share of episodes solved is 4.7 times the null adapter's (5.8% against 1.25%). 207's rule reads D over its four attempts; the fourth (x207_D_d, running) cannot undo it: on the two tasks the null adapter solved (once in two attempts each) D's rate is 2 of 3 and cannot fall below 2 of 4, so the count ends at no fewer than 9 vs 0 (p = 0.004), and 14 of 320 is still 3.5 times the null adapter's share. So by 207's rule D helps live against the like-for-like control. Its loop gate fails: 37 error endings per 160 against the null adapter's 9 plus 6; the 4x D adapter has 17 (misses by 2). Limits: 80 tasks that E4B failed, counts of 2 to 14, one adapter per arm (seed 1; the A100 pair's attempts follow), and the null adapter stands in for "no training", not for "any training" (C's adapter is that control: 5 of 160, D over C 8 vs 3, p = 0.23).

**206, interim 9 (03:31, Nov 10): the 4x self-corrector at seed 2 on a G4 (xqgSCx2, 00:48-03:30).** Fixed 22 of 52, broken 7 of 168, net +15; held CE at item 600 0.745; prequential gain 0.066; gate cross-entropy to E4B 0.753; hardware flips 3; untaught drift at item 600 0.034, held-question drift 0.038. Its seed-1 run: 22 / 5, +17; 0.744; 0.070. Against the 1x self-corrector at seed 2 (21 / 6, +15; 0.754; 0.060): paired 4 vs 4; held CE 0.009 lower (0.012 at seed 1); gain 0.006 higher. So the longer fit's closeness and speed gain for the self-corrector repeats at seed 2, and at this seed it adds no decisions (it added 7 net at seed 1, where the 1x student was low). Means over two seeds: S +12.5, 0.755, 0.061; D +17.5, 0.753, 0.064; S4 +16.0, 0.745, 0.068. A caution on the drift measure: the untaught drift at single checkpoints moves between 0.010 and 0.047 within one run (this run: 0.011, 0.011, 0.016, 0.017, 0.014, 0.034 over its last six checkpoints; the 1x self-corrector at seed 2 had 0.047 at item 350 and 0.016 at 600; the 4x self-corrector at seed 1 0.036 at item 500 and 0.017 at 600). Predictions 7 and 8 read drift at item 600 as pre-registered; the final table will also give the mean over the last six checkpoints, marked as added after the fact. D4 at seed 2 ends about 03:55, DI about 04:55; C at seed 2 started at 03:30.

**206, interim 10 (03:56, Nov 10): D with the 4x corrector at seed 2 on a G4 (xqgDx2, 01:11-03:55); the 4x pair at seed 2.** D4: fixed 25 of 52, broken 4 of 168, net +21; held CE at item 600 0.741; untaught drift 0.012, held-question drift 0.035; prequential gain 0.067; gate cross-entropy to E4B 0.751; hardware flips 3. Against the 4x self-corrector at seed 2 (22 / 7, +15; 0.745; 0.066): paired on the 220 decisions 8 vs 2 (p = 0.11; on the 52 differing 3 vs 0, on the 168 agreeing 5 vs 2); net 6 higher; held CE 0.0042 lower (0.0034 at seed 1); the held CE curves run together to item 400 and part after it (0.7449 / 0.7431 / 0.7401 / 0.7407 against 0.7475 / 0.7467 / 0.7461 / 0.7449 at items 450-600). The 4x pair over two seeds: D4 +20, +21 (mean 20.5), S4 +17, +15 (mean 16.0); a lead for D4 in each seed (3, 6; mean 4.5); held CE 0.741, 0.741 against 0.744, 0.745. The two seeds' paired counts together: 16 vs 7 (p = 0.093; a sum over seeds, not in 206's rules). By 206's rule: the closeness route to "the teacher's states teach" is closed (it needed 0.005 in every seed; 0.0034 and 0.0042); the decision route needs a mean lead of 6 with a lead in each seed, so seed 3 would have to give D4 a lead of 9 or more. Otherwise the 4x pair ends in between: a consistent small lead for the cross corrector on both measures (3-6 net, 0.003-0.004), short of the pre-registered sizes. Against D with the 1x corrector at seed 2: held CE 0.011 lower (0.013 at seed 1), net +21 against +16, paired 8 vs 3. Predictions: 5 (D4's held CE at least 0.005 below D's in two of three seeds) held, 0.013 and 0.011; 6 (D4 at least 0.005 below S4 in every seed) failed at seed 1; 8 (D4's drift above D's in every seed) failed, it is below in both (0.014 / 0.017, 0.012 / 0.017). Dose rule so far: the longer-fitted corrector is the better teacher in both seeds. Means over two seeds (net; held CE; gain): S +12.5; 0.755; 0.061. D +17.5; 0.753; 0.064. S4 +16.0; 0.745; 0.068. D4 +20.5; 0.741; 0.069. D4 is the best student on every measure in both seeds, and the least variable on net (+20, +21; its A100 run +17).

**Amendment 206c (04:03, Nov 10; after seeds 1-2 of S, D, S4, D4 and seed 1 of DI and C were scored; xqgSC3 had started at 03:56 and no seed-3 number had been read; user: "i think the 0.005 at every step is wrong. thst does not make sense. also you said its teaching faster... which the original paper on telepatgy also called out").** The objection is accepted in part. 206's bars (held CE lower by at least 0.005 in every seed, or 6 net decisions on average with a lead in each) are sizes chosen before any data; they do not ask whether a difference is real, and a fixed bar in every seed fails a small difference that is there in every run. They stay scored and reported as written. Added beside them, with the three-seed rule fixed now:

*Closeness, row by row (dreamer/g4_rows.py).* Every student is scored on the same 240 held gate rows (cross-entropy to E4B along its reply), from 118 episodes in 17 held repositories (data/online/gate_rows_groups.json, built from s3_calls.jsonl, sha256 2df5f431, with the gate's rng). Paired mean differences, 95% bootstrap intervals over repositories (intervals over rows and over episodes agree to 0.0003): D minus S, seed 1 -0.0027 [-0.0038, -0.0017], seed 2 -0.0010 [-0.0018, -0.0001]; D4 minus S4, seed 1 -0.0011 [-0.0021, +0.0000], seed 2 -0.0028 [-0.0040, -0.0015]; the four pairs together -0.0019 [-0.0026, -0.0012], the cross student closer in 15 of the 17 repositories and in 4 of 4 pairs. For scale, on the same rows: D minus C at seed 1 -0.0170 [-0.0202, -0.0131]; the 4x fit minus the 1x fit -0.0081 (D) and -0.0080 (S) over two seeds; DI minus D +0.0008 [-0.0002, +0.0020]. So what E4B's states add to the student's closeness is real on these four runs and small: about a ninth of the corrector's lead over the plain teacher and about a quarter of the longer fit's gain. The same students differ from seed to seed by about 0.001 (D's gate CE 0.7593 and 0.7593, S's 0.7619 and 0.7603), so the pair is the unit that counts: 4 of 4 has p = 0.125.

*Decisions, per decision.* For each of the 220 decisions, the number of cross students matching E4B minus the number of self students (D and D4 against S and S4, both seeds): more on 19 decisions, fewer on 7 (p = 0.029), a total of 19 more matches over the four pairs (9, 1, 3, 6).

*Speed.* On the original paper: its abstract (arXiv 2610.15095) claims that KV-cache transfer beats text between agents in accuracy (40% higher on average) and in wall-clock time (4x faster on average), a claim about communication at inference, not about learning. The learning-speed result here is the nearest relative, not a replication: the students taught through the corrector learn the stream faster than the student that copies E4B's output distribution. Prequential gain at seed 1: C 0.039; S 0.063, D 0.065, DI 0.064, S4 0.070, D4 0.070 (1.6 to 1.8 times C's). Items to reach C's held CE at item 600 (0.767, C's best checkpoint): S, D, S4 and D4 are at or below it at item 100, DI at item 150: a sixth to a quarter of the stream. Cross against self on speed: the cross student's mean prequential gain is the higher in 4 of 4 pairs, by 0.0001 to 0.0035, and not in the first 100 items (0.047 / 0.046, 0.044 / 0.049, 0.050 / 0.049, 0.049 / 0.052): the speed comes with the corrector's form, and E4B's states add little to it.

Rules for the three-seed reading, fixed now:
- The teacher's states add closeness if the cross student is closer in all six pairs (three seeds, two fits; exact p = 0.031) and the six-pair mean difference has a repository-bootstrap interval below zero. The paper then gives the size beside the corrector-against-C gap and the fit gap, and marks the analysis as added after seeds 1-2. Five of six with the interval below zero: "closer on average, not in every pair". Otherwise no claim.
- Decisions: the per-decision count over the six pairs, as above, reported with 206's own rule.
- Speed against C: reported per seed as the item at which each corrector student first reaches C's item-600 held CE, with the prequential gains.
- The paper's wording follows from these: the method teaches through a state corrector; most of its gain over copying E4B's distribution comes with the corrector's form (the self-corrector has it), and feeding the corrector E4B's states adds the measured remainder.

Predictions (seed 3; C's seed-2 run is at an early item and has not been read past item 0):
1. D's student is closer than S's on the 240 rows at seed 3 (75%); D4's than S4's (75%).
2. Cross closer in all six pairs (55%).
3. The six-pair interval is below zero (85%), with a mean between -0.001 and -0.003 (75%).
4. In seeds 2 and 3, S, D, S4 and D4 reach C's item-600 held CE by item 150 (80%); DI by item 200 (75%).
5. The cross student's mean prequential gain is the higher in both seed-3 pairs (65%).
6. Per decision over the six pairs, more decisions favour the cross students than the self students, with p < 0.05 (60%).
Limits: one stream order per seed, one model pair, one task family; the row-level intervals describe these students, the pair count describes the teachers; the analysis was chosen after seeing four pairs.

**207, interim 4 (04:22, Nov 10): D's fourth live attempt (x207_D_d, 02:23-04:21); D's reading by 207's rule.** 5 of 80 resolved in 117 min; 17 error endings, 10 time-outs; 20.3 model calls per task. D's four attempts: 5, 6, 3, 5 of 80 (19 of 320, 5.9%; 12 tasks solved at least once; 73 error endings, 48 time-outs). By task on the solve rate (x207_pertask.py): D over the null adapter 12 vs 0 (p = 0.0005); over plain E2B (three attempts so far) 10 vs 3 (p = 0.092); over C's LoRA 9 vs 3 (p = 0.15); the 4x D adapter against D 6 vs 5. Predictions: 4, first half (D's two new attempts solve at least 8 of 160) held exactly, 3 + 5; 5 (D above the null adapter on more tasks, 75%; with p < 0.05, 35%) held on both. 207's rule: pooled over its four attempts D beats the null adapter by task at p < 0.05 and solves 4.75 times its share of episodes (5.9% against 1.25%; the rule asks for twice), so D helps live against the like-for-like control, and the paper may say so with the counts. The loop gate fails: 36.5 error endings per 160 against the null adapter's 9 plus 6. Next: x207_plain_d, then the A100 pair (CA_a, DA_a, CA_b, DA_b).

## Addendum 208 (04:57, Nov 10; built, in review, not yet run): a second round for the paper's main pattern (user: "yes improve this as much as we can before submission. the focus should be the novel pattern")

Why: the paper now leads with telepathy-like teaching (a corrected student as the online teacher). Its two weakest points are (1) the pair count behind "E4B's states add closeness" (six pairs at most with three seeds: 6 of 6 is p = 0.031), and (2) the live test: D's LoRA beats the null adapter, but no self-corrector student has run live, so the live gain cannot be put down to the corrector's form or to E4B's states. Seed 3 of 206 has not ended (xqgSC3 is at about item 250; no gate, row or decision result of any seed-3 run exists or has been read; one mid-run held CE of xqgSC3, 0.762 at item 100, was read at 04:36).

**Amendment 206d: seeds 4 and 5 of the cross-against-self pairs on G4.** dreamer/xq_onl12_launch.sh: S4, D4, S and D at seeds 4 and 5 (8 sessions, the frozen job script of 206, G4 only, at most 3 sessions; it starts nothing until 206's launcher has ended, because that launcher counts only its own sessions; no start after 21:30 on Nov 10; about 200 CU of the 874 left). They run whatever seed 3 shows. Scoring as in 206 and 206c (g4_table.py, g4_rows.py, the 4090 partition). Rule for the final reading, which replaces 206c's six-pair rule as the paper's reading if all ten pairs are complete by the freeze (the morning of Nov 11; otherwise the complete pairs are read by the same proportions and the six-pair rule is reported as set): the teacher's states add closeness if the cross student is closer on the 240 rows in at least 9 of the 10 pairs (exact two-sided p at most 0.021) and the ten-pair mean difference has a repository-bootstrap interval below zero; with 7 or 8 of 10 and the interval below zero, "closer on average, not in every pair"; otherwise no claim. Decisions: the per-decision count over the ten pairs. 206c's six-pair reading is still reported as its own step. Predictions: (1) in the four new pairs the cross student is closer in at least 3 (70%), in all 4 (40%); (2) at least 9 of 10 in the end (45%); (3) the cross student's net is the higher on average over the four new pairs (70%); (4) every new session's hardware flips against the 4090 partition are at most 6 of 220 (85%).

**208 proper: the self-corrector students' LoRAs live.** dreamer/x208_live.sh on the Mac, after 207's end line and before 203: the same 80 tasks, harness, time scale, QAT E2B checkpoint and adapter submission as 196, 202 and 207. Attempts, in this order: x208_S_a, x208_S4_a, x208_S_b, x208_S4_b. S is the 1x self-corrector's student of seed 1 trained on the 4090 (189; net +8), the same GPU, stream and seed as 196's D adapter; S4 is the 4x self-corrector's student of seed 1 trained on a G4 (206, xqgSCx1; net +17), the twin of 202's 4x D adapter (trained on an A100; same stream and seed). Not started if 207 has not ended by 18:00 on Nov 10; no new attempt after 20:00. Analysis dreamer/x208_pertask.py (by task on the solve rate, exact sign tests) and live_loops.py. Predictions: (1) S's LoRA solves at least 6 of 160 (60%; the null adapter 2, C's LoRA 5, D's 9.5 per 160); (2) S4's solves at least 8 of 160 (50%; the 4x D adapter 13); (3) each beats the null adapter by task at p < 0.05 (35% each); (4) D's solve rate is the higher on more tasks than S's (55%) and the 4x D adapter's than S4's (55%), neither at p < 0.05 (80%); (5) S's and S4's error endings per 160 are above the null adapter's 9 plus 6 (70% each). Reading rules, fixed now: the live gain comes with the corrector's form if the two self-corrector adapters pooled (320 episodes) beat the null adapter by task at p < 0.05 with at least twice its share of solved episodes, and their pooled share is within a quarter of the two cross adapters' pooled share (D and the 4x D adapter, 32 of 480, 6.7%). E4B's states matter live only if the cross adapters lead in both pairs and the pooled by-task test (D and 4x D against S and S4) has p < 0.05. Anything else is reported as counts. Limits: one adapter per arm and seed 1 only; S's seed-1 student is its low draw on decisions; the S4 and 4x D adapters were trained on different GPU types.

**Amendment 203c.** 203's runner on the 4090 is replaced by dreamer/dense203c_4090.sh: the same commands and cutoffs (not started after 23:00 on Nov 10, no new arm after 03:00 on Nov 11), now also after 208's end line (X208_DONE or X208_HALTED). The old runner is stopped with SIGKILL after checking its command line, so that its exit trap does not write DENSE203_DONE and release 195. 195 (a separate study) moves back by the length of 208, about 7 hours.

**206, interim 11 (04:58, Nov 10): the imitable corrector as D's teacher at seed 2 on a G4 (xqgIM2, 02:12-04:55).** DI: fixed 22 of 52, broken 2 of 168, net +20; held CE at item 600 0.751; untaught drift 0.015, held-question drift 0.025; prequential gain 0.061; gate cross-entropy to E4B 0.758; hardware flips 3. Against D with the 1x corrector at seed 2 (19 / 3, +16; 0.752; drift 0.017; gain 0.063): paired 6 vs 2 on the 220 decisions (p = 0.29); held CE 0.0014 lower; on the 240 rows -0.0009 [-0.0025, +0.0003] (seed 1: +0.0008 [-0.0003, +0.0019]). Against the 1x self-corrector at seed 2: 8 vs 3. Over two seeds DI nets +19 and +20 with 1 and 2 broken decisions (D: +19 and +16 with 4 and 3 broken; the other teachers 3 to 8). Prediction 7: DI's held CE within 0.004 of D's holds in both seeds (0.0002, 0.0014); its drift at or below D's in at least two of three seeds is settled, held (0.015 / 0.017 in both). By 206's rule for DI so far: within 0.005 on held CE and within 3 on net on average (+19.5 against +17.5), so no difference at this dose; the fewer broken decisions (3 against 7 over two seeds) are a lead to watch at seed 3, not a result.

**207, interim 5 (05:04, Nov 10): plain E2B's fourth attempt (x207_plain_d, 04:21-05:04).** 1 of 80 resolved in 43 min; 4 error endings, no time-out; 13.5 model calls per task. Plain E2B's four attempts: 4, 1, 1, 1 of 80 (7 of 320, 2.2%; 5 tasks; 19 error endings, 1 time-out; 55, 48, 53, 43 min). By task on the solve rate, four attempts each: D over plain E2B 9 vs 1 (p = 0.021). The null adapter against plain 1 vs 4 (p = 0.38). Predictions: 1 (the null adapter takes at least 1.5 times the plain attempts' wall time) held, 92 min against 50, 1.85 times; 4, second half (plain's two new attempts solve at most 8 of 160) held, 2. So D's LoRA beats both controls by task, the null adapter (12 vs 0, p = 0.0005) and plain E2B (9 vs 1, p = 0.021); against C's LoRA it leads 9 vs 3 (p = 0.15). The A100 pair's attempts follow (CA_a, DA_a, CA_b, DA_b).

**Amendment 208a (05:11, Nov 10; review changes and start; user: "yes push the new code and run the new tasks").** An independent review of x208_live.sh, dense203c_4090.sh, xq_onl12_launch.sh and x208_pertask.py found no blocking bug. Changes made before the start: x208_live.sh never writes a second X208 line of either kind, waits for vLLM on port 8000 to be gone before its final marker as well (a failed ssh is no longer read as "gone"), tries the adapter check three times, and does not start while the old 203 runner (which does not wait for 208) is alive; its per-task log line keeps 8,000 characters. g4_table.py and g4_rows.py accept seeds 4 and 5, and g4_rows.py prints the exact sign test for any split of the pairs. x207_pertask.py printed its error kinds empty (the update sat inside a comment; the totals were right); x208_pertask.py counts them. Control flow of x208_live.sh checked on a simulated clock (207 ending on time, late and never; slow, failed and missing-adapter cases): exactly one X208 line in each. The 4x self-corrector's seed-1 G4 adapter was copied to the 4090 (data_x208/lora_S4_g4, sha256 86f3d3f8, equal to the Mac's copy). The swap on the 4090 at 05:10: the old 203 runner (PID 608313, command line checked, in its wait loop, only child a sleep; no X207, X208 or DENSE203_DONE line in the queue log) was stopped with SIGKILL and dense203c_4090.sh started (one process; DENSE203_DONE not written). x208_live.sh started at 05:10 and waits for 207's end line; xq_onl12_launch.sh started at 05:11 and waits for 206's launcher to end.
