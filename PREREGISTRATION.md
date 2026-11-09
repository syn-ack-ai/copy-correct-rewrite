# Pre-registration log: addenda 165-197

This file is an excerpt of the project's running pre-registration log. It holds the addenda from "Addendum 165" through the end of
"Addendum 197", copied verbatim and in their original order (addendum 180 sits between 177 and 178 in the source). Each addendum states its
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

**189, interim 4 (2026-11-08, 21:31 gate; read 21:40): the blend teacher, seed 1.** Same code7, settings, stream and gate as D seed 1, with the target log_softmax(w × E4B + (1 − w) × base E2B). Calibration on 24 points outside the stream (10,987 tokens): KL(base ‖ E4B) 0.323, KL(base ‖ corrector teacher) 0.064, so the blend was run, with w = 0.433.

| Setup (keep + replay) | Fixed (of 52) | Broken (of 168) | Net | Gate CE | Held CE (item 600) | Untaught drift | Held-question drift |
|---|---|---|---|---|---|---|---|
| D, corrector, seed 1 | 23 | 4 | +19 | 0.758 | 0.751 | 0.017 | 0.041 |
| Blend, w = 0.433, seed 1 | 14 | 4 | +10 | 0.787 | 0.781 | 0.014 | 0.022 |
| C, teacher's distribution, seed 1 | 14 | 6 | +8 | 0.783 | 0.775 | 0.014 | 0.034 |
| Control, self-corrector, seed 1 | 13 | 5 | +8 | 0.762 | 0.755 | 0.025 | 0.073 |

Paired over all 220 decisions: D seed 1 vs blend 11 vs 2 (p = 0.022; on the 52 differing, 10 vs 1, p = 0.012); D seed 2 vs blend 11 vs 4 (p = 0.12).

Prediction 5 held (w = 0.433, inside 0.2-0.8). Prediction 6 held (the blend fixes fewer than D seed 1, 14 vs 23; untaught drift 0.014, within 0.01 of D's 0.017). Success-bar part "D seed 1 beats both control and blend by at least 4 fixed": met (by 10 and 9). Still open: D's three-seed mean (seed 3 on Colab) and D's drift against C's three seeds.

Reading: a softer target alone does not explain D. Matched to the corrector's pull from base E2B (a fifth of E4B's), the blend behaves like setup C: close drift, few breaks, but it fixes no more decisions than C and sits further from the teacher on held text (CE 0.781 vs D's 0.751). D's target is as close to E2B as the blend's but points elsewhere; the self-corrector control shows that the direction needs E4B's states. One blend seed; hardware as D seed 1 (the 4090).


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
