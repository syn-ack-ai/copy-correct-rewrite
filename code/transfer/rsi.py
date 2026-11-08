"""Addendum 177 (user 11-07: "this could, if it continues to work, be a way to do RSI loops?"; "yes, design the pilot now - I think self
improvement in this case could come either from the internet, or a stronger models outputs"): the RSI pilot, a loop of attempt -> verify ->
update on LiveCodeBench problems with hidden tests. One process per loop arm keeps the student in memory across rounds. Model m = the student
after m updates (model 0 = the student itself).
  round r = 1..R
    attempt  model r-1 writes greedy + K samples (T 0.8, xlearn.gen_sample) on the train pool (test.jsonl-test4.jsonl, easy and medium);
             the greedy run is model r-1's measure, reused
    wins     per problem the first passing attempt with its own closing fence, greedy first; reply = the attempt up to and including that
             fence (model text is never edited). Source S: own wins only. Source T: own wins + the stronger model's verified rows (RSI_TROWS,
             the format of xlearn build's xl_rows.jsonl) for the train-pool problems with no own win this round
    update   a rank-64 LoRA on the MLPs (gate/up/down) of the write layer and every later layer, RSI_UPD_STEPS x 2 rows for every rule:
             U1 ReST: cross-entropy on the wins
             U2 KL-SFT: + RSI_LAMBDA x KL(model r-1 || student) over the written span (xbridge seqself)
             U5 on-policy self-bridge (177a; user 11-07: "the RSI loop should also try to first translate offpolicy to onpolicy using the
               translators/self-bridge"): the translator is fitted as in U3, but the LoRA trains on the model's OWN samples of this round (all
               K samples of every train-pool problem, cut at their own fence), toward the corrected model on those tokens (per-token KL over the
               sampled reply, the first U5_MASK tokens masked): the wins (own or stronger) only ever fit the translator, nothing is copied
            U3 self-bridge: (a) the control translator fitted on model r-1's own states toward the wins (xbridge.train's recipe; its steps
                are extra and logged), (b) the LoRA trained toward the translator-corrected model r-1's full next-token distribution over the
                written span (outputs-only dense KL, xbridge.train_graft's lora form)
    merge    the LoRA is merged into the weights (PEFT merge_and_unload) = model r; the adapter is saved
    measure  greedy on both pools: fixed (train-pool problems model 0 failed, now solved), new (held problems model 0 failed, now solved),
             kept (model 0 solved, still solved; held and train pool apart); exact sign tests against model 0 and model r-1; off-task (held
             agent replies, fresh questions' p(true), untaught prose; when XB_DIR has the s3 / question files); health (pass@K and distinct
             samples on the train pool, `pass`-stub share, mean answer length, answers cut at the token cap without a closing fence)
Model 0 is measured once per session and reused by every later arm (rsi_{name}_m0.json, keyed by seed, pools, settings and a weight
fingerprint; RSI_R0 = such a file from another session), so the arms are paired against the same model-0 results and their round-1 wins.
  python rsi.py loop SOURCE:UPDATE      (S|T):(U1|U2|U3); env XB_STUDENT XB_DIR XB_SEED XC_DATA [XB_TEACHER: a Gemma tokenizer for the
                                        off-task rows when the student is not Gemma; RSI_TROWS for T; XB_SMALL; RSI_* knobs below]
  python rsi.py summary                 every arm in XB_DIR/out: counts per round, arm-vs-arm sign tests, the decision rules of 177"""
import gc, hashlib, json, math, os, re, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as Fn

HERE = Path(__file__).parent; sys.path.insert(0, str(HERE))
import xbridge as XB  # noqa: E402
import xcode as XC  # noqa: E402
import xlearn as XL  # noqa: E402

OD, SMALL, SEED = XB.OD, XB.SMALL, XB.SEED
E_ = os.environ.get
NAME = re.sub(r"[^A-Za-z0-9_.-]", "_", E_("RSI_NAME") or Path(E_("XB_STUDENT", "student")).name.lower())
NTRAIN, NHELD = (12, 4) if SMALL else (int(E_("RSI_NTRAIN", "1000000")), int(E_("RSI_NHELD", "1000000")))
K = 2 if SMALL else int(E_("RSI_K", "4"))                                   # samples per problem and round (T = xlearn.TEMP, top-p 0.95)
ROUNDS = int(E_("RSI_ROUNDS")) if E_("RSI_ROUNDS") else (2 if SMALL else 3)
SSTEPS = int(E_("RSI_SMALL_STEPS", "4"))                                    # every training phase in small mode
UPD_STEPS = SSTEPS if SMALL else int(E_("RSI_UPD_STEPS", "400"))            # the LoRA, every rule (x XB.ACC rows per step)
TR_STEPS = SSTEPS if SMALL else int(E_("RSI_TR_STEPS", "600"))              # U3's translator, first phase (noise and depth dropout)
TR_CLEAN = [int(c) for c in E_("RSI_TR_CLEAN_STEPS", "0").split(",") if c.strip() and int(c) > 0]   # 175's extra clean phases, e.g. "300,300"
TR_CLEAN = [SSTEPS] * len(TR_CLEAN) if SMALL else TR_CLEAN
TR_CLEAN_LR = float(E_("RSI_TR_CLEAN_LR", "1e-4")); LAM = float(E_("RSI_LAMBDA", "1.0"))
RANK = int(E_("RSI_RANK", "64")); LR = float(E_("RSI_LR", "3e-4")); WARM = 30   # the LoRA schedule of xbridge.train_graft
FINAL_SAMPLE = E_("RSI_FINAL_SAMPLE", "1") == "1"                           # sample the last model too (its pass@K)
ADTYPE = E_("RSI_ADAPTER_DTYPE", "fp32")                                    # saved adapters (the merge always uses the trained fp32 LoRA)
if E_("RSI_BATCH"):
    XC.BATCH = int(E_("RSI_BATCH"))                                         # generation batch (xcode default 16); this process only
VERSION = "rsi-1"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sd(purpose, r):
    """seeds from XB_SEED and the round (or model index): 1 sampling, 2 row order, 3 LoRA init, 4 translator"""
    return 177_000_000 + 1_000_000 * purpose + 1000 * SEED + r


def tag(arm):
    return arm.replace(":", "-")


def free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def pools():
    """train pool: test.jsonl-test4.jsonl easy and medium (the attempts); held: test5 + test6 easy and medium (172's problems); sorted by
    date and id, the same order in every arm"""
    tr = XL.load(XL.TRAIN_FILES, ("easy", "medium"), NTRAIN); he = XL.load(XL.HELD_FILES, ("easy", "medium"), NHELD)
    assert not {p["id"] for p in tr} & {p["id"] for p in he}, "train and held pools overlap"; return tr, he


def enc(tok, bos, ctx, reply):
    """ids of bos + ctx + reply (no special tokens added), the first reply position and the written span's start, as xbridge.encode
    with no second tokenizer: never position 0 (Qwen's attention sink / Gemma's BOS), at most XB.SPAN prompt positions"""
    e = tok(bos + ctx + reply, return_offsets_mapping=True, add_special_tokens=False); ids = e["input_ids"]
    st = np.array([s_ for s_, _ in e["offset_mapping"]]); rs = np.nonzero(st >= len(bos) + len(ctx))[0]
    if len(rs) == 0 or rs[0] < 2:
        return None
    rstart = int(rs[0]); return ids, rstart, max(1, rstart - XB.SPAN)


class Student:
    def __init__(s):
        from transformers import AutoTokenizer
        s.path = E_("XB_STUDENT"); s.m = XB.load(s.path); s.tok = AutoTokenizer.from_pretrained(s.path)
        s.gemma = "gemma" in str(getattr(s.m.config, "model_type", "")).lower(); s.bos = E_("RSI_BOS", "<bos>" if s.gemma else "")
        s.n = len(XB.layers(s.m)); s.WL = XB.rel(s.n, 0.35); s.sites = list(range(s.WL + 1, s.n)); s.SL = [XB.rel(s.n, f) for f in (0.5, 0.65)]
        cfg = s.m.config.get_text_config() if hasattr(s.m.config, "get_text_config") else s.m.config; s.ds = cfg.hidden_size; s.lay = [s.WL] + s.sites
        L = XB.layers(s.m); want = {id(getattr(L[l].mlp, f"{q}_proj")) for l in s.lay for q in ("gate", "up", "down")}
        s.targets = [n_ for n_, mod in s.m.named_modules() if id(mod) in want]   # exact names: model-agnostic (Qwen model.layers, Gemma language_model.layers)
        assert len(s.targets) == 3 * len(s.lay), (len(s.targets), s.lay)
        log("student", s.path, "| layers", s.n, "width", s.ds, "| write", s.WL, "sources", s.SL, "sites", f"{s.sites[0]}-{s.sites[-1]}", "| bos", repr(s.bos), "| device", XB.DEV)

    def fingerprint(s):
        """a hash of a strided sample of every trained weight (a LoRA merge changes all of them)"""
        h = hashlib.sha256(); L = XB.layers(s.m)
        for l in s.lay:
            for q in ("gate", "up", "down"):
                h.update(getattr(L[l].mlp, f"{q}_proj").weight.detach().flatten()[::97].float().cpu().numpy().tobytes())
        return h.hexdigest()[:16]

    @torch.no_grad()
    def probe(s, text):
        """mean next-token log-prob of a fixed text (the chaining check of the CPU test)"""
        ids = s.tok(s.bos + text, add_special_tokens=False)["input_ids"][:96]; x = torch.tensor([ids], device=XB.DEV)
        L = Fn.log_softmax(s.m(input_ids=x, use_cache=False).logits[0, :-1].float(), -1); return round(float(L.gather(-1, x[0, 1:, None]).mean()), 6)


class SelfBridge:
    """xbridge.Bridge with no teacher: the source is the student's own states at rel(n, 0.5) and rel(n, 0.65) (the control), the write layer
    rel(n, 0.35), the sites every later layer. look() and write() are Bridge's own (its teacher=False path); Grab is registered before
    QPatch as in Bridge, so it reads each layer before the patch"""
    def __init__(s, st):
        s.mS, s.SL, s.WL, s.sites, s.ds = st.m, st.SL, st.WL, st.sites, st.ds
        s.gS = XB.Grab(s.mS, sorted(set(s.SL) | {s.WL} | set(s.sites))); s.P = XB.QPatch(s.mS, s.WL, s.sites)

    def look(s, ids, rstart, sites=False):
        return XB.Bridge.look(s, None, ids, None, rstart, sites=sites, teacher=False)

    write = XB.Bridge.write

    def close(s):
        for h in s.gS.hs + s.P.hs:
            h.remove()
        s.P.slot.clear(); s.gS.out = {}


def sched(steps, warm):
    return lambda s_: min(1.0, (s_ + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s_ / steps)))


def stream(rng, n, need, first=None):
    """row indices: whole permutations back to back (each row once per pass); in small mode the longest row first (peak memory)"""
    o = np.concatenate([rng.permutation(n) for _ in range(math.ceil(need / n) + 1)])
    return np.concatenate([[first], o]) if first is not None else o


def fit_translator(st, sb, E, r):
    """xbridge.train('control') on the wins: spreads from XB.NCAL rows; XTr (gall); loss = cross-entropy on the reply + XB_SIZE x size + XB_HARM x
    do-no-harm where the model alone is already right; noise XB_NOISE and depth dropout XB_DROP on the source; AdamW XB_LR, warmup 30, cosine
    to 0.1; then each RSI_TR_CLEAN_STEPS phase: a fresh optimizer at RSI_TR_CLEAN_LR, no noise or dropout (175)"""
    rng = np.random.default_rng(sd(4, r)); torch.manual_seed(sd(4, r)); t0 = time.time()
    longest = int(np.argmax([len(e[0]) - e[2] for e in E])) if SMALL else None
    cal = list(dict.fromkeys(([longest] if SMALL else []) + rng.permutation(len(E)).tolist()))[:XB.NCAL]
    Xs, Ys, Sd = [], [], {l: [] for l in st.sites}
    for i in cal:
        ids, rstart, s0 = E[i]; o = sb.look(ids, rstart, sites=True); Xs.append(o["S"][s0:].cpu()); Ys.append(o["E"][s0:].cpu())
        for l in st.sites:
            Sd[l].append(o["sites"][l][s0:].cpu())
        del o
    Xa, Ya = torch.cat(Xs), torch.cat(Ys)
    sp = {"mx": Xa.mean(0), "sx": Xa.std(0) + 1e-3, "my": Ya.mean(0), "sy": Ya.std(0) + 1e-3, "dsy": torch.stack([torch.cat(Sd[l]).std(0) for l in st.sites]) + 1e-3}
    T = XB.XTr(Xa.shape[1], st.ds, len(st.sites), sp).to(XB.DEV); assert Xa.shape[1] == 2 * st.ds; del Xa, Ya, Xs, Ys, Sd
    phases = [(TR_STEPS, XB.LR, False)] + [(c, TR_CLEAN_LR, True) for c in TR_CLEAN]; idx = stream(rng, len(E), sum(c for c, _, _ in phases) * XB.ACC, longest); k = 0
    info = {"calibration rows": len(cal), "rows": len(E), "phases": [], "size": XB.SIZE, "harm": XB.HARM, "noise": XB.NOISE, "drop": XB.DROP}
    log(f"U3 translator: {len(E)} rows, calibration {len(cal)}, phases {[(c, lr, 'clean' if cl else 'noisy') for c, lr, cl in phases]}")
    for ph, (steps, lr, clean) in enumerate(phases):
        opt = torch.optim.AdamW(T.parameters(), lr=lr, weight_decay=0.0); sc = sched(steps, 30); hist = []
        for step in range(steps):
            for g_ in opt.param_groups:
                g_["lr"] = lr * sc(step)
            opt.zero_grad(set_to_none=True); vals = []
            for _ in range(XB.ACC):
                ids, rstart, s0 = E[int(idx[k])]; k += 1; n = len(ids); o = sb.look(ids, rstart)
                kb = torch.rand(2) >= XB.DROP
                if not kb.any():
                    kb[int(torch.randint(2, (1,)))] = True
                drop = torch.cat([torch.full((st.ds,), float(b_)) for b_ in kb.tolist()]).to(XB.DEV)
                d, g, aw = sb.write(T, o["S"], s0, n, h=o["E"], noise=0.0 if clean else XB.NOISE, drop=None if clean else drop)
                try:
                    lsm = Fn.log_softmax(st.m(input_ids=o["x"], logits_to_keep=n - rstart + 1, use_cache=False).logits[0, :-1].float(), -1)
                finally:
                    sb.P.slot.clear()
                y = torch.tensor(ids[rstart:], device=XB.DEV); ce = -lsm.gather(-1, y[:, None])[:, 0].mean()
                loss = ce + XB.SIZE * ((d ** 2).mean() + aw); hm = torch.zeros((), device=XB.DEV); ag = o["la"].argmax(-1) == y
                if XB.HARM > 0 and ag.any():                                 # the model alone is already right: keep its own distribution there
                    sv, si = o["la"][ag].topk(XB.TOPK, -1); hm = XB.tail_ce(lsm[ag], si, sv); loss = loss + XB.HARM * hm
                (loss / XB.ACC).backward()
                vals.append((float(ce.detach()), float(-o["la"].gather(-1, y[:, None])[:, 0].mean()), float(((d ** 2).mean() + aw).detach()), float(hm.detach()), float(g.mean().detach())))
                del o, lsm, d, g, aw, loss
            torch.nn.utils.clip_grad_norm_(T.parameters(), 1.0); opt.step(); hist.append(np.mean(vals, 0))
            if step % 25 == 0 or step == steps - 1:
                h_ = np.mean(hist[-25:], 0)
                log(f"U3 translator phase {ph} step {step} of {steps} | cross-entropy corrected {h_[0]:.4f} (alone {h_[1]:.4f}) | size {h_[2]:.4f} harm {h_[3]:.4f} gate {h_[4]:.3f} | {time.time() - t0:.0f} s")
        assert np.isfinite(np.array(hist[-10:])).all(), "translator loss is not finite"
        info["phases"].append({"steps": steps, "lr": lr, "clean": clean, "first25 (ce corrected, ce alone, size, harm, gate)": [round(float(v), 4) for v in np.mean(hist[:25], 0)],
                               "last25": [round(float(v), 4) for v in np.mean(hist[-25:], 0)]})
        del opt; free()
    T.eval()
    for p_ in T.parameters():
        p_.requires_grad_(False)
    info["seconds"] = round(time.time() - t0, 1); return T, info


U5_MASK = 3


def train_lora(st, rule, E, r, T=None, sb=None):
    """one LoRA (rank RANK, alpha 2 x RANK, MLP gate/up/down at the write layer and every later layer), UPD_STEPS x XB.ACC rows in the
    round's fixed order; U1 cross-entropy on the reply; U2 + LAM x KL(model r-1 || student) over [s0, n); U3 KL(corrected model r-1 ||
    student) over [s0, n) (the target with the LoRA off and the translator written over the span; no translator in the student's pass)"""
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(sd(3, r)); t0 = time.time()
    pm = get_peft_model(st.m, LoraConfig(r=RANK, lora_alpha=2 * RANK, lora_dropout=0.0, target_modules=st.targets, bias="none"))
    params = [p_ for n_, p_ in st.m.named_parameters() if "lora_" in n_]
    for p_ in params:
        p_.data = p_.data.float()
    rng = np.random.default_rng(sd(2, r)); idx = stream(rng, len(E), UPD_STEPS * XB.ACC, int(np.argmax([len(e[0]) - e[2] for e in E])) if SMALL else None)
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0); sc = sched(UPD_STEPS, WARM); hist = []
    log(f"{rule} LoRA: rank {RANK}, {len(st.targets)} modules at layers {st.lay[0]}-{st.lay[-1]}, {UPD_STEPS} steps x {XB.ACC} rows of {len(E)}, lr {LR}" + (f", lambda {LAM}" if rule == "U2" else ""))
    for step in range(UPD_STEPS):
        for g_ in opt.param_groups:
            g_["lr"] = LR * sc(step)
        opt.zero_grad(set_to_none=True); vals = []
        for a_ in range(XB.ACC):
            ids, rstart, s0 = E[int(idx[step * XB.ACC + a_])]; n = len(ids); x = torch.tensor([ids], device=XB.DEV); y = torch.tensor(ids[rstart:], device=XB.DEV); lt = None
            if rule == "U2":
                with pm.disable_adapter(), torch.no_grad():
                    lt = Fn.log_softmax(st.m(input_ids=x, logits_to_keep=n - s0, use_cache=False).logits[0].float(), -1)
            elif rule in ("U3", "U5"):
                with pm.disable_adapter(), torch.no_grad():
                    o = sb.look(ids, rstart); sb.write(T, o["S"], s0, n)
                    try:
                        lt = Fn.log_softmax(st.m(input_ids=x, logits_to_keep=n - s0, use_cache=False).logits[0].float(), -1)   # positions s0..n-1
                    finally:
                        sb.P.slot.clear()
                    del o
            if rule == "U5":                                             # on-policy: the KL only over the model's own sampled reply, the first tokens masked
                ls = Fn.log_softmax(st.m(input_ids=x, logits_to_keep=n - s0, use_cache=False).logits[0].float(), -1)
                a_, b_ = rstart - 1 - s0 + U5_MASK, n - 1 - s0; kl = (lt[a_:b_].exp() * (lt[a_:b_] - ls[a_:b_])).sum(-1).mean(); loss = kl
                ce = -ls[rstart - 1 - s0:n - 1 - s0].gather(-1, y[:, None])[:, 0].mean().detach(); tce = -lt[rstart - 1 - s0:n - 1 - s0].gather(-1, y[:, None])[:, 0].mean()
            elif rule == "U1":
                ls = Fn.log_softmax(st.m(input_ids=x, logits_to_keep=n - rstart + 1, use_cache=False).logits[0, :-1].float(), -1)
                ce = -ls.gather(-1, y[:, None])[:, 0].mean(); loss = ce; kl = tce = torch.zeros(())
            else:
                ls = Fn.log_softmax(st.m(input_ids=x, logits_to_keep=n - s0, use_cache=False).logits[0].float(), -1)
                ce = -ls[rstart - 1 - s0:n - 1 - s0].gather(-1, y[:, None])[:, 0].mean(); kl = (lt.exp() * (lt - ls)).sum(-1).mean()
                tce = -lt[rstart - 1 - s0:n - 1 - s0].gather(-1, y[:, None])[:, 0].mean(); loss = ce + LAM * kl if rule == "U2" else kl
            (loss / XB.ACC).backward(); vals.append((float(ce.detach()), float(kl.detach()), float(tce))); del ls, lt, loss
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); hist.append(np.mean(vals, 0))
        if step % 50 == 0 or step == UPD_STEPS - 1:
            h_ = np.mean(hist[-50:], 0)
            log(f"{rule} LoRA step {step} of {UPD_STEPS} | cross-entropy on the wins {h_[0]:.4f} | KL to the target {h_[1]:.4f} | target's cross-entropy {h_[2]:.4f} | {time.time() - t0:.0f} s")
    assert np.isfinite(np.array(hist[-10:])).all(), "LoRA loss is not finite"
    del opt; free()
    return pm, {"rule": rule, "steps": UPD_STEPS, "rows per step": XB.ACC, "rows": len(E), "rank": RANK, "lr": LR, "lambda": LAM if rule == "U2" else None, "layers": st.lay,
                "first50 (ce on wins, KL to target, target ce)": [round(float(v), 4) for v in np.mean(hist[:50], 0)], "last50": [round(float(v), 4) for v in np.mean(hist[-50:], 0)],
                "seconds": round(time.time() - t0, 1)}


@torch.no_grad()
def merge(st, pm, E, adir, r, arm):
    """save the adapter, merge it into the weights (model r), check the merge: KL(LoRA model || merged) and KL(LoRA model || model r-1) on the
    reply tokens of up to two wins, and the LoRA's size relative to the weights it changes"""
    from safetensors.torch import load_file, save_file
    t0 = time.time(); pm.save_pretrained(str(adir), selected_adapters=["default"])
    if ADTYPE == "bf16":
        f = adir / "adapter_model.safetensors"; w = load_file(str(f)); save_file({k_: v.to(torch.bfloat16) for k_, v in w.items()}, str(f))
    json.dump({"arm": arm, "round": r, "base": f"model {r - 1} = {st.path} with the adapters of rounds 1..{r - 1} merged in order", "dtype": ADTYPE}, open(adir / "rsi.json", "w"))
    pre, kl_up = [], []
    for ids, rstart, s0 in E[:2]:
        x = torch.tensor([ids], device=XB.DEV); kp = len(ids) - rstart + 1
        lp = Fn.log_softmax(st.m(input_ids=x, logits_to_keep=kp, use_cache=False).logits[0, :-1].float(), -1)
        with pm.disable_adapter():
            l0 = Fn.log_softmax(st.m(input_ids=x, logits_to_keep=kp, use_cache=False).logits[0, :-1].float(), -1)
        kl_up.append(float((lp.exp() * (lp - l0)).sum(-1).mean())); pre.append((x, kp, lp)); del l0
    rel = []; tset = set(st.targets)
    for n_, mod in st.m.named_modules():
        if n_ in tset and hasattr(mod, "get_delta_weight"):
            rel.append(float(mod.get_delta_weight("default").float().norm() / mod.get_base_layer().weight.float().norm().clamp_min(1e-12)))
    m = pm.merge_and_unload(); assert m is st.m, "merge_and_unload returned a new model object"
    for p_ in st.m.parameters():
        p_.requires_grad_(False)
    st.m.eval(); kl_m = []
    for x, kp, lp in pre:
        lm = Fn.log_softmax(st.m(input_ids=x, logits_to_keep=kp, use_cache=False).logits[0, :-1].float(), -1); kl_m.append(float((lp.exp() * (lp - lm)).sum(-1).mean())); del lm
    del pre; free()
    out = {"KL(LoRA model || merged) per token": round(float(np.mean(kl_m)), 6) if kl_m else None, "KL(LoRA model || model r-1) per token": round(float(np.mean(kl_up)), 6) if kl_up else None,
           "LoRA |dW| / |W| (mean over modules)": round(float(np.mean(rel)), 6) if rel else None, "modules": len(rel), "seconds": round(time.time() - t0, 1)}
    log(f"merged round {r}: KL(LoRA || merged) {out['KL(LoRA model || merged) per token']} vs KL(LoRA || before) {out['KL(LoRA model || model r-1) per token']} | |dW|/|W| {out['LoRA |dW| / |W| (mean over modules)']}")
    return out


def greedy(st, probs):
    t0 = time.time(); g = XC.gen_hf(st.m, st.tok, [p["prompt"] for p in probs], bos=st.bos); t1 = time.time(); ok = XL.score(probs, g)
    return {p["id"]: t for p, t in zip(probs, g)}, {p["id"]: bool(o) for p, o in zip(probs, ok)}, {"generate s": round(t1 - t0, 1), "score s": round(time.time() - t1, 1)}


def sample(st, tr, m_idx):
    """K samples per train-pool problem from model m_idx (seeded by XB_SEED and m_idx), scored in parallel"""
    t0 = time.time(); torch.manual_seed(sd(1, m_idx)); outs = XL.gen_sample(st.m, st.tok, [p["prompt"] for p in tr], K, bos=st.bos); t1 = time.time()
    flat = [(p, t) for p, row in zip(tr, outs) for t in row]; ok = XL.score([p for p, _ in flat], [t for _, t in flat])
    texts = {p["id"]: list(row) for p, row in zip(tr, outs)}; passes = {p["id"]: [bool(o) for o in ok[j * K:(j + 1) * K]] for j, p in enumerate(tr)}
    return texts, passes, {"generate s": round(t1 - t0, 1), "score s": round(time.time() - t1, 1)}


def attempt_stats(tr, g_pass, s_text, s_pass):
    ids = [p["id"] for p in tr]; n = len(ids)
    return {"problems": n, "greedy passes": sum(g_pass[i] for i in ids), f"pass@{K} (samples)": sum(any(s_pass[i]) for i in ids),
            "pass@1 (mean over samples)": round(float(np.mean([np.mean(s_pass[i]) for i in ids])), 4) if n else None,
            "greedy or any sample passes": sum(g_pass[i] or any(s_pass[i]) for i in ids),
            "distinct sample codes per problem (mean)": round(float(np.mean([len({XC.code_of(t).strip() for t in s_text[i]}) for i in ids])), 3) if n else None}


def pick_wins(tr, g_text, g_pass, s_text, s_pass):
    """per problem the first passing attempt with its own closing fence, greedy first: {id: (reply, source)}; reply = the attempt up to and
    including its first ``` (xlearn build's fence rule; nothing appended)"""
    out = {}
    for p in tr:
        i = p["id"]
        for t, o, k in [(g_text[i], g_pass[i], "own greedy")] + [(t, o, "own sample") for t, o in zip(s_text[i], s_pass[i])]:
            if o and "```" in t:
                out[i] = (t[:t.index("```") + 3], k); break
    return out


def load_trows(tr):
    """the stronger model's verified rows (xlearn build format: id, ctx, reply, source, difficulty), kept for train-pool problems only; the
    context must be this pool's prompt exactly and the reply must end at its own first closing fence"""
    path = E_("RSI_TROWS"); assert path and Path(path).is_file(), "source T needs RSI_TROWS (the stronger model's rows file)"
    byid = {p["id"]: p for p in tr}; rows, bad, outside = {}, 0, 0
    for line in open(path):
        r = json.loads(line)
        if r["id"] not in byid:
            outside += 1; continue
        if r["ctx"] != byid[r["id"]]["prompt"] or not r["reply"].endswith("```") or r["reply"].index("```") != len(r["reply"]) - 3:
            bad += 1; continue
        rows[r["id"]] = r
    log(f"stronger-model rows: {len(rows)} usable for the train pool ({len(tr)} problems) | outside the pool {outside} | rejected (prompt or fence mismatch) {bad} | {path}")
    assert bad <= max(2, 0.05 * (len(rows) + bad)), "too many stronger-model rows do not match this pool's prompts"
    return rows, {"file": path, "sha": hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16], "usable": len(rows), "outside the pool": outside, "rejected": bad}


def is_stub(text):
    """a skeleton answer: every code line is a class/def header, an import, a decorator or a docstring line, plus `pass` / `...` (the LeetCode
    starter echoed back)"""
    ls = [l_.strip() for l_ in XC.code_of(text).splitlines()]; ls = [l_ for l_ in ls if l_ and not l_.startswith("#")]
    body = [l_ for l_ in ls if not l_.startswith(("class ", "def ", "import ", "from ", "@", '"""', "'''"))]
    return bool(body) and all(l_ in ("pass", "...") for l_ in body)


def no_body(text):   # review 11-07: no runnable body (every code line a header, import, decorator, docstring, comment, pass or ..., or none)
    ls = [l_.strip() for l_ in XC.code_of(text).splitlines()]; ls = [l_ for l_ in ls if l_ and not l_.startswith('#')]
    return all(l_.startswith(('class ', 'def ', 'import ', 'from ', '@', '"' * 3, "'" * 3)) or l_ in ('pass', '...') for l_ in ls)


def health(st, probs, texts):
    """greedy answers: `pass`-stub share (all, functional), empty answers, mean length (re-tokenized), no closing fence: cut at the token cap
    (re-tokenized length >= 90% of the cap) or ended early"""
    ln = {p["id"]: len(st.tok(texts[p["id"]], add_special_tokens=False)["input_ids"]) for p in probs}; fn = [p for p in probs if p["func"]]
    nf = [p for p in probs if "```" not in texts[p["id"]]]; cut = [p for p in nf if ln[p["id"]] >= 0.9 * XC.MAXNEW]
    return {"answers": len(probs), "stub share": round(sum(is_stub(texts[p["id"]]) for p in probs) / max(1, len(probs)), 4),
            "stubs among functional (LeetCode)": [sum(is_stub(texts[p["id"]]) for p in fn), len(fn)], "empty answers": sum(not XC.code_of(texts[p["id"]]).strip() for p in probs),
            "mean answer tokens": round(float(np.mean(list(ln.values()))), 1) if ln else None, "cut at the token cap without a fence": [len(cut), len(probs)],
            "no fence, ended early": [len(nf) - len(cut), len(probs)], "no runnable body share": round(sum(no_body(texts[p["id"]]) for p in probs) / max(1, len(probs)), 4)}


def offtask_sets(st):
    """xlearn.offtask's rows (same picks): X1's held agent replies, fresh questions (2x), untaught prose; needs the s3 / question files in
    XB_DIR and a Gemma tokenizer to decode them (the student's own if Gemma, else XB_TEACHER, tokenizer only)"""
    if not all((XB.D / f).exists() for f in ("s3_ids.npy", "s3_calls.jsonl", "xq_questions.jsonl")):
        log("off-task: no s3 / question files in", XB.D, "- skipped"); return None
    if st.gemma:
        tg = st.tok
    elif E_("XB_TEACHER"):
        from transformers import AutoTokenizer
        tg = AutoTokenizer.from_pretrained(E_("XB_TEACHER"))
    else:
        log("off-task: no Gemma tokenizer (XB_TEACHER) to decode the s3 rows - skipped"); return None
    rows = XB.rows_all(tg); rng = np.random.default_rng(171); sets = {}
    for k in ("agent", "question", "question_fresh"):
        hk = [r for r in rows if r["kind"] == k and r["split"] == "held"]; picked = [hk[i] for i in rng.permutation(len(hk))[:XB.NX1H if k != "question_fresh" else 2 * XB.NX1H]]
        if k != "question":
            sets[f"held {k}"] = picked
    hk = [r for r in rows if r["kind"] == "prose" and r["split"] == "held"]; sets["held prose (untaught)"] = [hk[i] for i in rng.permutation(len(hk))[:3 if SMALL else 30]]
    log("off-task rows:", {k: len(v) for k, v in sets.items()}); return sets


@torch.no_grad()
def offtask(st, sets):
    """per row: next-token cross-entropy on the reply; on questions p(true) among A/B/C at the answer position"""
    if not sets:
        return None
    lq = {a: sorted({st.tok.encode(v, add_special_tokens=False)[0] for v in (a, " " + a)}) for a in "ABC"}; out = {}
    for name, rs in sets.items():
        ce, pt = [], []
        for r in rs:
            e = enc(st.tok, st.bos, r["ctx"], r["reply"])
            if e is None:
                ce.append(None); pt.append(None); continue
            ids, rstart, _ = e; x = torch.tensor([ids], device=XB.DEV); y = torch.tensor(ids[rstart:], device=XB.DEV)
            L = Fn.log_softmax(st.m(input_ids=x, logits_to_keep=len(ids) - rstart + 1, use_cache=False).logits[0, :-1].float(), -1); ce.append(float(-L.gather(-1, y[:, None])[:, 0].mean()))
            if r["kind"].startswith("question"):
                p_ = L[0].exp(); raw = {q: float(sum(p_[t_] for t_ in ts)) for q, ts in lq.items()}; ms = sum(raw.values()); pt.append(raw[r["true"]] / ms if ms else 0.0)
            else:
                pt.append(None)
            del L
        out[name] = {"cross-entropy": ce, "p(true)": pt}
    return out


def offtask_summary(cur, ref):
    """means, and the paired difference from model 0 per row (current - model 0) with a bootstrap 90% interval"""
    if not cur:
        return None
    out = {}
    for name, d in cur.items():
        o = {}
        for meas in ("cross-entropy", "p(true)"):
            a = [(x, y) for x, y in zip(d[meas], ref[name][meas]) if x is not None and y is not None]
            if not a:
                continue
            dd = np.array([x - y for x, y in a]); rr = np.random.default_rng(0); bs = [dd[rr.integers(len(dd), size=len(dd))].mean() for _ in range(1000)]
            o[meas] = {"mean": round(float(np.mean([x for x, _ in a])), 4), "minus model 0 [mean, 5%, 95%]": [round(float(dd.mean()), 4), round(float(np.percentile(bs, 5)), 4), round(float(np.percentile(bs, 95)), 4)], "rows": len(a)}
        out[name] = o
    return out


def counts(tr_ids, he_ids, p0, pr, prev):
    c = lambda ids, f: sum(1 for i in ids if f(i))  # noqa: E731
    return {"greedy passed": {"train": [c(tr_ids, lambda i: pr[i]), len(tr_ids)], "held": [c(he_ids, lambda i: pr[i]), len(he_ids)]},
            "fixed: train-pool problems model 0 failed, now solved": [c(tr_ids, lambda i: not p0[i] and pr[i]), c(tr_ids, lambda i: not p0[i])],
            "new: held problems model 0 failed, now solved": [c(he_ids, lambda i: not p0[i] and pr[i]), c(he_ids, lambda i: not p0[i])],
            "kept, held: model 0 solved, still solved": [c(he_ids, lambda i: p0[i] and pr[i]), c(he_ids, lambda i: p0[i])],
            "kept, train pool": [c(tr_ids, lambda i: p0[i] and pr[i]), c(tr_ids, lambda i: p0[i])],
            "paired vs model 0 (this only, model 0 only, exact p)": {"train": XC.mcnemar(pr, p0, tr_ids), "held": XC.mcnemar(pr, p0, he_ids)},
            "paired vs the previous model": {"train": XC.mcnemar(pr, prev, tr_ids), "held": XC.mcnemar(pr, prev, he_ids)}}


def rescore(byid, texts, passes, ref_texts, ref_pass, done):
    """xlearn's review rule: where this model and model 0 disagree, both are scored again one at a time (no parallel load near the time
    limit); model 0's serial result is kept once per problem (done) and used from then on"""
    n = ch = 0
    for i in list(passes):
        if passes[i] != ref_pass[i]:
            passes[i] = XC.run_problem(byid[i], texts[i]); n += 1
            if i not in done:
                done[i] = XC.run_problem(byid[i], ref_texts[i]); ch += done[i] != ref_pass[i]; ref_pass[i] = done[i]
    return n, ch


def save_jsonl(path, rows):
    with open(path, "w") as fh:
        for x in rows:
            fh.write(json.dumps(x) + "\n")


def model0(st, tr, he, sets):
    """model 0's greedy run (both pools), samples (train pool), health and off-task; computed once per session and reused by every arm when
    the key matches (or loaded from RSI_R0)"""
    probs = tr + he; byid = {p["id"]: p for p in probs}
    key = {"version": VERSION, "student": NAME, "seed": SEED, "small": SMALL, "train": hashlib.sha256(" ".join(p["id"] for p in tr).encode()).hexdigest()[:16],
           "held": hashlib.sha256(" ".join(p["id"] for p in he).encode()).hexdigest()[:16], "maxnew": XC.MAXNEW, "k": K, "temp": XL.TEMP, "batch": XC.BATCH,
           "fingerprint": st.fingerprint(), "offtask": sorted(sets) if sets else None}
    f = OD / f"rsi_{NAME}_m0.json"
    for src in [Path(E_("RSI_R0"))] if E_("RSI_R0") else [f]:
        if src.is_file():
            d = json.load(open(src))
            if d["key"] == key:
                log("model 0: reusing", src); return d
            log("model 0: cached key differs, recomputing:", {k_: (d["key"].get(k_), v) for k_, v in key.items() if d["key"].get(k_) != v})
    t0 = time.time(); gt, gp, tg = greedy(st, probs); log(f"model 0 greedy: train {sum(gp[p['id']] for p in tr)} of {len(tr)} | held {sum(gp[p['id']] for p in he)} of {len(he)} | {tg}")
    stx, sps, ts = sample(st, tr, 0); log(f"model 0 samples: {attempt_stats(tr, gp, stx, sps)} | {ts}")
    t1 = time.time(); ot = offtask(st, sets)
    d = {"key": key, "greedy": gt, "pass": gp, "samples": stx, "sample pass": sps, "health": {"train": health(st, tr, gt), "held": health(st, he, gt)}, "offtask": ot,
         "attempt": attempt_stats(tr, gp, stx, sps), "probe": st.probe(probs[0]["prompt"]), "timings": {"greedy": tg, "samples": ts, "offtask s": round(time.time() - t1, 1), "total s": round(time.time() - t0, 1)}}
    json.dump(d, open(f, "w"))
    save_jsonl(OD / f"rsi_{NAME}_m0_greedy.jsonl", [{"id": i, "pool": "train" if i in {p["id"] for p in tr} else "held", "text": gt[i], "pass": gp[i]} for i in byid])
    save_jsonl(OD / f"rsi_{NAME}_m0_samples.jsonl", [{"id": i, "k": j, "text": t, "pass": sps[i][j]} for i in stx for j, t in enumerate(stx[i])])
    return d


def loop(arm):
    src, rule = arm.split(":"); assert src in ("S", "T") and rule in ("U1", "U2", "U3", "U5"), f"arm must be S|T:U1|U2|U3|U5, not {arm}"
    XC.selfcheck(); T0 = time.time(); tr, he = pools(); probs = tr + he; byid = {p["id"]: p for p in probs}; tr_ids = [p["id"] for p in tr]; he_ids = [p["id"] for p in he]
    st = Student(); fp0 = st.fingerprint(); trows, tinfo = load_trows(tr) if src == "T" else ({}, None); sets = offtask_sets(st)
    log(f"== RSI arm {arm} | student {NAME} | seed {SEED} | rounds {ROUNDS} | pools: train {len(tr)}, held {len(he)} | K {K} | max new {XC.MAXNEW} | batch {XC.BATCH}"
        f" | update steps {UPD_STEPS} x {XB.ACC}" + (f" | translator {TR_STEPS} + clean {TR_CLEAN}" if rule in ("U3", "U5") else "") + (" | SMALL" if SMALL else ""))
    m0 = model0(st, tr, he, sets); p0 = dict(m0["pass"]); g_text, g_pass = m0["greedy"], p0; s_text, s_pass = m0["samples"], m0["sample pass"]   # g_pass is p0: serial re-scores refine both
    out = {"arm": arm, "source": src, "update": rule, "student": NAME, "student path": st.path, "seed": SEED, "small": SMALL, "version": VERSION,
           "config": {"rounds": ROUNDS, "K": K, "temperature": XL.TEMP, "max new tokens": XC.MAXNEW, "batch": XC.BATCH, "update steps": UPD_STEPS, "rows per step": XB.ACC,
                      "rank": RANK, "lr": LR, "lambda (U2)": LAM, "translator steps": TR_STEPS, "translator clean phases": TR_CLEAN, "clean lr": TR_CLEAN_LR,
                      "size": XB.SIZE, "harm": XB.HARM, "noise": XB.NOISE, "drop": XB.DROP, "layers": st.lay, "write": st.WL, "sources": st.SL, "bos": st.bos, "adapter dtype": ADTYPE},
           "pools": {"train": len(tr), "held": len(he)}, "ids": {"train": tr_ids, "held": he_ids}, "stronger rows": tinfo, "model 0 fingerprint": fp0, "model 0 key": m0["key"],
           "rounds": [{"round": 0, "measure": {**counts(tr_ids, he_ids, p0, g_pass, g_pass), "health": m0["health"], "off-task": offtask_summary(m0["offtask"], m0["offtask"]),
                                               "fingerprint": fp0, "probe": m0["probe"]}, "timings": m0["timings"]}],
           "per problem": {"m0": p0}}
    if m0["key"]["fingerprint"] != fp0:
        raise RuntimeError("model 0 fingerprint changed")
    jf = OD / f"rsi_{NAME}_{tag(arm)}.json"; json.dump(out, open(jf, "w"), indent=1); done = {}; ever = {}; ref_texts = m0["greedy"]
    for r in range(1, ROUNDS + 1):
        tm = {}; t0 = time.time(); log(f"== {arm} round {r} of {ROUNDS}: attempt by model {r - 1}")
        if r > 1:
            s_text, s_pass, tm["samples"] = sample(st, tr, r - 1)
        att = attempt_stats(tr, g_pass, s_text, s_pass); log(f"attempt (model {r - 1}): {att}")
        save_jsonl(OD / f"rsi_{NAME}_{tag(arm)}_m{r - 1}_samples.jsonl", [{"id": i, "k": j, "text": t, "pass": s_pass[i][j]} for i in s_text for j, t in enumerate(s_text[i])])
        own = pick_wins(tr, g_text, g_pass, s_text, s_pass); failed = [i for i in tr_ids if i not in own]; rows = []
        for i in tr_ids:                                                  # pool order: the same row order in every arm with the same win set
            if i in own:
                rows.append({"id": i, "ctx": byid[i]["prompt"], "reply": own[i][0], "source": own[i][1], "difficulty": byid[i]["difficulty"]})
            elif i in trows:
                rows.append({"id": i, "ctx": byid[i]["prompt"], "reply": trows[i]["reply"], "source": "stronger " + str(trows[i].get("source", "")), "difficulty": byid[i]["difficulty"]})
        save_jsonl(OD / f"rsi_{NAME}_{tag(arm)}_r{r}_wins.jsonl", rows)
        E, kept_rows = [], []
        for x in rows:
            e = enc(st.tok, st.bos, x["ctx"], x["reply"])
            if e is not None:
                E.append(e); kept_rows.append(x)
        for x in kept_rows:
            ever[x["id"]] = x["source"]
        wins = {"own greedy": sum(1 for x in rows if x["source"] == "own greedy"), "own sample": sum(1 for x in rows if x["source"] == "own sample"),
                "stronger": sum(1 for x in rows if x["source"].startswith("stronger")), "rows": len(rows), "rows encoded": len(E),
                "train-pool problems with no own win": len(failed), "of those with a stronger row": sum(1 for i in failed if i in trows)}
        log(f"wins for update {r}: {wins}"); tm["attempt and wins s"] = round(time.time() - t0, 1); entry = {"round": r, "attempt": {"by model": r - 1, **att}, "wins": wins}
        if not E:
            log(f"round {r}: no rows: no update, model {r} = model {r - 1}"); entry["update"] = None
        else:
            t1 = time.time(); T = sb = None; trinfo = None
            if rule in ("U3", "U5"):
                sb = SelfBridge(st); T, trinfo = fit_translator(st, sb, E, r)
            EL = E
            if rule == "U5":                                             # 177a: the LoRA's rows are the model's own samples of this round, not the wins
                EL = []
                for i in tr_ids:
                    for t in s_text[i]:
                        e = enc(st.tok, st.bos, byid[i]["prompt"], t[:t.index("```") + 3] if "```" in t else t)
                        if e is not None and len(e[0]) - e[1] > U5_MASK + 1:
                            EL.append(e)
                log(f"U5: {len(EL)} own samples as on-policy rows (the {len(E)} wins only fit the translator)")
            pm, linfo = train_lora(st, rule, EL, r, T, sb)
            if sb is not None:
                sb.close()
            del T; free(); tm["update s"] = round(time.time() - t1, 1)
            entry["update"] = {**linfo, "translator (U3, extra steps)": trinfo}; entry["merge"] = merge(st, pm, E, OD / f"rsi_{NAME}_{tag(arm)}_r{r}", r, arm); del pm; free()
        t2 = time.time(); prev = g_pass; g_text, g_pass, tm["greedy"] = greedy(st, probs); n_rs, n_ch = rescore(byid, g_text, g_pass, ref_texts, p0, done)
        tm["re-score s"] = max(0.0, round(time.time() - t2 - tm["greedy"]["generate s"] - tm["greedy"]["score s"], 1))
        t3 = time.time(); ot = offtask(st, sets); tm["off-task s"] = round(time.time() - t3, 1)
        cn = counts(tr_ids, he_ids, p0, g_pass, prev); fx = [i for i in tr_ids if not p0[i] and g_pass[i]]
        cn["fixed, by training"] = {"trained on own wins": sum(1 for i in fx if str(ever.get(i, "")).startswith("own")),
                                    "trained on stronger rows": sum(1 for i in fx if str(ever.get(i, "")).startswith("stronger")), "never trained on": sum(1 for i in fx if i not in ever)}
        entry["measure"] = {**cn, "re-scored serially": n_rs, "model 0 results changed on serial re-score (cumulative)": sum(1 for i in done if done[i] != m0["pass"][i]),
                            "health": {"train": health(st, tr, g_text), "held": health(st, he, g_text)}, "off-task": offtask_summary(ot, m0["offtask"]),
                            "fingerprint": st.fingerprint(), "probe": st.probe(probs[0]["prompt"])}
        tm["round s"] = round(time.time() - t0, 1); entry["timings"] = tm; out["rounds"].append(entry); out["per problem"][f"m{r}"] = dict(g_pass); out["per problem"]["m0"] = dict(p0)
        save_jsonl(OD / f"rsi_{NAME}_{tag(arm)}_m{r}_greedy.jsonl", [{"id": i, "pool": "train" if i in set(tr_ids) else "held", "text": g_text[i], "pass": g_pass[i]} for i in byid])
        log(f"RSI {arm} round {r}: fixed {cn['fixed: train-pool problems model 0 failed, now solved']} | new {cn['new: held problems model 0 failed, now solved']} | kept held "
            f"{cn['kept, held: model 0 solved, still solved']} | kept train {cn['kept, train pool']} | vs model 0 {cn['paired vs model 0 (this only, model 0 only, exact p)']} | "
            f"stub {entry['measure']['health']['held']['stub share']} | {tm['round s']:.0f} s")
        json.dump(out, open(jf, "w"), indent=1)
    if FINAL_SAMPLE:
        s_text, s_pass, ts = sample(st, tr, ROUNDS); out["final attempt (no update)"] = {"by model": ROUNDS, **attempt_stats(tr, g_pass, s_text, s_pass), "timings": ts}
        save_jsonl(OD / f"rsi_{NAME}_{tag(arm)}_m{ROUNDS}_samples.jsonl", [{"id": i, "k": j, "text": t, "pass": s_pass[i][j]} for i in s_text for j, t in enumerate(s_text[i])])
        log(f"final attempt (model {ROUNDS}): {out['final attempt (no update)']}")
    out["total s"] = round(time.time() - T0, 1); json.dump(out, open(jf, "w"), indent=1); print("RSI_LOOP_DONE", arm, flush=True)


def summary():
    """counts per arm and round; arm-vs-arm exact sign tests per round (held problems both arms' model 0 failed = the 'new' universe, all
    held, train pool); the decision rules of addendum 177"""
    arms = {}
    for f in sorted(OD.glob(f"rsi_{NAME}_*.json")):
        if not f.name.endswith(("_m0.json", "_summary.json")):
            d = json.load(open(f)); arms[d["arm"]] = d
    assert arms, f"no arm results in {OD}"
    K_ = {"fixed": "fixed: train-pool problems model 0 failed, now solved", "new": "new: held problems model 0 failed, now solved", "kept held": "kept, held: model 0 solved, still solved", "kept train": "kept, train pool"}
    table = {a: [{"round": e["round"], **{k: e["measure"][v] for k, v in K_.items()}, "wins": e.get("wins")} for e in d["rounds"]] for a, d in arms.items()}
    pairs = [(f"{s}:U3", f"{s}:U1") for s in "ST"] + [(f"{s}:U2", f"{s}:U1") for s in "ST"] + [(f"{s}:U3", f"{s}:U2") for s in "ST"] + [(f"T:{u}", f"S:{u}") for u in ("U1", "U2", "U3", "U5")] + [(f"{s}:U5", f"{s}:U1") for s in "ST"] + [(f"{s}:U5", f"{s}:U3") for s in "ST"]
    tests = {}
    for a, b in pairs:
        if a not in arms or b not in arms:
            continue
        A, B = arms[a], arms[b]; hs = [i for i in A["ids"]["held"] if i in set(B["ids"]["held"])]; tr = [i for i in A["ids"]["train"] if i in set(B["ids"]["train"])]
        uni = [i for i in hs if not A["per problem"]["m0"][i] and not B["per problem"]["m0"][i]]; t = {}
        for r in range(1, min(len(A["rounds"]), len(B["rounds"]))):
            pa, pb = A["per problem"].get(f"m{r}"), B["per problem"].get(f"m{r}")
            if pa is None or pb is None:
                continue
            ka = sum(1 for i in hs + tr if A["per problem"]["m0"][i] and pa[i]); kb = sum(1 for i in hs + tr if B["per problem"]["m0"][i] and pb[i])
            t[f"round {r}"] = {"new universe: held, both model 0s failed (a only, b only, exact p)": XC.mcnemar(pa, pb, uni), "all held": XC.mcnemar(pa, pb, hs),
                               "train pool": XC.mcnemar(pa, pb, tr), "kept (held + train), a / b": [ka, kb]}
        tests[f"{a} vs {b}"] = t
    rules = {}
    for s in "ST":
        a, b = f"{s}:U3", f"{s}:U1"
        if a in arms and b in arms and tests.get(f"{a} vs {b}"):
            last = max(tests[f"{a} vs {b}"], key=lambda k: int(k.split()[1])); x = tests[f"{a} vs {b}"][last]; mc = x["new universe: held, both model 0s failed (a only, b only, exact p)"]
            rules[f"self-bridge a better RSI update, source {s} ({last}: U3 beats U1 on new held problems, p < 0.1, and keeps at least as many)"] = bool(mc[0] > mc[1] and mc[2] < 0.1 and x["kept (held + train), a / b"][0] >= x["kept (held + train), a / b"][1])
    for s in "ST":                                                       # 177a: the same rule for the on-policy self-bridge
        a, b = f"{s}:U5", f"{s}:U1"
        if a in arms and b in arms and tests.get(f"{a} vs {b}"):
            last = max(tests[f"{a} vs {b}"], key=lambda k: int(k.split()[1])); x = tests[f"{a} vs {b}"][last]; mc = x["new universe: held, both model 0s failed (a only, b only, exact p)"]
            rules[f"on-policy self-bridge a better RSI update, source {s} ({last}: U5 beats U1 on new held problems, p < 0.1, and keeps at least as many)"] = bool(mc[0] > mc[1] and mc[2] < 0.1 and x["kept (held + train), a / b"][0] >= x["kept (held + train), a / b"][1])
    u3 = [f"{s}:U3" for s in "ST" if f"{s}:U3" in arms]
    if u3:
        rules["self-bridge compounds (U3's new count rises from round 1 to the last round in every source run)"] = all(len(table[a]) > 2 and table[a][-1]["new"][0] > table[a][1]["new"][0] for a in u3)
    ero = {}
    for a, rows in table.items():
        ero[a] = [rw["round"] for j, rw in enumerate(rows) if j > 0 and (rw["kept held"][0] + rw["kept train"][0]) <= (rows[j - 1]["kept held"][0] + rows[j - 1]["kept train"][0]) - 3]
    rules["erosion (kept, held + train, falls by >= 3 in a round): rounds per arm"] = ero
    for u in ("U1", "U3"):
        if f"T:{u}" in arms and f"S:{u}" in arms:
            rules[f"T beats S on new problems by >= 5 at the last round ({u})"] = table[f"T:{u}"][-1]["new"][0] - table[f"S:{u}"][-1]["new"][0] >= 5
    out = {"student": NAME, "arms": sorted(arms), "per round": table, "arm vs arm": tests, "decision rules (addendum 177)": rules}
    json.dump(out, open(OD / f"rsi_{NAME}_summary.json", "w"), indent=1)
    for a, rows in table.items():
        log(a, " | ".join(f"r{rw['round']}: fixed {rw['fixed'][0]}/{rw['fixed'][1]} new {rw['new'][0]}/{rw['new'][1]} kept {rw['kept held'][0]}/{rw['kept held'][1]}+{rw['kept train'][0]}/{rw['kept train'][1]}" for rw in rows))
    log("decision rules:", json.dumps(rules)); print("RSI_SUMMARY_DONE", flush=True)


if __name__ == "__main__":
    {"loop": lambda: loop(sys.argv[2]), "summary": summary}[sys.argv[1]]()
