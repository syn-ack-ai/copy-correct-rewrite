"""S3 of kaggle_g4agent/PLAN_E4B_BRIDGE.md: the 31B -> E4B translator trained online on E4B's own agent decision points (user 11-05:
"yes build the online translator now"; earlier: have the models "running at the same time"). Both models on one Colab G4; no states are stored.
  TR_STUDENT=e4b TR_TEACHERS=31b python transfer/online.py train ARCH_SOURCE   (ARCH mlp | all; SOURCE 31b: fed the 31B's states; e4b: the control, fed E4B's own)
  TR_STUDENT=e4b TR_TEACHERS=31b python transfer/online.py gate ARCH          (both sources of ARCH, on held-out decision points)
Data (ON_DIR): s3_ids.npy and s3_calls.jsonl: each decision point is E4B's context exactly as it saw it, then its reply (dreamer/s2_calls.py;
8 per taught run, 4 per held-out run; split by repo). Each sequence is cut to at most CAP tokens: the first HEAD context tokens (system prompt
and task) and the last ones, the same cut for both models; replies longer than RMAX are cut.
The translator writes over the last SPAN context tokens: it replaces E4B's layer-14 state there; arch all also adds at every later layer.
Training: E4B with the translated states learns to match the 31B's next-token distribution (top 32) along E4B's own reply: E4B's context and
reply (on-policy), the 31B's judgement. A small pull (BETA) keeps the translated state near E4B's own on the span.
Gate, forward only, held-out repos: (1) cross-entropy to the 31B along E4B's reply; (2) the tool choice: at the token where E4B's reply names its
tool (after "call:"), does each model's top token match the 31B's? (3) at E4B's own submit points, the probability each gives the token that
starts "submit". Arms: E4B alone, with the bridge (ARCH_31b), with the control (ARCH_e4b).
(4) added 11-05 before the full QAT run's gate (the small runs showed (2) and (3) at 100%: E4B's thinking already names the tool, so every arm
agrees): the decision without thinking. At each of E4B's tool calls, the context is followed directly by "<|tool_call>call:" and each model's
top next token (the tool name's first token) is read: E4B alone, bridge, control, and the 31B on the same input. Reported: agreement with the
31B, the probability of the 31B's choice, and, where E4B alone disagrees with the 31B, the share each arm moves onto the 31B's choice.
lora (11-06, S4b after R147: the control translator, fed only E4B's own states, got nearly all of the bridge's gain, so the gain is
on-policy distillation of the 31B on E4B's own contexts): a graft-shaped LoRA on E4B (MLP only, store.py's graft layers at the same relative
depth, rank ON_RANK 16, lr ON_LORA_LR 3e-4: R143) trained to the 31B's top 32 plus a tail bin (R145) along E4B's own replies, same data and
steps; saved as a PEFT folder (vLLM-servable); gated on the same 240 held-out points against E4B alone.
  TR_STUDENT=e4b TR_TEACHERS=31b python transfer/online.py lora train | lora gate"""
import json, math, os, sys, time
from contextlib import nullcontext
from pathlib import Path
import numpy as np, torch, torch.nn.functional as Fn
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE))
import pair as PR, translator as X
DEV = X.DEV; D = Path(os.environ.get("ON_DIR", str(HERE / "s3"))); OD = D / "out"; OD.mkdir(parents=True, exist_ok=True); SMALL = os.environ.get("ON_SMALL") == "1"
CAP = int(os.environ.get("ON_CAP", "6144")); HEAD = 1024; SPAN = int(os.environ.get("ON_SPAN", "1024")); RMAX = 1024
STEPS = 20 if SMALL else int(os.environ.get("ON_STEPS", "600")); ACC = int(os.environ.get("ON_ACC", "2")); LR = float(os.environ.get("ON_LR", "3e-4")); BETA = float(os.environ.get("ON_BETA", "0.05"))
NCAL = 6 if SMALL else int(os.environ.get("ON_NCAL", "48")); NGATE = 10 if SMALL else int(os.environ.get("ON_NGATE", "240")); TT, ST = X.T0, X.STUDENT
REPLY = os.environ.get("ON_REPLY") == "1"   # addendum 139: states taken and written over the context span AND the reply (teacher-forced), not the span only
LNAME = os.environ.get("ON_LORA_NAME", "lora_31b")   # the adapter gate_lora reads (S4c: lora_opd)
OPD_STEPS = 4 if SMALL else int(os.environ.get("ON_OPD_STEPS", "300")); OPD_K = int(os.environ.get("ON_OPD_K", "4")); OPD_GEN = int(os.environ.get("ON_OPD_GEN", "512"))
NLOOP = 3 if SMALL else int(os.environ.get("ON_NLOOP", "20")); NPOST = 3 if SMALL else int(os.environ.get("ON_NPOST", "16")); LOOP_GEN = 96 if SMALL else 1024; POST_GEN = 96 if SMALL else 512


def log(*a):
    print(time.strftime("%H:%M"), *a, flush=True)


IMIT = float(os.environ.get("ON_IMIT_BETA", "0"))   # addendum 200: > 0 penalises the part of the gall corrector's writes that a ridge from E2B's own states cannot predict
TRMIX = os.environ.get("ON_TRMIX") == "1"   # addendum 165: translator training on every kind of row (agent, prose, code, story, question)
SEEDOFF = 1000 * int(os.environ.get("ON_SEED", "0"))   # 11-07 replication seeds (0 = every earlier run; addendum 173)
TR_W = os.environ.get("ON_TR_W", "agent:0.4,onpolicy:0.2,prose:0.15,code:0.1,question:0.1,story:0.05")   # draw weights per kind (onpolicy: E2B's own sample at an agent point)
TR_SIZE = float(os.environ.get("ON_TR_SIZE", "0.1")); TR_HARM = float(os.environ.get("ON_TR_HARM", "0.5")); TR_NOISE = float(os.environ.get("ON_TR_NOISE", "0.1")); TR_DROP = float(os.environ.get("ON_TR_DROP", "0.15"))


def data(kinds=None):
    ids = np.load(D / "s3_ids.npy", mmap_mode="r"); rows = [json.loads(l) for l in open(D / "s3_calls.jsonl")]
    ok = (lambda r: True) if (TRMIX and kinds is None) else (lambda r: r.get("kind", "agent") in (kinds or ("agent",)))
    rows = [r for r in rows if ok(r)]
    return ids, [r for r in rows if r["split"] == "train"], [r for r in rows if r["split"] == "held"]


def seq(ids, r):
    """1 x L ids (context cut to head + tail, then the reply); c: where the reply starts; n: reply length; s0: the span's first position"""
    a, c0, n = r["off"], r["ctx_len"], min(r["reply_len"], RMAX); ctx = np.asarray(ids[a:a + c0]); rep = np.asarray(ids[a + c0:a + c0 + n]); room = CAP - n
    if len(ctx) > room:
        ctx = np.concatenate([ctx[:HEAD], ctx[-(room - HEAD):]])
    c = len(ctx); return torch.tensor(np.concatenate([ctx, rep])[None].astype(np.int64), device=DEV), c, n, max(1, c - SPAN)


class Grab:
    """keeps chosen layers' outputs over a slice of positions (off unless .sl is set)"""
    def __init__(s, m, layers):
        L = m.model.language_model.layers; s.sl, s.out = None, {}; s.hs = [L[l].register_forward_hook(s.make(l)) for l in layers]

    def make(s, l):
        def hook(_m, _i, out):
            if s.sl is not None:
                h = out[0] if isinstance(out, tuple) else out; s.out[l] = h[0, s.sl].detach().float()
            return out
        return hook


def load_models():
    mT, mS = X.load(TT), X.load(ST)
    return mT, mS, Grab(mT, X.SRC[TT]), Grab(mS, sorted(set(X.SRC[ST]) | set(X.ALL_SITES["all"]) | {X.WRITE}))


@torch.no_grad()
def look(mT, mS, gT, gS, x, c, n, s0, teacher=True, keep=False, stop=False):
    """the 31B's span states and top-32 along the reply; E4B's own span states (its source layers, its write layer, the all-level sites)"""
    out = {}
    if teacher:
        gT.sl = slice(s0, x.shape[1] if REPLY else c); gT.out = {}; lg = mT(input_ids=x, logits_to_keep=n + 1, use_cache=False).logits[0, :-1].float(); gT.sl = None
        out["T"] = torch.cat([gT.out[l] for l in X.SRC[TT]], -1); out["tv"], out["ti"] = Fn.log_softmax(lg, -1).topk(X.TOPK, -1)
        if keep:
            out["tlg"] = lg
        del lg                                                           # review 11-07: not alive during the student pass
    gS.sl = slice(s0, x.shape[1] if REPLY else c); gS.out = {}; lg = mS(input_ids=x, logits_to_keep=(n + 1) if (keep or stop) else 1, use_cache=False).logits[0, :-1].float(); gS.sl = None
    if stop:                                                             # addendum 165: E2B alone's top 32 (the do-no-harm target)
        out["sv"], out["si"] = Fn.log_softmax(lg, -1).topk(X.TOPK, -1)
    out["S"] = torch.cat([gS.out[l] for l in X.SRC[ST]], -1); out["E"] = gS.out[X.WRITE]; out["sites"] = {l: gS.out[l] for l in X.ALL_SITES["all"]}
    if keep:
        out["slg"] = lg
    return out


def pad(t, s0, c, L):
    """span rows (1 x span x d) into a 1 x L x d tensor of zeros, keeping the graph"""
    return torch.cat([t.new_zeros(1, s0, t.shape[-1]), t, t.new_zeros(1, L - c, t.shape[-1])], 1)


def write(P, T, src, s0, c, L, h12=None, noise=0.0, drop=None):
    """the translator's write over [s0, c) of a length-L sequence. gall (addendum 165): with h12 (E2B's own layer-12 state over [s0, c), from a
    pass without the bridge: exact, layers <= 12 are never changed) the new state and additions are computed here (training: the gradient
    reaches the translator); without it, inside the forward from the layer-12 state itself (no-grad uses). Returns (change, gate) for gall."""
    m = torch.zeros(1, L, dtype=torch.bool, device=DEV); m[0, s0:c] = True; P.slot["mask"] = m; one = torch.ones(1, c - s0, dtype=torch.bool, device=DEV)
    if T.arch == "gall":
        if h12 is not None:
            new, dl, d, g, aw = T.gapply(src[None], h12[None], noise, drop); P.slot["p"] = pad(new, s0, c, L); P.slot["d"] = [pad(d_.to(torch.bfloat16), s0, c, L) for d_ in dl]; P.slot["dl"] = dl; return d, g, aw   # 200c: dl unpadded for the penalty (popped at once); the padded copies in bf16, which the hook casts to anyway (half the memory, the same values)
        def fn(h):                                                       # review 11-07: the span only (no padding through the translator)
            new, dl, _, _, _ = T.gapply(src[None], h[:, s0:c]); P.slot["d"] = [pad(d_, s0, c, L) for d_ in dl]; return pad(new, s0, c, L)
        P.slot["fn"] = fn; return None, None, None
    wt = T.white(src[None], one); P.slot["p"] = pad(T.my + T.sy * wt, s0, c, L)
    if T.arch in ("multi", "all"):
        P.slot["d"] = [pad(d_, s0, c, L) for d_ in T.deltas(src[None])]
    return wt


def train(name):
    arch, source = name.split("_"); ids, tr, _ = data(); mT, mS, gT, gS = load_models(); rng = np.random.default_rng(7); t0 = time.time()
    CPOOL = int(os.environ.get("ON_CAL_POOL", "0"))                     # addendum 199: > 0 takes the calibration rows from the first CPOOL training rows
    assert 0 <= CPOOL <= len(tr), (CPOOL, len(tr)); ctr = tr[:CPOOL] if CPOOL else tr   # (the 1x pool; extra rows come after it), so a larger mix keeps the 1x calibration
    cal = [ctr[i] for i in rng.permutation(len(ctr))[:NCAL]]; Xs, Ys, Sd = [], [], {l: [] for l in X.ALL_SITES["all"]}
    longest = max(tr, key=lambda q: (min(q["ctx_len"] + min(q["reply_len"], RMAX), CAP), min(q["reply_len"], RMAX)))   # review 11-07: the longest reply among the longest rows
    if SMALL:
        cal[0] = longest                                                 # the small test reaches peak memory (review 11-06)
    CROWS = int(os.environ.get("ON_CAL_ROWS", "0"))                      # > 0: keep at most this many random rows per calibration point (R4 keeps 48 points)
    for r in cal:                                                        # calibration: spreads, the ridge start, the all-level scales
        x, c, n, s0 = seq(ids, r); o = look(mT, mS, gT, gS, x, c, n, s0); src_ = o["T"] if source == TT else o["S"]
        keep_ = torch.randperm(src_.shape[0], generator=torch.Generator().manual_seed(len(Xs)))[:CROWS].sort().values if 0 < CROWS < src_.shape[0] else slice(None)
        Xs.append(src_[keep_].cpu()); Ys.append(o["E"][keep_].cpu())
        for l in Sd:
            Sd[l].append(o["sites"][l][keep_].cpu())
        del o, src_; gT.out, gS.out = {}, {}
    grp = torch.cat([torch.full((len(x_),), k_) for k_, x_ in enumerate(Xs)]); Xa, Ya = torch.cat(Xs), torch.cat(Ys); st = {"mx": Xa.mean(0).numpy(), "sx": (Xa.std(0) + 1e-3).numpy(), "my": Ya.mean(0).numpy(), "sy": (Ya.std(0) + 1e-3).numpy()}
    T = X.Tr(Xa.shape[1], arch, st).to(DEV); din = Xa.shape[1]; log("calibration:", len(cal), "decision points,", len(Xa), "span tokens, source width", din)
    hg = set(torch.randperm(len(Xs), generator=torch.Generator().manual_seed(7))[:max(1, len(Xs) // 6)].tolist()); hm = torch.tensor([int(g_) in hg for g_ in grp]); ho, fi = hm.nonzero()[:, 0], (~hm).nonzero()[:, 0]   # whole decision points held out
    Yw = ((Ya - torch.tensor(st["my"])) / torch.tensor(st["sy"])).to(DEV); Z1 = torch.ones(len(Xa), Xa.shape[1] + 1, device=DEV); mx_, sx_ = torch.tensor(st["mx"]), torch.tensor(st["sx"])
    for i_ in range(0, len(Xa), 8192):                                   # filled in place: half the peak of building Z then concatenating (review 11-06)
        Z1[i_:i_ + 8192, :-1] = ((Xa[i_:i_ + 8192] - mx_) / sx_).to(DEV)
    G = torch.zeros(Z1.shape[1], Z1.shape[1], device=DEV); R = torch.zeros(Z1.shape[1], Yw.shape[1], device=DEV); best = None
    for i_ in range(0, len(fi), 8192):                                   # in chunks: no full copy of the training rows
        j_ = fi[i_:i_ + 8192].to(DEV); zc = Z1[j_]; G += zc.T @ zc; R += zc.T @ Yw[j_]; del zc
    for lam in (100.0, 1000.0, 10000.0):
        W = torch.linalg.solve(G + lam * torch.eye(G.shape[0], device=DEV), R); r2 = float(1 - ((Z1[ho] @ W - Yw[ho]) ** 2).sum() / ((Yw[ho] - Yw[ho].mean(0)) ** 2).sum())
        if best is None or r2 > best[0]:
            best = (r2, lam, W)
    T.lin.weight.data = best[2][:-1].T.contiguous(); T.lin.bias.data = best[2][-1].contiguous(); del Z1, G, R, Yw; log("ridge: held-out R2", round(best[0], 4), "strength", best[1], "(gall: reported only; its forward has no ridge term)" if T.arch == "gall" else "")
    if hasattr(T, "dsy"):
        T.dsy.copy_(torch.stack([torch.cat(Sd[l]).std(0) for l in T.sites]).to(DEV) + 1e-3)
    mS.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); mS.disable_input_require_grads(); mS.train(); P = X.Patch(mS, T.sites)   # the gradient starts at the hook, not at the embeddings
    opt = torch.optim.AdamW(T.parameters(), lr=LR, weight_decay=0.0); sched = lambda s_: min(1.0, (s_ + 1) / 30) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s_ / STEPS)))  # noqa: E731
    GALL = T.arch == "gall"; eos = eos_of(mS); tok = X.tok_(); torch.manual_seed(7)   # review 11-07: bridge and control draw alike
    W_ = dict((k_, float(v_)) for k_, v_ in (kv.split(":") for kv in TR_W.split(","))) if TRMIX else {"agent": 1.0}
    bykind = {k_: [r_ for r_ in tr if r_.get("kind", "agent") == k_] for k_ in W_ if k_ != "onpolicy"}; agent_tr = [r_ for r_ in tr if r_.get("kind", "agent") == "agent"]
    W_ = {k_: v_ for k_, v_ in W_.items() if k_ == "onpolicy" or bykind.get(k_)}; kinds = list(W_); pw = np.array([W_[k_] for k_ in kinds]); pw = pw / pw.sum()
    widths = [X.WIDTH[TT if source == TT else ST]] * len(X.SRC[TT if source == TT else ST]); assert sum(widths) == din, (widths, din)
    log("translator", name, "| kinds", {k_: round(float(w_), 3) for k_, w_ in zip(kinds, pw)}, "rows", {k_: len(v_) for k_, v_ in bykind.items()}, *(("| gall: size", TR_SIZE, "harm", TR_HARM, "noise", TR_NOISE, "depth drop", TR_DROP) if GALL else ()))

    assert IMIT == 0 or (GALL and source == TT), "ON_IMIT_BETA is for the cross-model gall corrector"
    IM = {"s": None, "ss": None, "n": 0, "mu": None, "sd": None, "G": {}, "R": {}, "W": {}}; IMWARM, IMEVERY, IMDECAY = (1, 1, 0.9) if SMALL else (50, 10, 0.99)   # addendum 200

    def imit_pen(step, phi, ys):
        """addendum 200: phi (span x 3*DS) E2B's own raw states at layers 12, 17, 22 from the pass without the corrector; ys {part: (span x DS) whitened
        write, with grad}: 'w12' is predicted from layer 12 only, 'add' (the additions averaged over sites) from layers 12, 17 and 22. Exponentially
        decayed ridge statistics (standardised features plus a bias), refitted every IMEVERY steps after IMWARM; returns the mean squared
        unpredicted part (grad through the writes only) and its share of the write energy"""
        with torch.no_grad():
            f = phi.float()
            if step < IMWARM:                                            # review 200: exact mean and sd over the warm-up rows, then frozen; no ridge statistics before
                IM["s"] = f.sum(0) + (IM["s"] if IM["s"] is not None else 0); IM["ss"] = (f ** 2).sum(0) + (IM["ss"] if IM["ss"] is not None else 0); IM["n"] += len(f)
                return ys["w12"].new_zeros(()), float("nan")
            if IM["mu"] is None:
                IM["mu"] = IM["s"] / IM["n"]; IM["sd"] = (IM["ss"] / IM["n"] - IM["mu"] ** 2).clamp_min(0).sqrt() + 1e-3
            fs = (f - IM["mu"]) / IM["sd"]; one = fs.new_ones(len(fs), 1)
            feats = {"w12": torch.cat([fs[:, :X.DS], one], 1), "add": torch.cat([fs, one], 1)}
            for k_, y_ in ys.items():
                F_ = feats[k_]; yd = y_.detach().float()
                IM["G"][k_] = IMDECAY * IM["G"].get(k_, 0) + F_.T @ F_; IM["R"][k_] = IMDECAY * IM["R"].get(k_, 0) + F_.T @ yd
                if step % IMEVERY == 0 or k_ not in IM["W"]:
                    Gk = IM["G"][k_]; lam = 0.05 * float(Gk.diagonal().mean()); IM["W"][k_] = torch.linalg.solve(Gk + lam * torch.eye(len(Gk), device=Gk.device), IM["R"][k_])
        if not IM["W"]:
            return ys["w12"].new_zeros(()), float("nan")
        res = [(y_.float() - feats[k_] @ IM["W"][k_]) for k_, y_ in ys.items()]; pen = sum((r_ ** 2).mean() for r_ in res) / len(res)
        en = sum(float((y_.detach().float() ** 2).mean()) for y_ in ys.values()) / len(ys); return pen, float(pen.detach()) / max(en, 1e-12)

    def draw(step):
        """a sequence to train on: (x, c, n, s0, kind)"""
        if SMALL and (step == 0 or (IMIT > 0 and step == IMWARM)):     # 200c: the longest row again once the penalty is on (and Adam's state exists)
            return (*seq(ids, longest), "agent")
        k_ = kinds[0] if len(kinds) == 1 else kinds[int(rng.choice(len(kinds), p=pw))]
        if k_ != "onpolicy":
            return (*seq(ids, bykind[k_][int(rng.integers(len(bykind[k_])))]), k_)
        r_ = agent_tr[int(rng.integers(len(agent_tr)))]; x_, c_, _, _ = seq(ids, r_); ctx = x_[:, :c_]   # E2B's own sample at an agent decision point (no bridge)
        if c_ > CAP - OPD_GEN:
            ctx = torch.cat([ctx[:, :HEAD], ctx[:, -(CAP - OPD_GEN - HEAD):]], 1)
        mS.eval(); torch.manual_seed(100003 * step + len(seen_draws)); seen_draws.append(step)   # review 11-07: the same sample for bridge and control
        with torch.no_grad():
            y = mS.generate(input_ids=ctx, attention_mask=torch.ones_like(ctx), do_sample=True, temperature=1.0, top_k=0, top_p=1.0, max_new_tokens=OPD_GEN, eos_token_id=eos, pad_token_id=tok.pad_token_id, use_cache=True)[:, ctx.shape[1]:]
        mS.train(); mm, _ = ends_of(y, eos); Ls = max(1, int(mm[0].sum())); cc = ctx.shape[1]
        return torch.cat([ctx, y[:, :Ls]], 1), cc, Ls, max(1, cc - SPAN), "onpolicy"
    hist = []; seen = {k_: 0 for k_ in kinds}; seen_draws = []
    for step in range(STEPS):
        for g in opt.param_groups:
            g["lr"] = LR * sched(step)
        opt.zero_grad(set_to_none=True); kds, ans, hms, gms, ims, hkl = [], [], [], [], [], []
        for _ in range(ACC):
            x, c, n, s0, kind = draw(step); seen[kind] += 1
            o = look(mT, mS, gT, gS, x, c, n, s0, stop=GALL and TR_HARM > 0); src = o["T"] if source == TT else o["S"]
            o.pop("sites", None); gT.out, gS.out = {}, {}                  # free the captured site layers (calibration only)
            if GALL:                                                     # addendum 165: identity by default, do no harm, regularised
                drop = None
                if TR_DROP > 0:
                    kb = torch.rand(len(widths)) >= TR_DROP
                    if not kb.any():
                        kb[int(torch.randint(len(widths), (1,)))] = True
                    drop = torch.cat([torch.full((w_,), float(k_)) for w_, k_ in zip(widths, kb.tolist())]).to(DEV)
                d_, g_, aw = write(P, T, src, s0, s0 + src.shape[0], x.shape[1], h12=o["E"], noise=TR_NOISE, drop=drop); dls = P.slot.pop("dl")   # review 200c: not kept through backward
                if TR_SIZE == 0:                                         # review 200c: its gradient is multiplied by 0; frees the 22 saved additions
                    aw = aw.detach()
            else:
                wt = write(P, T, src, s0, s0 + src.shape[0], x.shape[1])
            lsm = Fn.log_softmax(mS(input_ids=x, logits_to_keep=n + 1, use_cache=False).logits[0, :-1].float(), -1)
            kd = -(o["tv"].softmax(-1) * lsm.gather(-1, o["ti"])).sum(-1).mean()
            if GALL:
                anc = (d_ ** 2).mean() + aw; loss = kd + TR_SIZE * anc
                if IMIT > 0:                                             # addendum 200: the imitable corrector
                    assert o["S"].shape[-1] == 2 * X.DS, "ON_IMIT_BETA needs TR_SRC_DEPTHS with two E2B source layers (17, 22)"
                    aws = sum(d_k[0] / T.dsy[k_] for k_, d_k in enumerate(dls)) / len(T.sites)   # 200c: the same values from the unpadded writes (memory)
                    ip, ish = imit_pen(step, torch.cat([o["E"], o["S"]], -1).detach(), {"w12": d_[0], "add": aws}); loss = loss + IMIT * ip; ims.append(ish)
                if TR_HARM > 0:
                    ag = o["ti"][:, 0] == o["si"][:, 0]                  # E4B and E2B alone pick the same token: keep E2B's own distribution there
                    if ag.any():                                         # review 11-07: tail cross-entropy (top 32 + the rest), no pull when nothing changes
                        qk = lsm.gather(-1, o["si"])[ag]; pk = o["sv"][ag].exp(); pt = (1 - pk.sum(-1)).clamp_min(0)
                        hm = -((pk * qk).sum(-1) + pt * torch.log1p(-qk.exp().sum(-1).clamp(max=1 - 1e-6))).mean()
                        hp = -((pk * o["sv"][ag]).sum(-1) + pt * pt.clamp_min(1e-12).log()).mean(); hkl.append(float((hm - hp).detach()))   # 200b: E2B's own entropy there; hm - hp is the drift (KL)
                    else:
                        hm = lsm.new_zeros(())
                    loss = loss + TR_HARM * hm; hms.append(float(hm.detach()))
                gms.append(float(g_.detach().mean()))
            else:
                anc = ((wt[0] - (o["E"] - T.my) / T.sy) ** 2).mean(); loss = kd + BETA * anc
            dls = None                                                   # review 200c: free the unpadded writes before backward
            (loss / ACC).backward(); P.slot.clear(); kds.append(float(kd.detach())); ans.append(float(anc.detach())); del lsm, o
        torch.nn.utils.clip_grad_norm_(T.parameters(), 1.0); opt.step(); hist.append((np.mean(kds), np.mean(ans), np.mean(hms) if hms else 0.0, np.mean(gms) if gms else 0.0, np.nanmean(ims) if ims and not np.isnan(ims).all() else 0.0, np.mean(hkl) if hkl else 0.0))
        if step % 25 == 0 or step == STEPS - 1:
            h = np.mean(hist[-25:], 0); log(name, "step", step, "cross-entropy to the 31B", round(float(h[0]), 4), "pull" if not GALL else "change size", round(float(h[1]), 4),
                                            *(("harm", round(float(h[2]), 4), "harm KL", round(float(h[5]), 4), "gate", round(float(h[3]), 4)) if GALL else ()), *(("unpredicted share", round(float(h[4]), 4)) if IMIT > 0 else ()), "| seen", seen, round(time.time() - t0), "s")
    assert np.isfinite(np.array(hist[-25:])).all(), "training loss is not finite"
    torch.save({"arch": T.arch, "din": din, "state": T.state_dict(), "ridge R2": best[0], "last": [float(v) for v in np.mean(hist[-25:], 0)], "steps": STEPS, "cal pool": CPOOL, "reply": REPLY, "span": SPAN,
                "all_kv": os.environ.get("TR_ALL_KV"), "src depths": os.environ.get("TR_SRC_DEPTHS"), "mix": W_ if TRMIX else None, "seen": seen,
                "gall": {"size": TR_SIZE, "harm": TR_HARM, "noise": TR_NOISE, "depth drop": TR_DROP} if GALL else None, "imit beta": IMIT}, OD / f"{name}.pt"); log("peak memory", round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), "GiB allocated", round(torch.cuda.max_memory_reserved() / 2 ** 30, 2), "GiB reserved") if SMALL else None; print("TRAIN_DONE", name, flush=True)


@torch.no_grad()
def gate(arch):
    assert not REPLY, "gate() writes the context span only; use reach for reply-written translators"
    ids, _, held = data(); mT, mS, gT, gS = load_models(); tok = X.tok_(); mS.eval(); Ts = {}
    for src in (TT, ST):
        ck = torch.load(OD / f"{arch}_{src}.pt", weights_only=False, map_location=DEV); T = X.Tr(ck["din"], ck["arch"], {k: ck["state"][k].cpu().numpy() for k in ("mx", "sx", "my", "sy")}).to(DEV); T.load_state_dict(ck["state"]); Ts[src] = T.eval()
    P = X.Patch(mS, Ts[TT].sites); call = [tok.convert_tokens_to_ids("<|tool_call>")] + tok.encode("call:", add_special_tokens=False); sub = tok.encode("submit_patch", add_special_tokens=False)[0]
    pick = [held[i] for i in np.random.default_rng(11).permutation(len(held))[:NGATE]]; acc = {k: [] for k in ("E4B alone", "bridge", "control")}; agree = {k: [] for k in acc}; subp = {k: [] for k in list(acc) + ["31B"]}; dec = {k: [] for k in acc}
    for r in pick:
        x, c, n, s0 = seq(ids, r); o = look(mT, mS, gT, gS, x, c, n, s0, keep=True); rep = x[0, c:].tolist()
        at = next((i + len(call) for i in range(len(rep) - len(call)) if rep[i:i + len(call)] == call), None)   # the tool name's first token in E4B's reply
        lgs = {"E4B alone": o["slg"]}
        for k, src in (("bridge", o["T"]), ("control", o["S"])):
            write(P, Ts[TT if k == "bridge" else ST], src, s0, c, x.shape[1]); lgs[k] = mS(input_ids=x, logits_to_keep=n + 1, use_cache=False).logits[0, :-1].float(); P.slot.clear()
        for k, lg in lgs.items():
            acc[k].append(float(-(o["tv"].softmax(-1) * Fn.log_softmax(lg, -1).gather(-1, o["ti"])).sum(-1).mean()))
            if at is not None and at < len(lg):
                agree[k].append(int(lg[at].argmax() == o["tlg"][at].argmax()))
                if r["action"] == "submit_patch":
                    subp[k].append(float(Fn.softmax(lg[at], -1)[sub]))
        if at is not None and at < len(o["tlg"]) and r["action"] == "submit_patch":
            subp["31B"].append(float(Fn.softmax(o["tlg"][at], -1)[sub]))
        if r["action"] != "text":                                        # (4) the decision without thinking
            x2 = torch.cat([x[:, :c], torch.tensor([call], device=DEV)], 1); L2 = x2.shape[1]
            with torch.no_grad():
                d = {"31B": mT(input_ids=x2, logits_to_keep=1, use_cache=False).logits[0, -1].float(), "E4B alone": mS(input_ids=x2, logits_to_keep=1, use_cache=False).logits[0, -1].float()}
                for k, src in (("bridge", o["T"]), ("control", o["S"])):
                    write(P, Ts[TT if k == "bridge" else ST], src, s0, c, L2); d[k] = mS(input_ids=x2, logits_to_keep=1, use_cache=False).logits[0, -1].float(); P.slot.clear()
            t31 = int(d["31B"].argmax()); dis = int(d["E4B alone"].argmax()) != t31
            for k in acc:
                dec[k].append((int(int(d[k].argmax()) == t31), float(Fn.softmax(d[k], -1)[t31]), dis))
    res = {"arch": arch, "decision points": len(pick), "cross-entropy to the 31B along E4B's reply (lower is closer)": {k: round(float(np.mean(v)), 4) for k, v in acc.items()},
           "tool choice: top token matches the 31B's (share)": {k: round(float(np.mean(v)), 4) for k, v in agree.items()}, "tool tokens found": len(agree["E4B alone"]),
           "at E4B's own submits: probability of submit_patch": {k: round(float(np.mean(v)), 4) for k, v in subp.items()}, "E4B submits in the sample": len(subp["31B"]),
           "decision without thinking (context, then the tool-call opening), at E4B's tool calls": {
               "points": len(dec["E4B alone"]), "where E4B alone's choice differs from the 31B's": sum(v[2] for v in dec["E4B alone"]),
               "top token matches the 31B's (share)": {k: round(float(np.mean([v[0] for v in vs])), 4) if vs else None for k, vs in dec.items()},
               "probability of the 31B's choice": {k: round(float(np.mean([v[1] for v in vs])), 4) if vs else None for k, vs in dec.items()},
               "where E4B alone differs: share now on the 31B's choice": {k: round(float(np.mean([v[0] for v in vs if v[2]])), 4) if any(v[2] for v in vs) else None for k, vs in dec.items()}},
           "translators": {s: {"ridge R2": round(torch.load(OD / f"{arch}_{s}.pt", weights_only=False, map_location="cpu")["ridge R2"], 4)} for s in (TT, ST)}}
    json.dump(res, open(OD / f"gate_{arch}.json", "w"), indent=1); log(json.dumps(res)); print("GATE_DONE", arch, flush=True)


def lora_targets(m):
    """store.py's graft layers (E2B, 35 layers) at the same relative depth, MLP projections only"""
    nl = len(m.model.language_model.layers); lay = sorted({round(l * nl / 35) for l in (4, 6, 8, 10, 12, 16, 20, 24, 28)})
    return [f"model.language_model.layers.{l}.mlp.{q}_proj" for l in lay for q in ("gate", "up", "down")], lay


def tail_ce(lg, ti, tv):
    """cross-entropy to the teacher's top-K plus one bin for the rest of its probability (R145)"""
    qk = Fn.log_softmax(lg, -1).gather(-1, ti); pk = tv.exp(); pt = (1 - pk.sum(-1)).clamp_min(0); qt = torch.log1p(-qk.exp().sum(-1).clamp(max=1 - 1e-6))
    return -((pk * qk).sum(-1) + pt * qt).mean()


def train_lora():
    """S3-style LoRA on E4B's logged replies. ON_LORA_LOSS: kd (default, E4B's top 32 with a tail bin) or ce (addendum 192: plain SFT, hard
    labels on E4B's reply). ON_LORA_POOL: train (default, every training point) or stream (192: the online test's 600 stream items, seed 1).
    ON_LORA_OUT names the adapter (default lora_31b). Addendum 203: ON_LORA_LOSS=dense, a forward KL (top 32 + tail bin) to ON_LORA_TEACH (e4b,
    bridge = E2B corrected by ON_BRIDGE_TR, control = E2B corrected by TR_CONTROL) over the reply and up to ON_DENSE_CTX context positions before
    it (inside the corrector's written span), along E4B's logged replies; ON_LORA_LAYERS "A-B" puts the LoRA on layers A..B (MLP projections)."""
    from peft import LoraConfig, get_peft_model
    LLOSS, LPOOL, LOUT = os.environ.get("ON_LORA_LOSS", "kd"), os.environ.get("ON_LORA_POOL", "train"), os.environ.get("ON_LORA_OUT", "lora_31b")
    LTEACH, DCTX = os.environ.get("ON_LORA_TEACH", "e4b"), int(os.environ.get("ON_DENSE_CTX", "512"))   # addendum 203
    assert LLOSS in ("kd", "ce", "dense") and LPOOL in ("train", "stream") and LTEACH in ("e4b", "bridge", "control")
    ids, tr, _ = data(); rng = np.random.default_rng(7); t0 = time.time()
    if LLOSS == "ce":                                                    # review 192: plain SFT needs no teacher; E4B not loaded (about 9 GB free)
        mT = gT = gS = None; mS = X.load(ST)
    else:
        mT, mS, gT, gS = load_models()
    if LPOOL == "stream":                                                # 192: the same items, in the same draw, as train_online's stream (seed 1)
        call, sub = call_ids(X.tok_()); keep = [r for r in tr if not after_submit(ids, r, call, sub)]
        tr = [keep[i] for i in np.random.default_rng(170).permutation(len(keep))[:ONL_N]]
        if SMALL:                                                        # review 192: the small pass trains first on the longest reply (the memory peak)
            tr[0] = max((r for r in data()[1]), key=lambda q: (min(q["ctx_len"] + min(q["reply_len"], RMAX), CAP), min(q["reply_len"], RMAX)))
        torch.manual_seed(161)                                           # review 192: the online arms' LoRA start (seed 1)
    br = ctl = None
    if LLOSS == "dense" and LTEACH == "bridge":                          # addendum 203: the teachers' hooks on the base layers, before the LoRA
        br = bridge_setup(mS)
    if LLOSS == "dense" and LTEACH == "control":
        Tc_ = control_setup(); ctl = (Tc_, X.Patch(mS, Tc_.sites), Grab(mS, X.SRC[ST]))
    log("LoRA loss", LLOSS, *(("| teacher", LTEACH, "| context positions", DCTX) if LLOSS == "dense" else ()), "| pool", LPOOL, len(tr), "items | out", LOUT)
    tm, lay = lora_targets(mS); rk = int(os.environ.get("ON_RANK", "16")); llr = float(os.environ.get("ON_LORA_LR", "3e-4"))
    if LLOSS == "dense":                                                # review 203: the same LoRA start for every teacher
        torch.manual_seed(203)
    if os.environ.get("ON_LORA_LAYERS"):                                 # addendum 203: e.g. 12-34, the corrector's write zone
        a_, b_ = (int(v) for v in os.environ["ON_LORA_LAYERS"].split("-")); lay = list(range(a_, b_ + 1))
        tm = [f"model.language_model.layers.{l}.mlp.{q}_proj" for l in lay for q in ("gate", "up", "down")]
    pm = get_peft_model(mS, LoraConfig(r=rk, lora_alpha=2 * rk, lora_dropout=0.0, target_modules=tm, bias="none")); params = [p_ for n_, p_ in mS.named_parameters() if "lora_" in n_]
    log("LoRA layers", lay, "rank", rk, "lr", llr, "trainable", sum(p_.numel() for p_ in params))
    mS.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); mS.train()
    opt = torch.optim.AdamW(params, lr=llr, weight_decay=0.0); sched = lambda s_: min(1.0, (s_ + 1) / 30) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s_ / STEPS)))  # noqa: E731
    hist = []
    for step in range(STEPS):
        for g in opt.param_groups:
            g["lr"] = llr * sched(step)
        opt.zero_grad(set_to_none=True); kds = []
        for _ in range(ACC):
            r = tr[0] if SMALL and LPOOL == "stream" and step == 0 else tr[int(rng.integers(len(tr)))]; x, c, n, s0 = seq(ids, r)
            if LLOSS == "ce":                                            # 192: plain SFT on E4B's logged reply
                lg = mS(input_ids=x, logits_to_keep=n + 1, use_cache=False).logits[0, :-1].float(); kd = Fn.cross_entropy(lg, x[0, -n:])
            elif LLOSS == "dense":                                       # addendum 203: dense forward KL to the (corrected) teacher
                kp = max(1, min(x.shape[1] - s0 - 1, n + DCTX))
                with pm.disable_adapter(), torch.no_grad():
                    if LTEACH == "bridge":                                   # review 203: E4B's own logits are never built (memory)
                        tgt = teacher_lp(mT, mS, gT, x, c, kp, br)
                    elif LTEACH == "control":
                        tgt = control_lp(mS, ctl, x, c, kp)
                    else:
                        tgt = Fn.log_softmax(mT(input_ids=x, logits_to_keep=kp + 1, use_cache=False).logits[0, :-1].float(), -1)
                    tv, ti = tgt.topk(X.TOPK, -1); del tgt
                lg = mS(input_ids=x, logits_to_keep=kp + 1, use_cache=False).logits[0, :-1].float(); kd = tail_ce(lg, ti, tv); lg = None   # review 203: freed before backward
            else:
                with torch.no_grad():
                    tv, ti = Fn.log_softmax(mT(input_ids=x, logits_to_keep=n + 1, use_cache=False).logits[0, :-1].float(), -1).topk(X.TOPK, -1)
                lg = mS(input_ids=x, logits_to_keep=n + 1, use_cache=False).logits[0, :-1].float(); kd = tail_ce(lg, ti, tv)
            (kd / ACC).backward(); kds.append(float(kd.detach())); del lg
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); hist.append(np.mean(kds))
        if step % 25 == 0 or step == STEPS - 1:
            log("lora step", step, "cross-entropy to the 31B (with tail bin)", round(float(np.mean(hist[-25:])), 4), round(time.time() - t0), "s")
    assert np.isfinite(np.array(hist[-25:])).all(), "training loss is not finite"
    pm.save_pretrained(str(OD / LOUT)); json.dump({"layers": lay, "rank": rk, "lr": llr, "steps": STEPS, "loss": LLOSS, "pool": LPOOL, "items": len(tr), "last": float(np.mean(hist[-25:])), **({"teacher": LTEACH, "dense context": DCTX, "corrector": os.environ.get("ON_BRIDGE_TR") if LTEACH == "bridge" else os.environ.get("TR_CONTROL") if LTEACH == "control" else None} if LLOSS == "dense" else {})}, open(OD / LOUT / "train.json", "w"))
    if torch.cuda.is_available():
        log("peak GPU memory", round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), "GB")   # review 192
    print("TRAIN_DONE lora", flush=True)


@torch.no_grad()
def gate_lora():
    from peft import PeftModel
    ids, _, held = data(); mT, mS, gT, gS = load_models(); tok = X.tok_(); pm = PeftModel.from_pretrained(mS, str(OD / LNAME)); pm.eval()
    call = [tok.convert_tokens_to_ids("<|tool_call>")] + tok.encode("call:", add_special_tokens=False); sub = tok.encode("submit_patch", add_special_tokens=False)[0]
    pick = [held[i] for i in np.random.default_rng(11).permutation(len(held))[:NGATE]]; arms = ("E4B alone", "LoRA")   # the same points as gate()
    acc = {k: [] for k in arms}; agree = {k: [] for k in arms}; subp = {k: [] for k in list(arms) + ["31B"]}; dec = {k: [] for k in arms}

    def s_logits(x, keep, on):
        if on:
            return mS(input_ids=x, logits_to_keep=keep, use_cache=False).logits[0].float()
        with pm.disable_adapter():
            return mS(input_ids=x, logits_to_keep=keep, use_cache=False).logits[0].float()
    for r in pick:
        x, c, n, s0 = seq(ids, r); tlg = mT(input_ids=x, logits_to_keep=n + 1, use_cache=False).logits[0, :-1].float(); tv, ti = Fn.log_softmax(tlg, -1).topk(X.TOPK, -1)
        rep_ = x[0, c:].tolist(); at = next((i + len(call) for i in range(len(rep_) - len(call)) if rep_[i:i + len(call)] == call), None)
        lgs = {k: s_logits(x, n + 1, k == "LoRA")[:-1] for k in arms}
        for k, lg in lgs.items():
            acc[k].append(float(-(tv.softmax(-1) * Fn.log_softmax(lg, -1).gather(-1, ti)).sum(-1).mean()))
            if at is not None and at < len(lg):
                agree[k].append(int(lg[at].argmax() == tlg[at].argmax()))
                if r["action"] == "submit_patch":
                    subp[k].append(float(Fn.softmax(lg[at], -1)[sub]))
        if at is not None and at < len(tlg) and r["action"] == "submit_patch":
            subp["31B"].append(float(Fn.softmax(tlg[at], -1)[sub]))
        if r["action"] != "text":
            x2 = torch.cat([x[:, :c], torch.tensor([call], device=DEV)], 1); t31 = int(mT(input_ids=x2, logits_to_keep=1, use_cache=False).logits[0, -1].argmax())
            d = {k: s_logits(x2, 1, k == "LoRA")[-1] for k in arms}; dis = int(d["E4B alone"].argmax()) != t31
            for k in arms:
                dec[k].append((int(int(d[k].argmax()) == t31), float(Fn.softmax(d[k], -1)[t31]), dis))
    res = {"arch": "lora", "decision points": len(pick), "cross-entropy to the 31B along E4B's reply (lower is closer)": {k: round(float(np.mean(v)), 4) for k, v in acc.items()},
           "tool choice: top token matches the 31B's (share)": {k: round(float(np.mean(v)), 4) for k, v in agree.items()}, "tool tokens found": len(agree["E4B alone"]),
           "at E4B's own submits: probability of submit_patch": {k: round(float(np.mean(v)), 4) for k, v in subp.items()}, "E4B submits in the sample": len(subp["31B"]),
           "decision without thinking (context, then the tool-call opening), at E4B's tool calls": {
               "points": len(dec["E4B alone"]), "where E4B alone's choice differs from the 31B's": sum(v[2] for v in dec["E4B alone"]),
               "top token matches the 31B's (share)": {k: round(float(np.mean([v[0] for v in vs])), 4) for k, vs in dec.items()},
               "probability of the 31B's choice": {k: round(float(np.mean([v[1] for v in vs])), 4) for k, vs in dec.items()},
               "where E4B alone differs: share now on the 31B's choice": {k: round(float(np.mean([v[0] for v in vs if v[2]])), 4) if any(v[2] for v in vs) else None for k, vs in dec.items()}},
           "lora": json.load(open(OD / LNAME / "train.json")), "adapter": LNAME}
    json.dump({"adapter": LNAME, "decision points (matches the teacher, probability of its choice, student alone differs)": {k: dec[k] for k in arms},
               "cross-entropy per row": {k: acc[k] for k in arms}}, open(OD / f"gate_{LNAME}_points.json", "w"))   # 11-07: per point, for paired tests
    json.dump(res, open(OD / ("gate_lora.json" if LNAME == "lora_31b" else f"gate_{LNAME}.json"), "w"), indent=1); log(json.dumps(res)); print("GATE_DONE lora", flush=True)



def call_ids(tok):
    return [tok.convert_tokens_to_ids("<|tool_call>")] + tok.encode("call:", add_special_tokens=False), tok.encode("submit_patch", add_special_tokens=False)[0]


def after_submit(ids, r, call, sub):
    """True when E4B's context already holds a submit_patch call: a post-submit point (S4c leaves these out, as Kaggle rows end at submit)"""
    ctx = np.asarray(ids[r["off"]:r["off"] + r["ctx_len"]]); pat = np.asarray(call + [sub]); k = len(call)
    return any(h >= k and (ctx[h - k:h + 1] == pat).all() for h in np.nonzero(ctx == sub)[0])


def ends_of(y, eos):
    """K x T sampled ids -> K x T mask: tokens up to and including the first end token (all T when none)"""
    T = y.shape[1]; isend = torch.zeros_like(y, dtype=torch.bool)
    for e in eos:
        isend |= y == e
    first = torch.where(isend.any(1), isend.int().argmax(1), torch.full((y.shape[0],), T - 1, device=y.device))
    return torch.arange(T, device=y.device)[None] <= first[:, None], isend.any(1)


def eos_of(m):
    e = m.generation_config.eos_token_id
    return list(e) if isinstance(e, (list, tuple)) else [e]


BRIDGE_TR = os.environ.get("ON_BRIDGE_TR"); OTAG = "_bridge" if BRIDGE_TR else ""   # addendum 146: C and D, the bridged teacher
B1 = os.environ.get("ON_B1") == "1"; B1_KL = float(os.environ.get("ON_B1_KL", "1.0")); OTAG += "_b1" if B1 else ""   # addendum 148: E and F, the B1 mask
MIX = os.environ.get("ON_MIX") == "1"; MIX_W = float(os.environ.get("ON_MIX_W", "0.5")); OTAG += "_mix" if MIX else ""   # addendum 159: G, both teachers


def bridge_setup(mS):
    """the translator ON_BRIDGE_TR (in OD), trained in the configuration now in the environment (checked against its checkpoint)"""
    ck = torch.load(OD / f"{BRIDGE_TR}.pt", weights_only=False, map_location=DEV)
    want = {"reply": REPLY, "span": SPAN, "all_kv": os.environ.get("TR_ALL_KV"), "src depths": os.environ.get("TR_SRC_DEPTHS")}
    got = {k: ck.get(k) for k in want}; assert got == want, f"translator {BRIDGE_TR} was trained as {got}, the environment says {want}"
    assert ck.get("steps", 0) > 20, f"translator {BRIDGE_TR} has {ck.get('steps')} steps: a small-test translator"
    log("bridged teacher:", BRIDGE_TR, "steps", ck.get("steps"), "ridge R2", round(float(ck.get("ridge R2", float("nan"))), 4), "last", ck.get("last"))
    T = X.Tr(ck["din"], ck["arch"], {k: ck["state"][k].cpu().numpy() for k in ("mx", "sx", "my", "sy")}).to(DEV); T.load_state_dict(ck["state"]); T.eval()
    return T, X.Patch(mS, T.sites)


@torch.no_grad()
def teacher_lp(mT, mS, gT, s_i, c, L, br=None):
    """log-probabilities of the L sampled tokens of s_i under the teacher: the 31B itself, or (br = (T, P)) E4B with the translator writing the
    31B's states over [s0, end): the context span, and the sample too when ON_REPLY=1. The caller turns the student's own change off."""
    if br is None:
        return Fn.log_softmax(mT(input_ids=s_i, logits_to_keep=L + 1, use_cache=False).logits[0, :-1].float(), -1)
    T, P = br; n_ = s_i.shape[1]; s0 = max(1, c - SPAN); end = n_ if REPLY else c
    gT.sl = slice(s0, end); gT.out = {}; mT(input_ids=s_i, logits_to_keep=1, use_cache=False); gT.sl = None
    src = torch.cat([gT.out[l] for l in X.SRC[TT]], -1); gT.out = {}; write(P, T, src, s0, end, n_)
    lp = Fn.log_softmax(mS(input_ids=s_i, logits_to_keep=L + 1, use_cache=False).logits[0, :-1].float(), -1); P.slot.clear(); return lp


@torch.no_grad()
def teacher_lp_mix(mT, mS, gT, s_i, c, L, br, w):
    """addendum 159: both teachers from one pass of the 31B (its own log-probabilities, and its states for the bridge), then E4B bridged by br;
    returns log_softmax(w * 31B + (1 - w) * bridged E4B), the normalised geometric mean: reverse KL to it equals w KL(student || 31B) +
    (1 - w) KL(student || bridged E4B) up to a constant. The caller turns the student's own change off."""
    T, P = br; n_ = s_i.shape[1]; s0 = max(1, c - SPAN); end = n_ if REPLY else c
    gT.sl = slice(s0, end); gT.out = {}; l1 = Fn.log_softmax(mT(input_ids=s_i, logits_to_keep=L + 1, use_cache=False).logits[0, :-1].float(), -1); gT.sl = None
    src = torch.cat([gT.out[l] for l in X.SRC[TT]], -1); gT.out = {}; write(P, T, src, s0, end, n_)
    l2 = Fn.log_softmax(mS(input_ids=s_i, logits_to_keep=L + 1, use_cache=False).logits[0, :-1].float(), -1); P.slot.clear()
    return Fn.log_softmax(w * l1 + (1 - w) * l2, -1)


def train_opd():
    """S4c (addendum 132): true on-policy distillation. Each step, the LoRA student samples OPD_K replies (temperature 1, no top-k/top-p) at one
    of E4B's decision points; the 31B scores every sampled position; the loss is the exact per-position reverse KL(student || 31B) over the full
    vocabulary, averaged over the sampled tokens (GKD with the student's own samples, mode-seeking). Post-submit points are left out."""
    from peft import LoraConfig, get_peft_model
    ids, tr, _ = data(); mT, mS, gT, gS = load_models(); tok = X.tok_(); rng = np.random.default_rng(7); torch.manual_seed(7); t0 = time.time(); call, sub = call_ids(tok)
    keep = [r for r in tr if not after_submit(ids, r, call, sub)]; log("training points", len(tr), "post-submit left out", len(tr) - len(keep))
    tm, lay = lora_targets(mS); rk = int(os.environ.get("ON_RANK", "16")); llr = float(os.environ.get("ON_LORA_LR", "3e-4"))
    br = bridge_setup(mS) if BRIDGE_TR else None                         # before the LoRA, so the translator's hooks sit on the base layers
    assert not MIX or br is not None, "ON_MIX needs ON_BRIDGE_TR"
    pm = get_peft_model(mS, LoraConfig(r=rk, lora_alpha=2 * rk, lora_dropout=0.0, target_modules=tm, bias="none")); params = [p_ for n_, p_ in mS.named_parameters() if "lora_" in n_]
    log("LoRA layers", lay, "rank", rk, "lr", llr, "trainable", sum(p_.numel() for p_ in params), "samples", OPD_K, "x up to", OPD_GEN, "tokens", "| teacher:", (f"{MIX_W} x the 31B + {1 - MIX_W} x E4B bridged by {BRIDGE_TR} (geometric mean)" if MIX else f"E4B bridged by {BRIDGE_TR}") if br else "the 31B", "| B1 mask" if B1 else "")
    mS.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); eos = eos_of(mS)
    opt = torch.optim.AdamW(params, lr=llr, weight_decay=0.0); sched = lambda s_: min(1.0, (s_ + 1) / 15) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s_ / OPD_STEPS)))  # noqa: E731
    hist, lens, ended = [], [], []; ndec = nall = 0; tc_id = tok.convert_tokens_to_ids("<|tool_call>"); hdec = []
    for step in range(OPD_STEPS):
        for g in opt.param_groups:
            g["lr"] = llr * sched(step)
        r = max(keep, key=lambda q: q["ctx_len"]) if SMALL and step == 0 else keep[int(rng.integers(len(keep)))]   # the small test hits peak memory
        x, c, n, s0 = seq(ids, r); ctx = x[:, :c]
        if c > CAP - OPD_GEN:                                            # room for the sample: at most CAP tokens in all
            ctx = torch.cat([ctx[:, :HEAD], ctx[:, -(CAP - OPD_GEN - HEAD):]], 1); c = ctx.shape[1]
        pm.eval()
        with torch.no_grad():
            out = pm.generate(input_ids=ctx, attention_mask=torch.ones_like(ctx), do_sample=True, temperature=1.0, top_k=0, top_p=1.0, max_new_tokens=OPD_GEN,
                              num_return_sequences=OPD_K, eos_token_id=eos, pad_token_id=tok.pad_token_id, use_cache=True)
        y = out[:, c:]; m, fin = ends_of(y, eos); pm.train(); opt.zero_grad(set_to_none=True); kls = []; Ltot = int(m.sum())   # loss: mean over all sampled tokens
        kdec = []
        for i in range(OPD_K):
            L = int(m[i].sum()); s_i = torch.cat([ctx, y[i:i + 1, :L]], 1)
            if MIX:
                with pm.disable_adapter():
                    lt = teacher_lp_mix(mT, mS, gT, s_i, c, L, br, MIX_W)
            elif br is None:
                lt = teacher_lp(mT, mS, gT, s_i, c, L)
            else:
                with pm.disable_adapter():
                    lt = teacher_lp(mT, mS, gT, s_i, c, L, br)
            if B1:                                                       # addendum 148: the teacher on the action only, base E4B everywhere else
                yi = y[i, :L]; hit = (yi == tc_id).nonzero()
                dm = torch.zeros(L, dtype=torch.bool, device=DEV)
                if len(hit):
                    dm[int(hit[0, 0]):] = True                           # from the token that opens the tool call to the end of the sample
                with torch.no_grad(), pm.disable_adapter():
                    lb = Fn.log_softmax(mS(input_ids=s_i, logits_to_keep=L + 1, use_cache=False).logits[0, :-1].float(), -1)
                lt = torch.where(dm[:, None], lt, lb); wts = torch.where(dm, 1.0, B1_KL); del lb
                ndec += int(dm.sum()); nall += L
            ls = Fn.log_softmax(mS(input_ids=s_i, logits_to_keep=L + 1, use_cache=False).logits[0, :-1].float(), -1)
            assert ls.shape == lt.shape == (L, ls.shape[-1]), (ls.shape, lt.shape, L)
            kl = (ls.exp() * (ls - lt)).sum(-1)
            if B1:
                kdec.append((float(kl.detach()[dm].sum()), int(dm.sum()))); kl = kl * wts
            (kl.sum() / Ltot).backward(); kls.append(float(kl.detach().sum())); del lt, ls
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        hdec.append(sum(a_ for a_, _ in kdec) / max(1, sum(b_ for _, b_ in kdec)) if B1 else 0.0)   # this step's action tokens
        hist.append(sum(kls) / Ltot); lens.append(float(m.sum(1).float().mean())); ended.append(float(fin.float().mean()))
        if step % 10 == 0 or step == OPD_STEPS - 1:
            log("opd step", step, f"reverse KL to the {'mixed teacher' if MIX else 'bridged E4B' if BRIDGE_TR else '31B'} on own samples", round(float(np.mean(hist[-10:])), 4), "sample length", round(float(np.mean(lens[-10:]))),
                "ended", round(float(np.mean(ended[-10:])), 2), *(("| B1: action tokens", round(ndec / max(1, nall), 3), "reverse KL to the teacher on them", round(float(np.mean(hdec[-10:])), 4)) if B1 else ()), round(time.time() - t0), "s")
    assert np.isfinite(np.array(hist[-10:])).all(), "training loss is not finite"
    pm.save_pretrained(str(OD / f"lora_opd{OTAG}"))
    json.dump({"layers": lay, "rank": rk, "lr": llr, "steps": OPD_STEPS, "samples": OPD_K, "max_new": OPD_GEN, "first10": float(np.mean(hist[:10])),
               "last": float(np.mean(hist[-10:])), "post-submit left out": len(tr) - len(keep), "teacher": BRIDGE_TR or "31b", "mix": {"31b weight": MIX_W} if MIX else None, "b1": B1, "b1 base KL weight": B1_KL if B1 else None, "action token share": ndec / max(1, nall) if B1 else None}, open(OD / f"lora_opd{OTAG}" / "train.json", "w"))
    print("TRAIN_DONE opd", flush=True)


def rep_line(text, k=3, w=20):
    """a non-trivial line (>= w chars) that occurs at least k times: the runaway-thinking pattern seen live (addendum 131)"""
    from collections import Counter
    c = Counter(l_.strip() for l_ in text.split("<|tool_call>")[0].split("\n") if len(l_.strip()) >= w)   # thinking and text only, not call arguments
    return bool(c) and max(c.values()) >= k


@torch.no_grad()
def gate_loops():
    """S4c loop gate (addendum 132), student only, the harness's sampling (temperature 0.2, top-p 0.95). Arms: E4B alone, every adapter folder
    present among lora_31b (S4b) and lora_opd (S4c). (a) NLOOP held-out decision points before any submit, 2 samples each, up to LOOP_GEN tokens:
    share not ended, share with a line repeated 3+ times, mean length. (b) NPOST held-out post-submit points, 2 samples each, up to POST_GEN (512) tokens:
    share that open a tool call (live: S4b resubmits 95%, plain 2%)."""
    from peft import PeftModel
    ids, _, held = data(); mS = X.load(ST); tok = X.tok_(); call, sub = call_ids(tok); eos = eos_of(mS)
    names = [k for k in os.environ.get("ON_LOOP_ADAPTERS", "lora_31b,lora_opd").split(",") if (OD / k / "adapter_config.json").exists()]; assert names, "no adapter folder in " + str(OD)
    pm = PeftModel.from_pretrained(mS, str(OD / names[0]), adapter_name=names[0])
    for k in names[1:]:
        pm.load_adapter(str(OD / k), adapter_name=k)
    pm.eval(); rng = np.random.default_rng(13); order = rng.permutation(len(held))
    pre = [held[i] for i in order if not after_submit(ids, held[i], call, sub)][:NLOOP]; post = [held[i] for i in order if after_submit(ids, held[i], call, sub)][:NPOST]
    log("loop gate:", len(pre), "pre-submit points,", len(post), "post-submit points; arms E4B alone +", names)
    tc = tok.convert_tokens_to_ids("<|tool_call>")

    def sample(ctx, arm, new):
        kw = dict(input_ids=ctx, attention_mask=torch.ones_like(ctx), do_sample=True, temperature=0.2, top_p=0.95, top_k=0, max_new_tokens=new,
                  num_return_sequences=2, eos_token_id=eos, pad_token_id=tok.pad_token_id, use_cache=True)
        if arm == "E4B alone":
            with pm.disable_adapter():
                return pm.generate(**kw)[:, ctx.shape[1]:]
        pm.set_adapter(arm); return pm.generate(**kw)[:, ctx.shape[1]:]
    res = {"arms": ["E4B alone"] + names, "pre-submit points": len(pre), "post-submit points": len(post)}
    for arm in res["arms"]:
        torch.manual_seed(0); nend, rep, ln, opens = [], [], [], []
        for r in pre:
            x, c, n, s0 = seq(ids, r); y = sample(x[:, :c], arm, LOOP_GEN); m, fin = ends_of(y, eos)
            for i in range(y.shape[0]):
                L = int(m[i].sum()); nend.append(int(not fin[i])); ln.append(L); rep.append(int(rep_line(tok.decode(y[i, :L].tolist(), skip_special_tokens=False))))
        pend = []
        for r in post:
            x, c, n, s0 = seq(ids, r); y = sample(x[:, :c], arm, POST_GEN); m, fin = ends_of(y, eos)
            for i in range(y.shape[0]):
                opens.append(int((y[i, :int(m[i].sum())] == tc).any())); pend.append(int(not fin[i]))
        res[arm] = {"not ended within the cap (share)": round(float(np.mean(nend)), 3), "a line repeated 3+ times (share)": round(float(np.mean(rep)), 3),
                    "mean sample length": round(float(np.mean(ln))) if ln else None, "post-submit: opens a tool call (share)": round(float(np.mean(opens)), 3) if opens else None,
                    "post-submit: not ended within the cap (share)": round(float(np.mean(pend)), 3) if pend else None}
        log(arm, json.dumps(res[arm]))
    json.dump(res, open(OD / "gate_loops.json", "w"), indent=1); log(json.dumps(res)); print("GATE_DONE loops", flush=True)



NREACH = 10 if SMALL else int(os.environ.get("ON_NREACH", "120")); BUCKETS = ((0, 16), (16, 64), (64, 256), (256, 10 ** 6))


@torch.no_grad()
def gate_reach(arch):
    """What the bridge sends (addendum 138, user 11-06: "verify that both the outputs and desisions are bridged via the translator. we may have
    missed what is being sent when its a more complex task"). As trained and served, the 31B's translated states are written only over the last
    SPAN context tokens [s0, c); the reply [c, c+n) gets nothing from the 31B. Here, on held-out points, five arms: E4B alone; bridge and control
    over the context span only (as trained); bridge and control over the span and the whole reply (the 31B and E4B read E4B's reply teacher-forced,
    their states at every reply position are translated and written there too; the translators never saw reply positions). Per reply-position
    bucket: cross-entropy to the 31B (top 32) and top-1 agreement with the 31B; and the tool choice after thinking (the tool name's first token)."""
    ids, _, held = data(); mT, mS, gT, gS = load_models(); tok = X.tok_(); mS.eval(); Ts = {}
    for src in (TT, ST):
        ck = torch.load(OD / f"{arch}_{src}.pt", weights_only=False, map_location=DEV); T = X.Tr(ck["din"], ck["arch"], {k: ck["state"][k].cpu().numpy() for k in ("mx", "sx", "my", "sy")}).to(DEV); T.load_state_dict(ck["state"]); Ts[src] = T.eval()
    P = X.Patch(mS, Ts[TT].sites); call = [tok.convert_tokens_to_ids("<|tool_call>")] + tok.encode("call:", add_special_tokens=False)
    pick = [held[i] for i in np.random.default_rng(11).permutation(len(held))[:NREACH]]
    arms = ("E4B alone", "bridge, context only", "control, context only", "bridge, context and reply", "control, context and reply")
    ce = {k: {b: [] for b in BUCKETS} for k in arms}; top = {k: {b: [] for b in BUCKETS} for k in arms}; tool = {k: [] for k in arms}
    for r in pick:
        x, c, n, s0 = seq(ids, r); L = x.shape[1]
        gT.sl = slice(s0, L); gT.out = {}; tlg = mT(input_ids=x, logits_to_keep=n + 1, use_cache=False).logits[0, :-1].float(); gT.sl = None
        Tf = torch.cat([gT.out[l] for l in X.SRC[TT]], -1); tv, ti = Fn.log_softmax(tlg, -1).topk(X.TOPK, -1); tp = tv.softmax(-1); t1 = tlg.argmax(-1)
        gS.sl = slice(s0, L); gS.out = {}; slg = mS(input_ids=x, logits_to_keep=n + 1, use_cache=False).logits[0, :-1].float(); gS.sl = None
        Sf = torch.cat([gS.out[l] for l in X.SRC[ST]], -1)
        lgs = {"E4B alone": slg}
        for k in arms[1:]:
            src, T = (Tf, Ts[TT]) if k.startswith("bridge") else (Sf, Ts[ST]); end = c if k.endswith("context only") else L
            write(P, T, src[:end - s0], s0, end, L); lgs[k] = mS(input_ids=x, logits_to_keep=n + 1, use_cache=False).logits[0, :-1].float(); P.slot.clear()
        rep = x[0, c:].tolist(); at = next((i + len(call) for i in range(len(rep) - len(call)) if rep[i:i + len(call)] == call), None)
        for k, lg in lgs.items():
            cpos = -(tp * Fn.log_softmax(lg, -1).gather(-1, ti)).sum(-1); hit = (lg.argmax(-1) == t1).float()
            for b in BUCKETS:
                if n > b[0]:
                    ce[k][b].append(float(cpos[b[0]:min(n, b[1])].mean())); top[k][b].append(float(hit[b[0]:min(n, b[1])].mean()))
            if at is not None and at < n:
                tool[k].append(int(lg[at].argmax() == t1[at]))
    nm = lambda b: f"reply tokens {b[0]}-{b[1] - 1}" if b[1] < 10 ** 6 else f"reply tokens {b[0]}+"  # noqa: E731
    res = {"arch": arch, "points": len(pick), "points per bucket": {nm(b): len(ce["E4B alone"][b]) for b in BUCKETS},
           "cross-entropy to the 31B (lower is closer)": {k: {nm(b): round(float(np.mean(v)), 4) for b, v in ce[k].items() if v} for k in arms},
           "top-1 agrees with the 31B (share)": {k: {nm(b): round(float(np.mean(v)), 4) for b, v in top[k].items() if v} for k in arms},
           "tool choice after thinking matches the 31B's (share)": {k: round(float(np.mean(v)), 4) for k, v in tool.items() if v}, "tool tokens found": len(tool["E4B alone"])}
    json.dump(res, open(OD / f"gate_reach_{arch}.json", "w"), indent=1); log(json.dumps(res)); print("GATE_DONE reach", arch, flush=True)



BGL_STEPS = 20 if SMALL else int(os.environ.get("ON_BGL_STEPS", "600")); BGL_LR = float(os.environ.get("ON_BGL_LR", "1e-3")); BGL_LORA_STEPS = 20 if SMALL else int(os.environ.get("ON_BGL_LORA_STEPS", "600")); BGL_K = 64


def bgl_graft(name):
    """Addendum 140, step 1 (user 11-06: "combine this learning with the graft created lora once we hopefully fix the thinking tokens not being
    sent"): a gated graft (store.py Graft, at E4B's graft layers) learns the bridged E4B's next-token distribution along E4B's own replies.
    Teacher: E4B with translator NAME (ARCH_SOURCE, trained with ON_REPLY=1) writing the source's states over the span and the reply; the graft
    is off. Student: E4B with the graft, no bridge. Loss: top 64 plus the tail bin (R145). Post-submit points left out (as S4c)."""
    import store as STR
    direct = name == "direct"                                            # addendum 142: the teacher is the 31B's own next-token distribution (no translator)
    assert direct or REPLY, "bgl uses a translator trained with ON_REPLY=1"
    arch, source = ("direct", TT) if direct else name.split("_"); ids, tr, _ = data(); mT, mS, gT, gS = load_models(); tok = X.tok_(); rng = np.random.default_rng(7); torch.manual_seed(7); t0 = time.time()
    call, sub = call_ids(tok); keep = [r for r in tr if not after_submit(ids, r, call, sub)]
    if not direct:
        ck = torch.load(OD / f"{name}.pt", weights_only=False, map_location=DEV); assert ck.get("reply") is True, "the translator was not trained with ON_REPLY=1"
        T = X.Tr(ck["din"], ck["arch"], {k: ck["state"][k].cpu().numpy() for k in ("mx", "sx", "my", "sy")}).to(DEV); T.load_state_dict(ck["state"]); T.eval(); P = X.Patch(mS, T.sites)
    mods = {l: STR.Graft().to(DEV) for l in STR.GRAFT_LAYERS}; A = STR.Attach(mS, mods); A.on = False; params = [p_ for mod in mods.values() for p_ in mod.parameters()]
    log("graft layers", STR.GRAFT_LAYERS, "trainable", sum(p_.numel() for p_ in params), "teacher:", "the 31B's own outputs" if direct else f"E4B bridged by {name}", "| training points", len(keep), "of", len(tr))
    mS.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); mS.disable_input_require_grads(); mS.train()
    opt = torch.optim.AdamW(params, lr=BGL_LR, weight_decay=0.0); sched = lambda s_: min(1.0, (s_ + 1) / 30) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s_ / BGL_STEPS)))  # noqa: E731
    hist = []
    for step in range(BGL_STEPS):
        for g in opt.param_groups:
            g["lr"] = BGL_LR * sched(step)
        opt.zero_grad(set_to_none=True); ls_, gs_ = [], []
        for _ in range(ACC):
            r = keep[int(rng.integers(len(keep)))]; x, c, n, s0 = seq(ids, r); L = x.shape[1]; A.on = False
            with torch.no_grad():
                if direct:
                    tv, ti = Fn.log_softmax(mT(input_ids=x, logits_to_keep=n + 1, use_cache=False).logits[0, :-1].float(), -1).topk(BGL_K, -1)
                else:
                    o = look(mT, mS, gT, gS, x, c, n, s0, teacher=(source == TT)); src = o["T"] if source == TT else o["S"]
                    write(P, T, src, s0, s0 + src.shape[0], L); tv, ti = Fn.log_softmax(mS(input_ids=x, logits_to_keep=n + 1, use_cache=False).logits[0, :-1].float(), -1).topk(BGL_K, -1); P.slot.clear(); del o
            A.on = True; A.g = []; lg = mS(input_ids=x, logits_to_keep=n + 1, use_cache=False).logits[0, :-1].float()
            loss = tail_ce(lg, ti, tv); (loss / ACC).backward(); A.on = False   # on through backward: checkpointing re-runs the hooks (review 11-06)
            ls_.append(float(loss.detach())); gs_.append(float(torch.cat([g_.flatten() for g_ in A.g[:len(mods)]]).mean())); del lg
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); hist.append((np.mean(ls_), np.mean(gs_)))
        if step % 25 == 0 or step == BGL_STEPS - 1:
            h = np.mean(hist[-25:], 0); log("graft step", step, "cross-entropy to the teacher", round(float(h[0]), 4), "mean switch", round(float(h[1]), 3), round(time.time() - t0), "s")
    assert np.isfinite(np.array(hist[-25:])).all(), "graft loss is not finite"
    torch.save({"graft": {l: mod.state_dict() for l, mod in mods.items()}, "layers": STR.GRAFT_LAYERS, "translator": name, "steps": BGL_STEPS, "last": [float(v) for v in np.mean(hist[-25:], 0)]}, OD / f"graft_{name}.pt")
    print("TRAIN_DONE graft", name, flush=True)


def bgl_lora(name):
    """Addendum 140, step 2: the graft copied into a graft-shaped LoRA (rank 16, MLP at the same layers, lr ON_LORA_LR 3e-4, R146), trained to the
    grafted E4B's top 64 plus the tail bin along E4B's replies (the same points as step 1). Saved as lora_bgl_NAME (PEFT, vLLM-servable)."""
    import store as STR
    from peft import LoraConfig, get_peft_model
    ids, tr, _ = data(); mS = X.load(ST); tok = X.tok_(); rng = np.random.default_rng(8); torch.manual_seed(8); t0 = time.time(); call, sub = call_ids(tok)
    keep = [r for r in tr if not after_submit(ids, r, call, sub)]
    ck = torch.load(OD / f"graft_{name}.pt", weights_only=False, map_location=DEV); mods = {l: STR.Graft().to(DEV) for l in ck["layers"]}
    for l, mod in mods.items():
        mod.load_state_dict(ck["graft"][l]); mod.eval()
        for p_ in mod.parameters():
            p_.requires_grad_(False)
    A = STR.Attach(mS, mods); A.on = False
    tm, lay = lora_targets(mS); rk = int(os.environ.get("ON_RANK", "16")); llr = float(os.environ.get("ON_LORA_LR", "3e-4"))
    assert lay == list(ck["layers"]), (lay, ck["layers"])
    pm = get_peft_model(mS, LoraConfig(r=rk, lora_alpha=2 * rk, lora_dropout=0.0, target_modules=tm, bias="none")); params = [p_ for n_, p_ in mS.named_parameters() if "lora_" in n_]
    mS.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); mS.disable_input_require_grads(); mS.train()
    opt = torch.optim.AdamW(params, lr=llr, weight_decay=0.0); sched = lambda s_: min(1.0, (s_ + 1) / 30) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s_ / BGL_LORA_STEPS)))  # noqa: E731
    hist = []
    for step in range(BGL_LORA_STEPS):
        for g in opt.param_groups:
            g["lr"] = llr * sched(step)
        opt.zero_grad(set_to_none=True); ls_ = []
        for _ in range(ACC):
            r = keep[int(rng.integers(len(keep)))]; x, c, n, s0 = seq(ids, r)
            with torch.no_grad(), pm.disable_adapter():
                A.on = True; tv, ti = Fn.log_softmax(mS(input_ids=x, logits_to_keep=n + 1, use_cache=False).logits[0, :-1].float(), -1).topk(BGL_K, -1); A.on = False; A.g.clear()
            lg = mS(input_ids=x, logits_to_keep=n + 1, use_cache=False).logits[0, :-1].float(); loss = tail_ce(lg, ti, tv); (loss / ACC).backward(); ls_.append(float(loss.detach())); del lg
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); hist.append(np.mean(ls_))
        if step % 25 == 0 or step == BGL_LORA_STEPS - 1:
            log("graft -> LoRA step", step, "cross-entropy to the grafted E4B", round(float(np.mean(hist[-25:])), 4), round(time.time() - t0), "s")
    assert np.isfinite(np.array(hist[-25:])).all(), "graft -> LoRA loss is not finite"
    out = OD / f"lora_bgl_{name}"; pm.save_pretrained(str(out))
    json.dump({"layers": lay, "rank": rk, "lr": llr, "steps": BGL_LORA_STEPS, "last": float(np.mean(hist[-25:])), "from graft": f"graft_{name}.pt", "graft last": ck["last"]}, open(out / "train.json", "w"))
    print("TRAIN_DONE bgl lora", name, flush=True)



@torch.no_grad()
def gate_rung(arch):
    """Addendum 140 ladder gate: the translators ARCH_TEACHER (bridge) and ARCH_STUDENT (control, when its checkpoint exists) in the configuration
    they were trained in (ON_REPLY, ON_SPAN, TR_ALL_KV, TR_SRC_DEPTHS from the environment), against the student alone, on NREACH held-out points:
    per reply-position bucket, cross-entropy to the teacher (top 32) and top-1 agreement with it; the tool choice after thinking; the decision
    without thinking (context, then the tool-call opening; under ON_REPLY the opening tokens are written too)."""
    ids, _, held = data(); mT, mS, gT, gS = load_models(); tok = X.tok_(); mS.eval(); Ts = {}
    for src in (TT, ST):
        f = OD / f"{arch}_{src}.pt"
        if f.exists():
            ck = torch.load(f, weights_only=False, map_location=DEV); T = X.Tr(ck["din"], ck["arch"], {k: ck["state"][k].cpu().numpy() for k in ("mx", "sx", "my", "sy")}).to(DEV); T.load_state_dict(ck["state"]); Ts[src] = T.eval()
    assert TT in Ts, "no bridge translator in " + str(OD)
    P = X.Patch(mS, Ts[TT].sites); call = [tok.convert_tokens_to_ids("<|tool_call>")] + tok.encode("call:", add_special_tokens=False)
    arms = ["alone", "bridge"] + (["control"] if ST in Ts else []); pick = [held[i] for i in np.random.default_rng(11).permutation(len(held))[:NREACH]]
    ce = {k: {b: [] for b in BUCKETS} for k in arms}; top = {k: {b: [] for b in BUCKETS} for k in arms}; tool = {k: [] for k in arms}; dec = {k: [] for k in arms}

    def states(xx, a, b):
        gT.sl = slice(a, b); gT.out = {}; tl = mT(input_ids=xx, logits_to_keep=xx.shape[1] - c + 1, use_cache=False).logits[0].float(); gT.sl = None
        gS.sl = slice(a, b); gS.out = {}; sl_ = mS(input_ids=xx, logits_to_keep=xx.shape[1] - c + 1, use_cache=False).logits[0].float(); gS.sl = None
        return tl, sl_, torch.cat([gT.out[l] for l in X.SRC[TT]], -1), torch.cat([gS.out[l] for l in X.SRC[ST]], -1)

    def arm_logits(k, xx, a, b, srcs, keep):
        if k == "alone":
            return None
        T, src = (Ts[TT], srcs[0]) if k == "bridge" else (Ts[ST], srcs[1])
        write(P, T, src, a, b, xx.shape[1]); lg_ = mS(input_ids=xx, logits_to_keep=keep, use_cache=False).logits[0].float(); P.slot.clear(); return lg_
    whole = {k: [] for k in arms}; whole_top = {k: [] for k in arms}
    for r in pick:
        x, c, n, s0 = seq(ids, r); L = x.shape[1]; end = L if REPLY else c
        tl, sl_, Tsrc, Ssrc = states(x, s0, end); tlg = tl[:-1]; del tl
        tv, ti = Fn.log_softmax(tlg, -1).topk(X.TOPK, -1); tp = tv.softmax(-1); t1 = tlg.argmax(-1); del tlg
        rep = x[0, c:].tolist(); at = next((i + len(call) for i in range(len(rep) - len(call)) if rep[i:i + len(call)] == call), None)
        for k in arms:                                                   # one arm's full-vocabulary logits at a time (memory, review 11-06)
            lg = sl_[:-1] if k == "alone" else arm_logits(k, x, s0, end, (Tsrc, Ssrc), n + 1)[:-1]
            cpos = -(tp * Fn.log_softmax(lg, -1).gather(-1, ti)).sum(-1); hit = (lg.argmax(-1) == t1).float()
            whole[k].append(float(cpos.mean())); whole_top[k].append(float(hit.mean()))   # every reply token of the point counts once
            for b in BUCKETS:
                if n > b[0]:
                    ce[k][b].append(float(cpos[b[0]:min(n, b[1])].mean())); top[k][b].append(float(hit[b[0]:min(n, b[1])].mean()))
            if at is not None and at < n:
                tool[k].append(int(lg[at].argmax() == t1[at]))
            del lg
        del sl_
        if r["action"] != "text":
            x2 = torch.cat([x[:, :c], torch.tensor([call], device=DEV)], 1); L2 = x2.shape[1]; end2 = L2 if REPLY else c
            tl2, sl2, T2, S2 = states(x2, s0, end2); d = {k: (sl2[-1] if k == "alone" else arm_logits(k, x2, s0, end2, (T2, S2), 1)[-1]) for k in arms}
            t31 = int(tl2[-1].argmax()); dis = int(d["alone"].argmax()) != t31
            for k in arms:
                dec[k].append((int(int(d[k].argmax()) == t31), float(Fn.softmax(d[k], -1)[t31]), dis))
    nm = lambda b: f"reply tokens {b[0]}-{b[1] - 1}" if b[1] < 10 ** 6 else f"reply tokens {b[0]}+"  # noqa: E731
    res = {"arch": arch, "teacher": TT, "student": ST, "config": {"reply written": REPLY, "span": SPAN, "TR_ALL_KV": os.environ.get("TR_ALL_KV"), "source depths": os.environ.get("TR_SRC_DEPTHS", "0.35,0.5,0.65"),
           "source layers": {TT: X.SRC[TT], ST: X.SRC[ST]}, "write layer": X.WRITE, "addition layers": list(Ts[TT].sites)}, "points": len(pick),
           "cross-entropy to the teacher (lower is closer)": {k: {nm(b): round(float(np.mean(v)), 4) for b, v in ce[k].items() if v} for k in arms},
           "top-1 agrees with the teacher (share)": {k: {nm(b): round(float(np.mean(v)), 4) for b, v in top[k].items() if v} for k in arms},
           "whole reply, cross-entropy": {k: round(float(np.mean(v)), 4) for k, v in whole.items()}, "whole reply, top-1 agrees": {k: round(float(np.mean(v)), 4) for k, v in whole_top.items()},
           "tool choice after thinking matches the teacher's (share)": {k: round(float(np.mean(v)), 4) for k, v in tool.items() if v}, "tool tokens found": len(tool["alone"]),
           "decision without thinking": {"points": len(dec["alone"]), "where alone differs": sum(v[2] for v in dec["alone"]),
               "matches the teacher (share)": {k: round(float(np.mean([v[0] for v in vs])), 4) for k, vs in dec.items() if vs},
               "probability of the teacher's choice": {k: round(float(np.mean([v[1] for v in vs])), 4) for k, vs in dec.items() if vs},
               "where alone differs: share now on the teacher's choice": {k: round(float(np.mean([v[0] for v in vs if v[2]])), 4) for k, vs in dec.items() if any(v[2] for v in vs)}},
           "translators": {s_: {"ridge R2": round(torch.load(OD / f"{arch}_{s_}.pt", weights_only=False, map_location="cpu")["ridge R2"], 4)} for s_ in Ts}}
    json.dump(res, open(OD / f"gate_rung_{arch}.json", "w"), indent=1); log(json.dumps(res)); print("GATE_DONE rung", arch, flush=True)



OPDG_LR = float(os.environ.get("ON_OPDG_LR", "1e-3")); OPD_G2L_STEPS = 4 if SMALL else int(os.environ.get("ON_OPD_G2L_STEPS", "300"))


def opd_sample(model, ctx, tok, eos):
    with torch.no_grad():
        return model.generate(input_ids=ctx, attention_mask=torch.ones_like(ctx), do_sample=True, temperature=1.0, top_k=0, top_p=1.0, max_new_tokens=OPD_GEN,
                              num_return_sequences=OPD_K, eos_token_id=eos, pad_token_id=tok.pad_token_id, use_cache=True)


def opd_point(ids, keep, rng, step):
    r = max(keep, key=lambda q: q["ctx_len"]) if SMALL and step == 0 else keep[int(rng.integers(len(keep)))]
    x, c, n, s0 = seq(ids, r); ctx = x[:, :c]
    if c > CAP - OPD_GEN:
        ctx = torch.cat([ctx[:, :HEAD], ctx[:, -(CAP - OPD_GEN - HEAD):]], 1)
    return ctx


def train_opd_graft():
    """Addendum 145 (user 11-06: "prepare the on-policy graft code now"): S4c's objective (the student samples OPD_K replies at a decision point,
    the 31B scores every sampled position, exact reverse KL over the full vocabulary, post-submit points left out) trained into gated grafts
    (store.py Graft at E4B's graft layers, lr ON_OPDG_LR) instead of LoRA weights. The grafted E4B samples its own replies. Saved as graft_opd.pt:
    copy it into a LoRA with `opd g2l` (on-policy) or `bgl lora opd` (along E4B's recorded replies)."""
    import store as STR
    ids, tr, _ = data(); mT, mS, gT, gS = load_models(); tok = X.tok_(); rng = np.random.default_rng(7); torch.manual_seed(7); t0 = time.time(); call, sub = call_ids(tok)
    keep = [r for r in tr if not after_submit(ids, r, call, sub)]; log("training points", len(tr), "post-submit left out", len(tr) - len(keep))
    br = bridge_setup(mS) if BRIDGE_TR else None
    mods = {l: STR.Graft().to(DEV) for l in STR.GRAFT_LAYERS}; A = STR.Attach(mS, mods); A.on = True; params = [p_ for mod in mods.values() for p_ in mod.parameters()]
    log("graft layers", STR.GRAFT_LAYERS, "trainable", sum(p_.numel() for p_ in params), "lr", OPDG_LR, "samples", OPD_K, "x up to", OPD_GEN, "tokens")
    mS.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); mS.disable_input_require_grads(); eos = eos_of(mS)
    opt = torch.optim.AdamW(params, lr=OPDG_LR, weight_decay=0.0); sched = lambda s_: min(1.0, (s_ + 1) / 15) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s_ / OPD_STEPS)))  # noqa: E731
    hist, lens, ended, sw = [], [], [], []
    for step in range(OPD_STEPS):
        for g in opt.param_groups:
            g["lr"] = OPDG_LR * sched(step)
        ctx = opd_point(ids, keep, rng, step); c = ctx.shape[1]; mS.eval(); A.on = True
        out = opd_sample(mS, ctx, tok, eos); A.g = []
        y = out[:, c:]; m, fin = ends_of(y, eos); mS.train(); opt.zero_grad(set_to_none=True); kls = []; Ltot = int(m.sum())
        for i in range(OPD_K):
            L = int(m[i].sum()); s_i = torch.cat([ctx, y[i:i + 1, :L]], 1)
            A.on = br is None; lt = teacher_lp(mT, mS, gT, s_i, c, L, br); A.on = True   # the bridged teacher is E4B without the graft
            A.g = []; ls = Fn.log_softmax(mS(input_ids=s_i, logits_to_keep=L + 1, use_cache=False).logits[0, :-1].float(), -1)
            assert ls.shape == lt.shape == (L, ls.shape[-1]), (ls.shape, lt.shape, L)
            kl = (ls.exp() * (ls - lt)).sum(-1); (kl.sum() / Ltot).backward()   # the graft stays on through backward (checkpointing re-runs the hooks)
            kls.append(float(kl.detach().sum())); sw.append(float(torch.cat([g_.flatten() for g_ in A.g[:len(mods)]]).mean())); del lt, ls
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); A.g = []
        hist.append(sum(kls) / Ltot); lens.append(float(m.sum(1).float().mean())); ended.append(float(fin.float().mean()))
        if step % 10 == 0 or step == OPD_STEPS - 1:
            log("opd graft step", step, f"reverse KL to the {'bridged E4B' if BRIDGE_TR else '31B'} on own samples", round(float(np.mean(hist[-10:])), 4), "sample length", round(float(np.mean(lens[-10:]))),
                "ended", round(float(np.mean(ended[-10:])), 2), "mean switch", round(float(np.mean(sw[-40:])), 3), round(time.time() - t0), "s")
    assert np.isfinite(np.array(hist[-10:])).all(), "training loss is not finite"
    torch.save({"graft": {l: mod.state_dict() for l, mod in mods.items()}, "layers": STR.GRAFT_LAYERS, "translator": f"opd ({BRIDGE_TR or 'the 31B'}, on-policy)", "steps": OPD_STEPS,
                "last": [float(np.mean(hist[-10:])), float(np.mean(sw[-40:]))], "first10": float(np.mean(hist[:10]))}, OD / f"graft_opd{OTAG}.pt")
    print("TRAIN_DONE opd graft", flush=True)


def opd_graft_to_lora():
    """Addendum 145, step 2: graft_opd.pt copied into a graft-shaped LoRA on-policy: the LoRA student (graft off) samples OPD_K replies, the
    grafted E4B (LoRA off, graft on) scores them, exact reverse KL. No 31B needed. Saved as lora_opdg (PEFT, vLLM-servable)."""
    import store as STR
    from peft import LoraConfig, get_peft_model
    ids, tr, _ = data(); mS = X.load(ST); tok = X.tok_(); rng = np.random.default_rng(8); torch.manual_seed(8); t0 = time.time(); call, sub = call_ids(tok)
    keep = [r for r in tr if not after_submit(ids, r, call, sub)]
    ck = torch.load(OD / f"graft_opd{OTAG}.pt", weights_only=False, map_location=DEV); mods = {l: STR.Graft().to(DEV) for l in ck["layers"]}
    for l, mod in mods.items():
        mod.load_state_dict(ck["graft"][l]); mod.eval()
        for p_ in mod.parameters():
            p_.requires_grad_(False)
    A = STR.Attach(mS, mods); A.on = False
    tm, lay = lora_targets(mS); rk = int(os.environ.get("ON_RANK", "16")); llr = float(os.environ.get("ON_LORA_LR", "3e-4")); assert lay == list(ck["layers"]), (lay, ck["layers"])
    pm = get_peft_model(mS, LoraConfig(r=rk, lora_alpha=2 * rk, lora_dropout=0.0, target_modules=tm, bias="none")); params = [p_ for n_, p_ in mS.named_parameters() if "lora_" in n_]
    mS.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); mS.disable_input_require_grads(); eos = eos_of(mS)
    opt = torch.optim.AdamW(params, lr=llr, weight_decay=0.0); sched = lambda s_: min(1.0, (s_ + 1) / 15) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s_ / OPD_G2L_STEPS)))  # noqa: E731
    hist = []
    for step in range(OPD_G2L_STEPS):
        for g in opt.param_groups:
            g["lr"] = llr * sched(step)
        ctx = opd_point(ids, keep, rng, step); c = ctx.shape[1]; pm.eval(); A.on = False
        out = opd_sample(pm, ctx, tok, eos); y = out[:, c:]; m, fin = ends_of(y, eos); pm.train(); opt.zero_grad(set_to_none=True); kls = []; Ltot = int(m.sum())
        for i in range(OPD_K):
            L = int(m[i].sum()); s_i = torch.cat([ctx, y[i:i + 1, :L]], 1)
            with torch.no_grad(), pm.disable_adapter():
                A.on = True; lt = Fn.log_softmax(mS(input_ids=s_i, logits_to_keep=L + 1, use_cache=False).logits[0, :-1].float(), -1); A.on = False; A.g.clear()
            ls = Fn.log_softmax(mS(input_ids=s_i, logits_to_keep=L + 1, use_cache=False).logits[0, :-1].float(), -1)
            kl = (ls.exp() * (ls - lt)).sum(-1); (kl.sum() / Ltot).backward(); kls.append(float(kl.detach().sum())); del lt, ls
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); hist.append(sum(kls) / Ltot)
        if step % 10 == 0 or step == OPD_G2L_STEPS - 1:
            log("opd graft -> LoRA step", step, "reverse KL to the grafted E4B on own samples", round(float(np.mean(hist[-10:])), 4), round(time.time() - t0), "s")
    assert np.isfinite(np.array(hist[-10:])).all(), "graft -> LoRA loss is not finite"
    out = OD / f"lora_opdg{OTAG}"; pm.save_pretrained(str(out))
    json.dump({"layers": lay, "rank": rk, "lr": llr, "steps": OPD_G2L_STEPS, "last": float(np.mean(hist[-10:])), "from graft": f"graft_opd{OTAG}.pt", "graft last": ck["last"]}, open(out / "train.json", "w"))
    print("TRAIN_DONE opd graft -> LoRA", flush=True)



@torch.no_grad()
def opd_fidelity():
    """Addendum 145 measurement (review 11-06): on NFID held-out pre-submit points, 2 samples each (temperature 1, up to OPD_GEN tokens), the
    mean per-token reverse KL of each model's own samples to (a) the 31B and (b) the grafted E4B: base E4B, the grafted E4B (graft_opd.pt), and
    each adapter in ON_FID_ADAPTERS (graft off). Shows whether a weak live result comes from the graft (graft vs the 31B) or from the copy (copy
    vs the graft)."""
    import store as STR
    from peft import PeftModel
    NFID = 3 if SMALL else int(os.environ.get("ON_NFID", "20")); names = [k for k in os.environ.get("ON_FID_ADAPTERS", f"lora_bgl_opd{OTAG},lora_opdg{OTAG}").split(",") if (OD / k / "adapter_config.json").exists()]
    ids, _, held = data(); mT, mS, gT, gS = load_models(); tok = X.tok_(); call, sub = call_ids(tok); eos = eos_of(mS); torch.manual_seed(5)
    br = bridge_setup(mS) if BRIDGE_TR else None                         # before PEFT and the graft hooks (review 11-06)
    ck = torch.load(OD / f"graft_opd{OTAG}.pt", weights_only=False, map_location=DEV); mods = {l: STR.Graft().to(DEV).eval() for l in ck["layers"]}
    for l, mod in mods.items():
        mod.load_state_dict(ck["graft"][l])
    A = STR.Attach(mS, mods); A.on = False
    pm = PeftModel.from_pretrained(mS, str(OD / names[0]), adapter_name=names[0]) if names else None
    for k in names[1:]:
        pm.load_adapter(str(OD / k), adapter_name=k)
    if pm is not None:
        pm.eval()
    pts = [r for r in (held[i] for i in np.random.default_rng(17).permutation(len(held))) if not after_submit(ids, r, call, sub)][:NFID]

    def student(arm):                                                    # context manager-free switch: which model samples / is scored
        A.on = arm == "grafted E4B"
        if pm is not None:
            if arm in names:
                pm.set_adapter(arm); pm.base_model.enable_adapter_layers()
            else:
                pm.base_model.disable_adapter_layers()
    res = {"points": len(pts), "samples per point": 2, "adapters": names}
    for arm in ["E4B alone", "grafted E4B"] + names:
        to31, tog, tobr = [], [], []
        for r in pts:
            x, c, n, s0 = seq(ids, r); ctx = x[:, :c]
            if c > CAP - OPD_GEN:
                ctx = torch.cat([ctx[:, :HEAD], ctx[:, -(CAP - OPD_GEN - HEAD):]], 1)
            c = ctx.shape[1]; student(arm)
            out = mS.generate(input_ids=ctx, attention_mask=torch.ones_like(ctx), do_sample=True, temperature=1.0, top_k=0, top_p=1.0, max_new_tokens=OPD_GEN,
                              num_return_sequences=2, eos_token_id=eos, pad_token_id=tok.pad_token_id, use_cache=True); A.g = []
            y = out[:, c:]; m, fin = ends_of(y, eos)
            for i in range(y.shape[0]):
                L = int(m[i].sum()); s_i = torch.cat([ctx, y[i:i + 1, :L]], 1); student(arm)
                ls = Fn.log_softmax(mS(input_ids=s_i, logits_to_keep=L + 1, use_cache=False).logits[0, :-1].float(), -1); A.g = []
                lt = Fn.log_softmax(mT(input_ids=s_i, logits_to_keep=L + 1, use_cache=False).logits[0, :-1].float(), -1)
                student("grafted E4B"); lg = Fn.log_softmax(mS(input_ids=s_i, logits_to_keep=L + 1, use_cache=False).logits[0, :-1].float(), -1); A.g = []
                to31.append(float((ls.exp() * (ls - lt)).sum(-1).mean())); tog.append(float((ls.exp() * (ls - lg)).sum(-1).mean()))
                if br is not None:                                       # the teacher D was trained toward: E4B (no graft, no LoRA) with the translator
                    student("E4B alone"); lb = teacher_lp(mT, mS, gT, s_i, c, L, br); tobr.append(float((ls.exp() * (ls - lb)).sum(-1).mean())); del lb
                del ls, lt, lg
        res[arm] = {"reverse KL to the 31B on own samples": round(float(np.mean(to31)), 4), "reverse KL to the grafted E4B on own samples": round(float(np.mean(tog)), 4)}
        if tobr:
            res[arm]["reverse KL to the bridged E4B on own samples"] = round(float(np.mean(tobr)), 4)
        log(arm, json.dumps(res[arm]))
    A.on = False
    json.dump(res, open(OD / f"opd_fidelity{OTAG}.json", "w"), indent=1); log(json.dumps(res)); print("FIDELITY_DONE", flush=True)


# ---- addendum 161: the behaviour loop (E4B -> E2B on the Mac first; user 11-06: teach where the small model is unsure and the large one knows) ----
BEH = os.environ.get("ON_BEH_ARM", "gap"); BEH_TSURE = float(os.environ.get("ON_BEH_TSURE", "0.5")); BEH_SUNSURE = float(os.environ.get("ON_BEH_SUNSURE", "0.5"))
BEH_RANK = int(os.environ.get("ON_BEH_RANK", "5")); NCHK = 4 if SMALL else int(os.environ.get("ON_NCHK", "30"))


@torch.no_grad()
def teacher_both(mT, mS, gT, s_i, c, L, br):
    """the teacher's own log-probabilities over the L sampled tokens of s_i and, with br = (T, P), the bridged student's (the translator writing
    the teacher's states over [s0, end)), from one teacher pass. The caller turns the student's own change off."""
    if br is None:
        return Fn.log_softmax(mT(input_ids=s_i, logits_to_keep=L + 1, use_cache=False).logits[0, :-1].float(), -1), None
    T, P = br; n_ = s_i.shape[1]; s0 = max(1, c - SPAN); end = n_ if REPLY else c
    gT.sl = slice(s0, end); gT.out = {}; ld = Fn.log_softmax(mT(input_ids=s_i, logits_to_keep=L + 1, use_cache=False).logits[0, :-1].float(), -1); gT.sl = None
    src = torch.cat([gT.out[l] for l in X.SRC[TT]], -1); gT.out = {}; write(P, T, src, s0, end, n_)
    lb = Fn.log_softmax(mS(input_ids=s_i, logits_to_keep=L + 1, use_cache=False).logits[0, :-1].float(), -1); P.slot.clear(); return ld, lb


@torch.no_grad()
def gap_mask(lt, ls):
    """addendum 161: the 'teach here' positions of a sample: the teacher is sure (top probability >= BEH_TSURE); the student would choose
    differently or is unsure (its top probability < BEH_SUNSURE); and the teacher's choice is among the student's top BEH_RANK (elicitable: a
    behaviour gap, not missing knowledge). Also returns, over teacher-sure positions where the student differs, the elicitable and the missing count."""
    tmax, ta = lt.max(-1); smax, sa = ls.max(-1); sure = tmax.exp() >= BEH_TSURE; differs = sa != ta; unsure = smax.exp() < BEH_SUNSURE
    rank = (ls > ls.gather(-1, ta[:, None])).sum(-1); elic = rank < BEH_RANK
    return sure & (differs | unsure) & elic, int((sure & differs & elic).sum()), int((sure & differs & ~elic).sum())


@torch.no_grad()
def flip_mask(lt, ls, lh):
    """addendum 162, 'bridge-flip': the teacher is sure (top probability >= BEH_TSURE), the student differs or is unsure, and the base student
    with the HINT bridge (the teacher's mid-depth states only, ON_BRIDGE_TR trained on TR_SRC_DEPTHS=0.5,0.65) puts the teacher's choice on top:
    a correction the student's own later layers can carry out from a better upstream state. Returns the mask, the number of teacher-sure,
    student-differs positions the hint flips, and the number of teacher-sure, student-differs positions."""
    tmax, ta = lt.max(-1); smax, sa = ls.max(-1); sure = tmax.exp() >= BEH_TSURE; differs = sa != ta; unsure = smax.exp() < BEH_SUNSURE
    flips = lh.argmax(-1) == ta
    return sure & (differs | unsure) & flips, int((sure & differs & flips).sum()), int((sure & differs).sum())


def behave_check():
    """addendum 161, the translator range check: the R3 translators were trained along the teacher's (E4B's) logged replies; the loop scores the
    student's (E2B's) own samples. On NCHK held-out points, base E2B samples one reply (OPD_GEN tokens); along it and along E4B's logged reply,
    KL(E4B || arm) per token and top-1 agreement with E4B, for E2B alone, bridged (ON_BRIDGE_TR) and the control translator (TR_CONTROL, the
    student's own states). Rule: in range when, on E2B's own samples, the bridge is below the control by >= 0.05 nats with top-1 at least the control's."""
    ids, _, held = data(); mT, mS, gT, gS = load_models(); tok = X.tok_(); eos = eos_of(mS); call, sub = call_ids(tok); mS.eval()
    br = bridge_setup(mS); Tb, P = br; ck = torch.load(OD / f"{os.environ.get('TR_CONTROL', 'all_e2b')}.pt", weights_only=False, map_location=DEV)
    Tc = X.Tr(ck["din"], ck["arch"], {k: ck["state"][k].cpu().numpy() for k in ("mx", "sx", "my", "sy")}).to(DEV); Tc.load_state_dict(ck["state"]); Tc.eval()
    assert list(Tc.sites) == list(Tb.sites)
    pts = [r for r in held if not after_submit(ids, r, call, sub)]; pick = [pts[i] for i in np.random.default_rng(162).permutation(len(pts))[:NCHK]]
    arms = ("E2B alone", "bridged", "control"); res = {w: {k: {"kl": [], "top1": []} for k in arms} for w in ("E2B's own sample", "E4B's logged reply")}
    for j, r in enumerate(pick):
        x, c, n, s0 = seq(ids, r); ctx = x[:, :c]
        if c > CAP - OPD_GEN:
            ctx = torch.cat([ctx[:, :HEAD], ctx[:, -(CAP - OPD_GEN - HEAD):]], 1)
        with torch.no_grad():
            y = mS.generate(input_ids=ctx, attention_mask=torch.ones_like(ctx), do_sample=True, temperature=1.0, top_k=0, top_p=1.0, max_new_tokens=OPD_GEN, eos_token_id=eos, pad_token_id=tok.pad_token_id, use_cache=True)[:, ctx.shape[1]:]
        mm, _ = ends_of(y, eos); Ls = int(mm[0].sum())
        for w, s_i, cc, L in (("E2B's own sample", torch.cat([ctx, y[:, :Ls]], 1), ctx.shape[1], Ls), ("E4B's logged reply", x, c, n)):
            ss0 = max(1, cc - SPAN); end = s_i.shape[1] if REPLY else cc
            with torch.no_grad():
                gT.sl = slice(ss0, end); gT.out = {}; ld = Fn.log_softmax(mT(input_ids=s_i, logits_to_keep=L + 1, use_cache=False).logits[0, :-1].float(), -1); gT.sl = None
                gS.sl = slice(ss0, end); gS.out = {}; la = Fn.log_softmax(mS(input_ids=s_i, logits_to_keep=L + 1, use_cache=False).logits[0, :-1].float(), -1); gS.sl = None
                srcT = torch.cat([gT.out[l] for l in X.SRC[TT]], -1); srcS = torch.cat([gS.out[l] for l in X.SRC[ST]], -1); gT.out = {}; gS.out = {}
                t1 = ld.argmax(-1); pd = ld.exp()
                for k in arms:
                    if k == "E2B alone":
                        lk = la
                    else:
                        write(P, Tb if k == "bridged" else Tc, srcT if k == "bridged" else srcS, ss0, end, s_i.shape[1])
                        lk = Fn.log_softmax(mS(input_ids=s_i, logits_to_keep=L + 1, use_cache=False).logits[0, :-1].float(), -1); P.slot.clear()
                    res[w][k]["kl"].append(float((pd * (ld - lk)).sum(-1).mean())); res[w][k]["top1"].append(float((lk.argmax(-1) == t1).float().mean())); del lk
                del ld, la, pd, srcT, srcS
        if j % 5 == 0:
            log("check", j, "of", len(pick))
            if DEV == "mps":
                torch.mps.empty_cache()
    out = {w: {k: {m_: round(float(np.mean(v[m_])), 4) for m_ in ("kl", "top1")} for k, v in d.items()} for w, d in res.items()}
    own = out["E2B's own sample"]; out["points"] = len(pick); out["sample tokens"] = OPD_GEN
    out["in range (bridge below control by >= 0.05 nats, top-1 >= control's, on E2B's own samples)"] = bool(own["bridged"]["kl"] <= own["control"]["kl"] - 0.05 and own["bridged"]["top1"] >= own["control"]["top1"])
    json.dump(out, open(OD / "behave_check.json", "w"), indent=1); log(json.dumps(out)); print("CHECK_DONE behave", flush=True)


def train_behave():
    """Addendum 161: the behaviour loop. Each step the student with its LoRA samples OPD_K replies at a taught decision point; per sampled token
    one teacher pass gives E4B's own distribution and the bridged student's (R3 translator). Arms (ON_BEH_ARM): all = reverse KL to the bridged
    student on every token (C's recipe at this pair); gap = reverse KL to the bridged student only at the 'teach here' tokens (gap_mask, chosen
    with E4B's own distribution against the student's current one), reverse KL to the base student elsewhere; gapdirect = the same tokens with
    E4B's own distribution as the target; rand = as gap, with the same number of tokens per sample drawn at random (the sham).
    Amendment 162b: alldirect = every token, E4B's own distribution as the target (plain on-policy distillation, A's recipe at this pair);
    randdirect = rand's token count (the gap rule's) at random positions, E4B's own distribution as the target (the sham for gapdirect and flip)."""
    from peft import LoraConfig, get_peft_model
    assert BEH in ("all", "alldirect", "gap", "gapdirect", "rand", "randdirect", "flip"), BEH
    ids, tr, _ = data(); mT, mS, gT, gS = load_models(); tok = X.tok_(); rng = np.random.default_rng(161 + SEEDOFF); torch.manual_seed(161 + SEEDOFF); t0 = time.time(); call, sub = call_ids(tok)
    keep = [r for r in tr if not after_submit(ids, r, call, sub)]; log("training points", len(tr), "post-submit left out", len(tr) - len(keep))
    tm, lay = lora_targets(mS); rk = int(os.environ.get("ON_RANK", "16")); llr = float(os.environ.get("ON_LORA_LR", "3e-4"))
    br = bridge_setup(mS) if BRIDGE_TR else None; assert br is not None, "the behaviour loop needs ON_BRIDGE_TR (gap selection and the range log use it)"
    alpha = int(os.environ.get("ON_ALPHA", str(2 * rk)))   # addendum 164 rank sweep: alpha fixed at 32 for rank 64 (alpha / r scaling)
    torch.manual_seed(161 + SEEDOFF); pm = get_peft_model(mS, LoraConfig(r=rk, lora_alpha=alpha, lora_dropout=0.0, target_modules=tm, bias="none")); params = [p_ for n_, p_ in mS.named_parameters() if "lora_" in n_]   # 11-07: same initialisation in every arm
    log("behaviour loop, arm", BEH, "| LoRA layers", lay, "rank", rk, "lr", llr, "samples", OPD_K, "x up to", OPD_GEN, "tokens, steps", OPD_STEPS, "| teacher sure >=", BEH_TSURE, "student unsure <", BEH_SUNSURE, "elicitable: top", BEH_RANK)
    mS.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); eos = eos_of(mS)
    opt = torch.optim.AdamW(params, lr=llr, weight_decay=0.0); sched = lambda s_: min(1.0, (s_ + 1) / 15) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s_ / OPD_STEPS)))  # noqa: E731
    hist, lens, kd, rng_kl = [], [], [], []; sel_tok = all_tok = n_el = n_mi = n_fl = n_sd = 0
    for step in range(OPD_STEPS):
        for g in opt.param_groups:
            g["lr"] = llr * sched(step)
        ctx = opd_point(ids, keep, rng, step); c = ctx.shape[1]
        pm.eval(); out = opd_sample(pm, ctx, tok, eos); y = out[:, c:]; m, fin = ends_of(y, eos); pm.train(); opt.zero_grad(set_to_none=True); Ltot = max(1, int(m.sum())); kls = []
        for i in range(OPD_K):
            L = int(m[i].sum()); s_i = torch.cat([ctx, y[i:i + 1, :L]], 1)
            with pm.disable_adapter():
                ld, lbr = teacher_both(mT, mS, gT, s_i, c, L, br)
                with torch.no_grad():
                    l0 = Fn.log_softmax(mS(input_ids=s_i, logits_to_keep=L + 1, use_cache=False).logits[0, :-1].float(), -1)
            ls = Fn.log_softmax(mS(input_ids=s_i, logits_to_keep=L + 1, use_cache=False).logits[0, :-1].float(), -1)
            rng_kl.append(float((ld.exp() * (ld - lbr)).sum(-1).mean()))         # KL(E4B || bridged E2B) on the student's own sample: the translator's range
            truth = ld if BEH in ("gapdirect", "flip", "alldirect", "randdirect") else lbr   # amendment 162b: alldirect, randdirect
            if BEH in ("all", "alldirect"):
                dm = torch.ones(L, dtype=torch.bool, device=DEV)
            elif BEH == "flip":                                          # addendum 162: lbr is the student with the hint bridge
                dm, nf, nsd = flip_mask(ld, ls.detach(), lbr); n_fl += nf; n_sd += nsd
            else:
                gm, ne, nm = gap_mask(ld, ls.detach()); n_el += ne; n_mi += nm
                if BEH in ("rand", "randdirect"):
                    dm = torch.zeros(L, dtype=torch.bool, device=DEV); k_ = int(gm.sum())
                    if k_:
                        dm[torch.randperm(L, device=DEV)[:k_]] = True
                else:
                    dm = gm
            tgt = torch.where(dm[:, None], truth, l0); kl = (ls.exp() * (ls - tgt)).sum(-1)
            (kl.sum() / Ltot).backward(); kls.append(float(kl.detach().sum()))
            if dm.any():
                kd.append(float(kl.detach()[dm].mean()))
            sel_tok += int(dm.sum()); all_tok += L; del ld, lbr, l0, ls, truth, tgt, kl
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        hist.append(sum(kls) / Ltot); lens.append(float(m.sum(1).float().mean()))
        if step % 10 == 0 or step == OPD_STEPS - 1:
            log("behave", BEH, "step", step, "loss", round(float(np.mean(hist[-10:])), 4), "| teach-here share", round(sel_tok / max(1, all_tok), 3), "KL there", round(float(np.mean(kd[-20:])), 4) if kd else None,
                "| teacher-sure, student differs: elicitable", n_el, "missing", n_mi, *(("| hint flips", n_fl, "of", n_sd) if BEH == "flip" else ()), "| KL(E4B || bridged E2B) on own samples", round(float(np.mean(rng_kl[-20:])), 4), "| sample length", round(float(np.mean(lens[-10:]))), round(time.time() - t0), "s")
            if DEV == "mps":
                torch.mps.empty_cache()
    assert np.isfinite(np.array(hist[-10:])).all(), "training loss is not finite"
    name = f"lora_beh_{BEH}{os.environ.get('ON_BEH_TAG', '')}"; pm.save_pretrained(str(OD / name))
    json.dump({"arm": BEH, "layers": lay, "rank": rk, "alpha": alpha, "lr": llr, "steps": OPD_STEPS, "samples": OPD_K, "max_new": OPD_GEN, "teacher sure": BEH_TSURE, "student unsure": BEH_SUNSURE, "elicitable rank": BEH_RANK,
               "first10": float(np.mean(hist[:10])), "last": float(np.mean(hist[-10:])), "teach-here share": sel_tok / max(1, all_tok), "elicitable": n_el, "missing": n_mi, "hint flips": n_fl if BEH == "flip" else None, "teacher-sure student-differs": n_sd if BEH == "flip" else None,
               "KL(E4B || bridged E2B) on own samples": float(np.mean(rng_kl)), "post-submit left out": len(tr) - len(keep), "translator": BRIDGE_TR}, open(OD / name / "train.json", "w"))
    print("TRAIN_DONE behave", BEH, flush=True)


# ---- addendum 163: the resonance arm (user 11-07: "yes, set up the resonance arm after the queue"; then "stop all the other work on 4090 and
# start this arm"; the user's picture, 11-06: "where the smaller model is out of tune, the larger model could fill with its own in-tune resonance,
# and then that is what the gradient fills"). Report recipe (reports/Localized teacher corrections via bridge.md, 'what to stop', last paragraph):
# match only a low-rank, frozen, normalised subspace of the bridge delta at decision positions, always with the output loss, on-policy, faded.
RES_LAYERS = [int(v) for v in os.environ.get("ON_RES_LAYERS", "17,22").split(",")]   # E2B's 50/65% depths (the hint reads E4B's 50/65%)
RES_K = int(os.environ.get("ON_RES_K", "16")); RES_W = float(os.environ.get("ON_RES_W", "0.1")); RES_HOLD = int(os.environ.get("ON_RES_HOLD", "30")); RES_END = int(os.environ.get("ON_RES_END", "150"))
RES_SRC = os.environ.get("ON_RES_SRC", "bridge")   # bridge: the hint bridge's states (E4B's); control: the control hint translator's (E2B's own), the sham
NBASIS = 6 if SMALL else int(os.environ.get("ON_NBASIS", "160"))


class Cap:
    """chosen layers' outputs (after every earlier-registered hook, so after the bridge's writes) over a slice of positions; with grad when .grad"""
    def __init__(s, m, layers):
        L_ = m.model.language_model.layers; s.sl, s.out, s.grad = None, {}, False; s.hs = [L_[l].register_forward_hook(s.make(l)) for l in layers]

    def make(s, l):
        def hook(_m, _i, out):
            if s.sl is not None:
                h = out[0] if isinstance(out, tuple) else out; v = h[0, s.sl].float(); s.out[l] = v if s.grad else v.detach()
            return out
        return hook

    def take(s):
        o = s.out; s.out, s.sl, s.grad = {}, None, False; return o


def control_setup():
    ck = torch.load(OD / f"{os.environ['TR_CONTROL']}.pt", weights_only=False, map_location=DEV)
    want = {"reply": REPLY, "span": SPAN, "all_kv": os.environ.get("TR_ALL_KV"), "src depths": os.environ.get("TR_SRC_DEPTHS")}
    got = {k: ck.get(k) for k in want}; assert got == want, f"control {os.environ['TR_CONTROL']} was trained as {got}, the environment says {want}"
    assert ck.get("steps", 0) > 20, f"control {os.environ['TR_CONTROL']} has {ck.get('steps')} steps: a small-test translator"
    Tc = X.Tr(ck["din"], ck["arch"], {k: ck["state"][k].cpu().numpy() for k in ("mx", "sx", "my", "sy")}).to(DEV); Tc.load_state_dict(ck["state"]); Tc.eval(); return Tc


@torch.no_grad()
def res_passes(mT, mS, gT, cap, capS, s_i, c, L, br, Tc):
    """along s_i (its last L tokens scored): E4B's log-probabilities; base E2B's (adapter off: the caller) log-probabilities and RES_LAYERS states;
    the hint-bridged E2B's (br, E4B's states written over [s0, end)) log-probabilities and states; with want_ctl, the control translator's states
    (E2B's own source states). States at the positions that predict the L tokens."""
    T, P = br; n_ = s_i.shape[1]; s0 = max(1, c - SPAN); end = n_ if REPLY else c; sl = slice(n_ - L - 1, n_ - 1); want_ctl = Tc is not None
    gT.sl = slice(s0, end); gT.out = {}; ld = Fn.log_softmax(mT(input_ids=s_i, logits_to_keep=L + 1, use_cache=False).logits[0, :-1].float(), -1); gT.sl = None
    srcT = torch.cat([gT.out[l] for l in X.SRC[TT]], -1); gT.out = {}
    cap.sl = sl
    if want_ctl:
        capS.sl = slice(s0, end)
    l0 = Fn.log_softmax(mS(input_ids=s_i, logits_to_keep=L + 1, use_cache=False).logits[0, :-1].float(), -1); h0 = cap.take()
    srcS = None
    if want_ctl:
        oS = capS.take(); srcS = torch.cat([oS[l] for l in X.SRC[ST]], -1); del oS
    write(P, T, srcT, s0, end, n_); cap.sl = sl; lh = Fn.log_softmax(mS(input_ids=s_i, logits_to_keep=L + 1, use_cache=False).logits[0, :-1].float(), -1); hb = cap.take(); P.slot.clear()
    hc = None
    if want_ctl:
        write(P, Tc, srcS, s0, end, n_); cap.sl = sl; mS(input_ids=s_i, logits_to_keep=1, use_cache=False); hc = cap.take(); P.slot.clear()
    return ld, l0, h0, lh, hb, hc


def res_basis():
    """addendum 163, step 1 (no training): along E4B's logged replies at NBASIS training points, at the flip positions (E4B sure, base E2B differs
    or unsure, the hint bridge puts E4B's choice on top), the deltas bridged - base and control - base at RES_LAYERS. Per source and layer: the
    top RES_K directions (SVD, uncentred), their share of the delta's energy, and the ridge bound (held out by point): how much of the delta's
    RES_K coordinates a linear map of base E2B's own state at that layer predicts, an upper bound on what a LoRA can learn from E2B's own states."""
    ids, tr, _ = data(); mT, mS, gT, gS = load_models(); tok = X.tok_(); call, sub = call_ids(tok); mS.eval()
    br = bridge_setup(mS); Tc = control_setup(); assert list(Tc.sites) == list(br[0].sites); cap = Cap(mS, RES_LAYERS); capS = Cap(mS, X.SRC[ST]); t0 = time.time()
    pts = [r for r in tr if not after_submit(ids, r, call, sub)]; pick = [pts[i] for i in np.random.default_rng(163).permutation(len(pts))[:NBASIS]]
    rows = {src: {l: [] for l in RES_LAYERS} for src in ("bridge", "control")}; base = {l: [] for l in RES_LAYERS}; grp = []; ntok = nsel = 0
    for j, r in enumerate(pick):
        x, c, n, s0 = seq(ids, r)
        ld, l0, h0, lh, hb, hc = res_passes(mT, mS, gT, cap, capS, x, c, n, br, Tc)
        dm, _, _ = flip_mask(ld, l0, lh); ntok += n; nsel += int(dm.sum())
        if dm.any():
            for l in RES_LAYERS:
                rows["bridge"][l].append((hb[l] - h0[l])[dm].cpu()); rows["control"][l].append((hc[l] - h0[l])[dm].cpu()); base[l].append(h0[l][dm].cpu())
            grp += [j] * int(dm.sum())
        del ld, l0, h0, lh, hb, hc
        if j % 20 == 0:
            log("basis", j, "of", len(pick), "flip positions", nsel, "of", ntok, round(time.time() - t0), "s")
    assert nsel >= 4 * RES_K, f"only {nsel} flip positions: too few for a {RES_K}-dim basis"
    grp = np.array(grp); held = np.isin(grp, np.unique(grp)[::5])
    if held.all() or not held.any():
        held = np.arange(len(grp)) % 5 == 0
    out = {"points": len(pick), "reply tokens": ntok, "flip positions": nsel, "layers": RES_LAYERS, "k": RES_K, "per source and layer": {}}; save = {}
    for src in ("bridge", "control"):
        for l in RES_LAYERS:
            Dl = torch.cat(rows[src][l]).double(); Hl = torch.cat(base[l]).double()
            U = torch.linalg.svd(Dl[torch.from_numpy(~held)], full_matrices=False).Vh[:RES_K].T     # d x k, from the training rows only
            Y = Dl @ U; e_all = float((Dl ** 2).sum()); e_k = float((Y ** 2).sum())
            Xm = Hl[torch.from_numpy(~held)].mean(0); Xs = Hl[torch.from_numpy(~held)].std(0).clamp_min(1e-6); Z = (Hl - Xm) / Xs; Zt, Yt = Z[torch.from_numpy(~held)], Y[torch.from_numpy(~held)]
            best = -9.0
            for lam in (1e1, 1e2, 1e3, 1e4):
                W = torch.linalg.solve(Zt.T @ Zt + lam * torch.eye(Zt.shape[1], dtype=Zt.dtype), Zt.T @ (Yt - Yt.mean(0)))
                pr = Z[torch.from_numpy(held)] @ W + Yt.mean(0); yh = Y[torch.from_numpy(held)]
                r2 = 1 - float(((yh - pr) ** 2).sum() / ((yh - Yt.mean(0)) ** 2).sum()); best = max(best, r2)
            out["per source and layer"][f"{src} {l}"] = {"rows": int(Dl.shape[0]), "delta norm / state norm": round(float(Dl.norm(dim=1).mean() / Hl.norm(dim=1).mean()), 4),
                                                         "energy in top k": round(e_k / e_all, 4), "ridge bound, held-out R2 (best of 4 strengths)": round(best, 4)}
            save[f"{src} {l}"] = U.float()
    for l in RES_LAYERS:                                                  # overlap of the two subspaces: mean squared cosine of the principal angles
        out["per source and layer"][f"overlap {l}"] = round(float((torch.linalg.svdvals(save[f"bridge {l}"].T @ save[f"control {l}"]) ** 2).mean()), 4)
    torch.save({"U": save, "info": out}, OD / "resonance_basis.pt"); json.dump(out, open(OD / "resonance_basis.json", "w"), indent=1); log(json.dumps(out)); print("BASIS_DONE", flush=True)


class Leaf:
    """replaces layer l's output, once per armed forward, by a leaf that requires grad (frozen weights: the graph starts there)"""
    def __init__(s, m, l):
        s.on, s.v = False, None; s.h = m.model.language_model.layers[l].register_forward_hook(s.hook)

    def hook(s, _m, _i, out):
        if not s.on:
            return out
        h = out[0] if isinstance(out, tuple) else out; v = h.detach().requires_grad_(True); s.v, s.on = v, False
        return (v,) + tuple(out[1:]) if isinstance(out, tuple) else v


def res_decbasis():
    """addendum 164 (user 11-07, the decision-direction basis): the directions at RES_LAYERS in which moving base E2B's own state raises the
    probability of E4B's choice at the flip positions. A: as res_basis (same points), the flip positions, E4B's choices and the bridge and control
    deltas there; E4B is then freed. B: per point and layer, one forward of base E2B with layer l's output made a leaf, logits only at the flip
    positions, the gradient of the summed log-probability of E4B's choices. E2B's last 20 layers reuse the keys and values of layers 13-14, so a
    state at layer >= 15 reaches only its own position's logits: each row is that position's exact gradient. Basis: top RES_K right singular
    vectors of the unit-normalised gradient rows (training rows). Reported: the bridge's and the control's delta energy in it (and in the PCA
    basis), their cosine with the gradient, the first-order gain of the projected delta, the ridge bound, the overlap with the PCA basis."""
    ids, tr, _ = data(); mT, mS, gT, gS = load_models(); tok = X.tok_(); call, sub = call_ids(tok); mS.eval()
    br = bridge_setup(mS); Tc = control_setup(); assert list(Tc.sites) == list(br[0].sites); cap = Cap(mS, RES_LAYERS); capS = Cap(mS, X.SRC[ST]); t0 = time.time()
    pts = [r for r in tr if not after_submit(ids, r, call, sub)]; pick = [pts[i] for i in np.random.default_rng(163).permutation(len(pts))[:NBASIS]]
    keepA = []
    for j, r in enumerate(pick):
        x, c, n, s0 = seq(ids, r)
        ld, l0, h0, lh, hb, hc = res_passes(mT, mS, gT, cap, capS, x, c, n, br, Tc)
        dm, _, _ = flip_mask(ld, l0, lh)
        if dm.any():
            idx = dm.nonzero()[:, 0]
            keepA.append((j, r, idx.cpu(), ld.argmax(-1)[idx].cpu(), {l: (hb[l] - h0[l])[idx].cpu() for l in RES_LAYERS}, {l: (hc[l] - h0[l])[idx].cpu() for l in RES_LAYERS}, {l: h0[l][idx].cpu() for l in RES_LAYERS}))
        del ld, l0, h0, lh, hb, hc
        if j % 40 == 0:
            log("decision basis A", j, "of", len(pick), round(time.time() - t0), "s")
    import gc
    br[1].remove(); del mT, gT; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    leaves = {l: Leaf(mS, l) for l in RES_LAYERS}; G = {l: [] for l in RES_LAYERS}; grp = []
    for k_, (j, r, idx, ta, db, dc, hh) in enumerate(keepA):
        x, c, n, s0 = seq(ids, r); pos = (c - 1 + idx).to(DEV); tgt = ta.to(DEV)
        for l in RES_LAYERS:
            leaves[l].on = True
            with torch.enable_grad():
                lg = mS(input_ids=x, logits_to_keep=pos, use_cache=False).logits[0].float(); obj = Fn.log_softmax(lg, -1).gather(-1, tgt[:, None]).sum(); obj.backward()
            G[l].append(leaves[l].v.grad[0, pos].float().cpu()); leaves[l].v = None; del lg, obj
        grp += [j] * len(idx)
        if k_ % 40 == 0:
            log("decision basis B", k_, "of", len(keepA), round(time.time() - t0), "s")
    grp = np.array(grp); held = np.isin(grp, np.unique(grp)[::5])
    if held.all() or not held.any():
        held = np.arange(len(grp)) % 5 == 0
    trn = torch.from_numpy(~held); hel = torch.from_numpy(held)
    pca = torch.load(OD / "resonance_basis.pt", weights_only=False)["U"] if (OD / "resonance_basis.pt").exists() else {}
    out = {"points": len(pick), "flip positions": int(len(grp)), "layers": RES_LAYERS, "k": RES_K, "per layer": {}}; save = {}
    for l in RES_LAYERS:
        Gl = torch.cat(G[l]).double(); Db = torch.cat([a[4][l] for a in keepA]).double(); Dc = torch.cat([a[5][l] for a in keepA]).double(); Hl = torch.cat([a[6][l] for a in keepA]).double()
        Gn = Gl / Gl.norm(dim=1, keepdim=True).clamp_min(1e-12); U = torch.linalg.svd(Gn[trn], full_matrices=False).Vh[:RES_K].T; save[f"decision {l}"] = U.float()
        def en(D, B):
            return round(float(((D @ B) ** 2).sum() / (D ** 2).sum()), 4)
        def cosg(D):
            cs = (Gl * D).sum(1) / (Gl.norm(dim=1) * D.norm(dim=1)).clamp_min(1e-12); return round(float(cs.mean()), 4), round(float((cs > 0).double().mean()), 4)
        Y = Db @ U; Z = (Hl - Hl[trn].mean(0)) / Hl[trn].std(0).clamp_min(1e-6); best = -9.0
        for lam in (1e1, 1e2, 1e3, 1e4):
            W = torch.linalg.solve(Z[trn].T @ Z[trn] + lam * torch.eye(Z.shape[1], dtype=Z.dtype), Z[trn].T @ (Y[trn] - Y[trn].mean(0)))
            pr = Z[hel] @ W + Y[trn].mean(0); best = max(best, 1 - float(((Y[hel] - pr) ** 2).sum() / ((Y[hel] - Y[trn].mean(0)) ** 2).sum()))
        d = {"gradient rows": int(Gl.shape[0]), "gradient energy in top k (unit rows)": round(float(((Gn @ U) ** 2).sum() / (Gn ** 2).sum()), 4),
             "bridge delta energy in the decision basis": en(Db, U), "control delta energy in the decision basis": en(Dc, U),
             "cosine(gradient, bridge delta): mean, share > 0": cosg(Db), "cosine(gradient, control delta): mean, share > 0": cosg(Dc),
             "first-order log-prob gain of the projected delta (nats per position): bridge, control": [round(float((Gl * ((Db @ U) @ U.T)).sum(1).mean()), 4), round(float((Gl * ((Dc @ U) @ U.T)).sum(1).mean()), 4)],
             "first-order log-prob gain of the full delta: bridge, control": [round(float((Gl * Db).sum(1).mean()), 4), round(float((Gl * Dc).sum(1).mean()), 4)],
             "ridge bound of the bridge delta in this basis, held-out R2": round(best, 4)}
        if f"bridge {l}" in pca:
            Up = pca[f"bridge {l}"].double(); d["bridge delta energy in the PCA basis"] = en(Db, Up); d["overlap with the PCA basis"] = round(float((torch.linalg.svdvals(U.T @ Up) ** 2).mean()), 4)
        out["per layer"][str(l)] = d
    torch.save({"U": save, "info": out}, OD / "resonance_decbasis.pt"); json.dump(out, open(OD / "resonance_decbasis.json", "w"), indent=1); log(json.dumps(out)); print("DECBASIS_DONE", flush=True)


def train_res():
    """addendum 163, step 2: alldirect (every sampled token, reverse KL to E4B's own distribution) plus, at the flip positions of each on-policy
    sample, the resonance term: in the frozen RES_K-dim subspace U of RES_SRC's delta at each of RES_LAYERS, the student's state (no bridge, LoRA
    on) is pulled to the bridged base student's, || U^T (h - h_src) ||^2 / || U^T (h_base - h_src) ||^2 (1 at the start, 0 when matched), mean over
    the positions and layers; weight RES_W for RES_HOLD steps, then linearly to 0 at RES_END (warm start, then outputs only)."""
    from peft import LoraConfig, get_peft_model
    assert RES_SRC in ("bridge", "control"), RES_SRC
    ids, tr, _ = data(); mT, mS, gT, gS = load_models(); tok = X.tok_(); rng = np.random.default_rng(161 + SEEDOFF); torch.manual_seed(161 + SEEDOFF); t0 = time.time(); call, sub = call_ids(tok)
    keep = [r for r in tr if not after_submit(ids, r, call, sub)]; log("training points", len(tr), "post-submit left out", len(tr) - len(keep))
    br = bridge_setup(mS); Tc = control_setup() if RES_SRC == "control" else None; assert Tc is None or list(Tc.sites) == list(br[0].sites); cap = Cap(mS, RES_LAYERS)
    capS = Cap(mS, X.SRC[ST]) if Tc is not None else None
    DEC = os.environ.get("ON_RES_BASIS", "pca") == "dec"; assert not DEC or RES_SRC == "bridge"   # addendum 164: the decision-direction basis
    bas = torch.load(OD / ("resonance_decbasis.pt" if DEC else "resonance_basis.pt"), weights_only=False); U = {l: bas["U"][f"decision {l}" if DEC else f"{RES_SRC} {l}"].to(DEV) for l in RES_LAYERS}
    assert bas["info"]["k"] == RES_K and bas["info"]["layers"] == RES_LAYERS and (SMALL or bas["info"]["points"] >= 100), f"basis {bas['info']['k']}, {bas['info']['layers']}, {bas['info']['points']} points"
    log("basis:", bas["info"]["points"], "points,", bas["info"]["flip positions"], "flip positions")
    tm, lay = lora_targets(mS); rk = int(os.environ.get("ON_RANK", "16")); llr = float(os.environ.get("ON_LORA_LR", "3e-4"))
    torch.manual_seed(161 + SEEDOFF); pm = get_peft_model(mS, LoraConfig(r=rk, lora_alpha=2 * rk, lora_dropout=0.0, target_modules=tm, bias="none")); params = [p_ for n_, p_ in mS.named_parameters() if "lora_" in n_]
    log("resonance arm, basis", "decision" if DEC else "pca", "source", RES_SRC, "| state layers", RES_LAYERS, "k", RES_K, "weight", RES_W, "held", RES_HOLD, "steps, 0 at step", RES_END, "| LoRA layers", lay, "rank", rk, "lr", llr, "samples", OPD_K, "x up to", OPD_GEN, "tokens, steps", OPD_STEPS)
    mS.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); eos = eos_of(mS)
    opt = torch.optim.AdamW(params, lr=llr, weight_decay=0.0); sched = lambda s_: min(1.0, (s_ + 1) / 15) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s_ / OPD_STEPS)))  # noqa: E731
    wsched = lambda s_: RES_W if SMALL else RES_W * min(1.0, max(0.0, (RES_END - s_) / max(1, RES_END - RES_HOLD)))  # noqa: E731
    hist, hst, lens = [], [], []; sel_tok = all_tok = 0; checked = False
    for step in range(OPD_STEPS):
        for g in opt.param_groups:
            g["lr"] = llr * sched(step)
        w = wsched(step); ctx = opd_point(ids, keep, rng, step); c = ctx.shape[1]
        pm.eval(); out = opd_sample(pm, ctx, tok, eos); y = out[:, c:]; m, fin = ends_of(y, eos); pm.train(); opt.zero_grad(set_to_none=True); Ltot = max(1, int(m.sum())); kls, sts = [], []
        for i in range(OPD_K):
            L = int(m[i].sum()); s_i = torch.cat([ctx, y[i:i + 1, :L]], 1); n_ = s_i.shape[1]
            with pm.disable_adapter():
                if w > 0:
                    ld, l0, h0, lh, hb, hc = res_passes(mT, mS, gT, cap, capS, s_i, c, L, br, Tc); ht = hb if RES_SRC == "bridge" else hc
                else:
                    with torch.no_grad():
                        ld = Fn.log_softmax(mT(input_ids=s_i, logits_to_keep=L + 1, use_cache=False).logits[0, :-1].float(), -1)
            if w > 0:
                cap.sl, cap.grad = slice(n_ - L - 1, n_ - 1), True
            ls = Fn.log_softmax(mS(input_ids=s_i, logits_to_keep=L + 1, use_cache=False).logits[0, :-1].float(), -1); hs_ = cap.take()
            kl = (ls.exp() * (ls - ld)).sum(-1); loss = kl.sum() / Ltot; kls.append(float(kl.detach().sum())); all_tok += L if w > 0 else 0
            if w > 0:
                dm, _, _ = flip_mask(ld, ls.detach(), lh); sel_tok += int(dm.sum())
                if dm.any():
                    rel = []
                    for l in RES_LAYERS:
                        e = (hs_[l][dm] - ht[l][dm]) @ U[l]; ref = (h0[l][dm] - ht[l][dm]) @ U[l]
                        rel.append((e ** 2).sum() / (ref ** 2).sum().clamp_min(1e-6))   # review 11-07: ratio of sums, not mean of per-position ratios
                    st = torch.stack(rel).mean(); loss = loss + w * st / OPD_K; sts.append(float(st.detach()))
                    if not checked:
                        assert st.requires_grad, "the resonance term has no gradient path"; checked = True; log("state term has a gradient path")
                del l0, h0, lh, hb, hc, ht
            loss.backward(); del ld, ls, kl, loss, hs_
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        hist.append(sum(kls) / Ltot); hst.append(float(np.mean(sts)) if sts else float("nan")); lens.append(float(m.sum(1).float().mean()))
        if step % 10 == 0 or step == OPD_STEPS - 1:
            hh = [v for v in hst[-10:] if v == v]
            log("res", RES_SRC, "step", step, "KL to E4B", round(float(np.mean(hist[-10:])), 4), "| state term", round(float(np.mean(hh)), 4) if hh else None, "weight", round(w, 4),
                "| flip share", round(sel_tok / max(1, all_tok), 3), "| sample length", round(float(np.mean(lens[-10:]))), round(time.time() - t0), "s")
    assert np.isfinite(np.array(hist[-10:])).all(), "training loss is not finite"
    name = f"lora_beh_res{'' if RES_SRC == 'bridge' else 'ctl'}{'dec' if DEC else ''}{os.environ.get('ON_BEH_TAG', '')}"; pm.save_pretrained(str(OD / name))
    json.dump({"arm": ("res" if RES_SRC == "bridge" else "resctl") + ("dec" if DEC else ""), "source": RES_SRC, "basis": "decision" if DEC else "pca", "state layers": RES_LAYERS, "k": RES_K, "weight": RES_W, "hold": RES_HOLD, "end": RES_END, "layers": lay, "rank": rk, "lr": llr,
               "steps": OPD_STEPS, "samples": OPD_K, "max_new": OPD_GEN, "first10": float(np.mean(hist[:10])), "last": float(np.mean(hist[-10:])),
               "state term first10": float(np.nanmean(hst[:10])), "state term last10 before fade end": float(np.nanmean(hst[max(0, min(RES_END, len(hst)) - 10):min(RES_END, len(hst))])) if any(v == v for v in hst) else None,
               "flip share (while on)": sel_tok / max(1, all_tok), "post-submit left out": len(tr) - len(keep), "translator": BRIDGE_TR}, open(OD / name / "train.json", "w"))
    print("TRAIN_DONE res", RES_SRC, flush=True)


NMIX = 3 if SMALL else int(os.environ.get("ON_NMIX", "40")); NMIXQ = 3 if SMALL else int(os.environ.get("ON_NMIXQ", "150")); NMIXS = 2 if SMALL else int(os.environ.get("ON_NMIXS", "30"))


@torch.no_grad()
def gate_mix(name):
    """addendum 165, the genericity suite: one translator pair (NAME_e4b bridge, NAME_e2b control) on held-out rows of every kind, unchanged:
    agent (E4B's logged reply), agent along E2B's own sample, prose, code, story, question (training items' held split) and question_fresh (the
    asked test's fresh items, never in any translator's training). Arms: alone, bridge, control, sham (the bridge fed E4B's states shifted by half
    the written span: right distribution, wrong positions). Per kind and arm: cross-entropy to E4B (top 32) and top-1 agreement along the reply;
    harm: KL(E2B alone || arm) where E4B and E2B alone pick the same token; the state change against E2B's own state at the write layer and
    RES_LAYERS (norm ratio); for questions p(true) over A/B/C and the answer mass."""
    kinds = ("agent", "prose", "code", "story", "question", "question_fresh")
    ids, _, held = data(kinds); mT, mS, gT, gS = load_models(); tok = X.tok_(); mS.eval(); eos = eos_of(mS); Ts = {}
    want = {"reply": REPLY, "span": SPAN, "all_kv": os.environ.get("TR_ALL_KV"), "src depths": os.environ.get("TR_SRC_DEPTHS")}
    for src in (TT, ST):
        ck = torch.load(OD / f"{name}_{src}.pt", weights_only=False, map_location=DEV); got = {k: ck.get(k) for k in want}
        assert got == want, f"{name}_{src} was trained as {got}, the environment says {want}"
        T = X.Tr(ck["din"], ck["arch"], {k: ck["state"][k].cpu().numpy() for k in ("mx", "sx", "my", "sy")}).to(DEV); T.load_state_dict(ck["state"]); Ts[src] = T.eval()
    assert list(Ts[TT].sites) == list(Ts[ST].sites); P = X.Patch(mS, Ts[TT].sites); lay = [X.WRITE] + RES_LAYERS; cap = Cap(mS, lay)
    let = {a: sorted({tok.encode(v, add_special_tokens=False)[0] for v in (a, " " + a)}) for a in "ABC"}; rng = np.random.default_rng(165)
    jobs = []
    for k in kinds:
        hk = [r for r in held if r.get("kind", "agent") == k]; m_ = NMIXQ if k == "question_fresh" else NMIX
        jobs += [(k, hk[i]) for i in rng.permutation(len(hk))[:m_]]
        if SMALL and k == "agent":                                       # review 11-07: the smoke reaches peak memory
            jobs.append((k, max(hk, key=lambda q: (min(q["ctx_len"] + min(q["reply_len"], RMAX), CAP), min(q["reply_len"], RMAX)))))
    hk = [r for r in held if r.get("kind", "agent") == "agent"]; jobs += [("agent, E2B's own sample", hk[i]) for i in rng.permutation(len(hk))[:NMIXS]]
    arms = ("alone", "bridge", "control", "sham"); res = {}; t0 = time.time()
    for j, (k, r) in enumerate(jobs):
        x, c, n, s0 = seq(ids, r)
        if k == "agent, E2B's own sample":
            ctx = x[:, :c]
            if c > CAP - OPD_GEN:
                ctx = torch.cat([ctx[:, :HEAD], ctx[:, -(CAP - OPD_GEN - HEAD):]], 1)
            y = mS.generate(input_ids=ctx, attention_mask=torch.ones_like(ctx), do_sample=True, temperature=1.0, top_k=0, top_p=1.0, max_new_tokens=OPD_GEN, eos_token_id=eos, pad_token_id=tok.pad_token_id, use_cache=True)[:, ctx.shape[1]:]
            mm, _ = ends_of(y, eos); n = max(1, int(mm[0].sum())); c = ctx.shape[1]; x = torch.cat([ctx, y[:, :n]], 1); s0 = max(1, c - SPAN)
        L = x.shape[1]; end = L if REPLY else c
        gT.sl = slice(s0, end); gT.out = {}; tl = mT(input_ids=x, logits_to_keep=n + 1, use_cache=False).logits[0, :-1].float(); gT.sl = None
        tv, ti = Fn.log_softmax(tl, -1).topk(X.TOPK, -1); tp = tv.softmax(-1); t1 = ti[:, 0]; del tl
        srcT = torch.cat([gT.out[l] for l in X.SRC[TT]], -1); gT.out = {}
        gS.sl = slice(s0, end); gS.out = {}; cap.sl = slice(s0, end); la = Fn.log_softmax(mS(input_ids=x, logits_to_keep=n + 1, use_cache=False).logits[0, :-1].float(), -1); h_own = cap.take(); gS.sl = None
        srcS = torch.cat([gS.out[l] for l in X.SRC[ST]], -1); gS.out = {}; ag = la.argmax(-1) == t1
        d = res.setdefault(k, {a: {"ce": [], "top1": [], "harm": [], **{f"change {l}": [] for l in lay}, "p(true)": [], "answer mass": []} for a in arms})
        for a in arms:
            if a == "alone":
                lsm = la
            else:
                T = Ts[ST] if a == "control" else Ts[TT]; sv_ = srcS if a == "control" else (torch.roll(srcT, shifts=max(1, srcT.shape[0] // 2), dims=0) if a == "sham" else srcT)
                write(P, T, sv_, s0, end, L); cap.sl = slice(s0, end); lsm = Fn.log_softmax(mS(input_ids=x, logits_to_keep=n + 1, use_cache=False).logits[0, :-1].float(), -1); ha = cap.take(); P.slot.clear()
                for l in lay:
                    d[a][f"change {l}"].append(float(((ha[l] - h_own[l]).norm(dim=-1) / h_own[l].norm(dim=-1).clamp_min(1e-6)).mean()))
                if ag.any():                                             # review 11-07: in chunks (memory)
                    kl_ = torch.cat([(la[i:i + 128].exp() * (la[i:i + 128] - lsm[i:i + 128])).sum(-1) for i in range(0, la.shape[0], 128)])
                    d[a]["harm"].append(float(kl_[ag].mean())); del kl_
                del ha
            d[a]["ce"].append(float(-(tp * lsm.gather(-1, ti)).sum(-1).mean())); d[a]["top1"].append(float((lsm.argmax(-1) == t1).float().mean()))
            if k.startswith("question"):
                pr = lsm[0].exp(); raw = {q: float(sum(pr[t] for t in ts)) for q, ts in let.items()}; ms = sum(raw.values())
                d[a]["p(true)"].append(raw[r["true"]] / ms if ms else 0.0); d[a]["answer mass"].append(ms)
            if a != "alone":
                del lsm
        del la, h_own, srcT, srcS, tv, ti, tp
        if j % 50 == 0:
            log("mix gate", name, j, "of", len(jobs), k, round(time.time() - t0), "s")
    out = {"translators": name, "config": want, "rows": {k: len(v["alone"]["ce"]) for k, v in res.items()},
           "per kind": {k: {a: {m_: (round(float(np.mean(vs)), 4) if vs else None) for m_, vs in v.items()} for a, v in dk.items()} for k, dk in res.items()}}
    json.dump(out, open(OD / f"gate_mix_{name}.json", "w"), indent=1); log(json.dumps(out)); print("MIXGATE_DONE", name, flush=True)


# ---- addendum 170: online learning (user 11-07: "cant this be seen as an online learning problem? ... the smaller model, if it had the ability
# to learn while its running, could inheret the calculation and knowledge from the other model using this telepathy method... but we would need
# to figure out how to do online learning without catastrophic forgetting"; "yes, set up the online learning test on the 4090"). E2B with a
# LoRA meets a stream of ONL_N agent decision points it has never seen, once each, in a fixed order. Per item: first scored (prequential: the
# gate's measures, before learning from it; base E2B on the same item, paired), then learned: the student samples OPD_K replies, the teacher
# (ON_ONL_TEACH: direct = E4B's own distribution; bridge = base E2B bridged by ON_BRIDGE_TR) scores every token, reverse KL, one update.
# ON_ONL_KEEP > 0: + that weight x KL(student || base E2B) on one untaught row (prose / code / story) per item; ON_ONL_REPLAY = k > 0: every k-th
# item one earlier stream item again (a fresh sample, fresh targets). Every ONL_EVAL items, the bridge down: the gate's measures on ONL_HELD
# held points and KL(base || student) on ONL_MIX held untaught rows. Saved as lora_onl_TAG; then gate_lora on the 240 points as every arm.
ONL_N = 4 if SMALL else int(os.environ.get("ON_ONL_N", "600")); ONL_TEACH = os.environ.get("ON_ONL_TEACH", "direct"); ONL_KEEP = float(os.environ.get("ON_ONL_KEEP", "0"))
ONL_REPLAY = int(os.environ.get("ON_ONL_REPLAY", "0")); ONL_EVAL = 2 if SMALL else int(os.environ.get("ON_ONL_EVAL", "50")); ONL_HELD = 3 if SMALL else 60; ONL_MIX = 2 if SMALL else 15   # review 11-07: held curve every 50 items (the main learning curve)
# addendum 189 (user 11-08: "yes, queue D seeds 2-3 and the controls after the gate"): ON_SEED = s > 0 offsets the online run's seeds by 1000 s
# (stream subset and order, replay and keep picks, LoRA init, samples); two control teachers: control = base E2B corrected by the self-corrector
# TR_CONTROL (E2B's own source states, as res_passes' control); blend = log_softmax(w * E4B + (1 - w) * base E2B), w set before training so that
# the blend's pull at the start, KL(base || blend), equals the corrector teacher's, KL(base || bridged E2B), on base E2B's samples at ONL_NCALB
# decision points outside the stream (blend_alpha).
ONL_NCALB = 2 if SMALL else int(os.environ.get("ON_ONL_NCALB", "24"))


@torch.no_grad()
def point_scores(mT, mS, pm, x, c, n, r, call, which):
    """the gate's measures on one decision point for E2B with the LoRA on and/or off (which: subset of ('on', 'off')): cross-entropy to E4B's
    top 32 along E4B's logged reply; for tool actions, whether the no-thinking decision (context, then the tool-call opening) matches E4B's"""
    tlg = mT(input_ids=x, logits_to_keep=n + 1, use_cache=False).logits[0, :-1].float(); tv, ti = Fn.log_softmax(tlg, -1).topk(X.TOPK, -1); tp = tv.softmax(-1); del tlg
    out = {}; dec = r["action"] != "text"
    if dec:
        x2 = torch.cat([x[:, :c], torch.tensor([call], device=DEV)], 1); t2 = int(mT(input_ids=x2, logits_to_keep=1, use_cache=False).logits[0, -1].argmax())
    for w in which:
        ctxm = nullcontext() if w == "on" else pm.disable_adapter()
        with ctxm:
            lg = Fn.log_softmax(mS(input_ids=x, logits_to_keep=n + 1, use_cache=False).logits[0, :-1].float(), -1); ce = float(-(tp * lg.gather(-1, ti)).sum(-1).mean()); del lg
            m_ = int(int(mS(input_ids=x2, logits_to_keep=1, use_cache=False).logits[0, -1].argmax()) == t2) if dec else None
        out[w] = (ce, m_)
    return out


@torch.no_grad()
def mix_kl(mS, pm, ids, rows):
    """KL(base E2B || student) per token on the reply part of untaught rows (prose, code, story): what online learning moved where it was not taught"""
    v = []
    for r in rows:
        x, c, n, s0 = seq(ids, r)
        with pm.disable_adapter():
            lb = Fn.log_softmax(mS(input_ids=x, logits_to_keep=n + 1, use_cache=False).logits[0, :-1].float(), -1)
        ls = Fn.log_softmax(mS(input_ids=x, logits_to_keep=n + 1, use_cache=False).logits[0, :-1].float(), -1); v.append(float((lb.exp() * (lb - ls)).sum(-1).mean())); del lb, ls
    return float(np.mean(v)) if v else None


@torch.no_grad()
def control_lp(mS, ctl, s_i, c, L):
    """addendum 189: base E2B corrected by the self-corrector over the L sampled tokens of s_i: E2B's own source states over [s0, end) in a first
    pass, the correction written in a second (as res_passes' control). The caller turns the student's own change off."""
    Tc, P, g = ctl; n_ = s_i.shape[1]; s0 = max(1, c - SPAN); end = n_ if REPLY else c
    g.sl = slice(s0, end); g.out = {}; mS(input_ids=s_i, logits_to_keep=1, use_cache=False); g.sl = None
    src = torch.cat([g.out[l] for l in X.SRC[ST]], -1); g.out = {}; write(P, Tc, src, s0, end, n_)
    lp = Fn.log_softmax(mS(input_ids=s_i, logits_to_keep=L + 1, use_cache=False).logits[0, :-1].float(), -1); P.slot.clear(); return lp


@torch.no_grad()
def blend_alpha(mT, mS, gT, br, ids, items, tok, eos):
    """addendum 189: the blend teacher's weight w. On base E2B's own samples (OPD_K per point, as learn) at the calibration points: per token
    KL(base || bridged E2B) (the corrector teacher's pull at the start), KL(base || E4B), and KL(base || blend_w) on a 21-point grid of w; w is
    where the blend's mean pull equals the corrector's (linear interpolation; 1.0 if the corrector pulls at least as far as E4B). Runs before the
    LoRA exists (torch's seed is reset at the LoRA's creation, so the stream's samples do not depend on it)."""
    grid = np.linspace(0.0, 1.0, 21); kg = np.zeros(len(grid)); kb = kt = 0.0; nt = 0; torch.manual_seed(189 + SEEDOFF); mS.eval()
    for r in items:
        x, c, n, s0 = seq(ids, r); ctx = x[:, :c]
        if c > CAP - OPD_GEN:
            ctx = torch.cat([ctx[:, :HEAD], ctx[:, -(CAP - OPD_GEN - HEAD):]], 1)
        c = ctx.shape[1]; out = opd_sample(mS, ctx, tok, eos); y = out[:, c:]; m, _ = ends_of(y, eos)
        for k in range(OPD_K):
            L = int(m[k].sum())
            if L < 1:
                continue
            s_i = torch.cat([ctx, y[k:k + 1, :L]], 1); ld, lb = teacher_both(mT, mS, gT, s_i, c, L, br)
            l0 = Fn.log_softmax(mS(input_ids=s_i, logits_to_keep=L + 1, use_cache=False).logits[0, :-1].float(), -1); p0 = l0.exp()
            kb += float((p0 * (l0 - lb)).sum()); kt += float((p0 * (l0 - ld)).sum()); nt += L
            for j, w in enumerate(grid):
                kg[j] += float((p0 * (l0 - Fn.log_softmax(w * ld + (1 - w) * l0, -1))).sum())
            del ld, lb, l0, p0
    assert nt > 0, "no calibration tokens"
    kg, kb, kt = kg / nt, kb / nt, kt / nt; up = np.nonzero(kg >= kb)[0]; sg, sb = np.sqrt(np.maximum(kg, 0)), math.sqrt(max(kb, 0.0))   # 189a: sqrt(KL) is near-linear in w
    if kb >= kg[-1] or len(up) == 0:
        w = 1.0
    else:
        j = int(up[0]); w = float(grid[0]) if j == 0 else float(grid[j - 1] + (sb - sg[j - 1]) / (sg[j] - sg[j - 1]) * (grid[j] - grid[j - 1]))
    info = {"points": len(items), "tokens": nt, "KL(base || E4B)": round(kt, 5), "KL(base || corrector teacher)": round(kb, 5), "weight": round(w, 4),
            "KL(base || blend) on the grid": [round(float(v), 5) for v in kg]}
    log("blend calibration", json.dumps(info)); return w, info


def train_online():
    from peft import LoraConfig, get_peft_model
    assert ONL_TEACH in ("direct", "bridge", "control", "blend", "sft", "gated"); ids, tr, held = data(); mT, mS, gT, gS = load_models(); tok = X.tok_(); call, sub = call_ids(tok); eos = eos_of(mS)
    rng = np.random.default_rng(170 + SEEDOFF); torch.manual_seed(170 + SEEDOFF); t0 = time.time()   # 189: ON_SEED (0 = every earlier online run)
    keep = [r for r in tr if not after_submit(ids, r, call, sub)]; perm = rng.permutation(len(keep)); stream = [keep[i] for i in perm[:ONL_N]]
    calib = [keep[i] for i in perm[ONL_N:ONL_N + ONL_NCALB]] if ONL_TEACH == "blend" else []   # 189: outside the stream
    if SMALL:                                                           # review 11-07: the small pass reaches peak memory (the longest row first)
        stream[0] = max(keep, key=(lambda q: (min(q["ctx_len"] + min(q["reply_len"], RMAX), CAP), min(q["reply_len"], RMAX))) if ONL_TEACH == "sft" else (lambda q: min(q["ctx_len"] + min(q["reply_len"], RMAX), CAP)))   # review 192: sft also takes the longest reply
    hpick = [held[i] for i in np.random.default_rng(11).permutation(len(held))[:ONL_HELD]]   # the first ONL_HELD of the gate's own points
    _, mtr, mheld = data(kinds=("prose", "code", "story")); mpick = [mheld[i] for i in np.random.default_rng(12).permutation(len(mheld))[:ONL_MIX]]
    _, _, qheld = data(kinds=("question", "question_fresh")); qpick = [qheld[i] for i in np.random.default_rng(13).permutation(len(qheld))[:ONL_MIX]]   # review 11-07: untouched by the keep term
    if SMALL:
        mtr = sorted(mtr, key=lambda q: -q["reply_len"])
    br = bridge_setup(mS) if ONL_TEACH in ("bridge", "blend") else None   # 189: blend loads the corrector only to set its weight
    ctl, alpha, binfo = None, None, None
    if ONL_TEACH == "control":                                          # 189: the self-corrector as the teacher (hooks on the base layers, before the LoRA)
        Tc = control_setup(); ctl = (Tc, X.Patch(mS, Tc.sites), Grab(mS, X.SRC[ST])); log("control teacher:", os.environ["TR_CONTROL"])
    if ONL_TEACH == "blend":
        assert len(calib) == ONL_NCALB, f"only {len(calib)} of {ONL_NCALB} calibration points outside the stream"   # 189a
        alpha, binfo = blend_alpha(mT, mS, gT, br, ids, calib, tok, eos)
        if alpha >= 1.0 and SMALL:
            log("small pass: blend weight 1.0 on", len(calib), "points; going on (only the full run's calibration decides)")   # 189a: review
        elif alpha >= 1.0:
            log("BLEND_IS_TEACHER: the corrector teacher pulls at least as far as E4B; the blend would be the plain teacher (setup C)"); sys.exit(3)
    tm, lay = lora_targets(mS); torch.manual_seed(161 + SEEDOFF); pm = get_peft_model(mS, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=tm, bias="none"))
    params = [p_ for n_, p_ in mS.named_parameters() if "lora_" in n_]; mS.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=0.0); sched = lambda s_: min(1.0, (s_ + 1) / 15)  # noqa: E731   # review 11-07: flat after warm-up (online: no late damping)
    tag = f"{ONL_TEACH}{'_keep' if ONL_KEEP > 0 else ''}{'_replay' if ONL_REPLAY > 0 else ''}{os.environ.get('ON_BEH_TAG', '')}"
    log("online: teacher", ONL_TEACH, "| weight", alpha, "| seed offset", SEEDOFF, "| stream", len(stream), "items | keep", ONL_KEEP, "replay every", ONL_REPLAY, "| LoRA layers", lay, "| eval every", ONL_EVAL, "on", len(hpick), "held points and", len(mpick), "untaught rows")
    preq, curve, seen, runs_seen, repos_seen = [], [], [], set(), set(); gfrac = []   # 197: share of tokens where the gated teacher takes E4B's distribution

    def evaluate(i):
        pm.eval(); sc = [point_scores(mT, mS, pm, *seq(ids, r)[:3], r, call, ("on", "off") if i == 0 else ("on",)) for r in hpick]
        dm_ = [z["on"][1] for z in sc if z["on"][1] is not None]
        e = {"item": i, "held ce (on)": round(float(np.mean([z["on"][0] for z in sc])), 4), "held decisions match (on)": round(float(np.mean(dm_)), 4) if dm_ else None,
             "KL from base on untaught prose / code / story": round(mix_kl(mS, pm, ids, mpick), 5), "KL from base on held questions (never kept)": round(mix_kl(mS, pm, ids, qpick), 5), "seconds": round(time.time() - t0)}
        if i == 0:
            e["held ce (base)"] = round(float(np.mean([z["off"][0] for z in sc])), 4); e["held decisions match (base)"] = round(float(np.mean([z["off"][1] for z in sc if z["off"][1] is not None])), 4)
        curve.append(e); log("online eval", json.dumps(e)); pm.train()
        json.dump({"partial": True, "items": len(preq), "prequential": preq, "held curve (bridge down)": curve}, open(OD / f"online_{tag}_partial.json", "w"))   # review 11-07: nothing lost on a late crash

    def learn(r):
        if ONL_TEACH == "sft":                                           # 192: copy E4B's logged reply (cross-entropy), no sampling
            x, c, n, s0 = seq(ids, r); lg = mS(input_ids=x, logits_to_keep=n + 1, use_cache=False).logits[0, :-1].float()
            ce = Fn.cross_entropy(lg, x[0, -n:]); ce.backward(); v = float(ce.detach()); del lg, ce; return v
        x, c, n, s0 = seq(ids, r); ctx = x[:, :c]
        if c > CAP - OPD_GEN:
            ctx = torch.cat([ctx[:, :HEAD], ctx[:, -(CAP - OPD_GEN - HEAD):]], 1)
        c = ctx.shape[1]; pm.eval(); out = opd_sample(pm, ctx, tok, eos); y = out[:, c:]; m, _ = ends_of(y, eos); pm.train(); Ltot = max(1, int(m.sum())); kls = []
        for k in range(OPD_K):
            L = int(m[k].sum()); s_i = torch.cat([ctx, y[k:k + 1, :L]], 1)
            with pm.disable_adapter(), torch.no_grad():
                if ONL_TEACH == "bridge":
                    _, tgt = teacher_both(mT, mS, gT, s_i, c, L, br)
                elif ONL_TEACH == "control":                            # 189
                    tgt = control_lp(mS, ctl, s_i, c, L)
                elif ONL_TEACH == "blend":                              # 189: E4B and base E2B (the adapter is off here)
                    ld = Fn.log_softmax(mT(input_ids=s_i, logits_to_keep=L + 1, use_cache=False).logits[0, :-1].float(), -1)
                    l0 = Fn.log_softmax(mS(input_ids=s_i, logits_to_keep=L + 1, use_cache=False).logits[0, :-1].float(), -1)
                    tgt = Fn.log_softmax(alpha * ld + (1 - alpha) * l0, -1); del ld, l0
                elif ONL_TEACH == "gated":                              # 197: E4B's distribution where base E2B's top token differs from E4B's, base E2B's own elsewhere
                    ld = Fn.log_softmax(mT(input_ids=s_i, logits_to_keep=L + 1, use_cache=False).logits[0, :-1].float(), -1)
                    l0 = Fn.log_softmax(mS(input_ids=s_i, logits_to_keep=L + 1, use_cache=False).logits[0, :-1].float(), -1)
                    tgt = torch.where((ld.argmax(-1) != l0.argmax(-1))[:, None], ld, l0); gfrac.append(float((ld.argmax(-1) != l0.argmax(-1)).float().mean())); del ld, l0
                else:
                    tgt = Fn.log_softmax(mT(input_ids=s_i, logits_to_keep=L + 1, use_cache=False).logits[0, :-1].float(), -1)
            ls = Fn.log_softmax(mS(input_ids=s_i, logits_to_keep=L + 1, use_cache=False).logits[0, :-1].float(), -1); kl = (ls.exp() * (ls - tgt)).sum(-1)
            (kl.sum() / Ltot).backward(); kls.append(float(kl.detach().sum())); del tgt, ls, kl
        return sum(kls) / Ltot

    evaluate(0)
    for i, r in enumerate(stream):
        pm.eval(); x, c, n, s0 = seq(ids, r); z = point_scores(mT, mS, pm, x, c, n, r, call, ("on", "off")); pm.train()
        preq.append((z["on"][0], z["off"][0], z["on"][1], z["off"][1], r.get("run") in runs_seen, r.get("repo") in repos_seen)); runs_seen.add(r.get("run")); repos_seen.add(r.get("repo"))
        for g_ in opt.param_groups:
            g_["lr"] = 3e-4 * sched(i)
        opt.zero_grad(set_to_none=True); lv = learn(r); seen.append(r)
        assert math.isfinite(lv), f"loss is not finite at item {i}"
        if ONL_REPLAY > 0 and i > 0 and i % ONL_REPLAY == 0:
            learn(seen[int(rng.integers(len(seen) - 1))])                   # an earlier item again: a fresh sample, fresh targets
        if ONL_KEEP > 0:
            xm, cm, nm, _ = seq(ids, mtr[int(rng.integers(len(mtr)))])
            with pm.disable_adapter(), torch.no_grad():
                lb = Fn.log_softmax(mS(input_ids=xm, logits_to_keep=nm + 1, use_cache=False).logits[0, :-1].float(), -1)
            lsm = Fn.log_softmax(mS(input_ids=xm, logits_to_keep=nm + 1, use_cache=False).logits[0, :-1].float(), -1); kk = (lsm.exp() * (lsm - lb)).sum(-1).mean(); (ONL_KEEP * kk).backward(); del lb, lsm, kk
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if i % 25 == 0:
            w = np.array([p_[:2] for p_ in preq[-50:]]); log("online item", i, "learn KL", round(lv, 4), "| last 50 items, before learning: ce on", round(float(w[:, 0].mean()), 4), "base", round(float(w[:, 1].mean()), 4), round(time.time() - t0), "s")
        if (i + 1) % ONL_EVAL == 0 or i == len(stream) - 1:
            evaluate(i + 1)
    pq = np.array([(a, b) for a, b, *_ in preq]); fv = np.array([not rs for *_, rs, _ in preq]); dlist = [(a, b) for _, _, a, b, *_ in preq if a is not None]; wins = []
    for a in range(0, len(preq), 100 if not SMALL else 2):
        seg = pq[a:a + (100 if not SMALL else 2)]; fseg = seg[fv[a:a + len(seg)]]
        wins.append({"items": [a, a + len(seg)], "ce student": round(float(seg[:, 0].mean()), 4), "ce base": round(float(seg[:, 1].mean()), 4), "gain over base": round(float((seg[:, 1] - seg[:, 0]).mean()), 4),
                     "first-visit-run items": int(len(fseg)), "gain over base, first-visit-run items": round(float((fseg[:, 1] - fseg[:, 0]).mean()), 4) if len(fseg) else None})
    dd = np.array(dlist) if dlist else np.zeros((0, 2))
    out = {"teacher": ONL_TEACH, "blend weight": alpha, "blend calibration": binfo, "gated: mean per-sample share of positions from E4B (incl. replay)": round(float(np.mean(gfrac)), 4) if gfrac else None, "seed offset": SEEDOFF, "keep": ONL_KEEP, "replay every": ONL_REPLAY, "items": len(preq), "LoRA layers": lay, "prequential windows (before learning each item)": wins,
           "prequential decisions match, first / last half (student, base)": [[round(float(h[:, 0].mean()), 4), round(float(h[:, 1].mean()), 4)] for h in (dd[:len(dd) // 2], dd[len(dd) // 2:]) if len(h)],
           "held curve (bridge down)": curve}
    name = f"lora_onl_{tag}"; pm.save_pretrained(str(OD / name)); json.dump({"arm": f"online {tag}", "teacher": ONL_TEACH, "blend weight": alpha, "seed offset": SEEDOFF, "layers": lay, "rank": 16, "lr": 3e-4, "steps": len(preq), "samples": OPD_K, "max_new": OPD_GEN}, open(OD / name / "train.json", "w"))
    json.dump(out, open(OD / f"online_{tag}.json", "w"), indent=1); log(json.dumps(out)); print("ONLINE_DONE", tag, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "lora":
        {"train": train_lora, "gate": gate_lora}[sys.argv[2]]()
    elif sys.argv[1] == "rung":
        gate_rung(sys.argv[2])
    elif sys.argv[1] == "mixgate":
        gate_mix(sys.argv[2])
    elif sys.argv[1] == "online":
        train_online()
    elif sys.argv[1] == "bgl":
        {"graft": bgl_graft, "lora": bgl_lora}[sys.argv[2]](sys.argv[3])
    elif sys.argv[1] == "behave":
        {"check": behave_check, "train": train_behave, "basis": res_basis, "decbasis": res_decbasis, "res": train_res}[sys.argv[2]]()
    elif sys.argv[1] == "reach":
        gate_reach(sys.argv[2])
    elif sys.argv[1] == "opd":
        {"train": train_opd, "loops": gate_loops, "graft": train_opd_graft, "g2l": opd_graft_to_lora, "fidelity": opd_fidelity}[sys.argv[2]]()
    else:
        {"train": lambda: train(sys.argv[2]), "gate": lambda: gate(sys.argv[2])}[sys.argv[1]]()
