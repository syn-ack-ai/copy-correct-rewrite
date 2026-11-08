"""Addendum 166 (user 11-07: "when we try doing it with Qwen and Gemma the bridge may be better with the translator... can we try the smaller qwen
models with the e4b as the teacher and a translator that uses qwens tokenizer to bridge the gemma 4 one?"): a cross-tokenizer bridge, Gemma 4 E4B
(teacher, Gemma tokenizer, 262k vocabulary) -> Qwen3-1.7B (student, Qwen tokenizer, 151k), on Colab.
- Alignment: both models read the same text. Each Qwen position gets E4B's state from the last Gemma token that ends at or before that Qwen token
  ends (character offsets): E4B has read no further than Qwen has, so nothing leaks from the future.
- Translator (the gall design of addendum 165, sized for Qwen): from E4B's 50/65% states (layers 20, 27) and Qwen's own write-layer state, a gated
  change at Qwen's 35% layer and gated additions at every later layer; heads start at zero (Qwen unchanged). The control reads Qwen's own 50/65%
  states instead. Loss (no shared vocabulary, so the text is the target): the bridged Qwen's next-token cross-entropy on the reply part of each row
  (E4B's own replies on agent rows; the text itself on prose, code, stories; the true letter on questions) + XB_SIZE x the change's size + XB_HARM x
  do-no-harm (where Qwen alone already puts the true next token on top, keep its own top-32 distribution: tail cross-entropy). Noise 0.1 on the
  whitened source, each source depth dropped with probability 0.15.
- Rows: the agent decision points (E4B's contexts and replies, decoded from Gemma ids, context cut to XB_CAP Gemma tokens) and the addendum 165 mix
  (prose, code, stories; decoded), plus the pattern questions rebuilt in a plain shared format (xq_questions.jsonl: "...\\n\\nAnswer:" then " X").
- Suite: held rows of every kind, Qwen alone / bridge / control / sham (bridge fed E4B's states shifted by half the written span): cross-entropy and
  top-1 on the reply tokens, harm (KL from Qwen alone where it already puts the true token on top), state change at the write layer, p(true) and
  answer mass on questions; E4B's own p(true) on the same plain questions for reference.
  XB_TEACHER=... XB_STUDENT=... XB_DIR=... python xbridge.py train bridge|control ; python xbridge.py suite"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn

DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
D = Path(os.environ.get("XB_DIR", ".")); OD = D / "out"; OD.mkdir(parents=True, exist_ok=True); SMALL = os.environ.get("XB_SMALL") == "1"
STEPS = 20 if SMALL else int(os.environ.get("XB_STEPS", "600")); ACC = 2; LR = float(os.environ.get("XB_LR", "3e-4"))
SIZE = float(os.environ.get("XB_SIZE", "0.1")); HARM = float(os.environ.get("XB_HARM", "0.5")); NOISE = float(os.environ.get("XB_NOISE", "0.1")); DROP = float(os.environ.get("XB_DROP", "0.15"))
CAPG = int(os.environ.get("XB_CAP", "3072")); HEAD = 512; RMAX = 1024; SPAN = 1024; TOPK = 32
SBOS = os.environ.get("XB_SBOS", "")   # 186: "<bos>" for a Gemma student (its raw completions need the BOS); empty for Qwen (no BOS)
W = dict((k, float(v)) for k, v in (kv.split(":") for kv in os.environ.get("XB_W", "agent:0.5,prose:0.15,code:0.15,question:0.1,story:0.1").split(",")))
ROUND = 1000 * int(os.environ.get("XB_ROUND", "0"))   # review 11-07 (171): each reconciliation round draws fresh rows (0 = every earlier run)
SEED = int(os.environ.get("XB_SEED", "0"))   # addendum 167a: replicate seeds (0 = every earlier run)
NCAL = 6 if SMALL else 24; NSUITE = 3 if SMALL else int(os.environ.get("XB_NSUITE", "40")); NSUITEQ = 3 if SMALL else int(os.environ.get("XB_NSUITEQ", "150"))


def log(*a):
    print(time.strftime("%H:%M"), *a, flush=True)


def load(path):
    from transformers import AutoModelForCausalLM
    kw = {"attn_implementation": "eager"} if DEV == "mps" else {}
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=DEV, **kw).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def layers(m):
    return m.model.language_model.layers if hasattr(m.model, "language_model") else m.model.layers


def toks():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(os.environ["XB_TEACHER"]), AutoTokenizer.from_pretrained(os.environ["XB_STUDENT"])


def rel(n, f):
    return int(round(f * (n - 1)))


class Grab:
    """chosen layers' outputs (after earlier-registered hooks), all positions, detached, while .on"""
    def __init__(s, m, ls):
        s.on, s.out = False, {}; s.hs = [layers(m)[l].register_forward_hook(s.make(l)) for l in ls]

    def make(s, l):
        def hook(_m, _i, out):
            if s.on:
                s.out[l] = (out[0] if isinstance(out, tuple) else out)[0].detach().float()
            return out
        return hook

    def take(s):
        o = s.out; s.out, s.on = {}, False; return o


class QPatch:
    """Qwen: at the write layer the output becomes slot['p'] (or slot['fn'](h)) where slot['mask']; at the sites slot['d'][k] is added there"""
    def __init__(s, m, wl, sites):
        L = layers(m); s.slot = {}; s.hs = [L[wl].register_forward_hook(s.write)] + [L[l].register_forward_hook(s.adder(k)) for k, l in enumerate(sites)]

    def write(s, _m, _i, out):
        if "mask" not in s.slot:
            return out
        h = out[0] if isinstance(out, tuple) else out
        new = s.slot["fn"](h.float()) if "fn" in s.slot else s.slot["p"]
        hn = torch.where(s.slot["mask"][..., None], new.to(h.dtype), h)
        return (hn,) + tuple(out[1:]) if isinstance(out, tuple) else hn

    def adder(s, k):
        def hook(_m, _i, out):
            if "d" not in s.slot:
                return out
            h = out[0] if isinstance(out, tuple) else out
            hn = h + torch.where(s.slot["mask"][..., None], s.slot["d"][k].to(h.dtype), torch.zeros((), dtype=h.dtype, device=h.device))
            return (hn,) + tuple(out[1:]) if isinstance(out, tuple) else hn
        return hook


class XTr(nn.Module):
    """the gall translator (addendum 165) for a student of width ds: identity by default, a gated change at the write layer, gated additions"""
    def __init__(s, din, ds, nsites, st):
        super().__init__()
        for k, v in st.items():
            s.register_buffer(k, torch.as_tensor(v, dtype=torch.float32))
        s.gate = nn.Linear(din + ds, 1); nn.init.zeros_(s.gate.weight); nn.init.zeros_(s.gate.bias)
        s.mlp = nn.Sequential(nn.Linear(din + ds, 4096), nn.GELU(), nn.Linear(4096, ds)); nn.init.zeros_(s.mlp[2].weight); nn.init.zeros_(s.mlp[2].bias)
        s.trunk = nn.Sequential(nn.Linear(din + ds, 1024), nn.GELU()); s.outs = nn.ModuleList([nn.Linear(1024, ds) for _ in range(nsites)])
        for o_ in s.outs:
            nn.init.zeros_(o_.weight); nn.init.zeros_(o_.bias)

    def gapply(s, x, h, noise=0.0, drop=None):
        z = (x - s.mx) / s.sx
        if drop is not None:
            z = z * drop
        if noise > 0:
            z = z + noise * torch.randn_like(z)
        hw = (h - s.my) / s.sy; zh = torch.cat([z, hw], -1); g = torch.sigmoid(s.gate(zh)); d = g * s.mlp(zh); t = s.trunk(zh); aw = [g * o_(t) for o_ in s.outs]
        return h + s.sy * d, [s.dsy[k] * w_ for k, w_ in enumerate(aw)], d, g, torch.stack([(w_ ** 2).mean() for w_ in aw]).mean()


def rows_all(tg):
    """every row as text: (kind, split, context text, reply text, extra); agent and mix rows decoded from Gemma ids, questions from xq_questions"""
    ids = np.load(D / "s3_ids.npy", mmap_mode="r"); out = []
    for line in open(D / "s3_calls.jsonl"):
        r = json.loads(line); k = r.get("kind", "agent")
        if k.startswith("question"):
            continue
        a, c0, n = r["off"], r["ctx_len"], min(r["reply_len"], RMAX); ctx = np.asarray(ids[a:a + c0]); rep = np.asarray(ids[a + c0:a + c0 + n])
        if len(ctx) + n > CAPG:
            ctx = np.concatenate([ctx[:HEAD], ctx[-(CAPG - n - HEAD):]])
        out.append({"kind": k, "split": r["split"], "ctx": tg.decode(ctx.tolist(), skip_special_tokens=False), "reply": tg.decode(rep.tolist(), skip_special_tokens=False), "action": r.get("action")})
    for line in open(D / "xq_questions.jsonl"):
        out.append(json.loads(line))
    for r in out:                                                         # the teacher gets its BOS in encode(); the shared text has none
        if r["ctx"].startswith("<bos>"):
            r["ctx"] = r["ctx"][len("<bos>"):]
    return out


def encode(tg, tq, r):
    """both tokenizations of the same text and the alignment. Returns Gemma ids (with BOS), Qwen ids, for each Qwen position the index of
    the last Gemma token ending at or before it (BOS when none), the first reply position in Qwen tokens, the written span's start"""
    text = r["ctx"] + r["reply"]; cb = len(r["ctx"])
    eg = tg("<bos>" + text, return_offsets_mapping=True, add_special_tokens=False); gids = eg["input_ids"]
    gend = np.maximum.accumulate(np.array([e for _, e in eg["offset_mapping"]]) - len("<bos>"))
    eq = tq(SBOS + text, return_offsets_mapping=True, add_special_tokens=False); qids = eq["input_ids"]   # 186: the student's BOS (none for Qwen)
    qs = np.array([s_ for s_, _ in eq["offset_mapping"]]) - len(SBOS); qe = np.array([e for _, e in eq["offset_mapping"]]) - len(SBOS)
    j = np.clip(np.searchsorted(gend, qe, side="right") - 1, 0, len(gids) - 1)
    if gids == qids:                                                     # 187a: one tokenizer for both (Gemma to Gemma): the alignment is the identity
        j = np.arange(len(qids))
    rs = np.nonzero(qs >= cb)[0]
    if len(rs) == 0 or rs[0] < 1:
        return None
    rstart = int(rs[0]); s0 = max(1, int(np.argmax(j > 0)), rstart - SPAN)   # review 11-07: never Qwen's first token (its attention sink, no BOS) nor positions aligned only to E4B's BOS
    return (gids, qids, j, rstart, s0) if s0 < len(qids) else None


class Bridge:
    def __init__(s, src):
        tok = os.environ.get("XB_TEACHER_TOK_ONLY") == "1"   # 184 (the 4090, 24 GB): no teacher weights, only its tokenizer and config; for arms that never run E4B
        s.mT, s.mS = (None if tok else load(os.environ["XB_TEACHER"])), load(os.environ["XB_STUDENT"]); s.tg, s.tq = toks(); s.src = src
        if tok:
            from transformers import AutoConfig
            cT = AutoConfig.from_pretrained(os.environ["XB_TEACHER"]); cT = cT.get_text_config() if hasattr(cT, "get_text_config") else cT
        nT, nS = (cT.num_hidden_layers if tok else len(layers(s.mT))), len(layers(s.mS)); s.TL = [rel(nT, f) for f in (0.5, 0.65)]; s.SL = [rel(nS, f) for f in (0.5, 0.65)]
        s.WL = rel(nS, 0.35); s.sites = list(range(s.WL + 1, nS)); s.ds = s.mS.config.get_text_config().hidden_size if hasattr(s.mS.config, "get_text_config") else s.mS.config.hidden_size
        s.gT = None if tok else Grab(s.mT, s.TL); s.gS = Grab(s.mS, sorted(set(s.SL) | {s.WL} | set(s.sites))); s.P = QPatch(s.mS, s.WL, s.sites)
        s.wT = cT.hidden_size if tok else (s.mT.config.get_text_config().hidden_size if hasattr(s.mT.config, "get_text_config") else s.mT.config.hidden_size)
        log("teacher layers", nT, "sources", s.TL, "| student layers", nS, "width", s.ds, "write", s.WL, "sources (control)", s.SL, "| device", DEV)

    @torch.no_grad()
    def look(s, gids, qids, j, rstart, sites=False, teacher=True):
        """E4B's aligned source states (teacher=True), Qwen alone: its source states, write-layer state, (site states), log-probs over the reply"""
        srcT = None
        if teacher:                                                      # review 11-07: the control never needs E4B's pass
            s.gT.on = True; s.mT(input_ids=torch.tensor([gids], device=DEV), logits_to_keep=1, use_cache=False); oT = s.gT.take()
            jt = torch.as_tensor(j, device=DEV); srcT = torch.cat([oT[l][jt] for l in s.TL], -1); del oT
        x = torch.tensor([qids], device=DEV); n = len(qids); s.gS.on = True
        la = Fn.log_softmax(s.mS(input_ids=x, logits_to_keep=n - rstart + 1, use_cache=False).logits[0, :-1].float(), -1); oS = s.gS.take()
        o = {"T": srcT, "S": torch.cat([oS[l] for l in s.SL], -1), "E": oS[s.WL], "la": la, "x": x}
        if sites:
            o["sites"] = {l: oS[l] for l in s.sites}
        return o

    def write(s, T, src, s0, n, h=None, noise=0.0, drop=None):
        m = torch.zeros(1, n, dtype=torch.bool, device=DEV); m[0, s0:] = True; s.P.slot["mask"] = m
        pad = lambda t: torch.cat([t.new_zeros(1, s0, t.shape[-1]), t], 1)  # noqa: E731
        if h is not None:
            new, dl, d, g, aw = T.gapply(src[None, s0:], h[None, s0:], noise, drop); s.P.slot["p"] = pad(new); s.P.slot["d"] = [pad(d_) for d_ in dl]; return d, g, aw

        def fn(hh):
            new, dl, _, _, _ = T.gapply(src[None, s0:], hh[:, s0:]); s.P.slot["d"] = [pad(d_) for d_ in dl]; return pad(new)
        s.P.slot["fn"] = fn; return None


def tail_ce(lsm, si, sv):
    qk = lsm.gather(-1, si); pk = sv.exp(); pt = (1 - pk.sum(-1)).clamp_min(0)
    return -((pk * qk).sum(-1) + pt * torch.log1p(-qk.exp().sum(-1).clamp(max=1 - 1e-6))).mean()


def train(src):
    assert src in ("bridge", "control"); B = Bridge(src); rows = rows_all(B.tg); rng = np.random.default_rng(166 + SEED + ROUND); torch.manual_seed(7 + SEED + ROUND); t0 = time.time()
    tr = {k: [r for r in rows if r["kind"] == k and r["split"] == "train"] for k in W}; kinds = [k for k in W if tr[k]]; pw = np.array([W[k] for k in kinds]); pw /= pw.sum()
    log("rows (train):", {k: len(tr[k]) for k in kinds}, "| source", src)
    enc = lambda r: encode(B.tg, B.tq, r)  # noqa: E731
    cal, Xs, Ys, Sd = [], [], [], {l: [] for l in B.sites}
    longest = max(tr["agent"], key=lambda r: len(r["ctx"]) + len(r["reply"]))
    TINIT, RGRAFT, RBETA, TOUT = os.environ.get("XB_TR_INIT"), os.environ.get("XB_RECON_GRAFT"), float(os.environ.get("XB_RECON_BETA", "0")), os.environ.get("XB_TR_OUT")
    H = None
    if RGRAFT:                                                           # addendum 171: the graft whose reach the translator is pulled toward
        gck = torch.load(OD / f"{RGRAFT}.pt", weights_only=False, map_location=DEV); G = QGraft(B.ds, len(gck["layers"]), gck["width"]).to(DEV); G.load_state_dict(gck["state"])
        for p_ in G.parameters():
            p_.requires_grad_(False)
        assert gck["layers"] == [B.WL] + B.sites; H = GraftHooks(B.mS, gck["layers"], G.eval()); log("reconcile toward graft", RGRAFT, "beta", RBETA)
    while not TINIT and len(cal) < NCAL:                                 # calibration: the spreads (no ridge: gall has none); skipped when fine-tuning
        k = kinds[int(rng.choice(len(kinds), p=pw))]; r = longest if (SMALL and not cal) else tr[k][int(rng.integers(len(tr[k])))]; e = enc(r)
        if e is None:
            continue
        gids, qids, j, rstart, s0 = e; o = B.look(gids, qids, j, rstart, sites=True, teacher=src == "bridge"); cal.append(r)
        Xs.append((o["T"] if src == "bridge" else o["S"])[s0:].cpu()); Ys.append(o["E"][s0:].cpu())
        for l in B.sites:
            Sd[l].append(o["sites"][l][s0:].cpu())
        del o
    if TINIT:                                                            # addendum 171: fine-tune an existing translator (its spreads kept)
        T = load_tr(B, src, name=TINIT)[0]; T.train(); din = T.gate.in_features - B.ds; log("fine-tuning translator", TINIT)
    else:
        Xa, Ya = torch.cat(Xs), torch.cat(Ys)
        st = {"mx": Xa.mean(0), "sx": Xa.std(0) + 1e-3, "my": Ya.mean(0), "sy": Ya.std(0) + 1e-3, "dsy": torch.stack([torch.cat(Sd[l]).std(0) for l in B.sites]) + 1e-3}
        din = Xa.shape[1]; T = XTr(din, B.ds, len(B.sites), st).to(DEV); del Xa, Ya, Xs, Ys, Sd
        log("calibration:", len(cal), "rows; spread ratios max/median: source", round(float(st["sx"].max() / st["sx"].median()), 1), "write layer", round(float(st["sy"].max() / st["sy"].median()), 1), "sites", round(float((st["dsy"].max(1).values / st["dsy"].median(1).values).max()), 1))
    widths = [B.wT if src == "bridge" else B.ds] * 2; assert sum(widths) == din, (widths, din)
    opt = torch.optim.AdamW(T.parameters(), lr=LR, weight_decay=0.0); sched = lambda s_: min(1.0, (s_ + 1) / 30) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s_ / STEPS)))  # noqa: E731
    hist = []; seen = {k: 0 for k in kinds}; rks = []
    for step in range(STEPS):
        for g_ in opt.param_groups:
            g_["lr"] = LR * sched(step)
        opt.zero_grad(set_to_none=True); vals = []
        for a_ in range(ACC):
            while True:
                k = kinds[int(rng.choice(len(kinds), p=pw))]; r = longest if (SMALL and step == 0 and a_ == 0) else tr[k][int(rng.integers(len(tr[k])))]; e = enc(r)
                if e is not None:
                    break
            seen[k] += 1; gids, qids, j, rstart, s0 = e; n = len(qids); o = B.look(gids, qids, j, rstart, teacher=src == "bridge"); src_ = o["T"] if src == "bridge" else o["S"]
            kb = torch.rand(2) >= DROP
            if not kb.any():
                kb[int(torch.randint(2, (1,)))] = True
            drop = torch.cat([torch.full((w_,), float(b_)) for w_, b_ in zip(widths, kb.tolist())]).to(DEV)
            clean = (H is not None or CLEAN) and k in ("agent", "question") and rstart - 1 >= s0   # 175: XB_CLEAN=1 = the beta-0 session's clean rows, no graft
            rec = H is not None and clean            # review 11-07: pull only on the graft's own kinds, with the clean bridge it was fitted to
            d, g, aw = B.write(T, src_, s0, n, h=o["E"], noise=0.0 if clean else NOISE, drop=None if clean else drop); lall = None
            if not rec:
                lsm = Fn.log_softmax(B.mS(input_ids=o["x"], logits_to_keep=n - rstart + 1, use_cache=False).logits[0, :-1].float(), -1); B.P.slot.clear(); rk = None
            else:                                                        # addendum 171: the bridged distribution over the written span, pulled toward the grafted Qwen's
                lall = Fn.log_softmax(B.mS(input_ids=o["x"], logits_to_keep=n - s0, use_cache=False).logits[0].float(), -1); B.P.slot.clear()   # positions s0..n-1, as the graft trains
                with torch.no_grad():
                    H.on = True; lgr = Fn.log_softmax(B.mS(input_ids=o["x"], logits_to_keep=n - s0, use_cache=False).logits[0].float(), -1); H.on = False
                rk = (lall.exp() * (lall - lgr)).sum(-1).mean(); lsm = lall[rstart - 1 - s0:n - 1 - s0]; del lgr; rks.append(float(rk.detach()))
            y = torch.tensor(qids[rstart:], device=DEV); ce = -lsm.gather(-1, y[:, None])[:, 0].mean()
            loss = ce + SIZE * ((d ** 2).mean() + aw) + (RBETA * rk if rk is not None else 0.0); hm = torch.zeros((), device=DEV)
            ag = o["la"].argmax(-1) == y                                 # Qwen alone already right: keep its own distribution there
            if HARM > 0 and ag.any():
                sv, si = o["la"][ag].topk(TOPK, -1); hm = tail_ce(lsm[ag], si, sv); loss = loss + HARM * hm
            (loss / ACC).backward(); vals.append((float(ce.detach()), float(-o["la"].gather(-1, y[:, None])[:, 0].mean()), float(((d ** 2).mean() + aw).detach()), float(hm.detach()), float(g.mean().detach()), float(rk.detach()) if rk is not None else 0.0))
            del o, lsm, d, g, aw, loss, lall
        torch.nn.utils.clip_grad_norm_(T.parameters(), 1.0); opt.step(); hist.append(np.mean(vals, 0))
        if step % 25 == 0 or step == STEPS - 1:
            h_ = np.mean(hist[-25:], 0); log(src, "step", step, "cross-entropy (bridged)", round(float(h_[0]), 4), "(alone", round(float(h_[1]), 4), ") change size", round(float(h_[2]), 4), "harm", round(float(h_[3]), 4), "gate", round(float(h_[4]), 4), *(("| KL to the graft (agent and question rows)", round(float(np.mean(rks[-50:])), 4) if rks else None) if H is not None else ()), "| seen", seen, round(time.time() - t0), "s")
    assert np.isfinite(np.array(hist[-25:])).all(), "loss is not finite"
    torch.save({"src": src, "din": din, "ds": B.ds, "nsites": len(B.sites), "state": T.state_dict(), "TL": B.TL, "SL": B.SL, "WL": B.WL, "steps": STEPS, "seen": seen, "last": [float(v) for v in np.mean(hist[-25:], 0)],
                "size": SIZE, "harm": HARM, "noise": NOISE, "drop": DROP, "teacher": os.environ["XB_TEACHER"], "student": os.environ["XB_STUDENT"], "init": TINIT, "reconcile graft": RGRAFT, "beta": RBETA}, OD / f"{TOUT or 'xb_' + src}.pt")
    print("TRAIN_DONE", src, flush=True)


def load_tr(B, src, name=None):
    ck = torch.load(OD / f"{name or 'xb_' + src}.pt", weights_only=False, map_location=DEV); assert (ck["TL"], ck["SL"], ck["WL"]) == (B.TL, B.SL, B.WL), "layer mismatch"
    st = {k: ck["state"][k] for k in ("mx", "sx", "my", "sy", "dsy")}; T = XTr(ck["din"], ck["ds"], ck["nsites"], st).to(DEV); T.load_state_dict(ck["state"]); return T.eval(), ck["steps"]


@torch.no_grad()
def suite():
    B = Bridge("bridge"); Tb, sb = load_tr(B, "bridge"); Tc, sc = load_tr(B, "control"); rows = rows_all(B.tg); rng = np.random.default_rng(165); t0 = time.time()
    lq = {a: sorted({B.tq.encode(v, add_special_tokens=False)[0] for v in (a, " " + a)}) for a in "ABC"}; lg_ = {a: sorted({B.tg.encode(v, add_special_tokens=False)[0] for v in (a, " " + a)}) for a in "ABC"}
    cap = Grab(B.mS, [B.WL, B.sites[-1]]); jobs = []
    for k in ("agent", "prose", "code", "story", "question", "question_fresh"):
        hk = [r for r in rows if r["kind"] == k and r["split"] == "held"]; jobs += [(k, hk[i]) for i in rng.permutation(len(hk))[:NSUITEQ if k == "question_fresh" else NSUITE]]
    arms = ("alone", "bridge", "control", "sham"); res = {}
    for jx, (k, r) in enumerate(jobs):
        e = encode(B.tg, B.tq, r)
        if e is None:
            continue
        gids, qids, j, rstart, s0 = e; n = len(qids); o = B.look(gids, qids, j, rstart, sites=True); y = torch.tensor(qids[rstart:], device=DEV); la = o["la"]; ag = la.argmax(-1) == y
        h_own = {B.WL: o["E"], B.sites[-1]: o["sites"][B.sites[-1]]}; o.pop("sites")   # review 11-07: no extra pass
        d = res.setdefault(k, {a: {"ce": [], "top1": [], "harm": [], "change write": [], "change last": [], "p(true)": [], "answer mass": []} for a in arms + ("E4B",)})
        for a in arms:
            if a == "alone":
                lsm = la
            else:
                T = Tc if a == "control" else Tb; sv_ = o["S"] if a == "control" else o["T"]
                if a == "sham":
                    w_ = sv_[s0:]; sv_ = torch.cat([sv_[:s0], torch.roll(w_, shifts=max(1, w_.shape[0] // 2), dims=0)], 0)
                B.write(T, sv_, s0, n); cap.on = True; lsm = Fn.log_softmax(B.mS(input_ids=o["x"], logits_to_keep=n - rstart + 1, use_cache=False).logits[0, :-1].float(), -1); ha = cap.take(); B.P.slot.clear()
                for nm, l in (("change write", B.WL), ("change last", B.sites[-1])):
                    d[a][nm].append(float(((ha[l][s0:] - h_own[l][s0:]).norm(dim=-1) / h_own[l][s0:].norm(dim=-1).clamp_min(1e-6)).mean()))
                if ag.any():
                    d[a]["harm"].append(float((la[ag].exp() * (la[ag] - lsm[ag])).sum(-1).mean()))
            d[a]["ce"].append(float(-lsm.gather(-1, y[:, None])[:, 0].mean())); d[a]["top1"].append(float((lsm.argmax(-1) == y).float().mean()))
            if k.startswith("question"):
                pr = lsm[0].exp(); raw = {q: float(sum(pr[t] for t in ts)) for q, ts in lq.items()}; ms = sum(raw.values())
                d[a]["p(true)"].append(raw[r["true"]] / ms if ms else 0.0); d[a]["answer mass"].append(ms)
        if k.startswith("question"):                                     # E4B itself on the same plain question (its answer position)
            ctxg = B.tg("<bos>" + r["ctx"], add_special_tokens=False)["input_ids"]
            pr = Fn.softmax(B.mT(input_ids=torch.tensor([ctxg], device=DEV), logits_to_keep=1, use_cache=False).logits[0, -1].float(), -1)
            raw = {q: float(sum(pr[t] for t in ts)) for q, ts in lg_.items()}; ms = sum(raw.values()); d["E4B"]["p(true)"].append(raw[r["true"]] / ms if ms else 0.0); d["E4B"]["answer mass"].append(ms)
        del o, la
        if jx % 50 == 0:
            log("suite", jx, "of", len(jobs), k, round(time.time() - t0), "s")
    out = {"teacher": os.environ["XB_TEACHER"], "student": os.environ["XB_STUDENT"], "translator steps": [sb, sc], "rows": {k: len(v["alone"]["ce"]) for k, v in res.items()},
           "per kind": {k: {a: {m_: (round(float(np.mean(vs)), 4) if vs else None) for m_, vs in v.items()} for a, v in dk.items()} for k, dk in res.items()}}
    json.dump(out, open(OD / "xb_suite.json", "w"), indent=1); log(json.dumps(out)); print("SUITE_DONE", flush=True)


# ---- addendum 168 (user 11-07: "yes, set up step 2 after the sweep"): does the bridge give a better Qwen LoRA than cross-tokenizer distillation?
SELF_LAM = float(os.environ.get("XB_SELF_LAM", "1.0"))
CLEAN = os.environ.get("XB_CLEAN") == "1"   # 175: agent and question rows without noise or depth dropout (as in the reconciliation sessions), no graft needed   # 174q: the KL-to-self weight of the seqself arm
LSTEPS = 4 if SMALL else int(os.environ.get("XB_LSTEPS", "300")); LW = {"agent": 0.7, "question": 0.3}
LRANK = int(os.environ.get("XB_LRANK", "16")); LLAYERS = os.environ.get("XB_LLAYERS", "graft9"); LTAG = os.environ.get("XB_LTAG", "")   # dose-matched baselines (user 11-07): rank 64 at the span layers
ULD_W = float(os.environ.get("XB_ULD_W", "1.0")); ULD_K = int(os.environ.get("XB_ULD_K", "4096"))   # addendum 168b: ULD weight, sorted probabilities kept


def lora_layers(n):
    """the graft layers of the E4B/E2B work (4, 6, ... 28 of 35) at the same relative depth"""
    return sorted({round(l * n / 35) for l in (4, 6, 8, 10, 12, 16, 20, 24, 28)})


@torch.no_grad()
def e4b_letter(B, r):
    """E4B's own answer to a plain question (its top letter): the distillation target, right or wrong"""
    g = Fn.softmax(B.mT(input_ids=torch.tensor([B.tg("<bos>" + r["ctx"], add_special_tokens=False)["input_ids"]], device=DEV), logits_to_keep=1, use_cache=False).logits[0, -1].float(), -1)
    return max("ABC", key=lambda a: float(sum(g[t] for t in B.lg[a])))


def train_lora(arm):
    """seq: cross-entropy on E4B's text (agent replies; E4B's own answer letter on the training questions) = sequence-level distillation, the
    cross-tokenizer baseline. uld (addendum 168b): seq + XB_ULD_W x the Wasserstein-1 distance between E4B's and Qwen's sorted next-token
    probabilities (top XB_ULD_K), E4B's taken at the aligned Gemma position (Universal Logit Distillation, Boizard et al. 2024, with our alignment). bridge / control: the same rows, teacher-forced; the target is the bridged (or control-bridged) Qwen's full
    distribution over Qwen's own vocabulary at every reply position (forward KL), computed with the LoRA off. No bridge in the student's pass."""
    from peft import LoraConfig, get_peft_model
    assert arm in ("seq", "seqdense", "uld", "bridge", "control", "seqself"); B = Bridge("bridge"); rows = rows_all(B.tg); rng = np.random.default_rng(168 + SEED); torch.manual_seed(168 + SEED); t0 = time.time()
    B.lg = {a: sorted({B.tg.encode(v, add_special_tokens=False)[0] for v in (a, " " + a)}) for a in "ABC"}
    T = load_tr(B, arm)[0] if arm in ("bridge", "control") else None
    tr = {k: [r for r in rows if r["kind"] == k and r["split"] == "train"] for k in LW}; kinds = list(LW); pw = np.array([LW[k] for k in kinds]); pw /= pw.sum()
    lay = [B.WL] + B.sites if LLAYERS == "span" else lora_layers(len(layers(B.mS))); tm = [f"model.layers.{l}.mlp.{q}_proj" for l in lay for q in ("gate", "up", "down")]
    torch.manual_seed(161 + SEED); pm = get_peft_model(B.mS, LoraConfig(r=LRANK, lora_alpha=2 * LRANK, lora_dropout=0.0, target_modules=tm, bias="none")); params = [p_ for n_, p_ in B.mS.named_parameters() if "lora_" in n_]
    for p_ in params:
        p_.data = p_.data.float()
    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=0.0); sched = lambda s_: min(1.0, (s_ + 1) / 15) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s_ / LSTEPS)))  # noqa: E731
    log("LoRA arm", arm, "| layers", lay, "rank 16 | steps", LSTEPS, "| rows", {k: len(tr[k]) for k in kinds}); hist = []; agree = []
    for step in range(LSTEPS):
        for g_ in opt.param_groups:
            g_["lr"] = 3e-4 * sched(step)
        opt.zero_grad(set_to_none=True); vals = []
        for _ in range(ACC):
            while True:
                k = kinds[int(rng.choice(len(kinds), p=pw))]; r = dict(tr[k][int(rng.integers(len(tr[k])))])
                if SMALL and step == 0 and not vals:                       # review 11-07: the small pass reaches peak memory
                    k = "agent"; r = dict(max(tr["agent"], key=lambda q: len(q["ctx"]) + len(q["reply"])))
                if k == "question":
                    a_ = e4b_letter(B, r); agree.append(a_ == r["true"]); r["reply"] = " " + a_   # E4B's answer, not the key
                e = encode(B.tg, B.tq, r)
                if e is not None:
                    break
            gids, qids, j, rstart, s0 = e; n = len(qids); x = torch.tensor([qids], device=DEV); y = torch.tensor(qids[rstart:], device=DEV)
            if arm == "seqself":                                        # 174q (user 11-07: is the translator a missing piece?): SFT + KL(original Qwen || student)
                with pm.disable_adapter(), torch.no_grad():               # over the written span, positions s0..n-1 as the graft-objective LoRA; no translator
                    lt0 = Fn.log_softmax(B.mS(input_ids=x, logits_to_keep=n - s0, use_cache=False).logits[0].float(), -1)
                lsd = Fn.log_softmax(B.mS(input_ids=x, logits_to_keep=n - s0, use_cache=False).logits[0].float(), -1)
                ce = -lsd[rstart - 1 - s0:n - 1 - s0].gather(-1, y[:, None])[:, 0].mean() if rstart - 1 >= s0 else -Fn.log_softmax(B.mS(input_ids=x, logits_to_keep=n - rstart + 1, use_cache=False).logits[0, :-1].float(), -1).gather(-1, y[:, None])[:, 0].mean()
                loss = ce + SELF_LAM * (lt0.exp() * (lt0 - lsd)).sum(-1).mean(); (loss / ACC).backward(); vals.append(float(loss.detach())); del lsd, lt0; continue
            if arm == "seqdense":                                       # dense SFT: next-token loss over the whole written span (the graft's span), no teacher
                lsd = Fn.log_softmax(B.mS(input_ids=x, logits_to_keep=n - s0, use_cache=False).logits[0, :-1].float(), -1); yd = torch.tensor(qids[s0 + 1:], device=DEV)
                loss = -lsd.gather(-1, yd[:, None])[:, 0].mean(); (loss / ACC).backward(); vals.append(float(loss.detach())); del lsd; continue
            if arm == "uld":                                             # addendum 168b: E4B's distribution at the aligned Gemma position, sorted
                with torch.no_grad():
                    jj = torch.as_tensor(j[rstart - 1:n - 1], device=DEV)
                    pT = Fn.softmax(B.mT(input_ids=torch.tensor([gids], device=DEV), logits_to_keep=jj, use_cache=False).logits[0].float(), -1).topk(ULD_K, -1).values
            if arm in ("bridge", "control"):
                with pm.disable_adapter(), torch.no_grad():
                    o = B.look(gids, qids, j, rstart, teacher=arm == "bridge"); B.write(T, o["T"] if arm == "bridge" else o["S"], s0, n)
                    lt = Fn.log_softmax(B.mS(input_ids=x, logits_to_keep=n - rstart + 1, use_cache=False).logits[0, :-1].float(), -1); B.P.slot.clear(); del o
            ls = Fn.log_softmax(B.mS(input_ids=x, logits_to_keep=n - rstart + 1, use_cache=False).logits[0, :-1].float(), -1)
            if arm in ("seq", "uld"):
                loss = -ls.gather(-1, y[:, None])[:, 0].mean()
                if arm == "uld":                                         # ULD (Boizard et al. 2024): + the Wasserstein-1 distance of the sorted probability vectors
                    loss = loss + ULD_W * (pT - ls.exp().topk(ULD_K, -1).values).abs().sum(-1).mean(); del pT
            else:
                loss = (lt.exp() * (lt - ls)).sum(-1).mean()
            (loss / ACC).backward(); vals.append(float(loss.detach())); del ls
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); hist.append(np.mean(vals))
        if step % 25 == 0 or step == LSTEPS - 1:
            log("lora", arm, "step", step, "loss", round(float(np.mean(hist[-25:])), 4), "| E4B's answers right", round(float(np.mean(agree)), 3) if agree else None, round(time.time() - t0), "s")
    assert np.isfinite(np.array(hist[-10:])).all(), "loss is not finite"
    pm.save_pretrained(str(OD / f"xlora_{arm}{LTAG}")); json.dump({"arm": arm, "layers": lay, "rank": LRANK, "steps": LSTEPS, "first25": float(np.mean(hist[:25])), "last25": float(np.mean(hist[-25:])), "E4B's answers right": float(np.mean(agree)) if agree else None}, open(OD / f"xlora_{arm}{LTAG}" / "train.json", "w"))
    print("LORA_DONE", arm, flush=True)


@torch.no_grad()
def lora_eval(arm):
    """Qwen with the LoRA (or alone: arm 'alone'), no bridge, on the suite's held rows: cross-entropy and top-1 on the reply tokens, p(true) and
    answer mass on questions, KL from Qwen alone on prose / code / story (what the LoRA moved where it was not taught)"""
    from transformers import AutoTokenizer
    tq = AutoTokenizer.from_pretrained(os.environ["XB_STUDENT"]); tg = AutoTokenizer.from_pretrained(os.environ["XB_TEACHER"]); m = load(os.environ["XB_STUDENT"])
    if arm != "alone":
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, str(OD / f"xlora_{arm}")).eval()
    rows = rows_all(tg); rng = np.random.default_rng(165); lq = {a: sorted({tq.encode(v, add_special_tokens=False)[0] for v in (a, " " + a)}) for a in "ABC"}; res = {}
    for k in ("agent", "prose", "code", "story", "question", "question_fresh"):
        hk = [r for r in rows if r["kind"] == k and r["split"] == "held"]
        for i in rng.permutation(len(hk))[:NSUITEQ if k == "question_fresh" else NSUITE]:
            e = encode(tg, tq, hk[i])
            if e is None:
                continue
            _, qids, _, rstart, _ = e; n = len(qids); x = torch.tensor([qids], device=DEV); y = torch.tensor(qids[rstart:], device=DEV)
            lsm = Fn.log_softmax(m(input_ids=x, logits_to_keep=n - rstart + 1, use_cache=False).logits[0, :-1].float(), -1); d = res.setdefault(k, {"ce": [], "top1": [], "p(true)": [], "answer mass": [], "KL from alone": []})
            d["ce"].append(float(-lsm.gather(-1, y[:, None])[:, 0].mean())); d["top1"].append(float((lsm.argmax(-1) == y).float().mean()))
            if arm != "alone" and k in ("prose", "code", "story"):
                with m.disable_adapter():
                    la = Fn.log_softmax(m(input_ids=x, logits_to_keep=n - rstart + 1, use_cache=False).logits[0, :-1].float(), -1)
                d["KL from alone"].append(float((la.exp() * (la - lsm)).sum(-1).mean())); del la
            if k.startswith("question"):
                pr = lsm[0].exp(); raw = {q: float(sum(pr[t] for t in ts)) for q, ts in lq.items()}; ms = sum(raw.values()); d["p(true)"].append(raw[hk[i]["true"]] / ms if ms else 0.0); d["answer mass"].append(ms)
            del lsm
    out = {"arm": arm, "rows": {k: len(v["ce"]) for k, v in res.items()}, "per kind": {k: {m_: (round(float(np.mean(vs)), 4) if vs else None) for m_, vs in v.items()} for k, v in res.items()}}
    json.dump(out, open(OD / f"xlora_eval_{arm}.json", "w"), indent=1); log(json.dumps(out)); print("LEVAL_DONE", arm, flush=True)


# ---- report X1 (reports/Why LoRA loses bridge gains.md; user 11-07: "yes, start with X1 and the emulator graft"): retention accounting on the
# saved addendum 168b artifacts (this seed's translators and its bridge / control LoRAs in XB_DIR/out), no training. Review 11-07 applied: the
# rows the LoRAs saw, split by kind and all of them; question rows scored against E4B's letter (the LoRAs' target) and the true letter; rows the
# translators were trained on flagged; ratios from unrounded per-row values with a paired bootstrap; the lead's concentration against the net lead.
NX1H = 3 if SMALL else int(os.environ.get("XB_X1_HELD", "120"))


def replay_draws(rows, seed0, weights, steps, extra=0):
    """the rows a training run drew (its rng(seed0 + SEED) calls in order: a kind by weight, then a row; extra = draws before training, e.g. the
    translators' calibration rows; no redraws after encode() is None replayed), unique, in order"""
    rng = np.random.default_rng(seed0 + SEED); tr = {k: [r for r in rows if r["kind"] == k and r["split"] == "train"] for k in weights}; kinds = list(weights)
    pw = np.array([weights[k] for k in kinds]); pw /= pw.sum(); out, seen = [], set()
    for _ in range(extra + steps * ACC):
        k = kinds[int(rng.choice(len(kinds), p=pw))]; i = int(rng.integers(len(tr[k])))
        if (k, i) not in seen:
            seen.add((k, i)); out.append(tr[k][i])
    return out


def boot(a, b, n=1000):
    """mean(a) / mean(b) with a paired bootstrap 90% interval over rows"""
    a, b = np.asarray(a), np.asarray(b); rng = np.random.default_rng(0); r = []
    for _ in range(n):
        i = rng.integers(len(a), size=len(a)); d = b[i].mean()
        r.append(a[i].mean() / d if abs(d) > 1e-9 else np.nan)
    return [round(float(a.mean() / b.mean()), 3) if abs(b.mean()) > 1e-9 else None, round(float(np.nanpercentile(r, 5)), 3), round(float(np.nanpercentile(r, 95)), 3)]


@torch.no_grad()
def x1():
    from peft import PeftModel
    B = Bridge("bridge"); Tb = load_tr(B, "bridge")[0]; Tc = load_tr(B, "control")[0]; rows = rows_all(B.tg); t0 = time.time()
    lsteps = json.load(open(OD / "xlora_bridge" / "train.json"))["steps"]; tsteps = torch.load(OD / "xb_bridge.pt", weights_only=False, map_location="cpu")["steps"]
    pm = PeftModel.from_pretrained(B.mS, str(OD / "xlora_bridge"), adapter_name="b"); pm.load_adapter(str(OD / "xlora_control"), adapter_name="c"); pm.eval()
    B.lg = {a: sorted({B.tg.encode(v, add_special_tokens=False)[0] for v in (a, " " + a)}) for a in "ABC"}; lq = {a: sorted({B.tq.encode(v, add_special_tokens=False)[0] for v in (a, " " + a)}) for a in "ABC"}
    seen_lora = replay_draws(rows, 168, LW, lsteps); seen_tr = {id(r) for r in replay_draws(rows, 166, W, tsteps, extra=24)}   # train(): 24 calibration draws first (NCAL)
    sets = {f"train {k} (seen by the LoRAs)": [r for r in seen_lora if r["kind"] == k][:3 if SMALL else None] for k in LW}
    rng = np.random.default_rng(171)
    for k in ("agent", "question", "question_fresh"):
        hk = [r for r in rows if r["kind"] == k and r["split"] == "held"]; sets[f"held {k}"] = [hk[i] for i in rng.permutation(len(hk))[:NX1H if k != "question_fresh" else 2 * NX1H]]
    arms = ("alone", "bridge", "control", "LoRA bridge", "LoRA control"); out = {"seed": SEED, "LoRA steps": lsteps, "translator steps": tsteps, "sets": {}}; tok = {"lead": [], "kl": []}
    for name, rs in sets.items():
        d = {a: {"ce": [], "ce against E4B's letter (seq's target; not the KL arms')": [], "p(true)": []} for a in arms}; d["E4B"] = {"p(true)": []}; tr_seen = []
        for r in rs:
            e = encode(B.tg, B.tq, r)
            if e is None:
                continue
            gids, qids, j, rstart, s0 = e; n = len(qids); y = torch.tensor(qids[rstart:], device=DEV); L = {}; tr_seen.append(id(r) in seen_tr)
            with pm.disable_adapter():
                o = B.look(gids, qids, j, rstart); L["alone"] = o["la"]
                for a, T, src in (("bridge", Tb, o["T"]), ("control", Tc, o["S"])):
                    B.write(T, src, s0, n); L[a] = Fn.log_softmax(B.mS(input_ids=o["x"], logits_to_keep=n - rstart + 1, use_cache=False).logits[0, :-1].float(), -1); B.P.slot.clear()
            for a, ad in (("LoRA bridge", "b"), ("LoRA control", "c")):
                pm.set_adapter(ad); L[a] = Fn.log_softmax(B.mS(input_ids=o["x"], logits_to_keep=n - rstart + 1, use_cache=False).logits[0, :-1].float(), -1)
            ces = {a: -L[a].gather(-1, y[:, None])[:, 0] for a in arms}
            for a in arms:
                d[a]["ce"].append(float(ces[a].mean()))
            if name == "held agent":
                tok["lead"] += (ces["control"] - ces["bridge"]).tolist(); tok["kl"] += (L["bridge"].exp() * (L["bridge"] - L["control"])).sum(-1).tolist()
            if r["kind"].startswith("question"):
                with pm.disable_adapter():
                    ea = e4b_letter(B, r)
                ye = B.tq.encode(" " + ea, add_special_tokens=False)[0]
                for a in arms:
                    pr = L[a][0].exp(); raw = {q: float(sum(pr[t] for t in ts)) for q, ts in lq.items()}; ms = sum(raw.values()); d[a]["p(true)"].append(raw[r["true"]] / ms if ms else 0.0)
                    d[a]["ce against E4B's letter (seq's target; not the KL arms')"].append(float(-L[a][0, ye]))
                g = Fn.softmax(B.mT(input_ids=torch.tensor([B.tg("<bos>" + r["ctx"], add_special_tokens=False)["input_ids"]], device=DEV), logits_to_keep=1, use_cache=False).logits[0, -1].float(), -1)
                raw = {q: float(sum(g[t] for t in ts)) for q, ts in B.lg.items()}; ms = sum(raw.values()); d["E4B"]["p(true)"].append(raw[r["true"]] / ms if ms else 0.0)
            del o, L, ces
        C = {a: np.array(d[a]["ce"]) for a in arms}; live = C["control"] - C["bridge"]; stored = C["LoRA control"] - C["LoRA bridge"]; own = C["alone"] - C["control"]; own_st = C["alone"] - C["LoRA control"]
        res = {"rows": len(C["alone"]), "rows the translators saw": int(sum(tr_seen)),
               "means": {a: {k_: (round(float(np.mean(v)), 4) if v else None) for k_, v in d[a].items()} for a in list(arms) + ["E4B"]},
               "live lead (control - bridge)": round(float(live.mean()), 4), "stored lead (LoRA control - LoRA bridge)": round(float(stored.mean()), 4),
               "retention f = stored / live lead [mean, 5%, 95%]": boot(stored, live), "Qwen's own gain kept (alone - LoRA control) / (alone - control)": boot(own_st, own)}
        if sum(tr_seen) and sum(tr_seen) < len(tr_seen):
            un = ~np.array(tr_seen); res["retention f, rows the translators did not see"] = boot(stored[un], live[un])
        if d["E4B"]["p(true)"]:
            pb = np.array(d["bridge"]["p(true)"]); pe = np.array(d["E4B"]["p(true)"]); res["live bridge - E4B p(true), paired"] = {"mean": round(float((pb - pe).mean()), 4), "se": round(float((pb - pe).std() / math.sqrt(len(pb))), 4)}
        out["sets"][name] = res; log("x1", name, json.dumps(res), round(time.time() - t0), "s")
    if tok["lead"]:
        ld = np.array(tok["lead"]); kl = np.sort(np.array(tok["kl"]))[::-1]; top = np.sort(ld)[::-1][:max(1, len(ld) // 10)]
        out["held agent tokens"] = {"tokens": int(len(ld)), "net lead per token": round(float(ld.mean()), 4), "top 10% of tokens by lead: their summed lead / the net summed lead": round(float(top.sum() / ld.sum()), 3) if abs(ld.sum()) > 1e-9 else None,
                                    "bottom 10% by lead: summed lead / net": round(float(np.sort(ld)[:max(1, len(ld) // 10)].sum() / ld.sum()), 3) if abs(ld.sum()) > 1e-9 else None,
                                    "share of KL(bridged || control) in the top 10% of tokens": round(float(kl[:max(1, len(kl) // 10)].sum() / max(kl.sum(), 1e-9)), 3)}
    json.dump(out, open(OD / f"x1_s{SEED}.json", "w"), indent=1); log(json.dumps(out)); print("X1_DONE", flush=True)


# ---- the emulator graft, record stage (user 11-07: "what if the lora is running at recording while both models are being asked a question, we
# kinda did this by creating a graft ... we would have to figure out how to convert the graft in such a way that it updates the smaller model in
# a way that it preserves what it carried over"; "yes, start with X1 and the emulator graft"). With no E4B at its input, a student form learns
# to reproduce the bridged Qwen (target 'bridge') or the control-bridged Qwen (target 'control'); the bridge form's lead over the control form,
# against the live lead, says how much of E4B's contribution Qwen can compute from its own states. Forms: 'graft' = a gated residual MLP after
# each layer from the write layer to the last, each reading only its own layer's output (one forward pass), output starting at zero; 'lora' =
# a PEFT LoRA (rank XB_GRANK, MLP gate/up/down at the same layers) on the same objective: if it matches the graft, no compile step is needed.
# Loss (review 11-07): forward KL(target || student) at every position of the written span [s0, n) + XB_GSTATE_W x the per-layer state error on
# that span, whitened per dimension by the translator's spreads (sy at the write layer, dsy at the sites) and divided by a fixed per-layer
# normaliser (the alone-vs-target error on calibration rows), mean over layers. Rows: 70% agent / 30% question as step 2 (E4B's letter on
# questions), of which XB_GONP are Qwen's own samples (the current student, 256 tokens) at agent prompts, scored by the bridged target.
GSTEPS = 4 if SMALL else int(os.environ.get("XB_GSTEPS", "1000")); GWIDTH = int(os.environ.get("XB_GWIDTH", "1024")); GLR = float(os.environ.get("XB_GLR", "3e-4"))
GSTATE_W = float(os.environ.get("XB_GSTATE_W", "1.0")); GRANK = int(os.environ.get("XB_GRANK", "64")); GONP = float(os.environ.get("XB_GONP", "0.3")); GNCAL = 3 if SMALL else 16


class Cap:
    """chosen layers' outputs over a slice of positions, after every earlier-registered hook; with grad when .grad (the student's pass)"""
    def __init__(s, m, ls):
        s.sl, s.out, s.grad = None, {}, False; L_ = layers(m); s.hs = [L_[l].register_forward_hook(s.make(l)) for l in ls]

    def make(s, l):
        def hook(_m, _i, out):
            if s.sl is not None:
                h = out[0] if isinstance(out, tuple) else out; v = h[0, s.sl].float(); s.out[l] = v if s.grad else v.detach()
            return out
        return hook

    def take(s):
        o = s.out; s.out, s.sl, s.grad = {}, None, False; return o


class QGraft(nn.Module):
    def __init__(s, ds, n, width=None):
        super().__init__(); w = width or GWIDTH
        s.ln = nn.ModuleList([nn.LayerNorm(ds) for _ in range(n)]); s.up = nn.ModuleList([nn.Linear(ds, w) for _ in range(n)]); s.down = nn.ModuleList([nn.Linear(w, ds) for _ in range(n)])
        s.gate = nn.ModuleList([nn.Linear(ds, 1) for _ in range(n)])
        for k in range(n):
            nn.init.zeros_(s.down[k].weight); nn.init.zeros_(s.down[k].bias); nn.init.zeros_(s.gate[k].weight); nn.init.zeros_(s.gate[k].bias)

    def f(s, k, h):
        x = s.ln[k](h); return torch.sigmoid(s.gate[k](x)) * s.down[k](Fn.gelu(s.up[k](x)))


class GraftHooks:
    """adds graft.f(k, h) to the output of layer ls[k] while .on (every position)"""
    def __init__(s, m, ls, graft):
        s.on, s.g = False, graft; L = layers(m); s.hs = [L[l].register_forward_hook(s.make(k)) for k, l in enumerate(ls)]

    def make(s, k):
        def hook(_m, _i, out):
            if not s.on:
                return out
            h = out[0] if isinstance(out, tuple) else out; hn = h + s.g.f(k, h.float()).to(h.dtype)
            return (hn,) + tuple(out[1:]) if isinstance(out, tuple) else hn
        return hook


def gscale(T, gl):
    """per-layer, per-dimension spreads of Qwen's states from the translator's calibration (sy at the write layer, dsy at each site)"""
    return {gl[0]: T.sy.float()} | {l: T.dsy[k].float() for k, l in enumerate(gl[1:])}


def train_graft(form, tgt):
    assert form in ("graft", "lora") and tgt in ("bridge", "control")
    B = Bridge("bridge"); TB = {"bridge": load_tr(B, "bridge")[0], "control": load_tr(B, "control")[0]}; rows = rows_all(B.tg)
    T = load_tr(B, tgt, name=os.environ["XB_TRNAME"])[0] if os.environ.get("XB_TRNAME") else TB[tgt]   # addendum 171: the target translator by name (a reconciled one)
    rng = np.random.default_rng(172 + SEED + ROUND); torch.manual_seed(172 + SEED + ROUND); t0 = time.time()
    B.lg = {a: sorted({B.tg.encode(v, add_special_tokens=False)[0] for v in (a, " " + a)}) for a in "ABC"}; eos = B.tq.eos_token_id
    tr = {k: [r for r in rows if r["kind"] == k and r["split"] == "train"] for k in LW}; kinds = list(LW); pw = np.array([LW[k] for k in kinds]); pw /= pw.sum()
    gl = [B.WL] + B.sites; sc = gscale(TB["bridge"], gl)                # review 11-07: the same whitening for both arms
    if form == "graft":                                                 # the graft's hooks first, then Cap: Cap sees each layer after the bridge AND the graft
        gck = torch.load(OD / f"{os.environ['XB_GINIT']}.pt", weights_only=False, map_location=DEV) if os.environ.get("XB_GINIT") else None
        G = QGraft(B.ds, len(gl), gck["width"] if gck else None).to(DEV); H = GraftHooks(B.mS, gl, G); params = list(G.parameters()); pm = None
        if gck:                                                          # addendum 171: continue an earlier graft (review 11-07: its own width)
            G.load_state_dict(gck["state"]); log("graft warm start from", os.environ["XB_GINIT"], "(earlier steps", gck.get("steps"), "init", gck.get("init"), ")"); del gck
    else:
        from peft import LoraConfig, get_peft_model
        torch.manual_seed(161 + SEED); pm = get_peft_model(B.mS, LoraConfig(r=GRANK, lora_alpha=2 * GRANK, lora_dropout=0.0, target_modules=[f"model.layers.{l}.mlp.{q}_proj" for l in gl for q in ("gate", "up", "down")], bias="none"))
        params = [p_ for n_, p_ in B.mS.named_parameters() if "lora_" in n_]; H = None
        for p_ in params:
            p_.data = p_.data.float()
    cap = Cap(B.mS, gl)                                                 # registered last: the bridged states (no grad) and the student's (grad)

    class student:                                                      # the student's own forward: graft on, or LoRA on (the bridge is never on)
        def __enter__(s_):
            if H is not None:
                H.on = True
            return s_

        def __exit__(s_, *a):
            if H is not None:
                H.on = False

    class base:                                                         # the frozen base (graft off / LoRA off)
        def __enter__(s_):
            s_.c = pm.disable_adapter() if pm is not None else None
            if s_.c is not None:
                s_.c.__enter__()
            return s_

        def __exit__(s_, *a):
            if s_.c is not None:
                s_.c.__exit__(*a)

    def draw(step, a_):
        """(kind, row index or None, row): a teacher-forced row, or Qwen's own sample (the current student) at an agent prompt"""
        if SMALL and step == 0 and a_ == 0:                             # review 11-07: the longest written span in Qwen tokens (peak memory)
            i = max(range(len(tr["agent"])), key=lambda q: (lambda e_: (e_[0] and len(e_[1]) - e_[4]) or 0)(encode(B.tg, B.tq, tr["agent"][q]) or (None,) * 5)); return "agent", i, dict(tr["agent"][i])
        onp = rng.random() < GONP; k = "agent" if onp else kinds[int(rng.choice(len(kinds), p=pw))]; i = int(rng.integers(len(tr[k]))); r = dict(tr[k][i])   # questions keep the true letter: the KL arms' target is the distribution before it (review 11-07)
        if onp:
            ctx = B.tq(r["ctx"], add_special_tokens=False)["input_ids"][-(CAPG * 3 // 2):]; torch.manual_seed(100003 * step + a_ + 17 * SEED)
            with torch.no_grad(), student():
                y = B.mS.generate(input_ids=torch.tensor([ctx], device=DEV), attention_mask=torch.ones(1, len(ctx), dtype=torch.long, device=DEV), do_sample=True, temperature=1.0, top_k=0, top_p=1.0, max_new_tokens=256, eos_token_id=eos, pad_token_id=eos, use_cache=True)[0, len(ctx):].tolist()
            r = {"kind": "onpolicy", "ctx": B.tq.decode(ctx), "reply": B.tq.decode(y, skip_special_tokens=True)}; return "onpolicy", i, r
        return k, i, r

    def targets(r, e):
        gids, qids, j, rstart, s0 = e; n = len(qids); x = torch.tensor([qids], device=DEV); kp = n - s0   # review 11-07: logits at positions s0..n-1, the state span
        with torch.no_grad(), base():
            o = B.look(gids, qids, j, rstart, sites=True, teacher=tgt == "bridge"); h0 = {l: (o["E"] if l == B.WL else o["sites"][l])[s0:] for l in gl}
            B.write(T, o["T"] if tgt == "bridge" else o["S"], s0, n); cap.sl = slice(s0, n)
            lt = Fn.log_softmax(B.mS(input_ids=x, logits_to_keep=kp, use_cache=False).logits[0].float(), -1); ht = cap.take(); B.P.slot.clear()
        return x, kp, s0, n, h0, ht, lt

    norm = {l: [] for l in gl}; ncal = ci = 0                          # review 11-07: ONE normaliser for both arms (the mean of the bridge's and the
    while ncal < GNCAL and ci < 50 * GNCAL:                             # control's whitened deltas from Qwen alone on the same calibration rows)
        ci += 1; k_ = kinds[int(rng.choice(len(kinds), p=pw))]; r = dict(tr[k_][int(rng.integers(len(tr[k_])))]) if not SMALL else dict(tr["agent"][ci - 1])
        e = encode(B.tg, B.tq, r)
        if e is None:
            continue
        gids, qids, j, rstart, s0 = e; n = len(qids); x = torch.tensor([qids], device=DEV); ncal += 1
        with torch.no_grad(), base():
            o = B.look(gids, qids, j, rstart, sites=True); h0 = {l: (o["E"] if l == B.WL else o["sites"][l])[s0:] for l in gl}; dd = {l: 0.0 for l in gl}
            for tt in ("bridge", "control"):
                B.write(TB[tt], o["T"] if tt == "bridge" else o["S"], s0, n); cap.sl = slice(s0, n); B.mS(input_ids=x, logits_to_keep=1, use_cache=False); ht = cap.take(); B.P.slot.clear()
                for l in gl:
                    dd[l] += 0.5 * float((((h0[l] - ht[l]) / sc[l]) ** 2).sum(-1).mean())
        for l in gl:
            norm[l].append(dd[l])
        del o, h0, ht
    norm = {l: max(float(np.mean(v)), 1e-6) for l, v in norm.items()}
    opt = torch.optim.AdamW(params, lr=GLR, weight_decay=0.0); sched = lambda s_: min(1.0, (s_ + 1) / 30) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s_ / GSTEPS)))  # noqa: E731
    log("student form", form, "toward", tgt, "| layers", gl, "params", sum(p_.numel() for p_ in params), "| steps", GSTEPS, "lr", GLR, "state weight", GSTATE_W, "own samples", GONP, "| normaliser", {l: round(v, 2) for l, v in norm.items()})
    hist = []; seen = []
    for step in range(GSTEPS):
        for g_ in opt.param_groups:
            g_["lr"] = GLR * sched(step)
        opt.zero_grad(set_to_none=True); vals = []
        for a_ in range(ACC):
            while True:
                k, i, r = draw(step, a_); e = encode(B.tg, B.tq, r)
                if e is not None:
                    break
            seen.append((k, i)); x, kp, s0, n, h0, ht, lt = targets(r, e)
            with student():
                cap.sl, cap.grad = slice(s0, n), True; ls = Fn.log_softmax(B.mS(input_ids=x, logits_to_keep=kp, use_cache=False).logits[0].float(), -1); hg = cap.take()
            st = torch.stack([(((hg[l] - ht[l]) / sc[l]) ** 2).sum(-1).mean() / norm[l] for l in gl]).mean()
            kl = (lt.exp() * (lt - ls)).sum(-1).mean(); loss = kl + GSTATE_W * st
            (loss / ACC).backward(); vals.append((float(kl.detach()), float(st.detach()))); del h0, ht, hg, lt, ls, loss
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); hist.append(np.mean(vals, 0))
        if step % 50 == 0 or step == GSTEPS - 1:
            h_ = np.mean(hist[-50:], 0); log(form, tgt, "step", step, "KL to the target", round(float(h_[0]), 4), "state error", round(float(h_[1]), 4), round(time.time() - t0), "s")
    assert np.isfinite(np.array(hist[-10:])).all(), "loss is not finite"
    tag = f"{form}_{tgt}_w{GSTATE_W:g}{os.environ.get('XB_GTAG', '')}"; info = {"form": form, "tgt": tgt, "layers": gl, "width": G.up[0].out_features if form == "graft" else GWIDTH, "init": os.environ.get("XB_GINIT"), "target translator": os.environ.get("XB_TRNAME"), "rank": GRANK, "steps": GSTEPS, "seed": SEED, "state weight": GSTATE_W, "own samples": GONP,
                                                 "seen": seen, "first50": [float(v) for v in np.mean(hist[:50], 0)], "last50": [float(v) for v in np.mean(hist[-50:], 0)]}
    if form == "graft":
        torch.save({**info, "state": G.state_dict(), "ds": B.ds}, OD / f"x{tag}.pt")
    else:
        pm.save_pretrained(str(OD / f"x{tag}")); json.dump(info, open(OD / f"x{tag}" / "train.json", "w"))
    print("GRAFT_DONE", tag, flush=True)


@torch.no_grad()
def graft_eval():
    """every saved student form in XB_DIR/out (x{graft,lora}_{bridge,control}_w{W}): next-token cross-entropy on the reply for Qwen alone, the
    live bridge, the live control, and each form (no E4B, no translator), on X1's held rows and on the agent / question rows each pair trained
    on; per pair: fidelity of each arm (alone - form) / (alone - live target) and retention (control form - bridge form) / (live control - live
    bridge), with paired bootstrap intervals; p(true) on questions"""
    import glob
    B = Bridge("bridge"); Tb = load_tr(B, "bridge")[0]; Tc = load_tr(B, "control")[0]; rows = rows_all(B.tg); t0 = time.time(); gl = [B.WL] + B.sites; F = {}; pm = None
    tr = {k: [r for r in rows if r["kind"] == k and r["split"] == "train"] for k in LW}; seen = {}
    for f in sorted(glob.glob(str(OD / "xgraft_*_w*.pt"))):
        ck = torch.load(f, weights_only=False, map_location=DEV); assert ck["layers"] == gl
        G = QGraft(B.ds, len(gl), ck["width"]).to(DEV); G.load_state_dict(ck["state"]); tag = Path(f).stem[1:]; F[tag] = ("graft", GraftHooks(B.mS, gl, G.eval())); seen[tag] = ck["seen"]
    for d_ in sorted(glob.glob(str(OD / "xlora_*_w*"))):
        from peft import PeftModel
        tag = Path(d_).name[1:]; info = json.load(open(Path(d_) / "train.json")); seen[tag] = info["seen"]
        if pm is None:
            pm = PeftModel.from_pretrained(B.mS, d_, adapter_name=tag)
        else:
            pm.load_adapter(d_, adapter_name=tag)
        F[tag] = ("lora", tag)
    import re
    for d_ in sorted(p_ for p_ in OD.glob("xlora_*") if p_.is_dir() and not re.search(r"_w\d", p_.name)):   # bar 3: step-2 and dose-matched baselines on the same rows
        nm = d_.name[len("xlora_"):]
        if True:
            from peft import PeftModel
            tag = f"step2 {nm} LoRA"
            if pm is None:
                pm = PeftModel.from_pretrained(B.mS, str(d_), adapter_name=tag.replace(" ", "_"))
            else:
                pm.load_adapter(str(d_), adapter_name=tag.replace(" ", "_"))
            F[tag] = ("lora", tag.replace(" ", "_"))
    if pm is not None:
        pm.eval()
    pairs = sorted({t.replace("_bridge_", "_X_").replace("_control_", "_X_") for t in F if not t.startswith("step2")}); log("forms", sorted(F), "pairs", pairs)
    lq = {a: sorted({B.tq.encode(v, add_special_tokens=False)[0] for v in (a, " " + a)}) for a in "ABC"}; rng = np.random.default_rng(171); sets = {}
    for k in ("agent", "question", "question_fresh"):
        hk = [r for r in rows if r["kind"] == k and r["split"] == "held"]; sets[f"held {k}"] = [hk[i] for i in rng.permutation(len(hk))[:NX1H if k != "question_fresh" else 2 * NX1H]]
    sb = next((v for v in seen.values() if v), [])                      # review 11-07: every form drew the same teacher-forced rows; any form's record
    for k in LW:
        u = list(dict.fromkeys(i for kk, i in sb if kk == k)); sets[f"train {k}"] = [tr[k][i] for i in u[:3 if SMALL else NX1H]]
    for k in ("prose", "code", "story"):                                # do-no-harm (review 11-07): untaught kinds, KL from Qwen alone per form
        hk = [r for r in rows if r["kind"] == k and r["split"] == "held"]; sets[f"held {k} (untaught)"] = [hk[i] for i in rng.permutation(len(hk))[:3 if SMALL else 30]]

    class off:
        def __enter__(s_):
            s_.c = pm.disable_adapter() if pm is not None else None
            if s_.c is not None:
                s_.c.__enter__()

        def __exit__(s_, *a):
            if s_.c is not None:
                s_.c.__exit__(*a)
    out = {"seed": SEED, "forms": sorted(F), "sets": {}}
    for name, rs in sets.items():
        C = {a: [] for a in ["alone", "bridge", "control"] + sorted(F)}; P = {a: [] for a in C}; KLa = {a: [] for a in C}
        for r in rs:
            e = encode(B.tg, B.tq, r)
            if e is None:
                continue
            gids, qids, j, rstart, s0 = e; n = len(qids); y = torch.tensor(qids[rstart:], device=DEV); kr = n - rstart + 1; L = {}
            with off():
                o = B.look(gids, qids, j, rstart); L["alone"] = o["la"]
                for a, T, src in (("bridge", Tb, o["T"]), ("control", Tc, o["S"])):
                    B.write(T, src, s0, n); L[a] = Fn.log_softmax(B.mS(input_ids=o["x"], logits_to_keep=kr, use_cache=False).logits[0, :-1].float(), -1); B.P.slot.clear()
                for tag, (fm, h) in F.items():
                    if fm == "graft":
                        h.on = True; L[tag] = Fn.log_softmax(B.mS(input_ids=o["x"], logits_to_keep=kr, use_cache=False).logits[0, :-1].float(), -1); h.on = False
            for tag, (fm, h) in F.items():
                if fm == "lora":
                    pm.set_adapter(h); L[tag] = Fn.log_softmax(B.mS(input_ids=o["x"], logits_to_keep=kr, use_cache=False).logits[0, :-1].float(), -1)
            for a in C:
                C[a].append(float(-L[a].gather(-1, y[:, None])[:, 0].mean()))
                if a != "alone":
                    KLa[a].append(float((L["alone"].exp() * (L["alone"] - L[a])).sum(-1).mean()))
                if r["kind"].startswith("question"):
                    p_ = L[a][0].exp(); raw = {q: float(sum(p_[t] for t in ts)) for q, ts in lq.items()}; ms = sum(raw.values()); P[a].append(raw[r["true"]] / ms if ms else 0.0)
            del o, L
        A = {a: np.array(v) for a, v in C.items()}; live = A["control"] - A["bridge"]
        res = {"rows": len(A["alone"]), "means": {a: round(float(v.mean()), 4) for a, v in A.items()}, "p(true)": {a: round(float(np.mean(v)), 4) for a, v in P.items() if v},
               "KL from Qwen alone (per token)": {a: round(float(np.mean(v)), 4) for a, v in KLa.items() if v}, "live lead": round(float(live.mean()), 4), "pairs": {}}
        for pr_ in pairs:
            bt, ct = pr_.replace("_X_", "_bridge_"), pr_.replace("_X_", "_control_")
            if bt in A and ct in A:
                res["pairs"][pr_] = {"lead": round(float((A[ct] - A[bt]).mean()), 4), "gain over alone, bridge form / control form": [round(float((A["alone"] - A[bt]).mean()), 4), round(float((A["alone"] - A[ct]).mean()), 4)],
                                     "retention [mean, 5%, 95%]": boot(A[ct] - A[bt], live),
                                     "fidelity bridge form": boot(A["alone"] - A[bt], A["alone"] - A["bridge"]), "fidelity control form": boot(A["alone"] - A[ct], A["alone"] - A["control"])}
        out["sets"][name] = res; log("graft eval", name, json.dumps(res), round(time.time() - t0), "s")
    json.dump(out, open(OD / f"xgraft_eval_s{SEED}.json", "w"), indent=1); print("GEVAL_DONE", flush=True)


@torch.no_grad()
def recon_eval():
    """addendum 171: the live translators XB_RE_TR (stems in out/; plus the control) and the grafts XB_RE_G (stems; the bridge down) on X1's held
    agent rows, held fresh questions and held prose: next-token cross-entropy on the reply (p(true) on questions), with paired bootstrap
    differences of every graft against the first graft and against the control graft (xgraft_control_w0, when present)"""
    B = Bridge("bridge"); rows = rows_all(B.tg); t0 = time.time(); gl = [B.WL] + B.sites; Tc = load_tr(B, "control")[0]
    trs = {n_: load_tr(B, "bridge", name=n_)[0] for n_ in os.environ["XB_RE_TR"].split(",")}; gfs = {}
    for n_ in dict.fromkeys(os.environ["XB_RE_G"].split(",") + [c_ for c_ in os.environ.get("XB_RE_C", "").split(",") if c_] + (["xgraft_control_w0"] if (OD / "xgraft_control_w0.pt").exists() else [])):
        ck = torch.load(OD / f"{n_}.pt", weights_only=False, map_location=DEV); G = QGraft(B.ds, len(gl), ck["width"]).to(DEV); G.load_state_dict(ck["state"]); gfs[n_] = GraftHooks(B.mS, gl, G.eval())
    lq = {a: sorted({B.tq.encode(v, add_special_tokens=False)[0] for v in (a, " " + a)}) for a in "ABC"}; rng = np.random.default_rng(171); sets = {}
    for k in ("agent", "question", "question_fresh"):
        hk = [r for r in rows if r["kind"] == k and r["split"] == "held"]; picked = [hk[i] for i in rng.permutation(len(hk))[:NX1H if k != "question_fresh" else 2 * NX1H]]
        if k != "question":
            sets[f"held {k}"] = picked
    hk = [r for r in rows if r["kind"] == "prose" and r["split"] == "held"]; sets["held prose (untaught)"] = [hk[i] for i in rng.permutation(len(hk))[:3 if SMALL else 30]]
    out = {"seed": SEED, "translators": list(trs), "grafts": list(gfs), "sets": {}}
    for name, rs in sets.items():
        C = {a: [] for a in ["alone", "control"] + [f"live {t}" for t in trs] + list(gfs)}; P = {a: [] for a in C}
        for r in rs:
            e = encode(B.tg, B.tq, r)
            if e is None:
                continue
            gids, qids, j, rstart, s0 = e; n = len(qids); y = torch.tensor(qids[rstart:], device=DEV); kr = n - rstart + 1; o = B.look(gids, qids, j, rstart); L = {"alone": o["la"]}
            B.write(Tc, o["S"], s0, n); L["control"] = Fn.log_softmax(B.mS(input_ids=o["x"], logits_to_keep=kr, use_cache=False).logits[0, :-1].float(), -1); B.P.slot.clear()
            for t, T in trs.items():
                B.write(T, o["T"], s0, n); L[f"live {t}"] = Fn.log_softmax(B.mS(input_ids=o["x"], logits_to_keep=kr, use_cache=False).logits[0, :-1].float(), -1); B.P.slot.clear()
            for g_, h in gfs.items():
                h.on = True; L[g_] = Fn.log_softmax(B.mS(input_ids=o["x"], logits_to_keep=kr, use_cache=False).logits[0, :-1].float(), -1); h.on = False
            for a in C:
                C[a].append(float(-L[a].gather(-1, y[:, None])[:, 0].mean()))
                if r["kind"].startswith("question"):
                    p_ = L[a][0].exp(); raw = {q: float(sum(p_[t_] for t_ in ts)) for q, ts in lq.items()}; ms = sum(raw.values()); P[a].append(raw[r["true"]] / ms if ms else 0.0)
            del o, L
        A = {a: np.array(v) for a, v in C.items()}; g0 = os.environ["XB_RE_G"].split(",")[0]; res = {"rows": len(A["alone"]), "means": {a: round(float(v.mean()), 4) for a, v in A.items()}}
        if any(P.values()):
            res["p(true)"] = {a: round(float(np.mean(v)), 4) for a, v in P.items() if v}

        def bd(a, b, n=1000):                                            # mean(a - b) per row with a paired bootstrap 90% interval
            d_ = A[a] - A[b]; rr = np.random.default_rng(0); bs = [d_[rr.integers(len(d_), size=len(d_))].mean() for _ in range(n)]
            return [round(float(d_.mean()), 4), round(float(np.percentile(bs, 5)), 4), round(float(np.percentile(bs, 95)), 4)]
        res["each graft minus the first graft (negative = better) [mean, 5%, 95%]"] = {g_: bd(g_, g0) for g_ in gfs if g_ != g0}
        if "xgraft_control_w0" in gfs:
            res["control graft (1,000 steps) minus each graft (steps not matched beyond round 0)"] = {g_: bd("xgraft_control_w0", g_) for g_ in gfs if g_ != "xgraft_control_w0"}
        for k_ in (1, 2):                                                # review 11-07: the comparisons that decide addendum 171
            r_, d_, c_ = f"{g0}_r{k_}", f"{g0}_d{k_}", f"xgraft_control_w0_c{k_}"
            if r_ in gfs and d_ in gfs:
                res.setdefault("re-attached minus dose control, same steps (negative = reconciliation helps)", {})[f"round {k_}"] = bd(r_, d_)
            for g_ in (r_, d_):
                if g_ in gfs and c_ in gfs:
                    res.setdefault("step-matched control graft minus graft (the stored E4B-specific lead)", {})[g_] = bd(c_, g_)
        res["per row"] = {a: [round(float(x), 5) for x in v] for a, v in A.items()}
        res["live lead of each translator (control minus live)"] = {t: bd("control", f"live {t}") for t in trs}
        out["sets"][name] = res; log("recon eval", name, json.dumps({k_: v for k_, v in res.items() if k_ != "per row"}), round(time.time() - t0), "s")
    json.dump(out, open(OD / f"recon_eval_s{SEED}{os.environ.get('XB_RE_TAG', '')}.json", "w"), indent=1); print("RECON_EVAL_DONE", flush=True)

if __name__ == "__main__":
    {"train": lambda: train(sys.argv[2]), "suite": suite, "lora": lambda: train_lora(sys.argv[2]), "leval": lambda: lora_eval(sys.argv[2]), "x1": x1, "graft": lambda: train_graft(sys.argv[2], sys.argv[3]), "geval": graft_eval, "reval": recon_eval}[sys.argv[1]]()
