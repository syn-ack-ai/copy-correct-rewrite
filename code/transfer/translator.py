"""Addendum 114: a translator from E4B's states into E2B's, trained through E2B, proven before any edit (user: "I think we need to teach a very good translator then, first... once we know it will do the job then we can test it").
python transfer/translator.py prep          -> transfer/pair/tr/: token ids; E4B's states at its source layers and its top-32 next-token log-probs; E2B's states at its source layers (layer 12 first)
python transfer/translator.py train NAME    NAME = arch_source: arch ridge | linear | mlp | attn | multi | gated | all | alladd, source e4b | e2b (the control: the same translator fed E2B's own states)
python transfer/translator.py gate NAME     -> transfer/pair/tr/gate_NAME.json
The translator writes E2B's layer-12 state (35% depth; layers 13 and 14 still compute the K/V that E2B's shared layers 15-34 read).
Stage 1, closed form: ridge to E2B's own layer-12 states on general text, in whitened coordinates. Stage 2: the translated states replace E2B's at layer 12, and the translator is trained until E2B's next-token outputs match E4B's (top-32), with a small pull toward E2B's own states on general text.
Facts held out for the gate: half of those E4B writes on the first wording and E2B does not; the 128 known facts of patch2.py."""
import gc, json, math, os, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE))
import pair as PR
OUT = PR.OUT; DEV = PR.DEV; SMALL = os.environ.get("TR_SMALL") == "1"; TR = OUT / ("tr_small" if SMALL else "tr"); TR.mkdir(exist_ok=True)
TOPK = 32; ALEN = 16
# One student and any number of teachers (user 11-05: "it should also support 3 models at the same time"). All three Gemma 4 sizes share one tokenizer.
# TR_STUDENT: the model written into. TR_TEACHERS: comma-separated; the first one's outputs are the training target, and an arm's source can pool several ("31b+e2b").
# Arm names are ARCH_SOURCE: the source is the model(s) whose states feed the translator; the student's own name is the control.
MODELS = {"e2b": {"nl": 35, "width": 1536, "kvfrom": 15}, "e4b": {"nl": 42, "width": 2560, "kvfrom": 24}, "31b": {"nl": 60, "width": 5376, "kvfrom": 60}}
WRITE_ANSWER = os.environ.get("TR_WRITE_ANSWER") == "1"   # addendum 144: facts are written over the answer tokens too (teacher-forced on the teacher's answer), not the prompt only
REP = int(os.environ.get("TR_REP", "0"))   # replicate (11-06): shifts the held-out split and the training order; 0 = every earlier run
STUDENT = os.environ.get("TR_STUDENT", "e2b"); TEACHERS = os.environ.get("TR_TEACHERS", "e4b").split(","); T0 = TEACHERS[0]
rel = lambda k, f: int(round(f * (MODELS[k]["nl"] - 1)))  # noqa: E731
SRC = {k: [rel(k, 0.35), rel(k, 0.5), rel(k, 0.65)] for k in MODELS}; WIDTH = {k: v["width"] for k, v in MODELS.items()}   # the student's first layer is its write layer
WRITE0 = SRC[STUDENT][0]
SRCD = bool(os.environ.get("TR_SRC_DEPTHS"))
if os.environ.get("TR_SRC_DEPTHS"):                                      # addendum 140 ladder: the source layers at other relative depths (the write layer stays at 35%)
    SRC = {k: [rel(k, float(f)) for f in os.environ["TR_SRC_DEPTHS"].split(",")] for k in MODELS}
WRITE, NL, KVFROM, DS = WRITE0, MODELS[STUDENT]["nl"], MODELS[STUDENT]["kvfrom"], WIDTH[STUDENT]; SITES = [rel(STUDENT, f) for f in (0.5, 0.65, 0.79)]
ALL_SITES = {"all": list(range(WRITE + 1, KVFROM - 1 if os.environ.get("TR_ALL_KV") == "1" else NL)), "alladd": list(range(0, NL))}
# TR_ALL_KV=1 (S3 review): layers from KVFROM on reuse K/V computed from the outputs of layers KVFROM-2 and KVFROM-1, so an addition after layer
# KVFROM-2 or later at a context position reaches only that position; later tokens (a reply) see only additions up to layer KVFROM-2   # all: the write plus additions on every later layer; alladd: additions at every layer, nothing replaced
NWIN = 200 if SMALL else int(os.environ.get("TR_NWIN", "12000")); NHELD = 20 if SMALL else 500; NROWS = 400 if SMALL else None
STEPS = 40 if SMALL else int(os.environ.get("TR_STEPS", "3000")); BW, BF = 12, 12; LR = float(os.environ.get("TR_LR", "3e-4")); BETA = float(os.environ.get("TR_BETA", "0.1")); LAMD = float(os.environ.get("TR_LAMD", "0.05"))
STRICT = os.environ.get("TR_SPLIT") == "strict"; HARM = float(os.environ.get("TR_HARM", "0")); SUF = ("_s" if STRICT else "") + ("_h" if HARM > 0 else "")   # HARM (after R125): pull to E2B's own state on training facts E2B already gets right   # strict split (after R125): every strict correction fact is held out too   # gated: weight on the size of the change


def log(*a):
    print(time.strftime("%H:%M"), *a, flush=True)


def rows_():
    return [json.loads(l) for l in open(OUT / "facts.jsonl")]


def splits(rows):
    """the same 128 known facts as patch2.py; half of the facts E4B writes on the first wording and E2B does not are held out"""
    rng = np.random.default_rng(113); known_all = [i for i, r in enumerate(rows) if r[f"{STUDENT}_q_four"] and r[f"{STUDENT}_q2_four"] and r[f"{T0}_q_four"] and r[f"{T0}_q2_four"]]; test_known = [int(i) for i in rng.permutation(known_all)[:128]]
    cand = [i for i, r in enumerate(rows) if r[f"{T0}_q_says"] and not r[f"{STUDENT}_q_says"]]; held = sorted(int(i) for i in np.random.default_rng(114 + REP).permutation(cand)[:len(cand) // 2])
    corr = {i for i, r in enumerate(rows) if r[f"{T0}_q_four"] and r[f"{T0}_q2_four"] and not r[f"{STUDENT}_q_four"] and not r[f"{STUDENT}_q2_four"]}
    if STRICT:
        held = sorted(set(held) | corr)
    out = set(held) | set(test_known); key = lambda t: " ".join(t.lower().split())  # noqa: E731
    bad = {key(rows[i][w]) for i in out for w in ("q", "q2")}; train = [i for i in range(len(rows)) if i not in out and key(rows[i]["q"]) not in bad and key(rows[i]["q2"]) not in bad]
    return held, test_known, train, corr


def tok_():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(PR.M[STUDENT], local_files_only=True)


def load(k):
    from transformers import AutoModelForCausalLM
    kw = {"attn_implementation": "eager"} if DEV == "mps" else {}        # 11-06: on MPS the default (sdpa) path is not causal for Gemma 4 (later tokens change earlier logits); eager is exact
    m = AutoModelForCausalLM.from_pretrained(PR.M[k], dtype=torch.bfloat16, device_map=DEV, local_files_only=True, **kw).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    if os.environ.get("TR_PLE_CPU") == "1":                              # 24 GB GPU (addendum 140): the per-layer embedding table is a lookup only; keep it in CPU memory
        lm = m.model.language_model if hasattr(m.model, "language_model") else m.model
        if getattr(lm, "embed_tokens_per_layer", None) is not None:
            lm.embed_tokens_per_layer = CpuLookup(lm.embed_tokens_per_layer, DEV); torch.cuda.empty_cache()
    return m


class CpuLookup(nn.Module):
    """an embedding kept on the CPU: looks up there, returns the rows on the GPU"""
    def __init__(s, emb, dev):
        super().__init__(); s.emb = emb.to("cpu"); s.dev = dev

    def forward(s, ids):
        return s.emb(ids.to("cpu")).to(s.dev)


@torch.no_grad()
def prep():
    rows = rows_(); tok = tok_(); bos = tok.bos_token_id
    seqs = []                                                            # (kind, fact, wording, ids, prompt length)
    wf = os.environ.get("TR_WIN_FROM")
    if wf:                                                               # the windows of an earlier prep (same tokenizer): token ids, kinds win and held
        ids0, off0, meta0 = np.load(f"{wf}/ids.npy"), np.load(f"{wf}/off.npy"), json.load(open(f"{wf}/meta.json"))
        wj = [j for j, mm in enumerate(meta0) if mm[0] == "win"][:NWIN] + [j for j, mm in enumerate(meta0) if mm[0] == "held"][:NHELD]   # at most NWIN and NHELD (small mode keeps it small)
        seqs += [(meta0[j][0], -1, "", ids0[off0[j]:off0[j + 1]].tolist(), 0) for j in wj]
        log("windows reused from", wf, sum(1 for x in seqs if x[0] == "win"), "train", sum(1 for x in seqs if x[0] == "held"), "held")
    else:
        import found as Fd
        os.environ["FOUND_SEED"] = "112"; Wtr = Fd.windows(NWIN); seqs += [("win", -1, "", [bos] + tok.encode(w, add_special_tokens=False), 0) for w in Wtr]
        sh = {w[i:i + 40] for w in Wtr for i in range(max(1, len(w) - 39))}; os.environ["FOUND_SEED"] = "113"; Wh = Fd.windows(3 * NHELD)
        Wh2 = [w for w in Wh if not any(w[i:i + 40] in sh for i in range(max(1, len(w) - 39)))][:NHELD]; del sh; log("held-out windows", len(Wh2), "kept; drawn", len(Wh), "(no 40-character stretch shared with training)")
        seqs += [("held", -1, "", [bos] + tok.encode(w, add_special_tokens=False), 0) for w in Wh2]
    for i, r in enumerate(rows[:NROWS]):
        for w in ("q", "q2"):
            p = [bos] + tok.encode(PR.FMT.format(q=r[w]), add_special_tokens=False); a = tok.encode(r[f"{T0}_{w}_text"], add_special_tokens=False)[:ALEN]
            seqs.append(("fact", i, w, p + a, len(p)))
    lens = np.array([len(s[3]) for s in seqs]); off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64); T = int(off[-1])
    np.save(TR / "ids.npy", np.concatenate([np.array(s[3], np.int32) for s in seqs])); np.save(TR / "off.npy", off)
    json.dump([[s[0], s[1], s[2], s[4]] for s in seqs], open(TR / "meta.json", "w")); log("sequences", len(seqs), "tokens", T)
    order = np.argsort(lens); B = 32
    for k in dict.fromkeys(TEACHERS + [STUDENT]):
        m = load(k); S = np.lib.format.open_memmap(TR / f"{k}_states.npy", mode="w+", dtype=np.float16, shape=(T, len(SRC[k]), WIDTH[k]))
        if k == STUDENT and SRCD:                                        # the write layer is no longer the first source layer: keep it on its own
            Wm = np.lib.format.open_memmap(TR / f"{STUDENT}_write.npy", mode="w+", dtype=np.float16, shape=(T, 1, WIDTH[k]))
        if k == T0:
            TI = np.lib.format.open_memmap(TR / f"{T0}_topi.npy", mode="w+", dtype=np.int32, shape=(T, TOPK)); TV = np.lib.format.open_memmap(TR / f"{T0}_topv.npy", mode="w+", dtype=np.float16, shape=(T, TOPK))
        t0 = time.time()
        for j in range(0, len(seqs), B):
            bi = order[j:j + B]; L = int(lens[bi].max()); x = torch.full((len(bi), L), tok.pad_token_id, dtype=torch.long); am = torch.zeros((len(bi), L), dtype=torch.long)
            for b, s in enumerate(bi):
                x[b, :lens[s]] = torch.tensor(seqs[s][3]); am[b, :lens[s]] = 1
            o = m(input_ids=x.to(DEV), attention_mask=am.to(DEV), output_hidden_states=True); hs = o.hidden_states[1:]; H = torch.stack([hs[l] for l in SRC[k]], 2).float().cpu().numpy().astype(np.float16)
            if k == STUDENT and SRCD:
                Hw = hs[WRITE].float().cpu().numpy().astype(np.float16)
            for b, s in enumerate(bi):
                n = int(lens[s]); S[off[s]:off[s] + n] = H[b, :n]
                if k == STUDENT and SRCD:
                    Wm[off[s]:off[s] + n, 0] = Hw[b, :n]
                if k == T0:
                    v, ix = Fn.log_softmax(o.logits[b, :n].float(), -1).topk(TOPK, -1); TI[off[s]:off[s] + n] = ix.cpu().numpy(); TV[off[s]:off[s] + n] = v.cpu().numpy().astype(np.float16)
            if j % (B * 100) == 0:
                log(k, j, "of", len(seqs), round((j + B) / (time.time() - t0), 1), "sequences a second")
        S.flush(); del S, m, o, hs; PR.free(); log(k, "done", "source layers", SRC[k])
        if k == STUDENT and SRCD:
            Wm.flush(); del Wm
    print("PREP_DONE", flush=True)


class Data:
    def __init__(s, source):
        s.ids = np.load(TR / "ids.npy"); s.off = np.load(TR / "off.npy"); s.meta = json.load(open(TR / "meta.json")); s.e2 = np.load(TR / (f"{STUDENT}_write.npy" if SRCD else f"{STUDENT}_states.npy"), mmap_mode="r")   # [:, 0] is the write-layer state
        s.srcs = [np.load(TR / f"{q}_states.npy", mmap_mode="r") for q in source.split("+")]   # pooled sources are concatenated per token
        s.ti = np.load(TR / f"{T0}_topi.npy", mmap_mode="r"); s.tv = np.load(TR / f"{T0}_topv.npy", mmap_mode="r"); s.din = sum(x.shape[1] * x.shape[2] for x in s.srcs)

    def src_rows(s, t):
        return np.concatenate([np.asarray(x[t], np.float32).reshape(len(t), -1) for x in s.srcs], 1)

    def seq(s, j, n=None):
        a = int(s.off[j]); n = int(s.off[j + 1]) - a if n is None else n
        return a, n

    def src_of(s, j, n=None):
        a, n = s.seq(j, n); return np.concatenate([np.asarray(x[a:a + n], np.float32).reshape(n, -1) for x in s.srcs], 1)


def stats(D, idx, k=60000):
    """per-dimension mean and spread of the source and of E2B's layer-12 state, on general text"""
    rng = np.random.default_rng(0); toks = np.concatenate([np.arange(D.off[j], D.off[j + 1]) for j in idx]); t = np.sort(rng.choice(toks, min(k, len(toks)), replace=False))
    X = D.src_rows(t); Y = np.asarray(D.e2[t, 0], np.float32)
    return {"mx": X.mean(0), "sx": X.std(0) + 1e-3, "my": Y.mean(0), "sy": Y.std(0) + 1e-3}


class Tr(nn.Module):
    def __init__(s, din, arch, st):
        super().__init__(); s.arch = arch; s.lin = nn.Linear(din, DS)
        s.sites = ALL_SITES["all"] if arch == "gall" else ALL_SITES.get(arch, SITES)
        for n_, v in st.items():
            s.register_buffer(n_, torch.tensor(v))
        if arch in ("mlp", "attn", "multi", "all"):
            s.mlp = nn.Sequential(nn.Linear(din, 4096), nn.GELU(), nn.Linear(4096, DS)); nn.init.zeros_(s.mlp[2].weight); nn.init.zeros_(s.mlp[2].bias)
        if arch == "attn":
            s.inp = nn.Linear(din, 768); s.blocks = nn.ModuleList([nn.TransformerEncoderLayer(768, 8, 3072, dropout=0.0, activation="gelu", batch_first=True, norm_first=True) for _ in range(2)]); s.outp = nn.Linear(768, DS); nn.init.zeros_(s.outp.weight); nn.init.zeros_(s.outp.bias)

        if arch == "multi":
            s.heads = nn.ModuleList([nn.Sequential(nn.Linear(din, 1024), nn.GELU(), nn.Linear(1024, DS)) for _ in SITES]); s.register_buffer("dsy", torch.ones(len(SITES), DS))
            for h_ in s.heads:
                nn.init.zeros_(h_[2].weight); nn.init.zeros_(h_[2].bias)
        if arch in ("all", "alladd"):
            s.trunk = nn.Sequential(nn.Linear(din, 1024), nn.GELU()); s.outs = nn.ModuleList([nn.Linear(1024, DS) for _ in s.sites]); s.register_buffer("dsy", torch.ones(len(s.sites), DS))
            for o_ in s.outs:
                nn.init.zeros_(o_.weight); nn.init.zeros_(o_.bias)

        if arch == "gated":                                              # keeps E2B's own state and learns a change: a gate on the ridge translation, plus a correction; both start at zero
            s.gate = nn.Linear(din + DS, 1); nn.init.zeros_(s.gate.weight); nn.init.constant_(s.gate.bias, -4.0)
            s.mlp = nn.Sequential(nn.Linear(din + DS, 4096), nn.GELU(), nn.Linear(4096, DS)); nn.init.zeros_(s.mlp[2].weight); nn.init.zeros_(s.mlp[2].bias)

        if arch == "gall":                                               # addendum 165: identity by default. From the source states AND E2B's own layer-12
            s.gate = nn.Linear(din + DS, 1); nn.init.zeros_(s.gate.weight); nn.init.zeros_(s.gate.bias)   # state: a gated change at the write layer and gated
            s.mlp = nn.Sequential(nn.Linear(din + DS, 4096), nn.GELU(), nn.Linear(4096, DS)); nn.init.zeros_(s.mlp[2].weight); nn.init.zeros_(s.mlp[2].bias)   # additions at the later
            s.trunk = nn.Sequential(nn.Linear(din + DS, 1024), nn.GELU()); s.outs = nn.ModuleList([nn.Linear(1024, DS) for _ in s.sites]); s.register_buffer("dsy", torch.ones(len(s.sites), DS))   # sites; no
            for o_ in s.outs:                                            # replacement: with the gate shut or the heads at zero, E2B is unchanged
                nn.init.zeros_(o_.weight); nn.init.zeros_(o_.bias)

    def gapply(s, x, h, noise=0.0, drop=None):
        """gall: x raw source states, h E2B's own raw layer-12 state (same positions); noise: Gaussian on the whitened source; drop: a 0/1 mask
        over the source width (whole depths). Returns the new raw layer-12 state, the raw additions per site, the whitened change, the gate, the mean square of the whitened additions."""
        z = (x - s.mx) / s.sx
        if getattr(s, "erU", None) is not None:                          # 209l: directions erased from what the corrector reads (set_erase)
            z = z - (z @ s.erU.T - s.era) @ s.erU
        if drop is not None:
            z = z * drop
        if noise > 0:
            z = z + noise * torch.randn_like(z)
        hw = (h - s.my) / s.sy; zh = torch.cat([z, hw], -1); g = torch.sigmoid(s.gate(zh)); d = g * s.mlp(zh); t = s.trunk(zh); aw = [g * o_(t) for o_ in s.outs]
        return h + s.sy * d, [s.dsy[k] * w_ for k, w_ in enumerate(aw)], d, g, torch.stack([(w_ ** 2).mean() for w_ in aw]).mean()

    def set_erase(s, path, check=True):
        """209l: erase directions of the whitened source states before the gall corrector reads them: z <- z - U^T (U z - a). The file (stub_dir.py)
        holds U (k x din, orthonormal rows), a (k: the value the component is set to, the no-stub class's mean) and the mx, sx the directions
        were found with, which must be this corrector's. Not part of the state dict: the caller sets it after every load (online.py ON_ERASE)."""
        e = torch.load(path, map_location=s.mx.device, weights_only=True); U, a = e["U"].float(), e["a"].float()
        assert s.arch == "gall" and U.dim() == 2 and U.shape[1] == s.mx.shape[0] and a.shape == (U.shape[0],), (s.arch, tuple(U.shape), tuple(a.shape))
        assert float((U @ U.T - torch.eye(U.shape[0], device=U.device)).abs().max()) < 1e-4, "the erased directions are not orthonormal"
        dm, ds = float(((e["mx"].float() - s.mx).abs() / s.sx).max()), float((e["sx"].float() / s.sx - 1).abs().max())
        assert not check or (dm < 0.02 and ds < 0.02), f"the directions were found with other whitening statistics (mean off by {dm:.4f} sd, spread by {ds:.4f})"   # check=False: a small test fit, whose statistics come from 6 calibration points (review 209l)
        s.erU, s.era = U, a; return {"directions": U.shape[0], "set to": [round(float(v), 4) for v in a], "statistics off by (mean, in sd; spread, ratio)": [round(dm, 5), round(ds, 5)]}

    def apply(s, x, h):
        """gated: x source states, h E2B's own layer-12 state (raw); returns the new raw state and the change in whitened units"""
        z = (x - s.mx) / s.sx; hw = (h - s.my) / s.sy; zh = torch.cat([z, hw], -1); d = torch.sigmoid(s.gate(zh)) * (s.lin(z) - hw) + s.mlp(zh)
        return h + s.sy * d, d

    def deltas_white(s, x):
        t = s.trunk((x - s.mx) / s.sx); return [o_(t) for o_ in s.outs]

    def deltas(s, x):
        if hasattr(s, "outs"):
            return [s.dsy[k] * w for k, w in enumerate(s.deltas_white(x))]
        z = (x - s.mx) / s.sx; return [s.dsy[k] * h_(z) for k, h_ in enumerate(s.heads)]

    def white(s, x, am):
        """x: B x L x din, raw source states; returns E2B's layer-12 state in whitened coordinates"""
        z = (x - s.mx) / s.sx; y = s.lin(z)
        if s.arch in ("mlp", "attn", "multi", "all"):
            y = y + s.mlp(z)
        if s.arch == "attn":
            L = z.shape[1]; h = s.inp(z); causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=z.device), 1)
            for b in s.blocks:
                h = b(h, src_mask=causal, src_key_padding_mask=~am)
            y = y + s.outp(h)
        return y

    def forward(s, x, am):
        return s.my + s.sy * s.white(x, am)


def ridge_init(T, D, idx, k=120000):
    """stage 1 in closed form: the linear part fits E2B's own layer-12 states (whitened); the ridge strength is picked on held-out tokens"""
    rng = np.random.default_rng(1); toks = np.concatenate([np.arange(D.off[j], D.off[j + 1]) for j in idx]); t = rng.permutation(toks)[:k]; nh = len(t) // 6; ho, t = np.sort(t[:nh]), np.sort(t[nh:])
    def get(tt):
        X = (torch.tensor(D.src_rows(tt), device=DEV) - T.mx) / T.sx; Y = (torch.tensor(np.asarray(D.e2[tt, 0], np.float32), device=DEV) - T.my) / T.sy; return X, Y
    G = torch.zeros(D.din + 1, D.din + 1, device=DEV); R = torch.zeros(D.din + 1, DS, device=DEV)
    for a in range(0, len(t), 8000):
        X, Y = get(t[a:a + 8000]); X = torch.cat([X, torch.ones(len(X), 1, device=DEV)], 1); G += X.T @ X; R += X.T @ Y
    Xh, Yh = get(ho); Xh = torch.cat([Xh, torch.ones(len(Xh), 1, device=DEV)], 1); best = None
    for lam in (10.0, 100.0, 1000.0, 10000.0):
        W = torch.linalg.solve(G + lam * torch.eye(D.din + 1, device=DEV), R); r2 = float(1 - ((Xh @ W - Yh) ** 2).sum() / ((Yh - Yh.mean(0)) ** 2).sum())
        if best is None or r2 > best[0]:
            best = (r2, lam, W)
    T.lin.weight.data = best[2][:-1].T.contiguous(); T.lin.bias.data = best[2][-1].contiguous(); log("ridge: held-out R2 (whitened)", round(best[0], 4), "strength", best[1])
    return best[0], best[1]


def batch(D, js, patch_prompt_only):
    """right-padded batch: ids, mask, source states, E2B's own layer-12 states, E4B's top-32, which positions get the translated state"""
    ns = [int(D.off[j + 1] - D.off[j]) for j in js]; L = max(ns); B = len(js); din = D.din
    x = np.zeros((B, L), np.int64); am = np.zeros((B, L), bool); pm = np.zeros((B, L), bool); win = np.zeros((B, L), bool); S = np.zeros((B, L, din), np.float32); E = np.zeros((B, L, DS), np.float32); TI = np.zeros((B, L, TOPK), np.int64); TV = np.full((B, L, TOPK), -1e4, np.float32)
    for b, j in enumerate(js):
        a, n = D.seq(j); x[b, :n] = D.ids[a:a + n]; am[b, :n] = True; S[b, :n] = D.src_of(j); E[b, :n] = D.e2[a:a + n, 0]; TI[b, :n] = D.ti[a:a + n]; TV[b, :n] = D.tv[a:a + n]
        kind, _, _, plen = D.meta[j]; pm[b, :plen if (kind == "fact" and patch_prompt_only) else n] = True; win[b, :n] = kind != "fact"
    t = lambda v: torch.from_numpy(v).to(DEV)  # noqa: E731
    return t(x), t(am), t(S), t(E), t(TI), t(TV), t(pm), t(win)


class Patch:
    """replaces E2B's layer-12 output where the mask is set; for multi, also adds the learned corrections at SITES"""
    def __init__(s, m, sites=None):
        L = m.model.language_model.layers; s.slot = {}; s.hs = [L[WRITE].register_forward_hook(s.hook)] + [L[l].register_forward_hook(s.adder(k)) for k, l in enumerate(SITES if sites is None else sites)]

    def remove(s):
        for h in s.hs:
            h.remove()

    def adder(s, k):
        def hook(_m, _i, out):
            if "d" not in s.slot:
                return out
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] != s.slot["mask"].shape[1]:
                return out
            hn = h + torch.where(s.slot["mask"][..., None], s.slot["d"][k].to(h.dtype), torch.zeros((), dtype=h.dtype, device=h.device))
            return (hn,) + tuple(out[1:]) if isinstance(out, tuple) else hn
        return hook

    def hook(s, _m, _i, out):
        if "mask" not in s.slot:
            return out
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] != s.slot["mask"].shape[1]:
            return out
        if "p" not in s.slot and "fn" not in s.slot:
            return out
        new = s.slot["fn"](h.float()) if "fn" in s.slot else s.slot["p"]
        hn = torch.where(s.slot["mask"][..., None], new.to(h.dtype), h)
        return (hn,) + tuple(out[1:]) if isinstance(out, tuple) else hn


def train(name):
    arch, source = name.split("_"); rows = rows_(); held, test_known, trf, _ = splits(rows); D = Data(source); meta = D.meta
    wins = [j for j, mm in enumerate(meta) if mm[0] == "win"]; trs = set(trf); facts = [j for j, mm in enumerate(meta) if mm[0] == "fact" and mm[1] in trs]
    kn = {i for i in trf if rows[i][f"{STUDENT}_q_four"] and rows[i][f"{STUDENT}_q2_four"] and rows[i][f"{STUDENT}_q_says"]}   # training facts E2B already gets right: the do-no-harm set
    T = Tr(D.din, "linear" if arch == "ridge" else arch, stats(D, wins)).to(DEV); r2, lam = ridge_init(T, D, wins); info = {"source layers": {k_: SRC[k_] for k_ in source.split("+")}, "write layer": WRITE, "write answer": WRITE_ANSWER, "ridge R2": r2, "ridge strength": lam, "training windows": len(wins), "training fact sequences": len(facts)}
    if arch != "ridge":
        m = load(STUDENT); P = Patch(m, T.sites); opt = torch.optim.AdamW(T.parameters(), lr=LR, weight_decay=0.0); rng = np.random.default_rng(7 + REP); t0 = time.time(); hist = []
        if T.arch in ("multi", "all", "alladd"):                         # the additions are scaled by the spread of E2B's own layer outputs at each site
            got = {}; tmp = [m.model.language_model.layers[l].register_forward_hook(lambda _m, _i, out, l=l: got.__setitem__(l, (out[0] if isinstance(out, tuple) else out).detach())) for l in T.sites]
            with torch.no_grad():
                x, am, *_ = batch(D, list(rng.choice(wins, 64, replace=False)), True); m(input_ids=x, attention_mask=am.long()); T.dsy.copy_(torch.stack([got[l][am].float().std(0) for l in T.sites]) + 1e-3)
            for h_ in tmp:
                h_.remove()
            del got
        sched = lambda s_: min(1.0, (s_ + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s_ / STEPS)))  # noqa: E731
        for step in range(STEPS):
            for g in opt.param_groups:
                g["lr"] = LR * sched(step)
            jw, jf = list(rng.choice(wins, BW, replace=False)), list(rng.choice(facts, BF, replace=False)); opt.zero_grad(set_to_none=True); kds, ancs = [], []
            for half in (jw[:BW // 2] + jf[:BF // 2], jw[BW // 2:] + jf[BF // 2:]):   # two halves, gradients added: the same step with half the memory
                x, am, S, E, TI, TV, pm, win = batch(D, half, not WRITE_ANSWER); hk = torch.zeros_like(am)
                for b, j in enumerate(half):
                    if HARM > 0 and meta[j][0] == "fact" and meta[j][1] in kn:
                        hk[b, :meta[j][3]] = True
                st = {}
                if T.arch == "gated":
                    def fn(h, S=S, st=st):
                        new, st["d"] = T.apply(S, h); return new
                    P.slot["fn"] = fn
                elif T.arch != "alladd":
                    wt = T.white(S, am); P.slot["p"] = T.my + T.sy * wt
                P.slot["mask"] = pm
                if T.arch in ("multi", "all"):
                    P.slot["d"] = T.deltas(S)
                if T.arch == "alladd":
                    ws = T.deltas_white(S); P.slot["d"] = [T.dsy[k] * w for k, w in enumerate(ws)]
                lg = m(input_ids=x, attention_mask=am.long()).logits; P.slot.clear()
                lq = Fn.log_softmax(lg[am].float(), -1); del lg; p4 = torch.softmax(TV[am], -1); kd = -(p4 * lq.gather(-1, TI[am])).sum(-1).mean()
                harm = torch.zeros((), device=DEV)
                if T.arch == "gated":
                    anc = (st["d"] ** 2)[pm].mean()
                    if hk.any():
                        harm = (st["d"] ** 2)[hk].mean()
                    ((kd + LAMD * anc + HARM * harm) / 2).backward()   # the smallest change that makes E2B match E4B
                elif T.arch == "alladd":
                    anc = torch.stack([(w ** 2)[pm].mean() for w in ws]).mean()
                    if hk.any():
                        harm = torch.stack([(w ** 2)[hk].mean() for w in ws]).mean()
                    ((kd + LAMD * anc + HARM * harm) / 2).backward()
                else:
                    anc = ((wt - (E - T.my) / T.sy) ** 2)[win & am].mean()
                    if hk.any():
                        harm = ((wt - (E - T.my) / T.sy) ** 2)[hk].mean()
                    ((kd + BETA * anc + HARM * harm) / 2).backward()
                kds.append(float(kd.detach())); ancs.append(float(anc.detach())); del lq
            torch.nn.utils.clip_grad_norm_(T.parameters(), 1.0); opt.step(); hist.append((np.mean(kds), np.mean(ancs)))
            if step % 100 == 0 or step == STEPS - 1:
                h = np.mean(hist[-100:], 0); log(name, "step", step, "teacher cross-entropy", round(float(h[0]), 4), "pull to E2B", round(float(h[1]), 4), round(time.time() - t0), "s")
        assert np.isfinite(np.array(hist[-100:])).all(), "training loss is not finite"
        P.remove(); info["last 100 steps (teacher cross-entropy, pull)"] = [round(float(v), 4) for v in np.mean(hist[-100:], 0)]; del m; PR.free()
    torch.save({"arch": T.arch, "din": D.din, "state": T.state_dict(), "info": info}, TR / f"{name}{SUF}.pt"); print("TRAIN_DONE", name, json.dumps(info), flush=True)


@torch.no_grad()
def gate(name):
    arch, source = name.split("_"); rows = rows_(); held, test_known, _, corr = splits(rows); D = Data(source); meta = D.meta; ck = torch.load(TR / f"{name}{SUF}.pt", weights_only=False)
    T = Tr(ck["din"], ck["arch"], {k: ck["state"][k].cpu().numpy() for k in ("mx", "sx", "my", "sy")}).to(DEV); T.load_state_dict(ck["state"]); T.eval()
    m = load(STUDENT); P = Patch(m, T.sites); tok = tok_(); res = {"name": name, "train info": ck["info"]}
    hw = [j for j, mm in enumerate(meta) if mm[0] == "held"]; acc = {}

    def add(k, v):                                                       # token sums and counts, so every token weighs the same
        a_ = acc.setdefault(k, [0.0, 0]); a_[0] += float(v.sum()); a_[1] += int(v.numel())
    for a in range(0, len(hw), 16):
        x, am, S, E, TI, TV, pm, win = batch(D, hw[a:a + 16], False); p4 = torch.softmax(TV[am], -1); add("E4B's own entropy (top-32)", -(p4 * torch.log(p4 + 1e-12)).sum(-1))
        nxt = torch.zeros_like(am); nxt[:, :-1] = am[:, 1:] & am[:, :-1]
        for tag, patched in (("E2B unpatched", False), ("E2B patched", True)):
            if patched:
                if T.arch == "gated":
                    tr = T.apply(S, E)[0]; P.slot["fn"] = lambda h, S=S: T.apply(S, h)[0]
                elif T.arch == "alladd":
                    tr = E
                else:
                    tr = T(S, am); P.slot["p"] = tr
                P.slot["mask"] = pm
                if T.arch in ("multi", "all", "alladd"):
                    P.slot["d"] = T.deltas(S)
                add("cosine, translated vs E2B's own layer 12", Fn.cosine_similarity(tr[am], E[am], dim=-1))
            lg = m(input_ids=x, attention_mask=am.long()).logits.float(); P.slot.clear(); lq = Fn.log_softmax(lg, -1)
            add(tag if patched else "E2B unpatched: cross-entropy to E4B", -(p4 * lq[am].gather(-1, TI[am])).sum(-1))
            add(f"{tag}: loss on the text", -lq[:, :-1].gather(-1, x[:, 1:, None])[..., 0][nxt[:, :-1]])
    res["held-out text"] = {k: round(v[0] / v[1], 4) for k, v in acc.items()}; h = res["held-out text"]
    h["share of the gap to E4B closed"] = round((h["E2B unpatched: cross-entropy to E4B"] - h["E2B patched"]) / (h["E2B unpatched: cross-entropy to E4B"] - h["E4B's own entropy (top-32)"]), 4); log(name, "held-out text", json.dumps(h))
    where = {(mm[1], mm[2]): j for j, mm in enumerate(meta) if mm[0] == "fact"}

    def facts(idx, w, patched):
        idx = [i for i in idx if (i, w) in where]; four, said = [], []
        for a in range(0, len(idx), 16):
            rb = [rows[i] for i in idx[a:a + 16]]; x, am, pos = PR.left(tok, [PR.FMT.format(q=r[w]) for r in rb]); L = x.shape[1]
            if patched:
                p = torch.zeros(len(rb), L, DS, device=DEV); mk = torch.zeros(len(rb), L, dtype=torch.bool, device=DEV); dd = [torch.zeros(len(rb), L, DS, device=DEV) for _ in T.sites]; Sf = torch.zeros(len(rb), L, D.din, device=DEV)
                for b, i in enumerate(idx[a:a + 16]):
                    j = where[(i, w)]; plen = meta[j][3]; s = torch.tensor(D.src_of(j, plen), device=DEV)[None]; mk[b, L - plen:] = True; Sf[b, L - plen:] = s[0]
                    if T.arch not in ("gated", "alladd"):
                        p[b, L - plen:] = T(s, torch.ones(1, plen, dtype=torch.bool, device=DEV))[0]
                    if T.arch in ("multi", "all", "alladd"):
                        for k_, d_ in enumerate(T.deltas(s)):
                            dd[k_][b, L - plen:] = d_[0]
                    assert x[b, L - plen:].tolist() == D.ids[D.off[j]:D.off[j] + plen].tolist(), "prompt tokens differ"
                P.slot["mask"] = mk
                if T.arch == "gated":
                    P.slot["fn"] = lambda h, Sf=Sf: T.apply(Sf, h)[0]
                elif T.arch != "alladd":
                    P.slot["p"] = p
                if T.arch in ("multi", "all", "alladd"):
                    P.slot["d"] = dd
            lg = m(input_ids=x, attention_mask=am, position_ids=pos).logits[:, -1].float(); g = m.generate(input_ids=x, attention_mask=am, max_new_tokens=16, do_sample=False); P.slot.clear()
            for q_, r in enumerate(rb):
                v = lg[q_, torch.tensor(r["first"], device=DEV)]; four.append(int((v[0] > v[1:]).all().item())); said.append(int(PR.says(tok.decode(g[q_, x.shape[1]:], skip_special_tokens=True).split("\n")[0], r["answer"])))
        return round(float(np.mean(four)), 4), round(float(np.mean(said)), 4)
    h1 = [i for i in held if rows[i][f"{T0}_q_says"] and not rows[i][f"{STUDENT}_q_says"]]; h2 = [i for i in h1 if not rows[i][f"{STUDENT}_q2_says"]]; hc = [i for i in held if i in corr]   # h1 == held on the half split
    sets = {"held-out facts E4B writes and E2B does not, first wording": (h1, "q"), "same facts, second wording, where E2B does not write it either": (h2, "q2"), "held-out facts in the strict correction set, first wording": (hc, "q"), "strict correction set, second wording": (hc, "q2"), "known facts, first wording": (test_known, "q")}
    res["facts (four-way, written)"] = {}
    for k, (idx, w) in sets.items():
        idx = [i for i in idx if (i, w) in where]
        if idx:
            res["facts (four-way, written)"][k] = {"n": len(idx), "unpatched": facts(idx, w, False), "patched": facts(idx, w, True)}; log(name, k, json.dumps(res["facts (four-way, written)"][k]))
    res["split"] = "strict" if STRICT else "half"; json.dump(res, open(TR / f"gate_{name}{SUF}.json", "w"), indent=1); print("GATE_DONE", name, flush=True)


if __name__ == "__main__":
    assert not SRCD or sys.argv[1] == "prep" or (TR / f"{STUDENT}_write.npy").exists(), "TR_SRC_DEPTHS needs a prep run with it (the write-layer states)"
    {"prep": prep, "train": lambda: train(sys.argv[2]), "gate": lambda: gate(sys.argv[2])}[sys.argv[1]]()
