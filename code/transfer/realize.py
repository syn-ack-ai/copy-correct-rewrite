"""Addendum 200, part 1 (user 11-09: "run the regression check on the 4090"): how much of each corrector's writes can E2B compute from its own states?
The report "Corrector layer placement evidence", rank 1. For the pair in ON_DIR/out (gall_e4b = cross, reads E4B; gall_e2b = self, reads E2B's own
layers 17/22), on held rows (the rung's 120 agent points, 40 each of prose, code, story, question), at up to NPOS reply positions per row (the span's rows from c - s0 on):
  targets (raw units) per corrector: w12 the layer-12 change; a13, a14 the additions at layers 13 and 14 (the last that reach E2B's shared KV);
           a15+ the additions at 15-34 summed; and each target's difference, cross minus self (the E4B-specific part)
  features: E2B's own states from the pass without a corrector: f12 = layer 12; f22 = layers 12, 17, 22; fall = layers 12, 14, 17, 22, 28, 34
  fits: ridge (lambda picked on a validation split) and a one-hidden-layer MLP (all targets jointly, early stop on validation), split by row
        (every 5th row test, the next validation, the rest train; per kind); R2 on the test rows, pooled over dimensions, also agent vs the rest.
Reply positions only (review 200). Forward passes and small fits only. Writes OD/realize_gall.json (realize_gall_small.json with ON_SMALL=1) and prints REALIZE_DONE.
  env as the rung (BASE + TR_SRC_DEPTHS=0.5,0.65 ON_DIR=...), from code11: python realize.py"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE))
import online as O
import translator as X

NROW = {k: (3 if O.SMALL else v) for k, v in (("agent", 120), ("prose", 40), ("code", 40), ("story", 40), ("question", 40))}
NPOS = 16 if O.SMALL else 96
FEATS = {"f12": [12], "f22": [12, 17, 22], "fall": [12, 14, 17, 22, 28, 34]}
PARTS = ["w12", "a13", "a14", "a15+"]


def load_tr(name):
    ck = torch.load(O.OD / f"{name}.pt", weights_only=False, map_location=O.DEV)
    want = {"reply": O.REPLY, "span": O.SPAN, "all_kv": __import__("os").environ.get("TR_ALL_KV"), "src depths": __import__("os").environ.get("TR_SRC_DEPTHS")}
    got = {k: ck.get(k) for k in want}; assert got == want, f"{name} was trained as {got}, the environment says {want}"
    assert ck["arch"] == "gall", ck["arch"]
    T = X.Tr(ck["din"], ck["arch"], {k: ck["state"][k].cpu().numpy() for k in ("mx", "sx", "my", "sy")}).to(O.DEV); T.load_state_dict(ck["state"]); T.eval()
    assert list(T.sites[:2]) == [13, 14] and T.sites[-1] == 34, T.sites
    return T, {k: ck.get(k) for k in ("steps", "imit beta", "last")}


@torch.no_grad()
def parts(T, src, h):
    new, dl, _, g, _ = T.gapply(src[None].float(), h[None].float())
    return {"w12": (new - h[None].float())[0], "a13": dl[0][0], "a14": dl[1][0], "a15+": torch.stack([d_[0] for d_ in dl[2:]]).sum(0)}, float(g.mean())


def r2(Y, P, mu, sd=None):
    """pooled over dimensions; with sd (the training sd per dimension), each dimension weighs alike (review 200: Gemma's large activations)"""
    if sd is not None:
        Y, P, mu = Y / sd, P / sd, mu / sd
    return float(1 - ((Y - P) ** 2).sum() / ((Y - mu) ** 2).sum().clamp_min(1e-12))


def main():
    t0 = time.time(); ids = None; mT, mS, gT, gS = O.load_models(); mS.eval()
    Tc, ic = load_tr("gall_e4b"); Ts, is_ = load_tr("gall_e2b"); rng = np.random.default_rng(200)
    Fs = {k: [] for k in FEATS}; Ys = {f"{w} {p}": [] for w in ("cross", "self", "cross - self") for p in PARTS}; meta = []; gates = {"cross": [], "self": []}
    for kind, nr in NROW.items():
        ids, _, held = O.data(kinds=(kind,))
        pick = [held[i] for i in np.random.default_rng(11 if kind == "agent" else 12).permutation(len(held))[:nr]]
        for ri, r in enumerate(pick):
            x, c, n, s0 = O.seq(ids, r)
            with torch.no_grad():
                o = O.look(mT, mS, gT, gS, x, c, n, s0)
            gT.out, gS.out = {}, {}; L = o["E"].shape[0]; r0 = c - s0; assert 0 < r0 < L, (r0, L)   # review 200: reply positions only (rows r0.. of the span)
            sel = torch.as_tensor(r0 + np.sort(rng.choice(L - r0, min(NPOS, L - r0), replace=False)), device=O.DEV)
            pc, gc = parts(Tc, o["T"], o["E"]); ps, gs_ = parts(Ts, o["S"], o["E"]); gates["cross"].append(gc); gates["self"].append(gs_)
            st = {12: o["E"], 17: o["S"][:, :X.DS], 22: o["S"][:, X.DS:], **{l: o["sites"][l] for l in (14, 28, 34)}}
            for k, ls in FEATS.items():
                Fs[k].append(torch.cat([st[l][sel].float() for l in ls], -1).cpu())
            for p in PARTS:
                Ys[f"cross {p}"].append(pc[p][sel].cpu()); Ys[f"self {p}"].append(ps[p][sel].cpu()); Ys[f"cross - self {p}"].append((pc[p] - ps[p])[sel].cpu())
            sp = "test" if ri % 5 == 0 else "val" if ri % 5 == 1 else "train"; meta += [(kind, sp)] * len(sel)
            del o
        O.log("realize: rows of", kind, "done", round(time.time() - t0), "s")
    del mT, mS, gT, gS, Tc, Ts; __import__("gc").collect(); torch.cuda.empty_cache()                # the models are not needed for the fits
    F = {k: torch.cat(v) for k, v in Fs.items()}; Y = {k: torch.cat(v) for k, v in Ys.items()}
    sp = np.array([m[1] for m in meta]); kd = np.array([m[0] for m in meta]); tr, va, te = (torch.as_tensor(sp == s) for s in ("train", "val", "test")); ag = torch.as_tensor(kd == "agent")
    out = {"addendum": 200, "small": O.SMALL, "dir": str(O.OD), "cross": ic, "self": is_, "rows": NROW, "positions": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
           "gate mean (cross, self)": [float(np.mean(gates["cross"])), float(np.mean(gates["self"]))],
           "mean squared norm per position (raw)": {k: float((v ** 2).sum(-1).mean()) for k, v in Y.items()}, "ridge": {}, "mlp": {}}
    for fk, Fx in F.items():
        mu, sd = Fx[tr].mean(0), Fx[tr].std(0) + 1e-6; Z = ((Fx - mu) / sd).to(O.DEV); Z = torch.cat([Z, Z.new_ones(len(Z), 1)], 1)
        G = Z[tr.to(O.DEV)].T @ Z[tr.to(O.DEV)]; lam0 = float(G.diagonal().mean()); out["ridge"][fk] = {}
        for yk, Yx in Y.items():
            Yd = Yx.to(O.DEV); ym = Yd[tr.to(O.DEV)].mean(0); R = Z[tr.to(O.DEV)].T @ (Yd[tr.to(O.DEV)] - ym); best = None
            for lr_ in (1e-3, 1e-2, 1e-1, 1.0):
                W = torch.linalg.solve(G + lr_ * lam0 * torch.eye(len(G), device=O.DEV), R); P_ = Z @ W + ym; v = r2(Yd[va.to(O.DEV)], P_[va.to(O.DEV)], ym)
                if best is None or v > best[0]:
                    best = (v, lr_, P_)
            P_ = best[2]; m_ = te.to(O.DEV); a_ = (te & ag).to(O.DEV); b_ = (te & ~ag).to(O.DEV)
            ysd = Yd[tr.to(O.DEV)].std(0) + 1e-8
            out["ridge"][fk][yk] = {"R2 test": r2(Yd[m_], P_[m_], ym), "R2 test, dimensions standardised": r2(Yd[m_], P_[m_], ym, ysd), "R2 test agent": r2(Yd[a_], P_[a_], ym), "R2 test other": r2(Yd[b_], P_[b_], ym), "lambda (x mean diag)": best[1]}
            del Yd, R, P_
        # the MLP: all targets jointly, each target scaled by its training rms, hidden 2048, early stop on validation loss
        keys = list(Y); Yall = torch.cat([Y[k] for k in keys], 1); ymu = Yall[tr].mean(0); ysd = torch.cat([torch.full((Y[k].shape[1],), float((Y[k][tr] - Y[k][tr].mean(0)).pow(2).mean().sqrt()) + 1e-8) for k in keys])
        Yn = ((Yall - ymu) / ysd).to(O.DEV); torch.manual_seed(200)
        net = torch.nn.Sequential(torch.nn.Linear(Z.shape[1], 2048), torch.nn.GELU(), torch.nn.Linear(2048, Yn.shape[1])).to(O.DEV); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        itr = tr.nonzero()[:, 0].to(O.DEV); bestv, bests, pat = None, None, 0
        for ep in range(4 if O.SMALL else 60):
            net.train()
            for b in itr[torch.randperm(len(itr), device=O.DEV)].split(512):
                opt.zero_grad(); ((net(Z[b]) - Yn[b]) ** 2).mean().backward(); opt.step()
            net.eval()
            with torch.no_grad():
                v = float(((net(Z[va.to(O.DEV)]) - Yn[va.to(O.DEV)]) ** 2).mean())
            if bestv is None or v < bestv:
                bestv, bests, pat = v, {k: w.clone() for k, w in net.state_dict().items()}, 0
            else:
                pat += 1
                if pat >= 6:
                    break
        net.load_state_dict(bests); net.eval(); out["mlp"][fk] = {"epochs": ep + 1}
        with torch.no_grad():
            Pn = net(Z) * ysd.to(O.DEV) + ymu.to(O.DEV)
        o_ = 0
        for k in keys:
            w = Y[k].shape[1]; Yd = Y[k].to(O.DEV); P_ = Pn[:, o_:o_ + w]; ym = Yd[tr.to(O.DEV)].mean(0); o_ += w
            m_ = te.to(O.DEV); a_ = (te & ag).to(O.DEV); b_ = (te & ~ag).to(O.DEV)
            sdk = Yd[tr.to(O.DEV)].std(0) + 1e-8
            out["mlp"][fk][k] = {"R2 test": r2(Yd[m_], P_[m_], ym), "R2 test, dimensions standardised": r2(Yd[m_], P_[m_], ym, sdk), "R2 test agent": r2(Yd[a_], P_[a_], ym), "R2 test other": r2(Yd[b_], P_[b_], ym)}
        del Z, G, Yn, net, Pn; torch.cuda.empty_cache(); O.log("realize: features", fk, "done", round(time.time() - t0), "s")
    out["seconds"] = round(time.time() - t0)
    json.dump(out, open(O.OD / f"realize_gall{'_small' if O.SMALL else ''}.json", "w"), indent=1); O.log(json.dumps({k: out[k] for k in ("positions", "gate mean (cross, self)")})); print("REALIZE_DONE", flush=True)


if __name__ == "__main__":
    main()
