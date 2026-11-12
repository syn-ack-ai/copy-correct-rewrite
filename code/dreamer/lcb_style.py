"""209k: whose style do the students' answers have? A naive-Bayes style score (token unigrams and bigrams, add-one smoothing) is fitted on the
299 row problems to tell E4B's greedy answers from E2B alone's, checked on the 188 held problems, and read on the students' held answers
(runs/<session>/xl_gens_<arm>.jsonl). Score of an answer: mean log-likelihood ratio per feature, E4B over E2B alone; above 0 = E4B's side.
usage: lcb_style.py NAME=session/arm ...      (a name ending _sN is paired with the other names of that seed). Standard library only."""
import collections, json, math, os, re, sys
_H = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(_H, "..", "runs", "")
BASE = json.load(open(os.path.join(_H, "..", "..", "papers", "kaggle_bridge", "data", "lcb", "xl_base_ids.json"))); ROWS, HELD = BASE["rows"], BASE["held"]
TOK = re.compile(r"[A-Za-z_]+|\d+|\S")


def feats(text, first=None):
    t = TOK.findall(text.rstrip().removesuffix("```")); t = t[:first] if first else t
    return t + [a + " " + b for a, b in zip(t, t[1:])]


def sign(w, l):
    n = w + l
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(0, min(w, l) + 1)) / 2 ** n)


def load(spec):
    ses, arm = spec.split("/"); return {r["id"]: r for r in map(json.loads, open(f"{RUNS}{ses}/xl_gens_{arm}.jsonl"))}


def main():
    arms = collections.OrderedDict(a.split("=") for a in sys.argv[1:])
    e4b = {r["id"]: r for r in map(json.loads, open(RUNS + "xqlB/xl_build_gens.jsonl")) if r["model"] == "e4b" and r["kind"] == "greedy"}
    alone = load(arms.pop("alone")); assert all(i in e4b and i in alone for i in ROWS + HELD)
    for first in (None, 200):
        C = {"e4b": collections.Counter(), "e2b": collections.Counter()}
        for i in ROWS:
            C["e4b"].update(feats(e4b[i]["text"], first)); C["e2b"].update(feats(alone[i]["text"], first))
        V = len(set(C["e4b"]) | set(C["e2b"])); N = {k: sum(v.values()) for k, v in C.items()}
        llr = lambda f: math.log((C["e4b"][f] + 1) / (N["e4b"] + V)) - math.log((C["e2b"][f] + 1) / (N["e2b"] + V))  # noqa: E731
        def score(text):
            f = feats(text, first); return sum(map(llr, f)) / max(1, len(f))
        s4, s2 = {i: score(e4b[i]["text"]) for i in HELD}, {i: score(alone[i]["text"]) for i in HELD}
        acc = (sum(v > 0 for v in s4.values()) + sum(v <= 0 for v in s2.values())) / (2 * len(HELD)); auc = sum((a > b) + 0.5 * (a == b) for a in s4.values() for b in s2.values()) / len(HELD) ** 2
        print(f"\n== {'whole answer' if not first else f'first {first} tokens'}: held problems, E4B against E2B alone: accuracy {acc:.3f} | AUC {auc:.3f} | mean score E4B {sum(s4.values()) / len(HELD):+.3f}, E2B alone {sum(s2.values()) / len(HELD):+.3f}")
        if not first:
            top = sorted((f for f in set(C["e4b"]) | set(C["e2b"]) if C["e4b"][f] + C["e2b"][f] >= 40), key=llr)
            print("   marks of E2B alone:", " | ".join(repr(f) for f in top[:14])); print("   marks of E4B:", " | ".join(repr(f) for f in top[::-1][:14]))
        S = {}
        for name, spec in arms.items():
            g = load(spec); S[name] = ({i: score(g[i]["text"]) for i in HELD}, g)
            sc = S[name][0]; print(f"   {name:9s} mean {sum(sc.values()) / len(HELD):+.3f} | scored as E4B's {sum(v > 0 for v in sc.values()):3d} of {len(HELD)} | closer to E4B's own score than E2B alone's is, by problem: {sum(abs(sc[i] - s4[i]) < abs(s2[i] - s4[i]) for i in HELD)}")
        for a in S:
            for b in S:
                if a < b and a.split("_s")[-1] == b.split("_s")[-1]:
                    sa, sb = S[a][0], S[b][0]; w = sum(sa[i] > sb[i] for i in HELD); l = sum(sa[i] < sb[i] for i in HELD)
                    both = [i for i in HELD if S[a][1][i]["pass"] and S[b][1][i]["pass"]]; w2 = sum(sa[i] > sb[i] for i in both); l2 = sum(sa[i] < sb[i] for i in both)
                    print(f"   {a} vs {b}: higher score on {w} vs {l} held problems (p={sign(w, l):.4f}) | where both pass ({len(both)}): {w2} vs {l2} (p={sign(w2, l2):.4f})")


if __name__ == "__main__":
    main()
