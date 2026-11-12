"""Export the live test's per-task outcomes (addenda 196, 202, 207, 208) as one JSON: for every complete attempt (80 task results), each
task's outcome, error kind and number of model calls. Runs on box in ~/g4agent/rollouts (stdin); prints JSON.
  ssh box 'cd ~/g4agent/rollouts && python3 -' < live_export.py > papers/kaggle_bridge/data/live/live80_attempts.json"""
import glob, json
ARMS = {"plain E2B": ["x196_plain_a", "x196_plain_b", "x207_plain_c", "x207_plain_d"], "null adapter": ["x207_N_a", "x207_N_b"],
        "C LoRA (4090 seed 1)": ["x196_C_a", "x196_C_b"], "D LoRA (4090 seed 1)": ["x196_D_a", "x196_D_b", "x207_D_c", "x207_D_d"],
        "D4 LoRA (A100 seed 1)": ["x202_D4x_a", "x202_D4x_b"], "C LoRA (A100 seed 1)": ["x207_CA_a", "x207_CA_b"], "D LoRA (A100 seed 1)": ["x207_DA_a", "x207_DA_b"],
        "S LoRA (4090 seed 1)": ["x208_S_a", "x208_S_c"], "S4 LoRA (G4 seed 1)": ["x208_S4_a", "x208_S4_c"],   # 208c: x208_S_b and x208_S4_b lost their link to box (rollouts exited 255) and are not attempts; run again as _c
        "D LoRA (4090 seed 1), new driver": ["x208_D_e"], "null adapter, new driver": ["x208_N_c"]}   # 208b: control attempts after the 4090's kernel and driver upgrade
out = {"note": "80 held-out-repository agent tasks per attempt (tasks E4B failed in the run that chose them; in later attempts it solved 5 to 10 of the first 40 and 5 and 9 of the other 40); resolved = the hidden tests pass; error = how the episode ended if not by a submit (timeout or other); calls = model calls",
       "arms": {}}
for arm, names in ARMS.items():
    for n in names:
        d = {}
        for f in glob.glob(n + "/res_*/task_results.jsonl"):
            for l in open(f):
                if l.strip():
                    r = json.loads(l); e = str(r.get("error") or "")
                    d[r["instance_id"]] = {"resolved": bool(r["resolved"]), "error": "" if not e else ("timeout" if "timeout" in e.lower() else "other"), "calls": r["total_llm_calls"]}
        if len(d) == 80:
            out["arms"].setdefault(arm, {})[n] = d
print(json.dumps(out, sort_keys=True))
