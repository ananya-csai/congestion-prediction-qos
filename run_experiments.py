"""Runs the full experiment grid defined in experiments.json.

    python run_experiments.py            # everything (about 1.5-2 h on 2 CPU cores)
    python run_experiments.py --quick    # 1 seed, 500 test steps: checks the pipeline only

Resumable: a step whose output file already exists is skipped.
Every command and its duration is appended to outputs/experiment_log.txt.
"""
import argparse
import json
import subprocess
import sys
import time

import config as C
from features import run_tag

PY = sys.executable


def run(cmd, log):
    t0 = time.time()
    print("$", " ".join(cmd), flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    with open(log, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {dt:7.1f}s  rc={res.returncode}  {' '.join(cmd)}\n")
    if res.returncode != 0:
        print(res.stdout[-2000:], res.stderr[-2000:])
        raise SystemExit(f"command failed: {' '.join(cmd)}")
    return res.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    cfg = json.loads((C.ROOT / "experiments.json").read_text())
    C.OUT_DIR.mkdir(exist_ok=True)
    log = C.OUT_DIR / "experiment_log.txt"
    seeds = cfg["seeds"][:1] if a.quick else cfg["seeds"]
    th, al, mg = cfg["theta"], cfg["alpha"], cfg["margin"]

    jobs = []  # (dataset, horizon, scale)
    for ds in cfg["datasets"]:
        for _, sc in cfg["loads"].items():
            jobs.append((ds, cfg["main_horizon"], sc))
        eh = cfg.get("extra_horizon")
        if eh:
            jobs.append((ds, eh["horizon"], cfg["loads"][eh["load"]]))

    for ds, h, sc in jobs:
        out = C.OUT_DIR / run_tag(ds, h, th, sc)
        common = ["--dataset", ds, "--horizon", str(h), "--theta", str(th), "--scale", str(sc)]
        if not (out / "features.pkl").exists():
            run([PY, "features.py", *common], log)
        for s in seeds:
            if not (out / f"test_probabilities_seed{s}.npz").exists():
                run([PY, "train_model.py", *common, "--seed", str(s)], log)
            if not (out / f"routing_summary_seed{s}_alpha{al:g}_m{mg:g}.csv").exists():
                extra = ["--max-steps", "500"] if a.quick else []
                run([PY, "route_optimizer.py", *common, "--seed", str(s),
                     "--alpha", str(al), "--margin", str(mg), *extra], log)

    fa = cfg["feature_ablation"]
    for ds in cfg["datasets"]:
        sc = cfg["loads"][fa["load"]]
        out = C.OUT_DIR / run_tag(ds, fa["horizon"], th, sc)
        common = ["--dataset", ds, "--horizon", str(fa["horizon"]), "--theta", str(th),
                  "--scale", str(sc), "--seed", str(fa["seed"])]
        for name, drop in fa["variants"].items():
            if a.quick:
                break
            if not (out / f"metrics_prediction_seed{fa['seed']}_{name}.csv").exists():
                run([PY, "train_model.py", *common, "--drop-features", ",".join(drop),
                     "--suffix", name], log)
    print("All experiments finished. Log:", log)


if __name__ == "__main__":
    main()
