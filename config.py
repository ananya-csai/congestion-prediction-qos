"""Central settings for every module.

Change values here (or pass command-line flags) instead of editing the
other modules, so the paper and the code always use the same numbers.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"

# Where get_data.py downloads the CSV files from (public GitHub repository
# of Yuan et al., "Diffusion Models Meet Network Management", arXiv:2411.19493).
DATA_URL = "https://raw.githubusercontent.com/Y-debug-sys/DTM/main/Data/{file}"

DATASETS = {
    # n_nodes: routers; step_min: minutes per traffic matrix
    "abilene": {"n_nodes": 12, "step_min": 5},
    "geant": {"n_nodes": 23, "step_min": 15},
}

# ---- Congestion definition -------------------------------------------
THETA = 0.8            # link is congested when utilisation >= THETA
HORIZON = 1            # predict congestion h steps ahead

# ---- Capacity calibration (see paper, Sec. "Mathematical model") ------
# The datasets give traffic volumes but not link capacities. We assume all
# links have the same capacity C, chosen on the TRAINING period only, so
# that TARGET_CONGESTED_FRAC of training link-intervals have U >= THETA
# at LOAD_SCALE = 1.0.
TARGET_CONGESTED_FRAC = 0.05
LOAD_SCALE = 1.0       # multiply all demands (used for stress tests)
OUTLIER_QUANTILE = 0.999  # per-OD cap, fitted on training data only

# ---- Features ----------------------------------------------------------
N_LAGS = 6             # U(t), U(t-1), ..., U(t-5)
ROLL_WINDOW = 12       # steps used for rolling mean / std / max

# ---- Chronological split ----------------------------------------------
TRAIN_FRAC = 0.70
VAL_FRAC = 0.10        # test = remaining 0.20

# ---- Random Forest -----------------------------------------------------
RF_PARAMS = dict(
    n_estimators=150,
    max_depth=14,
    min_samples_leaf=20,
    max_features="sqrt",
    max_samples=0.3,           # each tree sees 30% of rows (speed on a laptop)
    class_weight="balanced_subsample",
    n_jobs=-1,
)
SEED = 42

# ---- Routing -----------------------------------------------------------
ALPHA = 5.0            # extra weight added to a link with congestion risk 1.0
MARGIN = 0.1           # greedy rerouting aims to keep links below (THETA - MARGIN)
