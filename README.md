# ML-Based Network Congestion Prediction and Traffic Optimization for QoS

Code and results for the paper **"Machine Learning-Based Network Congestion Prediction and Intelligent Traffic Optimization for QoS Enhancement"**. The paper is a Computer Networks Laboratory project, Manipal University Jaipur.

- Authors: Ananya Shukla and Dr. Praneet Saurabh, Department of Computer Science and Engineering, Manipal University Jaipur.
- Tools: Python, scikit-learn (Random Forest), NetworkX, pandas, matplotlib.

## What the code does

1. Loads real backbone traffic matrices and rebuilds the network topology as a graph:
   - Abilene: 12 nodes, 5-minute intervals;
   - GÉANT: 23 nodes, 15-minute intervals.
2. Routes the traffic on shortest paths and computes each link's utilisation. A link is labelled *congested* when utilisation is at least 0.8.
3. Trains a Random Forest to predict which links will be congested in the next interval. It is compared with two baselines: persistence ("nothing changes") and logistic regression.
4. Uses the predictions to reroute flows with a capacity-aware greedy procedure in NetworkX.
5. Compares four routing strategies on the test period:
   - static shortest path;
   - reactive rerouting;
   - predictive rerouting (Random Forest);
   - an oracle that knows the next interval.

   Each is tested at three traffic loads, with an ablation study and day-level Wilcoxon tests.

**Real or simulated:** the traffic demands are real measurements. Link capacities, routing decisions, delay (M/M/1 model) and loss (fluid overflow model) are simulated.

## Dataset

The data is not stored in this repository (about 115 MB). `get_data.py` downloads it from the public repository of Yuan et al. (arXiv:2411.19493): <https://github.com/Y-debug-sys/DTM/tree/main/Data>

| File | Network | Content |
|---|---|---|
| `abilene_tm.csv`, `abilene_rm.csv` | Abilene (Internet2) | 48,384 traffic matrices, 5-minute intervals, 1 Mar to 4 Sep 2004 |
| `geant_tm.csv`, `geant_rm.csv` | GÉANT | 10,772 traffic matrices, 15-minute intervals |

**Original sources:**
- Abilene: Y. Zhang, <https://www.cs.utexas.edu/~yzhang/research/AbileneTM/>
- GÉANT: Uhlig et al., "Providing public intradomain traffic matrices to the research community", ACM SIGCOMM CCR 36(1), 2006.

## Quick start

```bash
git clone https://github.com/<your-username>/congestion-prediction-qos.git
cd congestion-prediction-qos
python -m venv venv
# Windows: venv\Scripts\activate      Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
python get_data.py             # download the datasets (one time)
python run_experiments.py      # all experiments used in the paper
python analyze.py              # statistics and LaTeX tables -> outputs/results/
python make_figures.py         # figures                     -> outputs/figures/
```

`run_experiments.py` took about 25 minutes on a 2-core machine. `python run_experiments.py --quick` only checks that the pipeline runs; its numbers are not used in the paper.

## Running single steps

| Command | What it does |
|---|---|
| `python data_loader.py --dataset abilene` | Loads data and prints a topology summary |
| `python features.py --dataset abilene --scale 1.0` | Builds features and labels, with a chronological split |
| `python train_model.py --dataset abilene --scale 1.0` | Trains persistence, logistic regression and Random Forest |
| `python route_optimizer.py --dataset abilene --scale 1.0` | Runs the routing simulation (static / reactive / predictive / oracle) |
| `streamlit run dashboard.py` | Optional result browser (`pip install streamlit` first) |

Use `--dataset geant` for the second network. Use `--scale 0.8 / 1.0 / 1.25` for low, medium and high load.

## Experiments reported in the paper

All settings are in `experiments.json`:

| Setting | Value |
|---|---|
| Networks | Abilene, GÉANT |
| Loads | ×0.8 (low), ×1.0 (medium), ×1.25 (high) |
| Prediction horizon | 1 interval |
| Congestion threshold θ | 0.8 |
| Random Forest | 150 trees, max depth 14, seed 42 |
| Rerouting | α = 5, safety margin δ = 0.1 |

The recorded outputs are included:
- `outputs/results/*.csv`: prediction metrics, routing metrics, statistical tests, sensitivity check;
- `outputs/results/tables/*.tex`: LaTeX tables used in the paper;
- `outputs/figures/`: all result figures (PDF and PNG);
- `outputs/experiment_log.txt`: every command that was run, with its duration.

## How data leakage is prevented

- Chronological split: 70% train, 10% validation, 20% test. The last *h* steps before each boundary are dropped.
- The outlier cap and the link capacity are computed from training data only.
- The decision threshold is tuned on validation data only. The test period is used once.
- Features use only data up to time *t*. The label is for time *t + h*.

## Repository layout

```
config.py            all settings
get_data.py          dataset download
data_loader.py       loading + topology rebuild
network.py           routing and link-load helpers
features.py          features, labels, split
train_model.py       models and prediction metrics
route_optimizer.py   rerouting simulation and QoS metrics
run_experiments.py   runs the full experiment grid
analyze.py           statistics and LaTeX tables
make_figures.py      figures
dashboard.py         optional Streamlit viewer
templates/           empty result-table templates
outputs/             recorded results, tables, figures
```

## License

MIT. See `LICENSE`.
