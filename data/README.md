# Data folder

The datasets are not stored in this repository (about 115 MB). Run

```bash
python get_data.py
```

to download the four CSV files into this folder from the public repository of Yuan et al. (arXiv:2411.19493):
https://github.com/Y-debug-sys/DTM/tree/main/Data

- `abilene_tm.csv`, `abilene_rm.csv`: Abilene, 12 nodes, 48,384 five-minute traffic matrices (original data: Y. Zhang, https://www.cs.utexas.edu/~yzhang/research/AbileneTM/)
- `geant_tm.csv`, `geant_rm.csv`: GÉANT, 23 nodes, 10,772 fifteen-minute traffic matrices (Uhlig et al., ACM SIGCOMM CCR 2006)
