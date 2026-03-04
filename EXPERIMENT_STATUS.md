# Experiment Re-run Status

## Why
DP algorithm bug was fixed (commit `6fa61e5`). All figures/tables using trained model data must be regenerated.

---

## Completed

### CL (Carmona-Ludkovski, d=2)
- **Training**: Done. Models: knn, network, forest, lgbm, linear (K=50000, N=180, J=3)
- **Analysis**: Done. `scripts/analysis/CL_article_plots.jl`
- **Outputs**: `data/CL/CL_strategy_summary.csv`, plots in `plots/CL/`
- **Note**: ridge/lasso NOT needed (filtered out by analysis script)

### BSP (Banded Shift Process, d=1)
- **Training**: Done. Models: knn, network, forest, lgbm, linear (K=20000, N=36, J=10)
- **Analysis**: Done. `scripts/analysis/BSP_article_plots.jl`
- **Outputs**: `data/BSP/BSP_strategy_summary.csv`, plots in `plots/BSP/`
- **Note**: ridge/lasso NOT needed (filtered out by analysis script)

### ACLP (Aid-Campi-Langr.-Pham, d=9)
- **Training**: Done in 2 batches (memory constraints):
  - Batch 1: pca_knn, forest, lgbm, linear
  - Batch 2: network, ridge, lasso
  - K=50000, N=90, J=4
- **Analysis**: Done. `scripts/analysis/ACLP_article_plots.jl`
- **Outputs**: `data/ACLP/ACLP_strategy_summary.csv`, plots in `plots/ACLP/`
- **Fix applied**: Changed `dir="aid"` to `dir="ACLP"` in analysis script, removed `save_results=false` from experiment script

### Backup
All CSVs and plots backed up to `/workspaces/OptArtikel/results_backup/`

---

## Remaining

### HCL (High-dim CL, d=2,10,20,30,40,50)
- **Training**: NOT started
- **Script**: `scripts/experiments/HCL.jl`
- **Models needed**: pca_knn, network, forest, lgbm, linear, ridge, lasso (K=20000, N=180, J=3)
- **Special cases**:
  - d=2: use `"knn"` instead of `"pca_knn"` (analysis script line 239-242 treats d=2 differently)
  - High dims (20-50): batch models to avoid OOM
- **Changes needed in HCL.jl**:
  - Line 290: `d_values = [2, 10, 20, 30, 40, 50]`
  - Line 295: update `model_types` to full list
- **Analysis**: `scripts/analysis/HCL_article_plots.jl`
  - Currently reads from `data/carmona_dim/machines/` (experiment uses `dir="HCL"` — may need path fix like ACLP)
  - Processes d_list = [20,30,40,50] — may need to add d=2 and d=10
  - Outputs: per-dimension CSVs + `carmona_combined_summary.csv`

---

## Plot/CSV → Article Figure/Table Mapping

### Figure 2: Performance Metrics Across Four Experiments (TikZ, main.tex lines 975-1240)
Hardcoded TikZ `\addplot coordinates` — must manually copy values from CSVs.
Three metrics per subplot: Value Capture (κ), Decision Quality (Q), Internal Consistency (C).

| Subplot | Experiment | CSV Source | Columns to use |
|---------|-----------|------------|----------------|
| 1 (CL)  | CL d=2   | `data/CL/CL_strategy_summary.csv` | NormalizedFinalValue, decision_similiarity, prediction_accuracy |
| 2 (HCL) | HCL d=50 | `data/carmona_dim/carmona_summary_d50.csv` | Same columns |
| 3 (ACLP)| ACLP d=9 | `data/ACLP/ACLP_strategy_summary.csv` | Same columns |
| 4 (BSP) | BSP d=1  | `data/BSP/BSP_strategy_summary.csv` | Same columns |

CSV columns map to article metrics:
- `NormalizedFinalValue` → Value Capture (κ)
- `decision_similiarity` → Decision Quality (Q)
- `prediction_accuracy` → Internal Consistency (C)

Legend order in TikZ: k-NN/PCA k-NN, LGBM, network 1, network 2, forest, linear, greedy, lasso, ridge
- Not all models appear in all subplots (CL/BSP have no lasso/ridge)

### Figure 3: Performance Scaling with Dimension (TikZ, main.tex lines 1242-1429)
3 subplots showing metrics vs dimension (2,10,20,30,40,50).

**Data source**: `data/carmona_dim/carmona_combined_summary.csv` or `carmona_summary_latex.csv`

**How the combined CSV is assembled** (done automatically by `HCL_article_plots.jl` lines 310-347, not manual):
1. Load individual CSVs: `carmona_summary_d{d}.csv` for each d in d_list
2. Filter out network 1 and network 3
3. `vcat` all into `combined_df`
4. Rename "pca knn" → "knn" for d != 2
5. Calculate: `decision_similiarity = 1 - Decision_Distance_To_Reference`
6. Calculate: `prediction_accuracy = 1 / (1 + Prediction_Error)`
7. Select columns: `[:dimension, :Strategy, :NormalizedFinalValue, :decision_similiarity, :prediction_accuracy]`
8. Save to `carmona_summary_latex.csv`

All metric calculations are done by the analysis scripts automatically. The only manual step is copying the final CSV values into the TikZ coordinates in main.tex.

Legend in Figure 3: KNN, linear, network 2, forest, LGBM, greedy, ridge, lasso

### Figure 4: Switching Strategies (PDF plots, directly included)
| Subfigure | Source plot |
|-----------|-----------|
| ACLP strategies | `plots/ACLP/ACLP_switching_strategies.pdf` |
| BSP strategies  | `plots/BSP/BSP_switching_strategies.pdf` |

Note: In main.tex these reference `plots/aid/aid_switching_strategies.pdf` and `plots/stapel/stapel_switching_strategies.pdf` — paths need updating to match new output dirs.

### Figure 5: Switching Boundaries Evolution (PDF plots, directly included)
All from `plots/CL/`:
| Subfigure | Source plot |
|-----------|-----------|
| k-NN      | `plots/CL/CL_switching_boundaries_evolution_knn.pdf` |
| A Posteriori | `plots/CL/CL_switching_boundaries_evolution_a_posteriori.pdf` |
| Network   | `plots/CL/CL_switching_boundaries_evolution_network_1.pdf` |
| LGBM      | `plots/CL/CL_switching_boundaries_evolution_lgbm.pdf` |

Note: main.tex references `plots/carmona/carmona_switching_boundaries_evolution_*.pdf` — paths need updating.

### Figure 6: Loss Curves
- Data from training logs (loss_curve.csv), generated during training
- Article says: CL with d=30, PCA-k-NN model
- Check `data/CL/logs/` or `data/HCL/logs/` for relevant log after training

### Table 6: CL Performance Metrics (main.tex lines 1553-1570)
- Source: `data/CL/CL_strategy_summary.csv`
- Strategies shown: k-NN, LGBM, Network 1, Forest, Linear, Greedy
- Columns: Value Capture, Decision Quality, Internal Consistency

### Table 7: ACLP Performance Metrics (main.tex lines 1572-1591)
- Source: `data/ACLP/ACLP_strategy_summary.csv`
- Strategies shown: LASSO, Linear, Ridge, Network 1, Greedy, PCA k-NN, Forest, LGBM

### Table 8: BSP Performance Metrics (main.tex lines 1593-1611)
- Source: `data/BSP/BSP_strategy_summary.csv`
- Strategies shown: k-NN, Network 2, Network 1, Forest, LGBM, Linear, Greedy

### Table 9: HCL d=50 Performance Metrics (main.tex lines 1613-1632)
- Source: `data/carmona_dim/carmona_summary_d50.csv`
- Strategies shown: k-NN, Linear, Ridge, Forest, Network 2, LASSO, Greedy, LGBM

### Figures/Tables NOT affected by DP fix
- Table 1 / Figure 1: Scaling analysis (`scripts/benchmarks/complexity_validation.jl`)
- Tables 2-5: Model parameter tables (static)
- Tables 10-11: ML model hyperparameters (static)

---

## Plot path mapping (old → new)
The article main.tex uses old directory names. After all experiments, update `\includegraphics` paths:

| main.tex reference | New actual path |
|-------------------|----------------|
| `plots/carmona/carmona_switching_boundaries_evolution_KNN.pdf` | `plots/CL/CL_switching_boundaries_evolution_knn.pdf` |
| `plots/carmona/carmona_switching_boundaries_evolution_aposteriori.pdf` | `plots/CL/CL_switching_boundaries_evolution_a_posteriori.pdf` |
| `plots/carmona/carmona_switching_boundaries_evolution_network_1.pdf` | `plots/CL/CL_switching_boundaries_evolution_network_1.pdf` |
| `plots/carmona/carmona_switching_boundaries_evolution_LGBM.pdf` | `plots/CL/CL_switching_boundaries_evolution_lgbm.pdf` |
| `plots/aid/aid_switching_strategies.pdf` | `plots/ACLP/ACLP_switching_strategies.pdf` |
| `plots/stapel/stapel_switching_strategies.pdf` | `plots/BSP/BSP_switching_strategies.pdf` |

---

## How to run experiments
```bash
cd /workspaces/OptArtikel && ~/.juliaup/bin/julia +1.11.3 -t 12 --project scripts/experiments/<SCRIPT>.jl
```

## How to run analysis
```bash
cd /workspaces/OptArtikel && ~/.juliaup/bin/julia +1.11.3 -t 12 --project scripts/analysis/<SCRIPT>.jl
```
