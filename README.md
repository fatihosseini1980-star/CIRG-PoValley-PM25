# CIRG Po Valley reproducibility repository

This repository contains the frozen code and compact outputs for **Calibrated Inductive Residual Graphs for Prediction at Unseen Sites
under Temporal Shift: An Application to PM$_{2.5}$ Virtual Sensing**.

## What is included

- A deterministic data-construction script that recovers the 50-station, 101,913-row analysis file from the raw GRINS AQCLIM export.
- The complete component-level simulation with all numerical settings exposed in `config/simulation.json`.
- Frozen Stage-2 and locked 2023 model code preserved under `legacy/frozen_code/`.
- A relocatable full-refit launcher.
- Matched RBF25 uncertainty, event-probability analysis, 7/14-day moving-block bootstrap, and the two prespecified 203 µg/m³ sensitivity workflows.
- Precomputed simulation summaries, reports, and manuscript figures.

## Fast verification

The committed compact outputs can be inspected without rerunning the expensive model fit:

```bash
cat results/simulation/simulation_all_summary.csv
```

Run a small simulation smoke test:

```bash
CIRG_SIM_REPS=2 CIRG_SIM_SHIFT_REPS=2 bash scripts/run_simulation.sh
```

## Build the analysis data

```bash
python src/prepare_data.py \
  --input data/raw/Po_Valley_PM25_2013_2023_unbalanced.csv \
  --output data/processed/Po_Valley_PM25_Final_2018_2023_Observed.csv
```

Expected audit: 50 stations; 101,913 rows; 69,386 training rows; 16,660 validation rows; 15,867 external-year rows.

## Full real-data refit

```bash
python scripts/run_full_real_data.py \
  --data data/processed/Po_Valley_PM25_Final_2018_2023_Observed.csv \
  --workspace work_primary
```

This is computationally expensive because it performs five whole-station folds, learned-edge fitting, Student-t scale calibration, and the locked 2023 evaluation. It writes row-level predictions and all sensitivity inputs to `work_primary/`.

## Final controls before submission

```bash
bash scripts/run_post_analysis.sh work_primary
```

This produces:

- `Po_Valley_Sensitivity/moving_block_bootstrap.csv` for day-cluster, 7-day, and 14-day bootstrap intervals;
- `Po_Valley_Sensitivity/sensitivity_2023_203_metrics.csv`;
- a processed-data variant excluding the 2019 value of 203 for the required full refit;
- event metrics based on the calibrated Student-t predictive CDF.

The exact manuscript replacements for simulation settings and event probability are in `docs/MANUSCRIPT_VALUES.md`.

## Reproducibility boundary

The ten locked stations and 2023 outcomes were used only after the development specification was frozen. Source-thinning, outlier, and moving-block analyses are post-test sensitivity analyses and must be labelled accordingly.


## Completed sensitivity results

The final rerun generated all prespecified controls. Exact manuscript-output analyses are under `results/sensitivity/` and the insertion-ready text is in `docs/MANUSCRIPT_SENSITIVITY.tex`. The 2019 deletion required a controlled full refit; the 2023 deletion and 7/14-day moving-block bootstraps use the exact archived row-level predictions underlying the manuscript. See `docs/SENSITIVITY_RESULTS.md` for interpretation and the reproducibility boundary.

## Uncertainty-model implementation

The CIRG and RBF25 uncertainty workflows are implemented separately to preserve the matched evaluation design.

- `legacy/frozen_code/po_stage2_uncertainty_learned.py` implements the cross-fitted uncertainty model for CIRG development folds.
- `src/direct_uncertainty.py` applies the matched uncertainty procedure to the source-validated RBF25 comparator.
