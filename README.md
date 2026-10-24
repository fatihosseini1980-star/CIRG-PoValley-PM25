# CIRG-PoValley-PM25

## Paper and reproducibility archive

**Inductive Residual Transfer and Station-Block Calibration for PM2.5 Virtual Sensing at Unseen Sites**  
Fatemeh Hosseini and Omid Karimi — Semnan University.  
Manuscript destination: *Environmental Modeling & Assessment* (submission preparation; not a publication or an acceptance claim).

This is a clean, complete replacement for the public **code-and-results** repository, assembled from the author's supplied repository archive and the most recent EMA manuscript. It preserves the original frozen Stage-2 / Stage-3 code and the archived row-level outputs. See `docs/REPRODUCIBILITY_BOUNDARIES.md` for what has and has not been independently rerun.

### Where everything is

| Path | Description |
|---|---|
| `paper/` | Latest EMA LaTeX source, accompanying figures, class copy and compiled PDF; see `paper/README.md` |
| `config/` | Frozen study design, five station folds, ten locked sites, simulation settings |
| `src/` | Data construction and secondary evaluation/simulation scripts |
| `legacy/frozen_code/` | Original model scripts, preserved for provenance (some expect the historical `/mnt/data` layout) |
| `scripts/` | Relocatable runner and independent output validation |
| `results/` | Archived original article output, model parameters, numerical reports and figures |
| `validation/` | Five-fold 2022 independent rerun outputs and exploratory (not main-paper) blend analysis |
| `docs/` | Scope, execution, interpretation, checks and provenance |
| `data/` | Input location and acquisition instructions; large data files intentionally not tracked in public Git by default |

### Data

Original dataset: GRINS AQCLIM, Zenodo DOI: https://doi.org/10.5281/zenodo.17605148 . Follow the provider's attribution and licensing terms. The public repository does **not** include raw/processed CSVs; the authors also have a separate private/offline data ZIP with the exact inputs.

Put the two supplied files at the paths below (the names here are the exact offline-archive names):

```
data/raw/PoValleyPM2520132023csv.csv
data/processed/PoValleyPM25Final20182023Observed.csv
```

The processed input has 101,913 rows from 50 stations; the 40 development stations contribute 55,444 2018–2021 training rows and the ten locked stations contribute 3,237 rows to the 2023 test. The 69,386 training-period rows reported as the *full 50-station data inventory* must not be mistaken for development training rows. See `docs/DATA_AND_DESIGN.md`.

To build the archive-compatible processed CSV from the raw input (correct day-of-year periods: 365 normally and 366 for leap years):

```bash
python -m pip install -r requirements.txt
python src/prepare_data.py --input data/raw/PoValleyPM2520132023csv.csv --output data/processed/PoValleyPM25Final20182023Observed.csv
python scripts/check_repository.py --data data/processed/PoValleyPM25Final20182023Observed.csv
```

**Do not** replace archived experiment outputs with an alternative scaling specification or post-test exploratory results without documenting a new experiment. An alternative development-only scaling of the otherwise unused standardized PM2.5 column is provided for methodological exploration, not as the source of the published archived CSV.

### Verify archived external-test metrics without expensive refitting

```bash
python scripts/verify_archived_predictions.py --archive results/archived_article_outputs
python scripts/check_repository.py
```

Expected locked-2023 point RMSE: CIRG **5.897327**; RBF25 **5.682747**. The archive's probabilistic scores and intervals are computed from stored row-level predictions.

### Simulation smoke test and full model refit

```bash
CIRG_SIM_REPS=2 CIRG_SIM_SHIFT_REPS=2 bash scripts/run_simulation.sh
python scripts/run_full_real_data.py --data data/processed/PoValleyPM25Final20182023Observed.csv --workspace work_primary
bash scripts/run_post_analysis.sh work_primary
```

The five-fold full refit is computationally demanding and creates fresh outputs in the specified workspace. The historical model code is preserved unchanged; the launcher applies path relocation at execution time. Do **not** mistake a rerun for the frozen archive automatically: numerical environment, pipeline assumptions and inputs should be reconciled before replacing tables.

### Paper build

`cd paper && latexmk -pdf CIRG_EMA_Manuscript.tex` (with a compatible Springer Nature `sn-jnl.cls` and LaTeX dependencies installed). Check the publisher's current template before submission; the bundled class is a working copy, not a representation that it is the latest released publisher class.

### Main interpretation

CIRG improves strongly over the covariate-only model, while RBF25 outperforms it on dense-network external RMSE, MAE and CRPS. Source thinning is an explicitly exploratory post-test assessment, not a second independent locked confirmation. See `docs/REPRODUCIBILITY_BOUNDARIES.md`.
