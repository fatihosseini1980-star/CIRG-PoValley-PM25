# Data and frozen design

- Original inputs: GRINS AQCLIM, DOI https://doi.org/10.5281/zenodo.17605148, user-supplied original data extract.
- Selection: at least 80% PM2.5 coverage in every year 2018–2022, fixed before 2023 outcomes; 50 stations, 101,913 observed station-days.
- In the full **50-station inventory**: 69,386 observations in 2018–2021, 16,660 in 2022, 15,867 in 2023.
- **40 development stations**: 55,444 observations in 2018–2021, 13,323 observations used across five held-out station-fold evaluations in 2022.
- **Ten locked external stations**: 3,237 observations over 359 observed calendar days in 2023. Their IDs and the development folds are in `config/study_design.json`.
- The original archived file uses day-of-year sine/cosine denominators 365 for ordinary years and 366 for leap years. This version fixes the former repository script's denominator 365.25 to reproduce that archive.
- The archived standardized PM2.5 column was computed from the full 50-station inventory (including historic locked-station responses); it is unused by the five core frozen prediction scripts. A separate dev-only variant is available, but substituting its output means changing the historical processed input and is not the same exact archive rebuild. See `docs/REPRODUCIBILITY_BOUNDARIES.md`.
