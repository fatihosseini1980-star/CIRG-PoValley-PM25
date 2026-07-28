# Data placement

The raw GRINS AQCLIM export is approximately 241 MB and is intentionally not included in this repository.

Place it at:

`data/raw/Po_Valley_PM25_2013_2023_unbalanced.csv`

Then run:

```bash
python src/prepare_data.py \
  --input data/raw/Po_Valley_PM25_2013_2023_unbalanced.csv \
  --output data/processed/Po_Valley_PM25_Final_2018_2023_Observed.csv
```

The audit must return 50 stations and 101,913 observed station-day rows. Confirm redistribution rights before committing either raw or processed data to a public repository.
