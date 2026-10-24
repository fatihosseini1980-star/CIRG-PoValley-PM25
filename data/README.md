# Input data (kept separate from the public Git repository)

Dataset provenance: [GRINS AQCLIM / Zenodo v2.0.3](https://doi.org/10.5281/zenodo.17605148). Confirm original dataset redistribution and attribution conditions before making copies public.

The **offline bundle** for this project supplies two exact files:

* `raw/PoValleyPM2520132023csv.csv` (252,268,341 bytes)
* `processed/PoValleyPM25Final20182023Observed.csv` (59,544,696 bytes)

To reconstruct the latter using the correct archive-era day-of-year convention run:

```bash
python src/prepare_data.py --input data/raw/PoValleyPM2520132023csv.csv --output data/processed/PoValleyPM25Final20182023Observed.csv
```

The processed input is ignored in the public Git tree by default. If redistribution is permitted and you choose to track large inputs publicly, use Git LFS; do not force-add the 252 MB raw file as a normal Git object. See `docs/GITHUB_UPLOAD_FA.md`.
