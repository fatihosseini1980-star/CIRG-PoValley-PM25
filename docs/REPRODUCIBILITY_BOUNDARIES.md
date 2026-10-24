# Scientific provenance and reproduction status

The user-supplied original repository and final EMA manuscript are distinct versioned materials. This package preserves the frozen archive and aligns file names, data construction, documentation, and paper presentation. The stored locked-2023 row-level results have been independently rescored, and the five 2022 development folds have been rerun in the existing environment. Those checks do **not** establish that the whole data-to-figure pipeline has been independently rerun in a fresh environment or that every calibration variant is a new, nested out-of-sample guarantee.

### External test (archived results, unchanged)

| Metric | CIRG | RBF25 |
|---|---:|---:|
| RMSE | 5.897327 | 5.682747 |
| MAE | 3.763260 | 3.486626 |
| CRPS | 2.7799 | 2.5770 |
| nominal 90% interval coverage | 0.9574 | 0.9592 |

- The 2023 full-network performance of RBF25 is better for RMSE, MAE, and CRPS. Do not relabel or overwrite this result.
- The 10-source thinning comparison is a **post-test exploratory** stress test. It cannot be portrayed as the original untouched independent test.
- The retrospective 2022-weighted blend was investigated but did **not** beat RBF25 in locked 2023, nor yield a clear nested gain over CIRG in 2022; it is `validation/exploratory_blend_assessment_NOT_MAIN_ARTICLE.json`, not a manuscript replacement.
- Archived pooled point metrics and equal-day-weighted paired bootstrap contrasts use different averaging units. Do not compare their differences as if identical estimands.
- `legacy/frozen_code/` preserves original scripts. Runtime path patching is isolated in `scripts/run_full_real_data.py`.
- The raw source CSV and original processed file are kept in the offline data package, not the default public repository. Observe upstream data terms.
