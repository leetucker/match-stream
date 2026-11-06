# Stata reference scripts

These scripts are run by hand, from the repository root, and the files they
write in `tests/data/` are committed. Until then the tests that read them are
skipped. Never edit the outputs by hand.

The scripts (and the R ones in `tests/reference/`) write plain CSVs; the tests
read them gzipped. After running any of them, run
`python tests/reference/compress.py`, which gzips every `tests/data/ref_*.csv`
except `ref_data.csv`, the input the scripts read.

```bash
stata -b do tests/reference/stata/run_all.do
```

`run_all.do` runs every script even if one fails and says which failed;
look for "failed with return code" in `run_all.log`. All scripts import the
CSV with `asdouble` (Stata's default is single precision) and write JSON
numbers through `_json.do` (Stata's `%g` drops the leading zero, which JSON
needs).

| script | needs | writes | tests |
| --- | --- | --- | --- |
| `kmatch_kernel.do` | kmatch, moremata (`ssc install`) | `ref_kmatch_kernel.json`, `ref_kmatch_*_weights.csv` | tests/test_kernel.py |
| `kmatch_cv.do` | kmatch, moremata | `ref_kmatch_cv.json`, `ref_kmatch_cv_*_weights.csv` | tests/test_kernel.py (cross-validation without an outcome) |
| `kmatch_several.do` | kmatch, moremata | `ref_kmatch_several.json` | tests/test_several_outcomes.py (joint vcov of two outcomes) |
| `kmatch_nn.do` | kmatch, moremata | `ref_kmatch_nn.json`, `ref_kmatch_nn_*_ids.csv`, `ref_kmatch_nn_*_weights.csv` | tests/test_kmatch_nn.py |
| `psmatch2.do` | psmatch2 (`ssc install psmatch2`) | `ref_psmatch2.json`, `ref_psmatch2_*.csv` | tests/test_psmatch2.py |
| `teffects_ipw.do` | Stata 13+ | `ref_teffects_ipw.json` | tests/core/test_ipw.py |
| `teffects_lone.do` | Stata 13+ | `ref_teffects_lone.json`, `ref_teffects_lone_*_matches.csv` | tests/test_teffects_lone.py (how teffects treats a unit alone in its arm and exact cell; `effect(lone_unit_variance=)`) |
| `teffects_match.do` | Stata 13+ | `ref_teffects_match.json`, `ref_teffects_*_matches.csv`, `ref_teffects_pscore.csv` | tests/test_teffects.py |
