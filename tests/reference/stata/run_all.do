* All Stata references (see README.md). From the repository root:
*   stata -b do tests/reference/stata/run_all.do
* Each script runs even if an earlier one fails; check the log for errors.
foreach f in teffects_match teffects_ipw kmatch_kernel kmatch_nn psmatch2 teffects_lone kmatch_cv kmatch_several kmatch_estimands {
    capture noisily do tests/reference/stata/`f'.do
    if _rc di as error "`f'.do failed with return code " _rc
    else di as text "`f'.do done"
}
