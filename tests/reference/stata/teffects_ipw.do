* Reference values from Stata's teffects ipw for match-stream's IPW core
* (match_stream/_core/ipw.py; tests/core/test_ipw.py::test_teffects_reference).
*
* Run from the repository root:   stata -b do tests/reference/stata/teffects_ipw.do
* Writes tests/data/ref_teffects_ipw.json. teffects ipw uses normalized
* (Hajek) weights and stacked-GMM standard errors that account for the
* estimated score: the same estimator and variance as ipw_mean_difference,
* up to the small-sample factor.
version 15
clear all
set more off
import delimited using "tests/data/ref_data.csv", clear asdouble   // full precision
do tests/reference/stata/_json.do

tempname fh
file open `fh' using "tests/data/ref_teffects_ipw.json", write replace
file write `fh' "{" _n
file write `fh' `"  "stata_version": "`c(stata_version)'","' _n
local cases "ate_logit ate logit atet_logit atet logit ate_probit ate probit atet_probit atet probit"
local n : word count `cases'
forvalues c = 1(3)`n' {
    local name : word `c' of `cases'
    local stat : word `=`c'+1' of `cases'
    local link : word `=`c'+2' of `cases'
    teffects ipw (y) (d x1 x2 x3, `link'), `stat'
    matrix b = e(b)
    matrix V = e(V)
    jnum est b[1,1]
    jnum se sqrt(V[1,1])
    file write `fh' `"  "`name'": {"est": `est', "se": `se', "N": "' (e(N)) "}"
    if `c' + 3 <= `n' file write `fh' ","
    file write `fh' _n
}
file write `fh' "}" _n
file close `fh'
