* Reference values from Stata's teffects psmatch and teffects nnmatch for
* match-stream's matching estimators and Abadie-Imbens standard errors.
*
* Run from the repository root:   stata -b do tests/reference/stata/teffects_match.do
* Writes, in tests/data/:
*   ref_teffects_match.json        per case: estimate, SE, N, the command run
*   ref_teffects_<case>_matches.csv  id, d and the ids of the matches teffects
*                                  used (generate(); ties included), per case
*   ref_teffects_pscore.csv        id and the fitted logit and probit scores
*                                  (also in the JSON: their coefficients)
*
* Part 1, teffects psmatch: matching on an estimated score, SEs with the
* Abadie-Imbens (2016) correction (not built yet in match-stream; these are
* the references it will be checked against). For each psmatch case the
* same matches are also run through teffects nnmatch on the fitted score
* as a known variable (case name + "_known"): the gap between the two SEs is
* the 2016 correction alone.
* Part 2, teffects nnmatch: AI (2006) SEs and the AI (2011) bias adjustment,
* on the true score (column ps) or on x1-x3, within exact strata r, i: checks
* the existing effect() against teffects (so far it is checked against R
* Matching only).
*
* teffects' defaults kept unless the case says otherwise: ties are kept,
* vce(robust, nn(2)) (heteroskedastic, 2 same-arm matches for the
* conditional variances; match-stream's var_calc=2), metric Mahalanobis.
version 15
clear all
set more off
import delimited using "tests/data/ref_data.csv", clear asdouble   // full precision
do tests/reference/stata/_json.do
sort id                                     // generate() stores observation numbers
assert id == _n

tempname fh
file open `fh' using "tests/data/ref_teffects_match.json", write replace
file write `fh' "{" _n
file write `fh' `"  "stata_version": "`c(stata_version)'""'

* ------------------------------------------------------------------ helpers
capture program drop tewrite
program tewrite
    * tewrite fh name : writes e(b)[1,1], its SE, e(N) and the command line
    args fh name
    matrix b = e(b)
    matrix V = e(V)
    local cmd = subinstr(`"`e(cmdline)'"', `"""', "'", .)
    jnum est b[1,1]
    jnum se sqrt(V[1,1])
    file write `fh' "," _n `"  "`name'": {"est": `est', "se": `se', "N": "' (e(N)) ///
        `", "cmd": "`cmd'"}"'
end

capture program drop matches_out
program matches_out
    * matches_out name stub : id, d, and the ids of the matches (stub1, ...)
    args name stub
    preserve
    capture unab mv : `stub'*
    if _rc == 0 {
        foreach v of local mv {
            replace `v' = id[`v'] if `v' < .        // observation number -> id
        }
        keep id d `mv'
        keep if `: word 1 of `mv'' < .
        export delimited using "tests/data/ref_teffects_`name'_matches.csv", replace
    }
    restore
    capture drop `stub'*
end

* ------------------------------------------------------- the fitted scores
foreach link in logit probit {
    `link' d x1 x2 x3
    matrix g = e(b)
    predict double ps_`link', pr
    file write `fh' "," _n `"  "`link'_coef": {"'
    local names : colnames g
    local k = colsof(g)
    forvalues j = 1/`k' {
        local nm : word `j' of `names'
        if "`nm'" == "_cons" local nm "Intercept"
        if `j' > 1 file write `fh' ", "
        jnum c g[1,`j']
        file write `fh' `""`nm'": `c'"'
    }
    file write `fh' "}"
}
export delimited id ps_logit ps_probit using "tests/data/ref_teffects_pscore.csv", replace

* ------------------------------------------------- part 1: teffects psmatch
* name            stat  link    nn  vce
local ps_cases ///
    psm_atet_logit_m1      atet  logit   1  robust ///
    psm_ate_logit_m1       ate   logit   1  robust ///
    psm_atet_logit_m4      atet  logit   4  robust ///
    psm_ate_probit_m1      ate   probit  1  robust ///
    psm_atet_logit_m1_iid  atet  logit   1  iid
local n : word count `ps_cases'
forvalues c = 1(5)`n' {
    local name : word `c' of `ps_cases'
    local stat : word `=`c'+1' of `ps_cases'
    local link : word `=`c'+2' of `ps_cases'
    local nn   : word `=`c'+3' of `ps_cases'
    local vce  : word `=`c'+4' of `ps_cases'
    teffects psmatch (y) (d x1 x2 x3, `link'), `stat' nneighbor(`nn') vce(`vce') ///
        generate(_m)
    tewrite `fh' `name'
    matches_out `name' _m
    * the same matching with the fitted score taken as known (no 2016 term)
    teffects nnmatch (y ps_`link') (d), `stat' nneighbor(`nn') vce(`vce') generate(_m)
    tewrite `fh' `name'_known
    matches_out `name'_known _m
}

* ------------------------------------------------- part 2: teffects nnmatch
* on the true score within exact strata (as tests/test_effects.py's R cases)
teffects nnmatch (y ps) (d), atet nneighbor(1) ematch(r i) vce(iid) generate(_m)
tewrite `fh' nnm_atet_ps_m1_iid
matches_out nnm_atet_ps_m1_iid _m
teffects nnmatch (y ps) (d), atet nneighbor(4) ematch(r i) vce(robust, nn(4)) generate(_m)
tewrite `fh' nnm_atet_ps_m4_vc4
matches_out nnm_atet_ps_m4_vc4 _m
teffects nnmatch (y ps) (d), ate nneighbor(2) ematch(r i) vce(iid) generate(_m)
tewrite `fh' nnm_ate_ps_m2_iid
matches_out nnm_ate_ps_m2_iid _m
teffects nnmatch (y ps) (d), atet nneighbor(1) ematch(r i) vce(iid) ///
    biasadj(x1 x2 x3) generate(_m)
tewrite `fh' nnm_atet_ps_biasadj_iid
matches_out nnm_atet_ps_biasadj_iid _m
* Mahalanobis on x1-x3 (teffects: the full-sample covariance)
teffects nnmatch (y x1 x2 x3) (d), atet nneighbor(1) ematch(r i) metric(maha) ///
    vce(iid) generate(_m)
tewrite `fh' nnm_atet_maha_m1_iid
matches_out nnm_atet_maha_m1_iid _m
* a caliper, in teffects' distance units (recorded, not assumed). In the
* first run teffects stopped with "no exact matches for observation 7328"
* (a treated unit whose nearest control is 0.0057 away on ps, 0.049 SD) and
* osample() did not flag it, so this case only records the outcome: the
* estimate if teffects runs, else its return code. It must not stop the
* script.
capture drop _viol
capture noisily teffects nnmatch (y ps) (d), atet nneighbor(1) ematch(r i) vce(iid) ///
    caliper(0.05) osample(_viol)
local rc = _rc
if `rc' == 0 {
    tewrite `fh' nnm_atet_ps_cal_iid
}
local nviol "null"
capture confirm variable _viol
if _rc == 0 {
    count if _viol
    local nviol = r(N)
}
file write `fh' "," _n `"  "nnm_atet_ps_cal_iid_rc": `rc', "nnm_atet_ps_cal_iid_n_osample": `nviol'"' _n "}" _n
file close `fh'
