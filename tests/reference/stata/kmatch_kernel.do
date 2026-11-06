* Reference values from Stata's kmatch for match-stream's kernel and ridge
* matching (tests/test_kernel.py::test_kmatch_reference).
*
* Run from the repository root:   stata -b do tests/reference/stata/kmatch_kernel.do
* Needs kmatch (1.1.6 was read when match-stream was written) and moremata:
*   ssc install kmatch
*   ssc install moremata
* Writes tests/data/ref_kmatch_kernel.json (estimates, standard errors,
* bandwidths) and tests/data/ref_kmatch_<case>_weights.csv (id, d, weight).
* All cases match on the true score (variable ps) within exact strata r, i.
version 15
clear all
set more off
import delimited using "tests/data/ref_data.csv", clear asdouble   // full precision
do tests/reference/stata/_json.do

tempname fh
file open `fh' using "tests/data/ref_kmatch_kernel.json", write replace
file write `fh' "{" _n
file write `fh' `"  "stata_version": "`c(stata_version)'","' _n

capture program drop kmrun
program kmrun
    args fh name last
    gettoken opts 0 : 0, parse(",")
    syntax [, *]
    * the rest of the line after the comma are kmatch options
    local kmopts `options'
    capture drop _W
    kmatch ps d x1 x2 x3 (y), pscore(ps) ematch(r i) `kmopts' wgenerate(_W)
    matrix b = e(b)
    matrix V = e(V)
    matrix bw = e(bwidth)
    jnum est b[1,1]
    jnum se sqrt(V[1,1])
    file write `fh' `"  "`name'": {"est": `est', "se": `se', "N": "' (e(N)) `", "bwidth": ["'
    forvalues j = 1/`=colsof(bw)' {
        if `j' > 1 file write `fh' ", "
        jnum h bw[1,`j']
        file write `fh' "`h'"
    }
    file write `fh' `"], "options": "`kmopts'"}"'
    if "`last'" == "" file write `fh' ","
    file write `fh' _n
    preserve
    keep if _W < . & _W != 0
    keep id d _W
    export delimited using "tests/data/ref_kmatch_`name'_weights.csv", replace
    restore
end

* name             last?  kmatch options
kmrun `fh' kernel_att_pm     "" , att kernel(epan)
kmrun `fh' ridge_att_pm      "" , att kernel(epan) ridge
kmrun `fh' kernel_atc_fixed  "" , atc kernel(biweight) bwidth(0.01)
kmrun `fh' ll_att_fixed      "" , att kernel(epan) ridge(0) bwidth(0.02)
kmrun `fh' kernel_att_cv     "" , att kernel(epan) bwidth(cv y)
kmrun `fh' kernel_att_cvw    last , att kernel(epan) bwidth(cv y, weighted)
file write `fh' "}" _n
file close `fh'
