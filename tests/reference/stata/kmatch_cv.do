* Reference values from Stata's kmatch for match-stream's cross-validated
* bandwidth without an outcome (kmatch's bwidth(cv) with no variable: the
* criterion is the balance of the score's mean; tests/test_kernel.py).
*
* Run from the repository root:   stata -b do tests/reference/stata/kmatch_cv.do
* Needs kmatch and moremata (ssc install). Writes, in tests/data/:
*   ref_kmatch_cv.json          per case: return code, estimate, SE, the
*                               bandwidth, and kmatch's search (e(cv) and
*                               e(cv_treated) where present: rows of
*                               bandwidth, criterion, in evaluation order)
*   ref_kmatch_cv_<case>_weights.csv   id, d, _W
* As kmatch_kernel.do: the true score (pscore(ps)), exact cells r, i.
version 15
clear all
set more off
import delimited using "tests/data/ref_data.csv", clear asdouble
do tests/reference/stata/_json.do
sort id
tempname fh
file open `fh' using "tests/data/ref_kmatch_cv.json", write replace
file write `fh' "{" _n
file write `fh' `"  "stata_version": "`c(stata_version)'""'

capture program drop cvmat
program cvmat
    * cvmat fh key matname : writes  , "key": [[h, mise], ...]
    args fh key M
    file write `fh' `", "`key'": ["'
    forvalues i = 1/`=rowsof(`M')' {
        if `i' > 1 file write `fh' ", "
        jnum h `M'[`i', 1]
        jnum v `M'[`i', 2]
        file write `fh' "[`h', `v']"
    }
    file write `fh' "]"
end

capture program drop cvrun
program cvrun
    gettoken fh 0 : 0
    gettoken name 0 : 0
    syntax [, *]
    local kmopts `options'
    capture drop _W
    capture noisily kmatch ps d x1 x2 x3 (y), pscore(ps) ematch(r i) `kmopts' wgenerate(_W)
    local rc = _rc
    if `rc' {
        file write `fh' "," _n `"  "`name'": {"rc": `rc', "options": "`kmopts'"}"'
        exit
    }
    matrix b = e(b)
    matrix V = e(V)
    matrix bw = e(bwidth)
    jnum est b[1,1]
    jnum se sqrt(V[1,1])
    file write `fh' "," _n `"  "`name'": {"rc": 0, "est": `est', "se": `se', "N": "' (e(N)) ///
        `", "options": "`kmopts'", "bwidth": ["'
    forvalues j = 1/`=colsof(bw)' {
        if `j' > 1 file write `fh' ", "
        jnum h bw[1,`j']
        file write `fh' "`h'"
    }
    file write `fh' "]"
    foreach m in cv cv_treated {
        capture confirm matrix e(`m')
        if _rc == 0 {
            tempname M
            matrix `M' = e(`m')
            cvmat `fh' `m' `M'
        }
    }
    file write `fh' "}"
    preserve
    keep if _W < . & _W != 0
    keep id d _W
    export delimited using "tests/data/ref_kmatch_cv_`name'_weights.csv", replace
    restore
end

* name               kmatch options
cvrun `fh' kernel_att_cvs     , att kernel(epan) bwidth(cv)
cvrun `fh' ridge_att_cvs      , att kernel(epan) ridge bwidth(cv)
cvrun `fh' kernel_atc_cvs     , atc kernel(epan) bwidth(cv)
cvrun `fh' kernel_att_cvs_tri , att kernel(triangle) bwidth(cv)
* on a grid: every point evaluated, with the limit factor
cvrun `fh' kernel_att_cvs_grid , att kernel(epan) bwidth(cv, grid(0.01 0.02 0.04 0.08 0.16))
file write `fh' _n "}" _n
file close `fh'
