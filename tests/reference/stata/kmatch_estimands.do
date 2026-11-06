* Reference values from Stata's kmatch for match-stream's joint vcov of several estimands
* (ate att atc nate po in one call; influence-function SEs; tests/test_several_estimands.py).
*
* Run from the repository root:   stata -b do tests/reference/stata/kmatch_estimands.do
* Needs kmatch and moremata (ssc install). Writes tests/data/ref_kmatch_estimands.json:
* per case, the return code, e(b) and e(V) (row by row) for the outcome y
* (and y2 = 0.5 y + x3), estimated in one kmatch call with all five estimands.
* As kmatch_kernel.do and kmatch_nn.do: the true score (pscore(ps)), exact
* cells r, i.
version 15
clear all
set more off
import delimited using "tests/data/ref_data.csv", clear asdouble
gen double y2 = 0.5 * y + x3
do tests/reference/stata/_json.do
sort id
tempname fh
file open `fh' using "tests/data/ref_kmatch_estimands.json", write replace
file write `fh' "{" _n
file write `fh' `"  "stata_version": "`c(stata_version)'""'

capture program drop sevrun
program sevrun
    gettoken fh 0 : 0
    gettoken name 0 : 0
    syntax [, *]
    local kmopts `options'
    capture noisily kmatch ps d x1 x2 x3 (y y2), pscore(ps) ematch(r i) `kmopts'
    local rc = _rc
    if `rc' {
        file write `fh' "," _n `"  "`name'": {"rc": `rc', "options": "`kmopts'"}"'
        exit
    }
    matrix b = e(b)
    matrix V = e(V)
    local cn : colfullnames b
    file write `fh' "," _n `"  "`name'": {"rc": 0, "options": "`kmopts'", "N": "' (e(N)) ///
        `", "names": ""' `"`cn'"' `"", "b": ["'
    forvalues j = 1/`=colsof(b)' {
        if `j' > 1 file write `fh' ", "
        jnum v b[1,`j']
        file write `fh' "`v'"
    }
    file write `fh' "], " `""V": ["'
    forvalues i = 1/`=rowsof(V)' {
        if `i' > 1 file write `fh' ", "
        file write `fh' "["
        forvalues j = 1/`=colsof(V)' {
            if `j' > 1 file write `fh' ", "
            jnum v V[`i',`j']
            file write `fh' "`v'"
        }
        file write `fh' "]"
    }
    file write `fh' "]}"
end

* name           kmatch options
sevrun `fh' nn_all     , ate att atc nate po nn(1)
sevrun `fh' kernel_all , ate att atc nate po kernel(epan) bwidth(0.02)
sevrun `fh' ridge_all  , ate att atc nate po kernel(epan) ridge bwidth(0.02)
file write `fh' _n "}" _n
file close `fh'
