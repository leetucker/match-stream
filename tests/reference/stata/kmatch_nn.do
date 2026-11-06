* Reference values from Stata's kmatch for match-stream's nearest-neighbor
* matching (tests/test_kmatch_nn.py).
*
* Run from the repository root:   stata -b do tests/reference/stata/kmatch_nn.do
* Needs kmatch and moremata (ssc install kmatch; ssc install moremata).
* Writes, in tests/data/:
*   ref_kmatch_nn.json             per case: estimate, SE, N, options
*   ref_kmatch_nn_<case>_ids.csv   id, d and the ids of the matches
*                                  (idgenerate(), ties included)
*   ref_kmatch_nn_<case>_weights.csv  id, d and kmatch's weight
* Matching on the true score (pscore(ps)) within exact strata r, i, as the
* MatchIt and R Matching references. Calipers (bwidth()) are in score units.
version 15
clear all
set more off
import delimited using "tests/data/ref_data.csv", clear asdouble   // full precision
do tests/reference/stata/_json.do
sort id

tempname fh
file open `fh' using "tests/data/ref_kmatch_nn.json", write replace
file write `fh' "{" _n
file write `fh' `"  "stata_version": "`c(stata_version)'""'

capture program drop kmnn
program kmnn
    * kmnn fh name kind , kmatch options     (kind: ps or md)
    gettoken fh 0 : 0
    gettoken name 0 : 0
    gettoken kind 0 : 0
    syntax [, *]
    local kmopts `options'
    capture drop _W
    capture drop _ID_*
    if "`kind'" == "md" {
        kmatch md d x1 x2 x3 (y), ematch(r i) `kmopts' wgenerate(_W) ///
            idgenerate(_ID_) idvar(id)
    }
    else {
        kmatch ps d x1 x2 x3 (y), pscore(ps) ematch(r i) `kmopts' wgenerate(_W) ///
            idgenerate(_ID_) idvar(id)
    }
    matrix b = e(b)
    matrix V = e(V)
    jnum est b[1,1]
    jnum se sqrt(V[1,1])
    file write `fh' "," _n `"  "`name'": {"est": `est', "se": `se', "N": "' (e(N)) ///
        `", "options": "`kind' `kmopts'"}"'
    preserve
    capture unab ids : _ID_*
    if _rc == 0 {
        keep if `: word 1 of `ids'' < .
        keep id d `ids'
        export delimited using "tests/data/ref_kmatch_nn_`name'_ids.csv", replace
    }
    restore
    preserve
    keep if _W < . & _W != 0
    keep id d _W
    export delimited using "tests/data/ref_kmatch_nn_`name'_weights.csv", replace
    restore
end

* with replacement (kmatch keeps every control tied at the k-th distance)
kmnn `fh' nn_att_m1      ps , att nn(1)
kmnn `fh' nn_att_m3      ps , att nn(3)
kmnn `fh' nn_atc_m2      ps , atc nn(2)
kmnn `fh' nn_ate_m1      ps , ate nn(1)
kmnn `fh' nn_att_m1_cal  ps , att nn(1) bwidth(0.01)
* k = 3 with a caliper: kmatch drops a treated unit with fewer than 3
* matches inside it (MatchIt and match-stream keep it with fewer)
kmnn `fh' nn_att_m3_cal  ps , att nn(3) bwidth(0.01)
kmnn `fh' nn_att_m3_cal_keepall ps , att nn(3) bwidth(0.01) keepall
* without replacement: greedy in ascending pair distance (m.order closest)
set sortseed 20261102
kmnn `fh' wor_att_m1     ps , att nn(1) wor
kmnn `fh' wor_att_m1_cal ps , att nn(1) wor bwidth(0.01)
* k = 2 without replacement: kmatch makes one pass in distance order, each
* treated unit taking controls until it has 2; MatchIt (and match-stream)
* match in rounds
kmnn `fh' wor_att_m2     ps , att nn(2) wor
* Mahalanobis (kmatch md's default metric), recorded for information
kmnn `fh' md_att_m1      md , att nn(1)
file write `fh' _n "}" _n
file close `fh'
