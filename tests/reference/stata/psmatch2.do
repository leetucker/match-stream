* Reference values from Stata's psmatch2 for match-stream's nearest-neighbor
* matching (tests/test_psmatch2.py).
*
* Run from the repository root:   stata -b do tests/reference/stata/psmatch2.do
* Needs psmatch2 (ssc install psmatch2; written against 4.0.12). Writes, in
* tests/data/:
*   ref_psmatch2.json               per case: ATT and its SE, ATE and ATU (with
*                                   ate), matched treated, options, return code
*   ref_psmatch2_<case>.csv         id, d, _support, _weight and the ids of
*                                   the matches (psmatch2's _n1.._nk, mapped
*                                   from its _id to our id)
* All cases use the true score (pscore(ps)); psmatch2 has no exact strata.
* From psmatch2.ado: it sorts by arm and score (descending score with
* `descending`), not data order, so data order only breaks ties in the
* score (the data have none). tests/psmatch2_port.py ports its search.
version 15
clear all
set more off
import delimited using "tests/data/ref_data.csv", clear asdouble
do tests/reference/stata/_json.do
sort id

tempname fh
file open `fh' using "tests/data/ref_psmatch2.json", write replace
file write `fh' "{" _n
file write `fh' `"  "stata_version": "`c(stata_version)'""'

capture program drop p2run
program p2run
    * p2run fh name k , psmatch2 options
    * A failing case is recorded with its return code and the run goes on.
    gettoken fh 0 : 0
    gettoken name 0 : 0
    gettoken k 0 : 0
    syntax [, *]
    local opts `options'
    preserve
    capture noisily psmatch2 d, outcome(y) pscore(ps) `opts'
    local rc = _rc
    if `rc' {
        file write `fh' "," _n `"  "`name'": {"rc": `rc', "options": "`opts'"}"'
        restore
        exit
    }
    * psmatch2 returns r(att) and r(seatt) for the last outcome; with ate
    * also r(ate) and r(atu) (their SEs only with ai())
    foreach r in att seatt ate atu {
        jnum `r' r(`r')
    }
    capture confirm variable _support
    if _rc gen byte _support = 1
    count if d == 1 & _support == 1 & _weight < .
    local nmatched = r(N)
    file write `fh' "," _n `"  "`name'": {"rc": 0, "est": `att', "se": `seatt', "ate": `ate', "atu": `atu', "matched_treated": `nmatched', "options": "`opts'"}"'
    * psmatch2's _id numbers the observations; _n1.._nk hold the _id of
    * each unit's matches (one per slot, not every tied control)
    sort _id
    assert _id == _n
    forvalues j = 1/`k' {
        capture confirm variable _n`j'
        if _rc == 0 {
            gen long match`j' = id[_n`j'] if _n`j' < .
        }
    }
    sort id
    keep id d _support _weight match*
    export delimited using "tests/data/ref_psmatch2_`name'.csv", replace nolabel
    restore
end

* with replacement
p2run `fh' rep_n1            1 , neighbor(1)
p2run `fh' rep_n1_ties       1 , neighbor(1) ties
p2run `fh' rep_n3_cal        3 , neighbor(3) caliper(0.01)
p2run `fh' rep_n1_common     1 , neighbor(1) common
p2run `fh' rep_n1_ate        1 , neighbor(1) ate
* Abadie-Imbens SEs, own-arm variance from 1 neighbor: population (AI 2006,
* as R Matching) and psmatch2's default sample (conditional) version
p2run `fh' rep_n1_ai1        1 , neighbor(1) ai(1) population
p2run `fh' rep_n1_ai1_sample 1 , neighbor(1) ai(1)
* without replacement (1:1 only in psmatch2): ascending score (the default),
* descending score, and with a caliper
p2run `fh' wor_asc           1 , noreplacement
p2run `fh' wor_desc          1 , noreplacement descending
p2run `fh' wor_cal           1 , noreplacement caliper(0.005)
* Mahalanobis on x1-x3 (pooled within-arm covariance, as MatchIt)
p2run `fh' mahal             1 , mahalanobis(x1 x2 x3)
file write `fh' _n "}" _n
file close `fh'
