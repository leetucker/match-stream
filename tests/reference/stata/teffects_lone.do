* How teffects nnmatch handles a unit alone in its arm within its exact
* stratum when the robust variance needs that unit's conditional variance
* (tests/test_teffects_lone.py; match-stream's effect(var_fallback=)).
*
* Run from the repository root:   stata -b do tests/reference/stata/teffects_lone.do
* Writes, in tests/data/:
*   ref_teffects_lone.json            per case: return code, and when it ran,
*                                     estimate, SE, N and the command
*   ref_teffects_lone_<case>_matches.csv   id, d and the ids of the matches
*
* Two variants of tests/data/ref_data.csv, matched on the true score (ps)
* within the exact cells r, i, as teffects_match.do's nnmatch cases:
*   lone_treated  the cell r = 0, i = 0 keeps only its first treated unit
*                 (lowest id): the ATE needs that unit's conditional variance
*                 (it is the match of every control in the cell); teffects
*                 has no ATC
*   lone_control  the same cell keeps only its first control: the ATET needs
*                 that control's variance (it is the match of every treated
*                 unit in the cell)
* Each with vce(robust, nn(1)) and nn(2) (match-stream's var_calc 1, 2) and
* vce(iid) as a baseline. Possible outcomes, which the test tells apart: an
* error (the return code is recorded and the run goes on), a variance with
* the unit's conditional variance set to 0 (R Matching's), or one from its
* nearest same-arm units outside the cell (match-stream's
* var_fallback="nearest").
version 15
clear all
set more off
do tests/reference/stata/_json.do
tempname fh
file open `fh' using "tests/data/ref_teffects_lone.json", write replace
file write `fh' "{" _n
file write `fh' `"  "stata_version": "`c(stata_version)'""'

capture program drop lonerun
program lonerun
    * lonerun fh name : runs the teffects command that follows the colon
    gettoken fh 0 : 0
    gettoken name 0 : 0
    gettoken colon 0 : 0
    capture noisily `0' generate(_m)
    local rc = _rc
    if `rc' {
        file write `fh' "," _n `"  "`name'": {"rc": `rc'}"'
        capture drop _m*
        exit
    }
    matrix b = e(b)
    matrix V = e(V)
    local cmd = subinstr(`"`e(cmdline)'"', `"""', "'", .)
    jnum est b[1,1]
    jnum se sqrt(V[1,1])
    file write `fh' "," _n `"  "`name'": {"rc": 0, "est": `est', "se": `se', "N": "' ///
        (e(N)) `", "cmd": "`cmd'"}"'
    preserve
    capture unab mv : _m*
    if _rc == 0 {
        foreach v of local mv {
            replace `v' = id[`v'] if `v' < .        // observation number -> id
        }
        keep id d `mv'
        keep if `: word 1 of `mv'' < .
        export delimited using "tests/data/ref_teffects_lone_`name'_matches.csv", replace
    }
    restore
    capture drop _m*
end

foreach variant in treated control {
    import delimited using "tests/data/ref_data.csv", clear asdouble
    local arm = cond("`variant'" == "treated", 1, 0)
    sort id
    * the cell's units of that arm, in id order; keep the first
    bysort r i d (id): gen long _k = _n
    drop if r == 0 & i == 0 & d == `arm' & _k > 1
    drop _k
    sort id                                      // generate() stores observation numbers
    count if r == 0 & i == 0 & d == `arm'
    assert r(N) == 1
    local stat = cond("`variant'" == "treated", "ate", "atet")
    foreach v in "robust, nn(1)" "robust, nn(2)" "iid" {
        local tag = cond("`v'" == "iid", "iid", "vc" + substr("`v'", -2, 1))
        lonerun `fh' lone_`variant'_`stat'_`tag' : ///
            teffects nnmatch (y ps) (d), `stat' nneighbor(1) ematch(r i) vce(`v')
    }
}
file write `fh' _n "}" _n
file close `fh'
