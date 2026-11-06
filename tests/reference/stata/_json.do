* Shared helper for the reference scripts: jnum <local> <expression> puts
* the expression's value in <local> as a JSON number: 17 significant
* digits, with the leading zero Stata's %g omits (".5" is not JSON), and
* missing as null.
capture program drop jnum
program jnum
    gettoken lname 0 : 0
    local v = strtrim(strofreal(`0', "%21.17g"))
    if "`v'" == "." local v "null"
    else if substr("`v'", 1, 1) == "." local v "0`v'"
    else if substr("`v'", 1, 2) == "-." local v = "-0" + substr("`v'", 2, .)
    c_local `lname' `v'
end
