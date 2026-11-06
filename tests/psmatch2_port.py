"""A line-by-line Python port of psmatch2 4.0.12's nearest-neighbor search
on a score (its Mata functions match_pscore, match2, match_ties and
next_unmatched), used by tests/test_psmatch2.py to say what psmatch2 does
where match-stream does something else.

psmatch2 sorts the data by arm and score (descending score with
`descending`; ties in the score keep data order), puts the controls first,
and walks the treated units in that order with a pointer into the sorted
controls. For each treated unit the pointer steps forward while the
distance falls. Without replacement the pointer then moves to the next free
control, so a free control below it is never looked at again: the result is
not greedy nearest-available matching (MatchIt's, and match-stream's
match_order "smallest" or "largest"), and on tests/data/ref_data.csv about a
quarter of the treated units get a different control. With replacement the
search finds each unit's nearest controls exactly. The caliper is strict
(distance < caliper). Positions are 1-based, as in Mata.
"""

import numpy as np
import polars as pl


def psmatch2_pairs(data: pl.DataFrame, *, score="ps", treat="d", id="id", n_matches=1,
                   caliper=np.inf, noreplacement=False, keep_ties=False, descending=False):
    """Each treated unit's matched control ids, as psmatch2 makes them
    (ATT matching; units with no match within the caliper are absent)."""
    key = -pl.col(score) if descending else pl.col(score)
    s = data.with_columns(key.alias("__key"), pl.int_range(pl.len()).alias("__order")) \
        .sort([treat, "__key", "__order"])
    ps = np.concatenate([[np.nan], s[score].to_numpy()])
    ids = np.concatenate([[0], s[id].to_numpy()])
    n = s.height
    n0 = int((s[treat] == 0).sum())
    i0, i1, j0, j1 = n0 + 1, n, 1, n0
    taken = np.zeros(n + 2, bool)

    def next_unmatched(obs, j, forward):
        while True:
            j = j + 1 if forward else j - 1
            if obs != j and not noreplacement:
                return j
            if j < 1 or j > n or not (obs == j or taken[j]):
                return j

    out = {}
    i, jmatch, forward = i0, j0, True
    while i <= i1 and j0 <= jmatch <= j1:
        j = jmatch
        dif1 = abs(ps[i] - ps[j])
        while j0 <= j < j1:
            j = next_unmatched(i, j, forward)
            if j < j0 or j > j1:
                break
            dif0, dif1 = dif1, abs(ps[i] - ps[j])
            if dif1 > dif0:
                j = j1
            if dif1 < dif0:
                jmatch = j
        found = abs(ps[i] - ps[jmatch]) < caliper
        if found:
            taken[jmatch] = True
            got = [jmatch]
            if keep_ties:                  # match_ties: every control as near, forward
                d0 = abs(ps[i] - ps[jmatch])
                q = next_unmatched(i, jmatch, forward)
                while j0 <= q <= j1 and abs(ps[i] - ps[q]) <= d0:
                    got.append(q)
                    taken[q] |= noreplacement
                    q = next_unmatched(i, q, forward)
            if n_matches > 1:                   # match2: grow outward, the lower side wins ties
                kk, p0, p1, jm = 1, jmatch - 1, jmatch + 1, jmatch
                d0 = d1 = np.inf
                while kk < n_matches:
                    if p0 >= j0:
                        d0 = abs(ps[i] - ps[p0])
                    if p1 <= j1:
                        d1 = abs(ps[i] - ps[p1])
                    if (d0 <= d1 and p0 >= j0) or (p0 >= j0 and p1 > j1):
                        jm, p0 = p0, p0 - 1
                    elif p1 <= j1:
                        jm, p1 = p1, p1 + 1
                    else:
                        break
                    if abs(ps[i] - ps[jm]) < caliper:
                        got.append(jm)
                        kk += 1
                    else:
                        break
            out[int(ids[i])] = [int(ids[q]) for q in got]
        if jmatch == j1 and i < i1:
            forward = False
        if noreplacement and found:
            jmatch = next_unmatched(i, jmatch, forward)
        i += 1
    return out
