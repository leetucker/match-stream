# Effects and standard errors

The choices for inference after matching, side by side, on one data set
(4 matches with replacement, ties kept).

```{literalinclude} ../../examples/effects.py
:language: python
:start-at: import
```

Output:

```text
                                          estimate      s.e.
AI, homoskedastic                           0.9937    0.0117
AI, n_variance_matches=4, score taken as known    0.9937    0.0117
AI, n_variance_matches=4, score corrected    0.9937    0.0093
AI, bias-adjusted                           0.9958    0.0085
AI, ATE                                     0.9969    0.0116
IF, by unit                                 0.9937    0.0131
IF, subclass + id                           0.9937    0.0122
without replacement, CRV1 pair              1.0086    0.0122
```

**Notes.**

- **Abadie-Imbens (`vcov="ai"`)** is the default with replacement.
  - `n_variance_matches=J` estimates each unit's conditional variance from its J
    nearest same-arm units (R Matching's `Var.calc`).
  - `bias_adjustment=` applies Abadie and Imbens's (2011) regression adjustment
    (`True`: on the matching covariates).
- **The estimated score.** When match-stream fitted the score
  (`distance="ps"`), `n_variance_matches >= 1` also applies Abadie and Imbens's
  (2016) correction (as Stata's `teffects psmatch`). It lowers
  the standard error here from 0.0117 to 0.0093; `score_correction=False`
  takes the score as known. With `n_variance_matches=0` there is no correction,
  and a warning says so.
- **The ATE** matches both ways: every unit is focal once.
- **Influence-function SEs** hold the matching weights fixed. Two forms:
  - `vcov="if"`: by unit.
  - `vcov={"CRV1": "a+b"}`: clustered, on the subclass and/or source columns.
  - They equal the HC1/CRV1 standard errors of the weighted regression
    `y ~ treated` on the matched units with the matching weights, the
    usual analysis after matching without Abadie-Imbens.
- **Without replacement** the default clusters on the pair (Abadie and
  Spiess 2022).
