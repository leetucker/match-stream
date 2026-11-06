# Reference standard errors for effect(vce=...) (influence function, weights
# held fixed): MatchIt's recommended analysis, a weighted regression of the
# outcome on the treatment with sandwich standard errors, on
# tests/data/ref_data.csv.
#
#   Rscript tests/reference/reference_r_if.R
#
# Writes tests/data/ref_r_if.json: per spec, the vce argument to pass to
# effect(), the estimate and the standard error. tests/test_effects_if.py
# reads it (and skips while it does not exist).
#
# The matchit() calls are those of reference_r.R (same names). Conventions:
# sandwich::vcovCL's defaults (type "HC1", cadjust = TRUE), i.e.
# G/(G-1) * (N-1)/(N-K), which is match-stream's. For two-way clustering
# sandwich scales each term by its own G/(G-1) while match-stream (as
# fixest and hdfe-stream) uses the smallest G for every term, so the
# two-way reference is computed with fixest when it is installed.
# Needs: MatchIt 4.8.2, sandwich, jsonlite; optionally fixest.
suppressPackageStartupMessages({ library(MatchIt); library(sandwich); library(jsonlite) })

here <- dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))))
data_dir <- file.path(here, "..", "data")
df <- read.csv(file.path(data_dir, "ref_data.csv"))
rownames(df) <- df$id
df$rf <- factor(df$r); df$if_ <- factor(df$i)
cal3 <- c(0.2, x1 = 0.25, x2 = 0.25)
std3 <- c(TRUE, FALSE, FALSE)
out <- list(versions = list(R = R.version.string, MatchIt = as.character(packageVersion("MatchIt")),
                            sandwich = as.character(packageVersion("sandwich"))))
se_d <- function(V) sqrt(V["d", "d"])

# 1-NN without replacement: lm on match.data(), clustered by subclass (pairs)
m <- matchit(d ~ x1 + x2 + x3, data = df, method = "nearest", distance = df$ps, replace = FALSE,
             m.order = "largest", caliper = cal3, std.caliper = std3, exact = ~ r + i)
md <- match.data(m)
fit <- lm(y ~ d, data = md, weights = weights)
out$nn_ps_wor_largest <- list(desc = "lm(y ~ d, match.data, weights), vcovCL(~subclass)",
                              vce = list(CRV1 = "subclass"), est = unname(coef(fit)["d"]),
                              se = se_d(vcovCL(fit, cluster = ~subclass)))

# 1-NN with replacement: lm on get_matches() (a row per unit and subclass)
m <- matchit(d ~ x1 + x2 + x3, data = df, method = "nearest", distance = df$ps, replace = TRUE,
             caliper = 0.2, exact = ~ r + i)
gm <- get_matches(m, id = "unit")
fit <- lm(y ~ d, data = gm, weights = weights)
out$nn_ps_rep <- list(desc = "lm(y ~ d, get_matches, weights), vcovCL(~subclass)",
                      vce = list(CRV1 = "subclass"), est = unname(coef(fit)["d"]),
                      se = se_d(vcovCL(fit, cluster = ~subclass)))
if (requireNamespace("fixest", quietly = TRUE)) {
  ff <- fixest::feols(y ~ d, data = gm, weights = ~weights, cluster = ~subclass + id)
  out$nn_ps_rep_2way <- list(desc = "fixest::feols(y ~ d, get_matches, weights), cluster subclass + id",
                             vce = list(CRV1 = "subclass+id"), est = unname(coef(ff)["d"]),
                             se = unname(fixest::se(ff)["d"]),
                             fixest = as.character(packageVersion("fixest")))
}

# CEM: lm on match.data(), HC1 by unit (MatchIt advises robust, not cell-clustered, SEs)
m <- matchit(d ~ x1 + x2 + x3 + rf + if_, data = df, method = "cem",
             cutpoints = list(x1 = 5, x2 = 5, x3 = 5))
md <- match.data(m)
fit <- lm(y ~ d, data = md, weights = weights)
out$cem5 <- list(desc = "lm(y ~ d, match.data, weights), vcovHC HC1", vce = "if",
                 est = unname(coef(fit)["d"]), se = se_d(vcovHC(fit, type = "HC1")))

writeLines(toJSON(out, auto_unbox = TRUE, digits = 15, pretty = TRUE),
           file.path(data_dir, "ref_r_if.json"))
cat("wrote", file.path(data_dir, "ref_r_if.json"), "\n")
