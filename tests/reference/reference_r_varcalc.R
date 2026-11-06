# Reference values from R Matching for Abadie-Imbens standard errors with
# conditional variances (Var.calc > 0) when matching on covariates, on
# tests/data/ref_data.csv.
#
#   Rscript tests/reference/reference_r_varcalc.R
#
# X = (x1, x2, x3, r, i) with exact matching on r and i and Matching's
# default metric (Weight = 1: each covariate scaled by its standard
# deviation), which is match-stream's distance = "ivariance". Ties kept,
# distance.tolerance = 0, as reference_r.R. Writes tests/data/ref_r_varcalc.json
# (estimate, SE, matched counts per case) and tests/data/ref_matching_varcalc_pairs.csv
# (Matching's pairs for the ATT, M = 1). Needs Matching (4.10-15 when written).
suppressPackageStartupMessages({ library(Matching); library(jsonlite) })

here <- dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))))
data_dir <- file.path(here, "..", "data")
df <- read.csv(file.path(data_dir, "ref_data.csv"))
X <- cbind(df$x1, df$x2, df$x3, df$r, df$i)
ex <- c(FALSE, FALSE, FALSE, TRUE, TRUE)
out <- list(versions = list(R = R.version.string, Matching = as.character(packageVersion("Matching"))))
cases <- list(
  att_m1_v0 = list(estimand = "ATT", M = 1, Var.calc = 0),
  att_m1_v2 = list(estimand = "ATT", M = 1, Var.calc = 2),
  att_m2_v4 = list(estimand = "ATT", M = 2, Var.calc = 4),
  atc_m2_v3 = list(estimand = "ATC", M = 2, Var.calc = 3),
  ate_m1_v2 = list(estimand = "ATE", M = 1, Var.calc = 2)
)
for (name in names(cases)) {
  a <- cases[[name]]
  m <- Match(Y = df$y, Tr = df$d, X = X, exact = ex, M = a$M, estimand = a$estimand,
             Var.calc = a$Var.calc, replace = TRUE, ties = TRUE, distance.tolerance = 0)
  if (name == "att_m1_v0") {
    write.csv(data.frame(treated = df$id[m$index.treated], control = df$id[m$index.control],
                         weight = m$weights),
              file.path(data_dir, "ref_matching_varcalc_pairs.csv"), row.names = FALSE)
  }
  out[[name]] <- list(est = unname(m$est[1, 1]), se = m$se, nobs = m$nobs, wnobs = m$wnobs,
                      n_dropped = length(m$index.dropped))
}
writeLines(toJSON(out, auto_unbox = TRUE, digits = 15, pretty = TRUE),
           file.path(data_dir, "ref_r_varcalc.json"))
cat("wrote", file.path(data_dir, "ref_r_varcalc.json"), "\n")
