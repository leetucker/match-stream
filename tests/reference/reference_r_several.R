# R Matching's Abadie-Imbens SEs for two outcomes and their sum, on the
# reference data: the reference for match-stream's joint vcov of several
# outcomes (tests/test_several_outcomes.py). No package reports the
# cross-outcome covariance; through the identity
#   Cov = [V(y1 + y2) - V(y1) - V(y2)] / 2
# the three SEs below determine it.
#
#   Rscript tests/reference/reference_r_several.R      (from the repository root)
# Writes tests/data/ref_r_several.json. Needs Matching and jsonlite.
#
# y2 = 0.5 y + x3, built here and in the test. Design: 4 nearest
# neighbors on the true score with replacement within the exact strata
# r, i, ties kept (Matching's default; avoids its 1-NN tie on these data),
# ATT and ATE, Var.calc 0 and 2, and the bias adjustment on x1-x3.
suppressPackageStartupMessages({library(Matching); library(jsonlite)})
df <- read.csv("tests/data/ref_data.csv")
df$y2 <- 0.5 * df$y + df$x3
df$ysum <- df$y + df$y2
out <- list(R = paste(R.version$major, R.version$minor, sep = "."),
            Matching = as.character(packageVersion("Matching")))
cases <- list(att_vc0 = list("ATT", 0, FALSE), att_vc2 = list("ATT", 2, FALSE),
              ate_vc2 = list("ATE", 2, FALSE), att_vc2_bias = list("ATT", 2, TRUE))
for (name in names(cases)) {
  cs <- cases[[name]]
  res <- list()
  for (y in c("y", "y2", "ysum")) {
    args <- list(Y = df[[y]], Tr = df$d, X = cbind(df$ps, df$r, df$i),
                 exact = c(FALSE, TRUE, TRUE), estimand = cs[[1]], M = 4, replace = TRUE,
                 ties = TRUE, Var.calc = cs[[2]], BiasAdjust = cs[[3]],
                 distance.tolerance = 0)
    if (cs[[3]]) args$Z <- cbind(df$x1, df$x2, df$x3)
    m <- do.call(Match, args)
    res[[y]] <- list(est = as.numeric(m$est), se = as.numeric(m$se))
  }
  out[[name]] <- res
}
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA), "tests/data/ref_r_several.json")
