# Reference outputs from R's MatchIt and Matching on tests/data/ref_data.csv.
#
#   Rscript tests/reference/reference_r.R
#
# Writes tests/data/ref_matchit_<spec>.csv (MatchIt's match.matrix: one row per
# treated unit, its id and the ids of its matched controls; NA = no match),
# tests/data/ref_matchit_<spec>_weights.csv (MatchIt's nonzero weights by id, and the
# subclass when MatchIt reports one) and
# tests/data/ref_r.json (versions, specs, and Matching's estimates and
# Abadie-Imbens standard errors).
#
# Generated with R 4.5.3, MatchIt 4.8.2 (CRAN source build) and Matching 4.10-15
# (conda-forge). conda-forge's r-matchit was 4.7.2 at the time, which predates
# m.order = "closest"; build MatchIt from CRAN (see benchmarks/README.md).
suppressPackageStartupMessages({ library(MatchIt); library(Matching); library(jsonlite) })

here <- dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))))
data_dir <- file.path(here, "..", "data")
df <- read.csv(file.path(data_dir, "ref_data.csv"))
rownames(df) <- df$id
df$rf <- factor(df$r); df$if_ <- factor(df$i)
out <- list(versions = list(R = R.version.string, MatchIt = as.character(packageVersion("MatchIt")),
                            Matching = as.character(packageVersion("Matching"))),
            matchit = list(), matching = list())

cal3 <- c(0.2, x1 = 0.25, x2 = 0.25)
std3 <- c(TRUE, FALSE, FALSE)
specs <- list(
  nn_ps_rep = list(desc = "1-NN on ps, replace, exact r+i, caliper 0.2 SD of ps",
                   args = list(method = "nearest", distance = df$ps, replace = TRUE, caliper = 0.2,
                               exact = ~ r + i)),
  nn_ps_rep_k3_cal = list(desc = "3-NN on ps, replace, exact r+i, calipers ps 0.2 SD, |dx1|,|dx2| <= 0.25",
                          args = list(method = "nearest", distance = df$ps, replace = TRUE, ratio = 3,
                                      caliper = cal3, std.caliper = std3, exact = ~ r + i)),
  nn_ps_wor_largest = list(desc = "1-NN on ps, without replacement, m.order largest, exact r+i, calipers as k3",
                           args = list(method = "nearest", distance = df$ps, replace = FALSE, m.order = "largest",
                                       caliper = cal3, std.caliper = std3, exact = ~ r + i)),
  nn_ps_wor_smallest = list(desc = "as wor_largest, m.order smallest",
                            args = list(method = "nearest", distance = df$ps, replace = FALSE, m.order = "smallest",
                                        caliper = cal3, std.caliper = std3, exact = ~ r + i)),
  nn_ps_wor_closest = list(desc = "as wor_largest, m.order closest",
                           args = list(method = "nearest", distance = df$ps, replace = FALSE, m.order = "closest",
                                       caliper = cal3, std.caliper = std3, exact = ~ r + i)),
  nn_ps_wor_data = list(desc = "as wor_largest, m.order data",
                        args = list(method = "nearest", distance = df$ps, replace = FALSE, m.order = "data",
                                    caliper = cal3, std.caliper = std3, exact = ~ r + i)),
  nn_ps_wor_k2 = list(desc = "2-NN on ps, without replacement, m.order largest, exact r+i, caliper 0.2 SD",
                      args = list(method = "nearest", distance = df$ps, replace = FALSE, ratio = 2,
                                  m.order = "largest", caliper = 0.2, exact = ~ r + i)),
  nn_maha_rep = list(desc = "1-NN Mahalanobis on x1..x3 (MatchIt's covariance), replace, exact r+i",
                     args = list(method = "nearest", distance = "mahalanobis", replace = TRUE, exact = ~ r + i)),
  nn_maha_wor_cal = list(desc = "1-NN Mahalanobis on x1..x3, without replacement, m.order data, exact r+i, calipers |dx1|,|dx2| <= 0.25 (the benchmarks' scenario H)",
                         args = list(method = "nearest", distance = "mahalanobis", replace = FALSE,
                                     m.order = "data", caliper = c(x1 = 0.25, x2 = 0.25),
                                     std.caliper = FALSE, exact = ~ r + i)),
  nn_glm_rep = list(desc = "1-NN on a logit score fitted by MatchIt (d ~ x1+x2+x3+factor(r)+factor(i)), replace, caliper 0.2 SD",
                    formula = d ~ x1 + x2 + x3 + rf + if_,
                    args = list(method = "nearest", distance = "glm", replace = TRUE, caliper = 0.2)),
  cem5 = list(desc = "CEM: 5 equal-width bins on x1..x3, exact r and i",
              formula = d ~ x1 + x2 + x3 + rf + if_,
              args = list(method = "cem", cutpoints = list(x1 = 5, x2 = 5, x3 = 5)))
)
for (name in names(specs)) {
  s <- specs[[name]]
  f <- if (is.null(s$formula)) d ~ x1 + x2 + x3 else s$formula
  m <- do.call(matchit, c(list(formula = f, data = df, estimand = "ATT"), s$args))
  if (!is.null(m$match.matrix)) {
    mm <- m$match.matrix
    tab <- data.frame(treated = as.integer(rownames(mm)))
    for (k in seq_len(ncol(mm))) tab[[paste0("control", k)]] <- as.integer(mm[, k])
    write.csv(tab, file.path(data_dir, sprintf("ref_matchit_%s.csv", name)), row.names = FALSE)
  }
  wt <- data.frame(id = df$id, weight = m$weights)
  if (!is.null(m$subclass)) wt$subclass <- as.integer(m$subclass)
  wt <- wt[wt$weight > 0, ]                         # units with weight 0 are left out
  write.csv(wt, file.path(data_dir, sprintf("ref_matchit_%s_weights.csv", name)), row.names = FALSE, na = "")
  extra <- list()
  if (name == "nn_glm_rep") extra$distance_head <- unname(head(m$distance, 10))
  if (name %in% c("nn_ps_rep", "nn_ps_wor_largest", "nn_ps_rep_k3_cal", "cem5")) {
    # MatchIt's balance table (summary(m, un = TRUE)): rows = covariates,
    # columns Means Treated, Means Control, Std. Mean Diff., Var. Ratio,
    # eCDF Mean, eCDF Max (and Std. Pair Dist. for matched pairs)
    sm <- summary(m, un = TRUE, addlvariables = ~ x1 + x2 + x3)
    tab <- function(x) lapply(seq_len(nrow(x)), function(r)
      c(list(term = rownames(x)[r]), as.list(setNames(unname(x[r, ]), colnames(x)))))
    extra$balance_all <- tab(sm$sum.all)
    extra$balance_matched <- tab(sm$sum.matched)
    extra$nn <- unname(as.list(as.data.frame(sm$nn)))
  }
  out$matchit[[name]] <- c(list(desc = s$desc, n_matched_treated = sum(df$d == 1 & m$weights > 0),
                                n_matched_control = sum(df$d == 0 & m$weights > 0)), extra)
}

# Matching: estimates and Abadie-Imbens standard errors (ties kept, distance.tolerance = 0)
X_ps <- cbind(df$ps, df$r, df$i)
big <- 1e6
W_maha <- diag(5); W_maha[1:3, 1:3] <- solve(cov(df[, c("x1", "x2", "x3")]))
mspecs <- list(
  ai_att_m1_vc0 = list(desc = "ATT, 1-NN on ps, replace, exact r,i, caliper 0.2 SD, Var.calc 0",
                       args = list(X = X_ps, exact = c(FALSE, TRUE, TRUE), caliper = c(0.2, big, big),
                                   estimand = "ATT", M = 1)),
  ai_att_m4_vc4 = list(desc = "ATT, 4-NN on ps, replace, exact r,i, no caliper, Var.calc 4 (heteroskedastic)",
                       args = list(X = X_ps, exact = c(FALSE, TRUE, TRUE), estimand = "ATT", M = 4, Var.calc = 4)),
  ai_ate_m1_vc0 = list(desc = "ATE, 1-NN on ps, replace, exact r,i, no caliper, Var.calc 0",
                       args = list(X = X_ps, exact = c(FALSE, TRUE, TRUE), estimand = "ATE", M = 1)),
  ai_ate_m2_vc0 = list(desc = "ATE, 2-NN on ps, replace, exact r,i, no caliper, Var.calc 0",
                       args = list(X = X_ps, exact = c(FALSE, TRUE, TRUE), estimand = "ATE", M = 2)),
  ai_atc_m2_vc2 = list(desc = "ATC, 2-NN on ps, replace, exact r,i, no caliper, Var.calc 2",
                       args = list(X = X_ps, exact = c(FALSE, TRUE, TRUE), estimand = "ATC", M = 2, Var.calc = 2)),
  ai_att_biasadj = list(desc = "ATT, 1-NN on ps, exact r,i, BiasAdjust on Z = x1..x3, Var.calc 0",
                        args = list(X = X_ps, exact = c(FALSE, TRUE, TRUE), estimand = "ATT", M = 1,
                                    BiasAdjust = TRUE, Z = as.matrix(df[, c("x1", "x2", "x3")]))),
  ai_att_maha = list(desc = "ATT, 1-NN Mahalanobis on x1..x3 (full-sample covariance), exact r,i, Var.calc 0",
                     args = list(X = cbind(df$x1, df$x2, df$x3, df$r, df$i),
                                 exact = c(FALSE, FALSE, FALSE, TRUE, TRUE), Weight = 3, Weight.matrix = W_maha,
                                 estimand = "ATT", M = 1))
)
for (name in names(mspecs)) {
  s <- mspecs[[name]]
  m <- do.call(Match, c(list(Y = df$y, Tr = df$d, replace = TRUE, ties = TRUE, distance.tolerance = 0), s$args))
  if (name == "ai_att_m1_vc0") {
    # Matching's own pairs (treated id, control id, weight). On these data it
    # treats one treated unit (id 18735) as tied between two controls whose
    # distances differ by a factor of two (unexplained), so
    # the variance formula is also tested on these exact pairs.
    write.csv(data.frame(treated = df$id[m$index.treated], control = df$id[m$index.control],
                         weight = m$weights),
              file.path(data_dir, sprintf("ref_matching_%s_pairs.csv", name)), row.names = FALSE)
  }
  out$matching[[name]] <- list(desc = s$desc, est = unname(m$est[1, 1]), se = m$se,
                               se_standard = m$se.standard, nobs = m$nobs, wnobs = m$wnobs,
                               n_dropped = length(m$index.dropped))
}
writeLines(toJSON(out, auto_unbox = TRUE, digits = 15, pretty = TRUE), file.path(data_dir, "ref_r.json"))
cat("wrote", file.path(data_dir, "ref_r.json"), "\n")
