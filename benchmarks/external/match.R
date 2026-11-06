# One match with an R package, for benchmarks/_harness.py: read the Parquet
# file into memory (timed as "read", as the Python libraries' reading is),
# match, compute the ATT with the matching weights, and print one line,
#
#   RESULT {"t_read": ..., "t_fit": ..., "peak_memory_mb": ..., "coefs": {"att": ...}, ...}
#
# Peak memory is the process's VmHWM, as for the Python configurations.
#
#   Rscript match.R '<spec as JSON>'     (see the benchmarks' external_command)
#
# spec$fit is one of
#   matchit    MatchIt::matchit, the scenario's design as match-stream's
#   matching   Matching::Match (version "fast", no standard errors), the
#              exact strata as exact columns of X
#   matchby    Matching::Matchby over the exact strata (scenario A)
#   matching_ai  Matching::Match with the outcome (version "standard"): the
#              estimate and the Abadie-Imbens SE, for effects_benchmark.py's
#              design (1:1 on the score with replacement, exact strata, no
#              caliper); spec$n_variance_matches, spec$bias_adjustment (on x1-x3)
# spec$scenario (A-H, benchmarks/_scenarios.py), spec$path, spec$cal_sd,
# spec$cal_x. The Parquet file is read with arrow when it is installed, else
# nanoparquet.
#
# Matching's designs differ from MatchIt's where its conventions do (the ATT
# then differs from match-stream's; docs/implementations.md): its calipers
# are in SDs of each column, so the raw x calipers are divided by the SDs;
# a treated unit whose nearest control fails a covariate caliper is dropped,
# not matched to the next; its Mahalanobis weight matrix is the inverse of the
# full-sample covariance (MatchIt's and match-stream's: pooled within arms);
# it matches without replacement in data order, so the data are sorted by
# descending score first (MatchIt's "largest").

spec <- jsonlite::fromJSON(commandArgs(TRUE)[1], simplifyVector = TRUE)

hwm_mb <- function() {
  line <- grep("^VmHWM:", readLines("/proc/self/status"), value = TRUE)
  as.numeric(strsplit(trimws(sub("VmHWM:", "", line)), " +")[[1]][1]) / 1024
}
now <- function() proc.time()[["elapsed"]]

cols <- c("r", "i", "d", "score", "y", "x1", "x2", "x3")
if (requireNamespace("arrow", quietly = TRUE)) {
  read <- function(p) as.data.frame(arrow::read_parquet(p, col_select = cols))
} else {
  read <- function(p) as.data.frame(nanoparquet::read_parquet(p, col_select = cols))
}
pkg <- if (spec$fit == "matchit") "MatchIt" else "Matching"
suppressPackageStartupMessages(library(pkg, character.only = TRUE))
baseline <- hwm_mb()
t0 <- now()
df <- read(spec$path)
df$d <- as.integer(df$d)
t_read <- now() - t0

s <- spec$scenario
cal_sd <- spec$cal_sd
cal_x <- spec$cal_x
t0 <- now()
if (spec$fit == "matching_ai") {
  # ties = TRUE: Matching's bias adjustment requires it; the scores have no
  # ties, so the matches are those of ties = FALSE
  args <- list(Y = df$y, Tr = df$d, X = cbind(df$score, df$r, df$i), estimand = "ATT",
               M = 1, replace = TRUE, exact = c(FALSE, TRUE, TRUE), ties = TRUE,
               Var.calc = as.integer(spec$n_variance_matches),
               BiasAdjust = isTRUE(spec$bias_adjustment),
               version = "standard", distance.tolerance = 0)
  if (isTRUE(spec$bias_adjustment)) args$Z <- cbind(df$x1, df$x2, df$x3)
  m <- do.call(Match, args)
  att <- as.numeric(m$est)
  se <- as.numeric(m$se)
  matched <- length(unique(m$index.treated))
} else if (spec$fit == "matchit") {
  df$r <- factor(df$r)
  df$i <- factor(df$i)
  cal <- c(cal_sd, x1 = cal_x, x2 = cal_x)
  std <- c(TRUE, FALSE, FALSE)
  m <- switch(s,
    A = matchit(d ~ x1 + x2 + x3, data = df, method = "nearest", distance = df$score,
                replace = TRUE, caliper = cal_sd, exact = ~ r + i, estimand = "ATT"),
    B = matchit(d ~ x1 + x2 + x3, data = df, method = "nearest", distance = "mahalanobis",
                replace = TRUE, exact = ~ r + i, estimand = "ATT"),
    C = matchit(d ~ x1 + x2 + x3 + r + i, data = df, method = "cem",
                cutpoints = list(x1 = 5, x2 = 5, x3 = 5), estimand = "ATT"),
    E = matchit(d ~ x1 + x2 + x3, data = df, method = "nearest", distance = df$score,
                replace = TRUE, ratio = 3, caliper = cal, std.caliper = std,
                exact = ~ r + i, estimand = "ATT"),
    F = matchit(d ~ x1 + x2 + x3, data = df, method = "nearest", distance = df$score,
                replace = FALSE, m.order = "largest", caliper = cal, std.caliper = std,
                exact = ~ r + i, estimand = "ATT"),
    G = matchit(d ~ x1 + x2 + x3, data = df, method = "nearest", distance = "mahalanobis",
                replace = TRUE, estimand = "ATT"),
    H = matchit(d ~ x1 + x2 + x3, data = df, method = "nearest", distance = "mahalanobis",
                replace = FALSE, m.order = "data", caliper = c(x1 = cal_x, x2 = cal_x),
                std.caliper = FALSE, exact = ~ r + i, estimand = "ATT"),
    stop(paste("MatchIt has no scenario", s)))
  w <- m$weights
  t <- df$d == 1
  k <- w > 0
  att <- weighted.mean(df$y[t & k], w[t & k]) - weighted.mean(df$y[!t & k], w[!t & k])
  matched <- sum(t & k)
} else {
  if (s == "F") df <- df[order(-df$score), ]
  tr <- df$d
  big <- 1e6                                   # no caliper on the exact columns
  calx <- cal_x / c(sd(df$x1), sd(df$x2))     # in SDs of each column
  if (spec$fit == "matchby") {
    stopifnot(s == "A")
    m <- Matchby(Y = df$y, Tr = tr, X = df$score, by = list(df$r, df$i), estimand = "ATT",
                 M = 1, caliper = cal_sd, replace = TRUE, ties = FALSE, AI = FALSE,
                 print.level = 0, distance.tolerance = 0)
  } else {
    m <- switch(s,
      A = Match(Tr = tr, X = cbind(df$score, df$r, df$i), estimand = "ATT", M = 1,
                replace = TRUE, exact = c(FALSE, TRUE, TRUE), caliper = c(cal_sd, big, big),
                ties = FALSE, version = "fast", distance.tolerance = 0),
      B = {
        W <- diag(5)
        W[1:3, 1:3] <- solve(cov(df[, c("x1", "x2", "x3")]))
        Match(Tr = tr, X = cbind(df$x1, df$x2, df$x3, df$r, df$i), estimand = "ATT", M = 1,
              replace = TRUE, exact = c(FALSE, FALSE, FALSE, TRUE, TRUE), Weight = 3,
              Weight.matrix = W, ties = FALSE, version = "fast", distance.tolerance = 0)
      },
      E = Match(Tr = tr, X = cbind(df$score, df$x1, df$x2, df$r, df$i), estimand = "ATT",
                M = 3, replace = TRUE, exact = c(FALSE, FALSE, FALSE, TRUE, TRUE),
                caliper = c(cal_sd, calx, big, big), Weight = 3,
                Weight.matrix = diag(c(1, 1e-12, 1e-12, 1, 1)), ties = FALSE,
                version = "fast", distance.tolerance = 0),
      F = Match(Tr = tr, X = cbind(df$score, df$x1, df$x2, df$r, df$i), estimand = "ATT",
                M = 1, replace = FALSE, exact = c(FALSE, FALSE, FALSE, TRUE, TRUE),
                caliper = c(cal_sd, calx, big, big), Weight = 3,
                Weight.matrix = diag(c(1, 1e-12, 1e-12, 1, 1)), ties = FALSE,
                version = "fast", distance.tolerance = 0),
      H = {
        W <- diag(5)
        W[1:3, 1:3] <- solve(cov(df[, c("x1", "x2", "x3")]))
        Match(Tr = tr, X = cbind(df$x1, df$x2, df$x3, df$r, df$i), estimand = "ATT", M = 1,
              replace = FALSE, exact = c(FALSE, FALSE, FALSE, TRUE, TRUE),
              caliper = c(calx, big, big, big), Weight = 3, Weight.matrix = W,
              ties = FALSE, version = "fast", distance.tolerance = 0)
      },
      stop(paste("Matching has no scenario", s)))
  }
  if (spec$fit == "matchby") {
    att <- as.numeric(m$est)
    matched <- NA
  } else {
    it <- m$index.treated
    ic <- m$index.control
    w <- m$weights
    att <- sum(w * (df$y[it] - df$y[ic])) / sum(w)
    matched <- length(unique(it))
  }
}
t_fit <- now() - t0

cat("RESULT", jsonlite::toJSON(list(
  t_read = t_read, t_fit = t_fit, peak_memory_mb = hwm_mb(), baseline_memory_mb = baseline,
  peak_disk_mb = 0, matched = matched, converged = TRUE,
  coefs = if (exists("se")) list(att = unname(att), se = unname(se)) else list(att = unname(att))),
  auto_unbox = TRUE, digits = NA, na = "null"), "\n")
