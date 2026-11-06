# Reference outputs from MatchIt for Mahalanobis matching within a score
# caliper (mahvars =) on tests/data/ref_data.csv.
#
#   Rscript tests/reference/reference_r_mahvars.R
#
# Writes tests/data/ref_matchit_mahvars_<spec>.csv (match.matrix), ..._weights.csv
# and tests/data/ref_r_mahvars.json (versions; per spec: balance tables, sample
# sizes, discarded counts). Same data and conventions as reference_r.R.
# Needs MatchIt (4.8.2 when written) and jsonlite.
suppressPackageStartupMessages({ library(MatchIt); library(jsonlite) })

here <- dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))))
data_dir <- file.path(here, "..", "data")
df <- read.csv(file.path(data_dir, "ref_data.csv"))
rownames(df) <- df$id
out <- list(versions = list(R = R.version.string, MatchIt = as.character(packageVersion("MatchIt"))))

specs <- list(
  wor_cal = list(desc = "1-NN Mahalanobis on x1..x3 within a 0.2 SD caliper on ps, without replacement, exact r+i",
                 args = list(distance = df$ps, mahvars = ~ x1 + x2 + x3, caliper = 0.2,
                             replace = FALSE, exact = ~ r + i)),
  rep_k2 = list(desc = "2-NN Mahalanobis on x1..x3 within a 0.1 SD ps caliper, replace, exact r+i",
                args = list(distance = df$ps, mahvars = ~ x1 + x2 + x3, caliper = 0.1,
                            replace = TRUE, ratio = 2, exact = ~ r + i)),
  glm_discard = list(desc = "1-NN Mahalanobis on x1, x2 within a 0.1 SD caliper on a glm score, discard both",
                     args = list(distance = "glm", mahvars = ~ x1 + x2, caliper = 0.1,
                                 discard = "both", replace = FALSE)),
  closest = list(desc = "1-NN Mahalanobis on x1..x3 within a 0.25 SD ps caliper, m.order closest",
                 args = list(distance = df$ps, mahvars = ~ x1 + x2 + x3, caliper = 0.25,
                             replace = FALSE, m.order = "closest", exact = ~ r + i)),
  atc_rep = list(desc = "ATC, 1-NN Mahalanobis on x1..x3 within a 0.2 SD ps caliper, replace",
                 args = list(distance = df$ps, mahvars = ~ x1 + x2 + x3, caliper = 0.2,
                             replace = TRUE, estimand = "ATC"))
)
tab <- function(x) lapply(seq_len(nrow(x)), function(r)
  c(list(term = rownames(x)[r]), as.list(setNames(unname(x[r, ]), colnames(x)))))
for (name in names(specs)) {
  s <- specs[[name]]
  m <- do.call(matchit, c(list(formula = d ~ x1 + x2 + x3, data = df, method = "nearest"), s$args))
  mm <- m$match.matrix
  mt <- data.frame(focal = as.integer(rownames(mm)))
  for (k in seq_len(ncol(mm))) mt[[paste0("match", k)]] <- as.integer(mm[, k])
  write.csv(mt, file.path(data_dir, sprintf("ref_matchit_mahvars_%s.csv", name)), row.names = FALSE)
  wt <- data.frame(id = df$id, weight = m$weights)
  wt <- wt[wt$weight > 0, ]
  write.csv(wt, file.path(data_dir, sprintf("ref_matchit_mahvars_%s_weights.csv", name)),
            row.names = FALSE)
  sm <- summary(m, un = TRUE)
  out[[name]] <- list(desc = s$desc, discarded = sum(m$discarded),
                      balance_all = tab(sm$sum.all), balance_matched = tab(sm$sum.matched),
                      nn = unname(as.list(as.data.frame(sm$nn))))
}
writeLines(toJSON(out, auto_unbox = TRUE, digits = 15, pretty = TRUE),
           file.path(data_dir, "ref_r_mahvars.json"))
cat("wrote", file.path(data_dir, "ref_r_mahvars.json"), "\n")
