# Reference outputs from MatchIt for common support (discard =, reestimate =)
# on tests/data/ref_data.csv.
#
#   Rscript tests/reference/reference_r_discard.R
#
# Writes tests/data/ref_matchit_discard_<spec>.csv (match.matrix: treated id,
# control ids; NA = unmatched), ..._weights.csv (nonzero weights by id) and
# tests/data/ref_r_discard.json (versions; per spec: the number discarded by
# arm, MatchIt's summary(m, un = TRUE) balance tables and sample sizes, and
# the first distances). Same data and conventions as reference_r.R.
# Needs MatchIt (4.8.2 when written) and jsonlite.
suppressPackageStartupMessages({ library(MatchIt); library(jsonlite) })

here <- dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))))
data_dir <- file.path(here, "..", "data")
df <- read.csv(file.path(data_dir, "ref_data.csv"))
rownames(df) <- df$id
out <- list(versions = list(R = R.version.string, MatchIt = as.character(packageVersion("MatchIt"))))

specs <- list(
  both = list(desc = "1-NN on ps, without replacement, discard both, exact r+i, caliper 0.2 SD",
              args = list(distance = df$ps, discard = "both", replace = FALSE, caliper = 0.2,
                          exact = ~ r + i)),
  treated_rep = list(desc = "1-NN on ps, replace, discard treated, exact r+i",
                     args = list(distance = df$ps, discard = "treated", replace = TRUE,
                                 exact = ~ r + i)),
  control_k2 = list(desc = "2-NN on ps, without replacement, discard control, exact r+i",
                    args = list(distance = df$ps, discard = "control", replace = FALSE, ratio = 2,
                                exact = ~ r + i)),
  atc_both = list(desc = "ATC, 1-NN on ps, replace, discard both",
                  args = list(distance = df$ps, discard = "both", replace = TRUE,
                              estimand = "ATC")),
  glm_both = list(desc = "1-NN on a glm score (d ~ x1+x2+x3), discard both, caliper 0.1 SD",
                  args = list(distance = "glm", discard = "both", replace = TRUE, caliper = 0.1)),
  glm_reest = list(desc = "as glm_both, reestimate = TRUE",
                   args = list(distance = "glm", discard = "both", reestimate = TRUE,
                               replace = TRUE, caliper = 0.1))
)
tab <- function(x) lapply(seq_len(nrow(x)), function(r)
  c(list(term = rownames(x)[r]), as.list(setNames(unname(x[r, ]), colnames(x)))))
for (name in names(specs)) {
  s <- specs[[name]]
  m <- do.call(matchit, c(list(formula = d ~ x1 + x2 + x3, data = df, method = "nearest"), s$args))
  mm <- m$match.matrix
  mt <- data.frame(focal = as.integer(rownames(mm)))
  for (k in seq_len(ncol(mm))) mt[[paste0("match", k)]] <- as.integer(mm[, k])
  write.csv(mt, file.path(data_dir, sprintf("ref_matchit_discard_%s.csv", name)), row.names = FALSE)
  wt <- data.frame(id = df$id, weight = m$weights)
  wt <- wt[wt$weight > 0, ]
  write.csv(wt, file.path(data_dir, sprintf("ref_matchit_discard_%s_weights.csv", name)),
            row.names = FALSE)
  sm <- summary(m, un = TRUE)
  out[[name]] <- list(desc = s$desc,
                      discarded_treated = sum(m$discarded & df$d == 1),
                      discarded_control = sum(m$discarded & df$d == 0),
                      discarded_ids = df$id[m$discarded],
                      distance_head = unname(head(m$distance, 10)),
                      balance_all = tab(sm$sum.all), balance_matched = tab(sm$sum.matched),
                      nn_rows = rownames(sm$nn), nn = unname(as.list(as.data.frame(sm$nn))))
}
writeLines(toJSON(out, auto_unbox = TRUE, digits = 15, pretty = TRUE),
           file.path(data_dir, "ref_r_discard.json"))
cat("wrote", file.path(data_dir, "ref_r_discard.json"), "\n")
