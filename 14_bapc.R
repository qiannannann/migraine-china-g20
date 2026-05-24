# =============================================================================
# 14_bapc.R — Bayesian Age-Period-Cohort projection of migraine burden
# v2.4: read std_pop from CSV with auto-validate (rewrites file if wrong values)
# v2.3: fixed std_pop decimal-shift bug (0.86 → 0.0860 etc.)
# v2.2: extraction uses slot 'agestd.rate' + cols 'mean'/'sd'
# v2.1: round epi to integers + impute pop NAs with per-year median
# v2.0: transpose data; agestd.proj single-arg API
# v1.5-1.9: see previous versions
#
# Run: setwd to analysis/ then source("14_bapc.R")
# =============================================================================

suppressMessages({
  library(BAPC)
  library(INLA)
  library(Epi)
})

INPUT_DIR_REL <- "./bapc_data"
N_PREDICT  <- 30
AGE_GF     <- 5

# ---- Step 1: Resolve INPUT_DIR ----
if (!dir.exists(INPUT_DIR_REL)) {
  stop(sprintf("INPUT_DIR not found at: %s\nCurrent getwd(): %s",
               INPUT_DIR_REL, getwd()))
}
INPUT_DIR_ORIG <- normalizePath(INPUT_DIR_REL, mustWork = TRUE)

# ---- Step 2: Detect Windows MAX_PATH issue and auto-copy to short path ----
# Check if any full file path would exceed 260 chars
example_path <- file.path(INPUT_DIR_ORIG, "cases_China_Female_prev.csv")
needs_copy <- nchar(example_path) > 250 ||
              !file.exists(file.path(INPUT_DIR_ORIG, "pop_China_Both.csv"))

if (needs_copy) {
  short_input <- file.path(Sys.getenv("USERPROFILE"), "Documents", "bapc_data")
  cat(sprintf("⚠ Long path detected (%d chars). Copying bapc_data to: %s\n",
              nchar(example_path), short_input))
  dir.create(short_input, showWarnings = FALSE, recursive = TRUE)
  src_files <- list.files(INPUT_DIR_ORIG, full.names = TRUE)
  copied <- file.copy(src_files, short_input, overwrite = TRUE)
  cat(sprintf("  Copied %d / %d files successfully.\n", sum(copied), length(src_files)))
  INPUT_DIR <- normalizePath(short_input, mustWork = TRUE)
} else {
  INPUT_DIR <- INPUT_DIR_ORIG
}

# ---- Step 3: OUTPUT_DIR (in Documents to avoid the same path issue) ----
OUTPUT_DIR_REL <- file.path(Sys.getenv("USERPROFILE"), "Documents", "bapc_output")
dir.create(OUTPUT_DIR_REL, showWarnings = FALSE, recursive = TRUE)
OUTPUT_DIR <- normalizePath(OUTPUT_DIR_REL, mustWork = TRUE)

cat("==============================================\n")
cat("Working directory:", getwd(), "\n")
cat("INPUT_DIR (final):", INPUT_DIR, "\n")
cat("OUTPUT_DIR:       ", OUTPUT_DIR, "\n")
cat("==============================================\n\n")

# ---- Step 4: Discover files ----
files_present <- list.files(INPUT_DIR, full.names = FALSE)
cat(sprintf("Total files in INPUT_DIR: %d\n", length(files_present)))
cases_files <- files_present[grepl("^cases_", files_present)]
pop_files   <- files_present[grepl("^pop_",   files_present)]
cat(sprintf("  cases_ files: %d\n  pop_ files:   %d\n\n",
            length(cases_files), length(pop_files)))
if (length(cases_files) == 0) stop("No 'cases_*.csv' files found.")

# ---- Step 5: Standard population — read from CSV with auto-validate ----
# v2.4: read GBD World 2014 standard population from CSV in INPUT_DIR.
# If file missing OR contains the known decimal-shift bug (10-14 yr > 0.5),
# we re-write a corrected version automatically.

STD_CSV <- file.path(INPUT_DIR, "gbd_world_standard_population.csv")

# Canonical WHO World 2014 (per IHME / GBD documentation), 20 age groups
correct_std <- data.frame(
  age_group = c("<5 years","5-9 years","10-14 years","15-19 years","20-24 years",
                "25-29 years","30-34 years","35-39 years","40-44 years","45-49 years",
                "50-54 years","55-59 years","60-64 years","65-69 years","70-74 years",
                "75-79 years","80-84 years","85-89 years","90-94 years","95+ years"),
  gbd_world_std_pop = c(0.0886, 0.0869, 0.0860, 0.0847, 0.0822,
                        0.0793, 0.0761, 0.0715, 0.0659, 0.0604,
                        0.0537, 0.0455, 0.0372, 0.0296, 0.0221,
                        0.0152, 0.0091, 0.0044, 0.0015, 0.0004),
  stringsAsFactors = FALSE
)

# v2.5: ALWAYS rewrite the CSV with canonical values to avoid any possibility
# of stale/wrong values being read. Then read it back.
write.csv(correct_std, STD_CSV, row.names = FALSE)
cat(sprintf("  Wrote canonical WHO World 2014 standard population to: %s\n", STD_CSV))

# Read back as the authoritative source
std_csv <- read.csv(STD_CSV, stringsAsFactors = FALSE)
std_pop_named <- std_csv$gbd_world_std_pop
names(std_pop_named) <- std_csv$age_group

# Validate the loaded values match correct values (sanity check)
if (abs(sum(std_pop_named) - 1.0) > 0.01) {
  warning(sprintf("std_pop sum = %.4f, not ~1.0. Check CSV.", sum(std_pop_named)))
}
# Confirm young ages have similar weights to middle ages (catches dec-shift bugs)
ratio_check <- std_pop_named[1] / std_pop_named[7]   # <5 vs 30-34
if (ratio_check < 0.5 || ratio_check > 2.0) {
  warning(sprintf("std_pop ratio <5/30-34 = %.3f, expected ~1.16. Decimal-shift bug?",
                  ratio_check))
}

# Normalize and strip names
std_pop <- std_pop_named / sum(std_pop_named)
std_pop <- unname(as.numeric(std_pop))
cat(sprintf("  Standard population: %d age groups, sum=%.4f\n",
            length(std_pop), sum(std_pop)))

# ---- Step 6: Parser — now accepts digits (G20) ----
parse_cases <- function(filename) {
  # FIX: allow digits in country name (was [A-Za-z]+, now [A-Za-z0-9]+)
  parts <- regmatches(filename,
    regexec("^cases_([A-Za-z0-9]+)_([A-Za-z]+)_([a-z]+)\\.csv$", filename))[[1]]
  if (length(parts) != 4) return(NULL)
  list(country = parts[2], sex = parts[3], measure = parts[4],
       cases_path = file.path(INPUT_DIR, filename),
       pop_path   = file.path(INPUT_DIR, sprintf("pop_%s_%s.csv", parts[2], parts[3])))
}

# ---- Step 7: BAPC fit function ----
run_bapc_one <- function(cfg, n_predict = N_PREDICT, gf = AGE_GF) {
  cat(sprintf("--- Fitting BAPC: %s × %s × %s ---\n", cfg$country, cfg$sex, cfg$measure))
  if (!file.exists(cfg$pop_path)) {
    cat(sprintf("  Skipping: missing pop file %s\n", cfg$pop_path))
    return(NULL)
  }
  cases <- read.csv(cfg$cases_path, row.names = 1, check.names = FALSE)
  pop   <- read.csv(cfg$pop_path,   row.names = 1, check.names = FALSE)
  stopifnot(all(dim(cases) == dim(pop)))

  # ---- Fix 1: Impute NAs in pop with a reasonable positive value ----
  # NAs appear in <5 years (migraine rate = 0). Set to median of OTHER ages
  # in the same year, rather than 1, to avoid huge magnitude reversals
  # that destabilize INLA.
  n_pop_na <- sum(is.na(pop))
  if (n_pop_na > 0) {
    # For each year column with NAs, use median of non-NA values that year
    for (j in seq_len(ncol(pop))) {
      col <- pop[, j]
      if (any(is.na(col))) {
        med <- median(col, na.rm = TRUE)
        if (is.na(med) || med <= 0) med <- 1e5  # fallback
        pop[is.na(col), j] <- med
      }
    }
    cat(sprintf("  [Note: %d NA values in pop matrix replaced with per-year median]\n", n_pop_na))
  }

  # ---- Fix 2: Round case counts to integers ----
  # INLA's Poisson likelihood requires non-negative integer epi.
  # GBD counts are derived (rate × pop / 100k) so they're floats.
  # Preserve NAs (which mark future years for projection).
  cases_rounded <- cases
  for (j in seq_len(ncol(cases_rounded))) {
    col <- cases_rounded[, j]
    not_na <- !is.na(col)
    cases_rounded[not_na, j] <- round(col[not_na])
  }
  cases <- cases_rounded

  # ---- Fix 3: BAPC expects rows = periods (years), cols = age groups ----
  cases_t <- as.data.frame(t(cases))
  pop_t   <- as.data.frame(t(pop))

  apc_obj <- APCList(
    epi  = cases_t,
    pyrs = pop_t,
    gf   = gf
  )

  bapc_fit <- BAPC(
    APCList = apc_obj,
    predict = list(npredict = n_predict, retro = TRUE),
    model = list(
      age    = list(model = "rw2", prior = "loggamma", param = c(1, 0.00005), initial = 4, scale.model = FALSE),
      period = list(model = "rw2", prior = "loggamma", param = c(1, 0.00005), initial = 4, scale.model = FALSE),
      cohort = list(model = "rw2", prior = "loggamma", param = c(1, 0.00005), initial = 4, scale.model = FALSE),
      overdis = list(model = "iid", prior = "loggamma", param = c(1, 0.005), initial = 4)
    ),
    secondDiff = FALSE,
    stdweight  = std_pop,
    verbose    = FALSE
  )

  # Extract age-standardized rate (per person), multiply by 1e5 → per 100,000
  # BAPC 0.0.37 stores: slot 'agestd.rate' = 64 rows × 2 cols (mean, sd)
  asr_rate_mat <- slot(bapc_fit, "agestd.rate")
  if (nrow(asr_rate_mat) < 2) {
    stop("agestd.rate slot is empty — BAPC fit did not produce projections")
  }
  years_all      <- as.numeric(rownames(asr_rate_mat))
  mean_per_100k  <- asr_rate_mat[, "mean"] * 100000
  sd_per_100k    <- asr_rate_mat[, "sd"]   * 100000
  asr_proj <- data.frame(
    year  = years_all,
    mean  = mean_per_100k,
    lower = mean_per_100k - 1.96 * sd_per_100k,
    upper = mean_per_100k + 1.96 * sd_per_100k
  )

  list(country = cfg$country, sex = cfg$sex, measure = cfg$measure,
       fit = bapc_fit, asr = asr_proj)
}

# ---- Step 8: Run all ----
results <- list()
asr_summary <- data.frame()
for (cf in sort(cases_files)) {
  cfg <- parse_cases(cf)
  if (is.null(cfg)) {
    cat(sprintf("  Could not parse filename: %s — skipping\n", cf))
    next
  }
  key <- paste(cfg$country, cfg$sex, cfg$measure, sep = "_")
  out <- tryCatch(run_bapc_one(cfg),
                  error = function(e) {
                    cat(sprintf("  ERROR in %s: %s\n", key, e$message))
                    NULL
                  })
  if (is.null(out)) next
  results[[key]] <- out
  asr_df <- out$asr
  asr_df$country <- cfg$country
  asr_df$sex     <- cfg$sex
  asr_df$measure <- cfg$measure
  asr_summary    <- rbind(asr_summary, asr_df)
}

cat(sprintf("\n=== BAPC complete: %d / %d fits successful ===\n",
            length(results), length(cases_files)))

if (nrow(asr_summary) == 0) {
  stop("No fits succeeded. Check error messages above.")
}

# ---- Step 9: Save outputs ----
saveRDS(results, file.path(OUTPUT_DIR, "bapc_results.rds"))
write.csv(asr_summary, file.path(OUTPUT_DIR, "bapc_projections_2024_2053.csv"), row.names = FALSE)

# ---- Step 10: Plot ----
pdf(file.path(OUTPUT_DIR, "bapc_projections.pdf"), width = 12, height = 8)
par(mfrow = c(2, 3), mar = c(4, 4, 3, 1))
for (country in unique(asr_summary$country)) {
  for (measure in c("prev", "inc", "daly")) {
    measure_label <- switch(measure,
      "prev" = "ASPR per 100,000",
      "inc"  = "ASIR per 100,000",
      "daly" = "ASDR per 100,000",
      paste(measure))
    cols <- c("Both" = "black", "Male" = "#1F77B4", "Female" = "#D62728")
    sub <- asr_summary[asr_summary$country == country & asr_summary$measure == measure, ]
    if (nrow(sub) == 0) {
      plot.new(); title(sprintf("%s — %s (no data)", country, toupper(measure))); next
    }
    ylim_use <- range(sub$lower, sub$upper, na.rm = TRUE)
    plot(NA, xlim = c(1990, 2053), ylim = ylim_use,
         xlab = "Year", ylab = measure_label,
         main = sprintf("%s — %s", country, toupper(measure)))
    for (sex in c("Female", "Male", "Both")) {
      d <- sub[sub$sex == sex, ]
      if (nrow(d) == 0) next
      polygon(c(d$year, rev(d$year)), c(d$lower, rev(d$upper)),
              col = adjustcolor(cols[sex], alpha.f = 0.2), border = NA)
      lines(d$year, d$mean, col = cols[sex], lwd = 2)
    }
    abline(v = 2023, col = "gray", lty = 2)
    legend("topleft", legend = names(cols), col = cols, lwd = 2, cex = 0.7, bty = "n")
  }
}
dev.off()

# ---- Step 11: Export for merge ----
asr_export <- asr_summary[, c("country", "sex", "measure", "year", "mean", "lower", "upper")]
colnames(asr_export) <- c("country", "sex", "measure", "year", "projection", "PI_lower", "PI_upper")
asr_export$model <- "BAPC"
write.csv(asr_export, file.path(OUTPUT_DIR, "bapc_projections_for_merge.csv"), row.names = FALSE)

cat(sprintf("\n=== Done. All outputs at: %s ===\n", OUTPUT_DIR))
cat("Files:\n")
cat("  - bapc_results.rds\n")
cat("  - bapc_projections_2024_2053.csv\n")
cat("  - bapc_projections_for_merge.csv\n")
cat("  - bapc_projections.pdf\n")
