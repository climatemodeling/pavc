#!/usr/bin/env Rscript

# Usage: Rscript convert_rdata_to_csv.R path/to/your_file.RData

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) {
  stop("Usage: Rscript convert_rdata_to_csv.R /mnt/poseidon/remotesensing/arctic/alaska_pft_fcover_harmonization/data/plot_data/splot/input_data/_raw/sPlotOpen.RData")
}

rdata_path <- args[1]
if (!file.exists(rdata_path)) {
  stop("File not found: ", rdata_path)
}

# Load all objects into the global environment
load(rdata_path, envir = .GlobalEnv)

# Identify which objects are data.frames
all_objs <- ls(envir = .GlobalEnv)
df_names <- Filter(function(nm) is.data.frame(get(nm, envir = .GlobalEnv)), all_objs)

if (length(df_names) == 0) {
  message("No data.frames found in ", basename(rdata_path))
  quit(status = 0)
}

# Write each data.frame to CSV
for (nm in df_names) {
  df <- get(nm, envir = .GlobalEnv)

  # 1) find list columns
  is_list_col <- sapply(df, is.list)
  if (any(is_list_col)) {
    # 2) collapse each list element to a single string
    df[is_list_col] <- lapply(
      df[is_list_col],
      function(col) {
        vapply(col,
               function(x) paste(unlist(x), collapse = "; "),
               FUN.VALUE = character(1))
      }
    )
    message("  • Converted list columns in ", nm)
  }

  # 3) write out the cleaned df
  out_csv <- paste0(nm, ".csv")
  write.csv(df, out_csv, row.names = FALSE)
  message("Wrote ", out_csv)
}