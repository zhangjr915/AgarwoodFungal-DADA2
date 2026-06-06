library(dada2)
library(vegan)
library(ggplot2)

cat("=== DADA2 Pipeline v3 (Optimized ITS parameters) ===\n")

data_dir <- "FungalAnalysis/omicsmaster-addl26040126-jt8zupyszorpnkrm/ITS_meta/ADDL26040126_std_1"
out_dir <- "FungalAnalysis/dada2_results_v3"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, "filtered"), recursive = TRUE, showWarnings = FALSE)

samples <- c("A_a_1", "A_a_2", "A_b_1", "A_b_2", "A_c_1", "A_c_2")
fnFs <- c(
  file.path(data_dir, "A_a_1_1.fq.gz"), file.path(data_dir, "A_a_2_1.fq.gz"),
  file.path(data_dir, "A_b_1_1.fq.gz"), file.path(data_dir, "A_b_2_1.fq.gz"),
  file.path(data_dir, "A_c_1_1.fq.gz"), file.path(data_dir, "A_c_2_1.fq.gz")
)
fnRs <- c(
  file.path(data_dir, "A_a_1_2.fq.gz"), file.path(data_dir, "A_a_2_2.fq.gz"),
  file.path(data_dir, "A_b_1_2.fq.gz"), file.path(data_dir, "A_b_2_2.fq.gz"),
  file.path(data_dir, "A_c_1_2.fq.gz"), file.path(data_dir, "A_c_2_2.fq.gz")
)
filtFs <- file.path(out_dir, "filtered", paste0(samples, "_F_filt.fastq.gz"))
filtRs <- file.path(out_dir, "filtered", paste0(samples, "_R_filt.fastq.gz"))

# OPTIMIZED parameters for fungal ITS (Rolling et al. 2022 mSphere)
cat("\n=== Filter & Trim (maxEE=8, truncQ=8) ===\n")
filt_out <- filterAndTrim(fnFs, filtFs, fnRs, filtRs,
                           truncLen = c(250, 200),
                           maxN = 0,
                           maxEE = c(8, 8),    # Changed from 2,2 to 8,8
                           truncQ = 8,           # Changed from 2 to 8
                           rm.phix = TRUE,
                           compress = TRUE,
                           multithread = FALSE)
print(filt_out)
write.csv(filt_out, file.path(out_dir, "dada2_stats_filter.csv"), row.names = TRUE)

cat("\n=== Learn Errors ===\n")
errF <- learnErrors(filtFs, multithread = FALSE, verbose = FALSE)
errR <- learnErrors(filtRs, multithread = FALSE, verbose = FALSE)

cat("\n=== Dereplicate & DADA2 ===\n")
derepFs <- derepFastq(filtFs); derepRs <- derepFastq(filtRs)
names(derepFs) <- samples; names(derepRs) <- samples
dadaFs <- dada(derepFs, err = errF, multithread = FALSE, verbose = FALSE)
dadaRs <- dada(derepRs, err = errR, multithread = FALSE, verbose = FALSE)

cat("\n=== Merge Pairs ===\n")
mergers <- mergePairs(dadaFs, derepFs, dadaRs, derepRs, minOverlap = 12, maxMismatch = 0, verbose = FALSE)

cat("\n=== Sequence Table ===\n")
seqtab <- makeSequenceTable(mergers)
cat("Dimensions:", dim(seqtab), "\n")

cat("\n=== Remove Chimeras ===\n")
seqtab_nochim <- removeBimeraDenovo(seqtab, method = "consensus", multithread = FALSE, verbose = TRUE)
cat("After chimera removal:", dim(seqtab_nochim), "\n")
cat("Non-chimeric fraction:", sum(seqtab_nochim)/sum(seqtab), "\n")

# Stats
getN <- function(x) sum(getUniques(x))
track <- data.frame(
  input = c(75109,82774,84971,86475,91736,92903),
  filtered = filt_out[, "reads.out"],
  denoisedF = sapply(dadaFs, getN),
  merged = sapply(mergers, getN),
  nonchim = rowSums(seqtab_nochim),
  row.names = samples
)
write.csv(track, file.path(out_dir, "dada2_stats.csv"), row.names = TRUE)
cat("\nPipeline stats:\n")
print(track)

# Save ASV table
asv_names <- paste0("ASV", sprintf("%04d", 1:ncol(seqtab_nochim)))
asv_tab <- as.data.frame(t(seqtab_nochim))
rownames(asv_tab) <- asv_names
colnames(asv_tab) <- samples
write.csv(asv_tab, file.path(out_dir, "asv_table.csv"), row.names = TRUE)

# Save FASTA
asv_seqs <- colnames(seqtab_nochim)
fasta_lines <- c()
for (i in seq_along(asv_seqs)) {
  fasta_lines <- c(fasta_lines, paste0(">", asv_names[i]))
  fasta_lines <- c(fasta_lines, asv_seqs[i])
}
writeLines(fasta_lines, file.path(out_dir, "asv_sequences.fasta"))

cat("\n=== DONE ===\n")
cat("ASVs:", ncol(seqtab_nochim), "\n")
cat("Total reads:", sum(seqtab_nochim), "\n")
