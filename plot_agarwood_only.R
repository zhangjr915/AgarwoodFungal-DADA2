#!/usr/bin/env Rscript
# ============================================================
# 沉香内生真菌多样性分析 —— 仅 A 组（沉香）6 个样本
# 排除 B 组（苍术）
# ============================================================

library(ggplot2)
library(reshape2)
library(vegan)
library(pheatmap)
library(RColorBrewer)
library(scales)
library(gridExtra)
library(grid)

# ---------- 通用设置 ----------
theme_pub <- theme_bw() +
  theme(
    text = element_text(size = 14, family = "sans"),
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 12),
    axis.text.y = element_text(size = 12),
    axis.title = element_text(size = 14),
    legend.text = element_text(size = 11),
    legend.title = element_text(size = 13),
    strip.text = element_text(size = 13, face = "bold"),
    panel.grid.minor = element_blank()
  )

agarwood_samples <- c("A_a_1", "A_a_2", "A_b_1", "A_b_2", "A_c_1", "A_c_2")
cols_group <- c("A_a" = "#E64B35", "A_b" = "#4DBBD5", "A_c" = "#00A087")

# ============================================================
# 1. DADA2 Pipeline 质控流程图
# ============================================================
cat(">>> 绘制图1: DADA2 Pipeline 质控流程图\n")

stats <- read.csv("FungalAnalysis/dada2_results/dada2_stats.csv", row.names = 1)
stats <- stats[agarwood_samples, ]

stages <- c("input", "filtered", "denoisedF", "merged", "nonchim")
stage_labels <- c("Raw Reads", "Filtered", "Denoised", "Merged", "Non-chimeric")

stats_melt <- melt(as.matrix(stats))
colnames(stats_melt) <- c("Sample", "Stage", "Reads")
stats_melt$Stage <- factor(stats_melt$Stage, levels = stages, labels = stage_labels)
stats_melt$Source <- ifelse(grepl("A_a", stats_melt$Sample), "Source A",
                     ifelse(grepl("A_b", stats_melt$Sample), "Source B", "Source C"))

p1 <- ggplot(stats_melt, aes(x = Sample, y = Reads, fill = Stage)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  scale_y_continuous(labels = comma, expand = c(0, 0), limits = c(0, 100000)) +
  scale_fill_brewer(palette = "Set2", name = "Pipeline Stage") +
  labs(title = "DADA2 Processing Pipeline — Agarwood Samples",
       x = "Sample", y = "Number of Reads") +
  theme_pub +
  theme(legend.position = "right")

ggsave("FungalAnalysis/figures/v2/fig1_dada2_pipeline.png", p1,
       width = 10, height = 6, dpi = 300)

# ---------- 补充：保留率注释条 ----------
retention <- data.frame(
  Sample = agarwood_samples,
  Retention = round(stats$nonchim / stats$input * 100, 1)
)
retention$Source <- ifelse(grepl("A_a", retention$Sample), "Source A",
                   ifelse(grepl("A_b", retention$Sample), "Source B", "Source C"))

p1b <- ggplot(retention, aes(x = Sample, y = Retention, fill = Source)) +
  geom_bar(stat = "identity", width = 0.6, color = "black", linewidth = 0.3) +
  geom_text(aes(label = paste0(Retention, "%")), vjust = -0.5, size = 4.5) +
  scale_fill_manual(values = cols_group) +
  scale_y_continuous(limits = c(0, 100), expand = c(0, 0)) +
  labs(title = "Read Retention Rate After DADA2 Pipeline",
       x = "Sample", y = "Retention Rate (%)") +
  theme_pub

ggsave("FungalAnalysis/figures/v2/fig1b_retention_rate.png", p1b,
       width = 8, height = 5, dpi = 300)

# ============================================================
# 2. Alpha 多样性箱线图（A_a vs A_b vs A_c）
# ============================================================
cat(">>> 绘制图2: Alpha 多样性箱线图\n")

alpha <- read.csv("FungalAnalysis/dada2_results/alpha_diversity.csv")
alpha <- alpha[alpha$group == "Agarwood", ]
alpha$Source <- factor(alpha$source, levels = c("A_a", "A_b", "A_c"),
                       labels = c("Source A", "Source B", "Source C"))

# Shannon
p2a <- ggplot(alpha, aes(x = Source, y = Shannon, fill = Source)) +
  geom_boxplot(width = 0.5, outlier.shape = NA, alpha = 0.7) +
  geom_jitter(width = 0.15, size = 3, shape = 21, fill = "white", color = "black") +
  scale_fill_manual(values = cols_group) +
  labs(title = "Alpha Diversity — Shannon Index",
       x = "", y = "Shannon Index") +
  theme_pub + theme(legend.position = "none")

# Simpson
p2b <- ggplot(alpha, aes(x = Source, y = Simpson, fill = Source)) +
  geom_boxplot(width = 0.5, outlier.shape = NA, alpha = 0.7) +
  geom_jitter(width = 0.15, size = 3, shape = 21, fill = "white", color = "black") +
  scale_fill_manual(values = cols_group) +
  labs(title = "Alpha Diversity — Simpson Index",
       x = "", y = "Simpson Index") +
  theme_pub + theme(legend.position = "none")

# Chao1
p2c <- ggplot(alpha, aes(x = Source, y = Chao1, fill = Source)) +
  geom_boxplot(width = 0.5, outlier.shape = NA, alpha = 0.7) +
  geom_jitter(width = 0.15, size = 3, shape = 21, fill = "white", color = "black") +
  scale_fill_manual(values = cols_group) +
  labs(title = "Alpha Diversity — Chao1 Richness",
       x = "", y = "Chao1") +
  theme_pub + theme(legend.position = "none")

# ASV counts
p2d <- ggplot(alpha, aes(x = Source, y = ASVs, fill = Source)) +
  geom_boxplot(width = 0.5, outlier.shape = NA, alpha = 0.7) +
  geom_jitter(width = 0.15, size = 3, shape = 21, fill = "white", color = "black") +
  geom_text(aes(label = ASVs), vjust = -1.5, size = 3.5) +
  scale_fill_manual(values = cols_group) +
  labs(title = "Observed ASV Richness",
       x = "", y = "Number of ASVs") +
  theme_pub + theme(legend.position = "none")

p2_combined <- arrangeGrob(p2a, p2b, p2c, p2d, nrow = 2, ncol = 2)
ggsave("FungalAnalysis/figures/v2/fig2_alpha_diversity.png", p2_combined,
       width = 12, height = 10, dpi = 300)

# ============================================================
# 3. PCoA 排序图（仅沉香 6 样本）
# ============================================================
cat(">>> 绘制图3: PCoA 排序图\n")

asv_tab <- read.csv("FungalAnalysis/dada2_results/asv_table.csv", row.names = 1)
asv_tab <- asv_tab[, agarwood_samples]
asv_tab_t <- t(asv_tab)

# Bray-Curtis PCoA
bc <- vegdist(asv_tab_t, method = "bray")
pcoa_bc <- cmdscale(bc, k = 2, eig = TRUE)
eig_pct <- pcoa_bc$eig / sum(pcoa_bc$eig) * 100

pcoa_df <- data.frame(
  Sample = rownames(pcoa_bc$points),
  PC1 = pcoa_bc$points[, 1],
  PC2 = pcoa_bc$points[, 2]
)
pcoa_df$Source <- ifelse(grepl("A_a", pcoa_df$Sample), "Source A",
                  ifelse(grepl("A_b", pcoa_df$Sample), "Source B", "Source C"))

# 画 95% 置信椭圆
p3 <- ggplot(pcoa_df, aes(x = PC1, y = PC2, color = Source, fill = Source)) +
  geom_point(size = 5, shape = 21, color = "black", stroke = 1) +
  geom_text(aes(label = Sample), vjust = -1.2, size = 3.5, fontface = "bold") +
  stat_ellipse(geom = "polygon", alpha = 0.15, level = 0.95, linetype = 2) +
  scale_fill_manual(values = cols_group) +
  scale_color_manual(values = cols_group) +
  labs(title = "PCoA (Bray-Curtis) — Agarwood Fungal Communities",
       x = paste0("PCoA1 (", round(eig_pct[1], 1), "%)"),
       y = paste0("PCoA2 (", round(eig_pct[2], 1), "%)")) +
  theme_pub +
  theme(legend.position = "top")

ggsave("FungalAnalysis/figures/v2/fig3_pcoa_braycurtis.png", p3,
       width = 8, height = 7, dpi = 300)

# Jaccard PCoA
jac <- vegdist(asv_tab_t, method = "jaccard")
pcoa_jac <- cmdscale(jac, k = 2, eig = TRUE)
eig_jac <- pcoa_jac$eig / sum(pcoa_jac$eig) * 100

pcoa_jac_df <- data.frame(
  Sample = rownames(pcoa_jac$points),
  PC1 = pcoa_jac$points[, 1],
  PC2 = pcoa_jac$points[, 2]
)
pcoa_jac_df$Source <- ifelse(grepl("A_a", pcoa_jac_df$Sample), "Source A",
                      ifelse(grepl("A_b", pcoa_jac_df$Sample), "Source B", "Source C"))

p3b <- ggplot(pcoa_jac_df, aes(x = PC1, y = PC2, color = Source, fill = Source)) +
  geom_point(size = 5, shape = 21, color = "black", stroke = 1) +
  geom_text(aes(label = Sample), vjust = -1.2, size = 3.5, fontface = "bold") +
  stat_ellipse(geom = "polygon", alpha = 0.15, level = 0.95, linetype = 2) +
  scale_fill_manual(values = cols_group) +
  scale_color_manual(values = cols_group) +
  labs(title = "PCoA (Jaccard) — Agarwood Fungal Communities",
       x = paste0("PCoA1 (", round(eig_jac[1], 1), "%)"),
       y = paste0("PCoA2 (", round(eig_jac[2], 1), "%)")) +
  theme_pub +
  theme(legend.position = "top")

ggsave("FungalAnalysis/figures/v2/fig3b_pcoa_jaccard.png", p3b,
       width = 8, height = 7, dpi = 300)

# ============================================================
# 4. 物种组成堆叠柱状图（Top 10 + Other）
# ============================================================
cat(">>> 绘制图4: 物种组成堆叠柱状图\n")

tax <- read.csv("FungalAnalysis/dada2_results/asv_taxonomy_blast.csv")
top20 <- read.csv("FungalAnalysis/dada2_results/top20_ASVs.csv")

# 只取 A 组列
pct_cols <- c("A_a_1_pct", "A_a_2_pct", "A_b_1_pct", "A_b_2_pct", "A_c_1_pct", "A_c_2_pct")
comp <- top20[, c("ASV", pct_cols)]

# 合并物种名
comp <- merge(comp, tax[, c("ASV", "Clean_Species")], by = "ASV", all.x = TRUE)

# 合并同种 ASV
comp_agg <- aggregate(comp[, pct_cols], by = list(Species = comp$Clean_Species), FUN = sum)

# 按总丰度排序
comp_agg$Total <- rowSums(comp_agg[, pct_cols])
comp_agg <- comp_agg[order(-comp_agg$Total), ]

# Top 10 + Other
top_n <- 10
if (nrow(comp_agg) > top_n) {
  other_row <- colSums(comp_agg[(top_n + 1):nrow(comp_agg), pct_cols])
  top_comp <- comp_agg[1:top_n, ]
  other_df <- data.frame(Species = "Other", t(other_row), Total = sum(other_row))
  colnames(other_df) <- colnames(top_comp)
  top_comp <- rbind(top_comp, other_df)
} else {
  top_comp <- comp_agg
}

comp_melt <- melt(top_comp, id.vars = "Species", measure.vars = pct_cols)
colnames(comp_melt) <- c("Species", "Sample", "Abundance")
comp_melt$Sample <- gsub("_pct", "", comp_melt$Sample)

comp_melt$Species <- factor(comp_melt$Species,
                            levels = rev(c(as.character(top_comp$Species[top_comp$Species != "Other"]), "Other")))

# 配色
n_species <- length(unique(comp_melt$Species))
species_colors <- colorRampPalette(brewer.pal(12, "Set3"))(n_species)

p4 <- ggplot(comp_melt, aes(x = Sample, y = Abundance, fill = Species)) +
  geom_bar(stat = "identity", width = 0.7, color = "black", linewidth = 0.2) +
  scale_fill_manual(values = species_colors) +
  labs(title = "Fungal Community Composition — Agarwood Samples (Top 10 Species)",
       x = "Sample", y = "Relative Abundance (%)") +
  theme_pub +
  theme(legend.position = "right",
        axis.text.x = element_text(angle = 0, hjust = 0.5))

ggsave("FungalAnalysis/figures/v2/fig4_species_composition.png", p4,
       width = 10, height = 6, dpi = 300)

# ============================================================
# 5. 热力图（Top 20 ASVs × 6 沉香样本）
# ============================================================
cat(">>> 绘制图5: 热力图\n")

top20_mat <- top20[, pct_cols]
rownames(top20_mat) <- top20$ASV
colnames(top20_mat) <- agarwood_samples

# 行注释：物种名
row_anno <- data.frame(Species = tax$Clean_Species[match(top20$ASV, tax$ASV)])
rownames(row_anno) <- top20$ASV

# 列注释：来源
col_anno <- data.frame(
  Source = c("Source A", "Source A", "Source B", "Source B", "Source C", "Source C")
)
rownames(col_anno) <- agarwood_samples

ann_colors <- list(
  Source = c("Source A" = "#E64B35", "Source B" = "#4DBBD5", "Source C" = "#00A087"),
  Species = colorRampPalette(brewer.pal(12, "Set3"))(length(unique(row_anno$Species)))
)
names(ann_colors$Species) <- unique(row_anno$Species)

png("FungalAnalysis/figures/v2/fig5_heatmap_top20.png",
    width = 10, height = 8, units = "in", res = 300)
pheatmap(top20_mat,
         annotation_row = row_anno,
         annotation_col = col_anno,
         annotation_colors = ann_colors,
         color = colorRampPalette(c("#F7FBFF", "#6BAED6", "#2171B5", "#08306B"))(100),
         scale = "row",
         cluster_cols = TRUE,
         cluster_rows = TRUE,
         show_rownames = TRUE,
         show_colnames = TRUE,
         fontsize = 11,
         fontsize_row = 8,
         main = "Heatmap of Top 20 ASVs — Agarwood Samples")
dev.off()

# ============================================================
# 6. Top 10 物种相对丰度对比柱状图
# ============================================================
cat(">>> 绘制图6: Top 10 物种丰度柱状图\n")

# 按来源聚合
source_map <- c("A_a_1" = "Source A", "A_a_2" = "Source A",
                "A_b_1" = "Source B", "A_b_2" = "Source B",
                "A_c_1" = "Source C", "A_c_2" = "Source C")

top10_species <- comp_agg$Species[1:min(10, nrow(comp_agg))]

# 计算每个来源的平均丰度
source_abundance <- data.frame(Species = top10_species)
for (sp in c("Source A", "Source B", "Source C")) {
  sp_samples <- names(source_map[source_map == sp])
  sp_cols <- paste0(sp_samples, "_pct")
  source_abundance[[sp]] <- rowMeans(comp_agg[match(top10_species, comp_agg$Species), sp_cols])
}

bar_df <- melt(source_abundance, id.vars = "Species")
colnames(bar_df) <- c("Species", "Source", "Abundance")

p6 <- ggplot(bar_df, aes(x = reorder(Species, -Abundance), y = Abundance, fill = Source)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  scale_fill_manual(values = cols_group) +
  labs(title = "Top 10 Species — Mean Relative Abundance by Source",
       x = "", y = "Mean Relative Abundance (%)") +
  theme_pub +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("FungalAnalysis/figures/v2/fig6_top10_species_by_source.png", p6,
       width = 10, height = 6, dpi = 300)

# ============================================================
# 7. 沉香来源间 Bray-Curtis 距离热力图
# ============================================================
cat(">>> 绘制图7: Bray-Curtis 距离热力图\n")

bc_mat <- as.matrix(bc)
png("FungalAnalysis/figures/v2/fig7_braycurtis_distance.png",
    width = 7, height = 6, units = "in", res = 300)
pheatmap(bc_mat,
         color = colorRampPalette(c("#FFF7EC", "#FEE8C8", "#FDD49E", "#FDBB84",
                                     "#E34A33", "#B30000", "#7F0000"))(100),
         cluster_rows = TRUE,
         cluster_cols = TRUE,
         display_numbers = TRUE,
         number_format = "%.3f",
         fontsize_number = 9,
         fontsize = 11,
         main = "Bray-Curtis Dissimilarity — Agarwood Samples")
dev.off()

# ============================================================
# 8. 稀释曲线（Rarefaction Curve）
# ============================================================
cat(">>> 绘制图8: 稀释曲线\n")

set.seed(42)
raremax <- min(colSums(asv_tab))
rare_curves <- rarescurve(asv_tab_t, sample = raremax, step = 500)

# 提取数据用于ggplot
rare_df <- data.frame()
for (i in seq_along(agarwood_samples)) {
  tmp <- data.frame(
    Reads = seq(1, length(rare_curves[[i]]), by = 1) * 500,
    ASVs = rare_curves[[i]][seq(1, length(rare_curves[[i]]), by = 1)],
    Sample = agarwood_samples[i]
  )
  # 实际上 rarescurve 返回的是矩阵，需要调整
  rare_df <- rbind(rare_df, tmp)
}

# 用 vegan 的 rarecurve 提取
rc <- rarecurve(asv_tab_t, step = 500, sample = raremax)

rare_df <- data.frame()
for (i in 1:length(rc)) {
  n <- as.numeric(names(rc[[i]]))
  s <- as.vector(rc[[i]])
  tmp <- data.frame(Reads = n, ASVs = s, Sample = agarwood_samples[i])
  rare_df <- rbind(rare_df, tmp)
}

rare_df$Source <- ifelse(grepl("A_a", rare_df$Sample), "Source A",
                  ifelse(grepl("A_b", rare_df$Sample), "Source B", "Source C"))

p8 <- ggplot(rare_df, aes(x = Reads, y = ASVs, color = Source, group = Sample)) +
  geom_line(linewidth = 1) +
  geom_point(data = rare_df[rare_df$Sample %in% agarwood_samples, ],
             aes(fill = Source), size = 3, shape = 21, color = "black", stroke = 0.5) +
  scale_color_manual(values = cols_group) +
  scale_fill_manual(values = cols_group) +
  labs(title = "Rarefaction Curves — Agarwood Samples",
       x = "Sequencing Depth (Reads)", y = "Observed ASVs") +
  theme_pub +
  theme(legend.position = "top")

ggsave("FungalAnalysis/figures/v2/fig8_rarefaction_curve.png", p8,
       width = 8, height = 6, dpi = 300)

# ============================================================
# 9. 属水平汇总饼图（每个来源一个）
# ============================================================
cat(">>> 绘制图9: 属水平饼图\n")

# 从物种名提取属名
comp_agg$Genus <- sapply(strsplit(as.character(comp_agg$Species), " "), `[`, 1)

# 按属合并
genus_agg <- aggregate(comp_agg[, pct_cols],
                       by = list(Genus = comp_agg$Genus), FUN = sum)

# 按来源汇总
genus_df <- data.frame(Genus = genus_agg$Genus)
genus_df$Source_A <- rowMeans(genus_agg[, c("A_a_1_pct", "A_a_2_pct")])
genus_df$Source_B <- rowMeans(genus_agg[, c("A_b_1_pct", "A_b_2_pct")])
genus_df$Source_C <- rowMeans(genus_agg[, c("A_c_1_pct", "A_c_2_pct")])

# Top 8 属 + Other
top_genera <- genus_agg$Genus[order(-rowSums(genus_agg[, pct_cols]))]
top_genera <- top_genera[1:min(8, length(top_genera))]

make_pie <- function(df, col, title_text) {
  df_sub <- df[, c("Genus", col)]
  colnames(df_sub) <- c("Genus", "Abundance")
  df_sub <- df_sub[order(-df_sub$Abundance), ]
  
  top <- df_sub[df_sub$Genus %in% top_genera, ]
  other_val <- sum(df_sub$Abundance[!df_sub$Genus %in% top_genera])
  if (other_val > 0) {
    top <- rbind(top, data.frame(Genus = "Other", Abundance = other_val))
  }
  
  top$Genus <- factor(top$Genus, levels = top$Genus[order(-top$Abundance)])
  
  n_col <- length(unique(top$Genus))
  pie_colors <- colorRampPalette(brewer.pal(9, "Set1"))(n_col)
  
  ggplot(top, aes(x = "", y = Abundance, fill = Genus)) +
    geom_bar(stat = "identity", width = 1, color = "white", linewidth = 0.5) +
    coord_polar("y", start = 0) +
    scale_fill_manual(values = pie_colors) +
    labs(title = title_text) +
    theme_void() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
          legend.text = element_text(size = 10),
          legend.title = element_blank())
}

p9a <- make_pie(genus_df, "Source_A", "Source A (A_a)")
p9b <- make_pie(genus_df, "Source_B", "Source B (A_b)")
p9c <- make_pie(genus_df, "Source_C", "Source C (A_c)")

p9_combined <- arrangeGrob(p9a, p9b, p9c, nrow = 1)
ggsave("FungalAnalysis/figures/v2/fig9_genus_pie_charts.png", p9_combined,
       width = 16, height = 6, dpi = 300)

# ============================================================
# 完成
# ============================================================
cat("\n========================================\n")
cat("所有图表已生成完毕！仅包含沉香 6 样本。\n")
cat("输出目录: FungalAnalysis/figures/v2/\n")
cat("========================================\n")
