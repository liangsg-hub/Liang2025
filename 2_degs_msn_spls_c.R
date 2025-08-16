##############################################################################
###  Identify peripheral blood gene sets that robustly covary with HA imaging phenotype
##############################################################################
rm(list = ls())
options(stringsAsFactors = FALSE)
Sys.setlocale("LC_CTYPE", "en_US.UTF-8")

plan(multisession, workers = parallel::detectCores() - 1)
options(future.seed = 42)
RNGkind("L'Ecuyer-CMRG")
set.seed(42)

########################  Parameter section  #################################
DIR_DATA  <- "/msn_degs/mRNA_data"
FILE_EXPR <- "edgeR_DEG_MDD_vs_HC_DEGexpr_logCPM_FDR0p05.csv"
FILE_MSN  <- "hc_mdd_msn_8_measures.xlsx"
N_COMP    <- 1                # Only 1 component
GRID_X    <- c(3,5,7,9)       # Grid for number of ROIs
GRID_Y <- seq(15, 45, by = 5) # Grid for number of genes
NBOOT     <- 10000            # Number of bootstrap iterations
##############################################################################
suppressPackageStartupMessages({
  library(readr); library(readxl); library(dplyr); library(tibble)
  library(stringr); library(mixOmics); library(future.apply)
  library(ComplexHeatmap); library(circlize)
})

plan(multisession, workers = parallel::detectCores() - 1)

### 1. Data loading and preprocessing ----------------------------------------
expr_mat <- read_csv(file.path(DIR_DATA, FILE_EXPR), show_col_types = FALSE) |>
  column_to_rownames("gene_symbol") |>
  as.matrix()

msn_df   <- read_xlsx(file.path(DIR_DATA, FILE_MSN), sheet = 1)

## 1.1 Handle duplicate IDs
msn_df$ID <- make.unique(msn_df$ID, sep = "_dup")
rownames(expr_mat) <- make.unique(rownames(expr_mat), sep = "_dup")

## 1.2 Group vector
group_vec <- msn_df |>
  dplyr::select(ID, Subgroup) |>
  tibble::deframe()

## 1.3 MSN values for 152 ROIs
msn_mat <- msn_df |>
  column_to_rownames("ID") |>
  dplyr::select(where(is.numeric)) |>
  dplyr::select(1:152) |>
  dplyr::mutate(across(everything(), ~ replace(., is.infinite(.), NA))) |>
  as.matrix()

msn_mat <- msn_mat[complete.cases(msn_mat), ]
expr_mat <- expr_mat[, apply(expr_mat, 2, sd) > 0]
expr_mat <- expr_mat[complete.cases(expr_mat), ]

### 2. Sample alignment and grouping -----------------------------------------
common <- intersect(colnames(expr_mat), rownames(msn_mat))
msn_all  <- msn_mat[common, ]
expr_all <- t(expr_mat)[common, ]
group_all<- group_vec[common]

X_HA <- msn_all[group_all == "HA", ]
Y_HA <- expr_all[group_all == "HA", ]
X_LA <- msn_all[group_all == "LA", ]
Y_LA <- expr_all[group_all == "LA", ]
X_HC <- msn_all[group_all == "HC", ]
Y_HC <- expr_all[group_all == "HC", ]

## Z-score normalization using HA mean and standard deviation
sc <- function(mat, mu = NULL, sdv = NULL){
  scale(mat, center = mu %||% TRUE, scale = sdv %||% TRUE)
}

X_HA <- sc(X_HA)
Y_HA <- sc(Y_HA)
X_LA <- sc(X_LA, attr(X_HA,"scaled:center"), attr(X_HA,"scaled:scale"))
Y_LA <- sc(Y_LA, attr(Y_HA,"scaled:center"), attr(Y_HA,"scaled:scale"))
X_HC <- sc(X_HC, attr(X_HA,"scaled:center"), attr(X_HA,"scaled:scale"))
Y_HC <- sc(Y_HC, attr(Y_HA,"scaled:center"), attr(Y_HA,"scaled:scale"))

### 3. Parameter tuning (tune.spls) ------------------------------------------
n_HA  <- nrow(X_HA)
kfold <- max(3, min(5, n_HA))
tune <- tune.spls(
  X = X_HA, Y = Y_HA,
  ncomp       = 1,
  test.keepX  = GRID_X,
  test.keepY  = GRID_Y,
  validation  = "Mfold",
  folds       = kfold,
  nrepeat     = 1,
  mode        = "canonical",
  measure     = "cor",
  progressBar = FALSE
)

keepX_opt <- tune$choice.keepX
keepY_opt <- tune$choice.keepY
cat("Tuning result: keepX =", keepX_opt, "  keepY =", keepY_opt, "\n")

######################## 4. Stability selection ##############################

stabX <- matrix(0, nrow = ncol(X_HA), ncol = NBOOT,
                dimnames = list(colnames(X_HA), NULL))
stabY <- matrix(0, nrow = ncol(Y_HA), ncol = NBOOT,
                dimnames = list(colnames(Y_HA), NULL))

skip <- 0L
for(b in 1:NBOOT){
  samp <- sample(seq_len(nrow(X_HA)), size = floor(0.8*nrow(X_HA)), replace = TRUE)
  Xb <- X_HA[samp, ];  Yb <- Y_HA[samp, ]
  rownames(Xb) <- rownames(Yb) <- paste0("boot", b, "_", seq_along(samp))

  ok <- tryCatch({
    mod <- spls(Xb, Yb,
                ncomp  = 1,
                keepX  = keepX_opt,
                keepY  = keepY_opt,
                mode   = "canonical",
                max.iter = 500, tol = 1e-07)
    TRUE
  }, warning = function(w){
    if(grepl("did not converge", w$message)) return(FALSE)
    invokeRestart("muffleWarning")
  })

  if(!ok){ skip <- skip + 1; next }

  stabX[selectVar(mod,1)$X$name, b] <- 1
  stabY[selectVar(mod,1)$Y$name, b] <- 1
}
cat("Number of bootstrap iterations not converged:", skip, "/", NBOOT, "\n")

THR_ROI  <- quantile(rowMeans(stabX), 0.75)
THR_GENE <- quantile(rowMeans(stabY), 0.25)
roi_sel  <- names(which(rowMeans(stabX) >= THR_ROI))
gene_sel <- names(which(rowMeans(stabY) >= THR_GENE))

cat("Stable ROIs:", length(roi_sel), "  Stable genes:", length(gene_sel), "\n")
stopifnot(length(roi_sel) >= 5, length(gene_sel) >= 15)

######################## 5. Final model (HA group) ##########################
final <- spls(
  X      = X_HA[, roi_sel],
  Y      = Y_HA[, gene_sel],
  ncomp  = 1,
  keepX  = length(roi_sel),
  keepY  = length(gene_sel),
  mode   = "canonical",
  max.iter = 500, tol = 1e-07
)

# HA scores
U_HA <- final$variates$X[,1]
V_HA <- final$variates$Y[,1]

################ 6. Projection to LA and HC groups ################

proj <- function(Xnew, Ynew){
  U <- as.numeric(predict(final, Xnew, type = "scores")$variates)
  V <- as.numeric(Ynew %*% final$loadings$Y[,1])
  list(U = U, V = V)
}
sc_LA <- proj(X_LA[, roi_sel], Y_LA[, gene_sel])
sc_HC <- proj(X_HC[, roi_sel], Y_HC[, gene_sel])

######################## 7. Permutation test ###############################
perm_p <- function(U, V, nperm = 5000){
  r0   <- cor(U, V)
  null <- replicate(nperm, cor(U, sample(V)))
  (sum(null >= abs(r0)) + 1) / (nperm + 1)
}

r_HA <- cor(U_HA,  V_HA);  p_HA <- perm_p(U_HA,  V_HA)
r_LA <- cor(sc_LA$U, sc_LA$V); p_LA <- perm_p(sc_LA$U, sc_LA$V)
r_HC <- cor(sc_HC$U, sc_HC$V); p_HC <- perm_p(sc_HC$U, sc_HC$V)

### 8. Output ---------------------------------------------------
cat("\n========  Results  ========\n")
cat("HA : r =", round(r_HA,3), "  p =", signif(p_HA,3), "\n")
cat("LA : r =", round(r_LA,3), "  p =", signif(p_LA,3), "\n")
cat("HC : r =", round(r_HC,3), "  p =", signif(p_HC,3), "\n")

write.csv(data.frame(ROI = roi_sel),
          file.path(DIR_DATA, "ROI_stable.csv"), row.names = FALSE)
write.csv(data.frame(Gene = gene_sel),
          file.path(DIR_DATA, "GENE_stable.csv"), row.names = FALSE)

#### Set output directory ----------------------------------------------------
out_dir <- file.path(DIR_DATA, "plsc_outputs")
if(!dir.exists(out_dir)) dir.create(out_dir)

#### 1) Sample scores (scores / variates) ------------------------------------
write.csv(
  cbind(Sample = rownames(final$variates$X), final$variates$X),
  file = file.path(out_dir, "scores_X.csv"),
  row.names = FALSE
)

write.csv(
  cbind(Sample = rownames(final$variates$Y), final$variates$Y),
  file = file.path(out_dir, "scores_Y.csv"),
  row.names = FALSE
)

#### 2) Loadings -------------------------------------------------------------
write.csv(
  cbind(Variable = rownames(final$loadings$X), final$loadings$X),
  file = file.path(out_dir, "loadings_X.csv"),
  row.names = FALSE
)

write.csv(
  cbind(Variable = rownames(final$loadings$Y), final$loadings$Y),
  file = file.path(out_dir, "loadings_Y.csv"),
  row.names = FALSE
)

#### 3) Proportion of explained variance -------------------------------------
pev_X <- data.frame(Component = seq_along(final$prop_expl_var$X),
                    Prop      = final$prop_expl_var$X)
write.csv(pev_X,
          file = file.path(out_dir, "prop_expl_var_X.csv"),
          row.names = FALSE)

pev_Y <- data.frame(Component = seq_along(final$prop_expl_var$Y),
                    Prop      = final$prop_expl_var$Y)
write.csv(pev_Y,
          file = file.path(out_dir, "prop_expl_var_Y.csv"),
          row.names = FALSE)

cat("All CSV files saved to:", out_dir, "\n")

pev_X
pev_Y

## Organize data frame for plotting
cv_df <- dplyr::bind_rows(
  data.frame(Group = "HA", U = U_HA,          V = V_HA),
  data.frame(Group = "LA", U = sc_LA$U,       V = sc_LA$V),
  data.frame(Group = "HC", U = sc_HC$U,       V = sc_HC$V)
)

## Plot scatter plot (facet mode)
library(ggplot2)

p_cv <- ggplot(cv_df, aes(x = U, y = V, colour = Group)) +
  geom_point(size = 3.5, alpha = 1.0) +
  stat_smooth(method = "lm", se = FALSE, linetype = 2) +
  facet_wrap(~ Group, nrow = 1) +
  labs(x = "Canonical variate U (ROI side)",
       y = "Canonical variate V (Gene side)",
       title = "sPLS-C canonical variates by subgroup") +
  theme_bw(base_size = 11) +
  theme(legend.position = "none")

ggsave(file.path(DIR_DATA, "CanonicalVariates_U_vs_V.pdf"),
       p_cv, width = 9, height = 3.5)

## Combined plot for all groups
p_cv_combined <- ggplot(cv_df, aes(U, V, colour = Group, shape = Group)) +
  geom_point(size = 3.5, alpha = 1.0) +
  stat_smooth(aes(group = Group), method = "lm", se = FALSE, linetype = 2) +
  scale_colour_manual(
    values = c(
      "HC" = "#d5d5d5",
      "HA" = "#e41a1c",
      "LA" = "#377eb8"
    )
  ) +
  labs(x = "Canonical variate U (IDP)",
       y = "Canonical variate V (Genes)",
       colour = NULL, shape = NULL,
       title  = "Canonical variates") +
  coord_cartesian(ylim = c(-10, 10)) +
  theme_bw(base_size = 11)

ggsave(file.path(DIR_DATA, "CanonicalVariates_AllGroups.pdf"),
       p_cv_combined, width = 5, height = 4.5)

#### ------------------------------------------------------------------ ####
####  Figure S4  Heatmap of 28 imaging-gene associations               ####
#### ------------------------------------------------------------------ ####

setwd('/msn_degs/mRNA_data/plsc_outputs_fdr0p05_0730')

library(pheatmap)
library(RColorBrewer)

## 1.  Choose genes and order ROIs  ------------------------------------------
ng <- 30
gene_ord <- names(sort(abs(final$loadings$Y[,1]), decreasing = TRUE))[1:ng]
roi_ord  <- names(sort(abs(final$loadings$X[,1]), decreasing = TRUE))

## 2.  Association matrix = outer product of loadings  -----------------------
assoc_mat <- outer(final$loadings$Y[gene_ord, 1],
                   final$loadings$X[roi_ord,  1],
                   FUN = "*")
rownames(assoc_mat) <- gene_ord
colnames(assoc_mat) <- roi_ord

## Optional row-wise z-score
assoc_mat_z <- t(scale(t(assoc_mat)))
assoc_mat_z[!is.finite(assoc_mat_z)] <- 0

## 3.  Simple row annotation: loading direction  -----------------------------
ann_row <- data.frame(Direction =
                        ifelse(final$loadings$Y[gene_ord,1] > 0,"Positive","Negative"))
rownames(ann_row) <- gene_ord
ann_cols <- list(Direction = c(Positive = "#e41a1c", Negative = "#377eb8"))

## 4.  Draw and save  --------------------------------------------------------
heat_file <- file.path("Fig_heatmap_top30.pdf")
pheatmap(assoc_mat_z,
         color = colorRampPalette(rev(brewer.pal(11, "RdBu")))(100),
         border_color = NA,
         cluster_rows = TRUE, cluster_cols = TRUE,
         clustering_distance_rows = "correlation",
         clustering_distance_cols = "correlation",
         annotation_row = ann_row,
         annotation_colors = ann_cols,
         fontsize_row = 6.5, fontsize_col = 7,
         cellheight = 10, cellwidth = 18,
         main = "Figure Imaging-gene associations",
         filename = heat_file,
         width = 6, height = 5)

cat("Figure heatmap saved to:", heat_file, "\n")