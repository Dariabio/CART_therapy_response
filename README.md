# Clinical trial: Analysis of CAR-T product in responder and nonresponder patients

This repository contains the bioinformatics workflow for analysis of CAR-T product RNA-seq data from a clinical trial.

The main goal is to characterize molecular differences between responders and nonresponders and to identify biological pathways and CAR-T-related molecular signatures associated with clinical response.

The analysis includes:

RNA-seq expression matrix preparation and gene annotation
Clinical metadata preparation
VST normalization and PCA
Differential expression analysis with DESeq2
Volcano plot visualization
Ranked GSEA
ssGSEA based on predefined CAR-T gene panels
Statistical comparison of ssGSEA scores between clinical groups
Visualization and integration of GSEA results
Longitudinal analysis of CAR-T signatures across timepoints
Signature discovery of response

# Project structure

The analysis is organized as a Nextflow workflow. Individual analysis steps are implemented as Python scripts in the `bin/` directory.

### `bin/`

Contains the main scripts executed by the Nextflow workflow:

- `01_prepare_expression.py ` — preprocessing of expression data
- `02_prepare_metadata.py` — preprocessing of clinical data
- `03_vst_pca.py` — normalization and PCA plots
- `04_de.py` — differential expression analysis between groups of clinical factors
- `05_volcano_plot.py` — creation of Volcano plots
- `06_ranked_GSEA.py` — preparation of gene panels and running ranked GSEA
- `07_ssgsea.py` — running ssGSEA
- `08_ssgsea_statistics.py` — Statistical analysis of ssGSEA score
- `09_visualization_GSEA.py` — heatmaps, dotplots of ranked GSEA, ssGSEA results
- `10_dynamics.py` — Interaction and delta analysis
- `11_global_gsea.py` — running ranked GSEA using GO db
- `12_timepoints_analysis.py` — differential expression analysis between timepoints, ranked GSEA GO

### `independent_cohorts/`

Contains independent scripts for checking CAR-T product DEGs of our cohort and predefined signatures in three independent cohorts

- `GSE151511_analysis.py`
- `GSE197628_analysis.py`
- `GSE223655_analysis.py`

# INPUT

Four CAR-T-related panels are currently used for ssGSEA and ranked GSEA analyses

The pipeline currently performs analyses for the following clinical variables:

group_analysis_1
response
aggressiveness
bone_marrow
lesion

Timepoints:

product
d7 after
d28 after

The baseline timepoint is the CAR-T product.

## Pipeline steps
# 01. Expression preparation

Script:

bin/01_prepare_expression.py

This step:

reads individual expression files
combines samples into a single expression matrix
separates TPM and raw counts
performs gene annotation
produces matrices used by downstream analyses

Outputs:

expression_tpm.tsv
expression_raw.tsv
expression_tpm_ann.tsv
expression_raw_ann.tsv

# 02. Metadata preparation

Script:

bin/02_prepare_metadata.py

This step:

combines clinical datasets
matches clinical information to expression samples
prepares analysis groups
incorporates sequencing batch information
generates the final metadata table

Outputs:

counts.tsv
meta_analysis.tsv

# 03. VST normalization and PCA

Script:

bin/03_vst_pca.py

Raw counts are processed using DESeq2 variance-stabilizing transformation (VST).

PCA is performed to evaluate:

overall sample structure
clinical group separation
timepoint effects
patient effects
experimental batch effects
sequencing batch effects

Output:

expr_all_vst.tsv

and PCA figures.

# 04. Differential expression analysis

Script:

bin/04_de.py

Differential expression analysis is performed separately for each clinical comparison and timepoint.

The analysis uses DESeq2 with experimental batch included in the design.

The resulting DE tables are generated for:

product
d7
d28

These tables are subsequently used for visualization and ranked GSEA.

# 05. Volcano plots

Script:

bin/05_volcano_plot.py

Volcano plots are generated from the DESeq2 results to visualize:

log2 fold change
statistical significance
significantly upregulated genes
significantly downregulated genes

# 06. Ranked GSEA - gene panels

Script:

bin/06_ranked_gsea.py

Ranked gene lists generated from differential expression analysis are used for GSEA.

The analysis is performed using the predefined CAR-T gene panels.

This allows pathway/signature enrichment to be assessed using the complete ranked gene list rather than only significantly differentially expressed genes.

# 07. ssGSEA

Script:

bin/07_ssgsea.py

Single-sample GSEA (ssGSEA) is performed using the predefined CAR-T gene panels.

Each sample receives an enrichment score for each gene set.

The resulting matrix contains:

samples × CAR-T signatures

Output:

scores_all.csv

# 08. Statistical analysis of ssGSEA scores

Script:

bin/08_ssgsea_statistics.py

ssGSEA scores are statistically compared between clinical groups.

The analysis is performed separately for the predefined clinical comparisons.

The resulting statistics are used for downstream visualization and interpretation.

# 09. Visualization of GSEA results

Script:

bin/09_visualization_GSEA.py

This step integrates:

DESeq2 results
ranked GSEA results
ssGSEA statistics
clinical metadata

and generates summary visualizations of pathway/signature differences.

# 10. Longitudinal analysis

Script:

bin/10_dynamics.py

Longitudinal ssGSEA scores are analyzed across:

product → d7 after → d28 after

Two complementary approaches are used:

Interaction analysis

A mixed-effects model is used to test whether the trajectory over time differs between clinical groups.

Conceptually:

signature score ~ timepoint × clinical group

with patient included as a random effect.

The key question is:

Does the change in a molecular signature over time differ between responders and nonresponders?

Delta analysis

Changes relative to the CAR-T product baseline are calculated:

Δd7  = d7 − product
Δd28 = d28 − product

The delta values are then compared between clinical groups.

# 11. Ranked GSEA - GO database

Script:

bin/11_global_gsea.py

Ranked gene lists generated from differential expression analysis are used for GSEA.

The analysis is performed using GO database.

This allows pathway/signature enrichment to be assessed using the complete ranked gene list rather than only significantly differentially expressed genes.


# 12. Timepoint analysis

Script:

bin/12_timepoints_analysis.py

Unsupervised clustering by variable genes across timepoints.
Differential expression analysis d28 vs product, d7 vs product.
Visualization via heatmaps.
Ranked gene lists generated from differential expression analysis are used for GSEA using GO database.

# 13. Signature search

Script:

bin/13_signature.py

Create DEGs product-response-assosiated signature
Logistic univariate and multivariable regression - effect of clinical factors on "signature"
Search signature via Elastic Net

# 14. Signature statistics

Script:

bin/14_signature_statistics.py

Logistic univariate and multivariable regression of significant signatures by ssGSEA (responders vs nonresponders) - effect of clinical factors

## Separate scripts


# GSE151511_analysis
# GSE197628_analysis

Scripts:

independent_cohorts/GSE151511_analysis.py
independent_cohorts/GSE197628_analysis.py

Preprocessing of scRNA-seq data, filtering, calculating signature score, test, ROC_AOC, pseudobulk, differential expression analysis, ranked GSEA using predefined panels, visualization

# GSE223655_analysis

Script:

independent_cohorts/GSE223655_analysis.py

Preprocessing of data, filtering, calculating signature score, test, ROC_AOC, differential expression analysis, ssGSEA, ranked GSEA using predefined panels, visualization


# Reproducibility

The analysis is implemented in Nextflow DSL2.

The main workflow is:

main.nf

Each analysis step is implemented as an independent Nextflow process, allowing:

reproducible execution
process-level caching
restarting failed analyses
parallel execution of independent comparisons
separation of intermediate files from final results

To rerun the pipeline using cached results:

nextflow run main.nf -resume

# Installation

Create the Conda environment:

conda env create -f environment.yml

Activate it:

conda activate cart-analysis

Make sure Nextflow is installed:

nextflow -version

The pipeline currently uses Nextflow DSL2.

Running the pipeline

From the project root:

nextflow run main.nf

To resume an interrupted or previously completed run:

nextflow run main.nf -resume

Nextflow stores temporary process directories in:

work/

Final outputs are published to:

results/
Results

The main results directories are:

Directory	Content
01_prepare_expression/	Prepared expression matrices
02_prepare_metadata/	Counts and analysis metadata
03_vst_pca/	VST expression and PCA
04_de/	DESeq2 results
05_volcano_plot/	Volcano plots
06_ranked_gsea/	Ranked GSEA results, predefined gene panels
07_ssgsea/	ssGSEA scores
08_ssgsea_statistics/	Statistical analysis of ssGSEA
09_visualization_GSEA/	Integrated GSEA visualizations
10_dynamics/	Longitudinal signature analysis
11_global_gsea.pys/	Ranked GSEA results, GO db
12_timepoints_analysis.py/ Visualization across timepoints, DEGs results, ranked GSEA results
13_signature.py/ Signatures: DEGs, searched by elastic net, model parametrs and perfomance
14_signature_statistics.py/ results of logistic univariate and multivariable regression
GSE151511/ signature score, heatmap of signature genes, statistic results, ROC curve, GSEA results
GSE197628/ signature score, heatmap of signature genes, statistic results, ROC curve, GSEA results
GSE223655/ signature score, heatmap of signature genes, statistic results, ROC curve, GSEA results

# Software

The workflow uses:

Nextflow
Python
Conda
DESeq2 / PyDESeq2
GSEA / ssGSEA
pandas
NumPy
SciPy
statsmodels
scikit-learn
matplotlib
seaborn

# Exact package versions should be specified in:

environment.yml

Repository policy

Raw clinical data, large expression matrices, generated results, and Nextflow working directories are excluded using .gitignore.
