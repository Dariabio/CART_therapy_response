# Clinical trial: Analysis of CAR-T product in responder and nonresponder patients
## Start
1. `conda env create -f environment.yml`
2. `conda activate crc-analysis`
car_t_analysis: DE and GSEA
3. Add batches to clinical data. Match clinical data and expression. Prepare gene sets
4. Preprocessing (one matrix) and gene annotation of data
5. Design, normalization with deseq2 and PCA to see data clusterization
6. DE analysis per each timepoint - DE list for GSEA ???
7. 4 gene panels (CAR-T signatures) processing ssGSEA
8. ssGSEA 4 panels and test - responders vs nonresponders
9. ranked GSEA 4 panels
10. ranked GSEA GO and KEGG
