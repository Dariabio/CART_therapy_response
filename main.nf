nextflow.enable.dsl=2


// ============================================================
// PARAMETERS
// ============================================================

params.input_dir = "${projectDir}/data/raw/sample_expression"
params.annotation_dir = "${projectDir}/data/annotation"
params.panel_dir = "${projectDir}/data/raw"
params.results_dir = "${projectDir}/results"


// ============================================================
// PROCESS 1 — PREPARE EXPRESSION
// ============================================================

process PREPARE_EXPRESSION {

    tag "Expression preparation"

    publishDir(
        "${params.results_dir}/01_prepare_expression",
        mode: "copy"
    )

    input:

    path sample_files
    path annotation_files

    output:

    path "expression_tpm.tsv", emit: expression_tpm
    path "expression_raw.tsv", emit: expression_raw
    path "expression_tpm_ann.tsv", emit: expression_tpm_ann
    path "expression_raw_ann.tsv", emit: expression_raw_ann

    script:

    """
    mkdir -p sample_expression
    mkdir -p annotation

    cp ${sample_files} sample_expression/
    cp ${annotation_files} annotation/

    /opt/anaconda3/envs/crc_scanpy/bin/python \
        ${projectDir}/bin/01_prepare_expression.py \
        --input_dir sample_expression \
        --annotation_dir annotation \
        --output_dir .
    """
}


// ============================================================
// PROCESS 2 — PREPARE METADATA
// ============================================================

process PREPARE_METADATA {

    tag "Metadata preparation"

    publishDir(
        "${params.results_dir}/02_prepare_metadata",
        mode: "copy"
    )

    input:

    path expression
    path clinical
    path clinical_new

    output:

    path "counts.tsv", emit: counts
    path "meta_analysis.tsv", emit: meta_analysis

    script:

    """
    /opt/anaconda3/envs/crc_scanpy/bin/python \
        ${projectDir}/bin/02_prepare_metadata.py \
        --expression ${expression} \
        --clinical ${clinical} \
        --clinical_new ${clinical_new} \
        --output_dir .
    """
}


// ============================================================
// PROCESS 3 — VST + PCA
// ============================================================

process PREPARE_VST_PCA {

    tag "VST and PCA"

    publishDir(
        "${params.results_dir}/03_vst_pca",
        mode: "copy"
    )

    input:

    path expression
    path clinical

    output:

    path "expr_all_vst.tsv", emit: expr_all_vst
    path "figures", emit: figures

    script:

    """
    mkdir -p figures

    /opt/anaconda3/envs/crc_scanpy/bin/python \
        ${projectDir}/bin/03_vst_pca.py \
        --expression ${expression} \
        --clinical ${clinical} \
        --output_dir . \
        --figures figures
    """
}


// ============================================================
// PROCESS 4 — DESeq2
// ============================================================

process DE_ANALYSIS {

    tag "DESeq2 - ${comparison}"

    publishDir(
        "${params.results_dir}/04_de/${comparison}",
        mode: "copy"
    )

    input:

    path counts
    path metadata
    each comparison

    output:

    tuple val(comparison), path("DESeq2_*.tsv"), emit: de_results

    script:

    """
    /opt/anaconda3/envs/crc_scanpy/bin/python \
        ${projectDir}/bin/04_de.py \
        --counts ${counts} \
        --metadata ${metadata} \
        --comparison ${comparison} \
        --output_dir .
    """
}


// ============================================================
// PROCESS 5 — VOLCANO PLOTS
// ============================================================

process VOLCANO_PLOTS {

    tag "Volcano - ${comparison}"

    publishDir(
        "${params.results_dir}/05_volcano_plot/${comparison}",
        mode: "copy"
    )

    input:

    tuple val(comparison), path(de_files)

    output:

    path "volcano_plots"

    script:

    """
    mkdir -p de_results
    mkdir -p volcano_plots

    cp ${de_files} de_results/

    /opt/anaconda3/envs/crc_scanpy/bin/python \
        ${projectDir}/bin/05_volcano_plot.py \
        --comparison ${comparison} \
        --de_dir de_results \
        --figures volcano_plots
    """
}


// ============================================================
// PROCESS 6 — RANKED GSEA
// ============================================================

process RANKED_GSEA {

    tag "Ranked GSEA - ${comparison}"

    publishDir(
        "${params.results_dir}/06_ranked_gsea/${comparison}",
        mode: "copy"
    )

    input:

    tuple val(comparison), path(de_files)
    path panel_files

    output:

    tuple val(comparison), path("ranked_gsea"), emit: ranked_gsea_results

    script:

    """
    mkdir -p de_results
    mkdir -p panel_dir
    mkdir -p ranked_gsea

    cp ${de_files} de_results/
    cp ${panel_files} panel_dir/

    /opt/anaconda3/envs/crc_scanpy/bin/python \
        ${projectDir}/bin/06_ranked_gsea.py \
        --comparison ${comparison} \
        --de_dir de_results \
        --panel_dir panel_dir \
        --output_dir ranked_gsea
    """
}

// ============================================================
// PROCESS 7 — ssGSEA
// ============================================================

process SSGSEA {

    tag "Run ssGSEA"

    publishDir(
        "${params.results_dir}/07_ssgsea",
        mode: "copy"
    )

    input:

    path expression
    path panel_files

    output:

    path "ssgsea/scores_all.csv", emit: scores_all

    script:

    """
    mkdir -p panel_dir
    mkdir -p ssgsea
    cp ${panel_files} panel_dir/

    /opt/anaconda3/envs/crc_scanpy/bin/python \
        ${projectDir}/bin/07_ssgsea.py \
        --expression ${expression} \
        --panel_dir panel_dir \
        --output_dir ssgsea
    """
}
// ============================================================
// PROCESS 8 — statistics ssGSEA
// ============================================================
process STAT_SSGSEA {

    tag "test groups by ssGSEA scores - ${comparison}"

    publishDir(
        "${params.results_dir}/08_ssgsea_statistics/${comparison}",
        mode: "copy"
    )

    input:

    path metadata
    path scores_all
    each comparison

    output:

    tuple val(comparison), path("ssgsea_statistics"), emit: ssgsea_statistics

    script:

    """
    mkdir -p ssgsea_statistics

    /opt/anaconda3/envs/crc_scanpy/bin/python \
        ${projectDir}/bin/08_ssgsea_statistics.py \
        --metadata ${metadata} \
        --scores_all ${scores_all} \
        --comparison ${comparison} \
        --output_dir ssgsea_statistics
    """
}

// ============================================================
// PROCESS 9 — VISUALIZATION of GSEA RESULTS
// ============================================================

process PLOTS_GSEA {

    tag "GSEA visualization - ${comparison}"

    publishDir(
        "${params.results_dir}/09_visualization_GSEA/${comparison}",
        mode: "copy"
    )

    input:

    path metadata
    path scores_all
    tuple val(comparison), path(stat_results), path(ranked_gsea_dir), path(de_results)

    output:

    path "gsea_plots"

    script:

    """
    mkdir -p gsea_plots
    mkdir -p stat_results
    mkdir -p de_results
    

    cp -r ${stat_results}/* stat_results/
    cp ${de_results} de_results/


    /opt/anaconda3/envs/crc_scanpy/bin/python \
        ${projectDir}/bin/09_visualization_GSEA.py \
        --comparison ${comparison} \
        --metadata ${metadata} \
        --scores_all ${scores_all} \
        --stat_ssgsea_dir stat_results \
        --ranked_gsea_dir ranked_gsea \
        --output_dir gsea_plots
    """
}

// ============================================================
// PROCESS 10 — Longitudinal ssGSEA analysis
// ============================================================
process DYNAMICS_GSEA {

    tag "Interaction analysis and delta analysis - ${comparison}"

    publishDir(
        "${params.results_dir}/10_dynamics/${comparison}",
        mode: "copy"
    )

    input:

    path metadata
    path scores_all
    each comparison

    output:

    tuple val(comparison), path("dynamics_results"), emit: dynamics_results

    script:

    """
    mkdir -p dynamics_results

    /opt/anaconda3/envs/crc_scanpy/bin/python \
        ${projectDir}/bin/10_dynamics.py \
        --metadata ${metadata} \
        --scores_all ${scores_all} \
        --comparison ${comparison} \
        --output_dir dynamics_results
    """
}


// ============================================================
// WORKFLOW
// ============================================================

workflow {

    // --------------------------------------------------------
    // Expression input
    // --------------------------------------------------------

    sample_files = Channel
        .fromPath(
            "${params.input_dir}/*.tsv",
            checkIfExists: true
        )
        .collect()


    // --------------------------------------------------------
    // Annotation
    // --------------------------------------------------------

    annotation_files = Channel
        .fromPath(
            "${params.annotation_dir}/*.gz",
            checkIfExists: true
        )
        .collect()


    // --------------------------------------------------------
    // Clinical data
    // --------------------------------------------------------

    clinical = Channel
        .fromPath(
            "${projectDir}/data/raw/clinical_data.csv",
            checkIfExists: true
        )

    clinical_new = Channel
        .fromPath(
            "${projectDir}/data/raw/clinical_data_new.csv",
            checkIfExists: true
        )


    // --------------------------------------------------------
    // Gene panels
    // --------------------------------------------------------

    panel_files = Channel
        .fromPath(
            "${params.panel_dir}/gene_panel_*.csv",
            checkIfExists: true
        )
        .collect()


    // --------------------------------------------------------
    // Comparisons 4 and 8 process
    // --------------------------------------------------------

    comparisons = [
        "group_analysis_1",
        "response",
        "aggressiveness",
        "bone_marrow",
        "lesion"
    ]
    
    


    // ========================================================
    // PROCESS 1
    // ========================================================

    PREPARE_EXPRESSION(
        sample_files,
        annotation_files
    )


    // ========================================================
    // PROCESS 2
    // ========================================================

    PREPARE_METADATA(
        PREPARE_EXPRESSION.out.expression_raw_ann,
        clinical,
        clinical_new
    )


    // ========================================================
    // PROCESS 3
    // ========================================================

    PREPARE_VST_PCA(
        PREPARE_METADATA.out.counts,
        PREPARE_METADATA.out.meta_analysis
    )


    // ========================================================
    // PROCESS 4
    // ========================================================

    DE_ANALYSIS(
        PREPARE_METADATA.out.counts,
        PREPARE_METADATA.out.meta_analysis,
        comparisons
    )


    // ========================================================
    // PROCESS 5
    // ========================================================

    VOLCANO_PLOTS(
        DE_ANALYSIS.out.de_results
    )


    // ========================================================
    // PROCESS 6
    // ========================================================

    RANKED_GSEA(
        DE_ANALYSIS.out.de_results,
        panel_files
    )

    // ========================================================
    // PROCESS 7
    // ========================================================

    SSGSEA(
        PREPARE_VST_PCA.out.expr_all_vst,
        panel_files
    )

    // ========================================================
    // PROCESS 8
    // ========================================================

    STAT_SSGSEA(
        PREPARE_METADATA.out.meta_analysis,
        SSGSEA.out.scores_all,
        comparisons
    )

    // join data by comparison group
    gsea_results = STAT_SSGSEA.out.ssgsea_statistics
        .join(RANKED_GSEA.out.ranked_gsea_results)
        .join(DE_ANALYSIS.out.de_results)
        .map { comparison, stat_results, ranked_gsea_dir, de_results ->
            tuple(
                comparison,
                stat_results,
                ranked_gsea_dir,
                de_results
            )
        }




    // ========================================================
    // CREATE VALUE CHANNELS
    // ========================================================

    metadata_value = PREPARE_METADATA.out.meta_analysis.first()
    scores_value = SSGSEA.out.scores_all.first()

    // ========================================================
    // PROCESS 9
    // ========================================================

    PLOTS_GSEA(
        metadata_value,
        scores_value,
        gsea_results
    )

    // ========================================================
    // PROCESS 10
    // ========================================================

    DYNAMICS_GSEA(
        PREPARE_METADATA.out.meta_analysis,
        SSGSEA.out.scores_all,
        comparisons
    )
}