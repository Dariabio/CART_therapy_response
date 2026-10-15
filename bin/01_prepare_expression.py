import os
import argparse
import pandas as pd
import glob

#Arguments
parser = argparse.ArgumentParser(
    description="Merge and annotate RNA-seq expression matrices"
)

parser.add_argument(
    "--input_dir",
    required=True,
    help="Directory containing sample-level expression TSV files"
)

parser.add_argument(
    "--annotation_dir",
    required=True,
    help="Directory containing gene2ensembl.gz and Homo_sapiens.gene_info.gz"
)

parser.add_argument(
    "--output_dir",
    required=True,
    help="Output directory"
)

args = parser.parse_args()

input_dir = args.input_dir
ann_dir = args.annotation_dir
output_dir = args.output_dir

os.makedirs(output_dir, exist_ok=True)

# Input files

pattern = os.path.join(
    input_dir,
    "*.tsv"
)

files = sorted(
    glob.glob(pattern)
)

if not files:
    raise FileNotFoundError(
        f"No TSV files found in {input_dir}"
    )

print(f"Found {len(files)} sample files")

# list of dfs
dfs1 = []
dfs2 = []

for fpath in files:
    fname = os.path.basename(fpath)
    sample_name = fname.replace(".tsv", "")

    df = pd.read_csv(fpath, sep="\t")

    df1 = df[["target_id", "tpm"]]
    df2 = df[["target_id", "est_counts"]]

    df1 = df1.rename(columns={"tpm": sample_name})
    df2 = df2.rename(columns={"est_counts": sample_name})

    dfs1.append(df1)
    dfs2.append(df2)

merged = dfs1[0]
for df in dfs1[1:]:
    merged = pd.merge(merged, df, on="target_id", how="outer")

merged.to_csv(os.path.join(output_dir, "expression_tpm.tsv"), sep="\t")

merged = dfs2[0]
for df in dfs2[1:]:
    merged = pd.merge(merged, df, on="target_id", how="outer")

merged.to_csv(os.path.join(output_dir, "expression_raw.tsv"), sep="\t")

# 2. Preprocess data
# Annotation and filtering
expression = pd.read_csv(os.path.join(output_dir, "expression_raw.tsv"), sep="\t")
expression_tpm = pd.read_csv(os.path.join(output_dir, "expression_tpm.tsv"), sep="\t")

def annotate_enst_ids_ncbi(expression, ann_dir, expr_type =None, id_col=None):
    df = expression.copy()

    
    mapping_file = os.path.join(ann_dir, "gene2ensembl.gz")
    gene_info_file = os.path.join(ann_dir, "Homo_sapiens.gene_info.gz")

    ensembl_df = pd.read_csv(
        mapping_file,
        compression="gzip",
        sep="\t",
        usecols=["#tax_id", "GeneID", "Ensembl_rna_identifier"],
    )

    # filter human (9606)
    ensembl_df = ensembl_df[ensembl_df["#tax_id"] == 9606]

    # filter zero rows
    ensembl_df = ensembl_df[
        (ensembl_df["Ensembl_rna_identifier"].notna())
        & (ensembl_df["Ensembl_rna_identifier"] != "-")
    ]

    # NCBI GeneID -> Official Symbol

    ncbi_df = pd.read_csv(
        gene_info_file, compression="gzip", sep="\t", usecols=["GeneID", "Symbol"]
    )

    #Merge tables
    mapping_df = pd.merge(ensembl_df, ncbi_df, on="GeneID", how="inner")

    # Re
    mapping_df["Ensembl_rna_clean"] = (
        mapping_df["Ensembl_rna_identifier"]
        .astype(str)
        .apply(lambda x: x.split(".")[0].strip())
    )

    # Mapping dict
    enst_to_symbol = dict(
        zip(mapping_df["Ensembl_rna_clean"], mapping_df["Symbol"])
    )

    # Orig IDs
    if id_col in df.columns:
        raw_ids = df[id_col].astype(str)
    else:
        raw_ids = df.index.to_series().astype(str)

    # remove ID versions
    query_ids = raw_ids.apply(lambda x: x.split(".")[0].strip())

    # Mapping
    df["symbol"] = query_ids.map(enst_to_symbol)

    if id_col in df.columns:
        df = df.drop(id_col, axis=1)

    # Remove duplicates

    if expr_type == "raw":
    
            df = df.groupby("symbol").sum().reset_index()
            numeric_cols = df.select_dtypes(include=['number']).columns
            df[numeric_cols] = df[numeric_cols].round().astype(int)

    else:
            df = df.groupby("symbol").mean(numeric_only=True).reset_index()
    

    # Remove unmapped
    df = df.dropna(subset=["symbol"])
    df = df.set_index("symbol")
    df = df.drop("Unnamed: 0", axis=1)

    return df

# Saving annotated raw and tpm files
expression_tpm = annotate_enst_ids_ncbi(
    expression_tpm, ann_dir=ann_dir, id_col='target_id', expr_type="tpm")

expression = annotate_enst_ids_ncbi(
    expression, ann_dir=ann_dir, id_col='target_id', expr_type="raw")

expression_tpm.to_csv(os.path.join(output_dir, "expression_tpm_ann.tsv"), sep="\t")

expression.to_csv(os.path.join(output_dir, "expression_raw_ann.tsv"), sep="\t")

print("Expression preparation completed")

