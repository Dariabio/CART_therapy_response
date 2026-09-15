import os
import argparse
import numpy as np
import pandas as pd
import gseapy as gp


parser = argparse.ArgumentParser()

parser.add_argument(
    "--comparison",
    required=True
)

parser.add_argument(
    "--de_dir",
    required=True
)

parser.add_argument(
    "--output_dir",
    required=True
)

args = parser.parse_args()


comparison = args.comparison

os.makedirs(
    args.output_dir,
    exist_ok=True
)

#Paths
GSEA_DIR = args.output_dir
DE_DIR = args.de_dir

# Run GSEA

# per each timepoint
for short_name in [
    "product",
    "d7",
    "d28"
]:

    input_file = os.path.join(
        DE_DIR,
        f"DESeq2_{short_name}.tsv"
    )


    df = pd.read_csv(
        input_file,
        sep="\t",
        index_col=0
    )

    df.index.name = "gene"

    df = df.reset_index()

    df = df[
        ["gene", "log2FoldChange"]
    ].dropna()

    df = df.drop_duplicates(
        subset="gene", keep='first'
    )

    # Use log2FoldChange as metric
    #We add micro-noise and sort (from the highest LFC among responders to the lowest)
    np.random.seed(42)
    
    df["rank_metric"] = (
        df["log2FoldChange"]
        +
        np.random.normal(
            0,
            1e-9,
            len(df)
        )
    )

    df = df.sort_values(
        "rank_metric",
        ascending=False)


    ranked = df[["gene", "rank_metric"]]


    outdir = os.path.join(
        GSEA_DIR,
        short_name
    )

    os.makedirs(
        outdir,
        exist_ok=True
    )


    # public db
    public_sets = ['GO_Biological_Process_2021']
        
    for db in public_sets:
            
        pr = gp.prerank(
        rnk = ranked,
        gene_sets=db,  # database
        outdir=outdir,
        permutation_num=1000,
        min_size=15,   # for big db higher
        max_size=500,
        seed=42, verbose=True)

        all_results = pr.res2d.copy()


        all_results.to_csv(
            os.path.join(
                outdir,
                f"all_results_{db}_{short_name}.tsv"
            ),
            sep="\t",
            index=False
        )
            
        # Filter significant (FDR < 0.05)
        significant_terms = pr.res2d[pr.res2d['FDR q-val'] < 0.05]
        significant_terms.to_csv(
        os.path.join(outdir, f"significant_results_{db}_{short_name}.tsv"), 
        sep="\t", index=False)


