from typing import Optional, List, Literal, Dict
import os
import numpy as np
import anndata as ad
import pandas as pd
from tqdm.auto import tqdm
from argparse import ArgumentParser
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from simba_plus.utils import setup_logging
from simba_plus.post_training.enrichment import run_enrichr
import simba_plus.plotting.factors
from matplotlib.backends.backend_pdf import PdfPages


def add_argument(parser: ArgumentParser) -> ArgumentParser:
    parser.description = "Analyze learned latent factors"
    parser.add_argument(
        "adata_prefix",
        type=str,
        help="Directory that contains adata_{C,P,G}{version_suffix}.h5ad files from simba+ train output.",
    )
    parser.add_argument(
        "--version-suffix",
        type=str,
        default="",
        help="Suffix to append to adata_{C,P,G} files from simba+ train output. ({adata_prefix}/adata_{C,P,G}{version_suffix}.h5ad will be loaded.)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for LDSC run output and heritability results.",
    )
    parser.add_argument(
        "--cell-type-label",
        type=str,
        default="cell_type",
        help="When provided, calculate baseline per-cell-type heritability.",
    )
    return parser


def factor_enrichments(
    adata_G: ad.AnnData,
    gene_sets: List[
        Literal["GO_Biological_Process_2021", "KEGG_2021_Human", "MSigDB_Hallmark_2020"]
    ] = ["GO_Biological_Process_2021", "KEGG_2021_Human", "MSigDB_Hallmark_2020"],
):
    if "X_normed" not in adata_G.layers:
        adata_G.layers["X_normed"] = (
            adata_G.X / np.linalg.norm(adata_G.X.astype(np.float64), axis=1)[:, None]
        )
    adata_G.var_names = [f"Factor {i}" for i in range(adata_G.shape[1])]
    gene_loadings = pd.DataFrame(
        adata_G.layers["X_normed"],
        index=adata_G.obs_names,
        columns=adata_G.var_names,
    )
    top_enrichments: Dict[str, Dict[str, pd.DataFrame]] = {}
    bot_enrichments: Dict[str, Dict[str, pd.DataFrame]] = {}
    n_top_genes = {}
    n_bot_genes = {}
    if "top_enrichments" not in adata_G.uns:
        adata_G.uns["factor_enrichments_summary"] = {}
        for i, factor in enumerate(gene_loadings.columns):
            if factor not in top_enrichments:
                top_enrichments[factor] = {}
                bot_enrichments[factor] = {}
            for gene_set in gene_sets:
                top_enrichments[factor][gene_set], n_top_genes[factor] = run_enrichr(
                    gene_loadings[factor],
                    gene_set=gene_set,
                )
                bot_enrichments[factor][gene_set], n_bot_genes[factor] = run_enrichr(
                    gene_loadings[factor],
                    top=False,
                    gene_set=gene_set,
                )
            adata_G.uns["factor_enrichments_summary"][factor] = summarize_enrichments(
                top_enrichments[factor], bot_enrichments[factor]
            )
        adata_G.uns["top_enrichments"] = top_enrichments
        adata_G.uns["bot_enrichments"] = bot_enrichments
        adata_G.uns["top_enrichments_n_genes"] = n_top_genes
        adata_G.uns["bot_enrichments_n_genes"] = n_bot_genes

    return adata_G


def summarize_enrichments(gene_enrichment, gene_bot_enrichment):
    # Print peak annotation side by side
    top_most_gene_enrichment = None
    top_gene_set = ""
    bot_most_gene_enrichment = None
    bot_gene_set = ""
    for gene_set in gene_enrichment.keys():
        if len(gene_enrichment[gene_set]) > 0:
            this_enrichment = (
                gene_enrichment[gene_set]
                .sort_values("Adjusted P-value")[["Adjusted P-value", "Term"]]
                .iloc[0]
            )
            if top_most_gene_enrichment is None:
                top_most_gene_enrichment = this_enrichment
                top_gene_set = gene_set
            elif (
                top_most_gene_enrichment["Adjusted P-value"]
                > this_enrichment["Adjusted P-value"]
            ):
                top_most_gene_enrichment = this_enrichment
                top_gene_set = gene_set

        if len(gene_bot_enrichment[gene_set]) > 0:
            this_enrichment = (
                gene_bot_enrichment[gene_set]
                .sort_values("Adjusted P-value")[["Adjusted P-value", "Term"]]
                .iloc[0]
            )
            if bot_most_gene_enrichment is None:
                bot_most_gene_enrichment = this_enrichment
                bot_gene_set = gene_set
            elif (
                bot_most_gene_enrichment["Adjusted P-value"]
                > this_enrichment["Adjusted P-value"]
            ):
                bot_most_gene_enrichment = this_enrichment
                bot_gene_set = gene_set

    if top_most_gene_enrichment is None:
        top_enriched = "No enrichment"
    else:
        top_enriched = top_most_gene_enrichment["Term"]
    if bot_most_gene_enrichment is None:
        bot_enriched = "No enrichment"
    else:
        bot_enriched = bot_most_gene_enrichment["Term"]

    label = f"{top_gene_set}:{top_enriched} <> {bot_gene_set}:{bot_enriched}"
    return label


def main(args, logger=None):
    if not logger:
        logger = setup_logging(
            "simba+factor", log_dir=os.path.dirname(args.adata_prefix)
        )
    # Loading pretrained results as `simba+ heritability` takes long time to run
    if args.output_dir is None:
        args.output_dir = f"{os.path.dirname(args.adata_prefix)}/factors/"
        os.makedirs(args.output_dir, exist_ok=True)

    adata_C = sc.read_h5ad(f"{args.adata_prefix}/adata_C{args.version_suffix}.h5ad")
    adata_G = sc.read_h5ad(f"{args.adata_prefix}/adata_G{args.version_suffix}.h5ad")
    output_filename = f"{args.output_dir}/simba+factors.pdf"
    # Create a PdfPages object
    with PdfPages(output_filename) as pdf:
        # correlation mat
        fig = simba_plus.plotting.factors.factor_correlation(adata_C, return_fig=True)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Generating factor enrichment plots...")
        for d in tqdm(range(adata_C.shape[1])):
            factor_enrichments(adata_G)
            fig_top, fig_bot = simba_plus.plotting.factors.factor_enrichment(
                adata_G, factor=adata_G.var_names[d], return_fig=True
            )
            pdf.savefig(fig_top, bbox_inches="tight")
            plt.close(fig_top)
            pdf.savefig(fig_bot, bbox_inches="tight")
            plt.close(fig_bot)
        sc.set_figure_params(vector_friendly=True)
        fig = sc.pl.umap(adata_C, color=args.cell_type_label, return_fig=True)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        factor_labels = adata_G.uns["factor_enrichments_summary"]
        factor_umap = simba_plus.plotting.factors.factor_umap(
            adata_C,
            return_fig=True,
            factor_labels=factor_labels,
        )
        pdf.savefig(factor_umap, bbox_inches="tight")
        plt.close(factor_umap)
    logger.info(f"Generated factor report in {output_filename}.")
    adata_C.write(f"{args.adata_prefix}/adata_C{args.version_suffix}_annotated.h5ad")
    adata_G.write(f"{args.adata_prefix}/adata_G{args.version_suffix}_annotated.h5ad")
    logger.info(
        f"Generated annotated data in {args.adata_prefix}/adata_{{C,G,P}}_annotated.h5ad."
    )
