from typing import List, Literal, Optional, Dict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scanpy as sc
import anndata as ad
from simba_plus.plotting.utils import enrichment
from textwrap import wrap


def factor_correlation(adata_C: ad.AnnData, return_fig: bool = False, figsize=(10, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    correlation_mat = np.corrcoef(adata_C.X.T)
    sns.heatmap(
        correlation_mat,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        ax=ax,
    )
    ax.set_aspect("equal")
    ax.set_title("Factor Correlation Matrix")
    if return_fig:
        return fig


def factor_enrichment(
    adata_G: ad.AnnData,
    factor: str,
    adj_p_thres=0.1,
    n_max_terms=10,
    gene_sets: List[str] = [
        "GO_Biological_Process_2021",
        "KEGG_2021_Human",
        "MSigDB_Hallmark_2020",
    ],
    figsize=(10, 15),
    return_fig: bool = False,
):
    """
    Plot enrichment results for a given factor and gene set.

    Args:
        adata_G: AnnData object containing gene data with enrichment results in .uns.
        factor: The factor for which to plot enrichment results.
        gene_set: The gene set to plot (e.g., "GO_Biological_Process_2021"). If None, plots for all gene sets.
        n_max_terms: Maximum number of enriched terms to display in the plot.
    """
    fig_top = enrichment(
        adata_G,
        "top_enrichments",
        factor,
        adj_p_thres,
        n_max_terms,
        gene_sets,
        "Top ",
        figsize,
    )
    fig_bot = enrichment(
        adata_G,
        "bot_enrichments",
        factor,
        adj_p_thres,
        n_max_terms,
        gene_sets,
        "Bottom ",
        figsize,
    )
    if return_fig:
        return fig_top, fig_bot


def factor_umap(
    adata_C: ad.AnnData,
    return_fig: bool = False,
    rasterize: bool = True,
    factor_labels: Dict[str, str] = None,
    cell_type_label: Optional[str] = None,
    **kwargs,
):
    adata_C = adata_C.copy()
    if "X_normed" not in adata_C.layers:
        adata_C.layers["X_normed"] = (
            adata_C.X / np.linalg.norm(adata_C.X.astype(np.float64), axis=1)[:, None]
        )
    for i in range(adata_C.X.shape[1]):
        adata_C.obs[f"Factor {i}"] = adata_C.layers["X_normed"][:, i]
    if rasterize:
        sc.set_figure_params(vector_friendly=True)

    def format_summary(summary_str):
        top, bot = summary_str.split(" <> ")
        return (
            "\n".join(wrap(top.split(":", 1)[-1], 30))
            + "<> \n"
            + "\n".join(wrap(bot.split(":", 1)[-1], 30))
        )

    colors = [f"Factor {i}" for i in range(adata_C.X.shape[1])]
    if factor_labels is not None:
        titles = [
            f"Factor {i}\n" + format_summary(factor_labels[f"Factor {i}"])
            for i in range(adata_C.X.shape[1])
        ]
    else:
        titles = None

    fig = sc.pl.umap(
        adata_C,
        color=colors,
        title=titles,
        legend_loc="right margin",
        cmap="coolwarm",
        vcenter=0,
        hspace=1,
        return_fig=return_fig,
        **kwargs,
    )
    fig.subplots_adjust(top=0.9)
    if return_fig:
        return fig
