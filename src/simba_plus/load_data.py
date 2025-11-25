# https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html
from typing import Dict, Iterable
import numpy as np
import torch
from torch_geometric.data import HeteroData
import anndata as ad
from simba_plus.utils import _assign_node_id, _make_tensor
import argparse


def validate_input(adata_CG, adata_CP):
    """
    Validate and extract dimensions from input AnnData objects.

    This function compares cell indices between adata_CG (cell x gene matrix)
    and adata_CP (cell x peak matrix), and makes sure that, if both are provided,
    the .obs indices are aligned. If not, adata_CP will be subset/reordered
    to match adata_CG.obs.index.

    It returns the number of cells, genes, peaks, and motifs (-1 if not determined).

    Args:
        adata_CG (anndata.AnnData or None): AnnData for cell by gene data.
        adata_CP (anndata.AnnData or None): AnnData for cell by peak data.

    Returns:
        tuple:
            n_cells (int): Number of cells determined from provided AnnDatas.
            n_genes (int): Number of genes (-1 if adata_CG not given).
            n_peaks (int): Number of peaks (-1 if adata_CP not given).
            n_motifs (int): Placeholder, always -1.
    """
    n_cells = n_genes = n_peaks = n_motifs = -1
    if adata_CG is not None:
        n_cells, n_genes = adata_CG.shape
        if adata_CP is not None:
            _, n_peaks = adata_CP.shape
    elif adata_CP is not None:
        n_cells, n_peaks = adata_CP.shape
    if adata_CG is not None and adata_CP is not None:
        if not (adata_CG.obs.index == adata_CP.obs.index).all():
            adata_CP = adata_CP[adata_CG.obs.index, :]
        assert (adata_CG.obs.index == adata_CG.obs.index).all()
    return n_cells, n_genes, n_peaks, n_motifs


def type_attribute(data):
    """
    Convert node attributes to PyTorch tensors.

    This function converts all node attributes in a HeteroData object to PyTorch tensors.
    It ensures that the attributes are in the correct format for use in a PyTorch model.

    Args:
        data (torch_geometric.data.HeteroData): The HeteroData object containing the node attributes.

    Returns:
        torch_geometric.data.HeteroData: The HeteroData object with node attributes converted to PyTorch tensors.
    """
    for node_type in data.node_types:
        data[node_type].x = torch.tensor(data[node_type].x, dtype=torch.float)
    return data


def make_sc_HetData(
    adata_CG: ad.AnnData = None,
    adata_CP: ad.AnnData = None,
    cell_cont_covariate_to_include: Dict[str, Iterable[str]] = None,
    cell_cat_cov: str = None,
):
    """
    Create a sc-HetData object from AnnData objects.

    This function creates a sc-HetData object from two AnnData objects, one for cell by gene data and one for cell by peak data.
    All node embeddings are initialized to zero. Edge attributes are set to the raw expression values (adata_CG) and binary accessibility values (adata_CP).
    It also includes optional covariates for the cell attributes.

    Args:
        adata_CG (anndata.AnnData or None): AnnData for cell by gene data.
        adata_CP (anndata.AnnData or None): AnnData for cell by peak data.
        cell_cont_covariate_to_include (Dict[str, Iterable[str]] or None): Dictionary of continuous covariates to include for the cell attributes.
        cell_cat_cov (str or None): Name of the categorical covariate to use for the cell attributes.

    Returns:
        torch_geometric.data.HeteroData: The sc-HetData object created from the AnnData objects.
    """
    if not adata_CG and not adata_CP:
        raise ValueError("No data provided for edge construction")
    n_cells, n_genes, n_peaks, n_motifs = validate_input(adata_CG, adata_CP)
    data = HeteroData()
    n_dims = 50
    if n_cells > 0:
        if adata_CG is not None:
            data["cell"].x = torch.zeros((adata_CG.n_obs, n_dims))
            data["cell"].size_factor = adata_CG.X.toarray().sum(axis=1) / np.median(
                adata_CG.X.toarray().sum(axis=1)
            )
            if cell_cat_cov is not None:
                data["cell"].batch = torch.tensor(
                    adata_CG.obs[cell_cat_cov].astype("category").cat.codes.values,
                    dtype=torch.long,
                )
            data["gene"].size_factor = adata_CG.X.toarray().max(axis=0) / 2
            if cell_cont_covariate_to_include:
                data["cell"].cont_cov = adata_CG.obs[
                    cell_cont_covariate_to_include["CG"]
                ].values
            else:
                data["cell"].cont_cov = None
        elif adata_CP is not None:
            data["cell"].x = torch.zeros((adata_CP.n_obs, n_dims))
            # data["cell"].size_factor = adata_CP.X.toarray().sum(axis=1) / np.median(
            #     adata_CP.X.toarray().sum(axis=1)
            # )
            if cell_cat_cov is not None:
                data["cell"].batch = torch.tensor(
                    adata_CP.obs[cell_cat_cov].astype("category").cat.codes.values,
                    dtype=torch.long,
                )
            # data["peak"].size_factor = adata_CP.X.toarray().max(axis=0) / 2
            if cell_cont_covariate_to_include:
                data["cell"].cont_cov = adata_CP.obs[
                    cell_cont_covariate_to_include["CP"]
                ].values
            else:
                data["cell"].cont_cov = None

    if n_genes > 0:
        data["gene"].x = torch.zeros((adata_CG.n_vars, n_dims))
    if n_peaks > 0:
        data["peak"].x = torch.zeros((adata_CP.n_vars, n_dims))
    if adata_CG is not None:
        data["cell", "expresses", "gene"].edge_index = torch.from_numpy(
            np.stack(adata_CG.X.nonzero(), axis=0)
        ).long()  # [2, num_edges_expresses]
        data["cell", "expresses", "gene"].edge_attr = torch.from_numpy(
            adata_CG.X[adata_CG.X.nonzero()]
        ).squeeze()  # [num_edges_expresses, num_features_expresses]
        data["cell", "expresses", "gene"].edge_dist = (
            "NegativeBinomial"
            if np.issubdtype(adata_CG.X.dtype, np.integer)
            else "Normal"
        )

    if adata_CP is not None:
        data["cell", "has_accessible", "peak"].edge_index = torch.from_numpy(
            np.stack(adata_CP.X.nonzero(), axis=0)
        ).long()  # [2, num_edges_has_accessible]
        data["cell", "has_accessible", "peak"].edge_attr = torch.ones(
            data["cell", "has_accessible", "peak"].num_edges
        )
        data["cell", "has_accessible", "peak"].edge_dist = (
            "Bernoulli"  # Move to NB later on
        )

    for node_type in data.node_types:
        data[node_type].x = torch.tensor(data[node_type].x, dtype=torch.float)
    return data


def load_from_path(path: str, device="cpu") -> HeteroData:
    """
    Load a HeteroData object from a given path and move it to the specified device.

    This function loads a HeteroData object from a given path and moves it to the specified device.
    It also assigns node IDs and converts the attributes to PyTorch tensors.

    Args:
        path (str): Path to the HeteroData object file.
        device (str): Device to move the HeteroData object to.

    Returns:
        torch_geometric.data.HeteroData: The HeteroData object loaded from the given path and moved to the specified device.
    """
    data = torch.load(path, map_location=device, weights_only=False)
    _assign_node_id(data)
    _make_tensor(data, device=device)
    return data


def add_argument(parser):
    parser.description = "Load a HeteroData object from a given path and move it to the specified device."
    parser.add_argument(
        "--gene-adata",
        type=str,
        help="Path to the cell by gene AnnData file (e.g., .h5ad).",
    )
    parser.add_argument(
        "--peak-adata",
        type=str,
        help="Path to the cell by gene AnnData file (e.g., .h5ad).",
    )
    parser.add_argument(
        "--batch-col",
        type=str,
        help="Batch column in AnnData.obs of gene AnnData. If gene AnnData is not provided, peak AnnData will be used.",
    )
    parser.add_argument(
        "out_path",
        type=str,
        help="Path to the saved HeteroData object (e.g., .pt file).",
    )
    return parser


def main(args):
    """
    Create a sc-HetData object from AnnData objects and save it to a given path.

    This function creates a sc-HetData object from two AnnData objects, one for cell by gene data and one for cell by peak data.
    It also includes optional covariates for the cell attributes.

    Args:
        args (argparse.Namespace): Command line arguments.
    """
    dat = make_sc_HetData(
        adata_CG=ad.read_h5ad(args.gene_adata) if args.gene_adata else None,
        adata_CP=ad.read_h5ad(args.peak_adata) if args.peak_adata else None,
        cell_cat_cov=args.batch_col,
    )
    torch.save(dat, args.out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_argument(parser)
    args = parser.parse_args()
    main(args)
