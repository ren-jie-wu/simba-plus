"""Get residual summary statistics after regressing out baseline covariates."""

import os
import pandas as pd
import subprocess
import pybedtools
from scipy.sparse import lil_matrix
import simba_plus.datasets._datasets

def run_ldsc(sumstat_paths, out_path, rerun=False, nproc=10):
    simba_plus.datasets._datasets.heritability()
    processes = []
    os.makedirs(out_path, exist_ok=True)
    for sumstat_path in list(sumstat_paths):
        command = f"bash {os.path.dirname(__file__)}/ldsc_h2_baseline.sh {sumstat_path} {out_path} {rerun}"
        processes.append(subprocess.Popen(command, shell=True))
        if len(processes) >= nproc:
            for process in processes:
                process.wait()
            processes = []
    for process in processes:
        process.wait()


def get_residual(sumstat_list_path, output_path, rerun=False, nproc=10):
    sumstat_paths = list(
        pd.read_csv(sumstat_list_path, sep="\t", index_col=0, header=None).values[:, 0]
    )
    run_ldsc(sumstat_paths, output_path, rerun=rerun, nproc=nproc)

    residual_paths = [
        os.path.join(
            output_path,
            os.path.basename(p).split(".gz")[0].split(".sumstats")[0] + ".residuals",
        )
        for p in sumstat_paths
    ]
    residuals = pd.concat(
        [
            pd.read_csv(sumstat_paths[0], sep="\t", compression="infer")[
                ["SNP", "CHR", "BP"]
            ]
        ]
        + [pd.read_csv(p, sep="\t")["residuals"] for p in residual_paths],
        axis=1,
        ignore_index=True,
    )
    residuals.columns = ["SNP", "CHR", "BP"] + [
        os.path.basename(p).replace(".sumstats", "") for p in sumstat_paths
    ]
    return residuals


def get_overlap(snp_df, peak_df):
    # For each peak, get overlapping SNPs
    overlap_dict = {}
    # Convert SNP and peak dataframes to BedTool objects
    snp_df = snp_df.rename(columns={"CHR": "chrom", "BP": "start"})
    snp_df.insert(2, "end", snp_df["start"] + 1)
    snp_bed = pybedtools.BedTool.from_dataframe(
        snp_df
    )
    peak_bed = pybedtools.BedTool.from_dataframe(
        peak_df.rename(columns={"chr": "chrom", "start": "start", "end": "end"})
    )

    # Perform intersection
    intersection = snp_bed.intersect(peak_bed, wa=True, wb=True)

    # Parse the intersection results
    # Initialize a sparse matrix with dimensions (number of peaks, number of SNPs)
    num_snps = len(snp_df)
    num_peaks = len(peak_df)
    overlap_matrix = lil_matrix((num_peaks, num_snps), dtype=int)

    for line in intersection:
        snp_index = int(line[3])  # Assuming SNP index is in the 4th column
        peak_index = int(line[6])  # Assuming peak index is in the 7th column
        overlap_matrix[peak_index, snp_index] = 1
    return overlap_matrix
