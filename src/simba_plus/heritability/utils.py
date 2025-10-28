import os
import pandas as pd
from simba_plus.utils import write_bed


def get_overlap(snp_df, output_dir, data_path, adata_CP):
    snp_to_peak_path = f"{output_dir}/snp_to_peak.txt"
    create_snp_to_peak = not os.path.exists(snp_to_peak_path)
    if create_snp_to_peak:
        write_bed(
            adata_CP,
            filename=f"{output_dir}/peaks.bed",
        )
        filedir = "/data/pinello/PROJECTS/2023_09_JF_SIMBAvariant/pre_ldsc_analysis/sldsc_analysis/ldsc_files"
        os.system(
            f"bedtools intersect -a {filedir}/ref_bed_files/ref.bed -b {output_dir}/peaks.bed -wb -loj > {output_dir}/snp_to_peak.txt"
        )
    snp_to_peak = pd.read_csv(snp_to_peak_path, sep="\t", header=None)
    snp_to_peak.columns = [
        "chrom",
        "start",
        "end",
        "snp_id",
        "peak_chrom",
        "peak_start",
        "peak_end",
    ]
    snp_df["pid"] = (
        snp_to_peak["peak_chrom"]
        + "_"
        + snp_to_peak["peak_start"].astype(str)
        + "_"
        + snp_to_peak["peak_end"].astype(str)
    )
    adata_CP.var["pid"] = (
        adata_CP.var["chr"]
        + "_"
        + adata_CP.var["start"].astype(str)
        + "_"
        + adata_CP.var["end"].astype(str)
    )

    snp_df_peaksubset = adata_CP.var[["pid"]].merge(snp_df, on="pid", how="left")
    return snp_df_peaksubset.values
