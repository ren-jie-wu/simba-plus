def build_candidate_peaks(adata_CP, snp_hits, region_hits):
    """
    Create the final unified set of peaks:
        - peaks from SNP overlap
        - peaks from region overlap
        - if none provided → all peaks
    """
    if not snp_hits and not region_hits:
        print("No SNPs or regions provided → using all peaks.")
        return list(adata_CP.var_names)

    peaks = set(snp_hits) | set(region_hits)
    return list(peaks)

import pandas as pd
import pyranges as pr

def find_region_peak_overlaps(regions_df, adata_CP):
    """
    regions_df must have columns:
        regionid, chr, start, end

    Returns:
        peak_hits: list of overlapping peaks
        region_meta: df with Peak + region annotations
    """
    if regions_df is None:
        return [], None

    required = {"regionid", "chr", "start", "end"}
    if not required.issubset(regions_df.columns):
        raise ValueError(f"regions must contain columns: {required}")

    print(f"Processing {len(regions_df)} genomic regions → peak overlaps (PyRanges)...")

    # -----------------------------
    # Convert regions → PyRanges
    # -----------------------------
    region_pr = pr.PyRanges(pd.DataFrame({
        "Chromosome": regions_df["chr"].astype(str),
        "Start": regions_df["start"].astype(int),
        "End": regions_df["end"].astype(int),
        "regionid": regions_df["regionid"],
        "region_chr": regions_df["chr"].astype(str),
        "region_start": regions_df["start"].astype(int),
        "region_end": regions_df["end"].astype(int),
        "region_coord": regions_df.apply(
            lambda r: f"{r['chr']}_{r['start']}_{r['end']}", axis=1
        ),
    }))

    # -----------------------------
    # Convert peaks → PyRanges
    # -----------------------------
    peak_rows = []
    for pk in adata_CP.var_names:
        if ":" in pk:
            chrom, coords = pk.split(":")
            start, end = coords.split("-")
        else:
            chrom, start, end = pk.split("_")

        peak_rows.append({
            "Chromosome": chrom,
            "Start": int(start),
            "End": int(end),
            "Peak": pk,
        })

    peak_pr = pr.PyRanges(pd.DataFrame(peak_rows))

    # -----------------------------
    # Overlap
    # -----------------------------
    ov = region_pr.join(peak_pr).df

    if ov.empty:
        print("No regions overlapped any peaks.")
        return [], None

    peak_hits = ov["Peak"].unique().tolist()

    # -----------------------------
    # Build region metadata table
    # -----------------------------
    region_meta = (
        ov[
            [
                "Peak",
                "regionid",
                "region_chr",
                "region_start",
                "region_end",
                "region_coord",
            ]
        ]
        .drop_duplicates()
        .groupby("Peak", as_index=False)
        .agg({
            "regionid": lambda x: ",".join(sorted(set(x))),
            "region_coord": lambda x: ",".join(sorted(set(x))),
        })
    )

    return peak_hits, region_meta

import pandas as pd
import pyranges as pr

def find_snp_peak_overlaps(snps, adata_CP):
    if snps is None:
        return [], None

    if not {"snpid", "chr", "pos"}.issubset(snps.columns):
        raise ValueError("snps must be a DataFrame with columns: snp, chr, pos")

    print(f"Overlapping {len(snps)} SNPs with peaks (PyRanges)...")

    peaks_df = pd.DataFrame({"Peak": adata_CP.var_names})

    snp_pr = pr.PyRanges(pd.DataFrame({
        "Chromosome": snps["chr"].astype(str),
        "Start": snps["pos"].astype(int),
        "End": snps["pos"].astype(int) + 1,
        "snpid": snps["snpid"],
    }))

    ov = overlap_variants_with_peaks(peaks_df, snp_pr)

    if ov.empty:
        print("No SNPs overlapped any peaks.")
        return [], None

    peak_hits = ov["Peak"].unique().tolist()

    snp_meta = (
        ov[["Peak", "CHROM", "POS", "snpid"]]
        .rename(columns={"CHROM": "snp_chr", "POS": "snp_pos"})
        .groupby("Peak", as_index=False)
        .agg({
            "snpid": lambda x: ",".join(sorted(set(x))),
            "snp_chr": lambda x: ",".join(sorted(set(x))),
            "snp_pos": lambda x: ",".join(map(str, sorted(set(x)))),
        })
        .rename(columns={"snpid": "snpid"})
    )

    return peak_hits, snp_meta

def overlap_variants_with_peaks(candidates: pd.DataFrame, variants_bed: pr.PyRanges) -> pd.DataFrame:
    # Parse candidate Peak strings → chrom/start/end
    peaks_df = (
        candidates["Peak"]
        .str.replace(":", "_", regex=False)
        .str.split("_", expand=True)
        .rename(columns={0: "CHROM", 1: "START", 2: "END"})
        .astype({"START": int, "END": int})
    )
    peaks_df["Peak"] = candidates["Peak"]

    peak_pr = pr.PyRanges(peaks_df.rename(columns={
        "CHROM": "Chromosome", "START": "Start", "END": "End"
    }))

    # Find overlaps (RSID survives as column snp)
    overlap_df = variants_bed.join(peak_pr).df.rename(columns={
        "Chromosome": "CHROM",
        "Start": "POS",
        "End": "POS2",
        "Start_b": "START_ANN",
        "End_b": "END_ANN",
    })

    # Standardize Peak ID (use annotation start/end from the joined peak)
    overlap_df["Peak"] = (
        overlap_df["CHROM"].astype(str) + "_" +
        overlap_df["START_ANN"].astype(str) + "_" +
        overlap_df["END_ANN"].astype(str)
    )

    print(f"Found {len(overlap_df):,} overlapping variant–peak records.")

    return overlap_df[["Peak", "CHROM", "POS", "snpid"]]
