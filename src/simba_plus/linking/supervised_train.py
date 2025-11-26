
import pandas as pd
from simba_plus.linking import model_training
import pyranges as pr
import os
import ast

def peak_gene_link_train(
    eval_df: pd.DataFrame,
    feature_sets: dict,
    dataset_name: str,
    output_path: str,
):
    """
    Train supervised XGBoost models for peak–gene link prediction using SIMBA+ features.

    This function:
      - Runs the SIMBA+ XGBoost training pipeline with custom feature sets.
      - Performs hyperparameter search + LOCO cross-validation.
      - Saves trained models, predictions, and metrics into `output_path`.

    Parameters
    ----------
    eval_df : pd.DataFrame
        Input dataframe containing SIMBA+ features and labels.
        Must contain at least:
            - "Peak"  (peak ID string)
            - "Gene"  (gene name string)
            - "label" (binary training label)
        Optional:
            - "peak_gene_pair" (if missing, it will be auto-constructed)

    feature_sets : dict
        Dictionary mapping:
            model_name → list_of_feature_column_names
        Example:
            {
                "simba_plus_distance": ["SIMBA+_path_score", "1/Distance"],
                "distance_only": ["1/Distance"]
            }

    dataset_name : str
        Prefix used to label saved artifacts (e.g., `"crispr"`, `"eqtl"`).

    output_path : str
        Directory where all trained model files, metrics, and predictions will be saved.

    Returns
    -------
    results_obj : dict
        Metadata about saved models and training details.

    metrics_df : pd.DataFrame
        Per-chromosome LOCO validation metrics for each feature set.

    preds_df : pd.DataFrame
        Combined predictions across all chromosomes and feature sets.

    Raises
    ------
    ValueError
        If required columns are missing from `eval_df`.
    """

    # ------------------------------------------------------------
    # 1. Ensure peak_gene_pair column exists
    # ------------------------------------------------------------
    if "peak_gene_pair" not in eval_df.columns:
        required = {"Peak", "Gene"}
        if not required.issubset(eval_df.columns):
            raise ValueError(
                "eval_df must contain either 'peak_gene_pair' or both 'Peak' and 'Gene' "
                "to construct the identifier."
            )

        eval_df = eval_df.copy()
        eval_df["peak_gene_pair"] = (
            eval_df["Peak"].astype(str) + "_" + eval_df["Gene"].astype(str)
        )

    # ------------------------------------------------------------
    # 2. Train supervised XGBoost models using SIMBA+ pipeline
    # ------------------------------------------------------------
    results_obj, metrics_df, preds_df = model_training.train_xgboost(
        df=eval_df,
        dataset_name=dataset_name,
        output_dir=output_path,
        feature_sets=feature_sets,
        n_search_iter=30,
        search_n_jobs=8,
        random_state=1,
    )

    return results_obj, metrics_df, preds_df


def build_feature_sets(feature_sets: dict, output_path: str, python_list_format: bool = False):    
    rows = []
    for name, features in feature_sets.items():
        if python_list_format:
            features_str = str(features)                    # "['a','b']"
        else:
            features_str = ",".join(features)               # "a,b"

        rows.append({
            "feature_set_name": name,
            "features": features_str
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Feature sets CSV written to: {output_path}")
    return df


# ======================================================
# CLI SUPPORT
# ======================================================

import pandas as pd
import ast

def parse_feature_sets(csv_path):
    df = pd.read_csv(csv_path)

    if "feature_set_name" not in df.columns or "features" not in df.columns:
        raise ValueError("CSV must contain columns: 'feature_set_name' and 'features'")

    feature_sets = {}
    for _, row in df.iterrows():
        name = row["feature_set_name"]
        value = str(row["features"]).strip()

        # --- Case 1: Python list string ---
        # Examples: "['a','b']", ["x", "y"]
        if value.startswith("[") and value.endswith("]"):
            try:
                features = ast.literal_eval(value)
                # ensure list of strings
                features = [str(f).strip() for f in features]
            except Exception:
                raise ValueError(f"Invalid Python list syntax in features column: {value}")

        # --- Case 2: comma-separated list ---
        else:
            features = [f.strip() for f in value.split(",") if f.strip()]

        feature_sets[name] = features

    return feature_sets

def add_parser(subparser):
    subparser.add_argument("--eval_csv", required=True, help="Evaluation dataframe CSV.")
    subparser.add_argument("--feature_sets_csv", required=True, help="Feature sets CSV.")
    subparser.add_argument("--dataset_name", required=True, help="Dataset identifier.")
    subparser.add_argument("--output_path", required=True, help="Directory for outputs.")
    return subparser

def main(args):
    eval_df = pd.read_csv(args.eval_csv)
    feature_sets = parse_feature_sets(args.feature_sets_csv)

    os.makedirs(args.output_path, exist_ok=True)

    _, metrics_df, preds_df = peak_gene_link_train(
        eval_df=eval_df,
        feature_sets=feature_sets,
        dataset_name=args.dataset_name,
        output_path=args.output_path,
    )

    metrics_df.to_csv(os.path.join(args.output_path, "metrics.csv"), index=False)
    preds_df.to_csv(os.path.join(args.output_path, "preds.csv"), index=False)

    print(f"Finished supervised training → {args.output_path}")
