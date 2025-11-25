#!/usr/bin/env python3

import joblib
import pandas as pd

# ======================================================
# Load Model Bundle (model, scaler, feature_names)
# ======================================================

def load_final_model(model_path: str):
    """
    Load final XGBoost model bundle saved by train_xgboost.py.
    Returns:
        {
          "model": XGBClassifier,
          "scaler": Optional[Scaler],
          "feature_names": list[str],
          "best_hyperparams": dict
        }
    """
    return joblib.load(model_path)


def peak_gene_link_predict(model_bundle, new_df: pd.DataFrame):
    """
    Predict probabilities for new_df using the final model bundle.

    Args:
        model_bundle: dict {"model", "scaler", "feature_names"}
        new_df: pd.DataFrame containing columns for feature_names

    Returns:
        numpy array of prediction probabilities (P(label=1))
    """
    model = model_bundle["model"]
    scaler = model_bundle["scaler"]
    feature_names = model_bundle["feature_names"]

    # Select only required feature columns
    X = new_df[feature_names].copy()

    # Scale if needed
    if scaler is not None:
        X = scaler.transform(X)
    else:
        X = X.values

    preds = model.predict_proba(X)[:, 1]
    return preds

# ======================================================
# CLI Parser
# ======================================================

def add_parser(parser):
    parser.add_argument(
        "--input_csv", 
        required=True,
        help="CSV with new data containing feature columns."
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Path to the FINAL model bundle (*.pkl) saved by train_xgboost.py."
    )
    parser.add_argument(
        "--output_csv",
        required=True,
        help="Where to save prediction results."
    )
    parser.add_argument(
        "--id_col",
        default="peak_gene_pair",
        help="Identifier column to keep in output."
    )
    return parser


# ======================================================
# Main
# ======================================================

def main(args):

    model_bundle = load_final_model(args.model_dir)

    df = pd.read_csv(args.input_csv)

    missing = [c for c in model_bundle["feature_names"] if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(
            f"Missing required feature columns for prediction: {missing}"
        )

    pred = peak_gene_link_predict(model_bundle, df)

    out_df = pd.DataFrame({
        args.id_col: df[args.id_col],
        "pred_prob": pred
    })

    out_df.to_csv(args.output_csv, index=False)
    print(f"Predictions saved → {args.output_csv}")

