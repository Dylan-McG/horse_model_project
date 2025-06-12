# src/processing/combine.py

import pandas as pd
from pathlib import Path

def create_edge_backtest_file(pred_path, market_path, labels_path, out_path):
    """
    Combines model predictions, market odds, and true labels into a single
    backtest-ready dataset with computed edge scores.
    """
    # Load data
    preds = pd.read_csv(pred_path)
    market = pd.read_csv(market_path)
    labels = pd.read_csv(labels_path)[["Race_ID", "Horse", "Position"]]

    # Merge
    df = preds.merge(market, on=["Race_ID", "Horse"], how="left")
    df = df.merge(labels, on=["Race_ID", "Horse"], how="left")

    # Compute edge
    df["Market_Prob"] = 1 / df["Market_Odds"]
    df["Edge_Score"] = df["Predicted_Probability"] - df["Market_Prob"]
    df["True_Label"] = (df["Position"] == 1).astype(int)

    df.drop(columns=["Position"], inplace=True)

    # Save
    df.to_csv(out_path, index=False)
    print(f"✅ Saved: {out_path}")
    return df
