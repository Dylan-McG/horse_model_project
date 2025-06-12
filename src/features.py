# src/features.py
import pandas as pd
import numpy as np

# ==========================================================
# 🧠 Feature Engineering Pipeline for Horse Race Modeling
# ----------------------------------------------------------
# Adds domain-informed, statistical, and market-based features
# for predictive modeling and diagnostics.
# ==========================================================


# === 1. Race-Relative Features ===
def add_race_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    df["rel_Speed_PreviousRun"] = df.groupby("Race_ID")["Speed_PreviousRun"].transform(lambda x: x - x.mean())
    df["rank_Speed_PreviousRun"] = df.groupby("Race_ID")["Speed_PreviousRun"].rank(ascending=False)
    df["z_Speed_PreviousRun"] = df.groupby("Race_ID")["Speed_PreviousRun"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-5)
    )
    df["Field_Size"] = df.groupby("Race_ID")["Horse"].transform("count")
    return df


# === 2. Interaction Features ===
def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df["JockeyTrainerCombo"] = df["JockeyRating"] * df["TrainerRating"]
    df["SpeedTrainerInt"] = df["Speed_PreviousRun"] * df["TrainerRating"]
    return df


# === 3. Form Features ===
def add_form_features(df: pd.DataFrame) -> pd.DataFrame:
    df["RecentRun"] = pd.cut(
        df["daysSinceLastRun"],
        bins=[-1, 7, 30, 90, 9999],
        labels=["<7d", "8–30d", "31–90d", "90d+"]
    )
    if "PastWins" in df.columns:
        df["Ever_Won"] = (df["PastWins"] > 0).astype(int)
    return df


# === 4. Market-Based Features ===
def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    df["log_MarketOdds"] = np.log1p(df["Market_Odds"])
    df["MarketRank"] = df.groupby("Race_ID")["Market_Odds"].rank()
    df["Market_Prob"] = 1 / df["Market_Odds"]
    df["Market_Prob_Z"] = df.groupby("Race_ID")["Market_Prob"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-5)
    )
    return df


# === 5. Form Aggregates ===
def add_form_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    df["SpeedAvg2"] = df[["Speed_PreviousRun", "Speed_2ndPreviousRun"]].mean(axis=1)
    df["OddsAvg2"] = df[["MarketOdds_PreviousRun", "MarketOdds_2ndPreviousRun"]].mean(axis=1)
    return df


# === 6. Race Context Features ===
def add_race_context_features(df: pd.DataFrame) -> pd.DataFrame:
    df["RaceMeanOdds"] = df.groupby("Race_ID")["Market_Odds"].transform("mean")
    df["RaceStdOdds"] = df.groupby("Race_ID")["Market_Odds"].transform("std")
    df["OddsToMeanRatio"] = df["Market_Odds"] / df["RaceMeanOdds"]
    return df


# === 7. Advanced Interactions ===
def add_advanced_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df["TrainerFormCombo"] = df["TrainerRating"] * df["Speed_PreviousRun"]
    df["TrainerRecencyCombo"] = df["TrainerRating"] / (df["daysSinceLastRun"] + 1)
    return df


# === 8. Rolling Win Rate Features ===
def add_rolling_win_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling win rates per Jockey and Trainer.
    Assumes df is sorted chronologically by race order.
    """
    for col in ["JockeyID", "TrainerID"]:
        if col in df.columns and "Winner" in df.columns:
            rolling = (
                df[[col, "Winner"]]
                .groupby(col)["Winner"]
                .rolling(window=50, min_periods=10)
                .mean()
                .reset_index()
                .rename(columns={"Winner": f"{col}_WinRate50"})
            )
            df = df.merge(rolling, on=["level_1", col], how="left")
    return df


# === 9. Track & Distance Encoding ===
def add_track_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds track familiarity and distance bucket features.
    """
    # Track familiarity: how many times this horse has run at this track
    if "Track" in df.columns and "HorseID" in df.columns:
        df["Track_Experience"] = df.groupby(["HorseID", "Track"]).cumcount()

    # Clean and bucket race distances
    if "Distance" in df.columns:
        # Extract numeric part from mixed string (e.g., "2400m", "1m4f")
        extracted = df["Distance"].astype(str).str.extract(r"(\d+\.?\d*)")[0]
        df["Distance"] = pd.to_numeric(extracted, errors="coerce")

        # Sanity check: how many didn't convert?
        num_failed = df["Distance"].isna().sum()
        if num_failed > 0:
            print(f"⚠️ Warning: {num_failed} Distance values could not be parsed and will be NaN.")

        # Apply bucket cuts (only if distance is numeric)
        df["Distance_Bucket"] = pd.cut(
            df["Distance"],
            bins=[0, 1200, 1600, 2000, 2400, 3000, np.inf],
            labels=["Sprint", "Mile", "Intermediate", "Classic", "Long", "Marathon"]
        )

    return df


# === Master Feature Pipeline ===
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_race_relative_features(df)
    df = add_interaction_features(df)
    df = add_form_features(df)
    df = add_market_features(df)
    df = add_form_aggregates(df)
    df = add_race_context_features(df)
    df = add_advanced_interactions(df)
    df = add_rolling_win_rates(df)
    df = add_track_distance_features(df)
    return df


# === Diagnostics: Feature Summary ===
def feature_summary(df: pd.DataFrame, label_col="Winner") -> pd.DataFrame:
    """
    Returns a diagnostic table showing:
    - Correlation to label (if numeric)
    - Null percentage
    - Data type
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns.drop(label_col, errors="ignore")

    summary = pd.DataFrame({
        "Null_%": df[numeric_cols].isnull().mean() * 100,
        "Corr_to_Label": df[numeric_cols].corrwith(df[label_col]) if label_col in df.columns else np.nan,
        "Type": df[numeric_cols].dtypes
    }).sort_values("Corr_to_Label", ascending=False)

    return summary.reset_index().rename(columns={"index": "Feature"})


