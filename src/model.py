# src/model.py

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, brier_score_loss
from lightgbm.callback import early_stopping, log_evaluation


def softmax_stable(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax function."""
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def train_and_evaluate(X: pd.DataFrame,
                       y: pd.Series,
                       race_ids: pd.Series,
                       num_folds: int = 5):
    """
    Train LightGBM LambdaRank model with GroupKFold CV.

    Returns:
    - models: Trained LightGBM models per fold
    - oof_probs: Out-of-fold predicted win probabilities (softmax-normalised)
    - feature_importance: DataFrame of average feature importances
    """
    gkf = GroupKFold(n_splits=num_folds)
    oof_scores = pd.Series(index=X.index, dtype=float)
    models = []
    fold_metrics = []
    feature_importances = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=race_ids)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        train_group_sizes = race_ids.iloc[train_idx].value_counts().sort_index().values
        val_group_sizes = race_ids.iloc[val_idx].value_counts().sort_index().values

        model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=5000,
            learning_rate=0.05,
            importance_type="gain",
            random_state=42
        )

        print(f"\n🚀 Training Fold {fold + 1}/{num_folds}...")
        model.fit(
            X_train, y_train,
            group=train_group_sizes,
            eval_set=[(X_val, y_val)],
            eval_group=[val_group_sizes],
            callbacks=[
                early_stopping(stopping_rounds=100),
                log_evaluation(period=100)
            ]
        )

        val_scores = model.predict(X_val)
        oof_scores.iloc[val_idx] = val_scores

        # Softmax transform within each race
        val_races = race_ids.iloc[val_idx]
        df_val = pd.DataFrame({
            "Race_ID": val_races,
            "score": val_scores,
            "True_Label": y_val
        })
        df_val["Predicted_Probability"] = df_val.groupby("Race_ID")["score"].transform(softmax_stable)

        # Metrics
        log = log_loss(df_val["True_Label"], df_val["Predicted_Probability"])
        brier = brier_score_loss(df_val["True_Label"], df_val["Predicted_Probability"])
        print(f"📊 Fold {fold + 1} — Log Loss: {log:.5f} | Brier: {brier:.5f}")
        fold_metrics.append((log, brier))

        # Importance
        imp_df = pd.DataFrame({
            "feature": X.columns,
            "importance": model.booster_.feature_importance(importance_type="gain"),
            "fold": fold + 1
        })
        feature_importances.append(imp_df)

        models.append(model)

    # Combine OOF predictions
    full_df = pd.DataFrame({
        "Race_ID": race_ids,
        "score": oof_scores,
        "True_Label": y
    })
    full_df["Predicted_Probability"] = full_df.groupby("Race_ID")["score"].transform(softmax_stable)

    # Aggregate metrics
    log = log_loss(full_df["True_Label"], full_df["Predicted_Probability"])
    brier = brier_score_loss(full_df["True_Label"], full_df["Predicted_Probability"])

    print("\n📈 Final Out-of-Fold Evaluation")
    print("-------------------------------")
    print(f"Mean Log Loss:    {log:.5f}")
    print(f"Mean Brier Score: {brier:.5f}")

    # Aggregate feature importance
    importance_df = pd.concat(feature_importances)
    avg_importance = (importance_df.groupby("feature")["importance"]
                      .mean()
                      .sort_values(ascending=False)
                      .reset_index())

    return models, full_df["Predicted_Probability"], avg_importance



