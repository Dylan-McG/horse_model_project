import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 4)

# === Race-Level Plots ===
def plot_field_size_histogram(df: pd.DataFrame):
    plt.figure()
    sns.histplot(df["Field_Size"], bins=10, color="skyblue")
    plt.title("📐 Field Size Distribution")
    plt.xlabel("Horses per Race")
    plt.ylabel("Number of Races")
    plt.tight_layout()
    return plt

def plot_margin_distribution(df: pd.DataFrame):
    plt.figure()
    sns.histplot(df["Margin_Top2"], bins=20, color="orange")
    plt.axvline(0.10, color='red', linestyle='--', label="Clear Favourite Threshold")
    plt.title("⚖️ Margin Between Top 2 Horses")
    plt.xlabel("Win Probability Margin")
    plt.ylabel("Number of Races")
    plt.legend()
    plt.tight_layout()
    return plt

def plot_race_edge_map(df: pd.DataFrame):
    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=df,
        x='Entropy',
        y='Margin_Top2',
        size='Adj_Confidence',
        hue='Adj_Confidence',
        palette='viridis',
        sizes=(20, 200),
        legend=False
    )
    plt.axhline(0.10, color='red', linestyle='--', linewidth=1, label="Strong Favourite Margin")
    plt.title("📊 Race Edge Map: Entropy vs. Margin")
    plt.xlabel("Race Entropy (Uncertainty)")
    plt.ylabel("Top 2 Margin (Confidence Gap)")
    plt.grid(True)
    plt.tight_layout()
    return plt

# === Horse-Level Plots ===
def plot_top_picks_barplot(df: pd.DataFrame, title: str, dynamic_xlim=False):
    plt.figure(figsize=(10, 5))
    sns.barplot(x="Predicted_Probability", y="Horse", data=df, palette="Blues_d")
    plt.title(title)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Horse")

    if dynamic_xlim:
        min_prob = df["Predicted_Probability"].min()
        max_prob = df["Predicted_Probability"].max()
        plt.xlim(max(min_prob - 0.05, 0), min(max_prob + 0.05, 1))
    else:
        plt.xlim(0.5, 1.0)

    plt.tight_layout()
    return plt

# === Confidence Distribution ===
def plot_confidence_distribution(df: pd.DataFrame):
    plt.figure()
    sns.barplot(x="Bin", y="Count", data=df, color="mediumpurple")
    plt.xticks(rotation=45)
    plt.title("🔢 Distribution of Predicted Probabilities")
    plt.xlabel("Confidence Bin")
    plt.ylabel("Number of Horses")
    plt.tight_layout()
    return plt

# === Mispriced Horses Plot ===
def plot_mispriced_edge_barplot(df: pd.DataFrame):
    plt.figure(figsize=(10, 5))
    sns.barplot(x="Edge_Score", y="Horse", data=df, palette="Greens_d")
    plt.title("Top 10 Mispriced Horses by Positive Edge")
    plt.xlabel("Model Edge (Model - Market Probability)")
    plt.ylabel("Horse")
    plt.tight_layout()
    return plt
