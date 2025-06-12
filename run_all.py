import subprocess
from pathlib import Path

# List your notebooks in order
notebooks = [
    "01_eda.ipynb",
    "02_features.ipynb",
    "03_models.ipynb",
    "04_hyperparam_tuning.ipynb",
    "05_ensemble_stack.ipynb",
    "06_metrics_evaluation.ipynb",
    "07_visuals.ipynb",
    "08_edge_analysis.ipynb",
    "09_portfolio_simulator.ipynb"
    "10_market_edge.ipynb"
]

NOTEBOOK_DIR = Path(__file__).parent / "notebooks"

def run_notebook(nb):
    nb_path = NOTEBOOK_DIR / nb
    print(f"\n=== Running {nb} ===")
    result = subprocess.run([
        "jupyter", "nbconvert", "--to", "notebook", "--execute",
        "--inplace", str(nb_path)
    ])
    if result.returncode != 0:
        print(f"\n[ERROR] {nb} failed.")
        exit(result.returncode)
    print(f"=== Finished {nb} ===\n")

if __name__ == "__main__":
    for nb in notebooks:
        run_notebook(nb)
    print("\nAll notebooks completed successfully.")
