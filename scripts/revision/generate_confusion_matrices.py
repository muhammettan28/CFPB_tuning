from project_paths import project_path
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cfpb-paper")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = project_path(ROOT, "experiment_archive", "outputs")
FIGURES = project_path(ROOT, "REVISION_PAPER", "figures")

LABELS = [
    "retail_banking",
    "credit_reporting",
    "credit_card",
    "mortgages_and_loans",
    "debt_collection",
]

SHORT_LABELS = [
    "retail\nbanking",
    "credit\nreporting",
    "credit\ncard",
    "mortgages\nloans",
    "debt\ncollection",
]

MODELS = {
    "Qwen3-8B": {
        "base": "qwen3_8b_final_eval_base_zero_shot_*/internal_test/predictions.csv",
        "fine_tuned": "qwen3_8b_final_eval_fine_tuned_seed*/internal_test/predictions.csv",
    },
    "Llama 3.1-8B": {
        "base": "llama31_8b_final_eval_base_zero_shot_*/internal_test/predictions.csv",
        "fine_tuned": "llama31_8b_final_eval_fine_tuned_seed*/internal_test/predictions.csv",
    },
}


def read_confusion(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    return confusion_matrix(df["true_label"], df["prediction"], labels=LABELS).astype(float)


def load_matrix(pattern: str, average: bool) -> np.ndarray:
    paths = sorted(ARCHIVE.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No prediction files matched {pattern}")
    matrices = [read_confusion(path) for path in paths]
    if average:
        return np.mean(matrices, axis=0)
    if len(matrices) != 1:
        raise ValueError(f"Expected one base file for {pattern}, found {len(matrices)}")
    return matrices[0]


def row_normalise(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 9.0), constrained_layout=True)

    panels = []
    for model_name, patterns in MODELS.items():
        panels.append((model_name, "Base zero-shot", load_matrix(patterns["base"], average=False)))
        panels.append((model_name, "QLoRA fine-tuned mean", load_matrix(patterns["fine_tuned"], average=True)))

    vmax = 1.0
    for ax, (model_name, variant, counts) in zip(axes.ravel(), panels):
        normalised = row_normalise(counts)
        image = ax.imshow(normalised, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_title(f"{model_name}: {variant}", fontsize=11)
        ax.set_xticks(range(len(LABELS)))
        ax.set_xticklabels(SHORT_LABELS, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(LABELS)))
        ax.set_yticklabels(SHORT_LABELS, fontsize=8)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")

        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                value = normalised[i, j]
                count = counts[i, j]
                text_color = "white" if value >= 0.55 else "black"
                count_text = f"{count:.0f}" if abs(count - round(count)) < 1e-6 else f"{count:.1f}"
                ax.text(
                    j,
                    i,
                    f"{value * 100:.0f}%\n({count_text})",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=text_color,
                )

    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.86)
    colorbar.set_label("Row-normalised proportion")
    fig.suptitle("Internal-test confusion matrices", fontsize=13)

    for ext in ["pdf", "png"]:
        fig.savefig(FIGURES / f"internal_confusion_matrices.{ext}", dpi=300)


if __name__ == "__main__":
    main()
