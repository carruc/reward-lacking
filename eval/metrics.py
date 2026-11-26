"""Evaluation metrics and summary report generators."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def compute_classification_metrics(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    num_classes: int = 10,
) -> dict:
    """Compute weighted F1, macro F1, accuracy, per-class F1, and confusion matrix."""
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    labels = list(range(num_classes))

    weighted_f1 = float(f1_score(y_true_arr, y_pred_arr, labels=labels, average="weighted", zero_division=0))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0))
    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    per_class_vals = f1_score(y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0)
    per_class_f1 = {str(idx): round(float(val), 6) for idx, val in enumerate(per_class_vals)}
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels).tolist()

    return {
        "weighted_f1": round(weighted_f1, 6),
        "macro_f1": round(macro_f1, 6),
        "accuracy": round(acc, 6),
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
    }


def _write_text_atomic(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_summary_tables(out_root: Path) -> pd.DataFrame:
    """Aggregate ``*/metrics.json`` under ``out_root`` into ``summary.csv`` and ``summary.md``."""
    out_root = Path(out_root)
    rows: list[dict] = []
    for metrics_file in sorted(out_root.glob("*/metrics.json")):
        data = json.loads(metrics_file.read_text(encoding="utf-8"))
        rows.append(
            {
                "run_id": data["run_id"],
                "arch": data["arch"],
                "init": data["init"],
                "stage": data["stage"],
                "best_epoch": int(data["best_epoch"]),
                "epochs_ran": int(data.get("epochs_ran", data["best_epoch"])),
                "best_val_weighted_f1": float(data["best_val_weighted_f1"]),
                "best_val_macro_f1": float(data["best_val_macro_f1"]),
                "best_val_acc": float(data["best_val_acc"]),
                "shifted_val_weighted_f1": float(data["shifted_val_weighted_f1"]),
                "shifted_val_macro_f1": float(data.get("shifted_val_macro_f1", float("nan"))),
                "shifted_val_acc": float(data.get("shifted_val_acc", float("nan"))),
                "hard_shifted_val_weighted_f1": float(data.get("hard_shifted_val_weighted_f1", float("nan"))),
                "train_time_sec": float(data["train_time_sec"]),
            }
        )

    columns = [
        "run_id",
        "arch",
        "init",
        "stage",
        "best_epoch",
        "epochs_ran",
        "best_val_weighted_f1",
        "best_val_macro_f1",
        "best_val_acc",
        "shifted_val_weighted_f1",
        "shifted_val_macro_f1",
        "shifted_val_acc",
        "hard_shifted_val_weighted_f1",
        "train_time_sec",
    ]
    df = pd.DataFrame(rows, columns=columns)
    if not df.empty:
        df = df.sort_values("best_val_weighted_f1", ascending=False).reset_index(drop=True)

    csv_path = out_root / "summary.csv"
    tmp_csv = csv_path.with_suffix(".csv.tmp")
    out_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp_csv, index=False, lineterminator="\n")
    tmp_csv.replace(csv_path)

    md_lines = [
        f"# Screening Summary ({len(df)} completed runs)",
        "",
        "| Rank | run_id | arch | init | stage | best_epoch | best_val_weighted_f1 | best_val_macro_f1 | best_val_acc | shifted_val_weighted_f1 | hard_shifted_val_weighted_f1 | train_time_sec |",
        "| :---: | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for rank, r in enumerate(df.itertuples(index=False), start=1):
        hard_str = "—" if np.isnan(r.hard_shifted_val_weighted_f1) else f"{r.hard_shifted_val_weighted_f1:.4f}"
        md_lines.append(
            f"| {rank} | `{r.run_id}` | {r.arch} | {r.init} | {r.stage} | {r.best_epoch} | "
            f"{r.best_val_weighted_f1:.4f} | {r.best_val_macro_f1:.4f} | {r.best_val_acc:.4f} | "
            f"{r.shifted_val_weighted_f1:.4f} | {hard_str} | {r.train_time_sec:.1f}s |"
        )
    _write_text_atomic("\n".join(md_lines) + "\n", out_root / "summary.md")
    return df
