import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    auc,
)
import wandb
from pathlib import Path
from tqdm import tqdm
import pickle
import os


# Load models from wandb
def get_wandb_path(
    dictionary_id,
    wandb_path="wandb/",
    model_name="best_model_5.keras",
):
    id_to_path = {}
    for folder_name in os.listdir(wandb_path):
        if folder_name.startswith("run-") and "-" in folder_name:
            run_id = folder_name.split("-")[-1]
            full_path = os.path.join(wandb_path, folder_name)
            id_to_path[run_id] = full_path

    # Create the new dictionary using the original one
    weight_path_dictionary = {
        key: os.path.join(id_to_path.get(run_id, None), "files", model_name)
        for key, run_id in dictionary_id.items()
    }
    return weight_path_dictionary


def check_models_exist(weight_path_dictionary):
    model_info_dict = {}

    for key, file_path in weight_path_dictionary.items():
        if os.path.isfile(file_path):
            size_bytes = os.path.getsize(file_path)
            size_mb = size_bytes / (1024 * 1024)  # Convert to MB
            model_info_dict[key] = {
                "path": file_path,
                "exists": True,
                "size_mb": round(size_mb, 2),
            }
        else:
            model_info_dict[key] = {
                "path": file_path,
                "exists": False,
                "size_mb": None,
            }

    # Optional: pretty print
    import pprint

    pprint.pprint(model_info_dict, sort_dicts=False)


# Save and load predictions into the dataframe
def save_predictions_to_df(df, y_probs, class_names, model_name=None):
    if model_name is None:
        model_name = ""
    else:
        model_name = f"{model_name}_"
    for i, cls in enumerate(class_names):
        df[f"{model_name}{cls}_pred"] = y_probs[:, i]
    return df


def get_predictions_from_df(df, class_names, model_name=None, dataset=None):
    if model_name is None:
        model_name = ""
    else:
        model_name = f"{model_name}_"
    if dataset is not None:
        df = df.loc[df.dataset == dataset]
    y_true = df[class_names].values
    class_names_pred = [
        model_name + clname + "_pred" for clname in class_names
    ]
    y_probs = df[class_names_pred].values
    return y_true, y_probs


# ==============================================================================
# Helper Functions
# ==============================================================================


def _ensure_dir(path):
    """Ensure the directory exists."""
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)


def _save_or_log_plot(
    fig, title, save_path=None, log_wandb=False, wandb_key=None, display=False
):
    """Helper to save plot, log to WandB, and optionally display."""
    if save_path:
        _ensure_dir(save_path)
        # Use a tight layout before saving
        try:
            fig.tight_layout()
        except Exception:
            # Some plots might not support tight_layout
            pass
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved plot: {save_path}")
    if log_wandb and wandb.run:
        wandb_key = wandb_key or title.replace(
            " ", "_"
        )  # Basic key generation
        try:
            wandb.log({wandb_key: wandb.Image(fig)})
        except Exception as e:
            # Shorten error message slightly
            print(f"Warn: Failed logging '{wandb_key}'. Error: {e}")
    if display:
        plt.show()
    plt.close(fig)


def format_global_summary(global_summary_dict, digits=3):
    """Formats global bootstrap summary dictionary into a DataFrame."""
    records = []
    categories_map = {
        "accuracy": ("Overall", "Accuracy"),
        "macro_precision": ("Macro Average", "Precision"),
        "macro_recall": ("Macro Average", "Recall (Sensitivity)"),
        "macro_f1": ("Macro Average", "F1-score"),
        "macro_auc": ("Macro Average", "AUC"),
        "micro_precision": ("Micro Average", "Precision"),
        "micro_recall": ("Micro Average", "Recall (Sensitivity)"),
        "micro_f1": ("Micro Average", "F1-score"),
        # Micro AUC is often same as accuracy or roc_auc_score(y_true.ravel(), y_pred.ravel())
        # For simplicity, we can omit it or calculate it explicitly if needed.
        "weighted_precision": ("Weighted Average", "Precision"),
        "weighted_recall": ("Weighted Average", "Recall (Sensitivity)"),
        "weighted_f1": ("Weighted Average", "F1-score"),
        "weighted_auc": ("Weighted Average", "AUC"),
    }

    # Initialize rows for each category
    output_categories = [
        "Overall",
        "Macro Average",
        "Micro Average",
        "Weighted Average",
    ]
    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall (Sensitivity)",
        "F1-score",
        "AUC",
    ]

    data_for_df = {
        cat: {metric: "N/A" for metric in metric_columns}
        for cat in output_categories
    }

    for metric_key, (cat_name, metric_name) in categories_map.items():
        if metric_key in global_summary_dict:
            val_dict = global_summary_dict[metric_key]
            if (
                isinstance(val_dict, dict)
                and "mean" in val_dict
                and "95% CI" in val_dict
            ):
                if not np.isnan(val_dict["mean"]):
                    mean = round(val_dict["mean"], digits)
                    ci_low = round(val_dict["95% CI"][0], digits)
                    ci_high = round(val_dict["95% CI"][1], digits)
                    data_for_df[cat_name][metric_name] = (
                        f"{mean} ({ci_low}–{ci_high})"
                    )
                else:
                    data_for_df[cat_name][metric_name] = "N/A"  # NaN mean
            else:  # Point estimate case (no CI)
                if not pd.isna(val_dict):  # Check for np.nan or float
                    data_for_df[cat_name][metric_name] = (
                        f"{val_dict:.{digits}f}"
                    )
                else:
                    data_for_df[cat_name][metric_name] = "N/A"

    # Convert to list of records for DataFrame constructor
    for cat in output_categories:
        row = {"Category": cat}
        row.update(data_for_df[cat])
        records.append(row)

    df = pd.DataFrame(records)
    # Ensure correct column order
    ordered_cols = ["Category"] + metric_columns
    df = df[ordered_cols]
    return df


def format_per_class_summary(per_class_dict, digits=3, y_true=None):
    """Formats per-class bootstrap summary dictionary into a DataFrame."""
    records = []
    class_names_ordered = list(per_class_dict.keys())
    # Reordered metrics as requested: Sensitivity, Specificity, PPV, NPV, F1, AUC, AP
    metric_names_ordered = [
        "prevalence",
        "recall",
        "specificity",
        "precision",
        "npv",
        "f1",
        "auc",
        "ap",
    ]
    display_names = {
        "auc": "AUC",
        "ap": "AP",
        "precision": "PPV (Precision)",
        "recall": "Sensitivity (Recall)",
        "specificity": "Specificity",
        "npv": "NPV",
        "f1": "F1-score",
        "prevalence": "Prevalence",
    }

    # Calculate prevalence if y_true is provided
    prevalence_dict = {}
    if y_true is not None:
        n_samples = y_true.shape[0]
        for i, cls in enumerate(class_names_ordered):
            pos_count = np.sum(y_true[:, i])
            prevalence_dict[cls] = {
                "mean": float(pos_count) / n_samples,
                "absolute": int(pos_count),
            }

    for cls in class_names_ordered:
        metrics_data = per_class_dict.get(cls, {})
        row = {"Class": cls}
        for metric_key in metric_names_ordered:
            # Special handling for prevalence
            if metric_key == "prevalence" and y_true is not None:
                if cls in prevalence_dict:
                    prev_val = prevalence_dict[cls]["mean"]
                    abs_val = prevalence_dict[cls]["absolute"]
                    row[display_names[metric_key]] = (
                        f"{prev_val:.1%} ({abs_val}/{y_true.shape[0]})"
                    )
                else:
                    row[display_names[metric_key]] = "N/A"
                continue

            values = metrics_data.get(metric_key)
            display_name = display_names.get(
                metric_key, metric_key.capitalize()
            )
            # Handle bootstrapped values (dict with mean/CI) or point estimates (float)
            if (
                isinstance(values, dict)
                and "mean" in values
                and "95% CI" in values
            ):
                if not np.isnan(values["mean"]):
                    mean = round(values["mean"], digits)
                    ci_low = round(values["95% CI"][0], digits)
                    ci_high = round(values["95% CI"][1], digits)
                    row[display_name] = f"{mean} ({ci_low}–{ci_high})"
                else:
                    row[display_name] = "N/A"  # NaN mean from bootstrap
            elif isinstance(values, (float, int, np.number)) and not np.isnan(
                values
            ):
                row[display_name] = f"{values:.{digits}f}"  # Point estimate
            else:
                row[display_name] = (
                    "N/A"  # Missing metric or NaN point estimate
                )
        records.append(row)

    df = pd.DataFrame(records)
    # Ensure all expected display columns are present and ordered
    cols = ["Class"] + [
        display_names.get(m, m.capitalize()) for m in metric_names_ordered
    ]
    # Filter out columns that were never added if a metric was consistently missing
    cols = [c for c in cols if c in df.columns or c == "Class"]
    df = df[cols]
    return df


# ==============================================================================
# Plotting Functions
# ==============================================================================


def plot_cooccurrence_matrix(
    y_true, class_names, save_dir=None, log_wandb=False, display=False
):
    """Calculates and plots/saves the label co-occurrence matrix."""
    print("🧩 Calculating Label Co-occurrence...")
    if y_true.ndim == 1 or y_true.shape[1] < 2:
        print(
            "Warning: Co-occurrence matrix requires multi-label format "
            "(at least 2 classes). Skipping."
        )
        return None

    n_classes = len(class_names)
    if y_true.shape[1] != n_classes:
        print(
            f"Warning: y_true shape {y_true.shape} doesn't match "
            f"n_classes {n_classes}. Skipping co-occurrence."
        )
        return None

    co_occurrence = pd.DataFrame(y_true, columns=class_names).astype(
        int
    ).T @ pd.DataFrame(y_true, columns=class_names).astype(int)

    # Convert to numpy array for manipulation
    co_occurrence_np = co_occurrence.values.copy()

    # Get class counts (diagonal elements)
    diag_values = np.diag(co_occurrence_np).copy()

    # Create a mask for the upper triangle
    mask_upper = np.triu(np.ones_like(co_occurrence_np, dtype=bool), k=1)

    # Plot triangular matrix with diagonal values on white background
    figsize_x = max(6, n_classes * 0.8)
    figsize_y = max(5, n_classes * 0.7)
    fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))

    # Plot without diagonal (zeros on diagonal, masked upper triangle)
    co_occurrence_viz = co_occurrence_np.copy()
    np.fill_diagonal(
        co_occurrence_viz, 0
    )  # Set diagonal to 0 for color scaling
    sns.heatmap(
        co_occurrence_viz,
        annot=True,
        fmt="d",
        cmap="Purples",
        ax=ax,
        mask=mask_upper,
        cbar=True,
    )

    # Get the font properties from the existing annotations
    existing_font = plt.rcParams.get("font.family")[0]
    existing_fontsize = plt.rcParams.get("font.size")

    # Plot diagonal values with white cells but matching font
    for i in range(len(diag_values)):
        # Add text annotations for diagonal values with white background
        text = ax.text(
            i + 0.5,
            i + 0.5,
            f"{diag_values[i]}",
            color="black",
            ha="center",
            va="center",
            fontsize=existing_fontsize,
            family=existing_font,
        )
        # Create white background for the text with minimal padding
        text.set_bbox(
            dict(
                facecolor="white",
                alpha=1,
                edgecolor="grey",
                boxstyle="round,pad=0.2",
                linewidth=0.5,
            )
        )

    # Make sure class names are displayed on both axes
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names, rotation=0)

    ax.set_title("Label Co-Occurrence Matrix")
    ax.set_xlabel("Class")
    ax.set_ylabel("Class")

    filename = (
        f"{save_dir}/label_cooccurrence_matrix.png" if save_dir else None
    )
    _save_or_log_plot(
        fig,
        "Label Co-Occurrence Matrix",
        filename,
        log_wandb,
        "Label_CoOccurrence_Matrix",
        display=display,
    )

    return co_occurrence


def plot_roc_curves(
    y_true,
    y_probs,
    class_names,
    save_dir=None,
    log_wandb=False,
    display=False,
    dataset_name=None,
):
    """Plots/saves ROC curves for each class."""
    dataset_suffix = f" - {dataset_name}" if dataset_name else ""
    print(f"📈 Plotting ROC Curves{dataset_suffix}...")
    roc_aucs = {}
    for i, class_name in enumerate(class_names):
        if len(np.unique(y_true[:, i])) < 2:
            print(
                f"Warning: Cannot compute ROC curve for class '{class_name}' "
                "- only one class present in y_true."
            )
            roc_aucs[class_name] = np.nan
            continue

        fpr, tpr, _ = roc_curve(y_true[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        roc_aucs[class_name] = roc_auc

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", color="#1f77b4")
        ax.plot([0, 1], [0, 1], "k--", label="_nolegend_")
        title = f"ROC Curve{dataset_suffix} - {class_name}"
        ax.set_title(title)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate (Sensitivity/Recall)")
        ax.legend(loc="lower right")
        ax.grid(True)

        filename = (
            f"{save_dir}/roc_curve_{class_name}.png" if save_dir else None
        )
        _save_or_log_plot(
            fig,
            f"ROC Curve - {class_name}",
            filename,
            log_wandb,
            f"ROC_{class_name}",
            display=display,
        )
    return roc_aucs


def plot_pr_curves(
    y_true,
    y_probs,
    class_names,
    save_dir=None,
    log_wandb=False,
    display=False,
    dataset_name=None,
):
    """Plots/saves Precision-Recall curves for each class."""
    dataset_suffix = f" - {dataset_name}" if dataset_name else ""
    print(f"📈 Plotting Precision-Recall Curves{dataset_suffix}...")
    avg_precisions = {}
    for i, class_name in enumerate(class_names):
        if len(np.unique(y_true[:, i])) < 2:
            print(
                f"Warning: Cannot compute PR curve for class '{class_name}' "
                "- only one class present in y_true."
            )
            avg_precisions[class_name] = np.nan
            continue

        precision, recall, _ = precision_recall_curve(
            y_true[:, i], y_probs[:, i]
        )
        ap = average_precision_score(y_true[:, i], y_probs[:, i])
        avg_precisions[class_name] = ap

        # Remove the artificial point at (0, 1) that sklearn adds
        if len(precision) > 1 and recall[0] == 0 and precision[0] == 1.0:
            recall = recall[1:]
            precision = precision[1:]

        fig, ax = plt.subplots()
        ax.plot(recall, precision, label=f"AP = {ap:.3f}", color="#1f77b4")
        title = f"Precision-Recall Curve{dataset_suffix} - {class_name}"
        ax.set_title(title)
        ax.set_xlabel("Recall (Sensitivity)")
        ax.set_ylabel("Precision")
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, 1.05)
        ax.legend()
        ax.grid(True)

        filename = (
            f"{save_dir}/pr_curve_{class_name}.png" if save_dir else None
        )
        _save_or_log_plot(
            fig,
            f"PR Curve - {class_name}",
            filename,
            log_wandb,
            f"PR_{class_name}",
            display=display,
        )
    return avg_precisions


def plot_confusion_matrices(
    y_true,
    y_pred,
    class_names,
    thresholds_dict=None,
    save_dir=None,
    log_wandb=False,
    display=False,
    dataset_name=None,
):
    """Plots/saves confusion matrices (raw, norm, combined) for each class."""
    dataset_suffix = f" - {dataset_name}" if dataset_name else ""
    print(f"📊 Plotting Confusion Matrices{dataset_suffix}...")
    cms_raw = {}
    cms_norm = {}

    if thresholds_dict is None:
        print(
            "Warning: No thresholds_dict for plot_confusion_matrices. Using N/A."
        )
        thresholds_dict = {}

    for i, class_name in enumerate(class_names):
        thresh = thresholds_dict.get(class_name, np.nan)  # Get threshold
        thresh_str = (
            f"{thresh:.3f}"
            if isinstance(thresh, (int, float, np.float32, np.float64))
            and not np.isnan(thresh)
            else "N/A"
        )
        title_suffix = f"{dataset_suffix} - {class_name}"

        cm_raw = confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1])
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_norm = confusion_matrix(
                y_true[:, i], y_pred[:, i], labels=[0, 1], normalize="true"
            )
            cm_norm = np.nan_to_num(cm_norm)
        cms_raw[class_name] = cm_raw
        cms_norm[class_name] = cm_norm

        fig_norm, ax_norm = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0,
            vmax=1,
            ax=ax_norm,
            cbar=False,
            square=True,
            xticklabels=["Neg", "Pos"],
            yticklabels=["Neg", "Pos"],
            annot_kws={"size": 20}  # Increased font size for annotations
        )
        ax_norm.set_title(
            f"Confusion Matrix (Norm, thr={thresh_str}){title_suffix}",
            fontsize=18  # Increased title font size
        )
        ax_norm.set_xlabel(
            "Predicted Label", fontsize=16
        )  # Increased label font size
        ax_norm.set_ylabel(
            "True Label", fontsize=16
        )  # Increased label font size
        ax_norm.tick_params(
            axis='both', which='major', labelsize=16
        )  # Increased tick label font size
        filename_norm = (
            f"{save_dir}/confusion_matrix_norm_{class_name}.png"
            if save_dir
            else None
        )
        _save_or_log_plot(
            fig_norm,
            f"CM_Norm_{class_name}",
            filename_norm,
            log_wandb,
            f"CM_Norm_{class_name}",
            display=display,
        )

        fig_raw, ax_raw = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm_raw,
            annot=True,
            fmt="d",
            cmap="Greens",
            ax=ax_raw,
            cbar=False,
            square=True,
            xticklabels=["Neg", "Pos"],
            yticklabels=["Neg", "Pos"],
            annot_kws={"size": 20}  # Increased font size for annotations
        )
        ax_raw.set_title(
            f"Confusion Matrix (Raw, thr={thresh_str}){title_suffix}",
            fontsize=18  # Increased title font size
        )
        ax_raw.set_xlabel(
            "Predicted Label", fontsize=16
        )  # Increased label font size
        ax_raw.set_ylabel(
            "True Label", fontsize=16
        )  # Increased label font size
        ax_raw.tick_params(
            axis='both', which='major', labelsize=16
        )  # Increased tick label font size
        filename_raw = (
            f"{save_dir}/confusion_matrix_raw_{class_name}.png"
            if save_dir
            else None
        )
        _save_or_log_plot(
            fig_raw,
            f"CM_Raw_{class_name}",
            filename_raw,
            log_wandb,
            f"CM_Raw_{class_name}",
            display=display,
        )

        fig_comb, ax_comb = plt.subplots(figsize=(8, 6))
        annot_comb = [["" for _ in range(2)] for _ in range(2)]
        for r_idx in range(2):
            for c_idx in range(2):
                raw_val = cm_raw[r_idx, c_idx]
                norm_val = cm_norm[r_idx, c_idx] * 100
                annot_comb[r_idx][c_idx] = f"$\\mathbf{{{norm_val:.1f}\\%}}$\n({raw_val})"
        sns.heatmap(
            cm_norm,
            annot=annot_comb,
            fmt="s",
            cmap="Blues",
            vmin=0,
            vmax=1,
            ax=ax_comb,
            cbar=True,
            square=True,
            cbar_kws={"label": "Row Normalization"},
            xticklabels=["Neg", "Pos"],
            yticklabels=["Neg", "Pos"],
            annot_kws={"size": 18}  # Increased font size for annotations
        )
        # User asked for Combined CM title change
        ax_comb.set_title(
            f"Confusion Matrix (thr={thresh_str}){title_suffix}",
            fontsize=18  # Increased title font size
        )
        ax_comb.set_xlabel(
            "Predicted Label", fontsize=16
        )  # Increased label font size
        ax_comb.set_ylabel(
            "True Label", fontsize=16
        )  # Increased label font size
        ax_comb.tick_params(
            axis='both', which='major', labelsize=16
        )  # Increased tick label font size
        filename_comb = (
            f"{save_dir}/confusion_matrix_combined_{class_name}.png"
            if save_dir
            else None
        )
        _save_or_log_plot(
            fig_comb,
            f"CM_Combined_{class_name}",
            filename_comb,
            log_wandb,
            f"CM_Combined_{class_name}",
            display=display,
        )
    return cms_raw, cms_norm


def plot_f1_vs_threshold(
    y_true,
    y_probs,
    class_names,
    best_thresholds,
    f1_scores,
    save_dir=None,
    log_wandb=False,
    display=False,
):
    """Plots F1 score vs. threshold and marks the optimal point."""
    print("🔁 Plotting F1 vs Threshold...")
    for i, class_name in enumerate(class_names):
        true = y_true[:, i]
        probs = y_probs[:, i]
        if len(np.unique(true)) < 2:
            print(f"Warning: F1 vs Threshold plot skipped for '{class_name}'.")
            continue
        precision, recall, thresholds_pr = precision_recall_curve(true, probs)
        f1s = 2 * (precision * recall) / (precision + recall + 1e-8)
        valid_f1s = f1s[1:] if len(f1s) > len(thresholds_pr) else f1s
        fig, ax = plt.subplots()
        min_len = min(len(thresholds_pr), len(valid_f1s))
        if min_len > 0:
            ax.plot(
                thresholds_pr[:min_len], valid_f1s[:min_len], label="F1 Score"
            )
        thresh_val = best_thresholds.get(class_name, np.nan)
        f1_val = f1_scores.get(class_name, np.nan)
        if not np.isnan(thresh_val) and not np.isnan(f1_val):
            ax.axvline(
                thresh_val,
                color="r",
                linestyle="--",
                label=f"Best Thr: {thresh_val:.3f} (F1={f1_val:.3f})",
            )
        ax.set_title(f"F1 Score vs Threshold – {class_name}")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("F1 Score")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="best")
        ax.grid(True)
        filename = (
            f"{save_dir}/f1_vs_threshold_{class_name}.png"
            if save_dir
            else None
        )
        _save_or_log_plot(
            fig,
            f"F1_vs_Thr_{class_name}",
            filename,
            log_wandb,
            f"F1_vs_Thr_{class_name}",
            display=display,
        )


def plot_threshold_sweep(
    y_true,
    y_probs,
    class_names,
    thresholds_sweep=None,
    save_dir=None,
    log_wandb=False,
    display=False,
):
    """Plots Precision, Recall, F1 vs. thresholds for each class."""
    print("🧹 Plotting Threshold Sweep Metrics...")
    if thresholds_sweep is None:
        thresholds_sweep = np.arange(0.0, 1.01, 0.1)

    all_rows = []
    for i, class_name in enumerate(class_names):
        true = y_true[:, i]
        probs = y_probs[:, i]
        unique_labels = np.unique(true)
        has_positives = 1 in unique_labels
        has_negatives = 0 in unique_labels

        # Calculate prevalence once per class
        prevalence = np.mean(true) if len(true) > 0 else 0
        prevalence_count = np.sum(true)
        prevalence_str = f"{prevalence:.1%} ({prevalence_count}/{len(true)})"

        f1s, precisions, recalls, specificities, npvs = [], [], [], [], []
        for t in thresholds_sweep:
            preds = (probs >= t).astype(int)
            precision = precision_score(true, preds, zero_division=0)
            recall = recall_score(true, preds, zero_division=0)
            f1 = f1_score(true, preds, zero_division=0)
            cm = confusion_matrix(true, preds, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            total = tn + fp + fn + tp

            # Calculate specificity and NPV
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0

            all_rows.append(
                {
                    "Class": class_name,
                    "Threshold": round(t, 3),
                    "Prevalence": prevalence_str,
                    "PPV (Precision)": (
                        round(precision, 4) if has_positives else np.nan
                    ),
                    "Sensitivity (Recall)": (
                        round(recall, 4) if has_positives else np.nan
                    ),
                    "Specificity": (
                        round(specificity, 4) if has_negatives else np.nan
                    ),
                    "NPV": round(npv, 4) if has_negatives else np.nan,
                    "F1 Score": (
                        round(f1, 4)
                        if has_positives and has_negatives
                        else np.nan
                    ),
                    "TP": int(tp),
                    "FP": int(fp),
                    "FN": int(fn),
                    "TN": int(tn),
                    "Total": int(total),
                }
            )
            precisions.append(precision if has_positives else np.nan)
            recalls.append(recall if has_positives else np.nan)
            specificities.append(specificity if has_negatives else np.nan)
            npvs.append(npv if has_negatives else np.nan)
            f1s.append(f1 if has_positives and has_negatives else np.nan)

        fig, ax = plt.subplots()
        ax.plot(thresholds_sweep, f1s, label="F1 Score", marker=".")
        ax.plot(
            thresholds_sweep, precisions, label="PPV (Precision)", marker="."
        )
        ax.plot(
            thresholds_sweep, recalls, label="Sensitivity (Recall)", marker="."
        )

        ax.set_title(f"Threshold Sweep – {class_name}")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Metric Value")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True)
        filename = (
            f"{save_dir}/threshold_sweep_{class_name}.png"
            if save_dir
            else None
        )
        _save_or_log_plot(
            fig,
            f"Thr_Sweep_{class_name}",
            filename,
            log_wandb,
            f"Thr_Sweep_{class_name}",
            display=display,
        )

    threshold_df = pd.DataFrame(all_rows)
    if save_dir:
        sweep_path = f"{save_dir}/threshold_sweep_table.csv"
        threshold_df.to_csv(sweep_path, index=False, float_format="%.4f")
        print(f"Saved threshold sweep table: {sweep_path}")
    if log_wandb and wandb.run:
        try:
            wandb.log(
                {"Threshold_Sweep_Table": wandb.Table(dataframe=threshold_df)}
            )
        except Exception as e:
            print(f"Warn: Failed log Thr_Sweep_Table. Err: {e}")
    return threshold_df


# ==============================================================================
# Core Calculation Functions
# ==============================================================================


def find_optimal_thresholds(
    y_true, y_probs, class_names, save_dir=None, log_wandb=False, display=False
):
    """Finds the best threshold per class based on maximizing F1 score."""
    print("⚙️ Finding Optimal Thresholds (Maximizing F1)...")
    best_thresholds = {}
    f1_scores_at_best = {}
    for i, class_name in enumerate(class_names):
        true = y_true[:, i]
        probs = y_probs[:, i]
        if len(np.unique(true)) < 2:
            best_thresholds[class_name] = np.nan
            f1_scores_at_best[class_name] = np.nan
            continue
        precision, recall, thresholds_pr = precision_recall_curve(true, probs)
        f1s = 2 * (precision * recall) / (precision + recall + 1e-8)
        if len(f1s) > 1 and len(thresholds_pr) > 0:
            best_idx = np.argmax(f1s[1:])
            best_thresholds[class_name] = thresholds_pr[best_idx]
            f1_scores_at_best[class_name] = f1s[1:][best_idx]
        else:
            best_thresholds[class_name] = 0.5
            f1_scores_at_best[class_name] = f1s[0] if len(f1s) > 0 else 0.0
    plot_f1_vs_threshold(
        y_true,
        y_probs,
        class_names,
        best_thresholds,
        f1_scores_at_best,
        save_dir,
        log_wandb,
        display=display,
    )
    if log_wandb and wandb.run:
        try:
            table_data = [
                [
                    name,
                    best_thresholds.get(name, np.nan),
                    f1_scores_at_best.get(name, np.nan),
                ]
                for name in class_names
            ]
            wandb.log(
                {
                    "Optimal_Thresholds_Max_F1": wandb.Table(
                        columns=[
                            "Class",
                            "Best Threshold (Max F1)",
                            "Best F1",
                        ],
                        data=table_data,
                    )
                }
            )
        except Exception as e:
            print(f"Warn: Log Optimal_Thresholds. Err: {e}")
    print("Optimal Thresholds Found (or N/A if not determinable):")
    for cn in class_names:
        thresh = best_thresholds.get(cn, np.nan)
        f1 = f1_scores_at_best.get(cn, np.nan)
        print(
            f"  {cn}: {thresh if not np.isnan(thresh) else 'N/A':<6} "
            f"(F1={f1 if not np.isnan(f1) else 'N/A'})"
        )
    return best_thresholds, f1_scores_at_best


def calculate_bootstrap_metrics(
    y_true,
    y_probs,
    class_names,
    thresholds_dict,
    n_bootstraps=1000,
    ci_level=95,
):
    """
    Performs bootstrapping to calculate metrics with CIs.
    `thresholds_dict` must map class_name to threshold value.
    """
    print(f"🔄 Calculating Metrics with Bootstrapping (n={n_bootstraps})...")
    print("DEBUG: calculate_bootstrap_metrics received:")
    print(
        f"DEBUG: y_true type: {type(y_true)}, "
        f"y_true shape: {getattr(y_true, 'shape', 'N/A')}"
    )
    print(
        f"DEBUG: y_probs type: {type(y_probs)}, "
        f"y_probs shape: {getattr(y_probs, 'shape', 'N/A')}"
    )
    print(
        f"DEBUG: thresholds_dict type: {type(thresholds_dict)}, "
        f"content: {thresholds_dict}"
    )
    print(
        f"DEBUG: n_bootstraps type: {type(n_bootstraps)}, value: {n_bootstraps}"
    )

    n_samples, n_classes = y_true.shape
    rng = np.random.RandomState(42)

    if not isinstance(thresholds_dict, dict):
        # This was the previous error the user encountered.
        print(
            f"ERROR: thresholds_dict is type {type(thresholds_dict)}, not dict!"
        )
        raise TypeError("thresholds_dict must be a dictionary.")

    final_thresholds_array = np.array(
        [thresholds_dict.get(name, 0.5) for name in class_names]
    )
    valid_thresholds = np.nan_to_num(final_thresholds_array, nan=0.5)
    thresh_disp = {
        k: f"{v:.3f}" for k, v in zip(class_names, valid_thresholds)
    }
    print(f"Using thresholds for bootstrapping: {thresh_disp}")
    print(
        f"DEBUG: valid_thresholds for y_pred_baseline shape: {valid_thresholds.shape}"
    )

    y_pred_baseline = (y_probs >= valid_thresholds).astype(int)
    print(
        f"DEBUG: y_pred_baseline type: {type(y_pred_baseline)}, "
        f"shape: {getattr(y_pred_baseline, 'shape', 'N/A')}"
    )

    bootstrap_metrics = {
        "accuracy": [],
        "macro_precision": [],
        "macro_recall": [],
        "macro_f1": [],
        "micro_precision": [],
        "micro_recall": [],
        "micro_f1": [],
        "weighted_precision": [],
        "weighted_recall": [],
        "weighted_f1": [],
        "weighted_auc": [],
        "macro_auc": [],
    }
    per_class_bootstrap = {
        cls: {
            "precision": [],
            "recall": [],
            "specificity": [],
            "npv": [],
            "f1": [],
            "auc": [],
            "ap": [],
        }
        for cls in class_names
    }

    print(
        f"DEBUG: Starting bootstrap loop. n_samples: {n_samples}, "
        f"type: {type(n_samples)}"
    )
    for i_boot in tqdm(range(n_bootstraps), desc="Bootstrapping"):
        # This is where the user's traceback pointed (their line 425)
        # Original user line: indices = rng.choice(np.arange(n_samples),
        # size=n_samples, replace=True)
        # Changed to use n_samples directly if it's an int, which is
        # equivalent and cleaner.
        if not isinstance(n_samples, int):
            print(
                "CRITICAL DEBUG: n_samples is NOT an int before "
                f"rng.choice! Type: {type(n_samples)}, "
                f"Value: {n_samples}"
            )

        try:
            indices = rng.choice(n_samples, size=n_samples, replace=True)
        except Exception as e:
            print(f"ERROR during rng.choice: {e}")
            print(
                f"DEBUG: n_samples type: {type(n_samples)}, value: {n_samples}"
            )
            raise

        if i_boot == 0:  # Print details for the first bootstrap sample
            print(
                f"DEBUG Boot {i_boot}: indices type: {type(indices)}, shape: {getattr(indices, 'shape', 'N/A')}"
            )
            # print(f"DEBUG Boot {i_boot}: indices (first 10): {indices[:10] if hasattr(indices, '__len__') and len(indices) > 0 else 'N/A'}")
            print(
                f"DEBUG Boot {i_boot}: y_true type: {type(y_true)}, shape: {getattr(y_true, 'shape', 'N/A')}"
            )
            print(
                f"DEBUG Boot {i_boot}: y_pred_baseline type: {type(y_pred_baseline)}, shape: {getattr(y_pred_baseline, 'shape', 'N/A')}"
            )
            print(
                f"DEBUG Boot {i_boot}: y_probs type: {type(y_probs)}, shape: {getattr(y_probs, 'shape', 'N/A')}"
            )

        try:
            y_true_bs = y_true[indices]
            y_pred_bs = y_pred_baseline[indices]
            y_probs_bs = y_probs[indices]
        except TypeError as te:
            print(
                "CRITICAL DEBUG: TypeError during indexing (e.g., y_true[indices])!"
            )
            print(
                f"DEBUG: y_true type: {type(y_true)}, shape: {getattr(y_true, 'shape', 'N/A')}"
            )
            print(
                f"DEBUG: indices type: {type(indices)}, shape: {getattr(indices, 'shape', 'N/A')}"
            )
            # print(f"DEBUG: indices value (first 10): {indices[:10] if hasattr(indices, 'shape') else indices}")
            raise te
        except Exception as e_idx:
            print(f"CRITICAL DEBUG: Exception during indexing: {e_idx}")
            raise e_idx

        bootstrap_metrics["accuracy"].append(
            accuracy_score(y_true_bs, y_pred_bs)
        )
        for avg in ["macro", "micro", "weighted"]:
            for metric_fn, name in [
                (precision_score, "precision"),
                (recall_score, "recall"),
                (f1_score, "f1"),
            ]:
                try:
                    bootstrap_metrics[f"{avg}_{name}"].append(
                        metric_fn(
                            y_true_bs, y_pred_bs, average=avg, zero_division=0
                        )
                    )
                except ValueError:
                    bootstrap_metrics[f"{avg}_{name}"].append(np.nan)
            try:
                metric_key = f"{avg}_auc"
                if (
                    metric_key in bootstrap_metrics
                ):  # Only append if the key exists
                    bootstrap_metrics[metric_key].append(
                        roc_auc_score(
                            y_true_bs,
                            y_probs_bs,
                            average=avg,
                            multi_class="ovr",
                        )
                    )
            except ValueError:
                if (
                    metric_key in bootstrap_metrics
                ):  # Only append if the key exists
                    bootstrap_metrics[metric_key].append(np.nan)
        # Filter out micro_auc as it can be ill-defined or same as accuracy
        if "micro_auc" in bootstrap_metrics:
            bootstrap_metrics.pop("micro_auc")

        for i, cls in enumerate(class_names):
            y_t_c, y_p_c, y_pr_c = (
                y_true_bs[:, i],
                y_pred_bs[:, i],
                y_probs_bs[:, i],
            )
            unq_labels = np.unique(y_t_c)
            has_pos = 1 in unq_labels
            has_neg = 0 in unq_labels

            if has_pos and has_neg:
                per_class_bootstrap[cls]["auc"].append(
                    roc_auc_score(y_t_c, y_pr_c)
                )
                per_class_bootstrap[cls]["ap"].append(
                    average_precision_score(y_t_c, y_pr_c)
                )
                per_class_bootstrap[cls]["f1"].append(
                    f1_score(y_t_c, y_p_c, zero_division=0)
                )
            else:
                for m in ["auc", "ap", "f1"]:
                    per_class_bootstrap[cls][m].append(np.nan)
            if has_pos:
                per_class_bootstrap[cls]["precision"].append(
                    precision_score(y_t_c, y_p_c, zero_division=0)
                )
                per_class_bootstrap[cls]["recall"].append(
                    recall_score(y_t_c, y_p_c, zero_division=0)
                )
            else:
                for m in ["precision", "recall"]:
                    per_class_bootstrap[cls][m].append(np.nan)

            cm = confusion_matrix(y_t_c, y_p_c, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            if has_neg:
                per_class_bootstrap[cls]["specificity"].append(
                    tn / (tn + fp) if (tn + fp) > 0 else np.nan
                )
                per_class_bootstrap[cls]["npv"].append(
                    tn / (tn + fn) if (tn + fn) > 0 else np.nan
                )
            else:
                per_class_bootstrap[cls]["specificity"].append(np.nan)
                per_class_bootstrap[cls]["npv"].append(np.nan)

    alpha = 100 - ci_level
    lower_p, upper_p = alpha / 2.0, 100 - (alpha / 2.0)
    global_summary, per_class_summary = {}, {}
    for metric, values in bootstrap_metrics.items():
        valid = [v for v in values if not np.isnan(v)]
        global_summary[metric] = {
            "mean": np.mean(valid) if valid else np.nan,
            "95% CI": (
                np.percentile(valid, lower_p) if valid else np.nan,
                np.percentile(valid, upper_p) if valid else np.nan,
            ),
        }
    for cls, metrics_dict in per_class_bootstrap.items():
        per_class_summary[cls] = {}
        for metric_name, values in metrics_dict.items():
            valid = [v for v in values if not np.isnan(v)]
            per_class_summary[cls][metric_name] = {
                "mean": np.mean(valid) if valid else np.nan,
                "95% CI": (
                    np.percentile(valid, lower_p) if valid else np.nan,
                    np.percentile(valid, upper_p) if valid else np.nan,
                ),
            }
    return global_summary, per_class_summary


# ==============================================================================
# Main Evaluation Function
# ==============================================================================


def _calculate_micro_average_datasets(dataset_dfs, overall_df):
    """Calculate micro-average metrics across datasets with proper formatting."""
    if not dataset_dfs:
        return pd.DataFrame()

    # Get all numeric columns (excluding 'Class' and 'Dataset')
    numeric_cols = [
        col
        for col in dataset_dfs[0].columns
        if col not in ["Class", "Dataset"]
        and dataset_dfs[0][col].dtype in ["float64", "int64", "object"]
    ]

    # Get unique classes
    classes = dataset_dfs[0]["Class"].unique()

    micro_avg_rows = []
    for class_name in classes:
        row = {"Class": class_name}

        # Get overall prevalence for this class
        overall_class_row = overall_df[overall_df["Class"] == class_name]
        overall_prevalence = (
            overall_class_row["Prevalence"].iloc[0]
            if not overall_class_row.empty
            else "N/A"
        )

        for col in numeric_cols:
            # Special handling for prevalence - use overall value
            if "Prevalence" in col:
                row[col] = overall_prevalence
                continue

            # Extract numeric values from all datasets for this class and column
            values = []
            for df in dataset_dfs:
                class_row = df[df["Class"] == class_name]
                if not class_row.empty:
                    val_str = class_row[col].iloc[0]
                    if isinstance(val_str, str) and val_str != "N/A":
                        # Try to extract numeric value (handle formats like "0.123 (0.100–0.150)")
                        try:
                            if "(" in val_str:
                                # Extract mean value before parentheses
                                numeric_val = float(
                                    val_str.split("(")[0].strip()
                                )
                            elif "%" in val_str:
                                # Handle prevalence format like "12.3% (123/1000)"
                                numeric_val = (
                                    float(val_str.split("%")[0].strip()) / 100
                                )
                            else:
                                numeric_val = float(val_str)
                            values.append(numeric_val)
                        except (ValueError, IndexError):
                            continue

            if values:
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1) if len(values) > 1 else 0

                # Format like the rest of the table: mean (lower, upper)
                lower_bound = mean_val - std_val
                upper_bound = mean_val + std_val
                row[col] = (
                    f"{mean_val:.3f} ({lower_bound:.3f}–{upper_bound:.3f})"
                )
            else:
                row[col] = "N/A"

        micro_avg_rows.append(row)

    return pd.DataFrame(micro_avg_rows)


def _evaluate_single_dataset(
    y_true_subset,
    y_probs_subset,
    class_names,
    final_thresholds_dict,
    n_bootstraps,
    dataset_name="All",
):
    """Helper function to evaluate a single dataset subset."""
    print(f"\n--- Evaluating Dataset: {dataset_name} ---")

    # Apply thresholds
    valid_thresholds_for_pred_dict = {
        name: final_thresholds_dict.get(name, 0.5) for name in class_names
    }
    valid_thresholds_for_pred_dict = {
        k: (0.5 if np.isnan(v) else v)
        for k, v in valid_thresholds_for_pred_dict.items()
    }
    valid_thresholds_array = np.array(
        [valid_thresholds_for_pred_dict[name] for name in class_names]
    )
    y_pred_subset = (y_probs_subset >= valid_thresholds_array).astype(int)

    if n_bootstraps > 0:
        global_summary, per_class_summary = calculate_bootstrap_metrics(
            y_true_subset,
            y_probs_subset,
            class_names,
            valid_thresholds_for_pred_dict,
            n_bootstraps,
        )
        global_metrics_df = format_global_summary(global_summary)
        per_class_metrics_df = format_per_class_summary(
            per_class_summary, y_true=y_true_subset
        )

        return {
            "global_metrics_df": global_metrics_df,
            "per_class_metrics_df": per_class_metrics_df,
            "global_metrics_summary": global_summary,
            "per_class_metrics_summary": per_class_summary,
            "y_pred": y_pred_subset,
        }
    else:
        # Point estimates
        global_point = {
            "accuracy": accuracy_score(y_true_subset, y_pred_subset)
        }
        for avg in ["macro", "micro", "weighted"]:
            for metric_fn, name in [
                (precision_score, "precision"),
                (recall_score, "recall"),
                (f1_score, "f1"),
            ]:
                global_point[f"{avg}_{name}"] = metric_fn(
                    y_true_subset, y_pred_subset, average=avg, zero_division=0
                )
            try:
                global_point[f"{avg}_auc"] = roc_auc_score(
                    y_true_subset,
                    y_probs_subset,
                    average=avg,
                    multi_class="ovr",
                )
            except ValueError:
                global_point[f"{avg}_auc"] = np.nan

        per_class_point = {cls: {} for cls in class_names}
        for i, cls in enumerate(class_names):
            ytc, ypc, yprc = (
                y_true_subset[:, i],
                y_pred_subset[:, i],
                y_probs_subset[:, i],
            )
            unq = np.unique(ytc)
            has_pos = 1 in unq
            has_neg = 0 in unq
            per_class_point[cls]["precision"] = (
                precision_score(ytc, ypc, zero_division=0)
                if has_pos
                else np.nan
            )
            per_class_point[cls]["recall"] = (
                recall_score(ytc, ypc, zero_division=0) if has_pos else np.nan
            )
            per_class_point[cls]["f1"] = (
                f1_score(ytc, ypc, zero_division=0)
                if has_pos and has_neg
                else np.nan
            )
            cm = confusion_matrix(ytc, ypc, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            per_class_point[cls]["specificity"] = (
                tn / (tn + fp) if has_neg and (tn + fp) > 0 else np.nan
            )
            per_class_point[cls]["npv"] = (
                tn / (tn + fn) if has_neg and (tn + fn) > 0 else np.nan
            )
            per_class_point[cls]["auc"] = (
                roc_auc_score(ytc, yprc) if has_pos and has_neg else np.nan
            )
            per_class_point[cls]["ap"] = (
                average_precision_score(ytc, yprc)
                if has_pos and has_neg
                else np.nan
            )

        global_metrics_df = format_global_summary(global_point)
        per_class_metrics_df = format_per_class_summary(
            per_class_point, y_true=y_true_subset
        )
        return {
            "global_metrics_df": global_metrics_df,
            "per_class_metrics_df": per_class_metrics_df,
            "global_metrics_point": global_point,
            "per_class_metrics_point": per_class_point,
            "y_pred": y_pred_subset,
        }


def evaluate_model(
    y_true,
    y_probs,
    class_names,
    threshold=0.5,
    use_optimal_thresholds=True,
    n_bootstraps=1000,
    save_dir=None,
    log_wandb=True,
    display=False,
    target_precisions_for_recall=None,  # e.g. [0.9, 0.95]
    target_recalls_for_precision=None,  # e.g. [0.9, 0.95]
    dataset_values=None,  # New parameter for dataset-specific evaluation
):
    """
    Comprehensive model evaluation with bootstrapping, plots, and PR tradeoffs.
    """
    print("--- Starting Model Evaluation ---")
    n_samples_eval, n_classes_eval = y_true.shape  # Renamed to avoid clash
    if target_precisions_for_recall is None:
        target_precisions_for_recall = [0.85, 0.90, 0.95]
    if target_recalls_for_precision is None:
        target_recalls_for_precision = [0.80, 0.85, 0.90]

    if n_classes_eval != len(class_names):
        raise ValueError(
            "Shape mismatch: y_true/y_probs columns vs class_names len."
        )
    if log_wandb and not wandb.run:
        print("Warn: log_wandb=True but no wandb run. Skipping logs.")
        log_wandb = False

    # Handle dataset-specific evaluation
    dataset_specific = dataset_values is not None
    unique_datasets = None
    dataset_indices = {}

    if dataset_specific:
        dataset_values = np.array(dataset_values)
        if len(dataset_values) != n_samples_eval:
            raise ValueError(
                f"dataset_values length ({len(dataset_values)}) must match "
                f"y_true length ({n_samples_eval})"
            )
        unique_datasets = np.unique(dataset_values)
        print(
            f"Dataset-specific evaluation enabled. Found datasets: {unique_datasets}"
        )

        # Create indices for each dataset
        for dataset in unique_datasets:
            dataset_indices[dataset] = np.where(dataset_values == dataset)[0]
            print(
                f"Dataset '{dataset}': {len(dataset_indices[dataset])} samples"
            )

    results = {}
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    results["cooccurrence_matrix"] = plot_cooccurrence_matrix(
        y_true, class_names, save_dir, log_wandb, display=display
    )

    final_thresholds_dict = {}
    if use_optimal_thresholds:
        print("\nFinding optimal thresholds (max F1)...")
        best_thresholds, _ = find_optimal_thresholds(
            y_true, y_probs, class_names, save_dir, log_wandb, display=display
        )
        final_thresholds_dict = {
            name: best_thresholds.get(name, 0.5) for name in class_names
        }
    else:
        if isinstance(threshold, (int, float)):
            print(f"\nUsing fixed threshold: {threshold}")
            final_thresholds_dict = {
                name: float(threshold) for name in class_names
            }
        elif isinstance(threshold, (list, np.ndarray)):
            if len(threshold) == n_classes_eval:  # Used renamed var
                print("\nUsing provided list/array of thresholds.")
                final_thresholds_dict = dict(
                    zip(class_names, map(float, threshold))
                )
            else:
                raise ValueError(
                    "Fixed threshold list len must match n_classes."
                )
        else:
            raise TypeError("Invalid type for fixed threshold.")

    results["final_thresholds"] = {
        k: (v if not np.isnan(v) else "N/A")
        for k, v in final_thresholds_dict.copy().items()
    }

    # Evaluate overall dataset
    print("\n" + "=" * 50)
    print("OVERALL DATASET EVALUATION")
    print("=" * 50)
    overall_results = _evaluate_single_dataset(
        y_true,
        y_probs,
        class_names,
        final_thresholds_dict,
        n_bootstraps,
        "Overall",
    )

    # Store overall results
    results.update(overall_results)
    y_pred_final = overall_results["y_pred"]

    # Dataset-specific evaluation
    dataset_results = {}
    dataset_metrics_combined = []

    if dataset_specific:
        print("\n" + "=" * 50)
        print("DATASET-SPECIFIC EVALUATION")
        print("=" * 50)

        for dataset in unique_datasets:
            indices = dataset_indices[dataset]
            y_true_subset = y_true[indices]
            y_probs_subset = y_probs[indices]

            # Create dataset-specific save directory
            dataset_save_dir = None
            if save_dir:
                dataset_save_dir = f"{save_dir}/dataset_{dataset}"
                Path(dataset_save_dir).mkdir(parents=True, exist_ok=True)

            # Evaluate this dataset
            dataset_result = _evaluate_single_dataset(
                y_true_subset,
                y_probs_subset,
                class_names,
                final_thresholds_dict,
                n_bootstraps,
                str(dataset),
            )

            dataset_results[dataset] = dataset_result

            # Add dataset column to per-class metrics for combined table
            per_class_df = dataset_result["per_class_metrics_df"].copy()
            per_class_df["Dataset"] = str(dataset)
            dataset_metrics_combined.append(per_class_df)

            # Save dataset-specific results
            if dataset_save_dir:
                dataset_result["global_metrics_df"].to_csv(
                    f"{dataset_save_dir}/global_metrics_summary.csv",
                    index=False,
                    na_rep="N/A",
                )
                dataset_result["per_class_metrics_df"].to_csv(
                    f"{dataset_save_dir}/per_class_metrics_summary.csv",
                    index=False,
                    na_rep="N/A",
                )

        # Create combined dataset metrics table
        overall_per_class = results["per_class_metrics_df"].copy()
        overall_per_class["Dataset"] = "Overall"
        dataset_metrics_combined.insert(0, overall_per_class)

        # Calculate micro-average across datasets (excluding overall)
        if len(dataset_metrics_combined) > 1:  # More than just overall
            micro_avg_per_class = _calculate_micro_average_datasets(
                dataset_metrics_combined[1:], overall_per_class
            )  # Exclude overall
            micro_avg_per_class["Dataset"] = "Micro-Average"
            dataset_metrics_combined.append(
                micro_avg_per_class
            )  # Insert at the end

        combined_dataset_metrics_df = pd.concat(
            dataset_metrics_combined, ignore_index=True
        )

        # Reorder columns to put Dataset first
        cols = combined_dataset_metrics_df.columns.tolist()
        cols = ["Dataset"] + [col for col in cols if col != "Dataset"]
        combined_dataset_metrics_df = combined_dataset_metrics_df[cols]

        results["dataset_results"] = dataset_results
        results["combined_dataset_metrics_df"] = combined_dataset_metrics_df

        if save_dir:
            combined_dataset_metrics_df.to_csv(
                f"{save_dir}/combined_dataset_metrics_summary.csv",
                index=False,
                na_rep="N/A",
            )
            print(
                f"Saved combined dataset metrics: {save_dir}/combined_dataset_metrics_summary.csv"
            )

    print("\n" + "=" * 30 + " METRICS SUMMARY " + "=" * 30)
    metric_type = "(Mean, 95% CI)" if n_bootstraps > 0 else "(Point Estimate)"
    print(f"🔍 Global Metrics {metric_type}:")
    pd.options.display.float_format = "{:.3f}".format
    print(
        overall_results["global_metrics_df"].to_string(
            index=False, na_rep="N/A"
        )
    )
    if save_dir:
        overall_results["global_metrics_df"].to_csv(
            f"{save_dir}/global_metrics_summary.csv", index=False, na_rep="N/A"
        )

    print(f"\n📊 Per-Class Metrics {metric_type}:")
    print(
        overall_results["per_class_metrics_df"].to_string(
            index=False, na_rep="N/A"
        )
    )
    if save_dir:
        overall_results["per_class_metrics_df"].to_csv(
            f"{save_dir}/per_class_metrics_summary.csv",
            index=False,
            na_rep="N/A",
        )
    print("=" * 80)

    print("\n📋 Classification Report (using final thresholds)...")
    target_names_report = (
        [f"{class_names[0]}_neg", f"{class_names[0]}_pos"]
        if n_classes_eval == 1
        else class_names
    )
    try:
        report_str = classification_report(
            y_true,
            y_pred_final,
            target_names=target_names_report,
            zero_division=0,
            digits=3,
        )
        report_df = pd.DataFrame(
            classification_report(
                y_true,
                y_pred_final,
                target_names=target_names_report,
                output_dict=True,
                zero_division=0,
            )
        ).transpose()
        print(report_str)
        results.update(
            {
                "classification_report_str": report_str,
                "classification_report_df": report_df,
            }
        )
        if save_dir:
            with open(f"{save_dir}/classification_report.txt", "w") as f:
                f.write(report_str)
            report_df.to_csv(f"{save_dir}/classification_report.csv")
        if log_wandb and wandb.run:
            wandb.log(
                {
                    "classification_report_table": wandb.Table(
                        dataframe=report_df.round(3)
                        .reset_index()
                        .rename(columns={"index": "Metric"})
                    )
                }
            )
    except Exception as e:
        print(f"Warn: Could not generate/log classification report. Err: {e}")

    print("\nGenerating diagnostic plots...")
    results["roc_aucs"] = plot_roc_curves(
        y_true, y_probs, class_names, save_dir, log_wandb, display=display
    )
    results["avg_precisions"] = plot_pr_curves(
        y_true, y_probs, class_names, save_dir, log_wandb, display=display
    )
    results["raw_cm"], results["norm_cm"] = plot_confusion_matrices(
        y_true,
        y_pred_final,
        class_names,
        thresholds_dict=final_thresholds_dict,
        save_dir=save_dir,
        log_wandb=log_wandb,
        display=display,
    )
    results["threshold_sweep_df"] = plot_threshold_sweep(
        y_true,
        y_probs,
        class_names,
        save_dir=save_dir,
        log_wandb=log_wandb,
        display=display,
    )

    # Generate dataset-specific plots
    if dataset_specific:
        print("\nGenerating dataset-specific plots...")
        for dataset in unique_datasets:
            indices = dataset_indices[dataset]
            y_true_subset = y_true[indices]
            y_probs_subset = y_probs[indices]
            y_pred_subset = dataset_results[dataset]["y_pred"]

            dataset_save_dir = (
                f"{save_dir}/dataset_{dataset}" if save_dir else None
            )

            print(f"\nPlots for dataset: {dataset}")
            plot_roc_curves(
                y_true_subset,
                y_probs_subset,
                class_names,
                dataset_save_dir,
                log_wandb=False,
                display=display,
                dataset_name=dataset,
            )
            plot_pr_curves(
                y_true_subset,
                y_probs_subset,
                class_names,
                dataset_save_dir,
                log_wandb=False,
                display=display,
                dataset_name=dataset,
            )
            plot_confusion_matrices(
                y_true_subset,
                y_pred_subset,
                class_names,
                thresholds_dict=final_thresholds_dict,
                save_dir=dataset_save_dir,
                log_wandb=False,
                display=display,
                dataset_name=dataset,
            )
            plot_threshold_sweep(
                y_true_subset,
                y_probs_subset,
                class_names,
                save_dir=dataset_save_dir,
                log_wandb=False,
                display=display,
            )
            plot_cooccurrence_matrix(
                y_true_subset,
                class_names,
                dataset_save_dir,
                log_wandb=False,
                display=display,
            )

        # Generate combined plots
        print("\nGenerating combined dataset plots...")
        plot_combined_roc_curves(
            dataset_results,
            y_true,
            y_probs,
            class_names,
            unique_datasets,
            dataset_indices,
            save_dir,
            log_wandb=log_wandb,
            display=display,
        )
        plot_combined_pr_curves(
            dataset_results,
            y_true,
            y_probs,
            class_names,
            unique_datasets,
            dataset_indices,
            save_dir,
            log_wandb=log_wandb,
            display=display,
        )
        plot_combined_confusion_matrices(
            dataset_results,
            y_true,
            y_probs,
            class_names,
            unique_datasets,
            dataset_indices,
            final_thresholds_dict,
            save_dir,
            log_wandb=log_wandb,
            display=display,
        )

        # Generate single plots with all datasets
        print("\nGenerating single plots with all datasets...")
        plot_single_roc_curves(
            dataset_results,
            y_true,
            y_probs,
            class_names,
            unique_datasets,
            dataset_indices,
            save_dir,
            log_wandb=log_wandb,
            display=display,
        )
        plot_single_pr_curves(
            dataset_results,
            y_true,
            y_probs,
            class_names,
            unique_datasets,
            dataset_indices,
            save_dir,
            log_wandb=log_wandb,
            display=display,
        )

    # --- PR Tradeoff Analysis ---
    pr_tradeoff_records = []
    print("\n🔍 Analyzing Precision-Recall Tradeoffs...")
    for i, class_name in enumerate(class_names):
        y_t_c, y_pr_c = y_true[:, i], y_probs[:, i]
        if len(np.unique(y_t_c)) < 2:
            continue  # Skip if only one class
        precisions, recalls, pr_thresholds = precision_recall_curve(
            y_t_c, y_pr_c
        )
        # thresholds are for p[1:] and r[1:]
        # For recall at target precision:
        for target_p in target_precisions_for_recall:
            indices = np.where(precisions[1:] >= target_p)[0]
            if len(indices) > 0:
                # Among these, find the one with highest recall
                best_idx_for_target_p = indices[
                    np.argmax(recalls[1:][indices])
                ]
                actual_p = precisions[1:][best_idx_for_target_p]
                found_r = recalls[1:][best_idx_for_target_p]
                thr = pr_thresholds[best_idx_for_target_p]
            else:
                found_r, actual_p, thr = np.nan, np.nan, np.nan
            pr_tradeoff_records.append(
                {
                    "Class": class_name,
                    "Target Metric": "Precision",
                    "Target Value": target_p,
                    "Achieved Metric": "Recall",
                    "Achieved Value": found_r,
                    "Actual Precision": actual_p,
                    "Threshold": thr,
                }
            )
        # For precision at target recall:
        for target_r in target_recalls_for_precision:
            indices = np.where(recalls[1:] >= target_r)[0]
            if len(indices) > 0:
                # Among these, find the one with highest precision
                best_idx_for_target_r = indices[
                    np.argmax(precisions[1:][indices])
                ]
                actual_r = recalls[1:][best_idx_for_target_r]
                found_p = precisions[1:][best_idx_for_target_r]
                thr = pr_thresholds[best_idx_for_target_r]
            else:
                found_p, actual_r, thr = np.nan, np.nan, np.nan
            pr_tradeoff_records.append(
                {
                    "Class": class_name,
                    "Target Metric": "Recall",
                    "Target Value": target_r,
                    "Achieved Metric": "Precision",
                    "Achieved Value": found_p,
                    "Actual Recall": actual_r,
                    "Threshold": thr,
                }
            )
    pr_tradeoffs_df = pd.DataFrame(pr_tradeoff_records)
    results["pr_tradeoffs_df"] = pr_tradeoffs_df
    print("\nPrecision-Recall Tradeoffs:")
    print(
        pr_tradeoffs_df.to_string(
            index=False, na_rep="N/A", float_format="%.3f"
        )
    )
    if save_dir:
        pr_tradeoffs_df.to_csv(
            f"{save_dir}/pr_tradeoffs_summary.csv",
            index=False,
            float_format="%.3f",
            na_rep="N/A",
        )
    if log_wandb and wandb.run:
        try:
            wandb.log(
                {"PR_Tradeoffs_Table": wandb.Table(dataframe=pr_tradeoffs_df)}
            )
        except Exception as e:
            print(f"Warn: Failed log PR_Tradeoffs_Table. Err: {e}")

    # Display dataset-specific summary
    if dataset_specific:
        print("\n" + "=" * 30 + " DATASET SUMMARY " + "=" * 30)
        print(f"📊 Combined Dataset Metrics {metric_type}:")
        print(
            results["combined_dataset_metrics_df"].to_string(
                index=False, na_rep="N/A"
            )
        )
        print("=" * 80)

    if save_dir is not None:
        with open(f"{save_dir}/evaluation_results.pkl", "wb") as f:
            pickle.dump(results, f)
        with open(f"{save_dir}/y_true.pkl", "wb") as f:
            pickle.dump(y_true, f)
        with open(f"{save_dir}/y_probs.pkl", "wb") as f:
            pickle.dump(y_probs, f)
        with open(f"{save_dir}/class_names.txt", "w") as f:
            for idx, name in enumerate(class_names):
                f.write(f"{idx}: {name}\n")

    print("\n--- Evaluation Complete ---")
    return results


def plot_combined_roc_curves(
    dataset_results,
    y_true,
    y_probs,
    class_names,
    unique_datasets,
    dataset_indices,
    save_dir=None,
    log_wandb=False,
    display=False,
):
    """Plots combined ROC curves for all datasets in subplots."""
    print("📈 Plotting Combined ROC Curves...")

    for i, class_name in enumerate(class_names):
        # Include overall in the subplot layout
        n_plots = len(unique_datasets) + 1  # +1 for overall
        
        # Calculate grid dimensions with max 3 plots per row
        max_cols = 3
        n_cols = min(n_plots, max_cols)
        n_rows = (n_plots + max_cols - 1) // max_cols  # Ceiling division
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        
        # Handle different subplot configurations
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.flatten()
        elif n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        # Calculate positioning for Overall plot - place at the end
        total_grid_positions = n_rows * n_cols
        
        # Position Overall at the last position among actual plots
        overall_idx = n_plots - 1

        # Plot overall
        if len(np.unique(y_true[:, i])) >= 2:
            fpr, tpr, _ = roc_curve(y_true[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            axes[overall_idx].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", color="#1f77b4")
            axes[overall_idx].plot([0, 1], [0, 1], "k--", alpha=0.5)
            axes[overall_idx].set_title("Overall")
            axes[overall_idx].set_xlabel("False Positive Rate")
            axes[overall_idx].set_ylabel("True Positive Rate")
            axes[overall_idx].legend(loc="lower right")
            axes[overall_idx].grid(True)
        else:
            axes[overall_idx].text(
                0.5, 0.5, "Insufficient data", ha="center", va="center"
            )
            axes[overall_idx].set_title("Overall")

        # Plot individual datasets
        dataset_idx = 0
        for ax_pos in range(total_grid_positions):
            if ax_pos == overall_idx:
                continue  # Skip overall position
            if dataset_idx >= len(unique_datasets):
                break  # No more datasets to plot
                
            dataset = unique_datasets[dataset_idx]
            indices = dataset_indices[dataset]
            y_true_subset = y_true[indices]
            y_probs_subset = y_probs[indices]

            if len(np.unique(y_true_subset[:, i])) < 2:
                axes[ax_pos].text(
                    0.5, 0.5, "Insufficient data", ha="center", va="center"
                )
                axes[ax_pos].set_title(f"{dataset}")
                dataset_idx += 1
                continue

            fpr, tpr, _ = roc_curve(y_true_subset[:, i], y_probs_subset[:, i])
            roc_auc = auc(fpr, tpr)

            axes[ax_pos].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
            axes[ax_pos].plot([0, 1], [0, 1], "k--", alpha=0.5)
            axes[ax_pos].set_title(f"{dataset}")
            axes[ax_pos].set_xlabel("False Positive Rate")
            axes[ax_pos].set_ylabel("True Positive Rate")
            axes[ax_pos].legend(loc="lower right")
            axes[ax_pos].grid(True)
            
            dataset_idx += 1

        # Hide any unused subplots
        for k in range(n_plots, len(axes)):
            axes[k].set_visible(False)

        fig.suptitle(f"Combined ROC Curves - {class_name}")
        plt.tight_layout()

        filename = (
            f"{save_dir}/combined_roc_curve_{class_name}.png"
            if save_dir
            else None
        )
        _save_or_log_plot(
            fig,
            f"Combined ROC - {class_name}",
            filename,
            log_wandb,
            f"Combined_ROC_{class_name}",
            display=display,
        )


def plot_combined_pr_curves(
    dataset_results,
    y_true,
    y_probs,
    class_names,
    unique_datasets,
    dataset_indices,
    save_dir=None,
    log_wandb=False,
    display=False,
):
    """Plots combined PR curves for all datasets in subplots."""
    print("📈 Plotting Combined PR Curves...")

    for i, class_name in enumerate(class_names):
        # Include overall in the subplot layout
        n_plots = len(unique_datasets) + 1  # +1 for overall
        
        # Calculate grid dimensions with max 3 plots per row
        max_cols = 3
        n_cols = min(n_plots, max_cols)
        n_rows = (n_plots + max_cols - 1) // max_cols  # Ceiling division
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        
        # Handle different subplot configurations
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.flatten()
        elif n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        # Calculate positioning for Overall plot - place at the end
        total_grid_positions = n_rows * n_cols
        
        # Position Overall at the last position among actual plots
        overall_idx = n_plots - 1

        # Plot overall
        if len(np.unique(y_true[:, i])) >= 2:
            precision, recall, _ = precision_recall_curve(
                y_true[:, i], y_probs[:, i]
            )
            ap = average_precision_score(y_true[:, i], y_probs[:, i])

            # Remove the artificial point at (0, 1) that sklearn adds
            if len(precision) > 1 and recall[0] == 0 and precision[0] == 1.0:
                recall = recall[1:]
                precision = precision[1:]

            axes[overall_idx].plot(recall, precision, label=f"AP = {ap:.3f}", color="#1f77b4")
            axes[overall_idx].set_title("Overall")
            axes[overall_idx].set_xlabel("Recall")
            axes[overall_idx].set_ylabel("Precision")
            axes[overall_idx].set_xlim(0, 1.05)
            axes[overall_idx].set_ylim(0, 1.05)
            axes[overall_idx].legend()
            axes[overall_idx].grid(True)
        else:
            axes[overall_idx].text(
                0.5, 0.5, "Insufficient data", ha="center", va="center"
            )
            axes[overall_idx].set_title("Overall")

        # Plot individual datasets
        dataset_idx = 0
        for ax_pos in range(total_grid_positions):
            if ax_pos == overall_idx:
                continue  # Skip overall position
            if dataset_idx >= len(unique_datasets):
                break  # No more datasets to plot
                
            dataset = unique_datasets[dataset_idx]
            indices = dataset_indices[dataset]
            y_true_subset = y_true[indices]
            y_probs_subset = y_probs[indices]

            if len(np.unique(y_true_subset[:, i])) < 2:
                axes[ax_pos].text(
                    0.5, 0.5, "Insufficient data", ha="center", va="center"
                )
                axes[ax_pos].set_title(f"{dataset}")
                dataset_idx += 1
                continue

            precision, recall, _ = precision_recall_curve(
                y_true_subset[:, i], y_probs_subset[:, i]
            )
            ap = average_precision_score(
                y_true_subset[:, i], y_probs_subset[:, i]
            )

            # Remove the artificial point at (0, 1) that sklearn adds
            if len(precision) > 1 and recall[0] == 0 and precision[0] == 1.0:
                recall = recall[1:]
                precision = precision[1:]

            axes[ax_pos].plot(recall, precision, label=f"AP = {ap:.3f}")
            axes[ax_pos].set_title(f"{dataset}")
            axes[ax_pos].set_xlabel("Recall")
            axes[ax_pos].set_ylabel("Precision")
            axes[ax_pos].set_xlim(0, 1.05)
            axes[ax_pos].set_ylim(0, 1.05)
            axes[ax_pos].legend()
            axes[ax_pos].grid(True)
            
            dataset_idx += 1

        # Hide any unused subplots
        for k in range(n_plots, len(axes)):
            axes[k].set_visible(False)

        fig.suptitle(f"Combined PR Curves - {class_name}")
        plt.tight_layout()

        filename = (
            f"{save_dir}/combined_pr_curve_{class_name}.png"
            if save_dir
            else None
        )
        _save_or_log_plot(
            fig,
            f"Combined PR - {class_name}",
            filename,
            log_wandb,
            f"Combined_PR_{class_name}",
            display=display,
        )


def plot_combined_confusion_matrices(
    dataset_results,
    y_true,
    y_probs,
    class_names,
    unique_datasets,
    dataset_indices,
    final_thresholds_dict,
    save_dir=None,
    log_wandb=False,
    display=False,
):
    """Plots combined confusion matrices for all datasets in subplots."""
    print("📊 Plotting Combined Confusion Matrices...")

    for i, class_name in enumerate(class_names):
        thresh = final_thresholds_dict.get(class_name, np.nan)
        thresh_str = (
            f"{thresh:.3f}"
            if isinstance(thresh, (int, float, np.float32, np.float64))
            and not np.isnan(thresh)
            else "N/A"
        )

        # Include overall in the subplot layout
        n_plots = len(unique_datasets) + 1  # +1 for overall
        
        # Calculate grid dimensions with max 3 plots per row
        max_cols = 3
        n_cols = min(n_plots, max_cols)
        n_rows = (n_plots + max_cols - 1) // max_cols  # Ceiling division
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        
        # Handle different subplot configurations
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.flatten()
        elif n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        # Calculate positioning for Overall plot
        total_grid_positions = n_rows * n_cols
        empty_positions = total_grid_positions - n_plots
        
        # Position Overall at the last position among actual plots
        overall_idx = n_plots - 1

        # Plot overall
        valid_thresholds_array = np.array(
            [final_thresholds_dict.get(name, 0.5) for name in class_names]
        )
        y_pred_overall = (y_probs >= valid_thresholds_array).astype(int)

        cm_raw = confusion_matrix(
            y_true[:, i], y_pred_overall[:, i], labels=[0, 1]
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_norm = confusion_matrix(
                y_true[:, i],
                y_pred_overall[:, i],
                labels=[0, 1],
                normalize="true",
            )
            cm_norm = np.nan_to_num(cm_norm)

        # Create combined annotation (raw counts with percentages)
        annot_comb = [["" for _ in range(2)] for _ in range(2)]
        for r_idx in range(2):
            for c_idx in range(2):
                raw_val = cm_raw[r_idx, c_idx]
                norm_val = cm_norm[r_idx, c_idx] * 100
                annot_comb[r_idx][c_idx] = f"$\\mathbf{{{norm_val:.1f}\\%}}$\n({raw_val})"

        sns.heatmap(
            cm_norm,
            annot=annot_comb,
            fmt="s",
            cmap="Blues",
            vmin=0,
            vmax=1,
            ax=axes[overall_idx],
            cbar=False,
            square=True,
            xticklabels=["Neg", "Pos"],
            yticklabels=["Neg", "Pos"],
            annot_kws={"size": 18}  # Increased font size for annotations
        )
        axes[overall_idx].set_title("Overall", fontsize=16)  # Increased to match dataset titles
        axes[overall_idx].set_xlabel("Predicted", fontsize=16)  # Increased label font size
        axes[overall_idx].set_ylabel("True", fontsize=16)  # Increased label font size
        axes[overall_idx].tick_params(axis='both', which='major', labelsize=16)  # Increased tick label font size

        # Plot individual datasets
        dataset_idx = 0
        for ax_pos in range(total_grid_positions):
            if ax_pos == overall_idx:
                continue  # Skip overall position
            if dataset_idx >= len(unique_datasets):
                break  # No more datasets to plot
                
            dataset = unique_datasets[dataset_idx]
            indices = dataset_indices[dataset]
            y_true_subset = y_true[indices]
            y_pred_subset = dataset_results[dataset]["y_pred"]

            cm_raw = confusion_matrix(
                y_true_subset[:, i], y_pred_subset[:, i], labels=[0, 1]
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                cm_norm = confusion_matrix(
                    y_true_subset[:, i],
                    y_pred_subset[:, i],
                    labels=[0, 1],
                    normalize="true",
                )
                cm_norm = np.nan_to_num(cm_norm)

            # Create combined annotation (raw counts with percentages)
            annot_comb = [["" for _ in range(2)] for _ in range(2)]
            for r_idx in range(2):
                for c_idx in range(2):
                    raw_val = cm_raw[r_idx, c_idx]
                    norm_val = cm_norm[r_idx, c_idx] * 100
                    annot_comb[r_idx][c_idx] = f"$\\mathbf{{{norm_val:.1f}\\%}}$\n({raw_val})"

            sns.heatmap(
                cm_norm,
                annot=annot_comb,
                fmt="s",
                cmap="Blues",
                vmin=0,
                vmax=1,
                ax=axes[ax_pos],
                cbar=False,
                square=True,
                xticklabels=["Neg", "Pos"],
                yticklabels=["Neg", "Pos"],
                annot_kws={"size": 18}  # Increased font size for annotations
            )
            axes[ax_pos].set_title(f"{dataset}", fontsize=18)  # Increased title font size
            axes[ax_pos].set_xlabel("Predicted", fontsize=16)  # Increased label font size
            axes[ax_pos].set_ylabel("True", fontsize=16)  # Increased label font size
            axes[ax_pos].tick_params(axis='both', which='major', labelsize=16)  # Increased tick label font size
            
            dataset_idx += 1

        # Hide any unused subplots
        for k in range(n_plots, len(axes)):
            axes[k].set_visible(False)

        fig.suptitle(
            f"Combined Confusion Matrices - {class_name} (thr={thresh_str})",
            fontsize=20  # Increased suptitle font size
        )
        plt.tight_layout()

        filename = (
            f"{save_dir}/combined_confusion_matrix_{class_name}.png"
            if save_dir
            else None
        )
        _save_or_log_plot(
            fig,
            f"Combined CM - {class_name}",
            filename,
            log_wandb,
            f"Combined_CM_{class_name}",
            display=display,
        )


def plot_single_roc_curves(
    dataset_results,
    y_true,
    y_probs,
    class_names,
    unique_datasets,
    dataset_indices,
    save_dir=None,
    log_wandb=False,
    display=False,
):
    """Plots single ROC curves with all datasets in different colors."""
    print("📈 Plotting Single ROC Curves (All Datasets)...")

    # Tableau 10 color palette - professional and visually appealing
    dataset_colors = [
        "#4e79a7",
        "#f28e2c",
        "#e15759",
        "#76b7b2",
        "#59a14f",
        "#edc949",
        "#af7aa1",
        "#ff9d9a",
        "#9c755f",
        "#bab0ab",
    ]
    overall_color = "#1f77b4"  # Changed from black to blue

    for i, class_name in enumerate(class_names):
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot overall first with solid blue line
        if len(np.unique(y_true[:, i])) >= 2:
            fpr, tpr, _ = roc_curve(y_true[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(
                fpr,
                tpr,
                color=overall_color,
                linewidth=3,
                linestyle="-",
                label=f"Overall (AUC = {roc_auc:.3f})",
                alpha=0.9,
            )

        # Plot individual datasets with solid lines, thinner and more transparent
        for j, dataset in enumerate(unique_datasets):
            indices = dataset_indices[dataset]
            y_true_subset = y_true[indices]
            y_probs_subset = y_probs[indices]

            if len(np.unique(y_true_subset[:, i])) >= 2:
                fpr, tpr, _ = roc_curve(
                    y_true_subset[:, i], y_probs_subset[:, i]
                )
                roc_auc = auc(fpr, tpr)
                color = dataset_colors[j % len(dataset_colors)]
                ax.plot(
                    fpr,
                    tpr,
                    color=color,
                    linewidth=1.5,
                    linestyle="-",
                    label=f"{dataset} (AUC = {roc_auc:.3f})",
                    alpha=0.6,
                )

        # Plot diagonal reference line
        ax.plot([0, 1], [0, 1], "k:", alpha=0.5, linewidth=1, label="Random")

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curves - {class_name}")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        filename = (
            f"{save_dir}/single_roc_curve_{class_name}.png"
            if save_dir
            else None
        )
        _save_or_log_plot(
            fig,
            f"Single ROC - {class_name}",
            filename,
            log_wandb,
            f"Single_ROC_{class_name}",
            display=display,
        )


def plot_single_pr_curves(
    dataset_results,
    y_true,
    y_probs,
    class_names,
    unique_datasets,
    dataset_indices,
    save_dir=None,
    log_wandb=False,
    display=False,
):
    """Plots single PR curves with all datasets in different colors."""
    print("📈 Plotting Single PR Curves (All Datasets)...")

    # Tableau 10 color palette - professional and visually appealing
    dataset_colors = [
        "#4e79a7",
        "#f28e2c",
        "#e15759",
        "#76b7b2",
        "#59a14f",
        "#edc949",
        "#af7aa1",
        "#ff9d9a",
        "#9c755f",
        "#bab0ab",
    ]
    overall_color = "#1f77b4"  # Changed from black to blue

    for i, class_name in enumerate(class_names):
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot overall first with solid blue line
        if len(np.unique(y_true[:, i])) >= 2:
            precision, recall, _ = precision_recall_curve(
                y_true[:, i], y_probs[:, i]
            )
            ap = average_precision_score(y_true[:, i], y_probs[:, i])

            # Remove the artificial point at (0, 1) that sklearn adds
            if len(precision) > 1 and recall[0] == 0 and precision[0] == 1.0:
                recall = recall[1:]
                precision = precision[1:]

            ax.plot(
                recall,
                precision,
                color=overall_color,
                linewidth=3,
                linestyle="-",
                label=f"Overall (AP = {ap:.3f})",
                alpha=0.9,
            )

        # Plot individual datasets with solid lines, thinner and more transparent
        for j, dataset in enumerate(unique_datasets):
            indices = dataset_indices[dataset]
            y_true_subset = y_true[indices]
            y_probs_subset = y_probs[indices]

            if len(np.unique(y_true_subset[:, i])) >= 2:
                precision, recall, _ = precision_recall_curve(
                    y_true_subset[:, i], y_probs_subset[:, i]
                )
                ap = average_precision_score(
                    y_true_subset[:, i], y_probs_subset[:, i]
                )

                # Remove the artificial point at (0, 1) that sklearn adds
                if (
                    len(precision) > 1
                    and recall[0] == 0
                    and precision[0] == 1.0
                ):
                    recall = recall[1:]
                    precision = precision[1:]

                color = dataset_colors[j % len(dataset_colors)]
                ax.plot(
                    recall,
                    precision,
                    color=color,
                    linewidth=1.5,
                    linestyle="-",
                    label=f"{dataset} (AP = {ap:.3f})",
                    alpha=0.6,
                )

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall Curves - {class_name}")
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)

        filename = (
            f"{save_dir}/single_pr_curve_{class_name}.png"
            if save_dir
            else None
        )
        _save_or_log_plot(
            fig,
            f"Single PR - {class_name}",
            filename,
            log_wandb,
            f"Single_PR_{class_name}",
            display=display,
        )


# ==============================================================================
# Multi-Model Evaluation Functions
# ==============================================================================


def _create_per_metric_csvs(dataset_comparison_df, save_dir):
    """
    Creates separate CSV files for each metric with models as columns and datasets as rows.

    Parameters:
    -----------
    dataset_comparison_df : pd.DataFrame
        Combined dataset comparison DataFrame
    save_dir : str
        Directory to save the CSV files
    """
    # Get all metric columns (exclude Model, Class, Dataset, Prevalence)
    excluded_cols = ["Model", "Class", "Dataset", "Prevalence"]
    metric_cols = [
        col
        for col in dataset_comparison_df.columns
        if col not in excluded_cols
    ]

    # Get unique classes
    unique_classes = dataset_comparison_df["Class"].unique()

    for metric_name in metric_cols:
        print(f"  Creating CSV for metric: {metric_name}")

        # Create one CSV per metric, with separate sections for each class
        metric_data_all = []

        for class_name in unique_classes:
            class_data = dataset_comparison_df[
                dataset_comparison_df["Class"] == class_name
            ]

            # Create pivot table: datasets as rows, models as columns
            pivot_data = class_data.pivot(
                index="Dataset", columns="Model", values=metric_name
            )

            # Add class information
            pivot_data.index = [
                f"{class_name}_{dataset}" for dataset in pivot_data.index
            ]

            metric_data_all.append(pivot_data)

        # Combine all classes
        if metric_data_all:
            combined_metric_df = pd.concat(metric_data_all, axis=0)

            # Save to CSV
            safe_metric_name = (
                metric_name.replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("/", "_")
            )
            csv_filename = (
                f"{save_dir}/metric_{safe_metric_name}_models_vs_datasets.csv"
            )
            combined_metric_df.to_csv(csv_filename, na_rep="N/A")
            print(f"    Saved: {csv_filename}")


def aggregate_model_results(model_results_dict, class_names, save_dir=None):
    """
    Aggregates evaluation results from multiple models into comparison tables.

    Parameters:
    -----------
    model_results_dict : dict
        Dictionary where keys are model names and values are the results
        dictionaries returned by evaluate_model()
    class_names : list
        List of class names
    save_dir : str, optional
        Directory to save the aggregated results

    Returns:
    --------
    dict containing aggregated DataFrames for comparison
    """
    print("🔄 Aggregating results from multiple models...")

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Aggregate global metrics
    global_comparison_rows = []
    for model_name, results in model_results_dict.items():
        global_df = results["global_metrics_df"]
        for _, row in global_df.iterrows():
            new_row = {"Model": model_name, "Category": row["Category"]}
            # Add all metric columns
            for col in global_df.columns:
                if col != "Category":
                    new_row[col] = row[col]
            global_comparison_rows.append(new_row)

    global_comparison_df = pd.DataFrame(global_comparison_rows)

    # Aggregate per-class metrics
    per_class_comparison_rows = []
    for model_name, results in model_results_dict.items():
        per_class_df = results["per_class_metrics_df"]
        for _, row in per_class_df.iterrows():
            new_row = {"Model": model_name, "Class": row["Class"]}
            # Add all metric columns
            for col in per_class_df.columns:
                if col != "Class":
                    new_row[col] = row[col]
            per_class_comparison_rows.append(new_row)

    per_class_comparison_df = pd.DataFrame(per_class_comparison_rows)

    # Aggregate dataset-specific metrics if available
    dataset_comparison_df = None
    if any(
        "combined_dataset_metrics_df" in results
        for results in model_results_dict.values()
    ):
        dataset_comparison_rows = []
        for model_name, results in model_results_dict.items():
            if "combined_dataset_metrics_df" in results:
                dataset_df = results["combined_dataset_metrics_df"]
                for _, row in dataset_df.iterrows():
                    new_row = {"Model": model_name}
                    # Add all columns from the dataset metrics
                    for col in dataset_df.columns:
                        new_row[col] = row[col]
                    dataset_comparison_rows.append(new_row)

        if dataset_comparison_rows:
            dataset_comparison_df = pd.DataFrame(dataset_comparison_rows)

    # Save aggregated results
    aggregated_results = {
        "global_comparison": global_comparison_df,
        "per_class_comparison": per_class_comparison_df,
    }

    if dataset_comparison_df is not None:
        aggregated_results["dataset_comparison"] = dataset_comparison_df

    if save_dir:
        global_comparison_df.to_csv(
            f"{save_dir}/multi_model_global_comparison.csv",
            index=False,
            na_rep="N/A",
        )
        per_class_comparison_df.to_csv(
            f"{save_dir}/multi_model_per_class_comparison.csv",
            index=False,
            na_rep="N/A",
        )
        if dataset_comparison_df is not None:
            dataset_comparison_df.to_csv(
                f"{save_dir}/multi_model_dataset_comparison.csv",
                index=False,
                na_rep="N/A",
            )

        # Create per-metric CSV files with models as columns and datasets as rows
        if dataset_comparison_df is not None:
            print("Creating per-metric CSV files...")
            _create_per_metric_csvs(dataset_comparison_df, save_dir)

        print(f"Saved aggregated comparison tables to: {save_dir}")

    return aggregated_results


def plot_multi_model_roc_curves(
    model_results_dict,
    class_names,
    save_dir=None,
    log_wandb=False,
    display=False,
):
    """
    Plots ROC curves comparing multiple models for each class in subplots.
    Uses the individual ROC curve plots created during evaluation.

    Parameters:
    -----------
    model_results_dict : dict
        Dictionary where keys are model names and values are evaluation results
    class_names : list
        List of class names
    save_dir : str, optional
        Directory to save plots
    log_wandb : bool
        Whether to log to WandB
    display : bool
        Whether to display plots
    """
    print("📈 Plotting Multi-Model ROC Curves...")

    model_names = list(model_results_dict.keys())
    n_models = len(model_names)

    for class_name in class_names:
        # Create subplots: 1 row x n_models columns
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
        if n_models == 1:
            axes = [axes]

        for j, model_name in enumerate(model_names):
            results = model_results_dict[model_name]
            ax = axes[j]

            # Try to recreate ROC curve from stored data
            roc_plotted = False

            # Check if we have the raw data to recreate ROC curves
            if (
                isinstance(results, dict)
                and "y_true" in results
                and "y_probs" in results
            ):
                try:
                    y_true_model = results["y_true"]
                    y_probs_model = results["y_probs"]
                    class_idx = class_names.index(class_name)

                    if len(np.unique(y_true_model[:, class_idx])) >= 2:
                        fpr, tpr, _ = roc_curve(
                            y_true_model[:, class_idx],
                            y_probs_model[:, class_idx],
                        )
                        roc_auc = auc(fpr, tpr)

                        ax.plot(
                            fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.3f}"
                        )
                        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1)
                        ax.legend()
                        roc_plotted = True
                        roc_plotted = True
                    else:
                        print(
                            f"Warning: Insufficient unique labels for {model_name}-{class_name}"
                        )
                except Exception as e:
                    print(
                        f"Warning: Could not recreate ROC for {model_name}-{class_name}: {e}"
                    )
            else:
                print(
                    f"Warning: No raw data available for {model_name}-{class_name}"
                )
                if isinstance(results, dict) and "y_true" not in results:
                    print("  Missing y_true and y_probs in model results")

            # Fallback: show AUC value if ROC curve couldn't be plotted
            if not roc_plotted:
                if "roc_aucs" in results and class_name in results["roc_aucs"]:
                    auc_value = results["roc_aucs"][class_name]
                    if not np.isnan(auc_value):
                        # Plot diagonal reference line
                        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1)

                        # Add AUC text in the center
                        ax.text(
                            0.5,
                            0.5,
                            f"AUC = {auc_value:.3f}",
                            ha="center",
                            va="center",
                            fontsize=14,
                            bbox=dict(
                                boxstyle="round,pad=0.3",
                                facecolor="lightblue",
                                alpha=0.8,
                            ),
                        )
                    else:
                        ax.text(
                            0.5, 0.5, "No valid AUC", ha="center", va="center"
                        )
                else:
                    ax.text(0.5, 0.5, "No ROC data", ha="center", va="center")

            # Always set the axis limits and labels regardless of whether ROC was plotted
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"{model_name}")
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        fig.suptitle(f"Multi-Model ROC Comparison - {class_name}")
        plt.tight_layout()

        safe_class_name = class_name.replace(" ", "_").replace("/", "_")
        filename = (
            f"{save_dir}/multi_model_roc_{safe_class_name}.png"
            if save_dir
            else None
        )
        _save_or_log_plot(
            fig,
            f"Multi-Model ROC - {class_name}",
            filename,
            log_wandb,
            f"Multi_Model_ROC_{safe_class_name}",
            display=display,
        )


def plot_multi_model_pr_curves(
    model_results_dict,
    class_names,
    save_dir=None,
    log_wandb=False,
    display=False,
):
    """
    Plots PR curves comparing multiple models for each class in subplots.
    Uses the individual PR curve plots created during evaluation.

    Parameters:
    -----------
    model_results_dict : dict
        Dictionary where keys are model names and values are evaluation results
    class_names : list
        List of class names
    save_dir : str, optional
        Directory to save plots
    log_wandb : bool
        Whether to log to WandB
    display : bool
        Whether to display plots
    """
    print("📈 Plotting Multi-Model PR Curves...")

    model_names = list(model_results_dict.keys())
    n_models = len(model_names)

    for class_name in class_names:
        # Create subplots: 1 row x n_models columns
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
        if n_models == 1:
            axes = [axes]

        for j, model_name in enumerate(model_names):
            results = model_results_dict[model_name]
            ax = axes[j]

            # Try to recreate PR curve from stored data
            pr_plotted = False

            # Check if we have the raw data to recreate PR curves
            if (
                isinstance(results, dict)
                and "y_true" in results
                and "y_probs" in results
            ):
                try:
                    y_true_model = results["y_true"]
                    y_probs_model = results["y_probs"]
                    class_idx = class_names.index(class_name)

                    if len(np.unique(y_true_model[:, class_idx])) >= 2:
                        precision, recall, _ = precision_recall_curve(
                            y_true_model[:, class_idx],
                            y_probs_model[:, class_idx],
                        )
                        ap = average_precision_score(
                            y_true_model[:, class_idx],
                            y_probs_model[:, class_idx],
                        )

                        # Remove the artificial point at (0, 1) that sklearn adds
                        if (
                            len(precision) > 1
                            and recall[0] == 0
                            and precision[0] == 1.0
                        ):
                            recall = recall[1:]
                            precision = precision[1:]

                        ax.plot(
                            recall,
                            precision,
                            linewidth=2,
                            label=f"AP = {ap:.3f}",
                        )
                        ax.legend(loc="lower left")
                        pr_plotted = True
                    else:
                        print(
                            f"Warning: Insufficient unique labels for {model_name}-{class_name}"
                        )
                except Exception as e:
                    print(
                        f"Warning: Could not recreate PR for {model_name}-{class_name}: {e}"
                    )
            else:
                print(
                    f"Warning: No raw data available for {model_name}-{class_name}"
                )
                if isinstance(results, dict) and "y_true" not in results:
                    print("  Missing y_true and y_probs in model results")

            # Fallback: show AP value if PR curve couldn't be plotted
            if not pr_plotted:
                if (
                    "avg_precisions" in results
                    and class_name in results["avg_precisions"]
                ):
                    ap_value = results["avg_precisions"][class_name]
                    if not np.isnan(ap_value):
                        # Add AP text in the center
                        ax.text(
                            0.5,
                            0.5,
                            f"AP = {ap_value:.3f}",
                            ha="center",
                            va="center",
                            fontsize=14,
                            bbox=dict(
                                boxstyle="round,pad=0.3",
                                facecolor="lightgreen",
                                alpha=0.8,
                            ),
                        )
                    else:
                        ax.text(
                            0.5, 0.5, "No valid AP", ha="center", va="center"
                        )
                else:
                    ax.text(0.5, 0.5, "No PR data", ha="center", va="center")

            # Always set the axis limits and labels regardless of whether PR was plotted
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title(f"{model_name}")
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"Multi-Model PR Comparison - {class_name}")
        plt.tight_layout()

        safe_class_name = class_name.replace(" ", "_").replace("/", "_")
        filename = (
            f"{save_dir}/multi_model_pr_{safe_class_name}.png"
            if save_dir
            else None
        )
        _save_or_log_plot(
            fig,
            f"Multi-Model PR - {class_name}",
            filename,
            log_wandb,
            f"Multi_Model_PR_{safe_class_name}",
            display=display,
        )


def plot_multi_model_roc_curves_with_datasets(
    model_results_dict,
    class_names,
    save_dir=None,
    log_wandb=False,
    display=False,
):
    """
    Plots ROC curves comparing multiple models for each class in subplots.
    Each subplot shows the overall ROC curve plus individual dataset curves for that model.

    Parameters:
    -----------
    model_results_dict : dict
        Dictionary where keys are model names and values are evaluation results
    class_names : list
        List of class names
    save_dir : str, optional
        Directory to save plots
    log_wandb : bool
        Whether to log to WandB
    display : bool
        Whether to display plots
    """
    print("📈 Plotting Multi-Model ROC Curves with Datasets...")

    model_names = list(model_results_dict.keys())
    n_models = len(model_names)

    # Tableau 10 color palette for datasets
    dataset_colors = [
        "#4e79a7",
        "#f28e2c",
        "#e15759",
        "#76b7b2",
        "#59a14f",
        "#edc949",
        "#af7aa1",
        "#ff9d9a",
        "#9c755f",
        "#bab0ab",
    ]
    overall_color = "#1f77b4"  # Changed from black to blue

    for class_name in class_names:
        # Create subplots: 1 row x n_models columns
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
        if n_models == 1:
            axes = [axes]

        for j, model_name in enumerate(model_names):
            results = model_results_dict[model_name]
            ax = axes[j]

            # Check if we have the raw data and dataset results
            if (
                isinstance(results, dict)
                and "y_true" in results
                and "y_probs" in results
            ):
                try:
                    y_true_model = results["y_true"]
                    y_probs_model = results["y_probs"]
                    class_idx = class_names.index(class_name)

                    # Plot overall ROC curve first (thick, dark line)
                    if len(np.unique(y_true_model[:, class_idx])) >= 2:
                        fpr, tpr, _ = roc_curve(
                            y_true_model[:, class_idx],
                            y_probs_model[:, class_idx],
                        )
                        roc_auc = auc(fpr, tpr)
                        ax.plot(
                            fpr,
                            tpr,
                            color=overall_color,
                            linewidth=3,
                            linestyle="-",
                            label=f"Overall (AUC = {roc_auc:.3f})",
                            alpha=0.9,
                        )

                    # Plot individual dataset ROC curves if available
                    if (
                        "dataset_results" in results
                        and "dataset_values" in results
                    ):
                        dataset_results = results["dataset_results"]
                        dataset_values = results["dataset_values"]
                        unique_datasets = np.unique(dataset_values)

                        for dataset_idx, dataset_name in enumerate(
                            unique_datasets
                        ):
                            # Get indices for this dataset
                            dataset_indices = np.where(
                                dataset_values == dataset_name
                            )[0]
                            y_true_subset = y_true_model[dataset_indices]
                            y_probs_subset = y_probs_model[dataset_indices]

                            if (
                                len(np.unique(y_true_subset[:, class_idx]))
                                >= 2
                            ):
                                fpr, tpr, _ = roc_curve(
                                    y_true_subset[:, class_idx],
                                    y_probs_subset[:, class_idx],
                                )
                                roc_auc = auc(fpr, tpr)
                                color = dataset_colors[
                                    dataset_idx % len(dataset_colors)
                                ]
                                ax.plot(
                                    fpr,
                                    tpr,
                                    color=color,
                                    linewidth=1.5,
                                    linestyle="-",
                                    label=f"{dataset_name} (AUC = {roc_auc:.3f})",
                                    alpha=0.6,
                                )
                    else:
                        # Fallback: Add note that individual datasets are not available
                        ax.text(
                            0.02,
                            0.98,
                            "Individual datasets\nnot available",
                            transform=ax.transAxes,
                            fontsize=8,
                            alpha=0.7,
                            verticalalignment="top",
                            bbox=dict(
                                boxstyle="round,pad=0.3",
                                facecolor="lightyellow",
                                alpha=0.8,
                            ),
                        )

                    # Plot diagonal reference line
                    ax.plot([0, 1], [0, 1], "k:", alpha=0.5, linewidth=1)

                except Exception as e:
                    print(
                        f"Warning: Could not plot ROC for {model_name}-{class_name}: {e}"
                    )
                    ax.text(
                        0.5,
                        0.5,
                        "Error plotting ROC",
                        ha="center",
                        va="center",
                    )
            else:
                # Fallback: show AUC value if available
                if "roc_aucs" in results and class_name in results["roc_aucs"]:
                    auc_value = results["roc_aucs"][class_name]
                    if not np.isnan(auc_value):
                        ax.plot([0, 1], [0, 1], "k:", alpha=0.5, linewidth=1)
                        ax.text(
                            0.5,
                            0.5,
                            f"AUC = {auc_value:.3f}",
                            ha="center",
                            va="center",
                            fontsize=14,
                            bbox=dict(
                                boxstyle="round,pad=0.3",
                                facecolor="lightblue",
                                alpha=0.8,
                            ),
                        )
                    else:
                        ax.text(
                            0.5, 0.5, "No valid AUC", ha="center", va="center"
                        )
                else:
                    ax.text(0.5, 0.5, "No ROC data", ha="center", va="center")

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"{model_name}")
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"Multi-Model ROC Aggregated - {class_name}")
        plt.tight_layout()

        safe_class_name = class_name.replace(" ", "_").replace("/", "_")
        filename = (
            f"{save_dir}/multi_model_roc_with_datasets_{safe_class_name}.png"
            if save_dir
            else None
        )
        _save_or_log_plot(
            fig,
            f"Multi-Model ROC Aggregated - {class_name}",
            filename,
            log_wandb,
            f"Multi_Model_ROC_Aggregated_{safe_class_name}",
            display=display,
        )


def plot_multi_model_pr_curves_with_datasets(
    model_results_dict,
    class_names,
    save_dir=None,
    log_wandb=False,
    display=False,
):
    """
    Plots PR curves comparing multiple models for each class in subplots.
    Each subplot shows the overall PR curve plus individual dataset curves for that model.

    Parameters:
    -----------
    model_results_dict : dict
        Dictionary where keys are model names and values are evaluation results
    class_names : list
        List of class names
    save_dir : str, optional
        Directory to save plots
    log_wandb : bool
        Whether to log to WandB
    display : bool
        Whether to display plots
    """
    print("📈 Plotting Multi-Model PR Curves with Datasets...")

    model_names = list(model_results_dict.keys())
    n_models = len(model_names)

    # Tableau 10 color palette for datasets
    dataset_colors = [
        "#4e79a7",
        "#f28e2c",
        "#e15759",
        "#76b7b2",
        "#59a14f",
        "#edc949",
        "#af7aa1",
        "#ff9d9a",
        "#9c755f",
        "#bab0ab",
    ]
    overall_color = "#1f77b4"  # Changed from black to blue

    for class_name in class_names:
        # Create subplots: 1 row x n_models columns
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
        if n_models == 1:
            axes = [axes]

        for j, model_name in enumerate(model_names):
            results = model_results_dict[model_name]
            ax = axes[j]

            # Check if we have the raw data and dataset results
            if (
                isinstance(results, dict)
                and "y_true" in results
                and "y_probs" in results
            ):
                try:
                    y_true_model = results["y_true"]
                    y_probs_model = results["y_probs"]
                    class_idx = class_names.index(class_name)

                    # Plot overall PR curve first (thick, dark line)
                    if len(np.unique(y_true_model[:, class_idx])) >= 2:
                        precision, recall, _ = precision_recall_curve(
                            y_true_model[:, class_idx],
                            y_probs_model[:, class_idx],
                        )
                        ap = average_precision_score(
                            y_true_model[:, class_idx],
                            y_probs_model[:, class_idx],
                        )

                        # Remove the artificial point at (0, 1) that sklearn adds
                        if (
                            len(precision) > 1
                            and recall[0] == 0
                            and precision[0] == 1.0
                        ):
                            recall = recall[1:]
                            precision = precision[1:]

                        ax.plot(
                            recall,
                            precision,
                            color=overall_color,
                            linewidth=3,
                            linestyle="-",
                            label=f"Overall (AP = {ap:.3f})",
                            alpha=0.9,
                        )

                    # Plot individual dataset PR curves if available
                    if (
                        "dataset_results" in results
                        and "dataset_values" in results
                    ):
                        dataset_results = results["dataset_results"]
                        dataset_values = results["dataset_values"]
                        unique_datasets = np.unique(dataset_values)

                        for dataset_idx, dataset_name in enumerate(
                            unique_datasets
                        ):
                            # Get indices for this dataset
                            dataset_indices = np.where(
                                dataset_values == dataset_name
                            )[0]
                            y_true_subset = y_true_model[dataset_indices]
                            y_probs_subset = y_probs_model[dataset_indices]

                            if (
                                len(np.unique(y_true_subset[:, class_idx]))
                                >= 2
                            ):
                                precision, recall, _ = precision_recall_curve(
                                    y_true_subset[:, class_idx],
                                    y_probs_subset[:, class_idx],
                                )
                                ap = average_precision_score(
                                    y_true_subset[:, class_idx],
                                    y_probs_subset[:, class_idx],
                                )

                                # Remove the artificial point at (0, 1) that sklearn adds
                                if (
                                    len(precision) > 1
                                    and recall[0] == 0
                                    and precision[0] == 1.0
                                ):
                                    recall = recall[1:]
                                    precision = precision[1:]

                                color = dataset_colors[
                                    dataset_idx % len(dataset_colors)
                                ]
                                ax.plot(
                                    recall,
                                    precision,
                                    color=color,
                                    linewidth=1.5,
                                    linestyle="-",
                                    label=f"{dataset_name} (AP = {ap:.3f})",
                                    alpha=0.6,
                                )
                    else:
                        # Fallback: Add note that individual datasets are not available
                        ax.text(
                            0.02,
                            0.98,
                            "Individual datasets\nnot available",
                            transform=ax.transAxes,
                            fontsize=8,
                            alpha=0.7,
                            verticalalignment="top",
                            bbox=dict(
                                boxstyle="round,pad=0.3",
                                facecolor="lightyellow",
                                alpha=0.8,
                            ),
                        )

                except Exception as e:
                    print(
                        f"Warning: Could not plot PR for {model_name}-{class_name}: {e}"
                    )
                    ax.text(
                        0.5, 0.5, "Error plotting PR", ha="center", va="center"
                    )
            else:
                # Fallback: show AP value if available
                if (
                    "avg_precisions" in results
                    and class_name in results["avg_precisions"]
                ):
                    ap_value = results["avg_precisions"][class_name]
                    if not np.isnan(ap_value):
                        ax.text(
                            0.5,
                            0.5,
                            f"AP = {ap_value:.3f}",
                            ha="center",
                            va="center",
                            fontsize=14,
                            bbox=dict(
                                boxstyle="round,pad=0.3",
                                facecolor="lightgreen",
                                alpha=0.8,
                            ),
                        )
                    else:
                        ax.text(
                            0.5, 0.5, "No valid AP", ha="center", va="center"
                        )
                else:
                    ax.text(0.5, 0.5, "No PR data", ha="center", va="center")

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title(f"{model_name}")
            ax.legend(loc="lower left")
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"Multi-Model PR Aggregated - {class_name}")
        plt.tight_layout()

        safe_class_name = class_name.replace(" ", "_").replace("/", "_")
        filename = (
            f"{save_dir}/multi_model_pr_with_datasets_{safe_class_name}.png"
            if save_dir
            else None
        )
        _save_or_log_plot(
            fig,
            f"Multi-Model PR Aggregated - {class_name}",
            filename,
            log_wandb,
            f"Multi_Model_PR_Aggregated_{safe_class_name}",
            display=display,
        )


def plot_multi_model_confusion_matrices(
    model_results_dict,
    class_names,
    save_dir=None,
    log_wandb=False,
    display=False,
):
    """
    Plots confusion matrices comparing multiple models for each class in subplots.
    Creates 1 x num_models layout showing overall confusion matrix for each model.

    Parameters:
    -----------
    model_results_dict : dict
        Dictionary where keys are model names and values are evaluation results
    class_names : list
        List of class names
    save_dir : str, optional
        Directory to save plots
    log_wandb : bool
        Whether to log to WandB
    display : bool
        Whether to display plots
    """
    print("📊 Plotting Multi-Model Confusion Matrices...")

    model_names = list(model_results_dict.keys())
    n_models = len(model_names)

    for class_name in class_names:
        # Create subplots: 1 row x n_models columns
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
        if n_models == 1:
            axes = [axes]

        for j, model_name in enumerate(model_names):
            results = model_results_dict[model_name]
            ax = axes[j]

            # Try to get confusion matrix data
            cm_plotted = False

            if isinstance(results, dict) and "raw_cm" in results:
                try:
                    raw_cm_dict = results["raw_cm"]
                    norm_cm_dict = results["norm_cm"]

                    if (
                        class_name in raw_cm_dict
                        and class_name in norm_cm_dict
                    ):
                        cm_raw = raw_cm_dict[class_name]
                        cm_norm = norm_cm_dict[class_name]

                        # Create combined annotation (raw counts with percentages)
                        annot_comb = [["" for _ in range(2)] for _ in range(2)]
                        for r_idx in range(2):
                            for c_idx in range(2):
                                raw_val = cm_raw[r_idx, c_idx]
                                norm_val = cm_norm[r_idx, c_idx] * 100
                                annot_comb[r_idx][c_idx] = (
                                    f"$\\mathbf{{{norm_val:.1f}\\%}}$\n({raw_val})"
                                )

                        sns.heatmap(
                            cm_norm,
                            annot=annot_comb,
                            fmt="s",
                            cmap="Blues",
                            vmin=0,
                            vmax=1,
                            ax=ax,
                            cbar=False,
                            square=True,
                            xticklabels=["Neg", "Pos"],
                            yticklabels=["Neg", "Pos"],
                            annot_kws={"fontsize": 16},  # Increased font size
                        )
                        cm_plotted = True

                except Exception as e:
                    print(
                        f"Warning: Could not plot CM for {model_name}-{class_name}: {e}"
                    )

            if not cm_plotted:
                ax.text(0.5, 0.5, "No CM data", ha="center", va="center")

            ax.set_title(f"{model_name}", fontsize=14)  # Increased font size
            ax.set_xlabel("Predicted", fontsize=12)  # Increased font size
            ax.set_ylabel("True", fontsize=12)  # Increased font size
            ax.tick_params(axis="both", which="major", labelsize=12)  # Increased font size

        fig.suptitle(f"Multi-Model Confusion Matrices - {class_name}", fontsize=16)  # Increased font size
        plt.tight_layout()

        safe_class_name = class_name.replace(" ", "_").replace("/", "_")
        filename = (
            f"{save_dir}/multi_model_confusion_matrices_{safe_class_name}.png"
            if save_dir
            else None
        )
        _save_or_log_plot(
            fig,
            f"Multi-Model Confusion Matrices - {class_name}",
            filename,
            log_wandb,
            f"Multi_Model_CM_{safe_class_name}",
            display=display,
        )


def plot_multi_model_dataset_confusion_matrices(
    model_results_dict,
    class_names,
    save_dir=None,
    log_wandb=False,
    display=False,
):
    """
    Plots confusion matrices in a num_models x num_datasets grid for each class.
    Each row represents a model, each column represents a dataset.

    Parameters:
    -----------
    model_results_dict : dict
        Dictionary where keys are model names and values are evaluation results
    class_names : list
        List of class names
    save_dir : str, optional
        Directory to save plots
    log_wandb : bool
        Whether to log to WandB
    display : bool
        Whether to display plots
    """
    print("📊 Plotting Multi-Model Dataset Confusion Matrices...")

    model_names = list(model_results_dict.keys())

    # First, determine all unique datasets across all models
    all_datasets = set()
    for results in model_results_dict.values():
        if "dataset_results" in results:
            all_datasets.update(results["dataset_results"].keys())

    if not all_datasets:
        print(
            "Warning: No dataset-specific results found. Skipping dataset confusion matrices."
        )
        return

    all_datasets = sorted(list(all_datasets))
    n_models = len(model_names)
    n_datasets = len(all_datasets) + 1  # +1 for overall

    for class_name in class_names:
        # Create subplots: n_models rows x (n_datasets+1) columns
        fig, axes = plt.subplots(
            n_models,
            n_datasets,
            figsize=(5 * n_datasets, 4 * n_models),
            constrained_layout=True,
        )
        # fig.set_constrained_layout_pads(h_pad=0.01, w_pad=0.02,
        #                                 hspace=0, wspace=0.02)
        if n_models == 1:
            axes = axes.reshape(1, -1)
        if n_datasets == 1:
            axes = axes.reshape(-1, 1)

        for i, model_name in enumerate(model_names):
            results = model_results_dict[model_name]

            # Plot overall confusion matrix (first column)
            ax = axes[i, 0]
            cm_plotted = False

            if isinstance(results, dict) and "raw_cm" in results:
                try:
                    raw_cm_dict = results["raw_cm"]
                    norm_cm_dict = results["norm_cm"]

                    if (
                        class_name in raw_cm_dict
                        and class_name in norm_cm_dict
                    ):
                        cm_raw = raw_cm_dict[class_name]
                        cm_norm = norm_cm_dict[class_name]

                        # Create combined annotation
                        annot_comb = [
                            ["" for _ in range(2)] for _ in range(2)
                        ]
                        for r_idx in range(2):
                            for c_idx in range(2):
                                raw_val = cm_raw[r_idx, c_idx]
                                norm_val = cm_norm[r_idx, c_idx] * 100
                                annot_comb[r_idx][c_idx] = (
                                    f"$\\mathbf{{{norm_val:.1f}\\%}}$\n({raw_val})"
                                )

                        sns.heatmap(
                            cm_norm,
                            annot=annot_comb,
                            fmt="s",
                            cmap="Blues",
                            vmin=0,
                            vmax=1,
                            ax=ax,
                            cbar=False,
                            square=True,
                            xticklabels=["Neg", "Pos"],
                            yticklabels=["Neg", "Pos"],
                            annot_kws={"fontsize": 12},  # Increased font size
                        )
                        cm_plotted = True

                except Exception as e:
                    print(
                        f"Warning: Could not plot overall CM for {model_name}-{class_name}: {e}"
                    )

            if not cm_plotted:
                ax.text(0.5, 0.5, "No CM data", ha="center", va="center")

            # Set title and labels
            if i == 0:  # Only set column title for first row
                ax.set_title("Overall", fontsize=16)  # Increased title font size
            if i == n_models - 1:  # Only set xlabel for last row
                ax.set_xlabel("Predicted", fontsize=12)  # Increased label font size
            ax.set_ylabel(f"{model_name}\nTrue", fontsize=12)  # Increased label font size
            ax.tick_params(axis='both', which='major', labelsize=12)  # Increased tick label font size

            # Plot dataset-specific confusion matrices
            for j, dataset_name in enumerate(all_datasets):
                ax = axes[i, j + 1]  # +1 because overall is at index 0
                cm_plotted = False

                if (
                    "dataset_results" in results
                    and dataset_name in results["dataset_results"]
                ):
                    dataset_result = results["dataset_results"][dataset_name]

                    # Try to get confusion matrix from dataset results
                    # We need to reconstruct it from the stored y_true and y_pred
                    if (
                        "y_true" in results
                        and "y_probs" in results
                        and "dataset_values" in results
                    ):
                        try:
                            y_true_model = results["y_true"]
                            dataset_values = results["dataset_values"]

                            # Get indices for this dataset
                            dataset_indices = np.where(
                                dataset_values == dataset_name
                            )[0]
                            y_true_subset = y_true_model[dataset_indices]

                            # Get y_pred for this dataset
                            y_pred_subset = dataset_result["y_pred"]

                            class_idx = class_names.index(class_name)
                            cm_raw = confusion_matrix(
                                y_true_subset[:, class_idx],
                                y_pred_subset[:, class_idx],
                                labels=[0, 1],
                            )

                            with np.errstate(
                                divide="ignore", invalid="ignore"
                            ):
                                cm_norm = confusion_matrix(
                                    y_true_subset[:, class_idx],
                                    y_pred_subset[:, class_idx],
                                    labels=[0, 1],
                                    normalize="true",
                                )
                                cm_norm = np.nan_to_num(cm_norm)

                            # Create combined annotation
                            annot_comb = [
                                ["" for _ in range(2)] for _ in range(2)
                            ]
                            for r_idx in range(2):
                                for c_idx in range(2):
                                    raw_val = cm_raw[r_idx, c_idx]
                                    norm_val = cm_norm[r_idx, c_idx] * 100
                                    annot_comb[r_idx][c_idx] = (
                                        f"$\\mathbf{{{norm_val:.1f}\\%}}$\n({raw_val})"
                                    )

                            sns.heatmap(
                                cm_norm,
                                annot=annot_comb,
                                fmt="s",
                                cmap="Blues",
                                vmin=0,
                                vmax=1,
                                ax=ax,
                                cbar=False,
                                square=True,
                                xticklabels=["Neg", "Pos"],
                                yticklabels=["Neg", "Pos"],
                                annot_kws={"fontsize": 12},  # Increased font size
                            )
                            cm_plotted = True

                        except Exception as e:
                            print(
                                f"Warning: Could not plot dataset CM for {model_name}-{dataset_name}-{class_name}: {e}"
                            )

                if not cm_plotted:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)

                # Set title and labels
                if i == 0:  # Only set column title for first row
                    ax.set_title(f"{dataset_name}", fontsize=14)  # Increased title font size
                if i == n_models - 1:  # Only set xlabel for last row
                    ax.set_xlabel("Predicted", fontsize=12)  # Increased label font size
                if j == 0:  # Only set ylabel for first column
                    ax.set_ylabel("True", fontsize=12)  # Increased label font size
                ax.tick_params(axis='both', which='major', labelsize=12)  # Increased tick label font size

        fig.suptitle(
            f"Multi-Model Dataset Confusion Matrices - {class_name}",
            fontsize=16,  # Increased suptitle font size
            y=0.89,
        )

        # Adjust subplot parameters to make space for the main title
        # Reduce the top of the subplots to make room for suptitle
        # fig.subplots_adjust(top=0.88)
        # plt.tight_layout()

        safe_class_name = class_name.replace(" ", "_").replace("/", "_")
        filename = (
            f"{save_dir}/multi_model_dataset_confusion_matrices_{safe_class_name}.png"
            if save_dir
            else None
        )
        _save_or_log_plot(
            fig,
            f"Multi-Model Dataset Confusion Matrices - {class_name}",
            filename,
            log_wandb,
            f"Multi_Model_Dataset_CM_{safe_class_name}",
            display=display,
        )


def plot_multi_model_metrics_comparison(
    aggregated_results,
    metric_name,
    save_dir=None,
    log_wandb=False,
    display=False,
):
    """
    Creates bar plots comparing a specific metric across models for each class separately.

    Parameters:
    -----------
    aggregated_results : dict
        Results from aggregate_model_results()
    metric_name : str
        Name of the metric to compare (e.g., 'AUC', 'F1-score', 'Sensitivity (Recall)')
    save_dir : str, optional
        Directory to save plots
    log_wandb : bool
        Whether to log to WandB
    display : bool
        Whether to display plots
    """
    print(f"📊 Plotting Multi-Model Comparison - {metric_name}...")

    per_class_df = aggregated_results["per_class_comparison"]

    if metric_name not in per_class_df.columns:
        print(f"Warning: Metric '{metric_name}' not found in results")
        return

    # Extract numeric values from the metric column (handle CI format)
    def extract_numeric_value(val_str):
        if isinstance(val_str, str) and val_str != "N/A":
            try:
                if "(" in val_str:
                    return float(val_str.split("(")[0].strip())
                else:
                    return float(val_str)
            except:
                return np.nan
        return np.nan

    per_class_df["numeric_metric"] = per_class_df[metric_name].apply(
        extract_numeric_value
    )

    # Get unique classes and models
    unique_classes = per_class_df["Class"].unique()
    unique_models = per_class_df["Model"].unique()

    # Create one plot per class
    for class_name in unique_classes:
        class_data = per_class_df[per_class_df["Class"] == class_name]

        fig, ax = plt.subplots(figsize=(max(10, len(class_data) * 1.5), 6))

        # Create bar plot for this class
        models = class_data["Model"].values
        values = class_data["numeric_metric"].values

        # Filter out NaN values
        valid_mask = ~np.isnan(values)
        models_valid = models[valid_mask]
        values_valid = values[valid_mask]

        if len(values_valid) > 0:
            bars = ax.bar(models_valid, values_valid, alpha=0.8, width=0.5)

            # Add value labels on bars
            for bar, value in zip(bars, values_valid):
                height = bar.get_height()
                # Position text with more space to avoid overlap
                text_y = min(height + 0.03, 0.92)
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    text_y,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        ax.set_title(f"Multi-Model Comparison - {metric_name} - {class_name}")
        ax.set_xlabel("Model")
        ax.set_ylabel(metric_name)
        ax.set_ylim(0, 1.05)  # Add more top margin to prevent text overlap
        ax.grid(True, alpha=0.3, axis="y")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Save plot
        safe_class_name = class_name.replace(" ", "_").replace("/", "_")
        safe_metric_name = (
            metric_name.replace(" ", "_").replace("(", "").replace(")", "")
        )
        filename = (
            f"{save_dir}/multi_model_comparison_{safe_metric_name}_{safe_class_name}.png"
            if save_dir
            else None
        )
        _save_or_log_plot(
            fig,
            f"Multi-Model Comparison - {metric_name} - {class_name}",
            filename,
            log_wandb,
            f"Multi_Model_Comparison_{safe_metric_name}_{safe_class_name}",
            display=display,
        )


def plot_multi_model_comprehensive_comparison(
    aggregated_results,
    comparison_metrics,
    save_dir=None,
    log_wandb=False,
    display=False,
):
    """
    Creates comprehensive plots showing all metrics for each class with different colors per model.

    Parameters:
    -----------
    aggregated_results : dict
        Results from aggregate_model_results()
    comparison_metrics : list
        List of metrics to compare
    save_dir : str, optional
        Directory to save plots
    log_wandb : bool
        Whether to log to WandB
    display : bool
        Whether to display plots
    """
    print("📊 Plotting Multi-Model Comprehensive Comparison...")

    per_class_df = aggregated_results["per_class_comparison"]

    # Extract numeric values for all metrics
    def extract_numeric_value(val_str):
        if isinstance(val_str, str) and val_str != "N/A":
            try:
                if "(" in val_str:
                    return float(val_str.split("(")[0].strip())
                else:
                    return float(val_str)
            except:
                return np.nan
        return np.nan

    # Get unique classes and models
    unique_classes = per_class_df["Class"].unique()
    unique_models = per_class_df["Model"].unique()

    # Tableau 10 color palette for models
    model_colors = [
        "#4e79a7",
        "#f28e2c",
        "#e15759",
        "#76b7b2",
        "#59a14f",
        "#edc949",
        "#af7aa1",
        "#ff9d9a",
        "#9c755f",
        "#bab0ab",
    ]

    # Create one comprehensive plot per class
    for class_name in unique_classes:
        class_data = per_class_df[per_class_df["Class"] == class_name]

        # Filter metrics that exist in the data
        available_metrics = [
            metric
            for metric in comparison_metrics
            if metric in per_class_df.columns
        ]

        if not available_metrics:
            print(f"Warning: No available metrics for class {class_name}")
            continue

        fig, ax = plt.subplots(
            figsize=(max(12, len(available_metrics) * 2.5), 8)
        )

        # Prepare data for plotting
        x_positions = np.arange(len(available_metrics))
        bar_width = 0.6 / len(unique_models)  # More spacing between bars

        for i, model_name in enumerate(unique_models):
            model_data = class_data[class_data["Model"] == model_name]
            if model_data.empty:
                continue

            values = []
            for metric in available_metrics:
                if metric in model_data.columns:
                    val_str = model_data[metric].iloc[0]
                    numeric_val = extract_numeric_value(val_str)
                    values.append(numeric_val)
                else:
                    values.append(np.nan)

            # Plot bars for this model
            x_pos = (
                x_positions
                + i * bar_width
                - (len(unique_models) - 1) * bar_width / 2
            )
            color = model_colors[i % len(model_colors)]

            # Filter out NaN values for plotting
            valid_mask = ~np.isnan(values)
            if np.any(valid_mask):
                bars = ax.bar(
                    x_pos[valid_mask],
                    np.array(values)[valid_mask],
                    bar_width,
                    label=model_name,
                    color=color,
                    alpha=0.8,
                )

                # Add value labels on bars
                for j, (pos, val) in enumerate(
                    zip(x_pos[valid_mask], np.array(values)[valid_mask])
                ):
                    if not np.isnan(val):
                        # Position text with more space to avoid overlap
                        text_y = min(val + 0.04, 0.92)
                        ax.text(
                            pos,
                            text_y,
                            f"{val:.3f}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                            rotation=45,
                        )

        ax.set_title(
            f"Multi-Model Comprehensive Comparison - {class_name}", fontsize=18  # Increased from 14
        )
        ax.set_xlabel("Metrics", fontsize=16)  # Increased from 12
        ax.set_ylabel("Metric Values", fontsize=16)  # Increased from 12
        ax.set_ylim(0, 1.08)  # Reset to original margin
        ax.set_xticks(x_positions)
        ax.set_xticklabels(available_metrics, rotation=45, ha="right", fontsize=14)  # Increased font size
        ax.tick_params(axis='y', labelsize=14)  # Increased y-axis tick label font size
        ax.legend(title="Models", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=14, title_fontsize=16)  # Increased legend font sizes
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()

        # Save plot
        safe_class_name = class_name.replace(" ", "_").replace("/", "_")
        filename = (
            f"{save_dir}/multi_model_comprehensive_{safe_class_name}.png"
            if save_dir
            else None
        )
        _save_or_log_plot(
            fig,
            f"Multi-Model Comprehensive - {class_name}",
            filename,
            log_wandb,
            f"Multi_Model_Comprehensive_{safe_class_name}",
            display=display,
        )


def check_dataset_availability(model_results):
    """
    Check which models have dataset-specific information available.

    Parameters:
    -----------
    model_results : dict
        Dictionary where keys are model names and values are evaluation results

    Returns:
    --------
    dict with availability information
    """
    availability = {}
    for model_name, results in model_results.items():
        has_dataset_values = "dataset_values" in results
        has_dataset_results = "dataset_results" in results
        availability[model_name] = {
            "has_dataset_values": has_dataset_values,
            "has_dataset_results": has_dataset_results,
            "has_both": has_dataset_values and has_dataset_results,
        }

    models_with_datasets = [
        name for name, info in availability.items() if info["has_both"]
    ]
    models_without_datasets = [
        name for name, info in availability.items() if not info["has_both"]
    ]

    print("\n📊 Dataset Availability Check:")
    print(f"✅ Models with dataset information: {models_with_datasets}")
    print(f"❌ Models without dataset information: {models_without_datasets}")

    if models_without_datasets:
        print(
            "\n💡 Note: Models without dataset information will only show overall curves"
        )
        print(
            "   in the 'with datasets' plots. To get individual dataset curves,"
        )
        print(
            "   re-run evaluate_model() or evaluate_multiple_models() with dataset_values parameter."
        )

    return availability


def aggregate_and_compare_models(
    model_results,
    class_names,
    save_dir=None,
    log_wandb=False,
    display=False,
    comparison_metrics=None,
):
    """
    Aggregates pre-computed model results and creates comparison plots.

    This function is useful when you already have individual model evaluation
    results and want to update aggregation/comparison without re-running
    individual evaluations.

    Parameters:
    -----------
    model_results : dict
        Dictionary where keys are model names and values are the results
        dictionaries returned by evaluate_model()
    class_names : list
        List of class names
    save_dir : str, optional
        Directory to save all results
    log_wandb : bool
        Whether to log to WandB
    display : bool
        Whether to display plots
    comparison_metrics : list, optional
        List of metrics to create comparison plots for. If None, will use all
        available metrics from the per-class results.

    Returns:
    --------
    dict containing aggregated comparisons
    """
    print(f"\n{'=' * 50}")
    print("AGGREGATING MULTI-MODEL RESULTS")
    print(f"{'=' * 50}")

    # Check dataset availability
    check_dataset_availability(model_results)

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Aggregate results
    aggregated_results = aggregate_model_results(
        model_results, class_names, save_dir
    )

    # Determine comparison metrics
    if comparison_metrics is None:
        # Auto-detect all available metrics from per-class results
        per_class_df = aggregated_results["per_class_comparison"]
        # Exclude non-metric columns
        excluded_cols = ["Model", "Class", "Prevalence"]
        available_metrics = [
            col for col in per_class_df.columns if col not in excluded_cols
        ]
        comparison_metrics = available_metrics
        print(f"Auto-detected metrics for comparison: {comparison_metrics}")
    else:
        # Use default comprehensive list if not specified
        if comparison_metrics == []:
            comparison_metrics = [
                "AUC",
                "AP",
                "F1-score",
                "Sensitivity (Recall)",
                "PPV (Precision)",
                "Specificity",
                "NPV",
            ]

    # Create comparison plots
    print("\nGenerating multi-model comparison plots...")

    # Plot ROC comparisons
    plot_multi_model_roc_curves(
        model_results, class_names, save_dir, log_wandb, display
    )

    # Plot PR comparisons
    plot_multi_model_pr_curves(
        model_results, class_names, save_dir, log_wandb, display
    )

    # Plot ROC comparisons with individual datasets
    plot_multi_model_roc_curves_with_datasets(
        model_results, class_names, save_dir, log_wandb, display
    )

    # Plot PR comparisons with individual datasets
    plot_multi_model_pr_curves_with_datasets(
        model_results, class_names, save_dir, log_wandb, display
    )

    # Plot metric comparisons
    for metric in comparison_metrics:
        plot_multi_model_metrics_comparison(
            aggregated_results, metric, save_dir, log_wandb, display
        )

    # Plot comprehensive comparison (all metrics per class)
    plot_multi_model_comprehensive_comparison(
        aggregated_results, comparison_metrics, save_dir, log_wandb, display
    )

    # Plot confusion matrix comparisons
    plot_multi_model_confusion_matrices(
        model_results, class_names, save_dir, log_wandb, display
    )

    # Plot dataset confusion matrix comparisons
    plot_multi_model_dataset_confusion_matrices(
        model_results, class_names, save_dir, log_wandb, display
    )

    # Print summary
    print(f"\n{'=' * 50}")
    print("MULTI-MODEL COMPARISON SUMMARY")
    print(f"{'=' * 50}")
    print(f"📊 Aggregated {len(model_results)} models")
    print("📈 Generated ROC and PR comparison plots")
    print("📈 Generated ROC and PR plots with individual datasets")
    print(
        f"📊 Generated comparison plots for {len(comparison_metrics)} metrics"
    )
    print("📊 Generated confusion matrix comparison plots")
    if save_dir:
        print(f"💾 All results saved to: {Path(save_dir).resolve()}")

    return aggregated_results


def evaluate_multiple_models(
    model_evaluation_dict,
    class_names,
    save_dir=None,
    log_wandb=False,
    display=False,
    comparison_metrics=None,
):
    """
    Comprehensive evaluation and comparison of multiple models.

    Parameters:
    -----------
    model_evaluation_dict : dict
        Dictionary where keys are model names and values are dictionaries containing:
        - 'y_true': true labels
        - 'y_probs': predicted probabilities
        - 'dataset_values': dataset labels (optional)
        - 'threshold': threshold values
        - 'use_optimal_thresholds': boolean
        - Any other parameters for evaluate_model()
    class_names : list
        List of class names
    save_dir : str, optional
        Directory to save all results
    log_wandb : bool
        Whether to log to WandB
    display : bool
        Whether to display plots
    comparison_metrics : list, optional
        List of metrics to create comparison plots for. If None, will use all
        available metrics from the per-class results.

    Returns:
    --------
    dict containing individual model results and aggregated comparisons
    """
    print("🚀 Starting Multi-Model Evaluation...")

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Evaluate each model individually
    model_results = {}
    for model_name, eval_params in model_evaluation_dict.items():
        print(f"\n{'=' * 50}")
        print(f"EVALUATING MODEL: {model_name}")
        print(f"{'=' * 50}")

        # Set up individual model save directory
        model_save_dir = f"{save_dir}/{model_name}" if save_dir else None

        # Extract parameters for evaluate_model
        y_true = eval_params["y_true"]
        y_probs = eval_params["y_probs"]

        # Set default parameters
        eval_kwargs = {
            "threshold": eval_params.get("threshold", 0.5),
            "use_optimal_thresholds": eval_params.get(
                "use_optimal_thresholds", True
            ),
            "n_bootstraps": eval_params.get("n_bootstraps", 1000),
            "save_dir": model_save_dir,
            "log_wandb": False,  # Don't log individual models to avoid clutter
            "display": False,  # Don't display individual plots
            "dataset_values": eval_params.get("dataset_values", None),
        }

        # Add any additional parameters
        for key, value in eval_params.items():
            if key not in ["y_true", "y_probs"] and key not in eval_kwargs:
                eval_kwargs[key] = value

        # Evaluate the model
        results = evaluate_model(y_true, y_probs, class_names, **eval_kwargs)

        # Store raw data for multi-model plotting - ensure it's a mutable dict
        if not isinstance(results, dict):
            results = dict(results) if hasattr(results, "items") else {}

        results["y_true"] = y_true.copy()  # Make a copy to ensure it persists
        results["y_probs"] = (
            y_probs.copy()
        )  # Make a copy to ensure it persists

        # Store dataset values if provided
        if eval_kwargs.get("dataset_values") is not None:
            results["dataset_values"] = eval_kwargs["dataset_values"].copy()

        model_results[model_name] = results

        # Verify the data was stored
        print(
            f"✓ Stored raw data for {model_name} (y_true: {results['y_true'].shape}, y_probs: {results['y_probs'].shape})"
        )

    # Use the new aggregation function
    aggregated_results = aggregate_and_compare_models(
        model_results,
        class_names,
        save_dir,
        log_wandb,
        display,
        comparison_metrics,
    )

    return {
        "individual_results": model_results,
        "aggregated_results": aggregated_results,
    }


# Example Usage (placeholder - replace with actual data loading)
if __name__ == "__main__":
    print("--- Running evaluation example ---")

    # --- Configuration ---
    N_SAMPLES = 500
    CLASS_NAMES = ["Cardiomegaly", "Effusion", "No Finding"]
    N_CLASSES = len(CLASS_NAMES)
    SAVE_DIR = "evaluation_output_test"  # Directory to save results
    LOG_WANDB = False  # Set to True to log (requires prior wandb.init)
    USE_OPTIMAL_THRESHOLDS = True
    N_BOOTSTRAPS = 50  # Set to 0 to disable CIs
    DISPLAY_PLOTS = False  # Set to True to display plots interactively
    FIXED_THRESHOLD_INPUT = [0.4, 0.6, 0.5]  # Example list threshold

    # --- Generate Dummy Data ---
    print("Generating dummy data...")
    np.random.seed(42)
    y_true_base = np.random.rand(N_SAMPLES, N_CLASSES) > 0.7
    has_finding = np.any(y_true_base[:, :-1], axis=1)
    y_true_base[has_finding, -1] = 0
    y_true_base[~has_finding, -1] = 1
    y_true = y_true_base.astype(int)
    y_probs = np.clip(
        y_true * np.random.normal(0.75, 0.25, size=(N_SAMPLES, N_CLASSES))
        + (1 - y_true)
        * np.random.normal(0.25, 0.25, size=(N_SAMPLES, N_CLASSES)),
        0.01,
        0.99,
    )

    # Generate dummy dataset values for demonstration
    dataset_values = np.random.choice(
        ["train", "val", "test"], size=N_SAMPLES, p=[0.6, 0.2, 0.2]
    )

    print(f"Dummy data shapes: y_true {y_true.shape}, y_probs {y_probs.shape}")
    print(
        f"Dataset distribution: {pd.Series(dataset_values).value_counts().to_dict()}"
    )
    prevalence = y_true.sum(axis=0) / N_SAMPLES * 100
    print(f"Class prevalence (%):\n{pd.Series(prevalence, index=CLASS_NAMES)}")

    if LOG_WANDB:
        try:
            run = wandb.init(
                project="cxr-evaluation-test",
                job_type="evaluation-example",
                reinit=True,
            )
            print(f"WandB Run URL: {wandb.run.get_url()}")
        except Exception as e:
            print(f"Could not initialize WandB: {e}. Disabling log.")
            LOG_WANDB = False

    eval_threshold_setting = (
        FIXED_THRESHOLD_INPUT if not USE_OPTIMAL_THRESHOLDS else 0.5
    )

    evaluation_results = evaluate_model(
        y_true=y_true,
        y_probs=y_probs,
        class_names=CLASS_NAMES,
        threshold=eval_threshold_setting,
        use_optimal_thresholds=USE_OPTIMAL_THRESHOLDS,
        n_bootstraps=N_BOOTSTRAPS,
        save_dir=SAVE_DIR,
        log_wandb=LOG_WANDB,
        display=DISPLAY_PLOTS,
        target_precisions_for_recall=[0.85, 0.9, 0.95],
        target_recalls_for_precision=[0.8, 0.85, 0.9],
        dataset_values=dataset_values,  # New parameter for dataset-specific evaluation
    )

    print("\nAccessing results dictionary keys:", evaluation_results.keys())

    if LOG_WANDB and wandb.run:
        wandb.finish()
        print("WandB run finished.")

    print(f"\nEvaluation artifacts saved in: {Path(SAVE_DIR).resolve()}")

    # --- Multi-Model Evaluation Example ---
    print("\n" + "=" * 60)
    print("MULTI-MODEL EVALUATION EXAMPLE")
    print("=" * 60)

    # Simulate multiple models with different performance
    model_evaluation_dict = {}

    for model_idx, model_name in enumerate(["Model_A", "Model_B", "Model_C"]):
        # Create slightly different predictions for each model
        noise_factor = 0.1 + model_idx * 0.05
        y_probs_model = np.clip(
            y_probs + np.random.normal(0, noise_factor, y_probs.shape),
            0.01,
            0.99,
        )

        model_evaluation_dict[model_name] = {
            "y_true": y_true,
            "y_probs": y_probs_model,
            "dataset_values": dataset_values,
            "threshold": [0.23, 0.5, 0.5],
            "use_optimal_thresholds": False,
            "n_bootstraps": 20,  # Reduced for demo
        }

    # Run multi-model evaluation with auto-detected metrics
    multi_model_results = evaluate_multiple_models(
        model_evaluation_dict=model_evaluation_dict,
        class_names=CLASS_NAMES,
        save_dir="evaluation_output_multi_model",
        log_wandb=False,
        display=DISPLAY_PLOTS,
        comparison_metrics=None,  # Will auto-detect all available metrics
    )

    print("\nMulti-model evaluation complete!")
    print(
        "Check the 'evaluation_output_multi_model' directory for comparison results."
    )
    print(
        "- Individual metric plots: one plot per class per metric showing models as bars"
    )
    print(
        "- Comprehensive plots: all metrics per class with different colors per model"
    )
    print(
        "- ROC comparison: subplots showing actual ROC curves for each model"
    )
    print("- Per-metric CSVs: models vs datasets matrices for each metric")

    # --- Example: Re-aggregate without re-evaluating ---
    print("\n" + "=" * 60)
    print("EXAMPLE: RE-AGGREGATION WITHOUT RE-EVALUATION")
    print("=" * 60)
    print(
        "Demonstrating how to update comparison plots without re-running evaluations..."
    )

    # Extract just the individual results
    individual_results = multi_model_results["individual_results"]

    # Re-aggregate with different settings (e.g., different save directory)
    updated_aggregation = aggregate_and_compare_models(
        model_results=individual_results,
        class_names=CLASS_NAMES,
        save_dir="evaluation_output_multi_model_updated",
        log_wandb=False,
        display=DISPLAY_PLOTS,
        comparison_metrics=["AUC", "F1-score"],  # Only specific metrics
    )

    print(
        "Re-aggregation complete! This is much faster since we didn't re-evaluate models."
    )
    print("New features demonstrated:")
    print("- Individual metric plots: one plot per class per metric")
    print("- Comprehensive plots: all metrics in one plot per class")
    print("- ROC and PR curves as subplots with actual curve data")
    print("- Confusion matrix comparisons: overall and dataset-specific")
    print("- Per-metric CSV files with models vs datasets")
