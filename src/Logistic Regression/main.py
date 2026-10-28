import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

# Ensure UTF-8 output encoding for Vietnamese characters on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

print("1. Loading datasets from local csv...")
train_df = pd.read_csv(r"../../dataset\vietnamese_sms_dataset\train.csv")
test_df = pd.read_csv(r"../../dataset\vietnamese_sms_dataset\test.csv")

X_train = train_df["message"].values
y_train = train_df["label"].values

X_test = test_df["message"].values
y_test = test_df["label"].values

print("2. Converting text to TF-IDF features...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("3. Training Logistic Regression model...")
model = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
model.fit(X_train_vec, y_train)

print("4. Evaluating model performance & calculating measurements...")
y_pred = model.predict(X_test_vec)
y_scores = model.predict_proba(X_test_vec)[
    :, 1
]  # Probability of being spam (positive class)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# 5 Core Measurements (Accuracy, Precision, Recall, F1-Score, Specificity)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
spec = tn / (tn + fp)
roc_auc = roc_auc_score(y_test, y_scores)

print("\n" + "=" * 55)
print("              5 EVALUATION MEASUREMENTS                ")
print("=" * 55)
print(f" 1. Accuracy            : {acc:.4f} ({acc * 100:.2f}%)")
print(f" 2. Precision           : {prec:.4f} ({prec * 100:.2f}%)")
print(f" 3. Recall (Sensitivity): {rec:.4f} ({rec * 100:.2f}%)")
print(f" 4. F1-Score            : {f1:.4f} ({f1 * 100:.2f}%)")
print(f" 5. Specificity         : {spec:.4f} ({spec * 100:.2f}%)")
print("=" * 55)
print("\nConfusion Matrix Breakdown:")
print(f"  True Negatives (TN) : {tn}")
print(f"  False Positives (FP): {fp}")
print(f"  False Negatives (FN): {fn}")
print(f"  True Positives (TP) : {tp}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

print("5. Exporting model to 'model...")
export_dir = os.path.join("model")
os.makedirs(export_dir, exist_ok=True)

# Export standalone model and vectorizer
model_path = os.path.join(export_dir, "model.joblib")
vectorizer_path = os.path.join(export_dir, "vectorizer.joblib")
joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)

# Also export with explicit descriptive names
joblib.dump(model, os.path.join(export_dir, "logistic_regression_model.joblib"))
joblib.dump(vectorizer, os.path.join(export_dir, "tfidf_vectorizer.joblib"))

# Export end-to-end Pipeline for convenient inference on raw text
pipeline = Pipeline(
    [
        ("vectorizer", vectorizer),
        ("classifier", model),
    ]
)
pipeline_path = os.path.join(export_dir, "pipeline.joblib")
joblib.dump(pipeline, pipeline_path)

print(f"Files saved in '{export_dir}':")
print(f" - {model_path}")
print(f" - {vectorizer_path}")
print(f" - {pipeline_path}")

print("\n6. Generating Matplotlib Report Dashboard...")
plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# --- Plot 1: Confusion Matrix ---
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["Ham (Legitimate)", "Spam / Phishing"]
)
disp.plot(ax=axes[0, 0], cmap=plt.cm.Blues, colorbar=False, values_format="d")
axes[0, 0].set_title(
    f"Confusion Matrix (Total Test Samples: {len(y_test)})",
    fontsize=13,
    fontweight="bold",
    pad=12,
)
axes[0, 0].set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
axes[0, 0].set_ylabel("True Label", fontsize=11, fontweight="bold")
axes[0, 0].grid(False)

# Annotation box for Confusion Matrix details
axes[0, 0].text(
    0.5,
    -0.20,
    f"TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}",
    ha="center",
    va="center",
    transform=axes[0, 0].transAxes,
    fontsize=11,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f4f8", edgecolor="#b0c4de"),
)

# --- Plot 2: 5 Performance Measurements ---
metrics_dict = {
    "Accuracy": acc,
    "Precision": prec,
    "Recall (Sensitivity)": rec,
    "F1-Score": f1,
    "Specificity": spec,
}
names = list(metrics_dict.keys())
values = list(metrics_dict.values())
colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]

bars = axes[0, 1].barh(
    names[::-1],
    values[::-1],
    color=colors[::-1],
    height=0.55,
    edgecolor="#333333",
    linewidth=0.8,
)
axes[0, 1].set_xlim(0, 1.18)
axes[0, 1].set_xlabel("Score (0.0 - 1.0)", fontsize=11, fontweight="bold")
axes[0, 1].set_title(
    "5 Key Performance Measurements", fontsize=13, fontweight="bold", pad=12
)
axes[0, 1].grid(axis="x", linestyle="--", alpha=0.6)

for bar in bars:
    w = bar.get_width()
    axes[0, 1].text(
        w + 0.02,
        bar.get_y() + bar.get_height() / 2,
        f"{w:.4f} ({w * 100:.2f}%)",
        va="center",
        ha="left",
        fontsize=10,
        fontweight="bold",
        color="#222222",
    )

# --- Plot 3: ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, y_scores)
axes[1, 0].plot(
    fpr, tpr, color="#d95f02", lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.4f})"
)
axes[1, 0].plot(
    [0, 1],
    [0, 1],
    color="#1f4e79",
    lw=1.8,
    linestyle="--",
    label="Random Chance (AUC = 0.50)",
)
axes[1, 0].set_xlim([0.0, 1.0])
axes[1, 0].set_ylim([0.0, 1.05])
axes[1, 0].set_xlabel(
    "False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold"
)
axes[1, 0].set_ylabel("True Positive Rate (Recall)", fontsize=11, fontweight="bold")
axes[1, 0].set_title(
    "Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold", pad=12
)
axes[1, 0].legend(loc="lower right", fontsize=11)
axes[1, 0].grid(True, linestyle="--", alpha=0.6)

# --- Plot 4: Precision-Recall Curve ---
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_scores)
ap_score = average_precision_score(y_test, y_scores)
spam_prevalence = np.mean(y_test)
axes[1, 1].plot(
    recall_vals,
    precision_vals,
    color="#2ca02c",
    lw=2.5,
    label=f"PR Curve (AP = {ap_score:.4f})",
)
axes[1, 1].plot(
    [0, 1],
    [spam_prevalence, spam_prevalence],
    color="#7f7f7f",
    lw=1.5,
    linestyle="--",
    label=f"Baseline (Spam Ratio = {spam_prevalence:.2f})",
)
axes[1, 1].set_xlim([0.0, 1.0])
axes[1, 1].set_ylim([0.0, 1.05])
axes[1, 1].set_xlabel("Recall", fontsize=11, fontweight="bold")
axes[1, 1].set_ylabel("Precision", fontsize=11, fontweight="bold")
axes[1, 1].set_title("Precision-Recall Curve", fontsize=13, fontweight="bold", pad=12)
axes[1, 1].legend(loc="lower left", fontsize=11)
axes[1, 1].grid(True, linestyle="--", alpha=0.6)

# Layout adjustments
plt.suptitle(
    "Spam Detection Model Performance Report - Logistic Regression",
    fontsize=16,
    fontweight="bold",
    y=0.99,
)
plt.tight_layout()

# Save the report png to local
report_filename = "model_performance_report.png"
plt.savefig(os.path.join(export_dir, report_filename), dpi=300, bbox_inches="tight")
print(f"Report saved successfully as '{report_filename}' and in '{export_dir}'!")
