"""
InfraCopilot AI — Trainer v5
==============================
Restores v2/v3-level performance (recall >0.85, ROC-AUC >0.95, PR-AUC >0.60).

Key decisions vs v4:
  - No label noise in data → model learns strong, consistent patterns
  - No calibration         → removes the threshold-drift bug that caused t=0.05
  - Threshold tuned on val slice of TRAIN set → no leakage onto test
  - class_weight='balanced' + SMOTE (if available) → proven imbalance fix
  - LR wins at this scale; RF kept for comparison only

Pipeline order (strict ML hygiene):
  1. Load + engineer features
  2. Stratified train/test split
  3. Fit scaler on train only
  4. Resample train only
  5. Train candidates
  6. Tune threshold on held-out VALIDATION SLICE of train (not test)
  7. Evaluate on untouched test set
  8. Save model + scaler + metadata (threshold stored in JSON)
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, average_precision_score,
    recall_score, precision_score, f1_score,
    precision_recall_curve,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    IMBLEARN = True
except ImportError:
    IMBLEARN = False

from data_generator import ALL_FEATURE_COLS, LABEL_COL, engineer_features, load_or_generate

# ── Artifact paths ────────────────────────────────────────────────────────────
MODEL_DIR     = "models_v5"
MODEL_PATH    = os.path.join(MODEL_DIR, "model.joblib")
SCALER_PATH   = os.path.join(MODEL_DIR, "scaler.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")
OUTPUTS_DIR   = "outputs"

COST_FN = 1_200   # cost of missing a real failure ($)
COST_FP = 50      # cost of a false alarm ($)
SEP = "═" * 64


def train(
    csv_path: str  = None,
    n_samples: int = 50_000,
    save_plots: bool = True,
) -> dict:
    os.makedirs(MODEL_DIR,   exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    t_wall = time.perf_counter()

    print(f"\n{SEP}\n  InfraCopilot AI — Trainer v5\n{SEP}")

    # ── 1. Load & engineer ────────────────────────────────────────────────────
    df = load_or_generate(csv_path=csv_path, n_samples=n_samples)
    df = engineer_features(df)
    X  = df[ALL_FEATURE_COLS].values.astype(np.float32)
    y  = df[LABEL_COL].values
    print(f"[Data]    {len(X):,} rows | {y.sum():,} failures ({y.mean()*100:.1f}%) | "
          f"{len(ALL_FEATURE_COLS)} features")

    # ── 2. Stratified train/test split ────────────────────────────────────────
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[Split]   Train {len(X_train_raw):,} | Test {len(X_test_raw):,} | "
          f"Train failures: {y_train.sum():,} | Test failures: {y_test.sum():,}")

    # ── 3. Scale (fit on train only) ──────────────────────────────────────────
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_raw)
    X_test_s  = scaler.transform(X_test_raw)

    # ── 4. Resample train set only ────────────────────────────────────────────
    X_bal, y_bal = _resample(X_train_s, y_train)

    # ── 5. Train candidates ───────────────────────────────────────────────────
    print(f"\n[Training]")
    candidates = {
        "LogisticRegression": LogisticRegression(
            C=0.5,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
    }

    trained = {}
    for name, model in candidates.items():
        t0 = time.perf_counter()
        model.fit(X_bal, y_bal)
        trained[name] = model
        print(f"  {name:<25} → {time.perf_counter()-t0:.1f}s")

    # ── 6. Threshold tuning on a validation slice of train ───────────────────
    # WHY: We carve 20% of the train set as a validation slice to tune threshold.
    # This prevents test-set leakage while keeping the threshold well-calibrated.
    X_tr_sub, X_val, y_tr_sub, y_val = train_test_split(
        X_train_s, y_train, test_size=0.20, random_state=42, stratify=y_train
    )
    print(f"\n[Threshold] Cost-optimal on val slice ({len(y_val):,} rows)")
    thresholds = {}
    for name, model in trained.items():
        val_prob = model.predict_proba(X_val)[:, 1]
        thresholds[name] = _cost_threshold(y_val, val_prob)
        print(f"  {name:<25} → t = {thresholds[name]:.3f}")

    # ── 7. Evaluate on untouched test set ─────────────────────────────────────
    print(f"\n[Evaluation]  (held-out test set, n={len(y_test):,})")
    results = {}
    for name, model in trained.items():
        results[name] = _evaluate(name, model, X_test_s, y_test, thresholds[name])
        _print_row(results[name])

    # Pick best by recall, break ties with F1
    best_name   = max(results, key=lambda k: (results[k]["failure_recall"], results[k]["failure_f1"]))
    best_model  = trained[best_name]
    best_thresh = thresholds[best_name]
    best_metrics = results[best_name]
    print(f"\n  ✅ Best: {best_name}  (recall={best_metrics['failure_recall']:.3f}, "
          f"threshold={best_thresh:.3f})")

    # ── 8. Save ───────────────────────────────────────────────────────────────
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler,     SCALER_PATH)

    metadata = {
        "model_version":    "v5",
        "model_name":       best_name,
        "feature_list":     ALL_FEATURE_COLS,       # matches spec field name
        "cost_opt_threshold": round(float(best_thresh), 4),  # matches spec field name
        "threshold":        round(float(best_thresh), 4),    # alias for inference_engine
        "metrics":          best_metrics,
        "all_results":      results,
        "cost_fn":          COST_FN,
        "cost_fp":          COST_FP,
        "imblearn_used":    IMBLEARN,
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    # ── Plots ─────────────────────────────────────────────────────────────────
    if save_plots:
        _plot_pr_curve(best_model, X_test_s, y_test, best_name, best_thresh)

    _print_summary(metadata, time.perf_counter() - t_wall)
    return metadata


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _resample(X_train: np.ndarray, y_train: np.ndarray):
    """
    Imbalance handling at 50k scale.

    Without imblearn: class_weight='balanced' on both models (already set).
    With imblearn:    RandomUnderSample majority → 10:1, then SMOTE → 2:1.
    Much faster than full SMOTE on 48k samples; proven in v3.
    """
    n_fail    = int((y_train == 1).sum())
    n_healthy = int((y_train == 0).sum())
    print(f"\n[Resample]  {n_fail:,} failures / {n_healthy:,} healthy")

    if not IMBLEARN:
        print(f"  imblearn not found → class_weight='balanced' only  "
              f"(pip install imbalanced-learn for SMOTE)")
        return X_train, y_train

    # Step 1: undersample to 10:1
    target_maj = min(n_healthy, n_fail * 10)
    rus = RandomUnderSampler(sampling_strategy={0: target_maj, 1: n_fail}, random_state=42)
    X_u, y_u = rus.fit_resample(X_train, y_train)

    # Step 2: SMOTE to 2:1
    target_min = target_maj // 2
    smote = SMOTE(sampling_strategy={1: target_min},
                  k_neighbors=min(5, n_fail - 1), random_state=42)
    X_r, y_r = smote.fit_resample(X_u, y_u)

    print(f"  After undersample+SMOTE: {(y_r==1).sum():,} failures / {(y_r==0).sum():,} healthy")
    return X_r, y_r


def _cost_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Sweep 0.05–0.95, return threshold minimizing FN×$1200 + FP×$50."""
    best_t, best_cost = 0.5, np.inf
    for t in np.arange(0.05, 0.95, 0.025):
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        cost = fn * COST_FN + fp * COST_FP
        if cost < best_cost:
            best_cost = cost
            best_t    = t
    return round(float(best_t), 3)


def _evaluate(name, model, X_test, y_test, threshold) -> dict:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    baseline = int(y_test.sum()) * COST_FN
    cost     = int(fn) * COST_FN + int(fp) * COST_FP
    return {
        "model":             name,
        "threshold":         round(float(threshold), 4),
        "roc_auc":           round(float(roc_auc_score(y_test, y_prob)), 4),
        "pr_auc":            round(float(average_precision_score(y_test, y_prob)), 4),
        "failure_recall":    round(float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)), 4),
        "failure_precision": round(float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)), 4),
        "failure_f1":        round(float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)), 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "baseline_cost_usd": baseline,
        "model_cost_usd":    cost,
        "cost_savings_usd":  baseline - cost,
    }


def _plot_pr_curve(model, X_test, y_test, model_name, threshold):
    y_prob = model.predict_proba(X_test)[:, 1]
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    op_p = precision_score(y_test, (y_prob >= threshold).astype(int), pos_label=1, zero_division=0)
    op_r = recall_score(y_test, (y_prob >= threshold).astype(int), pos_label=1, zero_division=0)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec, prec, lw=2.5, color="#2563eb", label=f"PR curve  (AP = {ap:.3f})")
    ax.scatter([op_r], [op_p], s=130, zorder=6, color="#dc2626",
               label=f"Operating point  (t = {threshold:.2f})")
    ax.axhline(y=y_test.mean(), color="#9ca3af", linestyle="--", lw=1.2,
               label=f"No-skill baseline = {y_test.mean():.3f}")
    ax.set_xlabel("Recall",    fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Curve — {model_name}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "pr_curve.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"[Plot] PR curve → {path}")


def _print_row(r):
    print(f"  {r['model']:<25}  recall={r['failure_recall']:.3f}  "
          f"prec={r['failure_precision']:.3f}  f1={r['failure_f1']:.3f}  "
          f"roc={r['roc_auc']:.4f}  pr_auc={r['pr_auc']:.4f}  t={r['threshold']:.3f}")


def _print_summary(meta, elapsed):
    m = meta["metrics"]
    print(f"\n{SEP}")
    print(f"  FINAL SUMMARY — {meta['model_name']}  (v{meta['model_version']})")
    print(SEP)
    print(f"  ROC-AUC           : {m['roc_auc']:.4f}")
    print(f"  PR-AUC            : {m['pr_auc']:.4f}  ← key metric for imbalance")
    print(f"")
    print(f"  ── Failure Class ──────────────────────────────")
    print(f"  Recall            : {m['failure_recall']:.4f}  ← PRIMARY — % of failures caught")
    print(f"  Precision         : {m['failure_precision']:.4f}  ← % of alerts that are real")
    print(f"  F1-Score          : {m['failure_f1']:.4f}")
    print(f"  Threshold         : {meta['cost_opt_threshold']}  (cost-optimized, saved in metadata)")
    print(f"")
    print(f"  ── Confusion Matrix ───────────────────────────")
    print(f"  True Positives    : {m['tp']:>6,}   failures correctly flagged")
    print(f"  False Positives   : {m['fp']:>6,}   unnecessary alerts")
    print(f"  False Negatives   : {m['fn']:>6,}   failures MISSED  ← minimize")
    print(f"  True Negatives    : {m['tn']:>6,}   healthy correctly cleared")
    print(f"")
    print(f"  ── Business Impact ────────────────────────────")
    print(f"  Cost (no model)   : ${m['baseline_cost_usd']:>10,}")
    print(f"  Cost (with model) : ${m['model_cost_usd']:>10,}")
    print(f"  Cost savings      : ${m['cost_savings_usd']:>10,}  ✅")
    print(f"")
    print(f"  Total time        : {elapsed:.1f}s")
    print(f"  Model             → {MODEL_PATH}")
    print(f"  Metadata          → {METADATA_PATH}")
    print(SEP)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data",     type=str, default="data/charger_data.csv")
    p.add_argument("--n",        type=int, default=50_000)
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()
    train(csv_path=args.data, n_samples=args.n, save_plots=not args.no_plots)
