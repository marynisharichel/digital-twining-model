"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        KIDNEY TRANSPLANT DIGITAL TWIN — COMPLETE ML PIPELINE                ║
║        Includes: Preprocessing · Models · Simulators · Graphs               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, roc_auc_score,
                              classification_report, confusion_matrix,
                              roc_curve, precision_recall_curve)
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ═══════════════════════════════════════════════════════════════
# 🎨  GLOBAL PLOT STYLE
# ═══════════════════════════════════════════════════════════════
PALETTE = {
    "bg":      "#0D1117",
    "surface": "#161B22",
    "border":  "#30363D",
    "accent1": "#00D4FF",
    "accent2": "#FF6B6B",
    "accent3": "#7CFC00",
    "accent4": "#FFD700",
    "text":    "#E6EDF3",
    "muted":   "#8B949E",
}

def apply_dark_theme():
    plt.rcParams.update({
        "figure.facecolor":  PALETTE["bg"],
        "axes.facecolor":    PALETTE["surface"],
        "axes.edgecolor":    PALETTE["border"],
        "axes.labelcolor":   PALETTE["text"],
        "axes.titlecolor":   PALETTE["text"],
        "xtick.color":       PALETTE["muted"],
        "ytick.color":       PALETTE["muted"],
        "text.color":        PALETTE["text"],
        "grid.color":        PALETTE["border"],
        "grid.linestyle":    "--",
        "grid.alpha":        0.5,
        "legend.facecolor":  PALETTE["surface"],
        "legend.edgecolor":  PALETTE["border"],
        "figure.dpi":        110,
        "font.family":       "DejaVu Sans",
        "font.size":         11,
    })

apply_dark_theme()


# ═══════════════════════════════════════════════════════════════
# 📂  STEP 1 — LOAD DATA
# ═══════════════════════════════════════════════════════════════
def load_data(filepath=None):
    """
    Load CSV from the provided path. If not found, generate synthetic data.
    UPDATE the CSV_PATH variable below to point to your actual file.
    """
    # ── UPDATE THIS PATH to your actual CSV file ───────────────
    CSV_PATH = r"C:\Users\maryn\Desktop\predictive\Kidney_Organ_SupplyChain_RawDataset.csv"

    # Use provided path or fall back to default
    path_to_try = filepath if filepath is not None else CSV_PATH

    if path_to_try and os.path.exists(path_to_try):
        df = pd.read_csv(path_to_try)
        print(f"✅  Loaded '{path_to_try}'  shape={df.shape}")
        return df
    else:
        if path_to_try:
            print(f"⚠️   '{path_to_try}' not found — generating synthetic data.")
        print("🔬  Generating synthetic kidney supply-chain dataset …")

    # ── Synthetic fallback ──────────────────────────────────────
    n = 500
    rng = np.random.default_rng(42)

    df = pd.DataFrame({
        "age":                   rng.integers(18, 75, n),
        "bp":                    rng.normal(80, 15, n).clip(50, 140),
        "sg":                    rng.choice([1.005,1.010,1.015,1.020,1.025], n),
        "al":                    rng.integers(0, 5, n),
        "su":                    rng.integers(0, 5, n),
        "bgr":                   rng.normal(130, 50, n).clip(50, 400),
        "bu":                    rng.normal(55, 30, n).clip(5, 200),
        "sc":                    rng.normal(2.5, 1.5, n).clip(0.4, 15),
        "sod":                   rng.normal(137, 8, n).clip(100, 160),
        "pot":                   rng.normal(4.5, 1.2, n).clip(2, 8),
        "hemo":                  rng.normal(12, 3, n).clip(3, 18),
        "pcv":                   rng.normal(38, 9, n).clip(9, 55),
        "wbcc":                  rng.normal(8000, 3000, n).clip(2000, 26000),
        "rbcc":                  rng.normal(4.5, 1.2, n).clip(1.5, 8),
        "wait_time_days":        rng.integers(30, 1800, n),
        "donor_age":             rng.integers(5, 70, n),
        "hla_mismatch":          rng.integers(0, 6, n),
        "cold_ischemia_hours":   rng.normal(18, 8, n).clip(1, 48),
        "transport_distance_km": rng.integers(10, 3000, n),
        "pra_percent":           rng.integers(0, 100, n),
        "dialysis_years":        rng.normal(3, 2, n).clip(0, 15),
        "donor_type":            rng.choice(["deceased", "living"], n, p=[0.65, 0.35]),
        "blood_group_match":     rng.choice([0, 1], n, p=[0.25, 0.75]),
        "comorbidities":         rng.integers(0, 5, n),
    })

    score = (
        0.20 * (df["sc"]                    < 3.0).astype(int) +
        0.18 * (df["hemo"]                  > 10).astype(int) +
        0.12 * (df["hla_mismatch"]          < 3).astype(int) +
        0.12 * (df["cold_ischemia_hours"]   < 20).astype(int) +
        0.10 * (df["bp"]                    < 90).astype(int) +
        0.08 * (df["bgr"]                   < 140).astype(int) +
        0.08 * (df["blood_group_match"]     == 1).astype(int) +
        0.06 * (df["donor_type"]            == "living").astype(int) +
        0.06 * (df["pra_percent"]           < 30).astype(int)
    )
    noise = rng.normal(0, 0.08, n)
    df["transplant_success"] = ((score + noise) > 0.42).astype(int)

    print(f"✅  Synthetic data ready  shape={df.shape}")
    print("   Success rate:", df["transplant_success"].mean().round(3))
    return df


# ═══════════════════════════════════════════════════════════════
# 🧹  STEP 2 — PREPROCESSING
# ═══════════════════════════════════════════════════════════════
def preprocess(df: pd.DataFrame):
    df = df.copy()
    df.replace("?", np.nan, inplace=True)

    TARGET_CANDIDATES = [
        "transplant_success", "classification", "ckd", "outcome",
        "result", "label", "target", "status", "diagnosis", "graft_survival"
    ]
    target_col = next(
        (c for c in df.columns if c.strip().lower() in TARGET_CANDIDATES), None
    )
    if target_col is None:
        target_col = min(df.columns, key=lambda c: df[c].nunique())
        print(f"⚠️   No obvious target found — guessing '{target_col}'")
    else:
        print(f"🎯  Target column: '{target_col}'")

    label_maps = {"ckd": 1, "notckd": 0, "success": 1, "failure": 0,
                  "yes": 1, "no": 0, "1": 1, "0": 0}
    if df[target_col].dtype == object:
        df[target_col] = (df[target_col].astype(str).str.strip().str.lower()
                                         .map(label_maps))

    dropped = df[target_col].isnull().sum()
    if dropped:
        print(f"⚠️   Dropping {dropped} rows with unrecognised target values.")
        df = df.dropna(subset=[target_col])

    df[target_col] = df[target_col].astype(int)
    y = df[target_col]

    X = df.drop(columns=[target_col]).copy()
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    X = X.apply(pd.to_numeric, errors="coerce")
    for col in X.columns:
        X[col] = X[col].fillna(0.0) if X[col].isnull().all() else X[col].fillna(X[col].median())

    feature_names = X.columns.tolist()
    print(f"✅  Preprocessing done — {len(feature_names)} features, "
          f"{len(y)} samples, {y.mean():.1%} positive rate")
    return X, y, feature_names, df


# ═══════════════════════════════════════════════════════════════
# 🤖  STEP 3 — TRAIN ENSEMBLE
# ═══════════════════════════════════════════════════════════════
def train_models(X, y):
    min_class = y.value_counts().min()
    stratify  = y if min_class >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print("\n🏋️  Training Random Forest …")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=3,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    print("🏋️  Training Gradient Boosting …")
    gbm = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        subsample=0.8, random_state=42
    )
    gbm.fit(X_train_sc, y_train)

    cv_rf  = cross_val_score(rf,  X_train, y_train, cv=5, scoring="roc_auc")
    cv_gbm = cross_val_score(gbm, X_train_sc, y_train, cv=5, scoring="roc_auc")
    print(f"   RF  CV AUC: {cv_rf.mean():.4f} ± {cv_rf.std():.4f}")
    print(f"   GBM CV AUC: {cv_gbm.mean():.4f} ± {cv_gbm.std():.4f}")

    return {
        "rf": rf, "gbm": gbm, "scaler": scaler,
        "X_train": X_train, "X_test": X_test,
        "X_train_sc": X_train_sc, "X_test_sc": X_test_sc,
        "y_train": y_train, "y_test": y_test,
    }


def make_predictor(models, feature_names):
    rf, gbm, scaler = models["rf"], models["gbm"], models["scaler"]

    def predict_proba(X_input):
        if not isinstance(X_input, pd.DataFrame):
            X_input = pd.DataFrame(X_input, columns=feature_names)
        X_aligned = X_input.reindex(columns=feature_names, fill_value=0)
        X_sc      = scaler.transform(X_aligned)
        p_rf      = rf.predict_proba(X_aligned)[:, 1]
        p_gbm     = gbm.predict_proba(X_sc)[:, 1]
        return 0.6 * p_rf + 0.4 * p_gbm

    return predict_proba


# ═══════════════════════════════════════════════════════════════
# 📊  STEP 4 — EVALUATION DASHBOARD  (2 separate figures)
# ═══════════════════════════════════════════════════════════════
def plot_evaluation_dashboard(models, predict_proba, feature_names):
    X_test, y_test = models["X_test"], models["y_test"]
    y_prob = predict_proba(X_test)
    y_pred = (y_prob > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    print(f"\n📊 Ensemble  Accuracy={acc:.4f}  ROC-AUC={auc:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # ── FIGURE 1: ROC · PR · Confusion Matrix ──────────────────
    fig1, axes1 = plt.subplots(1, 3, figsize=(22, 7))
    fig1.patch.set_facecolor(PALETTE["bg"])
    fig1.suptitle("🏥  Kidney Transplant — Evaluation Dashboard (Part 1)",
                  fontsize=16, fontweight="bold", color=PALETTE["accent1"], y=1.01)
    plt.subplots_adjust(wspace=0.38, left=0.06, right=0.97, top=0.92, bottom=0.13)

    # ROC
    ax = axes1[0]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax.plot(fpr, tpr, color=PALETTE["accent1"], lw=2.5, label=f"Ensemble  AUC = {auc:.3f}")
    ax.plot([0,1],[0,1], color=PALETTE["muted"], lw=1.2, ls="--", label="Random baseline")
    ax.fill_between(fpr, tpr, alpha=0.13, color=PALETTE["accent1"])
    ax.set_title("ROC Curve", fontweight="bold", fontsize=13, pad=14)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_facecolor(PALETTE["surface"])

    # Precision-Recall
    ax = axes1[1]
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ax.plot(rec, prec, color=PALETTE["accent2"], lw=2.5)
    ax.fill_between(rec, prec, alpha=0.13, color=PALETTE["accent2"])
    ax.set_title("Precision–Recall Curve", fontweight="bold", fontsize=13, pad=14)
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor(PALETTE["surface"])

    # Confusion Matrix
    ax = axes1[2]
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                linewidths=1.5, linecolor=PALETTE["bg"],
                annot_kws={"size": 20, "weight": "bold"},
                cbar_kws={"shrink": 0.7, "pad": 0.02})
    ax.set_title("Confusion Matrix", fontweight="bold", fontsize=13, pad=14)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_xticklabels(["Fail", "Success"], fontsize=11)
    ax.set_yticklabels(["Fail", "Success"], fontsize=11, rotation=0)

    plt.savefig("evaluation_dashboard_1.png", dpi=140, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.show()
    print("💾  Saved: evaluation_dashboard_1.png")

    # ── FIGURE 2: Prob dist · Feature Importance · Metrics ─────
    fig2, axes2 = plt.subplots(1, 3, figsize=(24, 8))
    fig2.patch.set_facecolor(PALETTE["bg"])
    fig2.suptitle("🏥  Kidney Transplant — Evaluation Dashboard (Part 2)",
                  fontsize=16, fontweight="bold", color=PALETTE["accent1"], y=1.01)
    plt.subplots_adjust(wspace=0.38, left=0.06, right=0.97, top=0.92, bottom=0.13)

    # Probability distribution
    ax = axes2[0]
    ax.hist(y_prob[y_test == 0], bins=28, alpha=0.72, color=PALETTE["accent2"],
            label="Failure", edgecolor=PALETTE["bg"])
    ax.hist(y_prob[y_test == 1], bins=28, alpha=0.72, color=PALETTE["accent1"],
            label="Success", edgecolor=PALETTE["bg"])
    ax.axvline(0.5, color=PALETTE["accent4"], lw=2, ls="--", label="Threshold = 0.5")
    ax.set_title("Predicted Probability Distribution", fontweight="bold", fontsize=13, pad=14)
    ax.set_xlabel("Predicted Probability", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor(PALETTE["surface"])

    # Feature importance
    ax = axes2[1]
    imp  = models["rf"].feature_importances_
    top  = np.argsort(imp)[-14:]
    cols = [feature_names[i] for i in top]
    vals = imp[top]
    colors = [PALETTE["accent1"] if v > np.median(vals) else PALETTE["accent3"] for v in vals]
    bars = ax.barh(cols, vals, color=colors, edgecolor=PALETTE["bg"], height=0.62)
    ax.set_title("Feature Importance (Random Forest)", fontweight="bold", fontsize=13, pad=14)
    ax.set_xlabel("Importance", fontsize=11)
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_facecolor(PALETTE["surface"])
    for bar, val in zip(bars, vals):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9, color=PALETTE["text"])

    # Metrics bar
    ax = axes2[2]
    metrics = {
        "Accuracy":          acc,
        "ROC-AUC":           auc,
        "Precision\n(Succ)": report.get("1", {}).get("precision", 0),
        "Recall\n(Succ)":    report.get("1", {}).get("recall", 0),
        "F1\n(Succ)":        report.get("1", {}).get("f1-score", 0),
    }
    bar_colors = [PALETTE["accent1"], PALETTE["accent4"],
                  PALETTE["accent2"], PALETTE["accent3"], PALETTE["accent1"]]
    xpos  = np.arange(len(metrics))
    bars2 = ax.bar(xpos, list(metrics.values()), color=bar_colors,
                   edgecolor=PALETTE["bg"], width=0.52)
    ax.set_xticks(xpos)
    ax.set_xticklabels(list(metrics.keys()), fontsize=10)
    ax.set_ylim(0, 1.18)
    ax.set_title("Model Performance Metrics", fontweight="bold", fontsize=13, pad=14)
    ax.set_ylabel("Score", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_facecolor(PALETTE["surface"])
    for bar, val in zip(bars2, metrics.values()):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.025,
                f"{val:.3f}", ha="center", fontsize=11,
                fontweight="bold", color=PALETTE["text"])

    plt.savefig("evaluation_dashboard_2.png", dpi=140, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.show()
    print("💾  Saved: evaluation_dashboard_2.png")


# ═══════════════════════════════════════════════════════════════
# 🔬  STEP 5 — FEATURE IMPACT ANALYSIS
# ═══════════════════════════════════════════════════════════════
def analyze_feature_impact(patient_dict, predict_proba, feature_names,
                            X_train, top_n=14):
    patient_df = pd.DataFrame([patient_dict]).reindex(columns=feature_names, fill_value=0)
    base_prob  = predict_proba(patient_df)[0]

    impacts = {}
    for feat in feature_names:
        mean_val        = X_train[feat].mean() if feat in X_train.columns else 0.0
        perturbed       = patient_df.copy()
        perturbed[feat] = mean_val
        impacts[feat]   = base_prob - predict_proba(perturbed)[0]

    sorted_impacts = sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]

    fig, ax = plt.subplots(figsize=(13, 8))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["surface"])

    feats  = [f for f, _ in sorted_impacts]
    vals   = [v for _, v in sorted_impacts]
    colors = [PALETTE["accent1"] if v > 0 else PALETTE["accent2"] for v in vals]

    bars = ax.barh(feats, vals, color=colors, edgecolor=PALETTE["bg"], height=0.62)
    ax.axvline(0, color=PALETTE["text"], lw=1.2)
    ax.set_title(f"🔬  Feature Impact Analysis   (Base Probability = {base_prob:.3f})",
                 fontweight="bold", fontsize=14, pad=16, color=PALETTE["accent1"])
    ax.set_xlabel("Impact on Transplant Success Probability", fontsize=12)
    ax.grid(True, alpha=0.3, axis="x")

    for bar, val in zip(bars, vals):
        xpos = val + (0.003 if val >= 0 else -0.003)
        ha   = "left" if val >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height()/2,
                f"{val:+.4f}", va="center", ha=ha, fontsize=9.5, color=PALETTE["text"])

    legend_patches = [
        mpatches.Patch(color=PALETTE["accent1"], label="↑ Helps outcome"),
        mpatches.Patch(color=PALETTE["accent2"], label="↓ Hurts outcome"),
    ]
    ax.legend(handles=legend_patches, fontsize=10, loc="lower right")
    plt.tight_layout(pad=2.0)
    plt.savefig("feature_impact.png", dpi=140, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.show()
    print("💾  Saved: feature_impact.png")

    print(f"\n🔬 Feature Impact  (Base Prob: {base_prob:.3f})")
    for feat, impact in sorted_impacts:
        print(f"  {feat:<28} {impact:+.4f}  {'↑ Helps' if impact > 0 else '↓ Hurts'}")

    return dict(sorted_impacts), base_prob


# ═══════════════════════════════════════════════════════════════
# 🔄  STEP 6 — COUNTERFACTUAL GENERATOR
# ═══════════════════════════════════════════════════════════════
def generate_counterfactual(patient_dict, predict_proba, feature_names,
                             target_prob=0.70, max_iterations=300):
    patient_df = pd.DataFrame([patient_dict]).reindex(columns=feature_names, fill_value=0)
    best       = patient_df.copy()
    best_prob  = predict_proba(best)[0]

    modifiable = {
        "sc":                  ("decrease", 0.10),
        "bp":                  ("decrease", 2.0),
        "bgr":                 ("decrease", 5.0),
        "hemo":                ("increase", 0.30),
        "hla_mismatch":        ("decrease", 1.0),
        "cold_ischemia_hours": ("decrease", 1.0),
        "pra_percent":         ("decrease", 2.0),
        "al":                  ("decrease", 0.5),
        "bu":                  ("decrease", 2.0),
    }
    modifiable = {k: v for k, v in modifiable.items() if k in feature_names}

    trajectory = [best_prob]
    if not modifiable:
        print("⚠️  No modifiable features found.")
        return best.iloc[0].to_dict(), [best_prob]

    for _ in range(max_iterations):
        if best_prob >= target_prob:
            break
        improved = False
        for feat, (direction, step) in modifiable.items():
            trial       = best.copy()
            val         = trial[feat].values[0]
            trial[feat] = max(0, val - step) if direction == "decrease" else val + step
            prob        = predict_proba(trial)[0]
            if prob > best_prob:
                best, best_prob = trial, prob
                trajectory.append(best_prob)
                improved = True
                break
        if not improved:
            break

    # ── Two separate figures ────────────────────────────────────
    # Left: trajectory
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    fig1.patch.set_facecolor(PALETTE["bg"])
    ax1.set_facecolor(PALETTE["surface"])
    ax1.plot(trajectory, color=PALETTE["accent3"], lw=2.5, marker="o",
             markersize=4, markerfacecolor=PALETTE["accent4"])
    ax1.axhline(target_prob, color=PALETTE["accent2"], lw=1.8, ls="--",
                label=f"Target = {target_prob}")
    ax1.fill_between(range(len(trajectory)), trajectory, alpha=0.15, color=PALETTE["accent3"])
    ax1.set_title("🔄  Counterfactual Optimisation Trajectory",
                  fontweight="bold", fontsize=14, pad=14, color=PALETTE["accent1"])
    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel("Transplant Success Probability", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout(pad=2.0)
    plt.savefig("counterfactual_trajectory.png", dpi=140, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.show()
    print("💾  Saved: counterfactual_trajectory.png")

    # Right: before/after
    original_df   = pd.DataFrame([patient_dict]).reindex(columns=feature_names, fill_value=0)
    changed_feats = [f for f in modifiable
                     if abs(best[f].values[0] - original_df[f].values[0]) > 1e-6][:10]

    if changed_feats:
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        fig2.patch.set_facecolor(PALETTE["bg"])
        ax2.set_facecolor(PALETTE["surface"])
        orig_vals  = [original_df[f].values[0] for f in changed_feats]
        after_vals = [best[f].values[0]         for f in changed_feats]
        x_idx      = np.arange(len(changed_feats))
        ax2.barh(x_idx - 0.22, orig_vals,  0.40, label="Before",
                 color=PALETTE["accent2"], edgecolor=PALETTE["bg"])
        ax2.barh(x_idx + 0.22, after_vals, 0.40, label="After",
                 color=PALETTE["accent3"], edgecolor=PALETTE["bg"])
        ax2.set_yticks(x_idx)
        ax2.set_yticklabels(changed_feats, fontsize=11)
        ax2.set_title("🔄  Feature Changes — Before vs After",
                      fontweight="bold", fontsize=14, pad=14, color=PALETTE["accent1"])
        ax2.set_xlabel("Feature Value", fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis="x")
        plt.tight_layout(pad=2.0)
        plt.savefig("counterfactual_changes.png", dpi=140, bbox_inches="tight",
                    facecolor=PALETTE["bg"])
        plt.show()
        print("💾  Saved: counterfactual_changes.png")

    print(f"\n🎯 Counterfactual: {trajectory[0]:.3f} → {best_prob:.3f}  "
          f"(target={target_prob}, iterations={len(trajectory)-1})")
    return best.iloc[0].to_dict(), trajectory


# ═══════════════════════════════════════════════════════════════
# ⏱️  STEP 7 — DETERIORATION SIMULATOR
# ═══════════════════════════════════════════════════════════════
def simulate_deterioration(patient_dict, predict_proba, feature_names,
                            days=180, intervention_day=None,
                            intervention_changes=None, scenarios=None):
    default_drift = {
        "sc":   +0.004,
        "bp":   +0.06,
        "hemo": -0.008,
        "bgr":  +0.10,
        "bu":   +0.05,
    }

    def _run_scenario(pd_dict, drift, int_day, changes, label):
        patient = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                   for k, v in pd_dict.items()}
        traj = []
        for day in range(days):
            if int_day is not None and day == int_day and changes:
                for k, v in changes.items():
                    patient[k] = v
                print(f"   💉 Intervention applied day {day} for '{label}'")
            for feat, rate in drift.items():
                if feat in patient and isinstance(patient[feat], float):
                    patient[feat] = max(0.0, patient[feat] + rate
                                        + np.random.normal(0, abs(rate) * 0.4))
            df_tmp = pd.DataFrame([patient]).reindex(columns=feature_names, fill_value=0)
            traj.append(predict_proba(df_tmp)[0])
        return traj

    if scenarios is None:
        scenarios = [
            {"name": "Natural Decline",   "drift": default_drift,
             "color": PALETTE["accent2"], "intervention": False},
            {"name": "With Intervention", "drift": default_drift,
             "color": PALETTE["accent3"], "intervention": True},
        ]

    day_axis = np.arange(days)
    all_traj = {}
    for sc in scenarios:
        use_int = sc.get("intervention", False)
        all_traj[sc["name"]] = _run_scenario(
            patient_dict, sc["drift"],
            intervention_day if use_int else None,
            intervention_changes if use_int else None,
            sc["name"],
        )

    # ── Figure A: Trajectory comparison ─────────────────────────
    fig_a, ax_a = plt.subplots(figsize=(14, 6))
    fig_a.patch.set_facecolor(PALETTE["bg"])
    ax_a.set_facecolor(PALETTE["surface"])
    for sc in scenarios:
        traj = all_traj[sc["name"]]
        ax_a.plot(day_axis, traj, color=sc["color"], lw=2.4, label=sc["name"])
        ax_a.fill_between(day_axis, traj, alpha=0.08, color=sc["color"])
    if intervention_day is not None:
        ax_a.axvline(intervention_day, color=PALETTE["accent4"], lw=1.8,
                     ls="--", label=f"Intervention (day {intervention_day})")
    ax_a.axhline(0.5, color=PALETTE["muted"], lw=1.2, ls=":")
    ax_a.set_title("⏱️  Trajectory Comparison — Natural vs Intervention",
                   fontweight="bold", fontsize=14, pad=14, color=PALETTE["accent1"])
    ax_a.set_xlabel("Day", fontsize=12)
    ax_a.set_ylabel("Transplant Success Probability", fontsize=12)
    ax_a.legend(fontsize=10)
    ax_a.grid(True, alpha=0.3)
    plt.tight_layout(pad=2.0)
    plt.savefig("deterioration_trajectory.png", dpi=140, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.show()
    print("💾  Saved: deterioration_trajectory.png")

    # ── Figure B: Rolling avg + Risk zone  ──────────────────────
    fig_b, axes_b = plt.subplots(1, 2, figsize=(18, 7))
    fig_b.patch.set_facecolor(PALETTE["bg"])
    fig_b.suptitle("⏱️  Deterioration Simulator — Risk Analysis",
                   fontsize=14, fontweight="bold", color=PALETTE["accent1"], y=1.01)
    plt.subplots_adjust(wspace=0.32, left=0.07, right=0.97, top=0.92, bottom=0.12)

    ax = axes_b[0]
    ax.set_facecolor(PALETTE["surface"])
    for sc in scenarios:
        s = pd.Series(all_traj[sc["name"]]).rolling(7, min_periods=1).mean()
        ax.plot(day_axis, s, color=sc["color"], lw=2.4, label=sc["name"])
    ax.set_title("7-Day Rolling Average", fontweight="bold", fontsize=13, pad=14)
    ax.set_xlabel("Day", fontsize=12)
    ax.set_ylabel("Smoothed Probability", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes_b[1]
    ax.set_facecolor(PALETTE["surface"])
    first_name = scenarios[0]["name"]
    risk_arr   = 1.0 - np.array(all_traj[first_name])
    ax.fill_between(day_axis, risk_arr, color=PALETTE["accent2"], alpha=0.5)
    ax.fill_between(day_axis, risk_arr, where=(risk_arr >= 0.5),
                    color="#FF2222", alpha=0.45, label="High Risk (>50%)")
    ax.fill_between(day_axis, risk_arr, where=(risk_arr < 0.5),
                    color=PALETTE["accent4"], alpha=0.4, label="Moderate Risk (<50%)")
    ax.axhline(0.5, color=PALETTE["text"], lw=1.2, ls="--")
    ax.set_title(f"Risk Zone — {first_name}", fontweight="bold", fontsize=13, pad=14)
    ax.set_xlabel("Day", fontsize=12)
    ax.set_ylabel("Failure Risk", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.savefig("deterioration_risk.png", dpi=140, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.show()
    print("💾  Saved: deterioration_risk.png")

    # ── Figure C: Final probability bar ─────────────────────────
    fig_c, ax_c = plt.subplots(figsize=(10, 6))
    fig_c.patch.set_facecolor(PALETTE["bg"])
    ax_c.set_facecolor(PALETTE["surface"])
    names  = [sc["name"] for sc in scenarios]
    finals = [all_traj[n][-1] for n in names]
    colors = [sc["color"]     for sc in scenarios]
    bars   = ax_c.bar(names, finals, color=colors, edgecolor=PALETTE["bg"], width=0.4)
    ax_c.set_ylim(0, 1.18)
    ax_c.set_title(f"⏱️  Final Probability after {days} Days",
                   fontweight="bold", fontsize=14, pad=14, color=PALETTE["accent1"])
    ax_c.set_ylabel("Success Probability", fontsize=12)
    ax_c.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, finals):
        ax_c.text(bar.get_x() + bar.get_width()/2, val + 0.025,
                  f"{val:.3f}", ha="center", fontweight="bold",
                  fontsize=13, color=PALETTE["text"])
    plt.tight_layout(pad=2.0)
    plt.savefig("deterioration_final.png", dpi=140, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.show()
    print("💾  Saved: deterioration_final.png")

    return all_traj


# ═══════════════════════════════════════════════════════════════
# 📦  STEP 8 — POPULATION STATISTICS
# ═══════════════════════════════════════════════════════════════
def plot_population_stats(X, y, feature_names):
    df_plot = X.copy()
    df_plot["outcome"] = y.values

    numeric_cols = [c for c in df_plot.columns
                    if c != "outcome" and df_plot[c].nunique() > 5][:10]

    n_cols = 2
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle("📦  Population Feature Distributions — Success vs Failure",
                 fontsize=15, fontweight="bold", color=PALETTE["accent1"], y=1.01)
    plt.subplots_adjust(hspace=0.45, wspace=0.30,
                        left=0.07, right=0.97, top=0.96, bottom=0.05)

    axes = axes.flatten()
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        ax.set_facecolor(PALETTE["surface"])
        for label, color, lbl in [(0, PALETTE["accent2"], "Failure"),
                                   (1, PALETTE["accent1"], "Success")]:
            data = df_plot.loc[df_plot["outcome"] == label, col].dropna()
            ax.hist(data, bins=28, alpha=0.68, color=color,
                    edgecolor=PALETTE["bg"], density=True, label=lbl)
        ax.set_title(col, fontsize=12, fontweight="bold", color=PALETTE["accent4"], pad=10)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    for ax in axes[len(numeric_cols):]:
        ax.set_visible(False)

    plt.savefig("population_stats.png", dpi=140, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.show()
    print("💾  Saved: population_stats.png")


# ═══════════════════════════════════════════════════════════════
# 🧬  DIGITAL TWIN CLASS
# ═══════════════════════════════════════════════════════════════
class DigitalTwin:
    def __init__(self, patient: dict, predict_proba, feature_names, X_train):
        self.patient       = patient
        self._predict      = predict_proba
        self.feature_names = feature_names
        self._X_train      = X_train

    def _prob(self):
        df_tmp = pd.DataFrame([self.patient]).reindex(
            columns=self.feature_names, fill_value=0)
        return self._predict(df_tmp)[0]

    def status(self):
        prob = self._prob()
        bar  = "█" * int(prob * 30) + "░" * (30 - int(prob * 30))
        print(f"\n🧬 Digital Twin — Transplant Success Probability")
        print(f"   [{bar}]  {prob:.1%}")
        risk = "🟢 Low" if prob > 0.65 else ("🟡 Moderate" if prob > 0.40 else "🔴 High")
        print(f"   Risk Level: {risk}")
        return prob

    def analyze(self, top_n=14):
        return analyze_feature_impact(
            self.patient, self._predict, self.feature_names,
            self._X_train, top_n=top_n)

    def counterfactual(self, target_prob=0.70):
        return generate_counterfactual(
            self.patient, self._predict, self.feature_names,
            target_prob=target_prob)

    def simulate(self, days=180, intervention_day=None,
                 intervention_changes=None, scenarios=None):
        return simulate_deterioration(
            self.patient, self._predict, self.feature_names,
            days=days,
            intervention_day=intervention_day,
            intervention_changes=intervention_changes,
            scenarios=scenarios)


# ═══════════════════════════════════════════════════════════════
# 🚀  MAIN
# ═══════════════════════════════════════════════════════════════
def main(filepath=None):
    print("=" * 65)
    print("  🏥  KIDNEY TRANSPLANT DIGITAL TWIN PIPELINE")
    print("=" * 65)

    df = load_data(filepath)
    X, y, feature_names, df_clean = preprocess(df)
    models       = train_models(X, y)
    predict_prob = make_predictor(models, feature_names)

    print("\n📊 Plotting evaluation dashboard …")
    plot_evaluation_dashboard(models, predict_prob, feature_names)

    print("\n📦 Plotting population statistics …")
    plot_population_stats(models["X_train"], models["y_train"], feature_names)

    probs_test = predict_prob(models["X_test"])
    idx_worst  = np.argmin(probs_test)
    patient_sample = models["X_test"].iloc[idx_worst].to_dict()
    print(f"\n🔍 Demo patient index {idx_worst}  (prob = {probs_test[idx_worst]:.3f})")

    twin = DigitalTwin(patient_sample, predict_prob, feature_names, models["X_train"])
    twin.status()

    print("\n🔬 Running feature impact analysis …")
    twin.analyze()

    print("\n🔄 Running counterfactual generator …")
    twin.counterfactual(target_prob=0.68)

    print("\n⏱️  Running deterioration simulator …")
    twin.simulate(
        days=180,
        intervention_day=45,
        intervention_changes={"sc": 1.2, "bp": 72, "hemo": 13.0},
    )

    print("\n" + "=" * 65)
    print("  ✅  PIPELINE COMPLETE — graphs saved as PNG files")
    print("  📁  evaluation_dashboard_1.png  (ROC · PR · Confusion)")
    print("  📁  evaluation_dashboard_2.png  (Prob dist · Importance · Metrics)")
    print("  📁  population_stats.png")
    print("  📁  feature_impact.png")
    print("  📁  counterfactual_trajectory.png")
    print("  📁  counterfactual_changes.png")
    print("  📁  deterioration_trajectory.png")
    print("  📁  deterioration_risk.png")
    print("  📁  deterioration_final.png")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ✏️  Update this path to your CSV file, or set to None for synthetic data
    main(filepath=r"C:\Users\maryn\Desktop\predictive\Kidney_Organ_SupplyChain_RawDataset.csv")