import subprocess, sys

def _install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

try:
    import sklearn
except ImportError:
    _install("scikit-learn")

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — works in Colab cells
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve,
)

np.random.seed(42)

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
def show_and_save(fig, filename):
    """Save figure and display inline in Colab."""
    fig.savefig(filename, dpi=140, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    try:
        from IPython.display import display, Image
        display(Image(filename))
    except Exception:
        pass
    print(f"  Saved: {filename}")

def load_data(filepath=None):
    path = "/content/Kidney_Organ_SupplyChain_RawDataset.csv"

    if path and os.path.exists(path):
        df = pd.read_csv(path)
        print(f"[OK]  Loaded '{path}'  shape={df.shape}")
        print("      Columns:", df.columns.tolist())
        return df

    if path:
        print(f"[!!]  '{path}' not found — falling back to synthetic data.")

    print("[>>]  Generating synthetic kidney supply-chain dataset ...")
    n   = 500
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
        0.20 * (df["sc"]                  < 3.0).astype(int) +
        0.18 * (df["hemo"]                > 10 ).astype(int) +
        0.12 * (df["hla_mismatch"]        < 3  ).astype(int) +
        0.12 * (df["cold_ischemia_hours"] < 20 ).astype(int) +
        0.10 * (df["bp"]                  < 90 ).astype(int) +
        0.08 * (df["bgr"]                 < 140).astype(int) +
        0.08 * (df["blood_group_match"]   == 1 ).astype(int) +
        0.06 * (df["donor_type"]          == "living").astype(int) +
        0.06 * (df["pra_percent"]         < 30 ).astype(int)
    )
    noise = rng.normal(0, 0.08, n)
    df["transplant_success"] = ((score + noise) > 0.42).astype(int)

    print(f"[OK]  Synthetic data ready  shape={df.shape}")
    print(f"      Success rate: {df['transplant_success'].mean():.1%}")
    return df

def preprocess(df: pd.DataFrame):
    df = df.copy()
    df.replace("?", np.nan, inplace=True)

    # ── Target detection ──────────────────────────────────────
    TARGET_CANDIDATES = [
        "transplant_success", "classification", "ckd", "outcome",
        "result", "label", "target", "status", "diagnosis",
        "graft_survival", "success", "failure",
    ]

    target_col = None

    # 1. Explicit override
    if EXPLICIT_TARGET:
        if EXPLICIT_TARGET in df.columns:
            target_col = EXPLICIT_TARGET
        else:
            # case-insensitive search
            match = next(
                (c for c in df.columns if c.lower() == EXPLICIT_TARGET.lower()), None
            )
            if match:
                target_col = match
            else:
                print(f"[!!]  EXPLICIT_TARGET='{EXPLICIT_TARGET}' not found in columns.")
                print(f"      Available columns: {df.columns.tolist()}")
                raise ValueError(
                    f"Column '{EXPLICIT_TARGET}' not found. "
                    "Update EXPLICIT_TARGET at the top of the script."
                )

    # 2. Name-based auto-detect
    if target_col is None:
        target_col = next(
            (c for c in df.columns if c.strip().lower() in TARGET_CANDIDATES), None
        )

    # 3. Last resort: lowest-cardinality binary column
    if target_col is None:
        binary_cols = [c for c in df.columns
                       if df[c].nunique() <= 5 and df[c].nunique() >= 2]
        if binary_cols:
            target_col = min(binary_cols, key=lambda c: df[c].nunique())
            print(f"[!!]  Auto-guessing target='{target_col}'  "
                  f"unique values: {df[target_col].unique().tolist()}")
            print(f"      If wrong, set EXPLICIT_TARGET at the top of the script.")
        else:
            raise ValueError(
                "Cannot auto-detect target column.\n"
                "Set EXPLICIT_TARGET = 'YourColumnName' at the top of the script.\n"
                f"Available columns: {df.columns.tolist()}"
            )

    print(f"[>>]  Target column: '{target_col}'")

    # ── Encode target ─────────────────────────────────────────
    label_maps = {
        "ckd": 1, "notckd": 0,
        "success": 1, "failure": 0,
        "yes": 1, "no": 0,
        "1": 1, "0": 0,
        "true": 1, "false": 0,
        "positive": 1, "negative": 0,
    }

    col_series = df[target_col]

    if col_series.dtype == object or str(col_series.dtype).startswith("string"):
        mapped = col_series.astype(str).str.strip().str.lower().map(label_maps)
        unmapped_mask = mapped.isnull() & col_series.notnull()
        if unmapped_mask.any():
            unique_unmapped = col_series[unmapped_mask].unique()
            print(f"[!!]  Could not map target values: {unique_unmapped}")
            print(f"      Only these values are handled automatically:")
            print(f"      {list(label_maps.keys())}")
            print(f"      Please preprocess your target column manually, or set")
            print(f"      EXPLICIT_TARGET to a numeric 0/1 column.")
            raise ValueError(
                f"Unmapped target values: {unique_unmapped}. "
                "See instructions above."
            )
        df[target_col] = mapped
    else:
        # Already numeric — ensure it's 0/1
        unique_vals = col_series.dropna().unique()
        if set(unique_vals) - {0, 1, 0.0, 1.0}:
            print(f"[!!]  Target '{target_col}' has values: {unique_vals}")
            print(f"      Expected binary 0/1. Binarising at median ...")
            median = col_series.median()
            df[target_col] = (col_series > median).astype(int)
        else:
            df[target_col] = col_series.astype(int)

    # Drop rows where target is still null
    n_before = len(df)
    df = df.dropna(subset=[target_col])
    dropped = n_before - len(df)
    if dropped:
        print(f"[!!]  Dropped {dropped} rows with null target values.")

    y = df[target_col].astype(int)

    # ── Features ──────────────────────────────────────────────
    X = df.drop(columns=[target_col]).copy()
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    X = X.apply(pd.to_numeric, errors="coerce")
    for col in X.columns:
        if X[col].isnull().all():
            X[col] = 0.0
        else:
            X[col] = X[col].fillna(X[col].median())

    feature_names = X.columns.tolist()
    print(f"[OK]  Preprocessing done — {len(feature_names)} features, "
          f"{len(y)} samples, {y.mean():.1%} positive rate")
    return X, y, feature_names, df

def train_models(X, y):
    min_class = y.value_counts().min()
    stratify  = y if min_class >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)
    X_test_sc   = scaler.transform(X_test)

    print("\n[>>]  Training Random Forest ...")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=3,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    print("[>>]  Training Gradient Boosting ...")
    gbm = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        subsample=0.8, random_state=42
    )
    gbm.fit(X_train_sc, y_train)

    cv_rf  = cross_val_score(rf,  X_train,    y_train, cv=5, scoring="roc_auc")
    cv_gbm = cross_val_score(gbm, X_train_sc, y_train, cv=5, scoring="roc_auc")
    print(f"      RF  CV AUC : {cv_rf.mean():.4f} +/- {cv_rf.std():.4f}")
    print(f"      GBM CV AUC : {cv_gbm.mean():.4f} +/- {cv_gbm.std():.4f}")

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

def plot_evaluation_dashboard(models, predict_proba, feature_names):
    X_test, y_test = models["X_test"], models["y_test"]
    y_prob = predict_proba(X_test)
    y_pred = (y_prob > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    print(f"\n[>>]  Ensemble  Accuracy={acc:.4f}  ROC-AUC={auc:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # ── Figure 1: ROC · PR · Confusion Matrix ──────────────────
    fig1, axes1 = plt.subplots(1, 3, figsize=(22, 7))
    fig1.patch.set_facecolor(PALETTE["bg"])
    fig1.suptitle("Kidney Transplant — Evaluation Dashboard (Part 1)",
                  fontsize=15, fontweight="bold", color=PALETTE["accent1"])
    plt.subplots_adjust(wspace=0.38, left=0.06, right=0.97, top=0.90, bottom=0.13)

    # ROC
    ax = axes1[0]
    ax.set_facecolor(PALETTE["surface"])
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax.plot(fpr, tpr, color=PALETTE["accent1"], lw=2.5,
            label=f"Ensemble  AUC = {auc:.3f}")
    ax.plot([0,1],[0,1], color=PALETTE["muted"], lw=1.2, ls="--",
            label="Random baseline")
    ax.fill_between(fpr, tpr, alpha=0.13, color=PALETTE["accent1"])
    ax.set_title("ROC Curve", fontweight="bold", fontsize=13, pad=14)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)

    # Precision-Recall
    ax = axes1[1]
    ax.set_facecolor(PALETTE["surface"])
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ax.plot(rec, prec, color=PALETTE["accent2"], lw=2.5)
    ax.fill_between(rec, prec, alpha=0.13, color=PALETTE["accent2"])
    ax.set_title("Precision-Recall Curve", fontweight="bold", fontsize=13, pad=14)
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.grid(True, alpha=0.3)

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

    show_and_save(fig1, "evaluation_dashboard_1.png")

    # ── Figure 2: Prob dist · Feature Importance · Metrics ─────
    fig2, axes2 = plt.subplots(1, 3, figsize=(24, 8))
    fig2.patch.set_facecolor(PALETTE["bg"])
    fig2.suptitle("Kidney Transplant — Evaluation Dashboard (Part 2)",
                  fontsize=15, fontweight="bold", color=PALETTE["accent1"])
    plt.subplots_adjust(wspace=0.38, left=0.06, right=0.97, top=0.90, bottom=0.13)

    # Probability distribution
    ax = axes2[0]
    ax.set_facecolor(PALETTE["surface"])
    ax.hist(y_prob[y_test == 0], bins=28, alpha=0.72, color=PALETTE["accent2"],
            label="Failure", edgecolor=PALETTE["bg"])
    ax.hist(y_prob[y_test == 1], bins=28, alpha=0.72, color=PALETTE["accent1"],
            label="Success", edgecolor=PALETTE["bg"])
    ax.axvline(0.5, color=PALETTE["accent4"], lw=2, ls="--", label="Threshold = 0.5")
    ax.set_title("Predicted Probability Distribution", fontweight="bold",
                 fontsize=13, pad=14)
    ax.set_xlabel("Predicted Probability", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Feature importance
    ax = axes2[1]
    ax.set_facecolor(PALETTE["surface"])
    imp  = models["rf"].feature_importances_
    top  = np.argsort(imp)[-14:]
    cols = [feature_names[i] for i in top]
    vals = imp[top]
    colors = [PALETTE["accent1"] if v > np.median(vals)
              else PALETTE["accent3"] for v in vals]
    bars = ax.barh(cols, vals, color=colors, edgecolor=PALETTE["bg"], height=0.62)
    ax.set_title("Feature Importance (Random Forest)", fontweight="bold",
                 fontsize=13, pad=14)
    ax.set_xlabel("Importance", fontsize=11)
    ax.grid(True, alpha=0.3, axis="x")
    for bar, val in zip(bars, vals):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9, color=PALETTE["text"])

    # Metrics bar chart
    ax = axes2[2]
    ax.set_facecolor(PALETTE["surface"])
    metrics = {
        "Accuracy":       acc,
        "ROC-AUC":        auc,
        "Precision\n(1)": report.get("1", {}).get("precision", 0),
        "Recall\n(1)":    report.get("1", {}).get("recall", 0),
        "F1\n(1)":        report.get("1", {}).get("f1-score", 0),
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
    for bar, val in zip(bars2, metrics.values()):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.025,
                f"{val:.3f}", ha="center", fontsize=11,
                fontweight="bold", color=PALETTE["text"])

    show_and_save(fig2, "evaluation_dashboard_2.png")

def analyze_feature_impact(patient_dict, predict_proba, feature_names,
                            X_train, top_n=14):
    patient_df = pd.DataFrame([patient_dict]).reindex(
        columns=feature_names, fill_value=0)
    base_prob = predict_proba(patient_df)[0]

    impacts = {}
    for feat in feature_names:
        mean_val        = X_train[feat].mean() if feat in X_train.columns else 0.0
        perturbed       = patient_df.copy()
        perturbed[feat] = mean_val
        impacts[feat]   = base_prob - predict_proba(perturbed)[0]

    sorted_impacts = sorted(
        impacts.items(), key=lambda x: abs(x[1]), reverse=True
    )[:top_n]

    feats  = [f for f, _ in sorted_impacts]
    vals   = [v for _, v in sorted_impacts]
    colors = [PALETTE["accent1"] if v > 0 else PALETTE["accent2"] for v in vals]

    fig, ax = plt.subplots(figsize=(13, 8))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["surface"])

    bars = ax.barh(feats, vals, color=colors, edgecolor=PALETTE["bg"], height=0.62)
    ax.axvline(0, color=PALETTE["text"], lw=1.2)
    ax.set_title(
        f"Feature Impact Analysis   (Base Probability = {base_prob:.3f})",
        fontweight="bold", fontsize=14, pad=16, color=PALETTE["accent1"]
    )
    ax.set_xlabel("Impact on Transplant Success Probability", fontsize=12)
    ax.grid(True, alpha=0.3, axis="x")

    for bar, val in zip(bars, vals):
        xpos = val + (0.003 if val >= 0 else -0.003)
        ha   = "left" if val >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height()/2,
                f"{val:+.4f}", va="center", ha=ha,
                fontsize=9.5, color=PALETTE["text"])

    legend_patches = [
        mpatches.Patch(color=PALETTE["accent1"], label="Helps outcome"),
        mpatches.Patch(color=PALETTE["accent2"], label="Hurts outcome"),
    ]
    ax.legend(handles=legend_patches, fontsize=10, loc="lower right")
    plt.tight_layout(pad=2.0)
    show_and_save(fig, "feature_impact.png")

    print(f"\n[>>]  Feature Impact  (Base Prob: {base_prob:.3f})")
    for feat, impact in sorted_impacts:
        direction = "Helps" if impact > 0 else "Hurts"
        print(f"      {feat:<30} {impact:+.4f}  {direction}")

    return dict(sorted_impacts), base_prob

def generate_counterfactual(patient_dict, predict_proba, feature_names,
                             target_prob=0.70, max_iterations=300):
    patient_df = pd.DataFrame([patient_dict]).reindex(
        columns=feature_names, fill_value=0)
    best      = patient_df.copy()
    best_prob = predict_proba(best)[0]

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
        print("[!!]  No modifiable features found in dataset.")
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

    # ── Figure A: Optimisation trajectory ───────────────────────
    fig_a, ax_a = plt.subplots(figsize=(12, 6))
    fig_a.patch.set_facecolor(PALETTE["bg"])
    ax_a.set_facecolor(PALETTE["surface"])
    ax_a.plot(trajectory, color=PALETTE["accent3"], lw=2.5,
              marker="o", markersize=5, markerfacecolor=PALETTE["accent4"])
    ax_a.axhline(target_prob, color=PALETTE["accent2"], lw=1.8,
                 ls="--", label=f"Target = {target_prob}")
    ax_a.fill_between(range(len(trajectory)), trajectory,
                      alpha=0.15, color=PALETTE["accent3"])
    ax_a.set_title("Counterfactual Optimisation Trajectory",
                   fontweight="bold", fontsize=14, pad=14,
                   color=PALETTE["accent1"])
    ax_a.set_xlabel("Iteration", fontsize=12)
    ax_a.set_ylabel("Transplant Success Probability", fontsize=12)
    ax_a.legend(fontsize=10)
    ax_a.grid(True, alpha=0.3)
    plt.tight_layout(pad=2.0)
    show_and_save(fig_a, "counterfactual_trajectory.png")

    # ── Figure B: Before / After per feature (FIXED) ────────────
    original_df   = pd.DataFrame([patient_dict]).reindex(
        columns=feature_names, fill_value=0)
    changed_feats = [
        f for f in modifiable
        if abs(best[f].values[0] - original_df[f].values[0]) > 1e-6
    ][:10]

    if changed_feats:
        n_feats  = len(changed_feats)
        fig_h    = max(5, n_feats * 1.6)
        fig_b, axes_b = plt.subplots(n_feats, 1, figsize=(11, fig_h))
        fig_b.patch.set_facecolor(PALETTE["bg"])
        fig_b.suptitle(
            f"Feature Changes — Before vs After\n"
            f"Probability: {trajectory[0]:.3f}  ->  {best_prob:.3f}  "
            f"(target={target_prob})",
            fontweight="bold", fontsize=13,
            color=PALETTE["accent1"], y=1.01
        )
        plt.subplots_adjust(
            hspace=0.65, left=0.22, right=0.90, top=0.93, bottom=0.06
        )

        if n_feats == 1:
            axes_b = [axes_b]   # make iterable for single subplot

        for ax_row, feat in zip(axes_b, changed_feats):
            orig_val  = float(original_df[feat].values[0])
            after_val = float(best[feat].values[0])
            ax_row.set_facecolor(PALETTE["surface"])

            bar_colors = [PALETTE["accent2"], PALETTE["accent3"]]
            bars = ax_row.barh(
                ["Before", "After"],
                [orig_val, after_val],
                color=bar_colors,
                edgecolor=PALETTE["bg"],
                height=0.45,
            )
            ax_row.set_title(feat, fontsize=10, fontweight="bold",
                             color=PALETTE["accent4"], pad=5, loc="left")
            ax_row.tick_params(labelsize=9)
            ax_row.grid(True, alpha=0.3, axis="x")

            # Value labels
            x_max = max(abs(orig_val), abs(after_val), 1e-9)
            for bar, val in zip(bars, [orig_val, after_val]):
                offset = x_max * 0.04
                ax_row.text(
                    bar.get_width() + offset,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9,
                    color=PALETTE["text"]
                )

        show_and_save(fig_b, "counterfactual_changes.png")

    print(f"\n[OK]  Counterfactual: {trajectory[0]:.3f} -> {best_prob:.3f}  "
          f"(target={target_prob}, iterations={len(trajectory)-1})")
    return best.iloc[0].to_dict(), trajectory

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
        patient = {
            k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in pd_dict.items()
        }
        traj = []
        for day in range(days):
            if int_day is not None and day == int_day and changes:
                for k, v in changes.items():
                    patient[k] = v
                print(f"      Intervention applied day {day} — '{label}'")
            for feat, rate in drift.items():
                if feat in patient and isinstance(patient[feat], float):
                    patient[feat] = max(
                        0.0,
                        patient[feat] + rate + np.random.normal(0, abs(rate)*0.4)
                    )
            df_tmp = pd.DataFrame([patient]).reindex(
                columns=feature_names, fill_value=0)
            traj.append(predict_proba(df_tmp)[0])
        return traj

    if scenarios is None:
        scenarios = [
            {
                "name":         "Natural Decline",
                "drift":        default_drift,
                "color":        PALETTE["accent2"],
                "intervention": False,
            },
            {
                "name":         "With Intervention",
                "drift":        default_drift,
                "color":        PALETTE["accent3"],
                "intervention": True,
            },
        ]

    day_axis  = np.arange(days)
    all_traj  = {}
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
    ax_a.axhline(0.5, color=PALETTE["muted"], lw=1.2, ls=":",
                 label="Decision threshold (0.5)")
    ax_a.set_title("Trajectory Comparison — Natural vs Intervention",
                   fontweight="bold", fontsize=14, pad=14,
                   color=PALETTE["accent1"])
    ax_a.set_xlabel("Day", fontsize=12)
    ax_a.set_ylabel("Transplant Success Probability", fontsize=12)
    ax_a.legend(fontsize=10)
    ax_a.grid(True, alpha=0.3)
    plt.tight_layout(pad=2.0)
    show_and_save(fig_a, "deterioration_trajectory.png")

    # ── Figure B: Rolling average + Risk zone ───────────────────
    fig_b, (ax_b1, ax_b2) = plt.subplots(1, 2, figsize=(18, 7))
    fig_b.patch.set_facecolor(PALETTE["bg"])
    fig_b.suptitle("Deterioration Simulator — Risk Analysis",
                   fontsize=14, fontweight="bold",
                   color=PALETTE["accent1"])
    plt.subplots_adjust(wspace=0.32, left=0.07, right=0.97,
                        top=0.90, bottom=0.12)

    ax_b1.set_facecolor(PALETTE["surface"])
    for sc in scenarios:
        s = pd.Series(all_traj[sc["name"]]).rolling(7, min_periods=1).mean()
        ax_b1.plot(day_axis, s, color=sc["color"], lw=2.4, label=sc["name"])
    ax_b1.set_title("7-Day Rolling Average", fontweight="bold",
                    fontsize=13, pad=14)
    ax_b1.set_xlabel("Day", fontsize=12)
    ax_b1.set_ylabel("Smoothed Probability", fontsize=12)
    ax_b1.legend(fontsize=10)
    ax_b1.grid(True, alpha=0.3)

    ax_b2.set_facecolor(PALETTE["surface"])
    first_name = scenarios[0]["name"]
    risk_arr   = 1.0 - np.array(all_traj[first_name])
    ax_b2.fill_between(day_axis, risk_arr,
                       color=PALETTE["accent2"], alpha=0.5)
    ax_b2.fill_between(day_axis, risk_arr, where=(risk_arr >= 0.5),
                       color="#FF2222", alpha=0.45, label="High Risk (>50%)")
    ax_b2.fill_between(day_axis, risk_arr, where=(risk_arr < 0.5),
                       color=PALETTE["accent4"], alpha=0.4, label="Moderate Risk (<50%)")
    ax_b2.axhline(0.5, color=PALETTE["text"], lw=1.2, ls="--")
    ax_b2.set_title(f"Risk Zone — {first_name}", fontweight="bold",
                    fontsize=13, pad=14)
    ax_b2.set_xlabel("Day", fontsize=12)
    ax_b2.set_ylabel("Failure Risk", fontsize=12)
    ax_b2.legend(fontsize=10)
    ax_b2.grid(True, alpha=0.3)

    show_and_save(fig_b, "deterioration_risk.png")

    # ── Figure C: Final probability bar ─────────────────────────
    fig_c, ax_c = plt.subplots(figsize=(10, 6))
    fig_c.patch.set_facecolor(PALETTE["bg"])
    ax_c.set_facecolor(PALETTE["surface"])
    names  = [sc["name"]  for sc in scenarios]
    finals = [all_traj[n][-1] for n in names]
    colors = [sc["color"] for sc in scenarios]
    bars   = ax_c.bar(names, finals, color=colors,
                      edgecolor=PALETTE["bg"], width=0.4)
    ax_c.set_ylim(0, 1.18)
    ax_c.set_title(f"Final Probability after {days} Days",
                   fontweight="bold", fontsize=14, pad=14,
                   color=PALETTE["accent1"])
    ax_c.set_ylabel("Success Probability", fontsize=12)
    ax_c.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, finals):
        ax_c.text(bar.get_x() + bar.get_width()/2, val + 0.025,
                  f"{val:.3f}", ha="center", fontweight="bold",
                  fontsize=13, color=PALETTE["text"])
    plt.tight_layout(pad=2.0)
    show_and_save(fig_c, "deterioration_final.png")

    return all_traj

def plot_population_stats(X, y, feature_names):
    df_plot = X.copy()
    df_plot["outcome"] = y.values

    numeric_cols = [
        c for c in df_plot.columns
        if c != "outcome" and df_plot[c].nunique() > 5
    ][:10]

    n_cols = 2
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle("Population Feature Distributions — Success vs Failure",
                 fontsize=15, fontweight="bold",
                 color=PALETTE["accent1"])
    plt.subplots_adjust(hspace=0.50, wspace=0.30,
                        left=0.07, right=0.97, top=0.94, bottom=0.05)

    axes_flat = axes.flatten() if n_rows > 1 else list(axes)
    for idx, col in enumerate(numeric_cols):
        ax = axes_flat[idx]
        ax.set_facecolor(PALETTE["surface"])
        for label, color, lbl in [
            (0, PALETTE["accent2"], "Failure"),
            (1, PALETTE["accent1"], "Success"),
        ]:
            data = df_plot.loc[df_plot["outcome"] == label, col].dropna()
            ax.hist(data, bins=28, alpha=0.68, color=color,
                    edgecolor=PALETTE["bg"], density=True, label=lbl)
        ax.set_title(col, fontsize=12, fontweight="bold",
                     color=PALETTE["accent4"], pad=10)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    for ax in axes_flat[len(numeric_cols):]:
        ax.set_visible(False)

    show_and_save(fig, "population_stats.png")

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
        bar  = "#" * int(prob * 30) + "." * (30 - int(prob * 30))
        print(f"\n[>>]  Digital Twin — Transplant Success Probability")
        print(f"      [{bar}]  {prob:.1%}")
        risk = ("LOW" if prob > 0.65 else
                "MODERATE" if prob > 0.40 else "HIGH")
        print(f"      Risk Level: {risk}")
        return prob

    def analyze(self, top_n=14):
        return analyze_feature_impact(
            self.patient, self._predict,
            self.feature_names, self._X_train, top_n=top_n)

    def counterfactual(self, target_prob=0.70):
        return generate_counterfactual(
            self.patient, self._predict,
            self.feature_names, target_prob=target_prob)

    def simulate(self, days=180, intervention_day=None,
                 intervention_changes=None, scenarios=None):
        return simulate_deterioration(
            self.patient, self._predict, self.feature_names,
            days=days,
            intervention_day=intervention_day,
            intervention_changes=intervention_changes,
            scenarios=scenarios)

def main():
    print("=" * 65)
    print("  KIDNEY TRANSPLANT DIGITAL TWIN PIPELINE")
    print("=" * 65)

    df                    = load_data()
    X, y, feature_names, df_clean = preprocess(df)
    models                = train_models(X, y)
    predict_prob          = make_predictor(models, feature_names)

    print("\n[>>]  Plotting evaluation dashboard ...")
    plot_evaluation_dashboard(models, predict_prob, feature_names)

    print("\n[>>]  Plotting population statistics ...")
    plot_population_stats(models["X_train"], models["y_train"], feature_names)

    probs_test = predict_prob(models["X_test"])
    idx_worst  = np.argmin(probs_test)
    patient_sample = models["X_test"].iloc[idx_worst].to_dict()
    print(f"\n[>>]  Demo patient index {idx_worst}  "
          f"(prob = {probs_test[idx_worst]:.3f})")

    twin = DigitalTwin(
        patient_sample, predict_prob, feature_names, models["X_train"]
    )
    twin.status()

    print("\n[>>]  Running feature impact analysis ...")
    twin.analyze()

    print("\n[>>]  Running counterfactual generator ...")
    twin.counterfactual(target_prob=0.68)

    print("\n[>>]  Running deterioration simulator ...")
    twin.simulate(
        days=180,
        intervention_day=45,
        intervention_changes={"sc": 1.2, "bp": 72, "hemo": 13.0},
    )

    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE — all images saved")
    print("  evaluation_dashboard_1.png  (ROC / PR / Confusion)")
    print("  evaluation_dashboard_2.png  (Dist / Importance / Metrics)")
    print("  population_stats.png")
    print("  feature_impact.png")
    print("  counterfactual_trajectory.png")
    print("  counterfactual_changes.png   (FIXED — per-feature rows)")
    print("  deterioration_trajectory.png")
    print("  deterioration_risk.png")
    print("  deterioration_final.png")
    print("=" * 65)

    try:
        from google.colab import files
        saved = [
            "evaluation_dashboard_1.png",
            "evaluation_dashboard_2.png",
            "population_stats.png",
            "feature_impact.png",
            "counterfactual_trajectory.png",
            "counterfactual_changes.png",
            "deterioration_trajectory.png",
            "deterioration_risk.png",
            "deterioration_final.png",
        ]
        print("\n[>>]  Downloading all PNGs to your local machine ...")
        for f in saved:
            if os.path.exists(f):
                files.download(f)
    except ImportError:
        pass   # Not in Colab — skip download

if __name__ == "__main__":
    main()
