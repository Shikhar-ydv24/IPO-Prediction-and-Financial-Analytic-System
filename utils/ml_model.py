"""
ML Model Module — IPO Listing Gain Predictor
---------------------------------------------
Models trained:
  1. Random Forest Regressor     — predict listing gain %
  2. Gradient Boosting Regressor — predict listing gain %
  3. XGBoost Regressor           — predict listing gain %
  4. Random Forest Classifier    — classify gain category
  5. Logistic Regression         — classify gain category

Pipeline:
  preprocessing → training → evaluation → feature importance → export
"""

import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection   import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing     import StandardScaler, LabelEncoder
from sklearn.pipeline          import Pipeline
from sklearn.compose           import ColumnTransformer
from sklearn.impute             import SimpleImputer
from sklearn.preprocessing     import OneHotEncoder
from sklearn.ensemble          import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble          import RandomForestClassifier
from sklearn.linear_model      import LinearRegression, LogisticRegression
from sklearn.metrics           import (mean_absolute_error, mean_squared_error, r2_score,
                                        accuracy_score, classification_report,
                                        confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from xgboost import XGBRegressor, XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠ XGBoost not installed. Skipping XGB models. Run: pip install xgboost")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "ipo_enriched.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",   "axes.labelcolor": "#c9d1d9",
    "xtick.color": "#8b949e",      "ytick.color": "#8b949e",
    "text.color": "#c9d1d9",       "grid.color": "#21262d",
    "grid.linestyle": "--",        "grid.alpha": 0.5,
    "figure.dpi": 120,
})
GREEN = "#3fb950"; RED = "#f85149"; ACCENT = "#58a6ff"; ORANGE = "#d29922"; PURPLE = "#bc8cff"

# ──────────────────────────────────────────────────────────────────────────
# Feature Configuration
# ──────────────────────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "issue_price", "issue_size_cr", "subscription_times",
    "qib_subscription", "nii_subscription", "retail_subscription",
    "pe_ratio", "revenue_cr", "profit_cr", "debt_equity",
    "promoter_holding_pct", "nifty_on_listing", "year", "month"
]

CATEGORICAL_FEATURES = ["sector", "issue_type", "market_sentiment"]

TARGET_REGRESSION     = "listing_gain_pct"
GAIN_BINS   = [-np.inf, 0, 10, 30, np.inf]
GAIN_LABELS = ["Negative", "Flat (0–10%)", "Moderate (10–30%)", "Strong (>30%)"]


def load_data() -> pd.DataFrame:
    path = DATA_PATH if os.path.exists(DATA_PATH) else os.path.join(BASE_DIR, "data", "ipo_data.csv")
    df   = pd.read_csv(path)
    for col in NUMERIC_FEATURES + [TARGET_REGRESSION]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_preprocessor():
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, NUMERIC_FEATURES),
        ("cat", cat_pipe, CATEGORICAL_FEATURES),
    ])


def prepare_data(df: pd.DataFrame):
    available_num = [c for c in NUMERIC_FEATURES    if c in df.columns]
    available_cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    all_features  = available_num + available_cat

    df_clean = df[all_features + [TARGET_REGRESSION]].dropna(subset=[TARGET_REGRESSION])
    X = df_clean[all_features]
    y_reg = df_clean[TARGET_REGRESSION]

    # Classification target
    y_clf = pd.cut(y_reg, bins=GAIN_BINS, labels=GAIN_LABELS).astype(str)

    return X, y_reg, y_clf, available_num, available_cat


# ──────────────────────────────────────────────────────────────────────────
# Regression Models
# ──────────────────────────────────────────────────────────────────────────
def train_regression_models(X_train, X_test, y_train, y_test, preprocessor):
    print("\n" + "="*55)
    print("REGRESSION MODELS — Predicting Listing Gain %")
    print("="*55)

    models = {
        "Random Forest":       RandomForestRegressor(n_estimators=200, max_depth=8,
                                                      min_samples_leaf=2, random_state=42),
        "Gradient Boosting":   GradientBoostingRegressor(n_estimators=200, learning_rate=0.08,
                                                          max_depth=5, random_state=42),
        "Linear Regression":   LinearRegression(),
    }
    if XGB_AVAILABLE:
        models["XGBoost"] = XGBRegressor(n_estimators=200, learning_rate=0.08,
                                          max_depth=5, random_state=42, verbosity=0)

    results   = {}
    pipelines = {}

    for name, model in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2   = r2_score(y_test, y_pred)

        results[name]   = {"MAE": mae, "RMSE": rmse, "R²": r2, "predictions": y_pred}
        pipelines[name] = pipe

        print(f"\n  {name}")
        print(f"    MAE : {mae:.2f}%")
        print(f"    RMSE: {rmse:.2f}%")
        print(f"    R²  : {r2:.3f}")

    # Save best model (by R²)
    best_name = max(results, key=lambda k: results[k]["R²"])
    best_pipe = pipelines[best_name]
    joblib.dump(best_pipe, os.path.join(MODELS_DIR, "best_regressor.pkl"))
    joblib.dump(pipelines, os.path.join(MODELS_DIR, "all_regressors.pkl"))
    print(f"\n  ✅ Best regressor: {best_name} (R²={results[best_name]['R²']:.3f})")
    print(f"     Saved → models/best_regressor.pkl")

    return results, pipelines, best_name


# ──────────────────────────────────────────────────────────────────────────
# Classification Models
# ──────────────────────────────────────────────────────────────────────────
def train_classification_models(X_train, X_test, y_train, y_test, preprocessor):
    print("\n" + "="*55)
    print("CLASSIFICATION MODELS — Predicting Gain Category")
    print("="*55)

    models = {
        "Random Forest Clf":   RandomForestClassifier(n_estimators=200, max_depth=8,
                                                       random_state=42, class_weight="balanced"),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42,
                                                   class_weight="balanced"),
    }
    if XGB_AVAILABLE:
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc  = le.transform(y_test)
        models["XGBoost Clf"] = (
            XGBClassifier(n_estimators=200, learning_rate=0.08, random_state=42, verbosity=0),
            le, y_train_enc, y_test_enc
        )

    results   = {}
    pipelines = {}

    for name, val in models.items():
        if isinstance(val, tuple):
            model, le, y_tr, y_te = val
            pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
            pipe.fit(X_train, y_tr)
            y_pred_enc = pipe.predict(X_test)
            y_pred = le.inverse_transform(y_pred_enc)
            acc = accuracy_score(y_te, y_pred_enc)
        else:
            pipe = Pipeline([("preprocessor", preprocessor), ("model", val)])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

        results[name]   = {"accuracy": acc, "predictions": y_pred}
        pipelines[name] = pipe

        print(f"\n  {name}")
        print(f"    Accuracy: {acc*100:.1f}%")
        print(classification_report(y_test, y_pred, zero_division=0, labels=GAIN_LABELS))

    best_name = max(results, key=lambda k: results[k]["accuracy"])
    joblib.dump(pipelines[best_name], os.path.join(MODELS_DIR, "best_classifier.pkl"))
    print(f"\n  ✅ Best classifier: {best_name} ({results[best_name]['accuracy']*100:.1f}%)")
    print(f"     Saved → models/best_classifier.pkl")

    return results, pipelines, best_name


# ──────────────────────────────────────────────────────────────────────────
# Feature Importance
# ──────────────────────────────────────────────────────────────────────────
def plot_feature_importance(pipeline, feature_names_num, feature_names_cat, title="Feature Importance", save_path=None):
    try:
        model       = pipeline.named_steps["model"]
        preprocessor = pipeline.named_steps["preprocessor"]

        # Get encoded cat feature names
        cat_enc   = preprocessor.named_transformers_["cat"]["encoder"]
        cat_names = list(cat_enc.get_feature_names_out(feature_names_cat))
        all_names = feature_names_num + cat_names

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0] if model.coef_.ndim > 1 else model.coef_)
        else:
            print("Model does not expose feature importances.")
            return

        # Top 15
        idx      = np.argsort(importances)[::-1][:15]
        top_feat = [all_names[i] if i < len(all_names) else f"feat_{i}" for i in idx]
        top_imp  = importances[idx]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(top_feat[::-1], top_imp[::-1], color=ACCENT, alpha=0.85, edgecolor="#0d1117")
        ax.set_xlabel("Importance Score")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.4)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            print(f"Saved → {save_path}")
        plt.show()
        plt.close()
    except Exception as e:
        print(f"Could not plot feature importance: {e}")


# ──────────────────────────────────────────────────────────────────────────
# Actual vs Predicted Plot
# ──────────────────────────────────────────────────────────────────────────
def plot_actual_vs_predicted(y_test, y_pred, model_name: str, save: bool = True):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{model_name} — Prediction Quality", fontsize=13, fontweight="bold", color="#c9d1d9")

    # Scatter
    ax = axes[0]
    ax.scatter(y_test, y_pred, alpha=0.55, s=28, color=ACCENT, edgecolors="none")
    lims = [min(y_test.min(), min(y_pred)), max(y_test.max(), max(y_pred))]
    ax.plot(lims, lims, color=ORANGE, lw=1.8, ls="--", label="Perfect prediction")
    ax.set_xlabel("Actual Listing Gain (%)")
    ax.set_ylabel("Predicted Listing Gain (%)")
    ax.set_title("Actual vs Predicted")
    ax.legend(fontsize=8, framealpha=0.3)

    # Residuals
    ax2 = axes[1]
    residuals = y_test.values - y_pred
    ax2.hist(residuals, bins=30, color=PURPLE, edgecolor="#0d1117", alpha=0.82)
    ax2.axvline(0, color=ORANGE, lw=2, ls="--")
    ax2.set_xlabel("Residual (Actual − Predicted)")
    ax2.set_ylabel("Count")
    ax2.set_title("Residual Distribution")

    plt.tight_layout()
    if save:
        path = os.path.join(ASSETS_DIR, "09_actual_vs_predicted.png")
        plt.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.show()
    plt.close()


# ──────────────────────────────────────────────────────────────────────────
# Confusion Matrix
# ──────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(y_test, y_pred, labels, title="Confusion Matrix", save: bool = True):
    cm   = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, ax=ax,
                annot_kws={"size": 10})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    plt.tight_layout()
    if save:
        path = os.path.join(ASSETS_DIR, "10_confusion_matrix.png")
        plt.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.show()
    plt.close()


# ──────────────────────────────────────────────────────────────────────────
# Predict for a new IPO
# ──────────────────────────────────────────────────────────────────────────
def predict_new_ipo(input_dict: dict, model_path: str = None) -> dict:
    """
    Predict listing gain for a new IPO given its features.

    Parameters
    ----------
    input_dict : dict with keys matching NUMERIC_FEATURES + CATEGORICAL_FEATURES
    model_path : path to a saved .pkl pipeline (defaults to best_regressor.pkl)

    Returns
    -------
    dict with predicted gain % and category
    """
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, "best_regressor.pkl")

    if not os.path.exists(model_path):
        return {"error": "Model not found. Run train_all_models() first."}

    pipeline  = joblib.load(model_path)
    input_df  = pd.DataFrame([input_dict])

    # Ensure all features present
    for col in NUMERIC_FEATURES:
        if col not in input_df.columns:
            input_df[col] = 0
    for col in CATEGORICAL_FEATURES:
        if col not in input_df.columns:
            input_df[col] = "Unknown"

    predicted_gain = float(pipeline.predict(input_df)[0])

    # Category
    cat = pd.cut([predicted_gain], bins=GAIN_BINS, labels=GAIN_LABELS)[0]

    # Confidence hint (heuristic)
    confidence = min(95, max(55, 75 + predicted_gain * 0.1))

    return {
        "predicted_listing_gain_pct": round(predicted_gain, 2),
        "gain_category":              str(cat),
        "confidence_score":           round(confidence, 1),
        "recommendation":             "✅ Invest" if predicted_gain > 10 else (
                                      "⚠️ Neutral" if predicted_gain >= 0 else "❌ Avoid")
    }


# ──────────────────────────────────────────────────────────────────────────
# Master function
# ──────────────────────────────────────────────────────────────────────────
def train_all_models():
    print("=" * 60)
    print("IPO ML MODEL TRAINING")
    print("=" * 60)

    df = load_data()
    print(f"Dataset loaded: {df.shape[0]} records")

    X, y_reg, y_clf, num_feats, cat_feats = prepare_data(df)
    print(f"Features: {len(num_feats)} numeric + {len(cat_feats)} categorical")
    print(f"Target (reg)  — mean={y_reg.mean():.2f}%, std={y_reg.std():.2f}%")
    print(f"Target (clf)  — classes: {y_clf.unique().tolist()}")

    # Train/test split — stratified on category
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    preprocessor = build_preprocessor()

    # ── Regression ──────────────────────────────────────────────────────
    reg_results, reg_pipes, best_reg = train_regression_models(
        X_train, X_test, y_reg_train, y_reg_test, preprocessor
    )

    # Feature importance for best regressor
    plot_feature_importance(
        reg_pipes[best_reg], num_feats, cat_feats,
        title=f"{best_reg} — Feature Importance (Regression)",
        save_path=os.path.join(ASSETS_DIR, "09a_feature_importance_reg.png")
    )

    # Actual vs Predicted
    plot_actual_vs_predicted(y_reg_test, reg_results[best_reg]["predictions"], best_reg)

    # ── Classification ──────────────────────────────────────────────────
    clf_results, clf_pipes, best_clf = train_classification_models(
        X_train, X_test, y_clf_train, y_clf_test, preprocessor
    )

    plot_confusion_matrix(
        y_clf_test, clf_results[best_clf]["predictions"],
        labels=GAIN_LABELS,
        title=f"{best_clf} — Confusion Matrix"
    )

    # Save metadata
    meta = {
        "best_regressor":    best_reg,
        "best_classifier":   best_clf,
        "reg_results":       {k: {m: v for m, v in v.items() if m != "predictions"}
                              for k, v in reg_results.items()},
        "clf_results":       {k: {m: v for m, v in v.items() if m != "predictions"}
                              for k, v in clf_results.items()},
        "numeric_features":  num_feats,
        "categorical_features": cat_feats,
        "gain_labels":       GAIN_LABELS,
    }
    joblib.dump(meta, os.path.join(MODELS_DIR, "model_metadata.pkl"))
    print("\n✅ All models trained and saved to models/")
    return meta


if __name__ == "__main__":
    meta = train_all_models()

    # Quick demo prediction
    demo = {
        "issue_price": 500, "issue_size_cr": 2000, "subscription_times": 80,
        "qib_subscription": 120, "nii_subscription": 100, "retail_subscription": 15,
        "pe_ratio": 35, "revenue_cr": 1500, "profit_cr": 120, "debt_equity": 0.3,
        "promoter_holding_pct": 65, "nifty_on_listing": 18000,
        "year": 2024, "month": 3,
        "sector": "IT", "issue_type": "OFS+Fresh", "market_sentiment": "Bullish"
    }
    result = predict_new_ipo(demo)
    print("\nDemo Prediction:")
    for k, v in result.items():
        print(f"  {k}: {v}")
