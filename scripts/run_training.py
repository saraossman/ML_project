"""
Run the full model training and evaluation pipeline.

Usage
-----
    conda activate ml_project
    python scripts/run_training.py                      # default: hotenc
    python scripts/run_training.py --encoding targetenc

What it does
------------
Loads the pre-split, feature-engineered parquet files from run_features.py,
then trains and evaluates every model in sequence:

  Baselines
    1. Median predictor  (DummyRegressor — absolute floor)
    2. Linear Regression (same features — isolates model complexity contribution)
    3. Ridge Regression

  Tree Models
    4. Random Forest
    5. XGBoost (default)
    6. XGBoost (tuned: expanded grid + early stopping)
    7. LightGBM (default)
    8. LightGBM (tuned: expanded grid + early stopping)
    9. Stacking Ensemble (RF + tuned XGB + tuned LGBM → Ridge meta-learner)

  Neural Network
   10. Residual MLP

Outputs
-------
  outputs/figures/model_comparison.png
  outputs/figures/feature_importance_{model}.png
  outputs/figures/shap_summary_{model}.png          (if shap installed)
  outputs/figures/residuals_{model}.png
  outputs/figures/nn_training_curve.png
  outputs/models/model_comparison.csv

Why same features for baselines?
---------------------------------
Linear Regression is trained on the *same* full feature set as the tree models.
This isolates the contribution of model complexity (linear vs. non-linear)
from feature choice. If we used fewer features for the baseline, we couldn't
tell whether the gap comes from the model or the features.
DummyRegressor(strategy='median') sets the absolute floor: any model worse
than this is doing worse than just predicting the training median every time.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models import (
    load_data,
    compute_metrics,
    evaluate_model,
    tune_xgboost,
    tune_lightgbm,
    build_stacking,
    plot_feature_importance,
    plot_shap,
    plot_residuals,
    plot_model_comparison,
    train_residual_mlp,
    predict_mlp,
    HAS_SHAP,
    FIGURES_DIR,
    MODELS_DIR,
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def main(encoding: str = 'hotenc'):
    # ── Load data ─────────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"Loading data  (encoding={encoding})...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(encoding=encoding)

    all_results = []   # single list, passed explicitly to every evaluate_model call

    # ── Baselines ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BASELINES")

    # Absolute floor: always predict the training-set median
    evaluate_model(
        'Median Predictor',
        DummyRegressor(strategy='median'),
        X_train, y_train, X_val, y_val, X_test, y_test,
        all_results,
    )

    # Linear baseline on same full feature set — isolates model complexity
    evaluate_model(
        'Linear Regression',
        LinearRegression(),
        X_train, y_train, X_val, y_val, X_test, y_test,
        all_results,
        scale=True,   # linear models benefit from standardisation
    )

    evaluate_model(
        'Ridge Regression',
        Ridge(alpha=1.0),
        X_train, y_train, X_val, y_val, X_test, y_test,
        all_results,
        scale=True,
    )

    # ── Random Forest ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RANDOM FOREST")

    rf = evaluate_model(
        'Random Forest',
        RandomForestRegressor(
            n_estimators=200, max_depth=15,
            min_samples_leaf=5, random_state=42, n_jobs=-1,
        ),
        X_train, y_train, X_val, y_val, X_test, y_test,
        all_results,
        skip_cv=True,   # val/test gives honest estimates; CV on trees is slow
    )

    plot_feature_importance(
        rf, X_train.columns,
        title='Random Forest — Top 20 Feature Importances',
        save_path=FIGURES_DIR / 'feature_importance_rf.png',
    )

    # ── XGBoost ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("XGBOOST — default")

    xgb_default = evaluate_model(
        'XGBoost (default)',
        XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            tree_method='hist',
            random_state=42, n_jobs=-1,
        ),
        X_train, y_train, X_val, y_val, X_test, y_test,
        all_results,
        skip_cv=True,   # tuned version evaluated below; CV here is redundant
    )

    print("\n" + "=" * 60)
    print("XGBOOST — tuned (expanded grid + early stopping)")

    xgb_tuned = tune_xgboost(X_train, y_train, X_val, y_val, n_iter=15)
    evaluate_model(
        'XGBoost (tuned)',
        xgb_tuned,
        X_train, y_train, X_val, y_val, X_test, y_test,
        all_results,
        skip_cv=True,
    )

    plot_feature_importance(
        xgb_tuned, X_train.columns,
        title='XGBoost (tuned) — Top 20 Feature Importances',
        save_path=FIGURES_DIR / 'feature_importance_xgb.png',
    )

    # ── LightGBM ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LIGHTGBM — default")

    lgbm_default = evaluate_model(
        'LightGBM (default)',
        LGBMRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbose=-1,
        ),
        X_train, y_train, X_val, y_val, X_test, y_test,
        all_results,
        skip_cv=True,
    )

    print("\n" + "=" * 60)
    print("LIGHTGBM — tuned (expanded grid + early stopping)")

    lgbm_tuned = tune_lightgbm(X_train, y_train, X_val, y_val, n_iter=15)
    evaluate_model(
        'LightGBM (tuned)',
        lgbm_tuned,
        X_train, y_train, X_val, y_val, X_test, y_test,
        all_results,
        skip_cv=True,
    )

    # ── Stacking Ensemble ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STACKING ENSEMBLE  (RF + tuned XGB + tuned LGBM  →  Ridge)")

    stacking = build_stacking(xgb_tuned, lgbm_tuned)
    evaluate_model(
        'Stacking (RF + XGB + LGBM)',
        stacking,
        X_train, y_train, X_val, y_val, X_test, y_test,
        all_results,
        skip_cv=True,   # stacking is slow to CV — base learners are already CV-evaluated
    )

    # ── Pick best tree model for SHAP + residual plots ────────────────────────
    results_df = pd.DataFrame(all_results)
    # exclude Median and MLP from "best tree" selection
    tree_rows = results_df[~results_df['model'].str.contains('Median|MLP')]
    best_name = tree_rows.loc[tree_rows['test_r2'].idxmax(), 'model']
    best_model_map = {
        'XGBoost (tuned)':          xgb_tuned,
        'LightGBM (tuned)':         lgbm_tuned,
        'Stacking (RF + XGB + LGBM)': stacking,
        'Random Forest':            rf,
        'XGBoost (default)':        xgb_default,
        'LightGBM (default)':       lgbm_default,
    }
    best_model = best_model_map.get(best_name, xgb_tuned)
    print(f"\nBest model: {best_name}")

    # refit best model on train (stacking already fitted; others too, but safe to redo)
    if best_name != 'Stacking (RF + XGB + LGBM)':
        best_model.fit(X_train, y_train)

    plot_residuals(
        y_test,
        best_model.predict(X_test),
        title=best_name,
        save_path=FIGURES_DIR / f'residuals_{best_name.replace(" ", "_").lower()}.png',
    )

    if HAS_SHAP and hasattr(best_model, 'feature_importances_'):
        plot_shap(
            best_model, X_test,
            title=f'SHAP — {best_name}',
            save_path=FIGURES_DIR / f'shap_{best_name.replace(" ", "_").lower()}.png',
        )
    elif not HAS_SHAP:
        print("  SHAP skipped (not installed). Run: pip install shap")

    # ── Residual MLP ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESIDUAL MLP")

    mlp, mlp_scaler, _ = train_residual_mlp(
        X_train, y_train, X_val, y_val,
        hidden_dim=256,
        dropout=0.2,
        epochs=200,
        lr=1e-3,
        patience=25,
        save_path=FIGURES_DIR / 'nn_training_curve.png',
    )

    val_preds_mlp  = predict_mlp(mlp, mlp_scaler, X_val)
    test_preds_mlp = predict_mlp(mlp, mlp_scaler, X_test)

    val_m  = compute_metrics(y_val,  val_preds_mlp)
    test_m = compute_metrics(y_test, test_preds_mlp)

    print("\n[Residual MLP — Validation]")
    for k, v in val_m.items():
        print(f"  {k:15s}: {v}")
    print("\n[Residual MLP — Test]")
    for k, v in test_m.items():
        print(f"  {k:15s}: {v}")

    all_results.append({
        'model':   'Residual MLP',
        'cv_r2':   '—', 'cv_rmse': '—',
        **{f'val_{k}':  v for k, v in val_m.items()},
        **{f'test_{k}': v for k, v in test_m.items()},
    })

    plot_residuals(
        y_test, test_preds_mlp,
        title='Residual MLP',
        save_path=FIGURES_DIR / 'residuals_mlp.png',
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    results_df = pd.DataFrame(all_results)

    summary_cols = [
        'model',
        'cv_r2', 'cv_rmse',
        'val_r2', 'val_rmse', 'val_mae', 'val_mape',
        'test_r2', 'test_rmse', 'test_mae', 'test_mape',
    ]
    summary = results_df[[c for c in summary_cols if c in results_df.columns]]
    print(summary.to_string(index=False))

    csv_path = MODELS_DIR / f'model_comparison_{encoding}.csv'
    summary.to_csv(csv_path, index=False)
    print(f"\nSaved to {csv_path}")

    plot_model_comparison(
        results_df[['model', 'test_r2', 'test_rmse']].dropna(),
        save_path=FIGURES_DIR / f'model_comparison_{encoding}.png',
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--encoding',
        choices=['hotenc', 'targetenc'],
        default='hotenc',
        help="Which feature encoding to use (default: hotenc)",
    )
    args = parser.parse_args()
    main(encoding=args.encoding)
