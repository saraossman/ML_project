"""
Run the full model training and evaluation pipeline.
"""

import argparse
import json
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
    tune_ridge,
    tune_random_forest,
    tune_xgboost,
    tune_lightgbm,
    build_stacking,
    plot_feature_importance,
    plot_residuals,
    plot_model_comparison,
    FIGURES_DIR,
    MODELS_DIR,
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def main(encoding: str = 'hotenc'):
    #Load data
    print("=" * 60)
    print(f"Loading data  (encoding={encoding})...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(encoding=encoding)

    all_results = []

    # Baselines
    print("\n" + "=" * 60)
    print("BASELINES")

    # Absolute floor
    evaluate_model(
        'Median Predictor',
        DummyRegressor(strategy='median'),
        X_train, y_train, X_val, y_val, X_test, y_test,
        all_results,
    )

    # Linear baseline
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

    print("\n[Ridge — tuned]")
    ridge_tuned = tune_ridge(X_train, y_train, X_val, y_val)
    evaluate_model(
        'Ridge (tuned)',
        ridge_tuned,
        X_train, y_train, X_val, y_val, X_test, y_test,
        all_results,
        scale=True,
    )

    # Random Forest
    print("\n" + "=" * 60)
    print("RANDOM FOREST — default")

    rf = evaluate_model(
        'Random Forest',
        RandomForestRegressor(
            n_estimators=200, max_depth=15,
            min_samples_leaf=5, random_state=42, n_jobs=-1,
        ),
        X_train, y_train, X_val, y_val, X_test, y_test,
        all_results,
        skip_cv=False,
    )

    print("\n" + "=" * 60)
    print("RANDOM FOREST — tuned")

    rf_tuned = tune_random_forest(X_train, y_train, X_val, y_val, n_iter=20)
    evaluate_model(
        'Random Forest (tuned)',
        rf_tuned,
        X_train, y_train, X_val, y_val, X_test, y_test,
        all_results,
        skip_cv=False,
    )

    plot_feature_importance(
        rf_tuned, X_train.columns,
        title='Random Forest (tuned) — Top 20 Feature Importances',
        save_path=FIGURES_DIR / 'feature_importance_rf.png',
    )

    # XGBoost
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
        skip_cv=False,
    )

    print("\n" + "=" * 60)
    print("XGBOOST — tuned (expanded grid + early stopping)")

    xgb_tuned = tune_xgboost(X_train, y_train, X_val, y_val, n_iter=30)
    evaluate_model(
        'XGBoost (tuned)',
        xgb_tuned,
        X_train, y_train, X_val, y_val, X_test, y_test,
        all_results,
        skip_cv=False,
    )

    plot_feature_importance(
        xgb_tuned, X_train.columns,
        title='XGBoost (tuned) — Top 20 Feature Importances',
        save_path=FIGURES_DIR / 'feature_importance_xgb.png',
    )

    #LightGBM
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
        skip_cv=False,
    )

    print("\n" + "=" * 60)
    print("LIGHTGBM — tuned (expanded grid + early stopping)")

    lgbm_tuned = tune_lightgbm(X_train, y_train, X_val, y_val, n_iter=30)
    evaluate_model(
        'LightGBM (tuned)',
        lgbm_tuned,
        X_train, y_train, X_val, y_val, X_test, y_test,
        all_results,
        skip_cv=False,
    )

    # Stacking Ensemble
    print("\n" + "=" * 60)
    print("STACKING ENSEMBLE  (RF + tuned XGB + tuned LGBM  →  Ridge)")

    stacking = build_stacking(xgb_tuned, lgbm_tuned, rf_tuned, ridge_tuned)
    evaluate_model(
        'Stacking (RF + XGB + LGBM)',
        stacking,
        X_train, y_train, X_val, y_val, X_test, y_test,
        all_results,
        skip_cv=True,   # stacking is slow to CV
    )

    #Pick best tree model for residual plots
    results_df = pd.DataFrame(all_results)
    tree_rows = results_df[~results_df['model'].str.contains('Median')]
    best_name = tree_rows.loc[tree_rows['test_r2'].idxmax(), 'model']
    best_model_map = {
        'XGBoost (tuned)':          xgb_tuned,
        'LightGBM (tuned)':         lgbm_tuned,
        'Stacking (RF + XGB + LGBM)': stacking,
        'Random Forest (tuned)':    rf_tuned,
        'Random Forest':            rf,
        'XGBoost (default)':        xgb_default,
        'LightGBM (default)':       lgbm_default,
    }
    best_model = best_model_map.get(best_name, xgb_tuned)
    print(f"\nBest model: {best_name}")

    # refit best model on train
    if best_name != 'Stacking (RF + XGB + LGBM)':
        if getattr(best_model, 'early_stopping_rounds', None):
            best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            best_model.fit(X_train, y_train)

    plot_residuals(
        y_test,
        best_model.predict(X_test),
        title=best_name,
        save_path=FIGURES_DIR / f'residuals_{best_name.replace(" ", "_").lower()}.png',
    )

    #Summary
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

    # Best hyper-parameters for every tuned model
    tuned_models = {
        'ridge':         ridge_tuned,
        'random_forest': rf_tuned,
        'xgboost':       xgb_tuned,
        'lightgbm':      lgbm_tuned,
    }
    best_params = {
        name: {
            'params':   getattr(m, 'tuned_params_', {}),
            'val_rmse': getattr(m, 'tuned_val_rmse_', None),
        }
        for name, m in tuned_models.items()
    }

    def _json_safe(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        return str(o)

    params_path = MODELS_DIR / f'best_params_{encoding}.json'
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=2, default=_json_safe)
    print(f"Best params saved to {params_path}")

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
