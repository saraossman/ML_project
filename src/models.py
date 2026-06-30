"""
Model training, evaluation, and plotting
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

RANDOM_STATE = 42
CV_FOLDS     = 5
VAL_SIZE     = 0.15   # fraction validation
FIGURES_DIR  = Path('outputs/figures')
MODELS_DIR   = Path('outputs/models')


#1. Data loading

def load_data(
    encoding: str = 'hotenc',
    processed_dir: str = 'data/processed',
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
):
    """
    Load train/test parquet files produced by run_features.py and split a
    validation set from the training data.

    Parameters
    encoding : 'hotenc' or 'targetenc'

    Returns
    X_train, X_val, X_test : pd.DataFrame
    y_train, y_val, y_test : pd.Series  (log_price)
    """
    base = Path(processed_dir)
    train = pd.read_parquet(base / f'train_{encoding}.parquet')
    test  = pd.read_parquet(base / f'test_{encoding}.parquet')

    id_cols   = [c for c in ['id', 'price', 'log_price'] if c in train.columns]
    feat_cols = [c for c in train.columns if c not in id_cols]

    X_all  = train[feat_cols].copy()
    y_all  = train['log_price'].copy()
    X_test = test[feat_cols].copy()
    y_test = test['log_price'].copy()

    for df in [X_all, X_test]:
        bool_cols = df.columns[df.dtypes == bool]
        df[bool_cols] = df[bool_cols].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=val_size, random_state=random_state
    )

    print(f"Encoding: {encoding}")
    print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# 2. Metrics

def compute_metrics(y_true, y_pred_log) -> dict:
    """
    Compute regression metrics in EUR (after reversing the log transform).
    R² is computed on log-price.
    """
    y_true_eur = np.expm1(np.asarray(y_true))
    y_pred_eur = np.expm1(np.asarray(y_pred_log))

    rmse       = float(np.sqrt(mean_squared_error(y_true_eur, y_pred_eur)))
    mae        = float(mean_absolute_error(y_true_eur, y_pred_eur))
    r2         = float(r2_score(y_true, y_pred_log))
    mape       = float(np.mean(np.abs((y_true_eur - y_pred_eur) / y_true_eur)) * 100)
    within_20  = float(np.mean(np.abs(y_true_eur - y_pred_eur) / y_true_eur < 0.20) * 100)

    return {
        'rmse':        round(rmse, 2),
        'mae':         round(mae, 2),
        'r2':          round(r2, 3),
        'mape':        round(mape, 2),
        'within_20pct': round(within_20, 1),
    }


# 3. Cross-validation

def _cv_safe_clone(model):
    """
    Return an unfitted clone of model that can be .fit(X, y) inside CV
    without an eval_set
    """
    m = clone(model)
    params = m.get_params()

    if params.get('early_stopping_rounds') is not None:        # XGBoost tuned
        best_it = getattr(model, 'best_iteration', None)
        n = (int(best_it) + 1) if isinstance(best_it, (int, np.integer)) else params.get('n_estimators')
        m.set_params(early_stopping_rounds=None, n_estimators=max(int(n or 1), 1))
    elif getattr(model, 'best_iteration_', None):              # LightGBM tuned
        m.set_params(n_estimators=max(int(model.best_iteration_), 1))

    return m


def cross_validate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = CV_FOLDS,
    scale: bool = False,
) -> dict:
    """
    K-fold cross-validation
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = {k: [] for k in ['rmse', 'mae', 'r2', 'mape', 'within_20pct']}
    template = _cv_safe_clone(model)

    for train_idx, val_idx in kf.split(X):
        X_tr = X.iloc[train_idx].copy()
        X_vl = X.iloc[val_idx].copy()
        y_tr = y.iloc[train_idx]
        y_vl = y.iloc[val_idx]

        if scale:
            sc = StandardScaler()
            X_tr = pd.DataFrame(sc.fit_transform(X_tr), columns=X_tr.columns)
            X_vl = pd.DataFrame(sc.transform(X_vl),     columns=X_vl.columns)

        fold_model = clone(template)
        fold_model.fit(X_tr, y_tr)
        metrics = compute_metrics(y_vl, fold_model.predict(X_vl))
        for k, v in metrics.items():
            fold_scores[k].append(v)

    return {
        k: (round(float(np.mean(v)), 3), round(float(np.std(v)), 3))
        for k, v in fold_scores.items()
    }


#4. Full evaluation

def evaluate_model(
    name: str,
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val:   pd.DataFrame,
    y_val:   pd.Series,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
    all_results: list,
    scale: bool = False,
    cv: int = CV_FOLDS,
    skip_cv: bool = False,
):
    """
    Cross-validate on train, fit on full train, evaluate on val and test.
    Returns the fitted model.
    """
    print(f"\n{'='*52}")
    print(f"  {name}")
    print(f"{'='*52}")

    # Cross-validation
    if skip_cv:
        print("\n[Cross-Validation] skipped")
        cv_res = {k: ('—', '—') for k in ['rmse', 'mae', 'r2', 'mape', 'within_20pct']}
    else:
        print("\n[Cross-Validation]")
        cv_res = cross_validate_model(model, X_train, y_train, cv=cv, scale=scale)
        for metric, (mean, std) in cv_res.items():
            print(f"  {metric:15s}: {mean:.3f} ± {std:.3f}")

    # Final fit
    if scale:
        sc = StandardScaler()
        Xtr = pd.DataFrame(sc.fit_transform(X_train), columns=X_train.columns)
        Xvl = pd.DataFrame(sc.transform(X_val), columns=X_val.columns)
        Xte = pd.DataFrame(sc.transform(X_test), columns=X_test.columns)
    else:
        Xtr, Xvl, Xte = X_train, X_val, X_test

    if getattr(model, 'early_stopping_rounds', None):
        model.fit(Xtr, y_train, eval_set=[(Xvl, y_val)], verbose=False)
    else:
        model.fit(Xtr, y_train)

    #Validation
    print("\n[Validation Set]")
    val_m = compute_metrics(y_val, model.predict(Xvl))
    for k, v in val_m.items():
        print(f"  {k:15s}: {v}")

    #Test
    print("\n[Test Set]")
    test_m = compute_metrics(y_test, model.predict(Xte))
    for k, v in test_m.items():
        print(f"  {k:15s}: {v}")

    all_results.append({
        'model':   name,
        **{f'cv_{k}': '—' if v[0] == '—' else f"{v[0]}±{v[1]}"
           for k, v in cv_res.items()},
        **{f'val_{k}':  v for k, v in val_m.items()},
        **{f'test_{k}': v for k, v in test_m.items()},
    })

    return model


#5. Hyperparameter tuning

def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val:   pd.DataFrame,
    y_val:   pd.Series,
    n_iter:  int = 30,
) -> XGBRegressor:
    """
    Randomized search where each candidate is trained with early stopping on thevalidation set.
    """
    from sklearn.metrics import mean_squared_error

    param_dist = {
        'max_depth':        [3, 4, 5, 6, 8, 10],
        'learning_rate':    [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
        'subsample':        [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 1.0],
        'min_child_weight': [1, 3, 5, 7, 10],
        'reg_alpha':        [0, 0.01, 0.1, 0.5, 1.0],
        'reg_lambda':       [0.5, 1.0, 2.0, 5.0, 10.0],
        'gamma':            [0, 0.1, 0.3, 0.5],
    }

    rng = np.random.RandomState(RANDOM_STATE)
    MAX_ROUNDS, STOP = 2000, 50
    best_score, best_params, best_n = np.inf, None, MAX_ROUNDS

    print(f"  Random search ({n_iter} candidates, early stopping)...")
    for i in range(n_iter):
        params = {k: rng.choice(v).item() for k, v in param_dist.items()}
        model = XGBRegressor(
            **params,
            n_estimators=MAX_ROUNDS,
            tree_method='hist',
            early_stopping_rounds=STOP,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        n_trees = model.best_iteration + 1            # best_iteration is 0 indexed
        score   = np.sqrt(mean_squared_error(y_val, model.predict(X_val)))
        print(f"  [{i+1}/{n_iter}] RMSE={score:.4f}  trees={n_trees}  {params}")
        if score < best_score:
            best_score, best_params, best_n = score, params, n_trees

    print(f"  Best RMSE={best_score:.4f} @ {best_n} trees")
    print(f"  Best params: {best_params}")
    final = XGBRegressor(
        **best_params,
        n_estimators=MAX_ROUNDS,
        tree_method='hist',
        early_stopping_rounds=STOP,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    final.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"  Best iteration: {final.best_iteration}")
    final.tuned_params_   = {**best_params, 'n_estimators': int(final.best_iteration) + 1}
    final.tuned_val_rmse_ = float(best_score)
    return final


def tune_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val:   pd.DataFrame,
    y_val:   pd.Series,
    n_iter:  int = 30,
) -> LGBMRegressor:
    """
    Same early-stopping random search as tune_xgboost
    """
    import lightgbm as lgb
    from sklearn.metrics import mean_squared_error

    param_dist = {
        'num_leaves':        [15, 31, 63, 127],
        'max_depth':         [-1, 4, 6, 8, 12],
        'learning_rate':     [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
        'subsample':         [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree':  [0.5, 0.6, 0.7, 0.8, 1.0],
        'min_child_samples': [5, 10, 20, 50, 100],
        'reg_alpha':         [0, 0.01, 0.1, 0.5, 1.0],
        'reg_lambda':        [0.5, 1.0, 2.0, 5.0, 10.0],
    }

    rng = np.random.RandomState(RANDOM_STATE)
    MAX_ROUNDS, STOP = 2000, 50
    best_score, best_params, best_n = np.inf, None, MAX_ROUNDS

    print(f"  Random search ({n_iter} candidates, early stopping)...")
    for i in range(n_iter):
        params = {k: rng.choice(v).item() for k, v in param_dist.items()}
        model = LGBMRegressor(
            **params,
            n_estimators=MAX_ROUNDS,
            subsample_freq=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(STOP, verbose=False), lgb.log_evaluation(-1)],
        )
        n_trees = model.best_iteration_ or MAX_ROUNDS
        score   = np.sqrt(mean_squared_error(y_val, model.predict(X_val)))
        print(f"  [{i+1}/{n_iter}] RMSE={score:.4f}  trees={n_trees}  {params}")
        if score < best_score:
            best_score, best_params, best_n = score, params, n_trees

    print(f"  Best RMSE={best_score:.4f} @ {best_n} trees")
    print(f"  Best params: {best_params}")
    final = LGBMRegressor(
        **best_params,
        n_estimators=MAX_ROUNDS,
        subsample_freq=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    final.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(STOP, verbose=False), lgb.log_evaluation(-1)],
    )
    print(f"  Best iteration: {final.best_iteration_}")
    final.tuned_params_   = {**best_params, 'n_estimators': int(final.best_iteration_ or best_n)}
    final.tuned_val_rmse_ = float(best_score)
    return final


def tune_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val:   pd.DataFrame,
    y_val:   pd.Series,
    alphas=None,
) -> Ridge:
    """
    Grid search over the L2 penalty alpha on a standardised data.
    """
    if alphas is None:
        alphas = np.logspace(-3, 3, 25)

    sc   = StandardScaler().fit(X_train)
    Xtr  = sc.transform(X_train)
    Xvl  = sc.transform(X_val)

    best_score, best_alpha = np.inf, 1.0
    print(f"  Alpha search ({len(alphas)} values)...")
    for a in alphas:
        m     = Ridge(alpha=float(a), random_state=RANDOM_STATE).fit(Xtr, y_train)
        score = np.sqrt(mean_squared_error(y_val, m.predict(Xvl)))
        if score < best_score:
            best_score, best_alpha = score, float(a)

    print(f"  Best RMSE={best_score:.4f} @ alpha={best_alpha:.4g}")
    final = Ridge(alpha=best_alpha, random_state=RANDOM_STATE)
    final.tuned_params_   = {'alpha': best_alpha}
    final.tuned_val_rmse_ = float(best_score)
    return final


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val:   pd.DataFrame,
    y_val:   pd.Series,
    n_iter:  int = 20,
) -> RandomForestRegressor:
    """
    Randomized search scored on the validation set
    """
    param_dist = {
        'n_estimators':      [200, 300, 500, 800],
        'max_depth':         [None, 10, 15, 20, 30],
        'min_samples_leaf':  [1, 2, 5, 10],
        'min_samples_split': [2, 5, 10],
        'max_features':      ['sqrt', 'log2', 0.5, 1.0],
    }

    rng = np.random.RandomState(RANDOM_STATE)
    best_score, best_params = np.inf, None

    print(f"  Random search ({n_iter} candidates)...")
    for i in range(n_iter):
        params = {k: v[rng.randint(len(v))] for k, v in param_dist.items()}
        model = RandomForestRegressor(
            **params, random_state=RANDOM_STATE, n_jobs=-1,
        )
        model.fit(X_train, y_train)
        score = np.sqrt(mean_squared_error(y_val, model.predict(X_val)))
        print(f"  [{i+1}/{n_iter}] RMSE={score:.4f}  {params}")
        if score < best_score:
            best_score, best_params = score, params

    print(f"  Best RMSE={best_score:.4f}")
    print(f"  Best params: {best_params}")
    final = RandomForestRegressor(**best_params, random_state=RANDOM_STATE, n_jobs=-1)
    final.tuned_params_   = best_params
    final.tuned_val_rmse_ = float(best_score)
    return final


# 6. Stacking ensemble

def build_stacking(
    xgb_model:   XGBRegressor,
    lgbm_model:  LGBMRegressor,
    rf_model:    RandomForestRegressor,
    ridge_model: Ridge,
    passthrough: bool = True,
) -> StackingRegressor:
    """
    Build a stacking ensemble from the tuned RF, XGBoost,LightGBM with a tuned-Ridge meta-learner.
    """
    xgb_params  = xgb_model.get_params()
    lgbm_params = lgbm_model.get_params()
    for key in ['n_estimators', 'callbacks', 'early_stopping_rounds',
                'tree_method', 'random_state', 'n_jobs', 'verbose']:
        xgb_params.pop(key,  None)
        lgbm_params.pop(key, None)

    xgb_best  = getattr(xgb_model, 'best_iteration',  None)
    lgbm_best = getattr(lgbm_model,'best_iteration_', None)
    n_xgb  = (xgb_best + 1) if isinstance(xgb_best,  (int, np.integer)) and xgb_best  >= 0 else 400
    n_lgbm = lgbm_best  if isinstance(lgbm_best, (int, np.integer)) and lgbm_best >  0 else 400
    n_xgb, n_lgbm = max(int(n_xgb), 1), max(int(n_lgbm), 1)

    rf_base    = clone(rf_model)
    meta_alpha = getattr(ridge_model, 'tuned_params_', {}).get('alpha', 1.0)
    meta       = make_pipeline(StandardScaler(), Ridge(alpha=meta_alpha, random_state=RANDOM_STATE))

    return StackingRegressor(
        estimators=[
            ('rf',   rf_base),
            ('xgb',  XGBRegressor(
                        n_estimators=n_xgb,
                        tree_method='hist',
                        **xgb_params, random_state=RANDOM_STATE, n_jobs=-1)),
            ('lgbm', LGBMRegressor(
                        n_estimators=n_lgbm,
                        **lgbm_params, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)),
        ],
        final_estimator=meta,
        cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        passthrough=passthrough,
        n_jobs=-1,
    )


# 7. Plots

def plot_feature_importance(
    model,
    feature_names,
    title: str,
    top_n: int = 20,
    save_path=None,
):
    """Bar chart of the top_n feature importance"""
    importance = pd.Series(model.feature_importances_, index=feature_names)
    importance = importance.sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    importance.plot(kind='bar', ax=ax, color='steelblue')
    ax.set_title(title)
    ax.set_ylabel('Importance')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def plot_residuals(
    y_true,
    y_pred_log,
    title: str,
    save_path=None,
):
    """Residuals vspredicted and residual distribution."""
    residuals = np.asarray(y_true) - np.asarray(y_pred_log)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_pred_log, residuals, alpha=0.25, s=5, color='steelblue')
    axes[0].axhline(0, color='coral', linestyle='--')
    axes[0].set_xlabel('Predicted log price')
    axes[0].set_ylabel('Residual')
    axes[0].set_title(f'{title} — Residuals vs Predicted')

    axes[1].hist(residuals, bins=60, color='steelblue', edgecolor='none')
    axes[1].axvline(0, color='coral', linestyle='--')
    axes[1].set_xlabel('Residual')
    axes[1].set_title(f'{title} — Residual Distribution')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_model_comparison(results_df: pd.DataFrame, save_path=None):
    """Horizontal bar chart comparing test R2 and RMSE across all models."""
    df = results_df.copy()
    df = df.sort_values('test_r2', ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(df) * 0.5 + 1)))

    axes[0].barh(df['model'], df['test_r2'], color='steelblue')
    axes[0].set_xlabel('Test R²')
    axes[0].set_title('Test R² (higher is better)')
    axes[0].axvline(df['test_r2'].max(), color='coral', linestyle='--', alpha=0.6)

    axes[1].barh(df['model'], df['test_rmse'], color='salmon')
    axes[1].set_xlabel('Test RMSE (€)')
    axes[1].set_title('Test RMSE (lower is better)')
    axes[1].axvline(df['test_rmse'].min(), color='steelblue', linestyle='--', alpha=0.6)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
