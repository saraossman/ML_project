"""
Model training, evaluation, and plotting utilities.

All functions are pure (no global state). Data is passed explicitly so cells
can run in any order without side-effects.

Fixes vs. the original notebook
---------------------------------
* Double-scaling bug: cross_validate_model always receives raw (unscaled) data
  and handles scaling per-fold internally. evaluate_model no longer pre-scales
  before calling cross_validate_model.
* all_results is passed as an explicit argument, not a global list.
* clean_col_names / price-cap / split are all done in the feature pipeline;
  load_data() just reads the parquet files and does a val split from train.
* XGBoost and LightGBM are tuned with an expanded grid and then refitted with
  early stopping on the validation set.
* Stacking uses the tuned base-learner params, not hand-picked ones.
* SHAP summary plot added for the best model (requires `pip install shap`).
* Simple MLP removed; only the Residual MLP is kept.
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
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

RANDOM_STATE = 42
CV_FOLDS     = 5
VAL_SIZE     = 0.15   # fraction of train set held out as validation
FIGURES_DIR  = Path('outputs/figures')
MODELS_DIR   = Path('outputs/models')


# ─── 1. Data loading ──────────────────────────────────────────────────────────

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
    ----------
    encoding : 'hotenc' or 'targetenc'

    Returns
    -------
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

    # coerce any leftover bool columns to int
    for df in [X_all, X_test]:
        bool_cols = df.columns[df.dtypes == bool]
        df[bool_cols] = df[bool_cols].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=val_size, random_state=random_state
    )

    print(f"Encoding: {encoding}")
    print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ─── 2. Metrics ───────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred_log) -> dict:
    """
    Compute regression metrics in EUR space (after reversing the log transform).
    R² is computed on log-price to be scale-invariant for model comparison.
    """
    y_true_eur = np.expm1(np.asarray(y_true))
    y_pred_eur = np.expm1(np.asarray(y_pred_log))

    rmse       = float(np.sqrt(mean_squared_error(y_true_eur, y_pred_eur)))
    mae        = float(mean_absolute_error(y_true_eur, y_pred_eur))
    r2         = float(r2_score(y_true, y_pred_log))
    mape       = float(np.mean(np.abs((y_true_eur - y_pred_eur) / y_true_eur)) * 100)
    within_20  = float(
        np.mean(np.abs(y_true_eur - y_pred_eur) / y_true_eur < 0.20) * 100
    )

    return {
        'rmse':        round(rmse, 2),
        'mae':         round(mae, 2),
        'r2':          round(r2, 3),
        'mape':        round(mape, 2),
        'within_20pct': round(within_20, 1),
    }


# ─── 3. Cross-validation (fixed: no double-scaling) ──────────────────────────

def cross_validate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = CV_FOLDS,
    scale: bool = False,
) -> dict:
    """
    K-fold cross-validation. Always receives *raw* (unscaled) data.
    Scaling is applied per-fold internally when scale=True, preventing
    data leakage between folds and avoiding the double-scaling bug in
    the original notebook.

    Returns dict of {metric: (mean, std)}.
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = {k: [] for k in ['rmse', 'mae', 'r2', 'mape', 'within_20pct']}

    for train_idx, val_idx in kf.split(X):
        X_tr = X.iloc[train_idx].copy()
        X_vl = X.iloc[val_idx].copy()
        y_tr = y.iloc[train_idx]
        y_vl = y.iloc[val_idx]

        if scale:
            sc = StandardScaler()
            X_tr = pd.DataFrame(sc.fit_transform(X_tr), columns=X_tr.columns)
            X_vl = pd.DataFrame(sc.transform(X_vl),     columns=X_vl.columns)

        model.fit(X_tr, y_tr)
        metrics = compute_metrics(y_vl, model.predict(X_vl))
        for k, v in metrics.items():
            fold_scores[k].append(v)

    return {
        k: (round(float(np.mean(v)), 3), round(float(np.std(v)), 3))
        for k, v in fold_scores.items()
    }


# ─── 4. Full evaluation (fixed: explicit args, no globals) ───────────────────

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

    Fixes vs. original
    ------------------
    - No globals: all data passed explicitly.
    - CV receives raw X_train regardless of scale flag; scaling is done
      per-fold inside cross_validate_model.
    - Final fit uses a fresh scaler fitted only on X_train.
    """
    print(f"\n{'='*52}")
    print(f"  {name}")
    print(f"{'='*52}")

    # ── Cross-validation (always raw data in, scale inside) ───────────────
    if skip_cv:
        print("\n[Cross-Validation] skipped")
        cv_res = {k: ('—', '—') for k in ['rmse', 'mae', 'r2', 'mape', 'within_20pct']}
    else:
        print("\n[Cross-Validation]")
        cv_res = cross_validate_model(model, X_train, y_train, cv=cv, scale=scale)
        for metric, (mean, std) in cv_res.items():
            print(f"  {metric:15s}: {mean:.3f} ± {std:.3f}")

    # ── Final fit ─────────────────────────────────────────────────────────
    if scale:
        sc = StandardScaler()
        Xtr = pd.DataFrame(sc.fit_transform(X_train), columns=X_train.columns)
        Xvl = pd.DataFrame(sc.transform(X_val),       columns=X_val.columns)
        Xte = pd.DataFrame(sc.transform(X_test),      columns=X_test.columns)
    else:
        Xtr, Xvl, Xte = X_train, X_val, X_test

    model.fit(Xtr, y_train)

    # ── Validation ────────────────────────────────────────────────────────
    print("\n[Validation Set]")
    val_m = compute_metrics(y_val, model.predict(Xvl))
    for k, v in val_m.items():
        print(f"  {k:15s}: {v}")

    # ── Test ──────────────────────────────────────────────────────────────
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


# ─── 5. Hyperparameter tuning with early stopping ────────────────────────────

def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val:   pd.DataFrame,
    y_val:   pd.Series,
    n_iter:  int = 8,
) -> XGBRegressor:
    """
    Random search on val set. Plain fit (no early stopping, no eval_set) per
    candidate to avoid XGBoost threading/verbosity quirks across versions.
    One final refit with early stopping after the best params are found.
    """
    from sklearn.metrics import mean_squared_error

    param_dist = {
        'max_depth':        [4, 6, 8],
        'learning_rate':    [0.03, 0.05, 0.1, 0.15],
        'subsample':        [0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'min_child_weight': [1, 3, 5],
        'reg_alpha':        [0, 0.1, 1.0],
        'reg_lambda':       [1.0, 2.0, 5.0],
    }

    rng = np.random.RandomState(RANDOM_STATE)
    best_score = np.inf
    best_params = None

    print(f"  Random search ({n_iter} candidates)...")
    for i in range(n_iter):
        params = {k: rng.choice(v).item() for k, v in param_dist.items()}
        model = XGBRegressor(
            **params,
            n_estimators=150,
            tree_method='hist',
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        score = mean_squared_error(y_val, model.predict(X_val))
        print(f"  [{i+1}/{n_iter}] RMSE={np.sqrt(score):.4f}  {params}")
        if score < best_score:
            best_score = score
            best_params = params

    print(f"  Best params: {best_params}")
    print(f"  Refitting with early stopping...")
    final = XGBRegressor(
        **best_params,
        n_estimators=500,
        tree_method='hist',
        early_stopping_rounds=30,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    final.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"  Best iteration: {final.best_iteration}")
    return final


def tune_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val:   pd.DataFrame,
    y_val:   pd.Series,
    n_iter:  int = 8,
) -> LGBMRegressor:
    """
    Same plain-fit random search as tune_xgboost, then one refit with early stopping.
    """
    import lightgbm as lgb
    from sklearn.metrics import mean_squared_error

    param_dist = {
        'max_depth':         [4, 6, 8],
        'learning_rate':     [0.03, 0.05, 0.1, 0.15],
        'subsample':         [0.7, 0.8, 0.9],
        'colsample_bytree':  [0.6, 0.8, 1.0],
        'min_child_samples': [10, 20, 50],
        'reg_alpha':         [0, 0.1, 1.0],
        'reg_lambda':        [1.0, 2.0, 5.0],
    }

    rng = np.random.RandomState(RANDOM_STATE)
    best_score = np.inf
    best_params = None

    print(f"  Random search ({n_iter} candidates)...")
    for i in range(n_iter):
        params = {k: rng.choice(v).item() for k, v in param_dist.items()}
        model = LGBMRegressor(
            **params,
            n_estimators=150,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
        model.fit(X_train, y_train)
        score = mean_squared_error(y_val, model.predict(X_val))
        print(f"  [{i+1}/{n_iter}] RMSE={np.sqrt(score):.4f}  {params}")
        if score < best_score:
            best_score = score
            best_params = params

    print(f"  Best params: {best_params}")
    print(f"  Refitting with early stopping...")
    final = LGBMRegressor(
        **best_params,
        n_estimators=500,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    final.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
    )
    print(f"  Best iteration: {final.best_iteration_}")
    return final


# ─── 6. Stacking ensemble (uses tuned params) ─────────────────────────────────

def build_stacking(
    xgb_model: XGBRegressor,
    lgbm_model: LGBMRegressor,
) -> StackingRegressor:
    """
    Build a stacking ensemble using the tuned XGBoost and LightGBM models
    as base learners (via their best params, not the fitted objects).
    RF is included as a diversity-adding base learner.
    Meta-learner is Ridge regression.
    """
    xgb_params  = xgb_model.get_params()
    lgbm_params = lgbm_model.get_params()

    # remove early-stopping bookkeeping keys that can't be passed to constructor
    for key in ['n_estimators', 'callbacks']:
        xgb_params.pop(key,  None)
        lgbm_params.pop(key, None)

    return StackingRegressor(
        estimators=[
            ('rf',   RandomForestRegressor(
                        n_estimators=200, max_depth=15,
                        min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1)),
            ('xgb',  XGBRegressor(
                        n_estimators=xgb_model.best_iteration or 400,
                        tree_method='hist',
                        **xgb_params, random_state=RANDOM_STATE, n_jobs=-1)),
            ('lgbm', LGBMRegressor(
                        n_estimators=getattr(lgbm_model, 'best_iteration_', 400),
                        **lgbm_params, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)),
        ],
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        n_jobs=-1,
    )


# ─── 7. Plots ─────────────────────────────────────────────────────────────────

def plot_feature_importance(
    model,
    feature_names,
    title: str,
    top_n: int = 20,
    save_path=None,
):
    """Bar chart of the top_n MDI feature importances."""
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


def plot_shap(model, X: pd.DataFrame, title: str, save_path=None):
    """
    SHAP summary plot (beeswarm). Shows direction and magnitude of each
    feature's effect — more informative than MDI importance alone.
    Requires: pip install shap
    """
    if not HAS_SHAP:
        print("SHAP not installed. Run: pip install shap")
        return

    print(f"  Computing SHAP values for {title}...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_residuals(
    y_true,
    y_pred_log,
    title: str,
    save_path=None,
):
    """Residuals-vs-predicted and residual distribution side-by-side."""
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
    """Horizontal bar chart comparing test R² and RMSE across all models."""
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


# ─── 8. Residual MLP ──────────────────────────────────────────────────────────

class ResidualMLP(
    __import__('torch').nn.Module
):
    """
    Two-block residual MLP for tabular price prediction.
    Architecture: input_proj → [ResBlock × 2] → output
    Each ResBlock: Linear → BN → ReLU → Dropout → Linear → BN, plus skip.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        import torch.nn as nn
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        def _block():
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
            )

        self.block1 = _block()
        self.block2 = _block()
        self.relu   = nn.ReLU()
        self.output = nn.Linear(hidden_dim, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.relu(self.input_proj(x))
        x = self.relu(x + self.block1(x))
        x = self.relu(x + self.block2(x))
        return self.output(x).squeeze(1)


def train_residual_mlp(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val:   pd.DataFrame,
    y_val:   pd.Series,
    hidden_dim: int = 256,
    dropout:    float = 0.2,
    epochs:     int = 200,
    lr:         float = 1e-3,
    patience:   int = 25,
    batch_size: int = 256,
    save_path=None,
):
    """
    Train the ResidualMLP and return the fitted model + training history.
    Scales features internally. Uses HuberLoss + AdamW + CosineAnnealingLR.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Training ResidualMLP on {device}  "
          f"(hidden={hidden_dim}, dropout={dropout})")

    sc = StandardScaler()
    Xtr = sc.fit_transform(X_train)
    Xvl = sc.transform(X_val)

    def _tensors(X, y):
        return (torch.FloatTensor(X),
                torch.FloatTensor(y.values if hasattr(y, 'values') else y))

    Xtr_t, ytr_t = _tensors(Xtr, y_train)
    Xvl_t, yvl_t = _tensors(Xvl, y_val)

    loader = DataLoader(
        TensorDataset(Xtr_t, ytr_t),
        batch_size=batch_size, shuffle=True,
    )

    model     = ResidualMLP(Xtr.shape[1], hidden_dim, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=1e-5
    )
    criterion = nn.HuberLoss(delta=1.0)

    Xvl_dev = Xvl_t.to(device)
    yvl_dev = yvl_t.to(device)

    best_val_loss  = float('inf')
    best_state     = None
    patience_count = 0
    history        = {'train': [], 'val': []}

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xvl_dev), yvl_dev).item()

        train_loss = float(np.mean(train_losses))
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if epoch % 25 == 0:
            print(f"    Epoch {epoch:3d} | train={train_loss:.4f} | "
                  f"val={val_loss:.4f} | patience={patience_count}/{patience}")

        if patience_count >= patience:
            print(f"    Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)

    # ── Training curve ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history['train'], label='Train loss', color='steelblue')
    ax.plot(history['val'],   label='Val loss',   color='coral')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Huber Loss')
    ax.set_title('Residual MLP — Training Curve')
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

    return model, sc, history


def predict_mlp(model, scaler, X: pd.DataFrame) -> np.ndarray:
    """Run inference with a trained ResidualMLP."""
    import torch
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_scaled = scaler.transform(X)
        return model(
            torch.FloatTensor(X_scaled).to(device)
        ).cpu().numpy()
