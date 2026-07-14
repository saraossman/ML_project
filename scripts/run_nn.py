#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader



# In[2]:


try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

RANDOM_STATE = 42
CV_FOLDS     = 5
VAL_SIZE     = 0.15   # fraction validation
FIGURES_DIR  = Path('.')
MODELS_DIR   = Path('.')


# In[3]:


def load_data(
    encoding: str = 'hotenc',
    processed_dir: str = '.',
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

    for df in [train, test]:
        if 'log_price' not in df.columns and 'price' in df.columns:
            df['log_price'] = np.log1p(df['price'])
        elif 'log_price' not in df.columns and 'price_eur' in df.columns:
            df['log_price'] = np.log1p(df['price_eur'])

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


# In[4]:


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


# In[6]:


# Call the function to actually create X_train, X_val, X_test, etc.
X_train, X_val, X_test, y_train, y_val, y_test = load_data()

torch.manual_seed(RANDOM_STATE)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

torch.manual_seed(RANDOM_STATE)

# Convert scaled arrays into PyTorch float tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_val_tensor   = torch.tensor(X_val_scaled, dtype=torch.float32)
X_test_tensor  = torch.tensor(X_test_scaled, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train.values.astype(np.float32), dtype=torch.float32).unsqueeze(1)
y_val_tensor   = torch.tensor(y_val.values.astype(np.float32), dtype=torch.float32).unsqueeze(1)

# Package into DataLoader
train_dataset  = TensorDataset(X_train_tensor, y_train_tensor)
train_loader   = DataLoader(train_dataset, batch_size=64, shuffle=True)



# In[7]:


class AirbnbPriceRegressor(nn.Module):
    def __init__(self, input_dim):
        super(AirbnbPriceRegressor, self).__init__()

        # Layer 1: Input features -> 128 neurons
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU() 
        self.dropout1 = nn.Dropout(0.2) # 20% dropout to prevent overfitting

        # Layer 2: 128 neurons -> 64 neurons
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)

        # Layer 3: Output layer (64 neurons -> 1 price prediction)
        self.output_layer = nn.Linear(64, 1)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        out = self.output_layer(x)
        return out

# Initialize the model with the exact working features count
input_features_count = X_train.shape[1]
model = AirbnbPriceRegressor(input_dim=input_features_count)

# Define Loss and use Adamax without Batch Norm getting in its way
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001) 
print(model)


# In[8]:


# 1. Initialize using your scaled train feature count (277)
input_features_count = X_train_tensor.shape[1]
model = AirbnbPriceRegressor(input_dim=input_features_count)

# 2. Define standard MSE Loss and standard Adam with 0.001 learning rate
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 3. Training Configurations
epochs = 100
train_losses = []
val_losses = []

print("Starting Neural Network Training...")
print(f"{'Epoch':<10}{'Train Loss (MSE)':<20}{'Val Loss (MSE)':<20}")
print("-" * 50)

# 4. The Deep Learning Training Loop
for epoch in range(epochs):
    model.train()  # Put model in training mode (enables dropout)
    running_loss = 0.0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()               # Clear past gradients
        predictions = model(batch_X)         # Forward pass: make predictions
        loss = criterion(predictions, batch_y) # Calculate training batch loss
        loss.backward()                     # Backward pass: calculate gradients
        optimizer.step()                    # Update model weights

        running_loss += loss.item() * batch_X.size(0)

    epoch_train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)

    # 5. Evaluate on Validation Set (No training/learning happens here)
    model.eval()  # Put model in evaluation mode (disables dropout)
    with torch.no_grad():
        val_preds = model(X_val_tensor)
        epoch_val_loss = criterion(val_preds, y_val_tensor).item()
        val_losses.append(epoch_val_loss)

    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"{epoch+1:<10}{epoch_train_loss:<20.4f}{epoch_val_loss:<20.4f}")

print("\nTraining Complete!")

# 6. Plot the Training Loss Curve (Perfect for your presentation slides!)
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error Loss')
plt.title('Neural Network Training History')
plt.legend()
plt.grid(True)
plt.show()

# 7. FINAL EVALUATION: Grade the Network on your hidden Validation and Test sets
model.eval()
with torch.no_grad():
    # Get raw tensor predictions and convert them back to standard NumPy arrays
    val_predictions_log = model(X_val_tensor).numpy().flatten()
    test_predictions_log = model(X_test_tensor).numpy().flatten()

print("\n" + "="*40)
print("       NEURAL NETWORK FINAL RESULTS")
print("="*40)

print("\n[Validation Set Metrics]")
val_scores = compute_metrics(y_val, val_predictions_log)
for metric, score in val_scores.items():
    print(f"  {metric:15s}: {score}")

print("\n[Test Set Metrics]")
test_scores = compute_metrics(y_test, test_predictions_log)
for metric, score in test_scores.items():
    print(f"  {metric:15s}: {score}")

