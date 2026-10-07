# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import common as common

# %%
class LSTMModel(nn.Module):
  def __init__(self, n_features, n_ventana):
    super(LSTMModel, self).__init__()

    self.lstm1 = nn.LSTM(
      input_size=n_features,
      hidden_size=100,
      batch_first=True,
      num_layers=1
    )

    self.lstm2 = nn.LSTM(
      input_size=100,
      hidden_size=50,
      batch_first=True,
      num_layers=1
    )

    self.fc1 = nn.Linear(50, 1)

  def forward(self, x):
    out, _ = self.lstm1(x)
    out, _ = self.lstm2(out)
    out = out[:, -1, :]
    out = self.fc1(out)
    return out

# %%
def train(obj: common.TrainDataCommon, product: str, verbose=1, epochs = 50, n_ventana=15):
  # Preparar datos para LSTM
  x_train, y_train = obj.get_train_data_LSTM()
  n_features = x_train.shape[2]

  # Convertir a tensores PyTorch
  x_train = torch.tensor(x_train, dtype=torch.float32)
  y_train = torch.tensor(y_train, dtype=torch.float32)

  # Dataset + dataloader
  dataset = TensorDataset(x_train, y_train)
  loader = DataLoader(dataset, batch_size=32, shuffle=True)
  
  # Crear modelo
  model = LSTMModel(n_features, n_ventana)
  if verbose == 2:
    for name, module in model.named_children():
      params = sum(p.numel() for p in module.parameters() if p.requires_grad)
      print(f"{name}: {params}")

  # Loss + optimizador
  criterion = nn.SmoothL1Loss()
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5
  )

  # Early stopping
  best_loss = float('inf')
  epochs_no_improve = 0
  patience = 10

  # Usar GPU si está disponible
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)

  # Entrenamiento
  for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch_x, batch_y in loader:
      batch_x = batch_x.to(device)
      batch_y = batch_y.to(device)

      optimizer.zero_grad()
      output = model(batch_x)
      loss = criterion(output.squeeze(), batch_y)
      loss.backward()

      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()

      total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)

    # Scheduler
    scheduler.step(avg_loss)

    # Early stopping
    if avg_loss < best_loss:
      best_loss = avg_loss
      epochs_no_improve = 0
      torch.save(model.state_dict(), f"modelos_clasico/{product}.pt")
    else:
      epochs_no_improve += 1
    
    if verbose == 1:
      lr = optimizer.param_groups[0]['lr']
      print(
          f"Epoch {epoch+1}/{epochs} | "
          f"Loss: {avg_loss:.4f} | "
          f"LR: {lr:.6f}"
      )
    
    if epochs_no_improve >= patience:
      if verbose == 1:
          print(f"⏹ Early stopping en epoch {epoch+1}")
      break
  
  # Cargar el mejor modelo guardado
  model.load_state_dict(torch.load(f"modelos_clasico/{product}.pt"))

  return model

# %%
def excute(products, verbose=0, epochs=50, n_ventana=15, n_pred=1):
  result =  pd.DataFrame(columns=['product', 'r2', 'pearson'])

  for product in products:
    if verbose == 1:
      print('### Producto:', product)
    df = all_sales[(all_sales['product'] == product)].copy()

    if df.empty:
      if verbose == 1:
        print('   No hay datos para este producto.')
      continue
    
    df = common.fix_dates(df, product, verbose)
    days_train = max(365 + n_ventana + n_pred - 1, len(df) - 365)
    days_test = min(max(30 + n_pred - 1, len(df) - days_train), 365)

    if len(df) < days_train + days_test:
      if verbose == 1:
        print('   No hay suficientes datos para entrenar y probar.')
      continue

    scaler_sales = StandardScaler()
    train_data = common.process_common(df, days_train, days_test, scaler_sales)
    model = train(train_data, product, verbose, epochs, n_ventana)

    (r2, pearson) = common.test_common(model, train_data, days_train, days_test, verbose)
    if verbose >= 1:
      print('   R^2:', r2)
      print('   pearson:', pearson)

    result.loc[len(result)] = [product, r2, pearson]

  return result

# %%

# Sample Sales Data - French bakery daily sales
# https://www.kaggle.com/datasets/matthieugimbert/french-bakery-daily-sales/data

all_sales = pd.read_csv('sales.csv')
all_sales = all_sales.rename(columns={'article': 'product', 'Quantity': 'sales'})
all_sales = all_sales[['date', 'product', 'sales']]
all_sales = all_sales.groupby(['product', 'date'], as_index=False)['sales'].sum()
'''
Top 20 most sold products

TRADITIONAL BAGUETTE    117463.0
CROISSANT                29654.0
PAIN AU CHOCOLAT         25236.0
COUPE                    23505.0
BANETTE                  22732.0
BAGUETTE                 22053.0
CEREAL BAGUETTE           7427.0
SPECIAL BREAD             5456.0
FORMULE SANDWICH          5181.0
TARTELETTE                5020.0
BOULE 400G                4824.0
CAMPAGNE                  4356.0
COOKIE                    3779.0
ECLAIR                    3654.0
VIK BREAD                 3619.0
COMPLET                   3535.0
FICELLE                   3405.0
MOISSON                   3362.0
BANETTINE                 3092.0
BOULE 200G                3080.0
'''

# required columns: date, product, sales
# date column as datetime
all_sales['date'] = pd.to_datetime(all_sales['date'])

print(all_sales['product'].unique())
len(all_sales)

# %%
product_sales = all_sales.groupby('product')['sales'].sum()
top_20_products = product_sales.sort_values(ascending=False).head(20)
print("First 20 products by sales:")
print(top_20_products)

# %%
#result = 
excute(
  [
  #'PAIN AU CHOCOLAT',
  #'BAGUETTE',
  #'CAMPAGNE',
  #'ECLAIR',
  'VIK BREAD',
  #'FICELLE',
  ]
, epochs=25
, verbose=2
, n_ventana=15
, n_pred=1
)

# %%
