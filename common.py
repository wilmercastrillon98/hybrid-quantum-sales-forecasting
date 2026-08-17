# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import calendar
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# %%
def fechas_faltantes(df, start='2013-01-01', end='2017-08-15', verbose=1):
    df = df.copy()
    #df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    full = pd.date_range(start=start, end=end, freq='D')
    presentes = pd.to_datetime(df['date'].dt.normalize().dropna().unique())
    presentes = pd.DatetimeIndex(sorted(presentes))
    faltantes = full.difference(presentes)
    if verbose == 1:
      resumen = {
          'start': start,
          'end': end,
          'expected_days': len(full),
          'present_days': len(presentes),
          'missing_days': len(faltantes),
          'missing_dates': list(faltantes[:10].strftime('%Y-%m-%d'))
      }
      print(resumen)
    return faltantes

def add_missing_days(df, faltantes, family, sales_value=0):
    if not isinstance(faltantes, pd.DatetimeIndex):
        faltantes = pd.to_datetime(faltantes)
    faltantes = faltantes.normalize()
    date_col = 'date'

    new_rows = pd.DataFrame({
        'date': faltantes,
        'product': family,
        'sales': sales_value
    })

    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce').dt.normalize()

    combined = pd.concat([df_copy, new_rows], ignore_index=True, sort=False)
    combined[date_col] = pd.to_datetime(combined[date_col]).dt.normalize()
    combined = combined.sort_values(by=date_col).reset_index(drop=True)
    return combined

def fix_dates(df, product, verbose=1):
  min_date = df['date'].min()
  max_date = df['date'].max()
  faltantes = fechas_faltantes(df, min_date, max_date, verbose)
  return add_missing_days(df, faltantes, product)

# %%
def r2(df_resultado):
  ss_res = ((df_resultado['Real'] - df_resultado['Predicción']) ** 2).sum()
  ss_tot = ((df_resultado['Real'] - df_resultado['Real'].mean()) ** 2).sum()
  r2 = 1 - ss_res / ss_tot
  #print('R^2:', r2)
  return r2
  
def pearson(df_resultado):
  corr = np.corrcoef(df_resultado['Real'], df_resultado['Predicción'])
  #print('pearson:', corr[0, 1])
  return corr[0, 1]

# %%
def test_common(model, train_data, offset, days_test, verbose=1, n_ventana=15, n_pred=1):
  train_data.build_test(offset, days_test)
  features_test = np.column_stack((train_data.test_days, train_data.dow_sin_test, train_data.dow_cos_test))
  x_test, y_test = split_data(train_data.test_days, features_test)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
  y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

  model.eval()
  x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
  with torch.no_grad():
      preds_tensor = model(x_test_tensor)
  predictions_scaled = preds_tensor.cpu().numpy().reshape(-1, 1)
  # predictions_scaled es un vector columna y cada fila es un vector fila de n_pred elementos
  predictions = train_data.descale_sales(predictions_scaled)

  y_test_real = []
  for i in range(len(train_data.test_days_original) - n_ventana - n_pred + 1):
    y_test_real.append(np.sum(train_data.test_days_original[i + n_ventana:i + n_ventana + n_pred]))
  y_test_real = np.array(y_test_real)

  df_resultado = pd.DataFrame({
      'Real': y_test_real,
      'Predicción': predictions
  })

  if verbose > 0:
    plt.figure(figsize=(12, 6))
    plt.plot(df_resultado['Real'].values, label='Real', color='blue')
    plt.plot(df_resultado['Predicción'].values, label='Predicción', color='orange')
    plt.title('Comparación de valores reales y predichos')
    plt.xlabel('Día')
    plt.ylabel('Ventas')
    plt.legend()
    plt.grid(True)
    plt.show()

  r2_value = r2(df_resultado)
  pearson_value = pearson(df_resultado)
  return (r2_value, pearson_value)

# ejemplo
# n_ventana = 5
# n_pred = 2
# data = [1, 2, 3 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# X = [ ([1, 2, 3 4, 5], sen, cos), ([2, 3, 4, 5, 6], sen, cos), ... ]
# y = [suma([6, 7]), suma([7, 8]), ... ]
# Para tener N datos se necesitan N + n_ventana + n_pred - 1 datos en data

def split_data(data, features, n_ventana=15, n_pred=1):
  X, y = [], []

  for i in range(len(data) - n_ventana - n_pred + 1):
      ventana_features = features[i:i+n_ventana]
      suma = np.sum(data[i + n_ventana:i + n_ventana + n_pred])
      X.append(ventana_features)
      y.append(suma)

  X = np.array(X)
  y = np.array(y)
  return X, y

# %%
class TrainDataCommon:
  def __init__(self, df, scaler_sales=MinMaxScaler()):
    self.df = df
    self.sales = df['sales'].values
    self.dow_sin = df['dow_sin'].values
    self.dow_cos = df['dow_cos'].values
    self.scale_data(scaler_sales)

  def scale_data(self, scaler_sales):
    self.df['sales_scaled'] = self.df['sales']

    #scaler_sales = MinMaxScaler(feature_range=(-np.pi, np.pi))
    self.df['sales_scaled'] = scaler_sales.fit_transform(self.df[['sales_scaled']])
    self.scaler_sales = scaler_sales
    self.sales_scaled = self.df['sales_scaled'].values
    return self.sales_scaled
  
  # devuelve un arreglo normal
  def descale_sales(self, predictions_scaled):
    pred_log = self.scaler_sales.inverse_transform(predictions_scaled).flatten()
    return pred_log

  def build(self, days_train, days_test, n_ventana=15, n_pred=1):
    # Preparar datos de entrenamiento
    train_days = self.sales_scaled[:days_train]
    dow_sin_train = self.dow_sin[:days_train]
    dow_cos_train = self.dow_cos[:days_train]
    self.train_days = train_days
    self.dow_sin_train = dow_sin_train
    self.dow_cos_train = dow_cos_train

    # Preparar datos de prueba
    test_days_original = self.sales[days_train-n_ventana:days_train+days_test]
    test_days = self.sales_scaled[days_train-n_ventana:days_train+days_test]
    dow_sin_test = self.dow_sin[days_train-n_ventana:days_train+days_test]
    dow_cos_test = self.dow_cos[days_train-n_ventana:days_train+days_test]
    self.test_days_original = test_days_original
    self.test_days = test_days
    self.dow_sin_test = dow_sin_test
    self.dow_cos_test = dow_cos_test

  def get_train_data_LSTM(self):
    features_train = np.column_stack((self.train_days, self.dow_sin_train, self.dow_cos_train))
    x_train, y_train = split_data(self.train_days, features_train)
    return x_train, y_train
  
  def build_test(self, offset=0, days_test = 30):
    test_days_original = self.sales[offset:offset+days_test]
    test_days = self.sales_scaled[offset:offset+days_test]
    dow_sin_test = self.dow_sin[offset:offset+days_test]
    dow_cos_test = self.dow_cos[offset:offset+days_test]
    self.test_days_original = test_days_original
    self.test_days = test_days
    self.dow_sin_test = dow_sin_test
    self.dow_cos_test = dow_cos_test

# %%
def process_common(df, days_train, days_test, scaler_sales=MinMaxScaler()) -> TrainDataCommon:
  # Crear variables de día de la semana
  df['dow'] = df['date'].dt.dayofweek
  df['dow_sin'] = np.sin(2 * np.pi * df['dow']/7) * np.pi
  df['dow_cos'] = np.cos(2 * np.pi * df['dow']/7) * np.pi

  obj = TrainDataCommon(df, scaler_sales)
  obj.build(days_train, days_test)
  return obj

# %%
import pennylane as qml
from pennylane.qnn import TorchLayer
import torch
import numpy as np
#import random

#semilla = random.randint(1, 10000)
#torch.manual_seed(semilla)
#np.random.seed(semilla)
#random.seed(semilla)
#qml.numpy.random.seed(semilla)

# %%
def train_quantum_common(
  model: nn.Module,
  obj: TrainDataCommon,
  product: str,
  folder: str,
  verbose=1,
  epochs = 50,
  patience = 10,
  lr = 0.001,
):
  # Preparar datos para LSTM
  x_train, y_train = obj.get_train_data_LSTM()
  #n_features = x_train.shape[2]

  os.makedirs(folder, exist_ok=True)
  best_path = f"{folder}/{product}_best.pt"

  # Convertir a tensores PyTorch
  x_train = torch.tensor(x_train, dtype=torch.float32)
  y_train = torch.tensor(y_train, dtype=torch.float32)

  # Dataset + dataloader
  dataset = TensorDataset(x_train, y_train)
  loader = DataLoader(dataset, batch_size=32, shuffle=True)
  
  # === CAMBIO IMPORTANTE: Mover el modelo ANTES de crear el optimizador ===
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)

  # Loss + optimizador
  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=patience//2
  )

  # Early stopping
  best_loss = float('inf')
  epochs_no_improve = 0
  logs = {}

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
        optimizer.step()

        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)

    # Scheduler
    scheduler.step(avg_loss)

    # Early stopping
    if avg_loss < best_loss:
      best_loss = avg_loss
      epochs_no_improve = 0
      torch.save(model.state_dict(), best_path)
    else:
      epochs_no_improve += 1
    
    if verbose > 1:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

    logs[epoch] = {
      'loss': avg_loss,
      'lr': optimizer.param_groups[0]['lr'],
    }

    if epochs_no_improve >= patience:
      if verbose > 1:
          print(f"⏹ Early stopping en epoch {epoch+1}")
      break
  
  if os.path.isfile(best_path):
    model.load_state_dict(torch.load(best_path, map_location=device))

  #if verbose > 0:
  #    print(model)
  torch.save(model.state_dict(), f"{folder}/{product}.pt")
  pd.DataFrame.from_dict(logs, orient='index').rename_axis('epoch').reset_index().to_csv(
    f"{folder}/{product}_logs.csv", index=False
  )
  return model

# %%
#verbose: 0 - sin salida, 1 - con gráficas, 2 - gráficas y entrenamiento detallado
def excute_quantum_common(
  all_sales: pd.DataFrame,
  products: list,
  model: nn.Module,
  folder: str,
  scaler_sales=MinMaxScaler(feature_range=(-np.pi, np.pi)),
  verbose=0,
  epochs = 50,
  patience = 10,
  lr=0.001,
  n_ventana=15,
  n_pred=1,
):
  result =  pd.DataFrame(columns=['product', 'r2', 'pearson'])
  for product in products:
    if verbose > 0:
      print('### Producto:', product)
    df = all_sales[(all_sales['product'] == product)].copy()

    if df.empty:
      if verbose > 0:
        print('   No hay datos para este producto.')
      continue
    
    df = fix_dates(df, product, verbose)
    days_train = max(365 + n_ventana + n_pred - 1, len(df) - 365)
    days_test = min(max(30 + n_pred - 1, len(df) - days_train), 365)

    if len(df) < days_train + days_test:
      if verbose > 0:
        print('   No hay suficientes datos para entrenar y probar.')
      continue

    train_data = process_common(df, days_train, days_test, scaler_sales)
    model = train_quantum_common(model, train_data, product, folder, verbose, epochs, patience, lr)
    (r2, pearson) = test_common(model, train_data, days_train, days_test, verbose)
    if verbose > 0:
      print('   R^2:', r2)
      print('   pearson:', pearson)

    result.loc[len(result)] = [product, r2, pearson]

  return result

# %%
def test_day_common(model, train_data, offset, days_test, day, verbose=1, n_ventana=15, n_pred=1):
  train_data.build_test(offset, days_test)
  features_test = np.column_stack((train_data.test_days, train_data.dow_sin_test, train_data.dow_cos_test))
  x_test, y_test = split_data(train_data.test_days, features_test)
  x_test = np.array([ x_test[day] ])

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
  y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

  model.eval()
  x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
  with torch.no_grad():
      preds_tensor = model(x_test_tensor)
  predictions_scaled = preds_tensor.cpu().numpy().reshape(-1, 1)
  # predictions_scaled es un vector columna y cada fila es un vector fila de n_pred elementos
  predictions = train_data.descale_sales(predictions_scaled)

  y_test_real = []
  for i in range(len(train_data.test_days_original) - n_ventana - n_pred + 1):
    y_test_real.append(np.sum(train_data.test_days_original[i + n_ventana:i + n_ventana + n_pred]))
  y_test_real = np.array(y_test_real)

  return (predictions[0], y_test_real[day])
