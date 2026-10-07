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
from torchinfo import summary

import common

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
def test(model, train_data, offset, days_test, verbose=1):
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

  if verbose == 1:
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

def split_data(data, features):
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
class TrainData:
  def __init__(self, df, scaler_range=(0, 1)):
    self.df = df
    self.sales = df['sales'].values
    self.dow_sin = df['dow_sin'].values
    self.dow_cos = df['dow_cos'].values
    self.scale_data(scaler_range)

  def scale_data(self, scaler_range=(0, 1)):
    self.df['sales_scaled'] = self.df['sales']

    scaler_sales = MinMaxScaler(feature_range=scaler_range)
    #scaler_sales = StandardScaler()
    self.df['sales_scaled'] = scaler_sales.fit_transform(self.df[['sales_scaled']])
    self.scaler_sales = scaler_sales
    self.sales_scaled = self.df['sales_scaled'].values
    return self.sales_scaled
  
  # devuelve un arreglo normal
  def descale_sales(self, predictions_scaled):
    pred_log = self.scaler_sales.inverse_transform(predictions_scaled).flatten()
    return pred_log

  def build(self, days_train, days_test):
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
class LSTMModel(nn.Module):
  def __init__(self, n_features, n_ventana):
    super(LSTMModel, self).__init__()

    self.lstm1 = nn.LSTM(
      input_size=n_features,
      hidden_size=128,
      batch_first=True,
      num_layers=1
    )

    self.lstm2 = nn.LSTM(
      input_size=128,
      hidden_size=64,
      batch_first=True,
      num_layers=1
    )

    self.fc1 = nn.Linear(64, 1)

  def forward(self, x):
    out, _ = self.lstm1(x)
    out, _ = self.lstm2(out)
    #out, _ = self.lstm3(out)
    out = out[:, -1, :]
    out = self.fc1(out)
    return out

# %%
def process(df, days_train, days_test, scaler_range=(0, 1)) -> TrainData:
  # Crear variables de día de la semana
  df['dow'] = df['date'].dt.dayofweek
  df['dow_sin'] = np.sin(2 * np.pi * df['dow']/7)
  df['dow_cos'] = np.cos(2 * np.pi * df['dow']/7)
  #df.tail(10)

  obj = TrainData(df, scaler_range)
  obj.build(days_train, days_test)
  return obj

def train(obj: TrainData, verbose=1, epochs = 50):
  # Preparar datos para LSTM
  x_train, y_train = obj.get_train_data_LSTM()
  n_features = x_train.shape[2]
  if verbose == 1:
    print(f"n_features: {n_features}")

  # Convertir a tensores PyTorch
  x_train = torch.tensor(x_train, dtype=torch.float32)
  y_train = torch.tensor(y_train, dtype=torch.float32)

  # Dataset + dataloader
  dataset = TensorDataset(x_train, y_train)
  loader = DataLoader(dataset, batch_size=32, shuffle=True)
  
  # Crear modelo
  model = LSTMModel(n_features, n_ventana)

  # Loss + optimizador
  criterion = nn.SmoothL1Loss()   # mejor que MSE para series
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
      torch.save(model.state_dict(), "best_lstm_model.pt")
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
  model.load_state_dict(torch.load("best_lstm_model.pt"))

  if verbose == 1:
      print(model)

  return model

# %%
# Debe tener columnas: date, product, sales

# conjuntos de productos
#“Retail Store Inventory and Demand Forecasting” y “Store Sales Forecasting Dataset”, 
#all_sales = pd.read_csv('train.csv') 
#all_sales = all_sales[all_sales['store_nbr'] == 1]
#all_sales = all_sales.rename(columns={'family': 'product', 'Units Sold': 'sales'})

# conjuntos Retail Store Inventory and Demand Forecasting -> simulados XXX
#all_sales = pd.read_csv('sales2.csv')
#all_sales = all_sales[all_sales['Store ID'] == 'S001']
#all_sales = all_sales.rename(columns={'Product ID': 'product', 'Units Sold': 'sales', 'Date': 'date'})
#all_sales = all_sales[['date', 'product', 'sales']]

# conjuntos Sample Sales Data French bakery daily sales
all_sales = pd.read_csv('sales3.csv')
all_sales = all_sales.rename(columns={'article': 'product', 'Quantity': 'sales'})
all_sales = all_sales[['date', 'product', 'sales']]
all_sales = all_sales.groupby(['product', 'date'], as_index=False)['sales'].sum()
# ['CROISSANT', 'TRADITIONAL BAGUETTE', 'BOULE 200G', 'VIK BREAD', 'BANETTINE']
# ['CROISSANT', 'TRADITIONAL BAGUETTE', 'BANETTINE']
'''
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
'''

# conjuntos Sales data for a chain of Brazilian stores
#all_sales = pd.read_csv('sales4.csv', delimiter=';')
#all_sales = all_sales.rename(columns={'Product': 'product', 'Sale Date Time': 'date', 'Amount': 'sales', 'Product Category': 'category'})
#all_sales = all_sales[['date', 'product', 'sales', 'category']]
#all_sales['date'] = pd.to_datetime(all_sales['date']).dt.normalize()
#all_sales = all_sales.groupby(['product', 'date', 'category'], as_index=False)['sales'].sum()
#['Additives And Fluids 128', 'Filters 3227', 'Lubricant 11670', 'Several 3393', 'Accessories 2308']
#['Filters 3227', 'Lubricant 11670', 'Additives And Fluids 128']

# Columna date como datetime
all_sales['date'] = pd.to_datetime(all_sales['date'])

print(all_sales['product'].unique())
len(all_sales)

# %%
test_sales = all_sales[(all_sales['category'] == 'Accessories')]
# Agrupar por producto y sumar las ventas
product_sales = test_sales.groupby('product')['sales'].sum()
# Ordenar de mayor a menor y seleccionar los 10 primeros
top_10_products = product_sales.sort_values(ascending=False).head(10)
# Mostrar el resultado
print("Los 10 productos con mayor cantidad de ventas son:")
print(top_10_products)

# %%
n_ventana = 15
n_pred = 1

# %%
product = 'SPECIAL BREAD'
df = all_sales[(all_sales['product'] == product)]
df.head(10)
len(df)

# %%
def excute(products, verbose=0):
  # verbose: 0 - sin salida, 1 - con gráficas, 2 - con entrenamiento detallado
  result =  pd.DataFrame(columns=['product', 'r2', 'pearson'])

  for product in products:
    if verbose == 1:
      print('### Producto:', product)
    df = all_sales[(all_sales['product'] == product)].copy()

    if df.empty:
      if verbose == 1:
        print('   No hay datos para este producto.')
      continue
    
    df = fix_dates(df, product, verbose)
    days_train = max(365 + n_ventana + n_pred - 1, len(df) - 365)
    days_test = min(max(30 + n_pred - 1, len(df) - days_train), 365)

    if len(df) < days_train + days_test:
      if verbose == 1:
        print('   No hay suficientes datos para entrenar y probar.')
      continue

    train_verbose = 1 if verbose == 2 else 0
    train_data = process(df, days_train, days_test)
    model = train(train_data, train_verbose)
    (r2, pearson) = test(model, train_data, days_train, days_test, verbose)
    if verbose == 1:
      print('   R^2:', r2)
      print('   pearson:', pearson)

    result.loc[len(result)] = [product, r2, pearson]

  return result

# %%
#result = 
excute(
  #['TRADITIONAL BAGUETTE','CROISSANT','PAIN AU CHOCOLAT','COUPE','BANETTE','BAGUETTE','CEREAL BAGUETTE','SPECIAL BREAD','FORMULE SANDWICH','TARTELETTE']
  ['TRADITIONAL BAGUETTE',
  'CROISSANT',
  'PAIN AU CHOCOLAT',
  'BANETTE',
  'CEREAL BAGUETTE',
  'SPECIAL BREAD',
  'FORMULE SANDWICH',
  'TARTELETTE',
  'COOKIE']
, verbose=1)

#print(result)
#print('Promedio R^2:', result['r2'].mean())
#print('Promedio pearson:', result['pearson'].mean())

# %%
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

def obtener_product_monthly(all_sales, family):
  products = all_sales[(all_sales['product'] == family)]

  if products.empty:
    print(f'No data found for product: {family}')
    return

  products = fix_dates(products, family, verbose=0)
  dayFirst = products['date'][0]
  dayLast = products['date'].iloc[-1]
  first_weekday, num_days = calendar.monthrange(dayLast.year, dayLast.month)

  products_monthly = (
      products.groupby(products['date'].dt.to_period('M'))['sales']
      .agg(['mean', 'median', 'sum', 'std'])
      .reset_index()
  )
  products_monthly['date'] = products_monthly['date'].dt.to_timestamp()

  if(dayFirst.day > 2):
    products_monthly.drop(products_monthly.index[0], inplace=True)
  if(dayLast.day < num_days - 2):
    products_monthly.drop(products_monthly.index[-1], inplace=True)

  return products_monthly

def graficar(all_sales, family):
  products_monthly = obtener_product_monthly(all_sales, family)
  period = products_monthly.shape[0] // 2

  # Analisis estacionalidad
  products_monthly['pct_change'] = products_monthly['sum'].pct_change() * 100
  result = seasonal_decompose(
    products_monthly.set_index('date')['sum'],
    model='additive',
    period=period
  )
  fig = result.plot()

  # Ajustar formato de fechas en eje X
  for ax in fig.axes:
      ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
      plt.setp(ax.get_xticklabels(), rotation=0, ha='right')
  fig.axes[0].set_title('')                # elimina el título superior
  fig.axes[0].set_ylabel('sum', rotation=90, labelpad=10)  # lo mueve al lado izquierdo
  
  plt.tight_layout()
  plt.show()

  # Serie mensual de ventas (usando la que ya creaste)
  sales = products_monthly.set_index('date')['sum']
  plt.figure(figsize=(8,4))
  plot_acf(sales, lags=period)  # 24 meses (~2 años)
  plt.title('Autocorrelograma - Ventas de ' + family)
  plt.show()

# %%

def buid_graphics(products):
  for product in products:
    print('### ### Producto:', product)
    graficar(all_sales, product)

# %%
buid_graphics(['Filters 3227', 'Lubricant 11670', 'Additives And Fluids 128'])


# %%

def grafica_triple(all_sales, family1, family2, family3):
  # Agrupar por mes
  products_monthly1 = obtener_product_monthly(all_sales, family1)
  products_monthly2 = obtener_product_monthly(all_sales, family2)
  products_monthly3 = obtener_product_monthly(all_sales, family3)

  period1 = products_monthly1.shape[0] // 2
  period2 = products_monthly2.shape[0] // 2
  period3 = products_monthly3.shape[0] // 2

  # Descomponer las tres series
  results = [
      seasonal_decompose(df.set_index('date')['sum'], model='additive', period=pd)
      for df, pd in [(products_monthly1,period1), (products_monthly2,period2), (products_monthly3,period3)]
  ]

  # Crear figura con 4 filas (observed, trend, seasonal, resid) y 3 columnas (una por producto)
  fig, axes = plt.subplots(4, 3, figsize=(12, 8), sharex=True)
  # Nombres de productos (ajusta según tus datos)
  product_names = [family1, family2, family3]

  # Componentes en orden
  components = [('observed', 'Suma'), ('trend', 'Tendencia'), 
                ('seasonal', 'Estacionalidad'), ('resid', 'Residuo')]

  for j, result in enumerate(results):          # columna = producto
      for i, comp in enumerate(components):     # fila = componente
          series = getattr(result, comp[0])
          axes[i, j].plot(series.index, series.values, color='tab:blue')
          if j == 0:
              axes[i, j].set_ylabel(comp[1], rotation=90, labelpad=10, ha='right', va='center')
          if i == 0:
              axes[i, j].set_title(product_names[j])

  # Formato de fechas
  for ax in axes[-1, :]:  # solo última fila (resid)
      ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
      ax.tick_params(axis='x', rotation=45)

  plt.tight_layout()
  plt.show()

  # Autocorrelogramas
  fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
  plot_acf(products_monthly1.set_index('date')['sum'], lags=period1, ax=axes[0])
  axes[0].set_title(f'Autocorrelograma - Ventas de {family1}')
  plot_acf(products_monthly2.set_index('date')['sum'], lags=period2, ax=axes[1])
  axes[1].set_title(f'Autocorrelograma - Ventas de {family2}')
  plot_acf(products_monthly3.set_index('date')['sum'], lags=period3, ax=axes[2])
  axes[2].set_title(f'Autocorrelograma - Ventas de {family3}')

  plt.tight_layout()
  plt.show()

# %%
grafica_triple(all_sales, 'Filters 3227', 'Lubricant 11670', 'Additives And Fluids 128')

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
n_qubits = 6
repetitions = 1

folder = 'modelos24'

# %%

def build_feature_map(inputs, n_qubits):
  # Encoding
  for i in range(n_qubits):
    x = inputs[:, i]

    qml.Hadamard(wires=i)
    qml.RY(torch.atan(x), wires=i)
    qml.RZ(torch.atan(x ** 2), wires=i)

def build_ansatz(weights, n_qubits):
  # Entanglement i -> i+1 (circular)
  for i in range(n_qubits):
    qml.CNOT(wires=[i, (i + 1) % n_qubits])

  # Entanglement i -> i+2 (circular)
  for i in range(n_qubits):
    qml.CNOT(wires=[i, (i + 2) % n_qubits])

  for layer in range(weights.shape[0]):
    for i in range(n_qubits):
      qml.RX(weights[layer, i, 0], wires=i)
      qml.RY(weights[layer, i, 1], wires=i)
      qml.RZ(weights[layer, i, 2], wires=i)

def observables(n_qubits):
  return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# %%
def create_qnode(n_qubits=4, repetitions=2):
  dev = qml.device("default.qubit", wires=n_qubits)

  @qml.qnode(dev, interface="torch")
  def circuit(inputs, weights):
    build_feature_map(inputs, n_qubits)
    build_ansatz(weights, n_qubits)
    return observables(n_qubits)

  weight_shapes = {"weights": (repetitions, n_qubits, 3)}
  return TorchLayer(circuit, weight_shapes)

dev = qml.device("default.qubit", wires=n_qubits)
@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    build_feature_map(inputs, n_qubits)
    qml.Barrier(wires=range(n_qubits), only_visual=True)
    build_ansatz(weights, n_qubits)
    qml.Barrier(wires=range(n_qubits), only_visual=True)
    return observables(n_qubits)

#dummy_inputs = torch.zeros((1, n_qubits))
#dummy_weights = torch.zeros((repetitions, n_qubits, 3))  # Cambiado de 2 a 3
#fig, ax = qml.draw_mpl(quantum_circuit)(dummy_inputs, dummy_weights)
#plt.show()


# %%
class QLSTMCell(nn.Module):
  def __init__(self, input_size, hidden_size):
    super(QLSTMCell, self).__init__()

    self.q_forget = create_qnode(n_qubits, repetitions)
    self.q_input = create_qnode(n_qubits, repetitions)
    self.q_output = create_qnode(n_qubits, repetitions)
    self.q_candidate = create_qnode(n_qubits, repetitions)

    self.linear = nn.Linear(input_size + hidden_size, 4*n_qubits)
    self.fc_out_f = nn.Linear(n_qubits, hidden_size)
    self.fc_out_i = nn.Linear(n_qubits, hidden_size)
    self.fc_out_o = nn.Linear(n_qubits, hidden_size)
    self.fc_out_g = nn.Linear(n_qubits, hidden_size)

  def forward(self, x, h, c):
    combined = torch.cat([x, h], dim=1)
    gates = self.linear(combined)
    wf, wi, wo, wg = torch.chunk(gates, 4, dim=1)

    f = torch.sigmoid(self.fc_out_f(self.q_forget(wf)))
    i = torch.sigmoid(self.fc_out_i(self.q_input(wi)))
    o = torch.sigmoid(self.fc_out_o(self.q_output(wo)))
    g = torch.tanh(self.fc_out_g(self.q_candidate(wg)))

    c = f * c + i * g
    h = o * torch.tanh(c)

    return h, c

# %%
class QLSTM(nn.Module):
  def __init__(self, input_size, hidden_size):
    super(QLSTM, self).__init__()

    self.hidden_size = hidden_size
    self.cell = QLSTMCell(input_size, hidden_size)

  def forward(self, x):
    batch_size, seq_len, _ = x.size()

    h = torch.zeros(batch_size, self.hidden_size).to(x.device)
    c = torch.zeros(batch_size, self.hidden_size).to(x.device)

    outputs = []

    for t in range(seq_len):
        h, c = self.cell(x[:, t, :], h, c)
        outputs.append(h.unsqueeze(1))

    return torch.cat(outputs, dim=1)

# %%
class HybridLSTM(nn.Module):
  def __init__(self, n_features, n_ventana):
    super(HybridLSTM, self).__init__()

    self.qlstm1 = QLSTM(n_features, 64)
    self.qlstm2 = QLSTM(64, 32)
    self.fc = nn.Linear(32, 1)

  def forward(self, x):
    out = self.qlstm1(x)
    out = self.qlstm2(out)
    out = out[:, -1, :]
    out = self.fc(out)

    return out

# %%
def train_quantum(obj, product, verbose=1, epochs = 50, patience = 10, lr = 0.001):
  # Preparar datos para LSTM
  x_train, y_train = obj.get_train_data_LSTM()
  n_features = x_train.shape[2]

  os.makedirs(folder, exist_ok=True)
  best_path = f"{folder}/{product}_best.pt"

  # Convertir a tensores PyTorch
  x_train = torch.tensor(x_train, dtype=torch.float32)
  y_train = torch.tensor(y_train, dtype=torch.float32)

  # Dataset + dataloader
  dataset = TensorDataset(x_train, y_train)
  loader = DataLoader(dataset, batch_size=32, shuffle=True)
  
  # Crear modelo
  model = HybridLSTM(n_features, n_ventana)
  if verbose == 2:
    print(summary(model, input_size=(1, 4, n_features)))

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

  #model.load_state_dict(torch.load(best_path, map_location=device))

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
    
    if verbose > 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

    logs[epoch] = {
      'loss': avg_loss,
      'lr': optimizer.param_groups[0]['lr'],
    }

    if epochs_no_improve >= patience:
      if verbose > 0:
          print(f"⏹ Early stopping en epoch {epoch+1}")
      break
  
  if os.path.isfile(best_path):
    model.load_state_dict(torch.load(best_path, map_location=device))

  #if verbose == 1:
  #    print(model)
  torch.save(model.state_dict(), f"{folder}/{product}.pt")
  pd.DataFrame.from_dict(logs, orient='index').rename_axis('epoch').reset_index().to_csv(
    f"{folder}/{product}_logs.csv", index=False
  )
  return model

# %%
def excute_quantum(products, verbose=0, epochs = 50, patience = 10, lr=0.001):
  result =  pd.DataFrame(columns=['product', 'r2', 'pearson'])
  for product in products:
    if verbose == 1:
      print('### Producto:', product)
    df = all_sales[(all_sales['product'] == product)].copy()

    if df.empty:
      if verbose == 1:
        print('   No hay datos para este producto.')
      continue
    
    df = fix_dates(df, product, verbose)
    days_train = max(365 + n_ventana + n_pred - 1, len(df) - 365)
    days_test = min(max(30 + n_pred - 1, len(df) - days_train), 365)

    if len(df) < days_train + days_test:
      if verbose == 1:
        print('   No hay suficientes datos para entrenar y probar.')
      continue

    scaler_range = (-np.pi, np.pi)
    #scaler_range = (0, 2*np.pi)
    train_data = process(df, days_train, days_test, scaler_range)
    model = train_quantum(train_data, product, verbose, epochs, patience, lr)
    (r2, pearson) = test(model, train_data, days_train, days_test, verbose)
    if verbose == 1:
      print('   R^2:', r2)
      print('   pearson:', pearson)

    result.loc[len(result)] = [product, r2, pearson]

    result.to_csv('Resultados_Quantum24.csv', index=False)
    print(result)

  return result

# %%
result_quantum = excute_quantum(
  [
  #'TRADITIONAL BAGUETTE',
  #'CROISSANT',
  #'PAIN AU CHOCOLAT',
  #'COUPE',
  #'BANETTE',
  #'BAGUETTE',
  #'CEREAL BAGUETTE',
  #'SPECIAL BREAD',
  #'FORMULE SANDWICH',
  #'TARTELETTE',

  #'BOULE 400G',
  #'CAMPAGNE',
  #'COOKIE',
  #'ECLAIR',
  'VIK BREAD',
  #'COMPLET',
  #'FICELLE',
  #'MOISSON',
  #'BANETTINE',
  #'BOULE 200G',
  ]
  , verbose=2
  , epochs = 25
  , patience = 10
  , lr = 0.012
)

#os.system("shutdown /s /t 30")


# %%
def load_model(product, verbose=0):
  """Carga el checkpoint `{folder}/{product}.pt` y ejecuta `test` con la misma
  partición train/test que `excute_quantum` para ese producto."""
  df = all_sales[(all_sales['product'] == product)].copy()
  if df.empty:
    raise ValueError(f'No hay datos para el producto: {product}')

  df = fix_dates(df, product, verbose)
  days_train = max(365 + n_ventana + n_pred - 1, len(df) - 365)
  days_test = min(max(30 + n_pred - 1, len(df) - days_train), 365)

  if len(df) < days_train + days_test:
    raise ValueError('No hay suficientes datos para entrenar y probar.')

  train_data = process(
    df, days_train, days_test, 
    scaler_range=(-np.pi, np.pi)
  )
  x_train, _ = train_data.get_train_data_LSTM()
  n_features = x_train.shape[2]

  path = f"{folder}/{product}.pt"
  if not os.path.isfile(path):
    raise FileNotFoundError(f'No existe el modelo guardado: {path}')

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = HybridLSTM(n_features, n_ventana)
  model.load_state_dict(torch.load(path, map_location=device))
  model.to(device)

  if verbose == 1:
    print(summary(model, input_size=(1, 4, n_features)))

  r2_val, pearson_val = common.test_common(model, train_data, days_train, days_test, verbose)
  if verbose == 1:
    print('   R^2:', r2_val)
    print('   pearson:', pearson_val)


  return model, r2_val, pearson_val

# %%
load_model('VIK BREAD', verbose=1)

# %%
def test_day(product, day):
  df = all_sales[(all_sales['product'] == product)].copy()
  if df.empty:
    raise ValueError(f'No hay datos para el producto: {product}')

  df = fix_dates(df, product, 0)
  days_train = max(365 + n_ventana + n_pred - 1, len(df) - 365)
  days_test = min(max(30 + n_pred - 1, len(df) - days_train), 365)

  if len(df) < days_train + days_test:
    raise ValueError('No hay suficientes datos para entrenar y probar.')

  train_data = process(
    df, days_train, days_test, 
    scaler_range=(-np.pi, np.pi)
  )
  x_train, _ = train_data.get_train_data_LSTM()
  n_features = x_train.shape[2]

  path = f"{folder}/{product}.pt"
  if not os.path.isfile(path):
    raise FileNotFoundError(f'No existe el modelo guardado: {path}')

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = HybridLSTM(n_features, n_ventana)
  model.load_state_dict(torch.load(path, map_location=device))
  model.to(device)

  return common.test_day_common(model, train_data, days_train, days_test, day, 0)

# %%
test_day('VIK BREAD', 150)

# %%
