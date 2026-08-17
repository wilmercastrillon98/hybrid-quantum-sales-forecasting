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
import common as common

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

  if verbose == 2:
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
  def __init__(self, df, scaler_sales=MinMaxScaler()):
    self.df = df
    self.sales = df['sales'].values
    self.dow_sin = df['dow_sin'].values
    self.dow_cos = df['dow_cos'].values
    self.scale_data(scaler_sales)

  def scale_data(self, scaler_sales):
    #self.df['sales_scaled'] = np.log1p(self.df['sales'])
    self.df['sales_scaled'] = self.df['sales']

    #scaler_sales = MinMaxScaler(feature_range=(0, 1))
    self.df['sales_scaled'] = scaler_sales.fit_transform(self.df[['sales_scaled']])
    self.scaler_sales = scaler_sales
    self.sales_scaled = self.df['sales_scaled'].values
    return self.sales_scaled
  
  # devuelve un arreglo normal
  def descale_sales(self, predictions_scaled):
    pred_log = self.scaler_sales.inverse_transform(predictions_scaled).flatten()
    #return np.expm1(pred_log)
    return pred_log

  def build(self, days_train, days_test):
    # Preparar datos de entrenamiento
    train_days = self.sales_scaled[:days_train]
    dow_sin_train = self.dow_sin[:days_train]
    dow_cos_train = self.dow_cos[:days_train]
    self.train_days = train_days
    self.dow_sin_train = dow_sin_train
    self.dow_cos_train = dow_cos_train

    ### REVISAR INTERVALOS DE PRUEBA
    
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
    #self.fc2 = nn.Linear(5, 1)

  def forward(self, x):
    out, _ = self.lstm1(x)
    out, _ = self.lstm2(out)
    #out, _ = self.lstm3(out)
    out = out[:, -1, :]
    out = self.fc1(out)
    #out = self.fc2(out)
    return out

# %%
def process(df, days_train, days_test, scaler_sales=MinMaxScaler()) -> TrainData:
  # Crear variables de día de la semana
  df['dow'] = df['date'].dt.dayofweek
  df['dow_sin'] = np.sin(2 * np.pi * df['dow']/7) * np.pi
  df['dow_cos'] = np.cos(2 * np.pi * df['dow']/7) * np.pi
  #df.tail(10)

  obj = TrainData(df, scaler_sales)
  obj.build(days_train, days_test)
  return obj

def train(obj: TrainData, product: str, verbose=1, epochs = 50):
  # Preparar datos para LSTM
  x_train, y_train = obj.get_train_data_LSTM()
  n_features = x_train.shape[2]
  if verbose == 2:
    print(f"n_features: {n_features}")

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
    #print(summary(model, input_size=(1, 4, n_features)))

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

  #if verbose == 1:
  #    print(model)

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

'''
Lubricant 11670    154559.996
Water 8975         147861.000
Cigarettes 864     137497.561
Lubricant 1045     106170.200
Cigarettes 9092     82586.000
Lubricant 3405      73114.500
Water 8487          71938.410
Lubricant 402       67632.890
The Bakery 9097     57825.020
The Bakery 2231     42481.000
'''

# Columna date como datetime
all_sales['date'] = pd.to_datetime(all_sales['date'])

print(all_sales['product'].unique())
len(all_sales)

# %%
#test_sales = all_sales[(all_sales['category'] == 'Accessories')]
# Agrupar por producto y sumar las ventas
product_sales = all_sales.groupby('product')['sales'].sum()
# Ordenar de mayor a menor y seleccionar los 10 primeros
top_10_products = product_sales.sort_values(ascending=False).head(15)
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
def excute(products, verbose=0, epochs=50):
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

    scaler_sales = StandardScaler()
    train_data = process(df, days_train, days_test, scaler_sales)
    model = train(train_data, product, verbose, epochs)

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
, epochs=100
, verbose=2)

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
import pennylane as qml
#import random

#semilla = random.randint(1, 10000)
#torch.manual_seed(semilla)
#np.random.seed(semilla)
#random.seed(semilla)
#qml.numpy.random.seed(semilla)

# %%
n_qubits = 6
repetitions = 4
parameters = 2

folder = 'modelos18'

# %%
dev = qml.device("default.qubit", wires=n_qubits)

def build_feature_map(inputs):
  for i in range(n_qubits):
    x = inputs[:, i]
    qml.RX(np.pi * x, wires=i)
    qml.RY(np.pi * x * 0.5, wires=i)

def build_ansatz(weights):
  for rep in range(repetitions):
    for q in range(n_qubits):
      qml.RY(weights[rep, q, 0], wires=q)
      qml.RZ(weights[rep, q, 1], wires=q)

    for q in range(n_qubits):
        target = (q + 1) % n_qubits
        qml.CNOT(wires=[q, target])
        qml.CRY(weights[rep, q, 1], wires=[q, target])
        qml.RZ(weights[rep, q, 1] * 0.5, wires=target)

def observables():
  return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    build_feature_map(inputs)
    qml.Barrier(wires=range(n_qubits), only_visual=True)
    build_ansatz(weights)
    qml.Barrier(wires=range(n_qubits), only_visual=True)
    return observables()

# %%
weight_shapes = {"weights": (repetitions, n_qubits, parameters)}

# Crear la capa de PennyLane
quantum_layer = TorchLayer(
    qnode=quantum_circuit, 
    weight_shapes=weight_shapes,
    #output_dim=n_qubits
)

#dummy_inputs = torch.zeros((1, n_qubits))
#dummy_weights = torch.zeros((repetitions, n_qubits, 2))
#fig, ax = qml.draw_mpl(quantum_circuit)(dummy_inputs, dummy_weights)
#plt.show()


# %%
class HybridLSTM(nn.Module):
  def __init__(self, n_features, n_ventana):
    super(HybridLSTM, self).__init__()
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

    self.fc_classical = nn.Linear(50, n_qubits)
    self.norm = nn.LayerNorm(n_qubits)
    #self.tanh = nn.Tanh()

    self.q_layer = quantum_layer
    
    self.fc_post = nn.Linear(n_qubits, 32)
    self.relu = nn.ReLU()

    self.fc_out = nn.Linear(32, 1)

  def forward(self, x):
    out, _ = self.lstm1(x)
    out, _ = self.lstm2(out)
    out = out[:, -1, :]
    
    out = self.fc_classical(out)
    out = self.norm(out)
    #out = self.tanh(out) * torch.pi
    out = self.q_layer(out)

    out = self.fc_post(out)
    out = self.relu(out)

    out = self.fc_out(out)
    return out

# %%
def train_quantum(obj, product, verbose=1, epochs = 50):
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
  model = HybridLSTM(n_features, n_ventana)

  # Loss + optimizador
  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5
  )

  # Usar GPU si está disponible
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)

  # Early stopping
  best_loss = float('inf')
  epochs_no_improve = 0
  patience = 10

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
      torch.save(model.state_dict(), "best_lstm_model.pt")
    else:
      epochs_no_improve += 1
    
    if verbose > 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

    if epochs_no_improve >= patience:
      if verbose > 0:
          print(f"⏹ Early stopping en epoch {epoch+1}")
      break
  
  if verbose == 1:
      print(model)

  torch.save(model.state_dict(), f"modelos18/{product}.pt")

  return model

# %%
def excute_quantum(products, verbose=0, epochs = 50):
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

    scaler_sales = MinMaxScaler(feature_range=(-np.pi, np.pi))
    train_data = process(df, days_train, days_test, scaler_sales)
    model = train_quantum(train_data, product, verbose, epochs)
    (r2, pearson) = test(model, train_data, days_train, days_test, verbose)
    if verbose > 0:
      print('   R^2:', r2)
      print('   pearson:', pearson)

    result.loc[len(result)] = [product, r2, pearson]

  return result

# %%
n_features = 3
model = HybridLSTM(n_features, n_ventana)
print(model)

from torchinfo import summary
summary(model, input_size=(1, 4, 3))

# %%
result_quantum = common.excute_quantum_common(
  all_sales,
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
  #'VIK BREAD',
  #'COMPLET',
  'FICELLE',
  #'MOISSON',
  #'BANETTINE',
  #'BOULE 200G',
  ],
  model,
  folder,
  #scaler_sales=StandardScaler(),
  scaler_sales=MinMaxScaler(feature_range=(-np.pi, np.pi)),
  #scaler_sales=MinMaxScaler(feature_range=(0, 1)),
  verbose=2,
  epochs=100,
  lr=0.001
)

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

  train_data = common.process_common(
    df, days_train, days_test,
    StandardScaler(),
    #MinMaxScaler(feature_range=(-np.pi, np.pi)),
    #MinMaxScaler(feature_range=(0, 1)),
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

  r2_val, pearson_val = common.test_common(model, train_data, days_train, days_test, verbose)
  if verbose == 1:
    print('   R^2:', r2_val)
    print('   pearson:', pearson_val)

  return model, r2_val, pearson_val

# %%
load_model('VIK BREAD', verbose=1)

# shutdown computer
#import os
#os.system("shutdown /s /t 1")

# %%
'''

RESULTADOS Clásico:

TRADITIONAL BAGUETTE	0.857792	0.926806
CROISSANT		          0.787264	0.889249
PAIN AU CHOCOLAT	    0.724889	0.852679
COUPE			            0.704118	0.840825
BANETTE			          0.606543	0.780923
BAGUETTE		          0.602767	0.783570
CEREAL BAGUETTE		    0.509998	0.717979
SPECIAL BREAD		      0.524084	0.728123
FORMULE SANDWICH	    0.222987	0.514630
TARTELETTE		        0.397959	0.649388

---

RESULTADOS Cuántico:
prediccion16.py

TRADITIONAL BAGUETTE	0.824448	0.915223
CROISSANT		          0.704030	0.864074
PAIN AU CHOCOLAT	    0.683443	0.839630
COUPE			            0.692271	0.838857
BANETTE			          0.596696	0.776284
BAGUETTE		          0.556460	0.751792
CEREAL BAGUETTE		    0.463251	0.701983
SPECIAL BREAD		      0.536145	0.734963
FORMULE SANDWICH	    0.281812	0.549213
TARTELETTE		        0.379291	0.628714

---

RESULTADOS Cuántico 2:
LSTM(100) → LSTM(50) → Linear(50 → 6) → VQC(6) → Linear(6 → 32) → Linear(32 → 1)
n_qubits = 6
repetitions = 3
parameters = 2

TRADITIONAL BAGUETTE	0.836335	0.922315 *
CROISSANT		          0.753603	0.881256 *
PAIN AU CHOCOLAT	    0.742100	0.864074 *
COUPE			            0.694228	0.835340
BANETTE			          0.602254	0.784417 *
BAGUETTE		          0.563147	0.760124 *
CEREAL BAGUETTE		    0.436655	0.681308 *
SPECIAL BREAD		      0.508501	0.717583 *
FORMULE SANDWICH	    0.217523	0.513812
TARTELETTE		        0.356434	0.629299

--- aumento de qubits

RESULTADOS Cuántico 3:
LSTM(100) → LSTM(50) → Linear(50 → 6) → VQC(6) → Linear(6 → 32) → Linear(32 → 1)
n_qubits = 7
repetitions = 3
parameters = 2
4 mejores que el clasico:

TRADITIONAL BAGUETTE	0.829514	0.920516
CROISSANT	            0.734558	0.862146
PAIN AU CHOCOLAT	    0.724780	0.857716
COUPE	                0.689251	0.839072
BANETTE	              0.616862	0.790205
BAGUETTE	            0.580126	0.770079 *
CEREAL BAGUETTE	      0.400292	0.658671
SPECIAL BREAD	        0.541395	0.745869 *
FORMULE SANDWICH	    0.231644	0.518152 *
TARTELETTE	          0.382614	0.633945 *


RESULTADOS Cuántico 4:
LSTM(100) → LSTM(50) → Linear(50 → 6) → VQC(6) → Linear(6 → 32) → Linear(32 → 1)
n_qubits = 8
repetitions = 3
parameters = 2
4 mejores que el clasico:

TRADITIONAL BAGUETTE	0.815272	0.915790
CROISSANT	            0.744144	0.877698 *
PAIN AU CHOCOLAT	    0.705815	0.840478
COUPE	                0.702611	0.841059 *
BANETTE	              0.613047	0.786207
BAGUETTE	            0.604374	0.785310 *
CEREAL BAGUETTE	      0.425909	0.681910 *
SPECIAL BREAD	        0.536111	0.741369
FORMULE SANDWICH	    0.231159	0.515751
TARTELETTE	          0.388920	0.643704 *

----------------------------------------------- 
Aumento de qubits parece mejorar los resultados de los productos con menor cantidad

4 mejores que el clasico
* -> los que mejoraron con el aumento de qubits
TRADITIONAL BAGUETTE	0.815272	0.915790
CROISSANT	            0.744144	0.877698
PAIN AU CHOCOLAT	    0.705815	0.840478
COUPE	                0.702611	0.841059 *
BANETTE	              0.613047	0.786207 *
BAGUETTE	            0.604374	0.785310 *
CEREAL BAGUETTE	      0.425909	0.681910
SPECIAL BREAD	        0.536111	0.741369 *
FORMULE SANDWICH	    0.231159	0.515751 *
TARTELETTE	          0.388920	0.643704 *
----------------------------------------------- 

--- aumento de repetitions

RESULTADOS Cuántico 5:
LSTM(100) → LSTM(50) → Linear(50 → 6) → VQC(6) → Linear(6 → 32) → Linear(32 → 1)
n_qubits = 6
repetitions = 4
parameters = 2
5 mejores que el clasico:

TRADITIONAL BAGUETTE	0.836229	0.918675
CROISSANT	            0.759059	0.877424
PAIN AU CHOCOLAT	    0.732483	0.856464
COUPE	                0.689553	0.836080
BANETTE	              0.677947	0.835866 *
BAGUETTE	            0.616687	0.791946 *
CEREAL BAGUETTE	      0.483282	0.702463 *
SPECIAL BREAD	        0.537399	0.741840 *
FORMULE SANDWICH	    0.292252	0.551467 *
TARTELETTE	          0.341256	0.606971


RESULTADOS Cuántico 6:
LSTM(100) → LSTM(50) → Linear(50 → 6) → VQC(6) → Linear(6 → 32) → Linear(32 → 1)
n_qubits = 6
repetitions = 5
parameters = 2
2 mejores que el clasico

TRADITIONAL BAGUETTE	0.849587	0.923320 *
CROISSANT	            0.769050	0.893073 *
PAIN AU CHOCOLAT	    0.719060	0.850475
COUPE	                0.712212	0.849955 *
BANETTE	              0.580813	0.781135
BAGUETTE	            0.574611	0.768782
CEREAL BAGUETTE	      0.395039	0.677892
SPECIAL BREAD	        0.510867	0.731144
FORMULE SANDWICH	    0.272398	0.534674
TARTELETTE	          0.362433	0.633992 *

----------------------------------------------- 
Aumento de repetitions parece mejorar los resultados de los productos en general
2 mejores que el clasico

* -> los que mejoraron con el aumento de repetitions
TRADITIONAL BAGUETTE	0.849587	0.923320 *
CROISSANT	            0.769050	0.893073 *
PAIN AU CHOCOLAT	    0.719060	0.850475
COUPE	                0.712212	0.849955 *
BANETTE	              0.580813	0.781135
BAGUETTE	            0.574611	0.768782 *
CEREAL BAGUETTE	      0.395039	0.677892
SPECIAL BREAD	        0.510867	0.731144 *
FORMULE SANDWICH	    0.272398	0.534674 *
TARTELETTE	          0.362433	0.633992 *
----------------------------------------------- 

-- aumento de qubits y repetitions

RESULTADOS Cuántico 7:
LSTM(100) → LSTM(50) → Linear(50 → 6) → VQC(6) → Linear(6 → 32) → Linear(32 → 1)
n_qubits = 8
repetitions = 5
parameters = 2
4 mejores que el clasico:

TRADITIONAL BAGUETTE	0.823012	0.910241
CROISSANT	            0.754188	0.882890 *
PAIN AU CHOCOLAT	    0.747206	0.865494 *
COUPE	                0.724369	0.852527 *
BANETTE	              0.612442	0.785380 *
BAGUETTE	            0.571439	0.780190 *
CEREAL BAGUETTE	      0.375680	0.660279
SPECIAL BREAD	        0.570416	0.758271 *
FORMULE SANDWICH	    0.197560	0.487751
TARTELETTE	          0.359145	0.638884

No mejoraron tanto como se esperaba

-----------------------------------------------

punto intermedio:
LSTM(100) → LSTM(50) → Linear(50 → 6) → VQC(6) → Linear(6 → 32) → Linear(32 → 1)
n_qubits = 7
repetitions = 4
parameters = 2
3 mejores que el clasico:

TRADITIONAL BAGUETTE	0.842586	0.922043 *
CROISSANT	            0.736784	0.872000
PAIN AU CHOCOLAT	    0.728550	0.858828
COUPE	                0.706907	0.847346 *
BANETTE	              0.612148	0.788958 *
BAGUETTE	            0.595558	0.781291 *
CEREAL BAGUETTE	      0.404686	0.662637
SPECIAL BREAD	        0.505052	0.723768 *
FORMULE SANDWICH	    0.165207	0.470762
TARTELETTE	          0.373880	0.625959 *

#####################################################################################33

RESULTADO PARA DOCUMENTO 2
LSTM(100) → LSTM(50) → Linear(50 → 6) → VQC(6) → Linear(6 → 32) → Linear(32 → 1)
n_qubits = 6
repetitions = 4
parameters = 2
5 mejores que el clasico:

TRADITIONAL BAGUETTE	0.836229	0.918675
CROISSANT	            0.759059	0.877424
PAIN AU CHOCOLAT	    0.732483	0.856464 *
COUPE	                0.689553	0.836080
BANETTE	              0.677947	0.835866 *
BAGUETTE	            0.616687	0.791946 *
CEREAL BAGUETTE	      0.483282	0.702463
SPECIAL BREAD	        0.537399	0.741840 *
FORMULE SANDWICH	    0.292252	0.551467 *
TARTELETTE	          0.341256	0.606971

-----------

EXTRAS - ninguno mejor que el clasico

-----------

RESULTADOS Cuántico 6:
LSTM(100) → LSTM(50) → Linear(50 → 6) → VQC(6) → Linear(6 → 32) → Linear(32 → 1)
n_qubits = 6
repetitions = 5
parameters = 2

TRADITIONAL BAGUETTE	0.849587	0.923320
CROISSANT	            0.769050	0.893073
PAIN AU CHOCOLAT	    0.719060	0.850475
COUPE	                0.712212	0.849955 *
BANETTE	              0.580813	0.781135
BAGUETTE	            0.574611	0.768782
CEREAL BAGUETTE	      0.395039	0.677892
SPECIAL BREAD	        0.510867	0.731144
FORMULE SANDWICH	    0.272398	0.534674 **
TARTELETTE	          0.362433	0.633992

-----------

scaler [-pi, pi]

TRADITIONAL BAGUETTE	0.826833	0.917450
CROISSANT	            0.765485	0.881172
PAIN AU CHOCOLAT	    0.679313	0.834532
COUPE	                0.672638	0.824891
BANETTE	              0.596296	0.784216
BAGUETTE	            0.577340	0.783747
CEREAL BAGUETTE	      0.445617	0.683727
SPECIAL BREAD	        0.541688	0.738830 *
FORMULE SANDWICH	    0.202619	0.488699
TARTELETTE	          0.349341	0.644429

-----------

scaler [-pi, pi] -> tanh(x) * pi

TRADITIONAL BAGUETTE	0.815592	0.907775
CROISSANT	            0.720286	0.860431
PAIN AU CHOCOLAT	    0.646968	0.822872
COUPE	                0.679389	0.826757
BANETTE	              0.600954	0.783403
BAGUETTE	            0.613788	0.788194 *
CEREAL BAGUETTE	      0.416661	0.673767
SPECIAL BREAD	        0.551365	0.763576 *
FORMULE SANDWICH	    0.233524	0.508250
TARTELETTE	          0.384229	0.632183

-------------------------

TRADITIONAL BAGUETTE	0.836229	0.918675
CROISSANT	            0.759059	0.877424
PAIN AU CHOCOLAT	    0.732483	0.856464 *
COUPE	                0.689553	0.836080
BANETTE	              0.677947	0.835866 *
BAGUETTE	            0.616687	0.791946 *
CEREAL BAGUETTE	      0.483282	0.702463
SPECIAL BREAD	        0.537399	0.741840 *
FORMULE SANDWICH	    0.292252	0.551467 *
TARTELETTE	          0.341256	0.606971

BOULE 400G	          0.338315	0.635515
CAMPAGNE	            0.317727	0.629931
COOKIE	              0.142896	0.478742
ECLAIR	              0.147730	0.568706
VIK BREAD	            0.689445	0.837608

COMPLET	    0.163067	0.530657
FICELLE	    0.483589	0.711714
MOISSON	    0.058353	0.435097
BANETTINE	  -0.347044	0.264410
BOULE 200G  0.214828	0.522381

'''
# %%
