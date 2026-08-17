# %%
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import torch.nn as nn
import common
from qiskit_ibm_runtime import QiskitRuntimeService
import pennylane as qml
from pennylane.qnn import TorchLayer
import torch
import numpy as np

# %%
n_ventana = 15
n_pred = 1

n_qubits = 6
repetitions = 5
parameters = 2

simulator = False

folder = 'modelos19'

# %%
def fechas_faltantes(df, start='2013-01-01', end='2017-08-15', verbose=1):
    df = df.copy()
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
from qiskit_ibm_runtime import QiskitRuntimeService

print(qml.about())

# %%
service = QiskitRuntimeService()
backend = service.least_busy(simulator=False, operational=True)
print(backend.name)

# %%
if simulator:
  print('Using simulator')
  dev = qml.device(
    "default.qubit",
    #"lightning.qubit",
    wires=n_qubits
  )
else:
  print('Using remote backend')
  dev = qml.device(
    "qiskit.remote",
    wires=n_qubits,
    backend=backend,
    shots=1024
  )

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
    build_ansatz(weights)
    return observables()

# %%
weight_shapes = {"weights": (repetitions, n_qubits, parameters)}

# Crear la capa de PennyLane
quantum_layer = TorchLayer(
    qnode=quantum_circuit,
    weight_shapes=weight_shapes,
    #output_dim=n_qubits
)

# %%
class HybridLSTM(nn.Module):
  def __init__(self, n_features, n_ventana):
    super(HybridLSTM, self).__init__()

    # LSTM layers
    self.lstm1 = nn.LSTM(
        input_size=n_features,
        hidden_size=128,
        batch_first=True
    )

    self.lstm2 = nn.LSTM(
        input_size=128,
        hidden_size=64,
        batch_first=True
    )

    # Shared representation
    self.shared = nn.Linear(64, 64)

    # Classical head
    self.classical_head = nn.Linear(64, 1)

    # Quantum head
    self.quantum_prep = nn.Linear(64, n_qubits)
    #self.norm = nn.LayerNorm(n_qubits)
    self.tanh = nn.Tanh()
    self.q_layer = quantum_layer
    self.quantum_out = nn.Linear(n_qubits, 1)

  def forward(self, x):
    out, _ = self.lstm1(x)
    out, _ = self.lstm2(out)
    out = out[:, -1, :]

    shared = self.shared(out)

    # Classical branch
    y_classical = self.classical_head(shared)

    # Quantum branch
    q = self.quantum_prep(shared)
    #q = self.norm(q)
    q = self.tanh(q) * torch.pi
    q = self.q_layer(q)
    y_quantum = self.quantum_out(q)

    # Residual sum
    y = y_classical + y_quantum

    return y

# %%
all_sales = pd.read_csv('sales3.csv')
all_sales = all_sales.rename(columns={'article': 'product', 'Quantity': 'sales'})
all_sales = all_sales[['date', 'product', 'sales']]
all_sales = all_sales.groupby(['product', 'date'], as_index=False)['sales'].sum()

# Columna date como datetime
all_sales['date'] = pd.to_datetime(all_sales['date'])

print(all_sales['product'].unique())
len(all_sales)

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

  return common.test_day_common(model, train_data, days_train, days_test, day, 0)

# %%
result = test_day('VIK BREAD', 156)
print(result)

#with open("output.txt", "w", encoding="utf-8") as file:
#  file.write(str(result))

# os.system("shutdown /s /t 30")

# %%
