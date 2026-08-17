# %%
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
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
repetitions = 1

folder = 'modelos24'

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
def process(df, days_train, days_test, scaler_range=(0, 1)) -> TrainData:
  # Crear variables de día de la semana
  df['dow'] = df['date'].dt.dayofweek
  df['dow_sin'] = np.sin(2 * np.pi * df['dow']/7)
  df['dow_cos'] = np.cos(2 * np.pi * df['dow']/7)
  #df.tail(10)

  obj = TrainData(df, scaler_range)
  obj.build(days_train, days_test)
  return obj

# %%
from qiskit_ibm_runtime import QiskitRuntimeService

print(qml.about())

# %%
service = QiskitRuntimeService()
#backend = service.least_busy(simulator=False, operational=True)
#print(backend.name)

# %%
#from qiskit_aer import AerSimulator
#from qiskit_ibm_runtime import QiskitRuntimeService
#backend = service.backend("ibm_marrakesh")
#noise_simulator = AerSimulator.from_backend(backend)

#from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
#backend = FakeMarrakesh()
#print(backend.name)

from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer.noise import NoiseModel

real_backend = service.least_busy(simulator=False, operational=True)
noise_model = NoiseModel.from_backend(real_backend)
#coupling_map = real_backend.configuration().coupling_map
#basis_gates = real_backend.configuration().basis_gates

sim_backend = AerSimulator(
  noise_model=noise_model,
  #coupling_map=coupling_map,
  #basis_gates=basis_gates
)

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
def create_qnode(n_qubits=4, repetitions=2, simulator=True):
  #dev = qml.device("default.qubit", wires=n_qubits)
  if simulator:
    dev = qml.device(
      #"default.qubit",
      #"lightning.qubit",
      "qiskit.aer",
      wires=n_qubits,
      backend=sim_backend,
      shots=1024
    )
  else:
    dev = qml.device(
      "qiskit.remote",
      wires=n_qubits,
      backend=real_backend,
      shots=1024
    )

  @qml.qnode(dev, interface="torch")
  def circuit(inputs, weights):
    build_feature_map(inputs, n_qubits)
    build_ansatz(weights, n_qubits)
    return observables(n_qubits)

  weight_shapes = {"weights": (repetitions, n_qubits, 3)}
  return TorchLayer(circuit, weight_shapes)

# %%
class QLSTMCell(nn.Module):
  def __init__(self, input_size, hidden_size):
    super(QLSTMCell, self).__init__()

    self.q_forget = create_qnode(n_qubits, repetitions, simulator=True)
    self.q_input = create_qnode(n_qubits, repetitions, simulator=True)
    self.q_output = create_qnode(n_qubits, repetitions, simulator=True)
    self.q_candidate = create_qnode(n_qubits, repetitions, simulator=True)

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
result = test_day('VIK BREAD', 155)
print(result)

with open("output.txt", "w", encoding="utf-8") as file:
  file.write(str(result))

# os.system("shutdown /s /t 30")

# %%
