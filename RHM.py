# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import common as common
import pennylane as qml
from pennylane.qnn import TorchLayer
import torch
import numpy as np
import pennylane as qml
#import random

# %%
# Debe tener columnas: date, product, sales

# conjuntos Sample Sales Data French bakery daily sales
all_sales = pd.read_csv('sales.csv')
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

# Columna date como datetime
all_sales['date'] = pd.to_datetime(all_sales['date'])

print(all_sales['product'].unique())
len(all_sales)

# %%
n_qubits = 6
repetitions = 4
parameters = 2

n_ventana = 15
n_pred = 1

folder = 'modelos19'

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

#print(qml.draw(quantum_circuit)(dummy_x, dummy_w))

# %%
class HybridLSTM(nn.Module):
  def __init__(self, n_features, n_ventana):
    super(HybridLSTM, self).__init__()

    # LSTM layers
    self.lstm1 = nn.LSTM(
        input_size=n_features,
        hidden_size=100,
        batch_first=True
    )

    self.lstm2 = nn.LSTM(
        input_size=100,
        hidden_size=50,
        batch_first=True
    )

    # Shared representation
    self.shared = nn.Linear(50, 50)

    # Classical head
    self.classical_head = nn.Linear(50, 1)

    # Quantum head
    self.quantum_prep = nn.Linear(50, n_qubits)
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
n_features = 3
model = HybridLSTM(n_features, n_ventana)

from torchinfo import summary
summary(model, input_size=(1, 5, 3))

# %%
result_quantum = common.excute_quantum_common(
  all_sales,
  [
  'TRADITIONAL BAGUETTE',
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
  #'FICELLE',
  #'MOISSON',
  #'BANETTINE',
  #'BOULE 200G',
  ],
  model,
  folder,
  #scaler_sales=StandardScaler(),
  #scaler_sales=MinMaxScaler(feature_range=(-np.pi, np.pi)),
  scaler_sales=MinMaxScaler(feature_range=(0, 1)),
  verbose=2,
  epochs=50,
  lr=0.001
)

# %%
common.load_model_common(
  all_sales, 'TRADITIONAL BAGUETTE', folder, HybridLSTM,
  scaler_sales=MinMaxScaler(feature_range=(0, 1)),
  verbose=1)

# %%
# torchview no puede ejecutar el circuito: sus tensores no son un backend de PennyLane.
class VQC(nn.Module):
  def forward(self, x):
    return x

q_layer = model.q_layer
model.q_layer = VQC()
try:
  import os
  os.environ["PATH"] = r"C:\Program Files\Graphviz\bin" + os.pathsep + os.environ["PATH"]
  from torchview import draw_graph
  graph = draw_graph(
      model,
      input_size=(1, n_ventana, 3),
      expand_nested=True,
      graph_name="HybridLSTM",
  )
  graph.visual_graph.render("hybrid_lstm", format="png")
  from IPython.display import Image, display
  display(Image("hybrid_lstm.png"))
finally:
  model.q_layer = q_layer

# %%
