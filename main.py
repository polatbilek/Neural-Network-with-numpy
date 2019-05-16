from layer import Layer
from model import Model

L1 = Layer(in_size=10, out_size=20, activation_func="tanh", initialization="normal", batch_size=1)
m = Model(activation_func="tanh", initialization="normal", numof_layers=2, layer_sizes=[5,2], dropout=0, seed=-1, batch_size=1)

print(m.cross_entropy([[0,1]], [[0.2, 0.8]]))

