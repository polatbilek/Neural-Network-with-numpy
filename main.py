from layer import Layer
from model import Model

m = Model(activation_func="tanh",
		  initialization="normal",
		  numof_layers=24,
		  layer_sizes=[5,7,7,3],
		  num_classes=3,
		  objective_function="softmax",
		  dropout=0,
		  seed=-1,
		  batch_size=1,
		  learning_rate=0.01)

print(m.cross_entropy([[0,1]], [[0.2, 0.8]]))

