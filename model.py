from layer import Layer
import math
import numpy as np


class Model():

	def __init__(self, activation_func, initialization, numof_layers, layer_sizes, dropout=0, seed=-1, batch_size=1):
		self.activation_function = activation_func
		self.initialization = initialization
		self.numof_layer = numof_layers
		self.layer_sizes = layer_sizes
		self.dropout=dropout
		self.seed = seed
		self.batch_size = batch_size



	def cross_entropy(self, Y, probs):
		return np.asarray(list(map(np.average, np.dot(np.asarray(Y), np.asarray(probs).transpose()))))



