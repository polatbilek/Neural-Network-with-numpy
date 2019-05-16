import numpy as np
import math

class Layer():

	def __init__(self, in_size, out_size, activation_func, initialization, dropout=0, seed=-1, batch_size=1):
		if seed != -1:
			np.random.seed(seed)

		if str.lower(initialization) == "normal":
			self.weights = np.random.normal(0, 1, size=(in_size, out_size))
			self.bias = np.random.normal(0, 1, size=(in_size, out_size))

		elif str.lower(initialization) == "xavier":
			self.weights = np.random.normal(0, math.sqrt(1/in_size), size=(in_size, out_size))
			self.bias = np.random.normal(0, math.sqrt(1/in_size), size=(in_size, out_size))

		else:
			assert("Invalid initizaliation parameter is passed. Valid parameters: normal, xavier")

		if str.lower(activation_func) == "relu":
			self.activation_function = self.relu_forward
			self.bacward_activation = self.relu_back

		elif str.lower(activation_func) == "sigmoid":
			self.activation_function = self.sigmoid_forward
			self.bacward_activation = self.sigmoid_back

		elif str.lower(activation_func) == "tanh":
			self.activation_function = self.tanh_forward
			self.bacward_activation = self.tanh_back

		else:
			assert ("Invalid activation function is passed. Valid functions: relu, sigmoid, tanh")

		self.dropout = dropout
		self.batch_size = batch_size
		self.in_size = in_size
		self.out_size = out_size

	################### FORWARD FUNCTIONS ###################
	def relu_forward(self, X):

		if self.batch_size != 1:
			for batch in range(len(X)):
				for neuron in range(len(X[batch])):
					if X[batch][neuron] < 0:
						X[batch][neuron] = 0

		else:
			for neuron in range(len(X)):
				if X[neuron] < 0:
					X[neuron] = 0

		return X

	def sigmoid_forward(self, X):
		activate = lambda z: 1/(1+math.e**-z)

		if self.batch_size != 1:
			for batch in range(len(X)):
				X[batch] = list(map(activate, X[batch]))

			return np.asarray(X)

		else:
			return np.asarray(list(map(activate, X)))




	def tanh_forward(self, X):
		activate = lambda z: (math.e**z - math.e**-z)/(math.e**z + math.e**-z)

		if self.batch_size != 1:
			for batch in range(len(X)):
				X[batch] = list(map(activate, X[batch]))

			return np.asarray(X)

		else:
			return np.asarray(list(map(activate, X)))


	################### BACKWARD FUNCTIONS ###################
	def relu_back(self, X):
		back = np.zeros((self.batch_size, self.in_size))

		if self.batch_size != 1:
			for batch in range(len(X)):
				for neuron in range(len(X[batch])):
					if X[batch][neuron] > 0:
						back[batch][neuron] = 1

		else:
			for neuron in range(len(X)):
				if X[neuron] > 0:
					back[neuron] = 1

		return back.squeeze()


	def sigmoid_back(self, X):
		back = np.zeros((self.batch_size, self.in_size))
		f = lambda z: 1/(1+math.e**-z)
		backward = lambda x: f(x)*(1-f(x))

		if self.batch_size != 1:
			for batch in range(len(X)):
				back[batch] = list(map(backward, X[batch]))

			return np.asarray(back)

		else:
			return np.asarray(list(map(backward, X))).squeeze()


	def tanh_back(self, X):
		back = np.zeros((self.batch_size, self.in_size))
		f = lambda z: (math.e ** z - math.e ** -z) / (math.e ** z + math.e ** -z)
		backward = lambda x: 1 - (f(x)**2)

		if self.batch_size != 1:
			for batch in range(len(X)):
				back[batch] = list(map(backward, X[batch]))

			return np.asarray(back)

		else:
			return np.asarray(list(map(backward, X))).squeeze()


	################### ITERATION FUNCTIONS ###################
	def forward(self, X):
		if self.batch_size != 1:
			print(np.asarray(X).shape)
			return self.activation_function(np.asarray(X).dot(self.weights) + self.bias)
		else:
			return self.activation_function(np.asarray(X).dot(self.weights) + self.bias)














