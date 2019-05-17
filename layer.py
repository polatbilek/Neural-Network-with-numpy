import numpy as np
import math

class Layer():

	def __init__(self, in_size, out_size, activation_func, initialization, learning_rate, dropout=0, seed=-1, batch_size=1):
		if seed != -1:
			np.random.seed(seed)

		if str.lower(initialization) == "normal":
			self.weights = np.random.normal(0, 1, size=(in_size, out_size))
			self.bias = np.random.normal(0, 1, size=(out_size))

		elif str.lower(initialization) == "xavier":
			self.weights = np.random.normal(0, math.sqrt(1/in_size), size=(in_size, out_size))
			self.bias = np.random.normal(0, math.sqrt(1/in_size), size=(out_size))

		else:
			assert("Invalid initizaliation parameter is passed. Valid parameters: normal, xavier")

		if str.lower(activation_func) == "relu":
			self.activation_function = self.relu_forward
			self.backward_activation = self.relu_backward

		elif str.lower(activation_func) == "sigmoid":
			self.activation_function = self.sigmoid_forward
			self.backward_activation = self.sigmoid_backward

		elif str.lower(activation_func) == "tanh":
			self.activation_function = self.tanh_forward
			self.backward_activation = self.tanh_backward

		elif str.lower(activation_func) == "softmax":
			self.activation_function = self.softmax_forward
			self.backward_activation = self.softmax_backward

		else:
			assert("Invalid activation function is passed. Valid functions: relu, sigmoid, tanh")

		self.dropout = dropout
		self.batch_size = batch_size
		self.in_size = in_size
		self.out_size = out_size
		self.learning_rate = learning_rate

	################### FORWARD FUNCTIONS ###################
	def relu_forward(self, X):

		if self.batch_size != 1:
			for batch in range(len(X)):
				for neuron in range(len(X[batch])):
					X[batch][neuron] = np.maximum(0, X[batch][neuron])

		else:
			for neuron in range(len(X)):
				X[neuron] = np.maximum(0, X[neuron])

		return X

	def sigmoid_forward(self, X):
		activate = lambda z: 1/(1+np.exp(-z))

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



	def softmax_forward(self, X):
		probs = []

		if self.batch_size != 1:
			denominators = []

			# first, determine denominators to not calculate it for every neuron
			for batch in range(len(X)):
				sum = 0
				for neuron in batch:
					sum += math.pow(math.e, neuron)
				denominators.append(sum)

			# apply softmax to each neuron
			for batch in range(len(X)):
				prob = []
				for neuron in batch:
					prob.append((math.pow(math.e, neuron)/denominators[batch]))

				probs.append(prob)

			return np.asarray(probs)

		else:
			denominator = 0

			# first, determine denominator to not calculate it for every neuron
			for neuron in X:
				denominator += math.pow(math.e, neuron)

			# apply softmax to each neuron
				for neuron in X:
					probs.append((math.pow(math.e, neuron) / denominator))


			return np.asarray(probs)


	################### BACKWARD FUNCTIONS ###################
	def relu_backward(self, err_prop, X):
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


	def sigmoid_backward(self, err_prop, X):
		back = np.zeros((self.batch_size, self.in_size))
		f = lambda z: 1/(1+np.exp(-z))
		backward = lambda x: err_prop*f(x)*(1-f(x))

		if self.batch_size != 1:
			for batch in range(len(X)):
				back[batch] = list(map(backward, X[batch]))

			return np.asarray(back)

		else:
			return np.asarray(list(map(backward, X))).squeeze()


	def tanh_backward(self, err_prop, X):
		back = np.zeros((self.batch_size, self.in_size))
		f = lambda z: (math.e ** z - math.e ** -z) / (math.e ** z + math.e ** -z)
		backward = lambda x: 1 - (f(x)**2)

		if self.batch_size != 1:
			for batch in range(len(X)):
				back[batch] = list(map(backward, X[batch]))

			return np.asarray(back)

		else:
			return np.asarray(list(map(backward, X))).squeeze()


	def softmax_backward(self, err_prop, X):
		if self.batch_size != 1:
			np.max(X)
		else:
			X[np.argmax(X)] = 1-np.max(X)
			return X





	################### ITERATION FUNCTIONS ###################
	def forward(self, X):
		Z = np.add(np.asarray(X).dot(self.weights), self.bias)
		return self.activation_function(Z), Z



	def backward(self, err_prop, X):
		Z = np.add(np.asarray(X).dot(self.weights), self.bias)
		return self.backward_activation(err_prop, Z)













