import numpy as np
import sys

class Layer():

	def __init__(self, in_size, out_size, activation_func, initialization, learning_rate, dropout=0, seed=-1, batch_size=1):
		if seed != -1:
			np.random.seed(seed)

		if str.lower(initialization) == "normal":
			self.weights = np.random.normal(0, 1, size=(in_size, out_size))
			self.bias = np.random.normal(0, 1, size=(out_size))

		elif str.lower(initialization) == "xavier":
			self.weights = np.random.normal(0, np.sqrt(1/in_size), size=(in_size, out_size))
			self.bias = np.random.normal(0, np.sqrt(1/in_size), size=(out_size))

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
		self.activated_output = 0
		self.input = 0


	def normalize(self, unnormalized_list, range_min, range_max):
		new_list = []

		for element in unnormalized_list:
			new_list.append((((element - max(unnormalized_list)) * (range_max - range_min)) / (max(unnormalized_list) - min(unnormalized_list)))+range_min)

		return  new_list

	def activation_clip(self, data, threshold):
		if self.batch_size != 1:
			pass
		else:
			for element in range(len(data)):
				if data[element] > threshold:
					data[element] = threshold
				elif data[element] < -1*threshold:
					data[element] = -1*threshold

		return data

	################### FORWARD FUNCTIONS ###################
	def relu_forward(self, X):

		if self.batch_size != 1:
			for batch in range(len(X)):
				for neuron in range(len(X[batch])):
					X[batch][neuron] = np.maximum(0, X[batch][neuron])

		else:
			for neuron in range(len(X)):
				X[neuron] = np.maximum(0, X[neuron])

		self.activated_output = X

		return X


	def sigmoid_forward(self, X):
		activate = lambda z: 1/(1+np.exp(-z))

		if self.batch_size != 1:
			for batch in range(len(X)):
				X[batch] = list(map(activate, X[batch]))

			self.activated_output = np.asarray(X)
			return np.asarray(X)

		else:
			X = np.squeeze(X)
			X = self.activation_clip(X, 700)

			self.activated_output = np.asarray(list(map(activate, X)))
			return self.activated_output




	def tanh_forward(self, X):
		activate = lambda z: (np.power(np.e, z) - np.power(np.e, -z))/(np.power(np.e, z) + np.power(np.e, -z))

		if self.batch_size != 1:
			for batch in range(len(X)):
				X[batch] = list(map(activate, X[batch]))

			self.activated_output = np.asarray(X)
			return np.asarray(X)

		else:
			self.activated_output = np.asarray(list(map(activate, X)))
			return np.asarray(list(map(activate, X)))



	def softmax_forward(self, X):
		probs = []

		if self.batch_size != 1:
			denominators = []

			# first, determine denominators to not calculate it for every neuron
			for batch in range(len(X)):
				sum = 0
				X[batch] = self.normalize(X[batch], range_max=200, range_min=-200)
				for neuron in X[batch]:
					sum += np.power(np.e, neuron)

				if sum == 0:
					denominators.append(0.000001)
				else:
					denominators.append(sum)

			# apply softmax to each neuron
			for batch in range(len(X)):
				prob = []
				for neuron in X[batch]:
					prob.append((np.power(np.e, neuron)/denominators[batch]))

				probs.append(prob)

			self.activated_output = np.asarray(probs)
			return np.asarray(probs)

		else:
			denominator = 0

			# first, determine denominator to not calculate it for every neuron
			for neuron in X:
				denominator += np.power(np.e, neuron)

			# apply softmax to each neuron
			for neuron in X:
				probs.append((np.power(np.e, neuron) / denominator))


			self.activated_output = np.squeeze(np.asarray(probs))
			return self.activated_output


	################### BACKWARD FUNCTIONS ###################
	def relu_backward(self, X):
		back = np.zeros(np.shape(X))

		if self.batch_size != 1:
			for batch in range(len(X)):
				for neuron in range(len(X[batch])):
					if X[batch][neuron] > 0:
						back[batch][neuron] = 1

		else:

			X = X[0]
			back = back[0]
			for neuron in range(len(X)):
				if X[neuron] > 0:
					back[neuron] = 1

		return back.squeeze()


	def sigmoid_backward(self, X):
		back = np.zeros((self.batch_size, self.in_size))
		f = lambda z: 1/(1+np.exp(-z))
		backward = lambda x: f(x)*(1-f(x))

		if self.batch_size != 1:
			for batch in range(len(X)):
				back[batch] = list(map(backward, X[batch]))

			return np.asarray(back)

		else:
			return np.asarray(list(map(backward, X))).squeeze()


	def tanh_backward(self, X):
		back = np.zeros((self.batch_size, self.in_size))
		f = lambda z: ((np.power(np.e,z) - np.power(np.e,-z))) / (np.power(np.e,z) + np.power(np.e,-z))
		backward = lambda x: 1 - (f(x)**2)

		if self.batch_size != 1:
			for batch in range(len(X)):
				back[batch] = list(map(backward, X[batch]))

			return np.asarray(back)

		else:
			return np.asarray(list(map(backward, X))).squeeze()


	def softmax_backward(self, X):
		return X


	################### ITERATION FUNCTIONS ###################
	def forward(self, X):
		X = list(X)
		X = np.squeeze(X)
		self.input = X.copy()

		if self.batch_size != 1:
			Z = np.add(np.asarray(X).dot(self.weights), self.bias)
		else:
			Z = np.add(np.asarray(X).dot(self.weights), self.bias)
			Z = np.squeeze(Z)


		return self.activation_function(Z)


	def backward(self, error_signal):

		if self.batch_size != 1:
			pass
		else:
			self.activated_output = np.squeeze(self.activated_output)
			self.activated_output = np.expand_dims(self.activated_output, axis=0)
			error_signal = np.expand_dims(error_signal, axis=0)
			self.input = np.expand_dims(self.input, axis=0)

			delta = np.multiply(error_signal, self.backward_activation(self.activated_output))
			error_to_next_layer = delta.dot(self.weights.transpose())

			self.weights = np.add(self.weights, np.multiply(self.learning_rate, self.input.transpose().dot(delta)))
			self.bias = np.add(self.bias, np.multiply(self.learning_rate, delta))

			error_to_next_layer = np.squeeze(error_to_next_layer)
			self.input = np.squeeze(self.input)
		return error_to_next_layer













