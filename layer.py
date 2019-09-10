import numpy as np
import sys

class Layer():

	def __init__(self, in_size, out_size, activation_func, initialization, learning_rate, dropout=0, seed=-1, batch_size=1):
		if seed != -1:
			np.random.seed(seed=seed)

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

		self.dropout_prob = dropout
		self.batch_size = batch_size
		self.in_size = in_size
		self.out_size = out_size
		self.learning_rate = learning_rate
		self.activated_output = 0
		self.input = 0

	####################################################################################
	#                                 HELPER FUNCTIONS                                 #
	####################################################################################
	def normalize(self, unnormalized_list, range_min, range_max): #Normalization for softmax, doesn't change the result
		new_list = []

		for element in unnormalized_list:
			new_list.append((((element - max(unnormalized_list)) * (range_max - range_min)) / (max(unnormalized_list) - min(unnormalized_list)))+range_min)

		return new_list

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

	def dropout(self, drop_prob, data):
		if drop_prob != 0:
			if self.batch_size != 1:
				pass

			else:
				probs = np.random.rand(len(data))
				probs[probs <= drop_prob] = 0
				probs[probs > drop_prob] = 1
				data = data * probs

		return data


	####################################################################################
	#                                 FORWARD FUNCTIONS                                #
	####################################################################################
	def relu_forward(self, X):

		self.activated_output = np.maximum(0, X)
		return self.activated_output


	def sigmoid_forward(self, X):
		if self.batch_size != 1:
			self.activated_output = 1 / (1 + np.exp(np.dot(X, -1)))
			return self.activated_output

		else:
			X = np.squeeze(X)

			self.activated_output = 1 / (1 + np.exp(np.dot(X, -1)))
			return self.activated_output



	def tanh_forward(self, X):
		activate = lambda z: (np.power(np.e, z) - np.power(np.e, -z))/(np.power(np.e, z) + np.power(np.e, -z))

		if self.batch_size != 1:
			for batch in range(len(X)):
				X[batch] = list(map(activate, X[batch]))

			self.activated_output = np.asarray(X)
			return self.activated_output

		else:
			self.activated_output = np.asarray(list(map(activate, X)))
			return self.activated_output



	def softmax_forward(self, X):
		probs = []

		if self.batch_size != 1:
			for batch in X:
				exps = np.exp(batch - np.max(batch))  # stable softmax
				probs.append(exps / np.sum(exps))

			self.activated_output = np.asarray(probs)
			return np.asarray(probs)

		else:
			exps = np.exp(X - np.max(X))  # stable softmax
			probs = exps / np.sum(exps)

			self.activated_output = np.asarray(probs)
			return self.activated_output


	####################################################################################
	#                                 BACKWARD FUNCTIONS                               #
	####################################################################################

	def relu_backward(self, X):

		if self.batch_size != 1:
			i = 0

			for batch in X:
				batch[batch > 0] = 1
				batch[batch <= 0] = 0
				X[i] = batch
				i += 1

		else:
			X[X > 0] = 1
			X[X <= 0] = 0

		return X


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


	def softmax_backward(self, X): # We take the derivative of cross-entropy with softmax, so it is already taken
		return X


	####################################################################################
	#                                ITERATION FUNCTIONS                               #
	####################################################################################

	def forward(self, X):
		X = list(X)
		X = np.squeeze(X)
		self.input = X.copy()

		if self.batch_size != 1:
			Z = np.add(np.asarray(X).dot(self.weights), self.bias)

		else:
			Z = np.add(np.asarray(X).dot(self.weights), self.bias)
			Z = np.squeeze(Z)
			layer_output = self.dropout(self.dropout_prob, self.activation_function(Z))
			self.activated_output = layer_output

		return layer_output


	def backward(self, error_signal):

		if self.batch_size != 1:
			pass
		else:
			# First we get rid of all possible empty dimensions then add only 1. ((1,1,2) may happen instead of (1,2))
			self.activated_output = np.squeeze(self.activated_output)
			self.activated_output = np.expand_dims(self.activated_output, axis=0)

			# Then add the empty dimension to be able to take dot product
			error_signal = np.expand_dims(error_signal, axis=0)
			self.input = np.expand_dims(self.input, axis=0)

			delta = np.multiply(error_signal, self.backward_activation(self.activated_output))
			error_to_next_layer = delta.dot(self.weights.transpose())

			self.weights = np.add(self.weights, np.multiply(self.learning_rate, self.input.transpose().dot(delta)))
			self.bias = np.add(self.bias, np.multiply(self.learning_rate, delta))

			error_to_next_layer = np.squeeze(error_to_next_layer)
			self.input = np.squeeze(self.input)

		return error_to_next_layer

