from layer import Layer
import math
import numpy as np


class Model():

	def __init__(self, activation_func, initialization, learning_rate, layer_sizes, num_classes, objective_function, dropout=0, seed=-1, batch_size=1):
		self.activation_function = activation_func
		self.initialization = initialization
		self.num_of_layers = len(layer_sizes)
		self.layer_sizes = layer_sizes
		self.dropout = dropout
		self.seed = seed
		self.batch_size = batch_size
		self.num_classes = num_classes
		self.learning_rate = learning_rate
		self.objective_function = objective_function
		self.memory = {}

		self.layers = []
		for layer in range(self.num_of_layers):
			if layer+1 != self.num_of_layers:
				self.layers.append(Layer(in_size = self.layer_sizes[layer],
										 out_size = self.layer_sizes[layer+1],
										 activation_func = self.activation_function,
										 initialization = self.initialization,
										 dropout = self.dropout,
										 seed = self.seed,
										 batch_size = self.batch_size,
										 learning_rate = self.learning_rate
										 ))
			else:
				self.layers.append(Layer(in_size = self.layer_sizes[layer],
										 out_size = self.num_classes,
										 activation_func = self.objective_function,
										 initialization = self.initialization,
										 dropout = self.dropout,
										 seed = self.seed,
										 batch_size = self.batch_size,
										 learning_rate = self.learning_rate
										 ))



	def cross_entropy(self, Y, probs):

		if self.batch_size != 1:
			loss = np.zeros(shape=(self.batch_size, self.num_classes))

			for batch in range(self.batch_size):
				error = -1*np.log(np.dot(np.asarray(Y[batch]), np.asarray(probs[batch]).transpose()))
				loss[batch][np.argmax(Y[batch])] = error

		else:
			loss = np.zeros(self.num_classes)
			error = -1*np.log(np.dot(np.asarray(Y), np.asarray(probs).transpose()))
			loss[np.argmax(Y)] = error

		return loss


	def cross_entropy_backward(self, Y, probs):
		if self.batch_size != 1:
			loss = np.zeros(shape=(self.batch_size, self.num_classes))

			for batch in range(self.batch_size):
				error = -1*np.log(np.dot(np.asarray(Y[batch]), np.asarray(probs[batch]).transpose()))
				loss[batch][np.argmax(Y[batch])] = error

		else:
			loss = np.zeros(self.num_classes)
			error = -1*np.log(np.dot(np.asarray(Y), np.asarray(probs).transpose()))
			loss[np.argmax(Y)] = error

		return loss

	# for each layer outs object will hold a list in this form [Z, activation(Z)]
	def forward(self, X):
		self.memory = {}
		prev_activated_z = X
		predictions = 0

		# here each layer is its neurons and weights, so it will hold the neuron values and Z value after forwarding
		for layer in self.num_of_layer:
			activated_z, z = self.layers[layer].forward(prev_activated_z)
			self.memory[str(layer)] = [prev_activated_z, z]
			prev_activated_z = activated_z

			if layer+1 == self.num_of_layer:
				return activated_z


	def backward(self):
		pass

	def train(self, X, Y):
		pass

	def accuracy(self, prediction_probs, Y):
		predictions = np.zeros((np.shape(prediction_probs)))

		for i in range(len(prediction_probs)):
			predictions[i][np.argmax(prediction_probs[i])] = 1

		true_pred = 0

		for prediction in predictions:
			if np.argmax(prediction) == np.argmax(Y):
				true_pred += 1

		return true_pred/len(predictions)


	def test(self, X, Y):
		final_accuracy = 0
		final_loss = 0
		total_num_of_data = 0

		for batch in range(math.ceil(len(X)/self.batch_size)):
			if batch == 0:
				preds = self.forward(X[0:batch*self.batch_size])
				loss = self.cross_entropy(preds, Y[0:batch*self.batch_size])

			elif batch+1 == math.ceil(len(X)/self.batch_size):
				preds = self.forward(X[batch*self.batch_size:-1])
				loss = self.cross_entropy(preds, Y[batch*self.batch_size:-1])

			else:
				preds = self.forward(X[(batch-1)*self.batch_size: batch*self.batch_size])
				loss = self.cross_entropy(preds, Y[(batch-1)*self.batch_size: batch*self.batch_size])

			accuracy = accuracy(preds, Y)

			final_accuracy += accuracy
			final_loss += loss
			total_num_of_data += len(preds)


		return final_loss/total_num_of_data, final_accuracy/total_num_of_data





