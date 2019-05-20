from layer import Layer
import math
import numpy as np
from tqdm import tqdm


class Model():

	def __init__(self, activation_func, initialization, learning_rate, num_epoch, layer_sizes, num_classes, objective_function, dropout=0, seed=-1, batch_size=1):
		self.activation_function = activation_func
		self.initialization = initialization
		self.num_of_layers = len(layer_sizes)-1
		self.layer_sizes = layer_sizes
		self.dropout = dropout
		self.seed = seed
		self.batch_size = batch_size
		self.num_classes = num_classes
		self.learning_rate = learning_rate
		self.objective_function = objective_function
		self.num_epoch = num_epoch

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
		loss = 0

		if self.batch_size != 1:
			for batch in range(len(Y)):
				for index in range(self.num_classes):
					error = Y[batch][index]*np.log(probs[batch][index]) + (1-Y[batch][index])*np.log((1-probs[batch][index]))
					loss += error

			return loss / (len(Y)*self.num_classes)

		else:
			for index in range(len(self.num_classes)):
				error = Y[index] * np.log(probs[index]) + (1-Y[index]) * np.log((1-probs[index]))
				loss += error

			return loss / self.num_classes


	def cross_entropy_backward(self, Y, probs):

		if self.batch_size != 1:
			loss = np.zeros(shape=(len(Y), self.num_classes))

			for batch in range(len(Y)):
				for index in range(self.num_classes):
					error = -1*((Y[batch][index]*(1/probs[batch][index])) +
								((1-Y[batch][index])*(1/(1-probs[batch][index]))))
					loss[batch][index] = error

			loss = np.mean(loss, axis=0)
			return loss

		else:
			loss = np.zeros(self.num_classes)
			for index in range(self.num_classes):
				error = -1 * ((Y[index] * (1 / probs[index])) +
							  ((1 - Y[index]) * (1 / (1 - probs[index]))))
				loss[index] = error

			return loss


	# for each layer outs object will hold a list in this form [Z, activation(Z)]
	def forward(self, X):
		prev_activated_z = X

		# here each layer is its neurons and weights, so it will hold the neuron values and Z value after forwarding
		for layer in range(self.num_of_layers):
			activated_z, _ = self.layers[layer].forward(prev_activated_z)
			prev_activated_z = activated_z

			if layer+1 == self.num_of_layers:
				return activated_z


	def backward(self, error):
		layer = self.num_of_layers-1

		while layer != 0:
			print("eeeeeeeeeeeeeeeeee")
			print(np.shape(self.layers[layer].input))
			print(np.shape(self.layers[layer-1].input))
			print(np.shape(self.layers[layer-2].input))
			print(np.shape(self.layers[layer-3].input))
			print("eeeeeeeeeeeeeeeeee")
			error = self.layers[layer].backward(self.layers[layer-1], error)
			layer -= 1


	def train(self, X, Y):
		X = list(X)
		Y = list(Y)

		for epoch in range(self.num_epoch):
			print("=!=!=!=!  Epoch "+str(epoch)+"  =!=!=!=!")

			for batch in tqdm(range(math.ceil(len(X)/self.batch_size))):
				if batch+1 == math.ceil(len(X)/self.batch_size):
					predictions = self.forward(X[(batch - 1) * self.batch_size:-1])
					loss = self.cross_entropy(Y[(batch - 1) * self.batch_size:-1], predictions)
					error_signal = self.cross_entropy_backward(Y[(batch - 1) * self.batch_size:-1],
															   predictions)

				else:
					predictions = self.forward(X[batch*self.batch_size:(batch+1)*self.batch_size])
					loss = self.cross_entropy(Y[batch*self.batch_size:(batch+1)*self.batch_size], predictions)
					error_signal = self.cross_entropy_backward(Y[batch*self.batch_size:(batch+1)*self.batch_size], predictions)

				self.backward(error_signal)


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





