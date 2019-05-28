from layer import Layer
import numpy as np
import sys

class Model():

	def __init__(self, activation_func, initialization, learning_rate, num_epoch, layer_sizes, num_classes, objective_function, dropout=0, seed=-1, batch_size=1, print_every=-1):
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
		self.print_every = print_every

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
					if Y[batch][index] == 1:
						if probs[batch][index] == 0:
							loss = 1000000
						else:
							loss += Y[batch][index]*np.log(probs[batch][index])
					else:
						if probs[batch][index] == 0:
							loss = 1000000
						else:
							loss += (1-Y[batch][index])*np.log((1-probs[batch][index]))

			return loss / (len(Y)*self.num_classes)

		else:
			Y = np.squeeze(Y)
			probs = np.squeeze(probs)
			for index in range(self.num_classes):
				if Y[index] == 1:
					if probs[index] == 0:
						loss = 1000000
					else:
						loss += np.log(probs[index])
				else:
					if probs[index] == 1:
						loss = 1000000
					else:
						loss += np.log((1-probs[index]))

			return loss / self.num_classes


	def cross_entropy_backward(self, Y, probs):

		if self.batch_size != 1:
			loss = np.zeros(shape=(self.batch_size, self.num_classes))

			for batch in range(self.batch_size):
				for index in range(self.num_classes):
					loss[batch][index] = Y[batch][index] - probs[batch][index]

			loss = np.mean(loss, axis=0)

		else:
			Y = np.squeeze(Y)
			probs = np.squeeze(probs)
			loss = np.zeros(self.num_classes)

			for index in range(self.num_classes):
				loss[index] = Y[index] - probs[index]

		return loss


	# for each layer outs object will hold a list in this form [Z, activation(Z)]
	def forward(self, X):
		prev_activated_z = X

		# here each layer is its neurons and weights, so it will hold the neuron values and Z value after forwarding
		for layer in range(self.num_of_layers):
			activated_z = self.layers[layer].forward(prev_activated_z)
			prev_activated_z = activated_z

			if layer+1 == self.num_of_layers:
				return activated_z


	def backward(self, error):
		layer = self.num_of_layers-1

		while layer >= 0:
			error = self.layers[layer].backward(error)
			layer -= 1


	def train(self, train_x, train_y, valid_x, valid_y):
		print("Training Started...")
		train_x = list(train_x)
		train_y = list(train_y)

		for epoch in range(self.num_epoch):
			data = train_x.copy()
			labels = train_y.copy()

			for batch in range(int(np.ceil(len(data)/self.batch_size))):
				if batch+1 == np.ceil(len(data)/self.batch_size):
					if len(data[batch * self.batch_size:-1]) != 0:
						predictions = self.forward(data[batch * self.batch_size:-1])
						loss = self.cross_entropy(labels[batch * self.batch_size:-1], predictions)
						error_signal = self.cross_entropy_backward(Y[(batch - 1) * self.batch_size:-1], predictions)

				else:
					predictions = self.forward(data[batch*self.batch_size:(batch+1)*self.batch_size])
					loss = self.cross_entropy(labels[batch*self.batch_size:(batch+1)*self.batch_size], predictions)
					error_signal = self.cross_entropy_backward(labels[batch*self.batch_size:(batch+1)*self.batch_size], predictions)

				if self.print_every != -1:
					if batch % self.print_every == 0:
						loss, accuracy = self.test(valid_x, valid_y)
						print("Epoch= " + str(epoch) + ", Batch coverage= %" +
								str(100*(batch/int(np.ceil(len(data)/self.batch_size)))) +
								", Loss= " + str(loss) + ", Accuracy= " + str(accuracy))


				self.backward(error_signal)


	def accuracy(self, prediction_probs, Y):
		predictions = np.zeros((np.shape(prediction_probs)))

		if self.batch_size != 1:
			for i in range(len(prediction_probs)):
				predictions[i][np.argmax(prediction_probs[i])] = 1

			true_pred = 0

			for prediction in predictions:
				if np.argmax(prediction) == np.argmax(Y):
					true_pred += 1

			return true_pred/len(predictions)

		else:
			if np.argmax(prediction_probs) == np.argmax(Y):
				return 1
			else:
				return 0



	def test(self, X, Y):

		final_accuracy = 0
		final_loss = 0

		for batch in range(int(np.ceil(len(X)/self.batch_size))):
			if batch+1 == int(np.ceil(len(X)/self.batch_size)):
				if len(X[batch*self.batch_size:-1]) != 0:
					preds = self.forward(X[batch*self.batch_size:-1])
					loss = self.cross_entropy(preds, Y[batch*self.batch_size:-1])

			else:
				preds = self.forward(X[batch*self.batch_size:(batch+1)*self.batch_size])
				loss = self.cross_entropy(preds, Y[batch*self.batch_size:(batch+1)*self.batch_size])

			acc = self.accuracy(preds, Y[batch*self.batch_size:(batch+1)*self.batch_size])

			final_accuracy += acc
			final_loss += loss

		total_num_of_data = int(np.ceil(len(X)/self.batch_size)) * self.batch_size


		return final_loss/total_num_of_data, final_accuracy/total_num_of_data





