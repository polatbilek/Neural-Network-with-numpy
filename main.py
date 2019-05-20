from layer import Layer
from model import Model
import numpy as np
import random

def data_gen():
	mu1 = [0,0]
	cov1 = [[1,0],[0,1]]

	mu2 = [1.5, 1]
	cov2= [[0.8, 0.5],[0.4,0.8]]

	test_size = 500
	train_size = 1500

	train_1 = np.random.multivariate_normal(mu1, cov1, train_size)
	train_label1 = [[0, 1] for i in range(train_size)]
	train_2 = np.random.multivariate_normal(mu2, cov2, train_size)
	train_label2 = [[1, 0] for i in range(train_size)]

	train_x = train_1.tolist() + train_2.tolist()
	train_y = train_label1 + train_label2

	c = list(zip(train_x, train_y))

	random.shuffle(c)

	train_x, train_y = zip(*c)

	test_1 = np.random.multivariate_normal(mu1, cov1, test_size)
	test_label1 = [[0, 1] for i in range(test_size)]
	test_2 = np.random.multivariate_normal(mu2, cov2, test_size)
	test_label2 = [[1, 0] for i in range(test_size)]

	test_x = test_1.tolist() + test_2.tolist()
	test_y = test_label1 + test_label2

	c = list(zip(test_x, test_y))

	random.shuffle(c)

	test_x, test_y = zip(*c)

	return train_x, train_y, test_x, test_y

model = Model(activation_func="relu",
		  initialization="normal",
		  layer_sizes=[2,5,5,2],
		  num_classes=2,
		  num_epoch=3,
		  objective_function="softmax",
		  dropout=0,
		  seed=-1,
		  batch_size=3,
		  learning_rate=0.01)


train_x, train_y, test_x, test_y = data_gen()

model.train(train_x, train_y)
print(model.test(test_x, test_y))


