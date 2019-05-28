from model import Model
import numpy as np
import random

def data_gen():
	mu1 = [2,2]
	cov1 = [[0.1,0],
			[0,0.1]]

	mu2 = [-2, -2]
	cov2= [[0.1, 0.1],
		   [0.1,0.1]]

	test_size = 500
	train_size = 2000

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
		  layer_sizes=[2,5,7,5,2],
		  num_classes=2,
		  num_epoch=2,
		  objective_function="softmax",
		  dropout=0,
		  batch_size=1,
		  learning_rate=0.01,
		  print_every=300)


train_x, train_y, test_x, test_y = data_gen() #get a random data to test the implementation if it learns

valid_x = train_x[int(0.8*len(train_x)):-1]
valid_y = train_y[int(0.8*len(train_y)):-1]
train_x = train_x[0:int(0.8*len(train_x))]
train_y = train_y[0:int(0.8*len(train_y))]


model.train(train_x, train_y, valid_x, valid_y)

print("Testing Started...")
test_loss, test_accuracy = model.test(test_x, test_y)
print("!!! Test Results !!!")
print("Test Loss= " + str(test_loss) + ", Test Accuracy= " + str(test_accuracy))


