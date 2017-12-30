import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import struct

dataset_dir = "./Dataset/"
training_data_path = "train-images.idx3-ubyte"
training_labels_path = "train-labels.idx1-ubyte"
test_data_path = "t10k-images.idx3-ubyte"
test_labels_path = "t10k-labels.idx1-ubyte"

def ascii_show(img):
	for y in img:
		row = ""
		for x in y:
			row += "{0: <4}".format(x)
		print(row)

def get_one_hot(labels):
	ret = np.zeros((len(labels), 10))
	ret[np.arange(len(labels)), labels] = 1
	return ret

def read_data(marker):
	if marker == "train":
		with open(dataset_dir + training_data_path, "rb") as fp:
			_, size, rows, cols = struct.unpack(">IIII", fp.read(16))
			training_data = np.fromfile(fp, dtype = np.uint8)
			training_data = np.reshape(training_data, (size, -1))
		# ascii_show(np.reshape(training_data[0], (rows, cols)))
		with open(dataset_dir + training_labels_path, "rb") as fp:
			_, _ = struct.unpack(">II", fp.read(8))
			training_labels = np.fromfile(fp, dtype = np.uint8)
			training_labels = get_one_hot(training_labels)
		return size, rows, cols, training_data, training_labels
	elif marker == "test":
		with open(dataset_dir + test_data_path, "rb") as fp:
			_, size, rows, cols = struct.unpack(">IIII", fp.read(16))
			testing_data = np.fromfile(fp, dtype = np.uint8)
			testing_data = np.reshape(testing_data, (size, -1))
		# ascii_show(np.reshape(training_data[0], (rows, cols)))
		with open(dataset_dir + test_labels_path, "rb") as fp:
			_, _ = struct.unpack(">II", fp.read(8))
			testing_labels = np.fromfile(fp, dtype = np.uint8)
			testing_labels = get_one_hot(testing_labels)
		return size, rows, cols, testing_data, testing_labels

def vis_imgs(data, labels, size):
	for k in range(size * size):
		plt.subplot(size, size, k+1)
		plt.title("Label : {label}".format(label = labels[k]), fontsize = 10)
		plt.imshow(data[k].reshape((28, 28)), cmap='gray')
	plt.show()

def build_model(learning_rate, batch_size, num_epochs):
	X = tf.placeholder(tf.float32, [None, 784]) # raw pixels as features
	Y = tf.placeholder(tf.float32, [None, 10]) # labels

	w = tf.Variable(tf.random_normal(shape=[784, 10], stddev=0.01), name="weights") # weights
	b = tf.Variable(tf.zeros([1, 10]), name="bias") # biases

	logits = tf.matmul(X, w) + b # Predicting Y values

	entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y) # Uses softmax function internally and computes the entropy
	loss = tf.reduce_mean(entropy) # Using mean softmanx entropy as the loss function as it is the unbiased estimate of the loss for softmax

	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss) # Using the basic Gradient Descent Rule to optimize the model.

	return {"w":w, "b":b, "logits":logits, "entropy":entropy, "loss":loss, "optimizer":optimizer, "X":X, "Y":Y}

def train_model(regres_model, training_data, training_labels, batch_size, num_epochs):
	init = tf.global_variables_initializer()
	losses = []
	sess = tf.Session()
	print("training starts")
	sess.run(init)
	num_batches = int(len(training_data)/batch_size)
	for i in range(num_epochs):
		loss = 0
		for _ in range(num_batches):
			idx = np.random.choice(len(training_data), size = batch_size, replace = False)
			x_batch = training_data[idx]
			idx = np.random.choice(len(training_labels), size = batch_size, replace = False)
			y_batch = training_labels[idx]
			_, temp = sess.run([regres_model["optimizer"], regres_model["loss"]], feed_dict = {regres_model["X"]: x_batch, regres_model["Y"]:y_batch})
			loss += temp
		loss /= num_batches
		print("epoch number : {i}, loss = {loss}".format(i = i + 1, loss = loss))
		losses.append(loss)
	return losses, sess

def test_model(regres_model, testing_data, testing_labels, batch_size, sess):
	total_correct = 0
	k = 0
	for x_batch, y_batch in zip(testing_data, testing_labels):
		# print(x_batch.shape, x_batch)
		x_batch = x_batch.reshape((1, len(x_batch)))
		y_batch = y_batch.reshape((1, len(y_batch)))
		_, loss, logits_batch = sess.run([regres_model["optimizer"], regres_model["loss"], regres_model["logits"]], feed_dict = {regres_model["X"]:x_batch, regres_model["Y"]:y_batch})

		soft_logits = tf.nn.softmax(logits_batch)
		predicts_correct = tf.equal(tf.argmax(soft_logits, 1), tf.argmax(y_batch, 1))
		x = sess.run(predicts_correct)
		
		if x[0]:
			total_correct += 1
		k += 1
		if k % 1000 == 0:
			print("{k} thousandth iteration of testing complete".format(k = k))
	accuracy = total_correct/len(testing_data)
	return accuracy

def main():
	size, rows, cols, training_data, training_labels = read_data("train")
	size, rows, cols, testing_data, testing_labels = read_data("test")
	print("data read in memory")
	# vis_imgs(training_data[:16], training_labels[:16], 4)
	
	learning_rate = 0.01
	batch_size = 128
	num_epochs = 25

	regres_model = build_model(learning_rate, batch_size, num_epochs)
	print("model built")
	losses, sess = train_model(regres_model, training_data, training_labels, batch_size, num_epochs)
	print("model trained")
	accuracy = test_model(regres_model, testing_data, testing_labels, batch_size, sess)
	print("model tested. Accuracy = {accuracy}".format(accuracy = accuracy))
	sess.close()

if __name__ == '__main__':
	main()