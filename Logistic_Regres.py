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
		return size, rows, cols, testing_data, testing_labels

def vis_imgs(data, labels, size):
	for k in range(size * size):
		plt.subplot(size, size, k+1)
		plt.title("Label : {label}".format(label = labels[k]), fontsize = 10)
		plt.imshow(data[k].reshape((28, 28)), cmap='gray')
	plt.show()

def main():
	size, rows, cols, training_data, training_labels = read_data("train")
	size, rows, cols, testing_data, testing_labels = read_data("test")
	vis_imgs(training_data[:16], training_labels[:16], 4)

if __name__ == '__main__':
	main()