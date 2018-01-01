from Logistic_Regres import *
import numpy as np

def ascii_show(img):
	for y in img:
		row = ""
		for x in y:
			row += "{0: <4}".format(x)
		print(row)

def vis_imgs(data, labels, size):
	for k in range(size * size):
		plt.subplot(size, size, k+1)
		plt.title("Label : {label}".format(label = labels[k]), fontsize = 10)
		plt.imshow(data[k].reshape((28, 28)), cmap='gray')
	plt.show()

def plot_losses(losses):
	X = np.array([i+1 for i, _ in enumerate(losses)])
	losses = np.array(losses)

	plt.title("loss function over training epochs")
	plt.xlabel("number of epochs")
	plt.ylabel("loss function value")
	plt.plot(X, losses, color = "red")
	plt.show()

def main():
	size, rows, cols, training_data, training_labels = read_data("train")

	temp = np.random.randint(0, 50000)
	ascii_show(np.reshape(training_data[temp], (rows, cols)))

	print("Do you want to run the visualization on the head of the data? Yes : Y")
	vis = input()
	if vis == "Y":
		vis_imgs(training_data[:16], training_labels[:16], 4)

if __name__ == '__main__':
	main()