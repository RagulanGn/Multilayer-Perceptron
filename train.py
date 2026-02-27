import pandas as pd
import numpy as np
import argparse

class layer():
	def __init__(self, in_size, nb_neuron, activation_function, activation_function_dx):
		self.in_size = in_size
		self.nb_neuron = nb_neuron
		# self.weight = np.random.randint(10, size=(nb_neuron, in_size)) + 1		# Need to update th weight in a good way 
		self.weight = np.random.normal(0, np.sqrt(2 / in_size), size=(nb_neuron, in_size))
		self.bias = np.zeros(nb_neuron)
		self.activation_function = activation_function
		self.activation_function_dx = activation_function_dx

class model():
	def __init__(self, loss_function, loss_function_dx, epoch, data, y):
		self.layer_list = []
		self.loss_function = loss_function
		self.loss_function_dx = loss_function_dx
		self.epoch = epoch
  
		self.data = data[:int(len(data) * 0.8)]
		self.data_val = data[int(len(data) * 0.8):]
		self.y = y[:int(len(data) * 0.8)]
		self.y_val = y[int(len(data) * 0.8):]

	def add_layer(self, nb_neuron, activation_function, activation_function_dx):
		if not self.layer_list :
			l = layer(len(self.data.columns), nb_neuron, activation_function, activation_function_dx)
		else :
			l = layer(self.layer_list[-1].nb_neuron, nb_neuron, activation_function, activation_function_dx)
		self.layer_list.append(l)
		return (l)

	def train(self):
		lr = 0.01																								#learning rate
		i = 0
		self.epoch = 1 #To remove (for testing)
		while (i < self.epoch):
			prediction = self.feed_forward(self.data)

			self.back_propagation(prediction, lr)
			val_loss = self.validation_step()
			if i % 100 == 0:
				loss = np.mean(self.loss_function(prediction, self.y))
				print(f"Epoch {i}, Loss: {loss:.4f}")
			i += 1
		print(f"Final prediction:\n{prediction}")

	def feed_forward(self, data):
		prediction = data
		for layer in self.layer_list:
			# print(f"prediction.shape : {prediction.shape}")
			# print(f"layer.weight.shape : {layer.weight.shape}")
			layer.z = prediction @ layer.weight.T + layer.bias
			# print(f"layer.z.shape : {layer.z.shape}")
			prediction = layer.activation_function(layer.z)
			layer.prediction = prediction
			# print(f"2 prediction.shape : {layer.z.shape}")
		print("End of feed forward \n")
		return (prediction)

	def back_propagation(self, prediction, lr):
		loss = self.loss_function(prediction, self.y)

		# Output Layer 
		previous_layer = self.layer_list[-1]
		da = previous_layer.activation_function_dx(previous_layer.z)
		dLoss = self.loss_function_dx(self.y, previous_layer.prediction)
		if (da.ndim > 2) : #(previous_layer.activation_function_dx() == "Matrix"):
			previous_layer.delta = np.squeeze(dLoss[:, np.newaxis, :] @ da)
		else:
			previous_layer.delta = dLoss * da
		
		# Hidden Layer
		for i in reversed(range(len(self.layer_list))):
			hidden_layer = self.layer_list[i]

			if (i == 0):
				layer_input = self.data
			else :
				layer_input = self.layer_list[i - 1].prediction
			# hidden_layer.delta = previous_layer.delta @ previous_layer.	* hidden_layer.activation_function_dx(hidden_layer.z)
			grad_W = hidden_layer.delta.T @ layer_input / len(self.data)
			grad_B = np.sum(hidden_layer.delta, axis=0) / len(self.data)
			if (i > 0):
				prev_layer = self.layer_list[i-1]
				# Chain Rule: (Current Delta @ Current Weights) * Derivative of previous activation
				prev_layer.delta = (hidden_layer.delta @ hidden_layer.weight) * prev_layer.activation_function_dx(prev_layer.z)
			hidden_layer.weight = hidden_layer.weight - (lr * grad_W)
			hidden_layer.bias = hidden_layer.bias - lr * grad_B
			if np.all(grad_W == 0):
				print(f"⚠️ Layer {i} gradient is ZERO. Weights won't move.")
		# print(f"Gradient_W = {grad_W}")

	def validation_step(self):
		prediction = self.feed_forward(self, self.data_val)
		loss = self.loss_function(prediction, self.y_val)
		return loss

#------------------------------Loss Function------------------------------------#

def distance(x, y):
    return (y - x) ** 2

def distance_dx(x, y):
    """ Partial Derivative of the loss function distance along the x variable"""
    return (2 * (y - x))

def binary_cross_entropy(y, x):
	ep = 1e-6
	return (-(y * np.log(x + ep) + (1 - y) * np.log(1 - x + ep)))

def binary_cross_entropy_dx(y, x):
	ep = 1e-6
	return ((x - y) / (x * (1 - x) + ep))

#------------------------------Activation Function------------------------------#

def softmax(x):																#X being entire Z (Matrix with all input)
	exp_x = np.exp(x - np.max(x, axis=1))								#To avoid big exponential
	return exp_x/np.sum(exp_x, axis=1)

#Derivation of soft max :
def softmax_dx(x):
	s = softmax(x)
	batch, n = s.shape
	jac = np.zeros((batch, n, n))
	for i in range(batch):
		row = s[i]
		jac[i] = np.diagflat(row) - row @ row.T
	return jac

#------------------------------Activation Function------------------------------#
	
def ReLU(x):
	return np.maximum(0, x)

def ReLU_dx(x):
    """ Derivative of the ReLU function """
    return ((np.array(x) > 0).astype(int))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_dx(x):
	s = sigmoid(x)
	return s * (1 - s)

#--------------------------------Main--------------------------------------#

# python train.py --layer 24 24 24 --epochs 84 --loss categoricalCrossentropy --batch_size 8 --learning_rate 0.0314

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("--layer", nargs="+", type=int, help="List of number of neuron in each layer", required=True)
	parser.add_argument("--epochs", type=int, help="Number of epoch", required=True)
	parser.add_argument("--loss", type=str, help="Lost function used", required=True)
	parser.add_argument("--batch_size", type=int, help="Size of batch", required=True)
	parser.add_argument("--learning_rate", type=float, help="learning rate", required=True)
	args = parser.parse_args()

	df = pd.read_csv("data_train.csv")
	y = df.iloc[:,-2:]
	data = df.iloc[:,:-2]			#maybe drop also 0 i think its the id
	if args.loss == "categoricalCrossentropy":
		perceptron = model(loss_function=binary_cross_entropy, loss_function_dx=binary_cross_entropy_dx, epoch=args.epochs, data=data, y=y)
	for i in args.layer:
		perceptron.add_layer(i, sigmoid, sigmoid_dx)
	perceptron.add_layer(i, softmax, softmax_dx) #output layer

	perceptron.train()
	return

if __name__ == "__main__":
    main()
    
#Split data
	#split data

#train
	#Compute weight (training)
	#Plot Loss and accuracy (for val and train)
	#Save weight + mean and std + argv

#predict
	#Predict the result