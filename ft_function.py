import numpy as np
from autograd import Value

#------------------------------Loss Function------------------------------------#

def distance(prediction, truth):
	return (truth - prediction) ** 2

def binary_cross_entropy(prediction: float, truth: float) -> float:
	eps = 1e-15
	loss = -np.mean((truth * np.log(prediction + eps) + (1.0 - truth) * np.log(1.0 - prediction + eps)))
	return loss 

def categorical_cross_entropy(prediction, truth):
	eps = 1e-15
	loss = -np.mean(truth * np.log(prediction + eps))
	return loss

#------------------------------Activation Function------------------------------#

def softmax(x):
	exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
	return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def ReLU(x):
	return np.maximum(0, x)

def sigmoid(x):
	x = np.clip(x, -50, 50)
	return 1 / (1 + np.exp(-x))

def ReLU(x):
	return x * (x > 0)
