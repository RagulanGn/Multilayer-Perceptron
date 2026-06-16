import numpy as np
from autograd import Value

#------------------------------Loss Function------------------------------------#

def distance(experiment, truth):
	return (truth - experiment) ** 2

def binary_cross_entropy(prediction: Value, truth: Value) -> Value:
	loss = -(truth * prediction.log() + (1.0 - truth) * (1.0 - prediction).log()).sum()
	return loss * (1.0 / prediction.number.shape[0])

def binary_cross_entropy_no_grad(prediction: float, truth: float) -> float:
	loss = -(truth * np.log(prediction) + (1.0 - truth) * np.log(1.0 - prediction)).sum()
	return loss * (1.0 / prediction.shape[0])

def categorical_cross_entropy(experiment, truth):
	ep = 1e-7
	return -np.sum(truth * np.log(experiment + ep), axis=1)

#------------------------------Activation Function------------------------------#

def softmax(x):
	exp_x = (x - np.max(x.number, axis=1, keepdims=True)).exp()
	return exp_x / exp_x.sum(axis=1, keepdims=True)

def softmax_nograd(x):#X being entire Z (Matrix with all input)
	x = np.array(x)
	exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))#To avoid big exponential
	return exp_x / np.sum(exp_x, axis=1, keepdims=True)

#Derivation of soft max :
# def softmax_dx(x):
# 	s = softmax(x)
# 	batch, n = s.shape
# 	jac = np.zeros((batch, n, n))
# 	for i in range(batch):
# 		row = s[i]
# 		jac[i] = np.diagflat(row) - np.outer(row,row)
# 		# jac[i] = np.diagflat(row) - row @ row.T
# 	return jac

#------------------------------Activation Function------------------------------#

def ReLU(x):
	return np.maximum(0, x)

def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))
