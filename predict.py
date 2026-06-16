import pandas as pd
import argparse
import numpy as np
from ft_function import binary_cross_entropy, softmax, ReLU, softmax_nograd

class layer():
	def __init__(self, in_size, nb_neuron, activation_function, weight, bias):
		self.in_size = in_size
		self.nb_neuron = nb_neuron
		self.weight = weight
		self.bias = bias
		self.activation_function = activation_function

class model():
	def __init__(self):
		self.layer_list = []

	def add_layer(self, nb_neuron, activation_function, weight, bias):
		if not self.layer_list :
			l = layer(31, nb_neuron, activation_function, weight, bias) #HARDCODED
		else :
			l = layer(self.layer_list[-1].nb_neuron, nb_neuron, activation_function, weight, bias)
		self.layer_list.append(l)
		return (l)

	def feed_forward(self, data):
		prediction = data
		for layer in self.layer_list:
			layer.z = prediction @ layer.weight.T + layer.bias
			if layer.activation_function == ReLU:
				prediction = ReLU(layer.z)
			elif layer.activation_function == softmax:
				prediction = softmax_nograd(layer.z)
			layer.prediction = prediction
		return (prediction)

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("test_dataset")
	args = parser.parse_args()
	try :
		df_test = pd.read_csv(args.test_dataset)
	except Exception as e:
		parser.error(str(e))

	npz_file = np.load("artefacts.npz", allow_pickle=True)
	MLPpredict = model()
	for i in range(len(npz_file['topology'])):
		MLPpredict.add_layer(npz_file['topology'][i], npz_file['activation_function'][i], npz_file['weight'][i], npz_file['bias'][i])

	# !!!!!!!!!!! df_val
	# df_val = pd.read_csv("datasets/data_val.csv")
	df_real = df_test.iloc[:,-2:].values.astype(np.float64)
	df_test = df_test.iloc[:,:-2].values.astype(np.float64)
	# !!!!!!!!!!! df_val
	df_test = (df_test - npz_file['mean']) / npz_file['std']
	prediction = MLPpredict.feed_forward(df_test)
	loss = binary_cross_entropy(prediction, df_real)
	print(loss)

if __name__ == "__main__":
    main()