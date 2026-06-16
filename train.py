import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

from ft_function import binary_cross_entropy, softmax, ReLU
from autograd import Value
from early_stopping import EarlyStopping
from optimizer import NesterovMomentum, Adam

np.random.seed(42) #Set the seed for the whole project

class layer():
	def __init__(self, in_size, nb_neuron, activation_function):
		self.in_size = in_size
		self.nb_neuron = nb_neuron
		self.weight = Value(np.random.normal(0, np.sqrt(2 / (in_size + nb_neuron)), size=(nb_neuron, in_size)))
		self.bias = Value(np.zeros(nb_neuron))
		self.activation_function = activation_function

	def parameters(self):
		return [self.weight, self.bias]

class model():
	def __init__(self, epoch, learning_rate, batch_size, dataloader, loss_function):
		self.layer_list = []
		self.epoch = epoch
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.dataloader = dataloader
		self.loss_function = loss_function

		self.validation_loss = []
		self.training_loss = []
		self.validation_acc = []
		self.training_acc = []

	def add_layer(self, nb_neuron, activation_function):
		if not self.layer_list :
			l = layer(31, nb_neuron, activation_function) #HARDCODED
		else :
			l = layer(self.layer_list[-1].nb_neuron, nb_neuron, activation_function)
		self.layer_list.append(l)
		return (l)

	def train(self, optimizer=None, EarlyStop=None):
		i = 0
		data, y = self.dataloader.train_dataloader()
		while (i < self.epoch):
			indices = np.random.permutation(len(data))
			data = data[indices]
			y = y[indices]
			loss_batch = []
			acc_batch = []
			for j in range(0, len(data), self.batch_size):
				batch = Value(data[j:j + self.batch_size])
				batch_y = Value(y[j:j + self.batch_size])

				#Feed Forward
				prediction = self.feed_forward(batch)
				loss = self.loss_function(prediction, batch_y)

				#Back Propagation

				for p in self.parameters():
					p._grad = np.zeros_like(p.number)

				loss.backward()
				if optimizer == None :
					self.back_propagation(self.learning_rate)
				else :
					optimizer(self.parameters())
				#Appends Metrics
				loss_batch.append(loss.number)
				pred_labels = prediction.number.argmax(axis=1)
				true_labels = batch_y.number.argmax(axis=1)
				acc_batch.append((pred_labels == true_labels).mean())

			self.training_loss.append(np.mean(loss_batch))
			self.training_acc.append(np.mean(acc_batch))
			self.validation_step()
			
			print(f"epoch {i+1}/{self.epoch} - loss: {self.training_loss[i]} - val_loss: {self.validation_loss[i]} - accuracy: {self.training_acc[i]} - val_accuracy: {self.validation_acc[i]}")
			
			#Early Stopping
			if (EarlyStop and EarlyStop(self.validation_loss[i])):
				print(f"Early Stopping at epoch {i+1} (Patience {EarlyStop.patience})")
				break

			i += 1

	def feed_forward(self, data):
		prediction = data
		for layer in self.layer_list:
			layer.z = prediction @ layer.weight.T + layer.bias
			if layer.activation_function == ReLU:
				prediction = layer.z.ReLU()
			elif layer.activation_function == softmax:
				prediction = softmax(layer.z)
			layer.prediction = prediction
		return (prediction)

	def back_propagation(self, lr):
		for p in self.parameters():
			p.number += -lr * p._grad

	def validation_step(self):
		data_val, y_val = self.dataloader.val_dataloader()

		prediction = self.feed_forward(Value(data_val.astype(np.float64)))
		loss = self.loss_function(prediction, Value(y_val.astype(np.float64)))
		self.validation_loss.append(loss.number)

		pred_labels = prediction.number.argmax(axis=1)
		true_labels = y_val.argmax(axis=1)
		self.validation_acc.append((pred_labels == true_labels).mean())

	def parameters(self):
		return [p for layer in self.layer_list for p in layer.parameters()]

	def show_graph(self):
		Path("graphs").mkdir(exist_ok=True)
		plt.plot(self.training_loss, label='Training Loss')
		plt.plot(self.validation_loss, label='Validation Loss')
		plt.legend()
		plt.xlabel("epoch")
		plt.ylabel("Loss")
		plt.savefig("graphs/graph.png")
		plt.show()
		plt.clf()
		plt.plot(self.validation_acc, label='Validation accuracy')
		plt.plot(self.training_acc, label='Training accuracy')
		plt.legend()
		plt.xlabel("epoch")
		plt.ylabel("Accuracy")
		plt.savefig("graphs/graph2.png")
		plt.show()

	def export_npz(self):
		weight = []
		bias = []
		topology = []
		activation_function = []
		for layer in self.layer_list:
			weight.append(layer.weight.number)
			bias.append(layer.bias.number)
			topology.append(layer.nb_neuron)
			activation_function.append(layer.activation_function)
		np.savez("artefacts.npz", 
			weight=np.array(weight, dtype=object),
			bias=np.array(bias, dtype=object),
			topology=np.array(topology),
			activation_function=np.array(activation_function, dtype=object),
			mean=self.dataloader.data_mean,
			std=self.dataloader.data_std)
		return

class MLPDataLoader():
	"""
	Simple Dataloader (load data and normalize in init)
	Usage: MLPDataLoader(data, test_ratio, val_ratio, shuffle=True)
	"""
	def __init__(self, data_train, data_val, shuffle=True):
		print(f"data_train : {data_train.shape}")
		self.features_train = data_train.iloc[:,:-2].values.astype(np.float64)
		self.y_train = data_train.iloc[:,-2:].values.astype(np.float64)

		self.features_val = data_val.iloc[:, :-2].values.astype(np.float64)
		self.y_val = data_val.iloc[:, -2:].values.astype(np.float64)

		self.shuffle = shuffle

		self.data_mean = np.mean(self.features_train, axis=0)
		self.data_std = np.std(self.features_train, axis=0) + 1e-8
		self.features_train = (self.features_train - self.data_mean) / self.data_std
		self.features_val = (self.features_val - self.data_mean) / self.data_std

	def train_dataloader(self):
		if (self.shuffle == True):
			indices = np.random.permutation(len(self.features_train))
			np.random.shuffle(indices)
			self.features_train = self.features_train[indices]
			self.y_train = self.y_train[indices]
		data = [self.features_train, self.y_train]
		return data

	def val_dataloader(self):
		if (self.shuffle == True):
			indices = np.random.permutation(len(self.features_val))
			np.random.shuffle(indices)
			self.features_val = self.features_val[indices]
			self.y_val = self.y_val[indices]
		data = [self.features_val, self.y_val]
		return data

# python train.py --layer 24 24 24 --epochs 84 --loss categoricalCrossentropy --batch_size 8 --learning_rate 0.0314
# python train.py --layer 16 8 8 --epochs 130 --loss binaryCrossentropy --batch_size 8 --learning_rate 0.01 --early_stopping 100
def main():
	parser = argparse.ArgumentParser(prog="MLP",
		description="""Small MLP from scratch (numpy), can modify numbers of layers, epoch, batch size, learning rate,
optimizer and earlystopping with the args of the program. You can also modify more EarlyStopping parameters directly in the code""",
		epilog="Code by Ragulan (Github: RagulanGn, Discord: .ragux)")

	parser.add_argument("--layer", nargs="+", type=int, help="List of number of neuron in each layer", required=True)
	parser.add_argument("--epochs", type=int, help="Number of epoch", required=True)
	parser.add_argument("--loss", type=str, choices=["binaryCrossentropy"], help="Lost function used", required=True)
	parser.add_argument("--batch_size", type=int, help="Size of batch", required=True)
	parser.add_argument("--learning_rate", type=float, help="learning rate", required=True)
	parser.add_argument("--optimizer", type=str, choices=["Nesterov", "Adam"], help="Optimizer choice between Nesterov or Adam")
	parser.add_argument("--early_stopping", type=int, help="Early stopping patience")
	args = parser.parse_args()

	try:
		df_train = pd.read_csv("datasets/data_train.csv")
		df_val = pd.read_csv("datasets/data_val.csv")
	except Exception as e:
		parser.error(str(e))

	dataloader = MLPDataLoader(df_train, df_val)

	if args.loss == "binaryCrossentropy":
		MLP = model(
		epoch=args.epochs, 
		batch_size=args.batch_size, 
		learning_rate=args.learning_rate,
		dataloader=dataloader,
		loss_function=binary_cross_entropy)

	optimizer = None
	if args.optimizer == "Adam":
		optimizer = Adam(lr=10e-3, b1=0.9, b2=0.99)
	if args.optimizer == "Nesterov":
		optimizer = NesterovMomentum(lr=10e-3, b1=0.9)

	EarlyStop = None
	if args.early_stopping:
		EarlyStop = EarlyStopping(patience=args.early_stopping, min_delta=0.0, mode='min')
	for i in args.layer:
		MLP.add_layer(i, ReLU)
	MLP.add_layer(2, softmax)

	MLP.train(EarlyStop=EarlyStop, optimizer=optimizer)
	MLP.show_graph()
	MLP.export_npz()
	return

if __name__ == "__main__":
    main()