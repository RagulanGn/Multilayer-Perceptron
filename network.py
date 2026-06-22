import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from autograd import Value

from ft_function import softmax, ReLU, softmax_nograd

class Layer():
	def __init__(self, in_size, out_size, activation_function, weight, bias):
		self.in_size = in_size
		self.out_size = out_size
		self.weight = weight if weight is not None else Value(np.random.normal(0, np.sqrt(2 / (in_size + out_size)), size=(out_size, in_size)))
		self.bias = bias if bias is not None else Value(np.zeros(out_size))
		self.activation_function = activation_function

	def parameters(self):
		return [self.weight, self.bias]

class MLP():
	def __init__(self, epoch=None, learning_rate=None, batch_size=None, dataloader=None, loss_function=None, loss_name=None):
		self.layer_list = []
		self.epoch = epoch
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.dataloader = dataloader
		self.loss_function = loss_function
		self.loss_name = loss_name

		self.validation_loss = []
		self.training_loss = []
		self.validation_acc = []
		self.training_acc = []

	def add_layer(self, in_size, out_size, activation_function, weight=None, bias=None):
		l = Layer(in_size, out_size, activation_function, weight, bias)
		self.layer_list.append(l)
		return (l)

	def train(self, optimizer=None, EarlyStop=None):
		i = 0
		while (i < self.epoch):
			data, y = self.dataloader.train_dataloader()
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
				if optimizer is None :
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
			z = prediction @ layer.weight.T + layer.bias
			if layer.activation_function == "ReLU":
				prediction = z.ReLU()
			elif layer.activation_function == "softmax":
				prediction = softmax(z)
		return (prediction)

	#Feed forward with no grad / plain numpy no Value object
	def feed_forward_no_grad(self, data):
		prediction = data
		for layer in self.layer_list:
			z = prediction @ layer.weight.T + layer.bias
			if layer.activation_function == "ReLU":
				prediction = ReLU(z)
			elif layer.activation_function == "softmax":
				prediction = softmax_nograd(z)
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
		plt.clf()
		plt.plot(self.validation_acc, label='Validation accuracy')
		plt.plot(self.training_acc, label='Training accuracy')
		plt.legend()
		plt.xlabel("epoch")
		plt.ylabel("Accuracy")
		plt.savefig("graphs/graph2.png")

	def export_npz(self):
		weight = []
		bias = []
		topology = []
		activation_function = []
		for layer in self.layer_list:
			weight.append(layer.weight.number)
			bias.append(layer.bias.number)
			topology.append([layer.in_size, layer.out_size])
			activation_function.append(layer.activation_function)
		np.savez("artefacts.npz", 
			weight=np.array(weight, dtype=object),
			bias=np.array(bias, dtype=object),
			topology=np.array(topology),
			loss_name=self.loss_name,
			activation_function=np.array(activation_function, dtype=object),
			mean=self.dataloader.data_mean,
			std=self.dataloader.data_std)
		return

class MLPDataLoader():
	def __init__(self, data_train, data_val, shuffle=True):
		self.features_train = data_train.iloc[:,:-2].values.astype(np.float64)
		self.y_train = data_train.iloc[:,-2:].values.astype(np.float64)

		self.features_val = data_val.iloc[:, :-2].values.astype(np.float64)
		self.y_val = data_val.iloc[:, -2:].values.astype(np.float64)

		self.shuffle = shuffle

		self.data_mean = np.mean(self.features_train, axis=0)
		self.data_std = np.std(self.features_train, axis=0) + 1e-8
		self.features_train = (self.features_train - self.data_mean) / self.data_std
		self.features_val = (self.features_val - self.data_mean) / self.data_std
		self.features_len = self.features_train.shape[1]

	def train_dataloader(self):
		if (self.shuffle == True):
			indices = np.random.permutation(len(self.features_train))
			return self.features_train[indices], self.y_train[indices]
		return self.features_train, self.y_train


	def val_dataloader(self):
		return self.features_val, self.y_val
