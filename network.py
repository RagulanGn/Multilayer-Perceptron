import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from autograd import Value, no_grad
import pandas as pd

from ft_function import softmax, ReLU

ACTIVATION_FUNCTION = {
	"ReLU" : ReLU,
	"softmax" : softmax
}

GRAPH_DIR = "graphs"
ARTEFACTS_DIR = "artefacts"

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

		self.training_loss = []
		self.training_acc = []
		self.training_rmse = []
		self.training_mae = []

		self.validation_loss = []
		self.validation_acc = []
		self.validation_rmse = []
		self.validation_mae = []

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
			rmse_batch = []
			mae_batch = []
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
				rmse_batch.append(np.sqrt(np.mean((prediction.number - batch_y.number) ** 2)))
				mae_batch.append(np.mean(np.abs(prediction.number - batch_y.number)))

			self.training_loss.append(np.mean(loss_batch))
			self.training_acc.append(np.mean(acc_batch))
			self.training_rmse.append(np.mean(rmse_batch))
			self.training_mae.append(np.mean(mae_batch))
			with no_grad():
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
			prediction = ACTIVATION_FUNCTION[layer.activation_function](z)
		return (prediction)

	def back_propagation(self, lr):
		for p in self.parameters():
			p.number += -lr * p._grad

	def validation_step(self):
		data_val, y_val = self.dataloader.val_dataloader()


		loss_batch = []
		acc_batch = []
		rmse_batch = []
		mae_batch = []
		for j in range(0, len(data_val), self.batch_size):
			batch = Value(data_val[j:j + self.batch_size])
			batch_y = Value(y_val[j:j + self.batch_size])
	
			#Feed Forward
			prediction = self.feed_forward(batch)
			loss = self.loss_function(prediction, batch_y)
	
			#Appends Metrics
			loss_batch.append(loss.number)
			pred_labels = prediction.number.argmax(axis=1)
			true_labels = batch_y.number.argmax(axis=1)
			acc_batch.append((pred_labels == true_labels).mean())
			rmse_batch.append(np.sqrt(np.mean((prediction.number - batch_y.number) ** 2)))
			mae_batch.append(np.mean(np.abs(prediction.number - batch_y.number)))
		pred_labels = prediction.number.argmax(axis=1)
		true_labels = y_val.argmax(axis=1)
		self.validation_loss.append(np.mean(loss_batch))
		self.validation_acc.append(np.mean(acc_batch))
		self.validation_rmse.append(np.mean(rmse_batch))
		self.validation_mae.append(np.mean(mae_batch))

	def parameters(self):
		return [p for layer in self.layer_list for p in layer.parameters()]

	def plot_graph(self, name, training_metrics, validation_metrics, show_graph_flag=True):
		plt.clf()
		plt.plot(training_metrics, label=f"Training {name}")
		plt.plot(validation_metrics, label=f"Validation {name}")
		plt.legend()
		plt.title(f"{name} comparison between training and validation")
		plt.xlabel("epoch")
		plt.ylabel(f"{name}")
		plt.savefig(f"{GRAPH_DIR}/{name}.png")
		if show_graph_flag:
			plt.show()

	def show_graph(self, show_graph_flag):
		Path(GRAPH_DIR).mkdir(exist_ok=True)

		self.plot_graph("Loss", self.training_loss, self.validation_loss, show_graph_flag)
		self.plot_graph("Accuracy", self.training_acc, self.validation_acc, show_graph_flag)
		self.plot_graph("RMSE", self.training_rmse, self.validation_rmse, show_graph_flag)
		self.plot_graph("MAE", self.training_mae, self.validation_mae, show_graph_flag)

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

	def get_next_id(self, parameters_path):
		if not parameters_path.is_file():
			return 0
		parameters = pd.read_csv(parameters_path)
		if "id" not in parameters.columns or parameters.empty:
			return 0
		return int(parameters["id"].max()) + 1

	def save_metrics(self):
		Path(ARTEFACTS_DIR).mkdir(exist_ok=True)
		parameters_path = Path(f"{ARTEFACTS_DIR}/parameters.csv")
		metrics_path = Path(f"{ARTEFACTS_DIR}/metrics.csv")

		id = self.get_next_id(parameters_path)
		metrics_df = pd.DataFrame({
			"id" : id, 
			"train loss" : self.training_loss,
			"train acc" : self.training_acc,
			"train mae" : self.training_mae,
			"train rmse" : self.training_rmse,
			"val loss" : self.validation_loss,
			"val acc" : self.validation_acc,
			"val mae" : self.validation_mae,
			"val rmse" : self.validation_rmse
			})
		parameters_df = pd.DataFrame({
			"id" : id,
			"layers" : str([layer.in_size for layer in self.layer_list] + [self.layer_list[-1].out_size]),
			"epochs" : self.epoch,
			"learning rate" : self.learning_rate,
			"batch size" : self.batch_size,
			"loss function" : self.loss_name
		}, index=[0])
		
		metrics_df.to_csv(metrics_path, mode="a", header=not metrics_path.is_file(), index=False)
		parameters_df.to_csv(parameters_path, mode="a", header=not parameters_path.is_file(), index=False)
		
	def show_graph_model_comparison(self, show_graph_flag):
		metrics_df = pd.read_csv(f"{ARTEFACTS_DIR}/metrics.csv")

		colors = plt.get_cmap("tab20").colors
		for metric in ["loss", "acc", "mae", "rmse"]:	
			plt.clf()
			for id, model in metrics_df.groupby("id"):
				color = colors[id % len(colors)]
				epochs = range(len(model))
				plt.plot(epochs, model[f"train {metric}"], color=color)
				plt.plot(epochs, model[f"val {metric}"], linestyle="--", color=color)
			plt.title(f"{metric} comparison between training and validation")
			plt.xlabel("epoch")
			plt.ylabel(f"{metric}")
			plt.savefig(f"graphs/{metric}_model_comparison.png")
			if show_graph_flag:
				plt.show()
			

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
