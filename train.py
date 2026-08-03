import matplotlib
import pandas as pd
import numpy as np
import argparse

from ft_function import binary_cross_entropy, categorical_cross_entropy
from early_stopping import EarlyStopping
from optimizer import NesterovMomentum, Adam
from network import MLP, MLPDataLoader

np.random.seed(42) #Set the seed for the whole project

LOSS_FUNCTIONS = {
	"binaryCrossentropy": binary_cross_entropy,
	"categoricalCrossentropy": categorical_cross_entropy
}

OPTIMIZERS = {
	"Nesterov": NesterovMomentum,
	"Adam": Adam
}

def main():
	parser = argparse.ArgumentParser(prog="train",
		description="""Small MLP from scratch (numpy), can modify numbers of layers, epoch, batch size, learning rate,
optimizer and earlystopping with the args of the program. You can also modify more EarlyStopping parameters directly in the code""",
		epilog="Code by Ragulan (Github: RagulanGn)")

	parser.add_argument("--layer", nargs="+", type=int, help="List of number of neuron in each layer", required=True)
	parser.add_argument("--epochs", type=int, help="Number of epoch", required=True)
	parser.add_argument("--loss", type=str, choices=["binaryCrossentropy", "categoricalCrossentropy"], help="Lost function used", required=True)
	parser.add_argument("--batch_size", type=int, help="Size of batch", required=True)
	parser.add_argument("--learning_rate", type=float, help="learning rate", required=True)
	parser.add_argument("--optimizer", type=str, choices=["Nesterov", "Adam"], help="Optimizer choice between Nesterov or Adam")
	parser.add_argument("--early_stopping", type=int, help="Early stopping patience")
	parser.add_argument("--train_dataset", type=str, help="Path of train dataset", default="datasets/data_train.csv")
	parser.add_argument("--val_dataset", type=str, help="Path of val dataset", default="datasets/data_val.csv")
	parser.add_argument("--hide_graphs", action="store_false", help="Hide all graphs (Still saved)")
	args = parser.parse_args()

	try:
		df_train = pd.read_csv(args.train_dataset, header=None, index_col=0)
		df_val = pd.read_csv(args.val_dataset, header=None, index_col=0)
	except Exception as e:
		parser.error(str(e))

	df_val = pd.get_dummies(df_val, columns=[1], dtype=int)
	df_train = pd.get_dummies(df_train, columns=[1], dtype=int)

	dataloader = MLPDataLoader(df_train, df_val)

	loss = LOSS_FUNCTIONS[args.loss]
	
	model = MLP(
	epoch=args.epochs, 
	batch_size=args.batch_size, 
	learning_rate=args.learning_rate,
	dataloader=dataloader,
	loss_function=loss,
	loss_name=args.loss)

	optimizer = None
	if args.optimizer :
		OPTIMIZERS[args.optimizer](args.learning_rate)

	if (len(args.layer) < 2):
		print(f"Need atleast 2 hidden layers, default to [8, 8]")
		args.layer = [8, 8]

	EarlyStop = None
	if args.early_stopping:
		EarlyStop = EarlyStopping(patience=args.early_stopping, min_delta=0.0, mode='min')
	layers_sizes = [dataloader.features_len, *args.layer, 2]

	for i in range(len(layers_sizes) - 1):
		activation_function = "ReLU" if i < len(layers_sizes) - 2 else "softmax"
		model.add_layer(layers_sizes[i], layers_sizes[i + 1], activation_function)

	model.train(EarlyStop=EarlyStop, optimizer=optimizer)
	model.show_graph(args.hide_graphs)
	model.export_npz()
	model.save_metrics()
	model.show_graph_model_comparison(args.hide_graphs)

	return

if __name__ == "__main__":
    main()