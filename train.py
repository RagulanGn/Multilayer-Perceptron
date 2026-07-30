import pandas as pd
import numpy as np
import argparse

from ft_function import binary_cross_entropy, categorical_cross_entropy
from early_stopping import EarlyStopping
from optimizer import NesterovMomentum, Adam
from network import MLP, MLPDataLoader

np.random.seed(42) #Set the seed for the whole project
FEATURES_NB = 30
# python train.py --layer 24 24 24 --epochs 84 --loss categoricalCrossentropy --batch_size 8 --learning_rate 0.0314
# python train.py --layer 16 8 8 --epochs 130 --loss binaryCrossentropy --batch_size 8 --learning_rate 0.01 --early_stopping 100
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
	args = parser.parse_args()

	try:
		df_train = pd.read_csv(args.train_dataset)
		df_val = pd.read_csv(args.val_dataset)
	except Exception as e:
		parser.error(str(e))

	if (not all(pd.api.types.is_numeric_dtype(dtype) for dtype in df_train.dtypes)):
		print("Non numeric colums detectected on train dataset")
		return
	if (not all(pd.api.types.is_numeric_dtype(dtype) for dtype in df_val.dtypes)):
		print("Non numeric colums detectected on val dataset")
		return

	dataloader = MLPDataLoader(df_train, df_val)

	if args.loss == "binaryCrossentropy":
		loss = binary_cross_entropy
	else:
		loss = categorical_cross_entropy
	model = MLP(
	epoch=args.epochs, 
	batch_size=args.batch_size, 
	learning_rate=args.learning_rate,
	dataloader=dataloader,
	loss_function=loss,
	loss_name=args.loss)

	optimizer = None
	if args.optimizer == "Adam":
		optimizer = Adam(args.learning_rate, b1=0.9, b2=0.999)
	if args.optimizer == "Nesterov":
		optimizer = NesterovMomentum(args.learning_rate, b1=0.9)

	if (len(args.layer) < 2):
		print("Need atleast 2 hidden layers")
		return

	EarlyStop = None
	if args.early_stopping:
		EarlyStop = EarlyStopping(patience=args.early_stopping, min_delta=0.0, mode='min')
	layers_sizes = [dataloader.features_len, *args.layer, 2]

	for i in range(len(layers_sizes) - 1):
		activation_function = "ReLU" if i < len(layers_sizes) - 2 else "softmax"
		model.add_layer(layers_sizes[i], layers_sizes[i + 1], activation_function)

	model.train(EarlyStop=EarlyStop, optimizer=optimizer)
	model.show_graph()
	model.export_npz()
	return

if __name__ == "__main__":
    main()