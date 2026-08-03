import pandas as pd
import argparse
import numpy as np
from ft_function import binary_cross_entropy
from network import MLP
from autograd import no_grad


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("test_dataset")
	args = parser.parse_args()
	try :
		df_test = pd.read_csv(args.test_dataset, header=None, index_col=0)
		npz_file = np.load("artefacts.npz", allow_pickle=True)
	except Exception as e:
		parser.error(str(e))

	df_test = pd.get_dummies(df_test, columns=[1], dtype=int)
	if (not all(pd.api.types.is_numeric_dtype(dtype) for dtype in df_test.dtypes)):
		print("Non numeric colums detectected on test dataset")
		return

	MLPpredict = MLP()
	for i in range(len(npz_file['topology'])):
		MLPpredict.add_layer(npz_file['topology'][i][0], npz_file['topology'][i][1], npz_file['activation_function'][i], npz_file['weight'][i], npz_file['bias'][i])

	df_real = df_test.iloc[:,-2:].values.astype(np.float64)
	df_test = df_test.iloc[:,:-2].values.astype(np.float64)
	df_test = (df_test - npz_file['mean']) / npz_file['std']

	with no_grad():
		prediction = MLPpredict.feed_forward(df_test)
		loss = binary_cross_entropy(prediction, df_real)
	print(loss)

if __name__ == "__main__":
	main()