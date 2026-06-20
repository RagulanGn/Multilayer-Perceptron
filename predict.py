import pandas as pd
import argparse
import numpy as np
from ft_function import binary_cross_entropy_no_grad, categorical_cross_entropy_no_grad
from network import MLP

FEATURES_NB = 31

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("test_dataset")
	args = parser.parse_args()
	try :
		df_test = pd.read_csv(args.test_dataset)
	except Exception as e:
		parser.error(str(e))

	npz_file = np.load("artefacts.npz", allow_pickle=True)
	MLPpredict = MLP()
	for i in range(len(npz_file['topology'])):
		MLPpredict.add_layer(npz_file['topology'][i][0], npz_file['topology'][i][1], npz_file['activation_function'][i], npz_file['weight'][i], npz_file['bias'][i])

	if df_test.shape[0] < FEATURES_NB:
		df_test = (df_test - npz_file['mean']) / npz_file['std']
		prediction = MLPpredict.feed_forward_no_grad(df_test)
		print(prediction)
	else :
		df_real = df_test.iloc[:,-2:].values.astype(np.float64)
		df_test = df_test.iloc[:,:-2].values.astype(np.float64)
		df_test = (df_test - npz_file['mean']) / npz_file['std']

		prediction = MLPpredict.feed_forward_no_grad(df_test)
		if npz_file['loss_function'] == 'binaryCrossentropy' :
			loss = binary_cross_entropy_no_grad(prediction, df_real)
		else :
			loss = categorical_cross_entropy_no_grad(prediction, df_real)
		print(loss)

if __name__ == "__main__":
	main()