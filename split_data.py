import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def soft_max():
	return

def data_vis(df):
	# print(df)
	df1 = df.iloc[:,21:]
	df1 = df
	print(df1.describe().to_string())

	# print(df.groupby(1).mean())
	# print(df.groupby(1).median())

	# sns.pairplot(df,hue=1)
	# plt.savefig("pair_plot.png")
	# plt.show()

def main():
	df = pd.read_csv("data.csv", header=None)
	df = df.iloc[:, 1:]

	data_vis(df)

	#Separation 50/50 pour le dataset (train/test)
	df_train = df[:int(0.5 * len(df.index))]
	df_test = df[int(0.5 * len(df.index)):]
	print(f"df train{df_train.iloc[:, 1:].shape}")
	#Standardization
	df_train.iloc[:, 1:] = (df_train.iloc[:, 1:] - df_train.iloc[:, 1:].mean()) / df_train.iloc[:, 1:].std()
	df_test.iloc[:, 1:] = (df_test.iloc[:, 1:] - df_train.iloc[:, 1:].mean()) / df_train.iloc[:, 1:].std()

	df_train = pd.get_dummies(df_train, columns=[1], dtype=int)
	df_test = pd.get_dummies(df_test, columns=[1], dtype=int)

	df_train.to_csv("data_train.csv")
	df_test.to_csv("data_test.csv")
	return

if __name__ == "__main__":
	main()