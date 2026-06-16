import pandas as pd

#Split data finished a lot of the work is in the dataloader on the train.py file

def main():
	"Read the data.csv, and separate the dataset into two, training and validation with 80/20 ratio"
	try : 
		df = pd.read_csv("datasets/data.csv", header=None)
	except Exception as e:
		print(f"Error reading the file {e}")
	
	#Remove the first columns which is ID
	df = df.iloc[:, 1:]

	#Transform label into boolean (int)
	df = pd.get_dummies(df, columns=[1], dtype=int)

	#Can shuffle before split if wanted
	# data_shuffled = np.random.permutation(data)

	#Split the dataset into 80/20 (train/test)
	df_train = df[:int(0.8 * len(df.index))]
	df_val = df[int(0.8 * len(df.index)):]

	#Export to csv
	df_train.to_csv("datasets/data_train.csv")
	df_val.to_csv("datasets/data_val.csv")
	return

if __name__ == "__main__":
	main()