import pandas as pd
import sys
import argparse
#Split data finished a lot of the work is in the dataloader on the train.py file

def main():
	"Read the data.csv, and separate the dataset into two, training and validation with 80/20 ratio"
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", type=str, help="Path of dataset to split", default="datasets/data.csv")
	parser.add_argument("--train_dataset", type=str, help="Path of train dataset", default="datasets/data_train.csv")
	parser.add_argument("--val_dataset", type=str, help="Path of val dataset", default="datasets/data_val.csv")
	args = parser.parse_args()

	try : 
		df = pd.read_csv(args.dataset, header=None)
	except Exception as e:
		sys.exit(f"Error reading the file {e}")

	print(df.groupby(1).size().sort_values())
	df_train = df.groupby(1).sample(frac=0.8, random_state=42)
	# df_train = df[:int(0.8 * len(df.index))]
	# df_val = df[int(0.8 * len(df.index)):]
	df_val = df.drop(df_train.index)

	#Export to csv
	df_train.to_csv(args.train_dataset, index=False, header=False)
	df_val.to_csv(args.val_dataset, index=False, header=False)
	return

if __name__ == "__main__":
	main()