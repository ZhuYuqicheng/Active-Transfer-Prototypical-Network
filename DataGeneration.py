# %%
# general packages
import numpy as np
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
# deep learning packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Conv1D
from tensorflow.python.keras.layers import MaxPooling1D
from tensorflow.python.keras.models import Model
# machine learning packages
from sklearn.preprocessing import StandardScaler

class GenerateHARData():
	def __init__(self) -> None:
		pass

	def get_group_data(self, group):
		# get data
		data_dir = "UCI HAR Dataset/"+group+"/Inertial Signals"
		filenames = list()
		filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
		filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
		X = []
		for filename in filenames:
			# load data
			data_path = os.path.join(data_dir, filename)
			data = pd.read_csv(data_path, header=None, delim_whitespace=True)
			X.append(data)
		X = np.stack(X, axis=2)
		# get labels
		label_path = "UCI HAR Dataset/"+group+"/y_"+group+".txt"
		label = pd.read_csv(label_path, header=None, delim_whitespace=True)
		label = label.values - 1
		y = tf.keras.utils.to_categorical(label)
		return X, y

	def scale_data(self, X):
		# remove overlap
		cut = int(X.shape[1] / 2)
		longX = X[:, -cut:, :]
		# flatten windows
		longX = longX.reshape((longX.shape[0] * longX.shape[1], longX.shape[2]))
		# flatten train and test
		flatX = X.reshape((X.shape[0] * X.shape[1], X.shape[2]))
		# flatTestX = testX.reshape((testX.shape[0] * testX.shape[1], testX.shape[2]))
		# standardize
		s = StandardScaler()
		s.fit(longX)
		flatX = s.transform(flatX)
		# reshape
		flatX = flatX.reshape((X.shape))
		return flatX

	def run(self):
		trainX, trainy = self.get_group_data("train")
		testX, testy = self.get_group_data("test")
		X = np.concatenate([trainX, testX], axis=0)
		# standardization
		X = self.scale_data(X)
		y = np.concatenate([trainy, testy], axis=0)
		return X, y
		
class GenerateHAPTData():
	def __init__(self) -> None:
		pass

	def get_exp_path(self, exp_num):
		raw_data_dir = "HAPT Dataset/RawData"
		for root, _, files in os.walk(raw_data_dir, topdown=False):
			path_list = [os.path.join(root, file) for file in files if "_exp" + exp_num + "_" in file]
		return path_list

	def get_exp_data(self, label_info, index):
		# get the right exp string
		if label_info["exp_num"][index] < 10: 
			exp_num = "0" + str(label_info["exp_num"][index])
		else:
			exp_num = str(label_info["exp_num"][index])
		# get clip indeces
		start = label_info["label_start"][index]
		end = label_info["label_end"][index]
		start_indeces = [start]
		window_len = 128
		while True:
			start += window_len
			if start+window_len > end:
				break
			start_indeces.append(start)
		# data preparation (concatenate + standardization)
		path_list = self.get_exp_path(exp_num)
		acc_data = pd.read_csv(path_list[0], header=None, delim_whitespace=True)
		gyro_data = pd.read_csv(path_list[1], header=None, delim_whitespace=True)
		concat_data = np.concatenate([acc_data, gyro_data], axis=1)
		concat_data = StandardScaler().fit_transform(concat_data)
		# stack clip data
		stack_data = [concat_data[i:i+window_len] for i in start_indeces]
		stack_data = np.stack(stack_data, axis=0)
		# create corresponding labels
		label = [label_info["act_num"][index]]*len(stack_data)
		label = np.array(label).reshape(-1,1)
		return stack_data, label

	def run(self, change=1):
		label_path = "HAPT Dataset/RawData/labels.txt"
		label_info = pd.read_csv(label_path, header=None, delim_whitespace=True)
		label_info.columns = ["exp_num", "user_num", "act_num", "label_start", "label_end"]
		data_list = []
		label_list = []
		for index in range(len(label_info)):
			data, label = self.get_exp_data(label_info, index)
			data_list.append(data)
			label_list.append(label)
		X = np.concatenate(data_list, axis=0)
		y = np.concatenate(label_list, axis=0)
		# select label
		mask = np.where(y>=change)[0]
		X = X[mask]
		y = y[mask]
		y = y - change
		# y = y - 1
		y = tf.keras.utils.to_categorical(y)
		return X, y

def plot_Learning_curve(train_history):
	"""Plot the learning curve of pre-trained encoder"""
	fig, ax = plt.subplots(figsize=(8, 6))
	plt.plot(train_history.history["acc"])
	plt.xlabel("epochs")
	plt.ylabel("accuracy")
	plt.title("Learning Curve of Pre-trained Encoder")

def train_model(X, y, verbose=1, epochs=10, batch_size=32, \
	filters=32, kernel=7, feature_num=100, plot_acc=False):
	"""pre-training process of the PN Encoder"""
	# get dimension
	n_timesteps =  X.shape[1]
	n_features = X.shape[2]
	n_outputs = y.shape[1]
	# define model structure
	model = Sequential()
	model.add(Conv1D(filters=filters, kernel_size=kernel, activation='relu', input_shape=(n_timesteps,n_features)))
	model.add(Conv1D(filters=filters, kernel_size=kernel, activation='relu'))
	model.add(Dropout(0.5))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Flatten())
	model.add(Dense(feature_num, activation='relu', name="feature"))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
	train_history = \
		model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose)
	# evaluate model on test set
	_, accuracy = model.evaluate(X, y, batch_size=batch_size, verbose=0)
	# result
	plot_Learning_curve(train_history)
	print(accuracy)

def train_encoder(X, y, verbose=1, epochs=10, batch_size=32, \
	filters=32, kernel=7, feature_num=100):
	# get dimension
	n_timesteps =  X.shape[1]
	n_features = X.shape[2]
	n_outputs = y.shape[1]
	# define model structure
	model = Sequential()
	model.add(Conv1D(filters=filters, kernel_size=kernel, activation='relu', \
		input_shape=(n_timesteps,n_features)))
	model.add(Conv1D(filters=filters, kernel_size=kernel, activation='relu'))
	model.add(Dropout(0.5))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Flatten())
	model.add(Dense(feature_num, activation='relu', name="feature"))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
	train_history = \
		model.fit(X, y, epochs=epochs, \
			batch_size=batch_size, verbose=verbose)
	# save the model
	current_time = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
	model_path = os.path.join("Encoder_models", current_time)
	model.save(model_path)

# %%
if __name__ == "__main__":
	# data generation test
	# X, y = GenerateHAPTData().run()
	# X, y = GenerateHARData().run()
	# train_model(X, y)

	# pre-trained model training
	X, y = GenerateHAPTData().run(change=7)
	train_encoder(X, y)

# %%
