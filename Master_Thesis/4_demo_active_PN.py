# %% Import
import random
from datetime import datetime
import os
import copy
from tkinter import Y
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Conv1D
from tensorflow.python.keras.layers import MaxPooling1D
from tensorflow.python.keras.models import Model
from tensorflow import set_random_seed
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from importlib import reload
reload(keras.models)

# for reproducibility
np.random.seed(1) # numpy
random.seed(2) # Python
set_random_seed(3) # tensorflow

# %% Load HAR Data
# load the data from file
def load_file(filepath):
	"""
	output: numpy array (samples, timesteps)
	"""
	dataframe = pd.read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

# load a list of files into a 3D array
def load_group(filenames, prefix=''):
	"""
	output: (samples, timesteps, features)
	"""
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
	# stack group so that features are the 3rd dimension
	loaded = np.dstack(loaded)
	return loaded

# load a dataset group, such as train or test
def load_dataset_group(group):
	filepath = "./UCI HAR Dataset/"+ group + "/Inertial Signals/"
	# load all 9 files as a single array
	filenames = list()
	# total acceleration
	filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
	# body acceleration
	filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
	# body gyroscope
	filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
	# load input data
	X = load_group(filenames, filepath)
	# load class output
	y = load_file('./UCI HAR Dataset/' + group + '/y_'+group+'.txt')
	return X, y

def scale_data(trainX, testX):
	# remove overlap
	cut = int(trainX.shape[1] / 2)
	longX = trainX[:, -cut:, :]
	# flatten windows
	longX = longX.reshape((longX.shape[0] * longX.shape[1], longX.shape[2]))
	# flatten train and test
	flatTrainX = trainX.reshape((trainX.shape[0] * trainX.shape[1], trainX.shape[2]))
	flatTestX = testX.reshape((testX.shape[0] * testX.shape[1], testX.shape[2]))
	# standardize
	s = StandardScaler()
	# fit on training data
	s.fit(longX)
	# apply to training and test data
	longX = s.transform(longX)
	flatTrainX = s.transform(flatTrainX)
	flatTestX = s.transform(flatTestX)
	# reshape
	flatTrainX = flatTrainX.reshape((trainX.shape))
	flatTestX = flatTestX.reshape((testX.shape))
	return flatTrainX, flatTestX

# load the dataset, returns train and test X and y elements
def load_dataset():
	# load all train
	trainX, trainy = load_dataset_group('train')
	# load all test
	testX, testy = load_dataset_group('test')
	# save the original label
	ori_trainy = copy.deepcopy(trainy)
	ori_testy = copy.deepcopy(testy)
	# zero-offset class values
	trainy = trainy - 1
	testy = testy - 1
	# one hot encode y
	trainy = tf.keras.utils.to_categorical(trainy)
	testy = tf.keras.utils.to_categorical(testy)
	# standardization
	flatTrainX, flatTestX = scale_data(trainX, testX)
	result = {"trainX": flatTrainX, "trainy": trainy, \
		"testX": flatTestX, "testy": testy, \
		"ori_trainy": ori_trainy, "ori_testy": ori_testy}
	return result

def input_preprocessing():
	# load the HAR data
	HAR_data = load_dataset()
	# retrieve data
	trainX, trainy = HAR_data["trainX"], HAR_data["trainy"]
	testX, testy = HAR_data["testX"], HAR_data["testy"]
	ori_trainy, ori_testy = HAR_data["ori_trainy"], HAR_data["ori_testy"]
	# for pre-training
	# 1 WALKING 2 WALKING_UPSTAIRS 3 WALKING_DOWNSTAIRS
	train_mask = np.where(trainy==1)
	test_mask = np.where(testy==1)
	pre_train_data = {}
	pre_train_data.update([("trainX", trainX[train_mask[0]]), ("trainy", trainy[train_mask[0]])])
	pre_train_data.update([("testX", testX[test_mask[0]]), ("testy", testy[test_mask[0]])])
	# for active learning
	X = np.concatenate((trainX, testX), axis=0)
	y = np.concatenate((ori_trainy, ori_testy), axis=0)
	# 4 SITTING 5 STANDING 6 LAYING
	active_mask = np.where((y==2) | (y==3) | (y==4) | (y==5) | (y==6))
	active_data = {"X": X[active_mask[0]], "y": y[active_mask[0]]}
	return pre_train_data, active_data

# %% Pre-trained Encoder class
class Encoder:
	def __init__(self, data):
		self.trainX = data["trainX"]
		self.trainy = data["trainy"]
		self.testX = data["testX"]
		self.testy = data["testy"]

	def train_model(self, verbose=0, epochs=10, batch_size=32, \
		filters=32, kernel=7, feature_num=100, plot_acc=False):
		"""pre-training process of the PN Encoder"""
		# get dimension
		n_timesteps =  self.trainX.shape[1]
		n_features = self.trainX.shape[2]
		n_outputs = self.trainy.shape[1]
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
			model.fit(self.trainX, self.trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
		# evaluate model on test set
		_, accuracy = model.evaluate(self.testX, self.testy, batch_size=batch_size, verbose=0)
		# save model
		current_time = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
		self.model_path = os.path.join("Encoder_models", current_time)
		model.save(self.model_path)

		# plot learning curve
		if plot_acc:
			self.plot_Learning_curve(train_history)
		return accuracy

	def plot_Learning_curve(self, train_history):
		"""Plot the learning curve of pre-trained encoder"""
		fig, ax = plt.subplots(figsize=(8, 6))
		plt.plot(train_history.history["acc"])
		plt.xlabel("epochs")
		plt.ylabel("accuracy")
		plt.title("Learning Curve of Pre-trained Encoder")

# %% Prototypical Network Class
class ActiveLearning:
	def __init__(self, model_path, data):
		# load the pre-trained encoder
		base_model = keras.models.load_model(model_path)
		# feature extraction
		model = Model(inputs=base_model.input, outputs=base_model.get_layer("feature").output)
		self.features = model.predict(data["X"])
		# initialize pool
		self.X_pool = copy.deepcopy(self.features)
		self.y = data["y"]
		self.y_pool = copy.deepcopy(self.y)

	def random_selection(self, X_pool):
		n_samples = len(X_pool)
		query_idx = np.random.choice(range(n_samples), len(np.unique(self.y)))
		return query_idx
	
	def pn_prediction(self, prototyps, x):
		dist = np.sum((np.array(prototyps)-x)**2, axis=1)
		idx = np.argmin(dist)
		pred = prototyps.index[idx]
		return pred

	def run_prototypical_network(self, support_X, support_y):
		X = np.concatenate(support_X, axis=0)
		y = np.concatenate(support_y, axis=0)
		support_set = pd.DataFrame(X)
		support_set["y"] = y
		prototyps = support_set.groupby("y").mean()
		y_pred = [self.pn_prediction(prototyps, x) for x in self.features]
		return y_pred

	def run_active_learning(self, n_queries=100):
		# initialization with K-means
		support_X = []
		support_y = []
		self.scores = []
		for index in range(n_queries):
			# query process
			query_idx = self.random_selection(self.X_pool)
			support_X.append(self.X_pool[query_idx])
			# labeling
			support_y.append(self.y_pool[query_idx])
			# training the model
			y_pred = self.run_prototypical_network(support_X, support_y)
			# evaluation
			score = accuracy_score(self.y, y_pred)
			self.scores.append(score)
			# delete from the pool
			self.X_pool = np.delete(self.X_pool, query_idx, axis=0)
			self.y_pool = np.delete(self.y_pool, query_idx, axis=0)


# %% main
if __name__ == "__main__":
	pre_train_data, active_data = input_preprocessing()
	# pre-trained model
	encoder = Encoder(pre_train_data)
	accuracy = encoder.train_model(epochs=5, verbose=0, plot_acc=False)
	print(accuracy)

	# Prototypical Network
	# # input with the same pre-processing (standardization)
	# HAR_data = load_dataset()
	# val_X = np.concatenate((HAR_data["trainX"], HAR_data["testX"]), axis=0)
	# # original label
	# val_label = np.concatenate((HAR_data["ori_trainy"], HAR_data["ori_testy"]), axis=0)
	# val_data = {"X": val_X, "y":val_label}
	# model_path = "Encoder_models/16_02_2022__00_42_06"
	model = ActiveLearning(encoder.model_path, active_data)
	model.run_active_learning(n_queries=10)
	plt.plot(model.scores)


# %%
