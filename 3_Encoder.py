# %% Import
import random
from datetime import datetime
import os
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
from tensorflow import set_random_seed
from sklearn.preprocessing import StandardScaler
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

# load the dataset, returns train and test X and y elements
def load_dataset():
	# load all train
	trainX, trainy = load_dataset_group('train')
	# load all test
	testX, testy = load_dataset_group('test')
	# zero-offset class values
	trainy = trainy - 1
	testy = testy - 1
	# one hot encode y
	trainy = tf.keras.utils.to_categorical(trainy)
	testy = tf.keras.utils.to_categorical(testy)
	result = {"trainX": trainX, "trainy": trainy, "testX": testX, "testy": testy}
	return result

# %% Pre-trained Encoder class
class Encoder:
	def __init__(self, data):
		self.trainX = data["trainX"]
		self.trainy = data["trainy"]
		self.testX = data["testX"]
		self.testy = data["testy"]

	def train_model(self, verbose=0, epochs=10, batch_size=32, \
		filters=32, kernel=7, feature_num=100, \
		standardize=True, plot_acc=False):
		"""pre-training process of the PN Encoder"""
		# get dimension
		n_timesteps =  self.trainX.shape[1]
		n_features = self.trainX.shape[2]
		n_outputs = self.trainy.shape[1]
		# standardization
		if standardize:
			self.scale_data()
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
		model.save(os.path.join("Encoder_models", current_time))

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

	def scale_data(self):
		"""standardization for HAR data only"""
		# remove overlap
		cut = int(self.trainX.shape[1] / 2)
		longX = self.trainX[:, -cut:, :]
		# flatten windows
		longX = longX.reshape((longX.shape[0] * longX.shape[1], longX.shape[2]))
		# flatten train and test
		flatTrainX = self.trainX.reshape((self.trainX.shape[0] * self.trainX.shape[1], self.trainX.shape[2]))
		flatTestX = self.testX.reshape((self.testX.shape[0] * self.testX.shape[1], self.testX.shape[2]))
		# standardize
		s = StandardScaler()
		# fit on training data
		s.fit(longX)
		# apply to training and test data
		longX = s.transform(longX)
		flatTrainX = s.transform(flatTrainX)
		flatTestX = s.transform(flatTestX)
		# reshape
		flatTrainX = flatTrainX.reshape((self.trainX.shape))
		flatTestX = flatTestX.reshape((self.testX.shape))
		# output
		self.trainX = flatTrainX
		self.testX = flatTestX

# %% Prototypical Network Class
class PrototypicalNetwork:
	def __init__(self, model_path):
		# load the pre-trained encoder
		self.model = keras.models.load_model(model_path)
	
	#def feature_extraction(self, data):


# %% main
if __name__ == "__main__":
	# pre-training process
	HAR_data = load_dataset()
	encoder = Encoder(HAR_data)
	accuracy = encoder.train_model(epochs=3, verbose=0, plot_acc=True)
	print(accuracy)

	# Prototypical Network
	# model_path = "Encoder_models/15_02_2022__23_37_24"

# %%
