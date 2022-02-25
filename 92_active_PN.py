# %%
import numpy as np
import pandas as pd
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Conv1D
from tensorflow.python.keras.layers import MaxPooling1D
from tensorflow.python.keras.models import Model

from sklearn.metrics import accuracy_score

from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from modAL.batch import uncertainty_batch_sampling

import matplotlib.pyplot as plt

from train_class import load_dataset

# %%
def random_sampling(classifier, X_pool):
	n_samples = len(X_pool)
	query_idx = np.random.choice(range(n_samples), size=1, replace=False)
	return query_idx, X_pool[query_idx]

def random_batch_sampling(classifier, X_pool):
	n_samples = len(X_pool)
	query_idx = np.random.choice(range(n_samples), size=6, replace=False)
	return query_idx, X_pool[query_idx]

class PrototypicalNetwork():
	def __init__(self) -> None:
		pass

	def fit(self, X, y, verbose=0, epochs=1, batch_size=32, \
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

		# get the feature extractor
		self.extractor = Model(inputs=model.input, outputs=model.get_layer("feature").output)
		features = self.extractor.predict(X)
		# calculate the prototyps
		support_set = pd.DataFrame(features)
		support_set["y"] = np.argmax(y, axis=1).reshape(-1,1)
		prototyps = support_set.groupby("y").mean()
		self.prototyps = prototyps

	def single_predict(self, feature):
		dist = np.sum((np.array(self.prototyps)-feature)**2, axis=1)
		idx = np.argmin(dist)
		return self.prototyps.index[idx]

	def predict(self, X):
		features = self.extractor.predict(X)
		y_pred = [self.single_predict(feature) for feature in features]
		return y_pred

	def predict_proba(self, X):
		features = self.extractor.predict(X)
		prob = []
		for feature in features:
			dist = np.sum((np.array(self.prototyps)-feature)**2, axis=1)
			prob.append(1 - np.exp(dist)/sum(np.exp(dist)))
		return np.array(prob)

	def score(self, X, y):
		y_pred = self.predict(X)
		y = np.argmax(y, axis=1).reshape(-1,1)
		return accuracy_score(y, y_pred)

# %%
if __name__ == "__main__":
	# load HAR data
	HAR_data = load_dataset()
	X_train = HAR_data["testX"]# just for test
	y_train = HAR_data["testy"]
	#X_test = HAR_data["testX"]
	#y_test = HAR_data["testy"]

	# Parameter Initialization
	n_queries = 20
	estimator = PrototypicalNetwork()
	# query_strategy = random_sampling
	# query_strategy = uncertainty_sampling
	# query_strategy = random_batch_sampling
	query_strategy = uncertainty_batch_sampling
	# initialization
	np.random.seed(0)
	initial_idx = np.random.choice(range(len(X_train)), size=1, replace=False)
	X_initial, y_initial = X_train[initial_idx], y_train[initial_idx]
	X_pool, y_pool = np.delete(X_train, initial_idx, axis=0), np.delete(y_train, initial_idx, axis=0)
	learner = ActiveLearner(
		estimator=estimator,
		query_strategy=query_strategy,
		X_training=X_initial, y_training=y_initial
	)
	# training
	train_accuracy = [learner.score(X_train, y_train)]
	#test_accuracy = [learner.score(X_test, y_test)]
	for i in range(n_queries):
		# Active Learning
		query_idx, _ = learner.query(X_pool)
		learner.teach(X_pool[query_idx], y_pool[query_idx])
		X_pool, y_pool = np.delete(X_pool, query_idx, axis=0), np.delete(y_pool, query_idx, axis=0)
		train_accuracy.append(learner.score(X_train, y_train))
		#test_accuracy.append(learner.score(X_test, y_test))
		print(f"run: {i+1}/{n_queries}")

# %%