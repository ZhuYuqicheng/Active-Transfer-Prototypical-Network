# %%
import numpy as np
import pandas as pd
from datetime import datetime
import os
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

from LearningPipeline import Evaluator
from DataGeneration import GenerateHAPTData, GenerateHARData

# %%
def random_sampling(classifier, X_pool):
	n_samples = len(X_pool)
	query_idx = np.random.choice(range(n_samples), size=1, replace=False)
	return query_idx, X_pool[query_idx]

def random_batch_sampling(classifier, X_pool):
	n_samples = len(X_pool)
	query_idx = np.random.choice(range(n_samples), size=6, replace=False)
	return query_idx, X_pool[query_idx]

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
	
class OfflinePrototypicalNetwork():
	def __init__(self) -> None:
		pass
	
	def fit(self, X, y):
		# load the pre-trained encoder
		model_path = "./Encoder_models/27_02_2022__23_06_08"
		base_model = keras.models.load_model(model_path)
		# feature extraction
		self.extractor = Model(inputs=base_model.input, outputs=base_model.get_layer("feature").output)
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
			dist = dist/max(dist) # avoid overflow of exp
			prob.append(1 - np.exp(dist)/sum(np.exp(dist)))
		return np.array(prob)

	def score(self, X, y):
		y_pred = self.predict(X)
		y = np.argmax(y, axis=1).reshape(-1,1)
		return accuracy_score(y, y_pred)

# %%
if __name__ == "__main__":
	evaluator = Evaluator(
		data_generator = GenerateHARData(), 
		estimator = OfflinePrototypicalNetwork(), 
		query_strategy = random_batch_sampling
	)

	evaluator.run(n_queries=10, iteration=1, visual=True)

# %%