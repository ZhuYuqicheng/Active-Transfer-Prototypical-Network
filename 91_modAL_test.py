#%%
import numpy as np
import pandas as pd
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from modAL.batch import uncertainty_batch_sampling

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Conv1D
from tensorflow.python.keras.layers import MaxPooling1D
from tensorflow.python.keras.models import Model

import matplotlib.pyplot as plt

from train_class import load_dataset
from DataGeneration import GenerateHAPTData, GenerateHARData

def random_sampling(classifier, X_pool):
	n_samples = len(X_pool)
	query_idx = np.random.choice(range(n_samples), size=1, replace=False)
	return query_idx, X_pool[query_idx]

def random_batch_sampling(classifier, X_pool):
	n_samples = len(X_pool)
	query_idx = np.random.choice(range(n_samples), size=6, replace=False)
	return query_idx, X_pool[query_idx]
class OneDCNN():
	def __init__(self) -> None:
		pass

	def fit(self, X, y, verbose=0, epochs=1, batch_size=32, \
		filters=32, kernel=7, feature_num=100):
		# get dimension
		n_timesteps =  X.shape[1]
		n_features = X.shape[2]
		n_outputs = y.shape[1]
		# define model structure
		self.model = Sequential()
		self.model.add(Conv1D(filters=filters, kernel_size=kernel, activation='relu', input_shape=(n_timesteps,n_features)))
		self.model.add(Conv1D(filters=filters, kernel_size=kernel, activation='relu'))
		self.model.add(Dropout(0.5))
		self.model.add(MaxPooling1D(pool_size=2))
		self.model.add(Flatten())
		self.model.add(Dense(feature_num, activation='relu', name="feature"))
		self.model.add(Dense(n_outputs, activation='softmax', name="prob"))
		self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		# fit network
		train_history = \
			self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose)

	def predict(self, X):
		return self.model.predict(X)
	
	def score(self, X, y):
		_, accuracy = self.model.evaluate(X, y, verbose=0)
		return accuracy
	
	def predict_proba(self, X):
		predictor = Model(inputs=self.model.input, outputs=self.model.get_layer("prob").output)
		return predictor.predict(X)

class Evaluator():
	def __init__(self, data_generator, estimator, query_strategy) -> None:
		self.X, self.y = data_generator.run()
		self.estimator = estimator
		self.query_strategy = query_strategy
		if (query_strategy == uncertainty_batch_sampling) or (query_strategy == random_batch_sampling):
			self.batch_mode = True
		else:
			self.batch_mode = False

	def single_evaluation(self, n_queries, index):
		np.random.seed(0)
		# initialization
		initial_idx = np.random.choice(range(len(self.X)), size=100, replace=False)
		X_initial, y_initial = self.X[initial_idx], self.y[initial_idx]
		X_pool, y_pool = np.delete(self.X, initial_idx, axis=0), np.delete(self.y, initial_idx, axis=0)
		# active learning
		learner = ActiveLearner(
			estimator=self.estimator,
			query_strategy=self.query_strategy,
			X_training=X_initial, y_training=y_initial
		)
		accuracy = [learner.score(self.X, self.y)]
		for i in range(n_queries):
			query_idx, _ = learner.query(X_pool)
			learner.teach(X_pool[query_idx], y_pool[query_idx])
			X_pool, y_pool = np.delete(X_pool, query_idx, axis=0), np.delete(y_pool, query_idx, axis=0)
			accuracy.append(learner.score(self.X, self.y))
			print(f"{index+1}. iteration: {i+1}/{n_queries} queries")
		# get x for visualization
		if self.batch_mode:
			self.plot_indeces = \
				np.linspace(1, (n_queries+1)*len(np.unique(self.y)), (n_queries+1), dtype=np.int16)
		else:
			self.plot_indeces = np.linspace(1, (n_queries+1), (n_queries+1), dtype=np.int16)
		return accuracy

	def run(self, n_queries, iteration, visual=False):
		self.accuracies = [self.single_evaluation(n_queries, index) \
			for index in range(iteration)]
		if visual:
			self.visualization()

	def bootstrap(self, values, confidence=0.95):
		return np.percentile(values,[100*(1-confidence)/2, 100*(1-(1-confidence)/2)])

	def visualization(self):
		# plot the result
		fig, ax = plt.subplots()
		y = np.apply_along_axis(np.mean, 0, np.array(self.accuracies))
		conf_int = np.apply_along_axis(self.bootstrap, 0, np.array(self.accuracies))
		ax.plot(self.plot_indeces, y)
		ax.fill_between(self.plot_indeces, conf_int[0], conf_int[1], alpha=0.1)

#%%
if __name__ == "__main__":
	evaluator = Evaluator(
		data_generator = GenerateHARData(), 
		estimator = OneDCNN(), 
		query_strategy = uncertainty_batch_sampling
	)

	evaluator.run(n_queries=10, iteration=3, visual=True)

# %%
