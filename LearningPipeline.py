#%%
# general packages
import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
# active learning packages
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from modAL.batch import uncertainty_batch_sampling
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
from sklearn.metrics import accuracy_score
# own packages
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
	"""
	The class of trivial 1D CNN
	"""
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

class OnlinePrototypicalNetwork():
	"""
	The class of active prototypical network
	Online: the encoder will be updated in real-time
	"""
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
			dist = dist/max(dist) # avoid overflow of exp
			prob.append(1 - np.exp(dist)/sum(np.exp(dist)))
		return np.array(prob)

	def score(self, X, y):
		y_pred = self.predict(X)
		y = np.argmax(y, axis=1).reshape(-1,1)
		return accuracy_score(y, y_pred)

class OfflinePrototypicalNetwork():
	"""
	The class of active prototypical network
	Offline: the encoder is pre-trained on HAPT dataset
	The encoder can be trained on 6_train_encoder.py
	"model_path" need to be changed after retrained the encoder
	"""
	def __init__(self) -> None:
		# load the pre-trained encoder
		model_path = "./Encoder_models/27_02_2022__23_06_08"
		base_model = keras.models.load_model(model_path)
		# get feature extractor
		self.extractor = tf.keras.models.Model(inputs=base_model.input, outputs=base_model.get_layer("feature").output)
	
	def fit(self, X, y):
		# feature extraction
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

class TransferLearning():
	"""
	pure active transfer learning
	"""
	def __init__(self, encoder_name) -> None:
		# load the pre-trained encoder
		model_path = "./Encoder_models/" + encoder_name
		base_model = keras.models.load_model(model_path)
		# fix the non-trainable part
		self.fixed_model = tf.keras.models.Model(inputs=base_model.input, outputs=base_model.get_layer("flatten").output)
		self.fixed_model.trainable = False
		self.feature_num = base_model.get_layer("feature").output.get_shape().as_list()[1]
		
	def fit(self, X, y):
		# add the trainable part on the top
		self.extractor = Sequential()
		self.extractor.add(self.fixed_model)
		self.extractor.add(Dense(self.feature_num, activation='relu', name="feature"))
		self.extractor.add(Dense(y.shape[1], activation='softmax', name="prob"))
		self.extractor.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		# fine-tune the extractor
		self.extractor.fit(X, y, epochs=1, batch_size=32, verbose=0)

	def predict(self, X):
		return self.extractor.predict(X)
	
	def score(self, X, y):
		_, accuracy = self.extractor.evaluate(X, y, verbose=0)
		return accuracy
	
	def predict_proba(self, X):
		predictor = Model(inputs=self.extractor.input, outputs=self.extractor.get_layer("prob").output)
		return predictor.predict(X)

class TransferPrototypicalNetwork():
	"""
	The class of active prototypical network
	Offline: the encoder is pre-trained on HAPT dataset
	and the encoder is online fine-tuned
	"""
	def __init__(self, encoder_name) -> None:
		# load the pre-trained encoder
		model_path = "./Encoder_models/" + encoder_name
		base_model = keras.models.load_model(model_path)
		# fix the non-trainable part
		self.fixed_model = tf.keras.models.Model(inputs=base_model.input, outputs=base_model.get_layer("flatten_12").output)
		self.fixed_model.trainable = False
		self.feature_num = base_model.get_layer("feature").output.get_shape().as_list()[1]
		
	def fit(self, X, y):
		# add the trainable part on the top
		self.extractor = Sequential()
		self.extractor.add(self.fixed_model)
		self.extractor.add(Dense(self.feature_num, activation='relu', name="feature"))
		self.extractor.add(Dense(y.shape[1], activation='softmax'))
		self.extractor.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		# fine-tune the extractor
		self.extractor.fit(X, y, epochs=1, batch_size=32, verbose=0)
		# feature extraction
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

class Evaluator():
	def __init__(self, data_generator, estimator, query_strategy, init_size=100) -> None:
		self.X, self.y = data_generator.run()
		self.estimator = estimator
		self.query_strategy = query_strategy
		self.init_size = init_size
		if (query_strategy == uncertainty_batch_sampling) or (query_strategy == random_batch_sampling):
			self.batch_mode = True
		else:
			self.batch_mode = False

	def single_evaluation(self, n_queries, index):
		np.random.seed(0)
		# initialization
		initial_idx = np.random.choice(range(len(self.X)), size=self.init_size, replace=False)
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
			label_len = len(np.unique(np.argmax(self.y, axis=1)))
			self.plot_indeces = \
				np.linspace(self.init_size, n_queries*label_len+self.init_size, n_queries+1, dtype=np.int16)
		else:
			self.plot_indeces = np.linspace(self.init_size, n_queries+self.init_size, n_queries+1, dtype=np.int16)
		return accuracy

	def run(self, n_queries, iteration, visual=False, save=False):
		self.accuracies = [self.single_evaluation(n_queries, index) \
			for index in range(iteration)]
		if save:
			self.save(n_queries, iteration)
		if visual:
			self.visualization()

	def save(self, n_queries, iteration):
		result = dict()
		result["accuracy"] = self.accuracies
		result["plot_indeces"] = self.plot_indeces
		result["estimator"] = f"{self.estimator}".split(" ")[0].split(".")[1]
		result["query_strategy"] = f"{self.query_strategy}".split(" ")[1]
		result["init_size"] = self.init_size
		result["n_queries"] = n_queries
		result["iteration"] = iteration
		# save the dict
		save_dir = "exp_results"
		save_name = result["estimator"]+"__"+result["query_strategy"]+"__"+str(result["init_size"])+".pkl"
		with open(os.path.join(save_dir, save_name), 'wb+') as f:
			pickle.dump(result, f)
		# with open('saved_dictionary.pkl', 'rb') as f:
    	# 	loaded_dict = pickle.load(f)


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
# uncertainty_sampling, uncertainty_batch_sampling, random_sampling, random_batch_sampling
# OneDCNN, OnlinePrototypicalNetwork, OfflinePrototypicalNetwork
# TransferLearning, TransferPrototypicalNetwork
if __name__ == "__main__":
	evaluator = Evaluator(
		data_generator = GenerateHARData(), 
		estimator = TransferLearning("10_03_2022__16_02_08"), 
		query_strategy = uncertainty_sampling,
		init_size=1
	)

	evaluator.run(n_queries=10, iteration=1, visual=True, save=False)

# %%
