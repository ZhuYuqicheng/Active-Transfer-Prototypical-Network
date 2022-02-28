#%%
# general packages
import numpy as np
import pandas as pd
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
from LearningPipeline import Evaluator

class TransferLearning():
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
		# fix the non-trainable part
		self.fixed_model = Model(inputs=base_model.input, outputs=base_model.get_layer("flatten_12").output)
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
	The encoder can be trained on 6_train_encoder.py
	"model_path" need to be changed after retrained the encoder
	"""
	def __init__(self) -> None:
		# load the pre-trained encoder
		model_path = "./Encoder_models/27_02_2022__23_06_08"
		base_model = keras.models.load_model(model_path)
		# fix the non-trainable part
		self.fixed_model = Model(inputs=base_model.input, outputs=base_model.get_layer("flatten_12").output)
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

#%% main test
if __name__ == "__main__":
	X, y = GenerateHARData().run()
	evaluator = Evaluator(
		data_generator = GenerateHARData(), 
		estimator = TransferLearning(), 
		query_strategy = uncertainty_sampling,
		init_size=1
	)

	evaluator.run(n_queries=10, iteration=1, visual=True)

# %%
