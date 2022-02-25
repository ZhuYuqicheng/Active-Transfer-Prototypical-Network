# %%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Conv1D
from tensorflow.python.keras.layers import MaxPooling1D
from tensorflow.python.keras.models import Model

from sklearn.preprocessing import StandardScaler

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

    def run(self):
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

class GenerateHAPTData():
    def __init__(self) -> None:
        pass

# %%
if __name__ == "__main__":
    # X, y = GenerateHAPTData().run()
    train_model(X, y)

# %%
