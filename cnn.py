from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical
import pandas as pd
import numpy as np
from PIL import Image

# Read in data
dataset_train = pd.read_csv('fashionmnist/fashion-mnist_train.csv')
training_set = dataset_train.iloc[:].values
x_train = training_set[:,1:]
# x_train = x_train.reshape(len(x_train), 28,28)
x_train = x_train.reshape(-1, 28, 28, 1)

x_trian = np.expand_dims(x_train, axis=0)
y_train = to_categorical(training_set[:, 0])

dataset_test = pd.read_csv('fashionmnist/fashion-mnist_test.csv')
test_set = dataset_test.iloc[:].values
x_test = test_set[:,1:]
x_test = x_test.reshape(-1, 28, 28, 1)
y_test = to_categorical(test_set[:, 0])

# Build the CNN
classifier = Sequential()
classifier.add(Conv2D(32, (2, 2), input_shape=(28, 28, 1), activation='relu'))  # using 3x3 pixels as a conv window
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(32, (2, 2), input_shape=(28, 28, 1), activation='relu'))  # using 3x3 pixels as a conv window
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units=128, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=10, kernel_initializer='uniform', activation='softmax'))
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
classifier.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))