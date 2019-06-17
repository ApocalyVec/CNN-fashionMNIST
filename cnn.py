from keras.models import Sequential
from keras.layers import Conv2D, Activation
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.utils import to_categorical
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Read in data
dataset_train = pd.read_csv('fashionmnist/fashion-mnist_train.csv')
training_set = dataset_train.iloc[:].values
x_train = training_set[:,1:]
# x_train = x_train.reshape(len(x_train), 28,28)
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0

x_trian = np.expand_dims(x_train, axis=0)
y_train = to_categorical(training_set[:, 0])

dataset_test = pd.read_csv('fashionmnist/fashion-mnist_test.csv')
test_set = dataset_test.iloc[:].values
x_test = test_set[:,1:]
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
y_test = to_categorical(test_set[:, 0])

# Build the CNN
classifier = Sequential()
classifier.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'))  # using 3x3 pixels as a conv window
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))  # using 3x3 pixels as a conv window
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))  # using 3x3 pixels as a conv window
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))


classifier.add(Flatten())
classifier.add(Dense(units=300, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=10, kernel_initializer='uniform', activation='softmax'))
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
r = classifier.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))

# plot some data
print('Fit return: ', r)
print(r.history.keys())

# plot losses
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# plot accuracies
plt.plot(r.history['acc'], label='acc')
plt.plot(r.history['val_acc'], label='val_acc')
plt.legend()
plt.show()