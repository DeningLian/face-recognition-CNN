# This module builds and trains the CNN model
import numpy as np
import os
# We use keras on top of TensorFlow
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

from skimage import io
from sklearn.cross_validation import train_test_split


# Load the path of dataset
DatasetPath = []
for i in os.listdir('./CNNdata'):
    DatasetPath.append(os.path.join('./CNNdata', i))

imageData = []
imageLabels = []

# Save the photos and labels
for i in DatasetPath:
    imgRead = io.imread(i,as_grey=True)
    imageData.append(imgRead)
    
    labelRead = int(os.path.split(i)[1].split("_")[0]) - 1
    imageLabels.append(labelRead)

# Split randomly the photos into 2 parts, 
# 90% for training, 10% for testing
X_train, X_test, y_train, y_test = train_test_split(np.array(imageData),np.array(imageLabels), train_size=0.9, random_state = 4)

X_train = np.array(X_train)
X_test = np.array(X_test)

y_train = np.array(y_train) 
y_test = np.array(y_test)

# nb_classes is the number of people(classifier)
nb_classes = 4
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# for tensorflow backend, the format is (nb_of_photo, size, size, channel)
X_train = X_train.reshape(X_train.shape[0], 46, 46, 1)
X_test = X_test.reshape(X_test.shape[0], 46, 46, 1)

# input_shape is for the first layer of model.
# 46, 46, 1 means size 46*46 pixels, 1 channel(because we read as grayscale,not RGB)
input_shape = (46, 46, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

# Start the build of model
model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# Commpile this model
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

# Training
model.fit(X_train, Y_train, batch_size=32, epochs=20,
                 verbose=1, validation_data=(X_test, Y_test))

# when the training finishes, save the trained model as nodel.json and model.h5
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# and use the 10% data as we have already splited to test the new model
scores = model.evaluate(X_test, Y_test, verbose=0)
print scores #print the accuracy
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


