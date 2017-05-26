from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.constraints import maxnorm
from keras.optimizers import SGD, RMSprop,Adam
from keras.utils.np_utils import to_categorical
from log import save_model, save_config, save_result,save_history
import numpy as np
import sys

# import dataset:
#Full training dataset
X_train = np.load('./data/X_train_full.npy')
y_train = np.load('./data/y_train_full.npy')
#Training set with validation
#X_train = np.load('./data/X_train.npy')
#y_train = np.load('./data/y_train.npy')
X_val = np.load('data/X_test_public.npy')
y_val = np.load('data/y_test_public.npy')
X_test = np.load('data/X_test_private.npy')
y_test = np.load('data/y_test_private.npy')
X_train = X_train.astype('float32');
X_val = X_val.astype('float32');
X_test = X_test.astype('float32');
print ('Lengths: ', len(X_train) ,len(y_train))

#y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)
# params:
batch_size = 128
nb_epoch = 300

# setup info:
print ('X_train shape: ', X_train.shape) # (n_sample, 1, 48, 48)
print ('y_train shape: ', y_train.shape) # (n_sample, n_categories)
print ('  img size: ', X_train.shape[2], X_train.shape[3])
print ('batch size: ', batch_size)
print ('  nb_epoch: ', nb_epoch)
print ('Lengths: ', len(X_train) ,len(y_train))
print ('Target: ', y_train)
print ('Target: ', y_train)
# model architecture:
model = Sequential()
# model.add(Conv2D(32,(3, 3), padding="same",activation='relu',input_shape=(1, X_train.shape[2], X_train.shape[3])))
# model.add(Conv2D(32,(3, 3), padding="same", activation='relu'))
# model.add(Conv2D(32,(3, 3), padding="same", activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

# model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
# model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
# model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
# model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

# model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
# model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
# model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
# model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

# model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
# model.add(Dense(512, activation='relu'))
# model.add(Dense(512, activation='relu'))

model.add(Conv2D(32, 3, 3, border_mode='same', activation='relu',
                        input_shape=(1, X_train.shape[2], X_train.shape[3])))
model.add(Dropout(0.2))
model.add(Conv2D(32, 3, 3, border_mode='same', activation='relu'))
model.add(Conv2D(32, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dropout(0.2))
#model.add(Dense(64, activation='relu'))
#model.add(Dense(64, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(6, activation='softmax'))

#optimizer
lrate = 0.001
decay = lrate/nb_epoch
optimizer = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
#optimizer = RMSprop(lr=lrate, rho=0.9, epsilon=1e-08, decay=decay)
#optimizer = Adam(lr=lrate,beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
print ('Training....')
#With validation
#hist = model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size
#	, validation_data=(X_val, y_val), shuffle=True, verbose=1)
#Without validation
hist = model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size
	, shuffle=True, verbose=1)
print(model.summary());
train_val_accuracy = hist.history;
# set callback: https://github.com/sallamander/headline-generation/blob/master/headline_generation/model/model.py

# model result:
loss_and_metrics = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)

print ('Loss: ', loss_and_metrics[0])
print (' Acc: ', loss_and_metrics[1])
print(hist.history.keys());
# model logging:
notes = 'medium set 100'
save_model(model, './data/results/')
save_history(train_val_accuracy, './data/results/');
#save_config(model.get_config(), './data/results/')
save_result(train_val_accuracy,loss_and_metrics, notes, './data/results/')
