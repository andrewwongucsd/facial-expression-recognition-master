
import numpy as np
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop
from log import load_model

X_train = np.load('data/X_train_full.npy')
y_train = np.load('data/y_train_full.npy')
X_test = np.load('data/X_test_private.npy')
y_test = np.load('data/y_test_private.npy')
nb_epoch = 200
batch_size = 128
lrate = 0.001
model = load_model('bestModel.json','bestModel.h5');
decay = lrate/nb_epoch
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
#rmsProp = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

loss_and_metrics = model.evaluate(X_train, y_train, batch_size=batch_size, verbose=1)
print("Training set")
print ('Loss: ', loss_and_metrics[0])
print (' Acc: ', loss_and_metrics[1])
loss_and_metrics = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
print("Testing set")
print ('Loss: ', loss_and_metrics[0])
print (' Acc: ', loss_and_metrics[1])
