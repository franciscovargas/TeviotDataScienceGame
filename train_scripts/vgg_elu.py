import numpy as np
np.random.seed(23455)
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D

from sklearn.cross_validation import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten

from keras.optimizers import SGD, Adam
import pandas as pd
import theano as th
import h5py
from keras.utils.np_utils import to_categorical
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Nadam
from keras.regularizers import activity_l1, l2

from keras.layers.advanced_activations import ELU

import logging, logging.config, yaml

with open ( 'logging.yaml', 'rb' ) as config:
    logging.config.dictConfig(yaml.load(config))
    logger = logging.getLogger('root')


final_filename = 'fine_rbm_weights_vgg.h5'

model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(3, 64, 64)))
model.add(ELU())
model.add(BatchNormalization(axis=1))
model.add(Convolution2D(32, 3, 3))
model.add(ELU())
model.add(BatchNormalization(axis=1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='valid'))
model.add(ELU())
model.add(BatchNormalization(axis=1))
model.add(Convolution2D(64, 3, 3))
model.add(ELU())
model.add(BatchNormalization(axis=1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(128, 3, 3, border_mode='valid'))
model.add(ELU())
model.add(BatchNormalization(axis=1))
model.add(Convolution2D(128, 3, 3))
model.add(ELU())
model.add(BatchNormalization(axis=1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#model.add(Flatten())
#model.add(Dense(512))
#model.add(Activation('relu'))
#model.add(BatchNormalization(axis=1))
#model.add(Dropout(0.5))

model.add(Flatten())
# Note: Keras does automatic shape inference.
model.add(Dense(256))
model.add(ELU())
model.add(BatchNormalization(axis=1))
model.add(Dropout(0.5))

model.add(Dense(4))
model.add(BatchNormalization(axis=1))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=Nadam(),
              metrics=['accuracy'])

print "loading train"
train = np.load("../data/pkl/train.npz")
print "done loading train"

x_tr, y_tr = train['x'], train['y']

x_tr=x_tr.transpose(0,3,1,2)

print "percentage split start"
#X_train, X_test, y_train, y_test = train_test_split(x_tr, y_tr-1,
                                                    #test_size=0.33,
                                                    #random_state=42)
print "percentage split done"

# valid = np.load('../data/pkl/valid.npz')
# x_tr, y_tr = valid['x_tr'].transpose(0,3,1,2), valid['y_tr']
# x_te, y_te = valid['x_te'].transpose(0,3,1,2), valid['y_te']


# x_tr = (x_tr - x_tr.mean((0), keepdims=True)) / (x_tr.std((0), keepdims=True) + 1e-10)
# x_te = (x_te - x_te.mean((0), keepdims=True)) / (x_te.std((0), keepdims=True) + 1e-10)
#trainX = pkl.load(open("../data/pkl/trainX.pkl"))
#trainX=trainX.transpose(0,3,1,2)
# train = np.load('../data/pkl/train.npz')
# x_tr, y_tr = train['x'].transpose(0,3,1,2), train['y']
# print x_tr.shape
#trainY = pkl.load(open("../data/pkl/trainY.pkl"))


print "GENERATED"
# model.fit(x_tr, to_categorical(y_tr-1,4), nb_epoch=100,
#                     validation_data=(x_te, to_categorical(y_te-1,4)))

model.fit(x_tr,
          to_categorical(y_tr-1,4), nb_epoch=12)

model.save_weights( final_filename, overwrite=True )

test = np.load("../data/pkl/test.npz" )
    # align dimensions such that channels are the
    # second axis of the tensor
x_te = test['x'].transpose(0,3,1,2)

results = np.argmax(model.predict(x_te),axis=-1) +1

logger.debug('Saving labels in file "../data/csv_lables/subfirst.csv"')

submission_example = pd.read_csv("../data/csv_lables/sample_submission4.csv")
submission_example["label"] = results
submission_example.to_csv("../data/csv_lables/subfirst.csv",index=False)
logger.debug( "Submitted at: " + ("../data/csv_lables/subfirst.csv") )

#print model.evaluate(x_te, to_categorical(y_te-1,4), batch_size=100)


#print model.evaluate(X_test, to_categorical(y_test,4), batch_size=100)
