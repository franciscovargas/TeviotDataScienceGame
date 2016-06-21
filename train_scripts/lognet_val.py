from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten

from keras.optimizers import SGD, Adam

import numpy as np
import theano as th

from keras.utils.np_utils import to_categorical
from sklearn.cross_validation import train_test_split
from keras.regularizers import l2
from keras.preprocessing.image import  ImageDataGenerator


import cPickle as pkl

LoG = np.array([[0, 1,0],
                [1,-4,1],
                [0, 1,0]])


Lox = np.array([[-1, 0,1],
                [-2, 0,2],
                [-1, 0,1]])

Loy = np.array([[1, 2,1],
                [0, 0,0],
                [-1, -2,-1]])



sha = np.array([[0, -1,0],
                [-1,5,-1],
                [0, -1,0]])


avg = (1/25.0)*np.ones((5,5))

weight = np.array([LoG for c in range(3)])
# weight = np.array([LoG, Lox, Loy])

# weights.swapaxes(0,-1)
weights = np.array([weight for c in range(10)])


weight1 = np.array([avg for c in range(10)])
# weights.swapaxes(0,-1)
weights1 = np.array([weight1 for c in range(10)])

"""
CNN Layer signature
Convolution1D(nb_filter,
              filter_length,
              init='uniform',
              activation='linear',
              weights=None,
              border_mode='valid',
              subsample_length=1,
              W_regularizer=None,
              b_regularizer=None,
              activity_regularizer=None,
              W_constraint=None, b_constraint=None, bias=True, input_dim=None, input_length=None)
"""
model = Sequential()
model.add(Convolution2D(nb_filter=10,border_mode='valid',nb_row=3,nb_col=3 ,
                        input_shape=(3,64,64),
                        weights=[weights, np.array([0]*10) ]) )
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(nb_filter=32,border_mode='valid',
                        nb_row=5, nb_col=5,))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


# model.add(Convolution2D(nb_filter=10,border_mode='valid',
#                         nb_row=5, nb_col=5,W_regularizer=l2(0.01)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
# MLP
model.add(Dense(200, W_regularizer=l2(0.01) ))
model.add(Activation('relu'))
model.add(Dropout(0.5))


#Classification Layer
model.add(Dense(4))
model.add(Activation('softmax'))


# compile model
sgd = SGD(lr=0.25)
adam = Adam(lr=0.1)


model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])






print "loading train"
trainX = pkl.load(open("../data/pkl/trainX.pkl"))
print "done loading train"


trainX=trainX.transpose(0,3,1,2)

trainY = pkl.load(open("../data/pkl/trainY.pkl"))

print "percentage split start"
X_train, X_test, y_train, y_test = train_test_split(trainX, trainY-1,
                                                    test_size=0.33,
                                                    random_state=42)
print "percentage split done"

datagen = ImageDataGenerator(
        horizontal_flip=True, rotation_range=5)
datagen.fit(X_train)
print "GENERATED"
generator = datagen.flow(X_train, to_categorical(y_train,4) , batch_size=32)
model.fit_generator(generator,
                    samples_per_epoch=len(X_train), nb_epoch=45,
                    validation_data=(X_test, to_categorical(y_test,4)))




# model.fit(X_train, to_categorical(y_train,4) , batch_size=100, nb_epoch=45)
print model.evaluate(X_test, to_categorical(y_test,4), batch_size=100)
