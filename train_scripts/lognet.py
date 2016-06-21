from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten

from keras.optimizers import SGD, Adam

import numpy as np
import theano as th

from keras.utils.np_utils import to_categorical
from keras.regularizers import l2

import cPickle as pkl

from  dutils import subtools

from keras.preprocessing.image import  ImageDataGenerator


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
                        nb_row=5, nb_col=5))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


# model.add(Convolution2D(nb_filter=10,border_mode='valid',
#                         nb_row=5, nb_col=5,))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
# MLP
model.add(Dense(200))
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
datagen = ImageDataGenerator(
        horizontal_flip=True, rotation_range=5)
datagen.fit(trainX)
print "GENERATED"
generator = datagen.flow(trainX, to_categorical(trainY-1,4) , batch_size=32)
model.fit_generator(generator,
                    samples_per_epoch=len(trainX), nb_epoch=60)

testX = pkl.load(open("../data/pkl/testX.pkl"))
testX=testX.transpose(0,3,1,2)

results = np.argmax(model.predict(testX),axis=-1) +1

subtools.create_submision(results,sub=200)
