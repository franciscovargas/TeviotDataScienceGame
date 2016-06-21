
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
import cPickle as pkl
from keras.layers import Dense, Dropout, Activation, Flatten

from keras.utils.np_utils import to_categorical
from sklearn.cross_validation import train_test_split
from keras.regularizers import l2
import h5py


def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D(((226 -64)/2,(226 -64)/2),input_shape=(3,64,64)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()

    return model

def pop():
    model.layers.pop() # Get rid of the classification layer
    # model.layers.pop() # Get rid of the dropout layer
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []

#
# if __name__ == "__main__":

    # Test pretrained model
print "COMPILING"
model = VGG_16('vgg16_weights.h5')
[pop()  for i in range(32)]
model.add(Flatten())
# MLP
model.add(Dense(200 ))
model.add(Activation('relu'))
model.add(Dropout(0.5))


#Classification Layer
model.add(Dense(4))
model.add(Activation('softmax'))
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')
print "LAYERS: ", model.layers
print "COMPILED"
print "loading train"
trainX = pkl.load(open("../data/pkl/trainX.pkl"))
print "done loading train"

trainX=trainX.transpose(0,3,1,2)
p = 224 - 64
trainY = pkl.load(open("../data/pkl/trainY.pkl"))

# np.pad(trainX,
#        ((0,0),(0,0),(p, k_x  -1),
#        (k_y-1, k_y -1)),
#        'constant',
#        constant_values=0 )

print "percentage split start"
X_train, X_test, y_train, y_test = train_test_split(trainX, trainY-1,
                                                    test_size=0.20,
                                                    random_state=42)
print "percentage split done"

model.fit(X_train, to_categorical(y_train,4) , batch_size=100, nb_epoch=45, validation_data=(X_test,to_categorical(y_test,4)) )
print model.evaluate(X_test, to_categorical(y_test,4), batch_size=100)
