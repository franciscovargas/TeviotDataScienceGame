from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.models import Model, Sequential
from keras.utils.np_utils import to_categorical
from sklearn.cross_validation import train_test_split
from keras.regularizers import l2
from keras.preprocessing.image import  ImageDataGenerator

import cPickle as pkl
import h5py
import numpy as np
import logging, logging.config, yaml


with open ( 'logging.yaml', 'rb' ) as config:
    logging.config.dictConfig(yaml.load(config))
    logger = logging.getLogger('root')


rbm_weights = 'autoencoder_weights.h5'
aft_weights = 'rbm_finetune_weights.h5'

def create_model(input_shape=(3, 64, 64), wfile=None):

    logger.debug( 'COMPILING' )

    model = Sequential()
    model.add(Convolution2D(5, 11, 11, input_shape=input_shape,
                activation='relu', border_mode='same'))
    model.add(MaxPooling2D((2,2), strides=(2, 2), border_mode='same'))
    model.add(Convolution2D(8, 5, 5, activation='relu', border_mode='same'))
    model.add(MaxPooling2D((2,2), border_mode='same'))
    model.add(Convolution2D(16, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D((2,2), border_mode='same'))

    if wfile:       # load the pretrained weights
        logger.debug( 'Loading weights.' )
        f = h5py.File(weights_path)
        for k in range(f.attrs['nb_layers']):
            if k >= len(model.layers):
                # we don't look at the last (fully-connected) layers in the savefile
                break
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            model.layers[k].set_weights(weights)


    model.add(Flatten())
    # MLP
    model.add(Dense(200, W_regularizer=l2(0.01) ))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))


    #Classification Layer
    model.add(Dense(4))
    model.add(Activation('softmax'))

    # compile model
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    logger.debug( 'DONE COMPILING' )

    return model


if __name__ == '__main__':
    logger.debug( "loading train" )
    # trainX = pkl.load(open("../data/pkl/trainX.pkl"))
    # trainY = pkl.load(open("../data/pkl/trainY.pkl"))
    # np.savez_compressed('../data/pkl/train.npz', x=trainX, t=trainY)
    train = np.load('../data/pkl/train.npz')
    x_tr = train['x']
    y_tr = train['t']

    logger.debug( "done loading train" )

    x_tr=x_tr.transpose(0,3,1,2)

    logger.debug( "percentage split start" )
    x_tr, x_te, y_tr, y_te = \
            train_test_split(x_tr, y_tr-1, test_size=0.33, random_state=42)
    logger.debug( "percentage split done" )


    model = create_model(wfile=rbm_weights)

    model.fit(x_tr, y_tr, nb_epoch=50, batch_size=128, shuffle=True,
            validation_data=(x_te,y_te))

    model.save_weights(aft_weights)
