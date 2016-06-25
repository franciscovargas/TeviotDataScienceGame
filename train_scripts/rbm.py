from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.models import Model, Sequential
import cPickle as pkl
import h5py
import numpy as np
import logging, logging.config, yaml

with open ( 'logging.yaml', 'rb' ) as config:
    logging.config.dictConfig(yaml.load(config))
    logger = logging.getLogger('root')


weights_filename = 'autoencoder_weights.h5'

def create_deep_rbm(input_shape=(3, 64, 64), wfile=None):

    logger.debug( 'COMPILING' )

    model = Sequential()

    # Convolution 1
    model.add(Convolution2D(5, 11, 11, border_mode='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same'))

    # Convolution 2
    model.add(Convolution2D(8, 5, 5, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same'))

    # Convolution 3
    model.add(Convolution2D(16, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same'))

    # Convolution 4
    model.add(Convolution2D(16, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(UpSampling2D((2, 2)))

    # Convolution 5
    model.add(Convolution2D(8, 5, 5, border_mode='same'))
    model.add(Activation('relu'))
    model.add(UpSampling2D((2, 2)))

    # Convolution 6
    model.add(Convolution2D(5, 11, 11, border_mode='same'))
    model.add(Activation('relu'))
    model.add(UpSampling2D((2, 2)))

    # Convolution 7
    model.add(Convolution2D(3, 1, 1, border_mode='same'))

    # autoencoder = Model(input_img, decoded)
    if wfile:
        logger.debug( 'LOADING WEIGHTS from file: %s.' % wfile )
        autoencoder.load_weights(wfile)


    model.compile(
            optimizer='rmsprop',
            loss='binary_crossentropy',
            metrics=['accuracy'])

    logger.debug( 'DONE COMPILING' )

    return model



if __name__ == '__main__':
    # load dataset
    logger.debug( "loading train" )
    train = np.load("../data/pkl/train.npz")
    test = np.load("../data/pkl/test.npz")
    x_tr = train['x']
    x_te = test['x']
    logger.debug( "done loading train" )

    x_tr=x_tr.transpose(0,3,1,2)
    x_te=x_te.transpose(0,3,1,2)

    # create model
    model = create_deep_rbm()

    # train model
    logger.debug( 'FITTING TRAINING SET...')
    model.fit(x_tr, x_tr, nb_epoch=50, batch_size=128, shuffle=True,
                    validation_data=(x_tr,x_tr))
    logger.debug( 'FITTING TEST SET...')
    model.fit(x_te, x_te, nb_epoch=50, batch_size=128, shuffle=True,
                    validation_data=(x_tr,x_tr))

    logger.debug( 'SAVING WEIGHTS in file: %s' % weights_filename )
    model.save_weights(weights_filename)
