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
import pandas as pd
import logging, logging.config, yaml

from rbm import create_deep_rbm


with open ( 'logging.yaml', 'rb' ) as config:
    logging.config.dictConfig(yaml.load(config))
    logger = logging.getLogger('root')


rbm_weights = 'autoencoder_weights.h5'
aft_weights = 'rbm_finetune_weights.h5'

def create_model(input_shape=(3, 64, 64), wfile=None, mfile=None):

    model = create_deep_rbm(input_shape, wfile)

    logger.debug('NUMBER OF LAYERS before pop = %d.' % len(model.layers))

    for i in xrange((len(model.layers)+1)/2):
        pop_layer(model)

    logger.debug('NUMBER OF LAYERS after pop = %d.' % len(model.layers))

    model.add(Flatten())
    # MLP
    model.add(Dense( 1000, W_regularizer=l2(0.01) ))
    model.add(Activation('relu'))
    model.add(Dense( 200, W_regularizer=l2(0.01) ))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))


    #Classification Layer
    model.add(Dense(4))
    model.add(Activation('softmax'))

    if mfile:
        logger.debug( 'LOADING WEIGHTS from file: %s.' % mfile )
        model.load_weights(mfile)

    # compile model
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    logger.debug( 'DONE COMPILING' )

    return model


def pop_layer(model):
    model.layers.pop() # Get rid of the classification layer
    # model.layers.pop() # Get rid of the dropout layer
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []


def get_results(model):
    test = pkl.load(open("../data/pkl/test.npz" ))
    # align dimensions such that channels are the
    # second axis of the tensor
    x_te = test['x'].transpose(0,3,1,2)
    logger.debug('Predicting labels...')
    results = np.argmax(model.predict(x_te),axis=-1) +1
    return results


def submit(model=None, sub=401):
    if model is None:
        model = create_model(mfile=aft_weights)


    results = get_results(model)
    logger.debug('Saving labels in file "../data/csv_lables/sub%d.csv"' % sub)

    submission_example = pd.read_csv("../data/csv_lables/sample_submission4.csv")
    submission_example["label"] = results
    submission_example.to_csv("../data/csv_lables/sub%d.csv"%sub,index=False)
    logger.debug( "Submitted at: " + ("../data/csv_lables/sub%d.csv"%sub) )


if __name__ == '__main__':
    logger.debug( "loading train" )
    # trainX = pkl.load(open("../data/pkl/trainX.pkl"))
    # trainY = pkl.load(open("../data/pkl/trainY.pkl"))
    # np.savez_compressed('../data/pkl/train.npz', x=trainX, t=trainY)
    train = np.load('../data/pkl/train.npz')
    x_tr=train['x'].transpose(0,3,1,2)
    y_tr = train['t']

    logger.debug( "done loading train" )

    logger.debug( "percentage split start" )
    # x_tr, x_te, y_tr, y_te = \
    #         train_test_split(x_tr, y_tr-1, test_size=0.33, random_state=42)
    logger.debug( "percentage split done" )


    model = create_model(wfile=rbm_weights)
    # model = create_model(mfile=aft_weights)

    datagen = ImageDataGenerator( vertical_flip=True,
            horizontal_flip=True, rotation_range=5, zoom_range=0.2)
    datagen.fit(x_tr)
    logger.debug( "GENERATED" )
    generator = datagen.flow(x_tr, to_categorical(y_tr-1,4) , batch_size=32)
    # model.fit_generator(generator,
    #                     samples_per_epoch=len(x_tr), nb_epoch=50,
    #                     validation_data=(x_te, to_categorical(y_te,4)))

    model.fit_generator(generator, samples_per_epoch=len(x_tr), nb_epoch=50)

    logger.debug( 'SAVING WEIGHTS in file: %s' % aft_weights )
    model.save_weights(aft_weights, overwrite=True)

    # logger.debug( model.evaluate(x_te, to_categorical(y_te-1,4), batch_size=100) )
    submit(model,401)
