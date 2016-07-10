from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers import Input, Dense, Dropout, Flatten
from keras.models import Model
from keras.preprocessing.image import  ImageDataGenerator
from keras.regularizers import l2
from keras.utils.np_utils import to_categorical
#import h5py
import numpy as np
import pandas as pd
import logging, logging.config, yaml
from dutils.laplace import *


with open ( 'logging.yaml', 'rb' ) as config:
    logging.config.dictConfig(yaml.load(config))
    logger = logging.getLogger('root')


weights_filename = 'rbm_%d_weights_top.h5'
final_filename = 'fine_rbm_weights_top.h5'

LoG = gen_lap()

weight = np.array([LoG for c in range(3)])
weights = np.array([weight for c in range(10)])





def create_rbms(input_shape=(3, 64, 64)):

    logger.debug( 'COMPILING' )

    encoded, decoded = [], []
    f0, k0, _ = input_shape

    # RBM 1
    input_img=Input(shape=input_shape)
    f1, k1, p1 = 10, 7, 2

    # encoder
    x = Convolution2D(f1, k1, k1, border_mode='same', activation='relu')(input_img)
    encoded.append(MaxPooling2D((p1,p1), border_mode='valid')(x))

    # decoder
    x = Convolution2D(f1, k1, k1, border_mode='same', activation='relu')(encoded[0])
    x = UpSampling2D((p1, p1))(x)
    decoded.append(Convolution2D(f0, 1, 1, border_mode='same')(x))

    # RBM 2
    f2, k2, p2 = 20, 5, 2

    # encoder
    x = Convolution2D(f2, k2, k2, border_mode='same', activation='relu' )(encoded[0])
    encoded.append(MaxPooling2D((p2, p2), border_mode='valid')(x))

    # decoder
    x = Convolution2D(f2, k2, k2, border_mode='same', activation='relu')(encoded[1])
    x = UpSampling2D((p2, p2))(x)
    decoded.append(Convolution2D(f1, 1, 1, border_mode='same')(x))

    # RBM 3
    f3, k3, p3 = 32, 3, 2

    # encoder
    x = Convolution2D(f3, k3, k3, border_mode='same', activation='relu')(encoded[1])
    encoded.append(MaxPooling2D((p3, p3), border_mode='valid')(x))

    # decoder
    x = Convolution2D(f3, k3, k3, border_mode='same', activation='relu')(encoded[2])
    x = UpSampling2D((p2, p2))(x)
    decoded.append(Convolution2D(f2, 1, 1, border_mode='same')(x))

    # Fully connected

    x = Flatten()(encoded[2])
    x = Dense(200, W_regularizer=l2(0.01), activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(4, activation='softmax')(x)

    rbms, hidden = [], []
    for encoder, decoder in zip(encoded, decoded):
        hid = Model(input_img, encoder)
        rbm = Model(input_img, decoder)

        rbms.append(rbm)
        hidden.append(hid)

        rbm.compile(optimizer='rmsprop', loss='mean_squared_error',
                    metrics=['accuracy'])

    fullmodel = Model(input_img, output)
    fullmodel.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                    metrics=['accuracy'])

    return rbms, hidden, fullmodel


def get_results(model):
    test = np.load("../data/pkl/test.npz" )
    # align dimensions such that channels are the
    # second axis of the tensor
    x_te = test['x'].transpose(0,3,1,2)

    logger.debug('Predicting labels...')
    results = np.argmax(model.predict(x_te),axis=-1) +1

    return results


def submit(model=None, sub=402):
    if model is None:
        model = create_model(mfile=aft_weights)


    results = get_results(model)
    logger.debug('Saving labels in file "../data/csv_lables/sub%d.csv"' % sub)

    submission_example = pd.read_csv("../data/csv_lables/sample_submission4.csv")
    submission_example["label"] = results
    submission_example.to_csv("../data/csv_lables/sub%d.csv"%sub,index=False)
    logger.debug( "Submitted at: " + ("../data/csv_lables/sub%d.csv"%sub) )


if __name__ == '__main__':
    # load dataset
    logger.debug( "loading train" )
    train = np.load("../data/pkl/trainXY.npz")
    unsupervised = np.load("../data/pkl/unsupervisedX.npz")
    x_tr, y_tr = train['x'].transpose(0,3,1,2), train['y']
    print x_tr.shape
    x_un = unsupervised['x'].transpose(0,3,1,2)
    print x_un.shape
    logger.debug( "done loading train" )
    ptrain = np.load("../data/pkl/ptrain.npz" )
    # align dimensions such that channels are the
    # second axis of the tensor
    x_pt = ptrain['x'].transpose(0,3,1,2)

    # create model
    decoders, encoders, full = create_rbms()

    # train model
    logger.debug( 'Start pretraining...')

    y = x_pt
    for encoder, decoder in zip(encoders, decoders):
        decoder.fit(x_pt, y, nb_epoch=20,
                batch_size=100,
                shuffle=True)

        logger.debug( 'Predicting next input...')
        y = encoder.predict(x_pt)

    logger.debug( 'Done preprocessing.' )

    logger.debug( 'Start training...' )

    full.fit(x_tr, to_categorical(y_tr-1,4), nb_epoch=20, batch_size=100)

    logger.debug( 'Done training.' )

    logger.debug( 'Submitting...' )
    submit(full, sub=00066)
