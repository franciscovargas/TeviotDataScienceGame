from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers import Input, Dense, Dropout, Flatten
from keras.models import Model
from keras.preprocessing.image import  ImageDataGenerator
from keras.regularizers import l2, activity_l1
from keras.utils.np_utils import to_categorical
import h5py
import numpy as np
import pandas as pd
import logging, logging.config, yaml
import sys

with open ( 'logging.yaml', 'rb' ) as config:
    logging.config.dictConfig(yaml.load(config))
    logger = logging.getLogger('root')


weights_filename = 'rbm_%d_weights_3.h5'
final_filename = 'fine_rbm_weights_3.h5'

def create_rbms(input_shape=(3, 64, 64), wfiles=[], ffile=None):

    logger.debug( 'COMPILING' )

    encoded, decoded = [], []
    f0, k0, _ = input_shape

    # RBM 1
    input_img=Input(shape=input_shape)
    f1, k1, p1 = 10, 11, 2

    # encoder
    x = Convolution2D(f1, k1, k1,
            border_mode='same',
            activity_regularizer=activity_l1(10e-7),
            activation='relu')(input_img)
    encoded.append(MaxPooling2D((p1,p1), border_mode='valid')(x))

    # decoder
    x = Convolution2D(f1, k1, k1, border_mode='same', activation='relu')(encoded[0])
    x = UpSampling2D((p1, p1))(x)
    decoded.append(Convolution2D(f0, 1, 1, border_mode='same')(x))

    # RBM 2
    f2, k2, p2 = 32, 7, 2

    # encoder
    x = Convolution2D(f2, k2, k2,
            border_mode='same',
            activity_regularizer=activity_l1(10e-7),
            activation='relu')(encoded[0])
    encoded.append(MaxPooling2D((p2, p2), border_mode='valid')(x))

    # decoder
    x = Convolution2D(f2, k2, k2,
            border_mode='same',
            activation='relu')(encoded[1])
    x = UpSampling2D((p2, p2))(x)
    decoded.append(Convolution2D(f1, 1, 1, border_mode='same')(x))

    # RBM 3
    f3, k3, p3 = 64, 5, 2

    # encoder
    x = Convolution2D(f3, k3, k3,
            border_mode='same',
            activity_regularizer=activity_l1(10e-7),
            activation='relu')(encoded[1])
    encoded.append(MaxPooling2D((p3, p3), border_mode='valid')(x))

    # decoder
    x = Convolution2D(f3, k3, k3, border_mode='same', activation='relu')(encoded[2])
    x = UpSampling2D((p2, p2))(x)
    decoded.append(Convolution2D(f2, 1, 1, border_mode='same')(x))

    # Fully connected

    x = Flatten()(encoded[2])
    x = Dense(500, activity_regularizer=activity_l1(10e-7), activation='relu')(x)
    x = Dense(200, W_regularizer=l2(0.01), activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(4, activation='softmax')(x)

    rbms, hidden = [], []
    for encoder, decoder in zip(encoded, decoded):
        hid = Model(input_img, encoder)
        rbm = Model(input_img, decoder)

        rbms.append(rbm)
        hidden.append(hid)

        rbm.compile(optimizer='adadelta', loss='binary_crossentropy',
                    metrics=['accuracy'])

    fullmodel = Model(input_img, output)
    fullmodel.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                    metrics=['accuracy'])

    # autoencoder = Model(input_img, decoded)
    for i, wfile in enumerate(wfiles):
        logger.debug( 'LOADING WEIGHTS from file: %s.' % wfile )
        try:
            rbms[i].load_weights(wfile)
        except:
            logger.error( 'File does not exit, or permission denied.' )

    if ffile:
        logger.debug( 'LOADING WEIGHTS from file: %s.' % ffile)
        try:
            fullmodel.load_weights(ffile)
        except:
            logger.error( 'File does not exit, or permission denied.' )

    logger.debug( 'DONE COMPILING' )

    return rbms, hidden, fullmodel


def get_results(model, whitening=True):
    test = np.load("../data/pkl/test.npz" )
    # align dimensions such that channels are the
    # second axis of the tensor
    x_te = test['x'].transpose(0,3,1,2)

    if whitening:
        logger.debug('Whitening test data...')
    datagen = ImageDataGenerator(zca_whitening=whitening)
    datagen.fit(x_te)
    generator = datagen.flow(x_te, batch_size=100)

    logger.debug('Predicting labels...')
    results = np.argmax(model.predict_generator(generator),axis=-1) +1

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

    full = None
    if len(sys.argv) <= 1 or 'train' in sys.argv:

        # create model
        decoders, encoders, full = create_rbms(
            wfiles=[weights_filename % (i + 1) for i in range(3)],
            ffile=final_filename
        )

        # load dataset
        logger.debug( "loading train" )
        train = np.load("../data/pkl/train.npz")
        test = np.load("../data/pkl/test.npz")
        x_tr, y_tr = train['x'], train['t']
        x_te = test['x']
        x_tr=x_tr.transpose(0,3,1,2)
        x_te=x_te.transpose(0,3,1,2)

        x = np.append(x_tr, x_te, axis=0)

        logger.debug( "done loading train" )


        logger.debug( "adding noise...")
        noise_factor = 0.5
        x_noisy = x + noise_factor * np.random.normal(loc=0., scale=1., size=x.shape)

        x_noisy = np.clip(x_noisy, 0., 1.)
        logger.debug( "noise added...")

        logger.debug( "generating data...")
        datagen = ImageDataGenerator(
                zca_whitening=True,
                vertical_flip=True,
                horizontal_flip=True,
                rotation_range=5,
                zoom_range=0.2)
        datagen.fit(x_noisy)
        logger.debug( "data generated.")

        # train model
        logger.debug( 'Start pretraining...')

        i = 0
        y = x
        for encoder, decoder in zip(encoders, decoders):
            generator = datagen.flow(x_noisy, y, batch_size=100)

            decoder.fit_generator(generator, samples_per_epoch=len(x), nb_epoch=10)

            filename = weights_filename % (i + 1)
            logger.debug( 'SAVING WEIGHTS in file: %s...' % filename )
            try:
                decoder.save_weights( filename, overwrite=True )
            except:
                logger.error( 'Permission denied.' )
            i += 1

            logger.debug( 'Predicting next input...')
            y = encoder.predict(x_noisy)

        logger.debug( 'Done preprocessing.' )

        logger.debug( 'Start training...' )

        datagen.fit(x_tr)
        generator = datagen.flow(x_tr, to_categorical(y_tr-1,4), batch_size=100)

        full.fit_generator(generator, samples_per_epoch=len(x_tr), nb_epoch=30)
        try:
            full.save_weights( final_filename, overwrite=True )
        except:
            logger.error( 'Permission denied.' )

        logger.debug( 'Done training.' )

    if len(sys.argv) <= 1 or 'submit' in sys.argv:
        logger.debug( 'Submitting...' )
        submit(model=full, sub=404)
