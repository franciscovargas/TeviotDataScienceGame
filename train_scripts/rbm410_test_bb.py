# in this script we use rotation of +/- 20 degrees, instead of +/- 5 degrees
# also change the l2 regularization of the last layer
# same as 408 without zca

from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, AveragePooling2D
from keras.layers import Input, Dense, Dropout, Flatten, MaxoutDense
from keras.models import Model
from keras.preprocessing.image import  ImageDataGenerator
from keras.regularizers import activity_l1, l2
from keras.utils.np_utils import to_categorical
from sklearn.cross_validation import train_test_split
#import h5py
import numpy as np
import numpy.matlib as ma
import pandas as pd
import logging, logging.config, yaml
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Nadam


with open ( 'logging.yaml', 'rb' ) as config:
    logging.config.dictConfig(yaml.load(config))
    logger = logging.getLogger('root')



# setup
load_weights_pt = True
load_weights_ft = False
pretrain = False
finetune = True
validate = True
submit_r = False

# parameters
nb_epochs_train = 30
nb_epochs_ptrain = 20
nb_sub = 777
zca = False

weights_filename = 'rbm_%d_weights_' + str(410) + '.h5'
final_filename = 'fine_rbm_weights_%d.h5' % nb_sub
#zca_filename = '../data/zca.npz'



def create_rbms(input_shape=(3, 64, 64), wfiles=[], ffile=None):

    logger.debug( 'COMPILING' )

    encoded, decoded = [], []
    f0, k0, _ = input_shape

    # RBM 1
    input_img=Input(shape=input_shape)
    f1, k1, p1 = 16, 7, 2

    # encoder
    x = Convolution2D(f1, k1, k1,
            border_mode='same',
            activation='relu')(input_img)
    encoded.append(AveragePooling2D((p1,p1), border_mode='valid')(x))

    # decoder
    x = Convolution2D(f1, k1, k1, border_mode='same', activation='relu')(encoded[0])
    x = UpSampling2D((p1, p1))(x)
    decoded.append(Convolution2D(f0, 1, 1, border_mode='same')(x))

    # RBM 2
    f2, k2, p2 = 64, 5, 2

    # encoder
    x = Convolution2D(f2, k2, k2,
            border_mode='same',
            activation='relu')(encoded[0])
    encoded.append(AveragePooling2D((p2, p2), border_mode='valid')(x))

    # decoder
    x = Convolution2D(f2, k2, k2,
            border_mode='same',
            activation='relu')(encoded[1])
    x = UpSampling2D((p2, p2))(x)
    decoded.append(Convolution2D(f1, 1, 1, border_mode='same')(x))

    # RBM 3
    f3, k3, p3 = 128, 5, 2

    # encoder
    x = Convolution2D(f3, k3, k3,
            border_mode='same',
            activation='relu')(encoded[1])
    encoded.append(AveragePooling2D((p3, p3), border_mode='valid')(x))

    # decoder
    x = Convolution2D(f3, k3, k3, border_mode='same', activation='relu')(encoded[2])
    x = UpSampling2D((p2, p2))(x)
    decoded.append(Convolution2D(f2, 1, 1, border_mode='same')(x))

    # Fully connected

    x = Flatten()(encoded[2])
    x = Dropout(0.5)(x)
    #x = MaxoutDense(500, W_regularizer=l2(0.05))(x)
    #x = MaxoutDense(200, W_regularizer=l2(0.05))(x)
    x = Dense(500, W_regularizer=l2(0.05), activation='relu')(x)
#    x = BatchNormalization(axis=1)(x)
    x = Dense(200, W_regularizer=l2(0.05), activation='relu')(x)
#    x = BatchNormalization(axis=1)(x)
    x = Dropout(0.5)(x)
    output = Dense(4, activation='softmax')(x)
#    output = BatchNormalization(axis=1)(x)

    rbms, hidden = [], []
    for encoder, decoder in zip(encoded, decoded):
        hid = Model(input_img, encoder)
        rbm = Model(input_img, decoder)

        rbms.append(rbm)
        hidden.append(hid)

        rbm.compile(optimizer=Nadam(), loss='mean_squared_error',
                    metrics=['accuracy'])

    fullmodel = Model(input_img, output)
    fullmodel.compile(optimizer=Nadam(), loss='categorical_crossentropy',
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




def whiten(x, w_zca):
    x_shape = x.shape
    x = x.reshape((x_shape[0], -1))
    m = ma.repmat(x.mean(axis=0), x_shape[0], 1)
    return (x - m).dot(w_zca).reshape(x_shape)



def get_results(model, whitening=zca):
    test = np.load("../data/pkl/test.npz" )
    # align dimensions such that channels are the
    # second axis of the tensor
    x_te = test['x'].transpose(0,3,1,2)
    if whitening:
        logger.debug('Whitening test data...')
        w_zca = np.load(zca_filename)['w']
        x_te = whiten(x_te,  w_zca)

    logger.debug('Predicting labels...')
    results = np.argmax(model.predict(x_te),axis=-1) +1

    return results



def submit(model=None, sub=None):
    if model is None:
        model = create_model(ffile=final_filename)


    results = get_results(model)
    logger.debug('Saving labels in file "../data/csv_lables/sub%d.csv"' % sub)

    submission_example = pd.read_csv("../data/csv_lables/sample_submission4.csv")
    submission_example["label"] = results
    submission_example.to_csv("../data/csv_lables/sub%d.csv"%sub,index=False)
    logger.debug( "Submitted at: " + ("../data/csv_lables/sub%d.csv"%sub) )






submit_r = submit_r and not validate

if __name__ == '__main__':

    # create model
    decoders, encoders, full = create_rbms(
        wfiles=[weights_filename % (i + 1) for i in range(3)] if load_weights_pt else [],
        ffile=final_filename if load_weights_ft else None
    )


    #w_zca = np.load(zca_filename)['w']
    datagen = ImageDataGenerator()

    if pretrain:
        # load dataset
        logger.debug( "loading pre-train" )

        ptrain = np.load('../data/pkl/ptrain.npz')
        x = ptrain['x'].transpose(0,3,1,2)
        if zca:
            x = whiten(x, w_zca)
        logger.debug( "done loading pre-train" )

        logger.debug( "adding noise...")
        noise_factor = 0.5
        x_noisy = x + noise_factor * np.random.normal(loc=0., scale=1., size=x.shape)
        # x_noisy = np.clip(x_noisy, x.min(), x.max())
        logger.debug( "noise added...")

        datagen.fit(x_noisy)

        # train model
        logger.debug( 'Start pretraining...')

        i = 0
        y = x
        for encoder, decoder in zip(encoders, decoders):
            generator = datagen.flow(x_noisy, y, batch_size=100)
            decoder.fit_generator(generator, samples_per_epoch=len(x), nb_epoch=nb_epochs_ptrain)

            filename = weights_filename % (i + 1)
            logger.debug( 'SAVING WEIGHTS in file: %s...' % filename )

            decoder.save_weights( filename, overwrite=True )
            i += 1

            logger.debug( 'Predicting next input...')
            y = encoder.predict(x_noisy)

        logger.debug( 'Done preprocessing.' )


    if finetune:
        # load dataset
        logger.debug( "loading train" )

        if validate:
            valid = np.load('../data/pkl/valid.npz')
            x_tr, y_tr = valid['x_tr'].transpose(0,3,1,2), valid['y_tr']
            x_te, y_te = valid['x_te'].transpose(0,3,1,2), valid['y_te']
        else:
            train = np.load('../data/pkl/train.npz')
            x_tr, y_tr = train['x'].transpose(0,3,1,2), train['y']

        if zca:
            x_tr = whiten(x_tr, w_zca)
            if validate:
                x_te = whiten(x_te, w_zca)

        logger.debug( "done loading train" )
       

        logger.debug( 'Start training...' )

        datagen.fit(x_tr)
        generator = datagen.flow(x_tr, to_categorical(y_tr-1,4), batch_size=70)

        full.fit_generator(generator, samples_per_epoch=len(x_tr), nb_epoch=40,
                           validation_data=(x_te, to_categorical(y_te-1,4)) if validate else None)
        full.save_weights( final_filename, overwrite=True )

        logger.debug( 'Done training.' )

    if submit_r:
        logger.debug( 'Submitting...' )
        submit(full, sub=nb_sub)

