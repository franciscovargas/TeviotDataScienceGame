# same as 410 with different augmentation

from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers import Input, Dense, Dropout, Flatten, BatchNormalization
from keras.models import Model
from keras.preprocessing.image import  ImageDataGenerator
from keras.regularizers import activity_l1, l2
from keras.utils.np_utils import to_categorical
from sklearn.cross_validation import train_test_split
import h5py
import numpy as np
import numpy.matlib as ma
import pandas as pd
import logging, logging.config, yaml


with open ( '../logging.yaml', 'rb' ) as config:
    logging.config.dictConfig(yaml.load(config))
    logger = logging.getLogger('root')



# setup
load_weights_pt = True
load_weights_ft = False
pretrain = True
finetune = True
validate = True
submit_r = False

# parameters
skip_pretrain = 0
nb_encoders = 7
nb_epochs_train = 100
nb_epochs_ptrain = 5
batch_size=100
nb_sub = 501
zca = False

weights_filename = 'rbm_%d_weights_' + str(nb_sub) + '.h5'
final_filename = 'fine_rbm_weights_%d.h5' % nb_sub
zca_filename = '../../data/zca.npz'
data_filepattern = '../../data/pkl/%s.npz'




def create_rbms(input_shape=(3, 64, 64), wfiles=[], ffile=None):

    logger.debug( 'COMPILING' )

    encoded, decoded = [], []
    f0, k0, _ = input_shape

    # VGG 1
    input_img=Input(shape=input_shape)
    f1, k1, p1 = 32, 3, 2
    z1 = int(k1 / 2) * 2

    # encoder 1.1
    x = Convolution2D( f1, k1, k1, border_mode='valid', activation='relu', name='encoder11' )( input_img )
    encoded.append( BatchNormalization( axis=1 )( x ) )
    
    # decoder 1.1
    x = ZeroPadding2D( ( z1, z1 ) )( encoded[-1] )
    x = Convolution2D( f1, k1, k1, border_mode='valid', activation='relu', name='decoder11a' )( x )
    x = BatchNormalization( axis=1 )( x )
    decoded.append( Convolution2D( f0, 1, 1, border_mode='same', activation='sigmoid', name='decoder11b' )( x ) )

    # encoder 1.2
    x = Convolution2D( f1, k1, k1, border_mode='valid', activation='relu', name='encoder12' )( encoded[-1] )
    e = BatchNormalization( axis=1 )( x )
    
    # decoder 1.2
    x = ZeroPadding2D( ( z1, z1 ) )( e )
    x = Convolution2D( f1, k1, k1, border_mode='valid', activation='relu', name='decoder12a' )( x )
    x = BatchNormalization( axis=1 )( x )
    decoded.append( Convolution2D( f1, 1, 1, border_mode='same', activation='sigmoid', name='decoder12b' )( x ) )
    
    x = MaxPooling2D( (p1, p1), border_mode='valid' )( e )
    encoded.append( Dropout( 0.25 )( x ) )

    

    # VGG 2
    f2, k2, p2 = 64, 3, 2
    z2 = int(k2 / 2) * 2

    # encoder 2.1
    x = Convolution2D( f2, k2, k2, border_mode='valid', activation='relu', name='encoder21' )( encoded[-1] )
    encoded.append( BatchNormalization( axis=1 )( x ) )

    # decoder 2.1
    x = ZeroPadding2D( ( z2, z2 ) )( encoded[-1] )
    x = Convolution2D( f2, k2, k2, border_mode='valid', activation='relu', name='decoder21a' )( x )
    x = BatchNormalization( axis=1 )( x )
    decoded.append( Convolution2D( f1, 1, 1, border_mode='same', activation='sigmoid', name='decoder21b' )( x ) )

    # encoder 2.2
    x = Convolution2D( f2, k2, k2, border_mode='valid', activation='relu' )( encoded[-1] )
    e = BatchNormalization( axis=1 )( x )

    # decoder 2.2
    x = ZeroPadding2D( ( z2, z2 ) )( e )
    x = Convolution2D( f2, k2, k2, border_mode='valid', activation='relu' )( x )
    x = BatchNormalization( axis=1 )( x )
    decoded.append( Convolution2D( f2, 1, 1, border_mode='same', activation='sigmoid' )( x ) )
    
    x = MaxPooling2D( (p2, p2), border_mode='valid' )( e )
    x = Dropout( 0.25 )( x )



    # VGG 3
    f3, k3, p3 = 128, 3, 2
    z3 = int(k3 / 2) * 2

    # encoder 3.1
    x = Convolution2D( f3, k3, k3, border_mode='valid', activation='relu', name='encoder31' )( encoded[-1] )
    encoded.append( BatchNormalization( axis=1 )( x ) )

    # decoder 3.1
    x = ZeroPadding2D( ( z3, z3 ) )( encoded[-1] )
    x = Convolution2D( f3, k3, k3, border_mode='valid', activation='relu', name='decoder31a' )( x )
    x = BatchNormalization( axis=1 )( x )
    decoded.append( Convolution2D( f2, 1, 1, border_mode='same', activation='sigmoid', name='decoder31b' )( x ) )

    # encoder 3.2
    x = Convolution2D( f3, k3, k3, border_mode='valid', activation='relu' )( encoded[-1] )
    e = BatchNormalization( axis=1 )( x )

    # decoder 3.2
    x = ZeroPadding2D( ( z3, z3 ) )( e )
    x = Convolution2D( f3, k3, k3, border_mode='valid', activation='relu' )( x )
    x = BatchNormalization( axis=1 )( x )
    decoded.append( Convolution2D( f3, 1, 1, border_mode='same', activation='sigmoid' )( x ) )
    
    x = MaxPooling2D( (p3, p3), border_mode='valid' )( e )
    x = Dropout( 0.25 )( x )



    # Fully connected
    inp_size = (k3 - z3) * (k3 - z3) * f3
    nb_units1 = 512

    encoded.append( Flatten()( x ) )

    # encoder 4.1
    x = Dense( nb_units1, W_regularizer=l2(0.05), activation='relu' )( encoded[-1] )
    e = BatchNormalization( axis=1 )( x )

    # decoder 4.1
    decoded.append( Dense( inp_size, activation='sigmoid' )( e ) )
    encoded.append( Dropout( 0.5 )( e ) )

    # encoder 4.2
    nb_units2 = 256

    x = Dense( nb_units2, W_regularizer=l2(0.05), activation='relu' )( encoded[-1] )
    e = BatchNormalization( axis=1 )( x )

    # decoder 4.2
    decoded.append( Dense( nb_units1, activation='sigmoid' )( e ) )
    encoded.append( Dropout( 0.5 )( e ) )


    # output
    output = Dense( 4, activation='softmax' )( encoded[-1] )

    rbms, hidden = [], []
    for encoder, decoder in zip(encoded, decoded):
        hid = Model(input_img, encoder)
        rbm = Model(input_img, decoder)

        rbms.append(rbm)
        hidden.append(hid)

        rbm.compile(optimizer='nadam', loss='mean_squared_error',
                    metrics=['accuracy'])


    fullmodel = Model(input_img, output)
    fullmodel.compile(optimizer='nadam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    
    for i, wfile in enumerate(wfiles):
        logger.debug( 'LOADING WEIGHTS from file: %s.' % wfile )
        try:
            rbms[i].load_weights(wfile)
        except:
            logger.error( 'File does not exit.' )

    if ffile:
        logger.debug( 'LOADING WEIGHTS from file: %s.' % ffile)
        try:
            fullmodel.load_weights(ffile)
        except:
            logger.error( 'File does not exit.' )

    logger.debug( 'DONE COMPILING' )

    return rbms, hidden, fullmodel




def whiten(x, w_zca):
    x_shape = x.shape
    x = x.reshape((x_shape[0], -1))
    m = ma.repmat(x.mean(axis=0), x_shape[0], 1)
    return (x - m).dot(w_zca).reshape(x_shape)



def get_results(model, whitening=zca):
    test = np.load( data_filepattern % 'test' )
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
    logger.debug('Saving labels in file "../../data/csv_lables/sub%d.csv"' % sub)

    submission_example = pd.read_csv("../../data/csv_lables/sample_submission4.csv")
    submission_example["label"] = results
    submission_example.to_csv("../../data/csv_lables/sub%d.csv"%sub,index=False)
    logger.debug( "Submitted at: " + ("../../data/csv_lables/sub%d.csv"%sub) )






submit_r = submit_r and not validate

if __name__ == '__main__':

    # create model
    decoders, encoders, full = create_rbms(
        wfiles=[weights_filename % (i + 1) for i in range(nb_encoders)] if load_weights_pt else [],
        ffile=final_filename if load_weights_ft else None
    )

    if zca:
        w_zca = np.load(zca_filename)['w']

    if pretrain:
        # load dataset
        logger.debug( "loading pre-train" )

        ptrain = np.load(data_filepattern % 'ptrain')
        x = ptrain['x'].transpose(0,3,1,2)
        if zca:
            x = whiten(x, w_zca)
        logger.debug( "done loading pre-train" )

        logger.debug( "adding noise...")
        noise_factor = 0.5
        x_noisy = x + noise_factor * np.random.normal(loc=0., scale=1., size=x.shape)
        # x_noisy = np.clip(x_noisy, x.min(), x.max())
        logger.debug( "noise added...")

        # train model
        logger.debug( 'Start pretraining...')

        i = 0
        for encoder, decoder in zip(encoders, decoders):
            i += 1
            logger.debug( 'Autoencoder %d:' % i )
            if i >= skip_pretrain: 
                logger.debug( '- training...' )
                decoder.fit(x_noisy, x, batch_size=batch_size, nb_epoch=nb_epochs_ptrain)

                filename = weights_filename % i
                logger.debug( '- SAVING WEIGHTS in file: %s...' % filename )

                decoder.save_weights( filename, overwrite=True )

            if i < nb_encoders:
                logger.debug( '- predicting next input...')
                x = encoder.predict(x_noisy)

        logger.debug( 'Done pretraining.' )


    if finetune:
        # load dataset

        if validate:
            logger.debug( "loading valid" )
            valid = np.load(data_filepattern % 'valid')
            x_tr, y_tr = valid['x_tr'].transpose(0,3,1,2), valid['y_tr']
            x_te, y_te = valid['x_te'].transpose(0,3,1,2), valid['y_te']
        else:
            logger.debug( "loading train" )
            train = np.load(data_filepattern % 'train')
            x_tr, y_tr = train['x'].transpose(0,3,1,2), train['y']

        if zca:
            x_tr = whiten(x_tr, w_zca)
            if validate:
                x_te = whiten(x_te, w_zca)

        logger.debug( "done loading" )
       

        logger.debug( 'Start training...' )

        full.fit(x_tr, to_categorical(y_tr-1,4), batch_size=batch_size, nb_epoch=nb_epochs_train,
                           validation_data=(x_te, to_categorical(y_te-1,4)) if validate else None)
        full.save_weights( final_filename, overwrite=True )

        logger.debug( 'Done training.' )

    if submit_r:
        logger.debug( 'Submitting...' )
        submit(full, sub=nb_sub)

