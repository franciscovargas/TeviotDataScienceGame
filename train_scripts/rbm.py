from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard
import cPickle as pkl
import h5py
import numpy as np


weights_filename = 'autoencoder_weights.h5'

def create_model(input_img=Input(shape=(3, 64, 64)), wfile=None):
    
    print 'COMPILING'
    
    x = Convolution2D(5, 11, 11, activation='relu', border_mode='same')(input_img)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(8, 5, 5, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
    encoded = MaxPooling2D((2, 2), border_mode='same')(x)
    
    # at this point the representation is (8, 12, 12) i.e. 1152
    
    x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(8, 5, 5, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(5, 11, 11, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Convolution2D(3, 3, 3, activation='sigmoid', border_mode='same')(x)
    
    autoencoder = Model(input_img, decoded)
    if wfile:
        autoencoder.load_weights(wfile)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    
    print 'DONE COMPILING'
    
    return autoencoder


    
if __name__ == '__main__':
    print "loading train"
    # np.save('train.npy', trainX)
    # trainX = np.load('train.npy')
    trainX = pkl.load(open("../data/pkl/trainX.pkl"))
    print "done loading train"

    trainX=trainX.transpose(0,3,1,2)
    

    model = create_model()

    model.fit(trainX, trainX, nb_epoch=50, batch_size=128, shuffle=True, validation_data=(trainX,trainX))
             # callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

    model.save_weights(weights_filename)