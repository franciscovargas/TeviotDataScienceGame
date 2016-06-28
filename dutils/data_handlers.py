import numpy as np

from keras import backend as K
from keras.preprocessing.image import  ImageDataGenerator



TRAIN_X_PKL = '../data/pkl/trainX.pkl'
TEST_X_PKL = '../data/pkl/testX.pkl'
TRAIN_Y_PKL = '../data/pkl/trainY.pkl'

TRAIN_NPZ = '../data/pkl/train.npz'
TEST_NPZ = '../data/pkl/train.npz'

DEFAULT_AUGMENT_NPZ = '../data/pkl/%s_augmented.npz'


def pkl2npz(dset='all'):

    import cPickle as pkl

    assert dset in ['train', 'test', 'all'], (
        "dset expected to be one of 'train', 'test' or 'all'. "
        "got %s." % dset
    )

    if dset in ['train', 'all']:
        x_tr = pkl.load(TRAIN_X_PKL)
        y_tr = pkl.load(TRAIN_Y_PKL)

        np.savez_compressed(TRAIN_NPZ, x=x_tr, y=y_tr)

    if dset in ['test', 'all']:
        x_te = pkl.load(TEST_X_PKL)

        np.savez_compressed(TEST_NPZ, x=x_te)


def augment_and_save(augmented_filename=None,
        dset='train', featurewise_center=False, samplewise_center=False,
        featurewise_std_normalization=False, samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0., width_shift_range=0., height_shift_range=0.,
        shear_range=0, zoom_range=0., channel_shift_range=0.,
        fill_mode='nearest', cval=0.,
        horizontal_flip=False, vertical_flip=False, rescale=None,
        dim_ordering=K.image_dim_orienting()):
    """
    For the documentation of these prameters see official keras documentation.
    """

    assert dset in ['train', 'test'], (
        "dset expected to be one of 'train', test. "
        "got %s." % dset
    )

    if augmented_filename is None:
        augmented_filename = DEFAULT_AUGMENT_NPZ % dset

    datagen = ImageDataGenerator(
        featurewise_center, samplewise_center,
        featurewise_std_normalization, samplewise_std_normalization,
        zca_whitening, rotation_range, width_shift_range, height_shift_range,
        shear_range, zoom_range, channel_shift_range, fill_mode, cval,
        horizontal_flip, vertical_flip, rescale, dim_ordering
    )

    if dset == 'train':
        data = np.load(TRAIN_NPZ)
        x, y = data['x'], data['y']

        datagen.fit(x)
        x_augment, y_augment = datagen.flow(x, y)
        np.savez_compressed(augmented_filename, x=x_augment, y=y_augment)
    else:
        data = np.load(TEST_NPZ)
        x = data['x']

        datagen.fit(x)
        x_augment = datagen.flow(x)
        np.savez_compressed(augmented_filename, x=x_augment)
