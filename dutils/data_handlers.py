import numpy as np

from keras import backend as K
from keras.preprocessing.image import  ImageDataGenerator
import cPickle as pkl


TRAIN_X_PKL = '../data/pkl/trainX.pkl'
TEST_X_PKL = '../data/pkl/testX.pkl'
TRAIN_Y_PKL = '../data/pkl/trainY.pkl'

TRAIN_NPZ = '../data/pkl/train.npz'
TEST_NPZ = '../data/pkl/test.npz'
PTRAIN_NPZ = '../data/pkl/ptrain.npz'

DEFAULT_AUGMENT_NPZ = '../data/pkl/%s_augmented.npz'


def pkl2npz(dset='all'):

    assert dset in {'train', 'test', 'all'}, (
        "dset expected to be one of 'train', 'test' or 'all'. "
        "got %s." % dset
    )

    if dset in {'train', 'all'}:
        x_tr = pkl.load(open(TRAIN_X_PKL, 'rb'))
        y_tr = pkl.load(open(TRAIN_Y_PKL, 'rb'))

        np.savez_compressed(TRAIN_NPZ, x=x_tr, y=y_tr)
        del x_tr
        del y_tr

    if dset in {'test', 'all'}:
        x_te = pkl.load(open(TEST_X_PKL, 'rb'))

        np.savez_compressed(TEST_NPZ, x=x_te)
        del x_te


def build(dset='train', save=False):

    assert dset in {'train', 'test', 'pretrain'}, (
        "dset expected to be one of 'train', 'test' or 'pretrain'. "
        "got %s." % dset
    )

    import os
    import pandas as pd
    import cv2
    from skimage import transform

    dir_conts = os.popen("ls ../data/images/roof_images").read().split("\n")[:-1]
    all_ids = map(lambda x: x.strip(".jpg"), dir_conts)

    df_all = pd.DataFrame({"k" : all_ids})

    # id_train.csv  sample_submission4.csv\n",
    df_train = pd.read_csv("../data/csv_lables/id_train.csv")
    df_test = pd.read_csv("../data/csv_lables/sample_submission4.csv")

    Index([u'k'], dtype='object')
    df_all.keys()

    df_pre = set(map(int,list(df_all["k"])))

    if dset == 'train':
        train = list()
        for i, img_id in enumerate(list(df_train["Id"])):
            img = cv2.imread("../data/images/roof_images/" + str(img_id) + ".jpg")
            resized = transform.resize(img, (64, 64) ),
            train.append(resized)

        train_matrix = np.array(train)
        trainY = df_train["label"].values

        if save:
            np.savez_compressed(TRAIN_NPZ, x=train_matrix, y=trainY)
        return train_matrix, trainY

    if dset == 'test':
        test = list()
        for i, img_id in enumerate(list(df_test["Id"])):
            img = cv2.imread("../data/images/roof_images/" + str(img_id) + ".jpg")
            resized = transform.resize(img, (64, 64) )
            test.append(resized)

        test_matrix = np.array(test)

        if save:
            np.savez_compressed(TEST_NPZ, x=test_matrix)
        return test_matrix

    if dset == 'pretrain':
        ptrain = list()
        for i, img_id in enumerate(df_pre):
            img = cv2.imread("../data/images/roof_images/" + str(img_id) + ".jpg")
            resized = transform.resize(img, (64, 64) )
            ptrain.append(resized)

        ptrain_matrix = np.asarray(ptrain)

        if save:
            np.savez_compressed(PTRAIN_NPZ, x=ptrain_matrix)
        return ptrain_matrix


def augment_and_save(augmented_filename=None,
        dset='train', featurewise_center=False, samplewise_center=False,
        featurewise_std_normalization=False, samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0., width_shift_range=0., height_shift_range=0.,
        shear_range=0, zoom_range=0., channel_shift_range=0.,
        fill_mode='nearest', cval=0.,
        horizontal_flip=False, vertical_flip=False, rescale=None,
        dim_ordering=K.image_dim_ordering()):
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

    if dset in {'train', 'all'}:
        data = np.load(TRAIN_NPZ)
        x, y = data['x'], data['y']

        datagen.fit(x)
        x_augment, y_augment = datagen.flow(x, y)
        np.savez_compressed(augmented_filename, x=x_augment, y=y_augment)

    if dset in {'test', 'all'}:
        data = np.load(TEST_NPZ)
        x = data['x']

        datagen.fit(x)
        x_augment = datagen.flow(x)
        np.savez_compressed(augmented_filename, x=x_augment)
