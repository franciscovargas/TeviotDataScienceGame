import numpy as np
import cPickle as pkl

TRAIN_X_PKL = '../data/pkl/trainX.pkl'
TEST_X_PKL = '../data/pkl/testX.pkl'
TRAIN_Y_PKL = '../data/pkl/trainY.pkl'

TRAIN_NPZ = '../data/pkl/train.npz'
TEST_NPZ = '../data/pkl/train.npz'


if pkl2npz(dset='all'):

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
