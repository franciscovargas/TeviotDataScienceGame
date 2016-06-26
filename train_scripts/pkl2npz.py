import numpy as np
import cPickle as pkl

TRAIN_X_PKL = '../data/pkl/trainX.pkl'
TEST_X_PKL = '../data/pkl/testX.pkl'
TRAIN_Y_PKL = '../data/pkl/trainY.pkl'

TRAIN_NPZ = '../data/npz/train.npz'
TEST_NPZ = '../data/npz/train.npz'


if __name__ == '__main__':
    x_tr = pkl.load(TRAIN_X_PKL)
    y_tr = pkl.load(TRAIN_Y_PKL)
    x_te = pkl.load(TEST_X_PKL)

    np.savez_compressed(TRAIN_NPZ, x=x_tr, y=y_tr)
    np.savez_compressed(TEST_NPZ, x=x_te)
