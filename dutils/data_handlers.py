import numpy as np
import numpy.matlib as ma

from keras import backend as K
from keras.preprocessing.image import  ImageDataGenerator
import cPickle as pkl
import cv2
from skimage import transform


TRAIN_X_PKL = '../data/pkl/trainX.pkl'
TEST_X_PKL = '../data/pkl/testX.pkl'
TRAIN_Y_PKL = '../data/pkl/trainY.pkl'

TRAIN_NPZ = '../data/pkl/train.npz'
VALID_NPZ = '../data/pkl/valid.npz'
TEST_NPZ = '../data/pkl/test.npz'
PTRAIN_NPZ = '../data/pkl/ptrain.npz'


DEFAULT_ZCA_NPZ = '../data/pkl/%s_zca.npz'


def pkl2npz(dset='all'):

    assert dset in ['train', 'test', 'all'], (
        "dset expected to be one of 'train', 'test' or 'all'. "
        "got %s." % dset
    )

    if dset in ['train', 'all']:
        x_tr = pkl.load(open(TRAIN_X_PKL, 'rb'))
        y_tr = pkl.load(open(TRAIN_Y_PKL, 'rb'))

        np.savez_compressed(TRAIN_NPZ, x=x_tr, y=y_tr)
        del x_tr
        del y_tr

    if dset in ['test', 'all']:
        x_te = pkl.load(open(TEST_X_PKL, 'rb'))

        np.savez_compressed(TEST_NPZ, x=x_te)
        del x_te


def build(dset='train', save=True, augment=True, zca_whitening=True):

    import os
    import pandas as pd

    assert dset in ['train', 'valid', 'test', 'pretrain'], (
        "dset expected to be one of 'train', 'valid', 'test' or 'pretrain'. "
        "got %s." % dset
    )

    dir_conts = os.popen("ls ../data/images/roof_images").read().split("\n")[:-1]
    all_ids = map(lambda x: x.strip(".jpg"), dir_conts)

    df_all = pd.DataFrame({"k" : all_ids})

    # id_train.csv  sample_submission4.csv\n",
    df_train = pd.read_csv("../data/csv_lables/id_train.csv")
    df_test = pd.read_csv("../data/csv_lables/sample_submission4.csv")

    df_pre = set(map(int,list(df_all["k"])))

    if dset == 'train':
        trainX, trainY = list(), list()
        for img_id, img_label in zip(list(df_train["Id"]), list(df_train["label"])):
            img = cv2.imread("../data/images/roof_images/" + str(img_id) + ".jpg")
            resized = transform.resize(img, (64, 64) ),

            trainX.append(resized[0])
            trainY.append(img_label)

        trainX = np.asarray(trainX)
        trainY = np.asarray(trainY)

        print 'Training set:'
        print 'x.shape = ', trainX.shape, ', y.shape = ', trainY.shape

        if augment:
            trainX, trainY = augment_data(trainX, trainY)
            print 'Augmnented: x.shape = ', trainX.shape, ', y.shape = ', trainY.shape

        if zca_whitening:
            trainX, _ = zca_whitening(trainX, build=False, save=False)
            if save:
                np.savez_compressed(DEFAULT_ZCA_NPZ % dset, x=trainX, y=trainY)

        elif save:
            np.savez_compressed(TRAIN_NPZ, x=trainX, y=trainY)
        return trainX, trainY
    
    if dset == 'valid':
        trainX, trainY = list(), list()
        for img_id, img_label in zip(list(df_train["Id"]), list(df_train["label"])):
            img = cv2.imread("../data/images/roof_images/" + str(img_id) + ".jpg")
            resized = transform.resize(img, (64, 64) ),

            trainX.append(resized[0])
            trainY.append(img_label)

        trainX = np.asarray(trainX)
        trainY = np.asarray(trainY)
       

        nb_val = .4
        
        x_tr, x_te, y_tr, y_te = [], [], [], []
        for i in xrange(4):
            xi, yi = trainX[trainY==(i+1)], trainY[trainY==(i+1)]
            n = xi.shape[0]
            xi_ind = np.arange(n)
            np.random.shuffle(xi_ind)
            
            n_tr = int((1-nb_val) * n)
            x_tr.append(xi[xi_ind[:n_tr]])
            x_te.append(xi[xi_ind[n_tr:]])
            y_tr.append(yi[xi_ind[:n_rr]])
            y_te.append(yi[xi_ind[n_tr:]])
        
        trainX = np.vstack(x_tr)
        trainY = np.vstack(y_tr)
        testX = np.vstack(x_te)
        testY = np.vstack(y_te)

        print 'Training set:'
        print 'x.shape = ', trainX.shape, ', y.shape = ', trainY.shape
        print 'Validation set:'
        print 'x.shape = ', testX.shape, ', y.shape = ', testY.shape

        if augment:
            trainX, trainY = augment_data(trainX, trainY)
            print 'Augmnented train: x.shape = ', trainX.shape, ', y.shape = ', trainY.shape

        if zca_whitening:
            trainX, _ = zca_whitening(trainX, build=False, save=False)
            testX, _ = zca_whitening(testX, build=False, save=False)
            if save:
                np.savez_compressed(DEFAULT_ZCA_NPZ % dset, x_tr=trainX, y_tr=trainY, x_te=testX, y_te=testX)

        elif save:
            np.savez_compressed(VALID_NPZ, x_tr=trainX, y_tr=trainY, x_te=testX, y_te=testY)
        return trainX, trainY, testX, testY

    
    if dset == 'test':
        test = list()
        for i, img_id in enumerate(list(df_test["Id"])):
            img = cv2.imread("../data/images/roof_images/" + str(img_id) + ".jpg")
            resized = transform.resize(img, (64, 64) )
            test.append(resized)

        testX = np.array(test)

        print 'Test set:'
        print 'x.shape = ', testX.shape

        if zca_whitening:
            testX, _ = zca_whitening(testX, build=False, save=False)
            if save:
                np.savez_compressed(DEFAULT_ZCA_NPZ % dset, x=testX)

        elif save:
            np.savez_compressed(TEST_NPZ, x=testX)
        return testX

    if dset == 'pretrain':
        ptrain = list()
        for i, img_id in enumerate(df_pre):
            img = cv2.imread("../data/images/roof_images/" + str(img_id) + ".jpg")
            resized = transform.resize(img, (64, 64) )
            ptrain.append(resized)

        ptrainX = np.asarray(ptrain)

        print 'All sets:'
        print 'x.shape = ', ptrainX.shape

        if augment:
            ptrainX, _ = augment_data(ptrainX)
            print 'Augmnented: x.shape = ', ptrainX.shape

        if zca_whitening:
            ptrainX, _ = zca_whitening(ptrainX, build=False, save=False)
            if save:
                np.savez_compressed(DEFAULT_ZCA_NPZ % dset, x=ptrainX)

        elif save:
            np.savez_compressed(PTRAIN_NPZ, x=ptrainX)
        return ptrainX


def augment_data(x_org, y_org=None):

    if y_org is None:
        y_org = [-1] * x_org.shape[0]
    trainX, trainY = list(), list()

    for x0, y0 in zip(x_org, y_org):

        num_rows, num_cols = x0.shape[:2]
        augmentX, augmentY = [], []

        augmentX.append(x0)
        augmentY.append(y0)

        # doublicate the data
        if 0 < y0 < 4:
            rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 90, 1)
            img_rotation = cv2.warpAffine(x0, rotation_matrix, (num_cols, num_rows))

            augmentX.append(img_rotation)
            if y0 == 1:
                augmentY.append(2)
            elif y0 == 2:
                augmentY.append(1)
            else:
                augmentY.append(y0)

        for resized, img_label in zip(augmentX, augmentY):

            # add original images
            trainX.append(resized)
            trainY.append(img_label)

            # flip images
            flipped_v = cv2.flip(resized,0)
            trainX.append(flipped_v)
            trainY.append(img_label)

            flipped_h = cv2.flip(resized,1)
            trainX.append(flipped_h)
            trainY.append(img_label)

            # # zooming
            # zoomed = cv2.resize(resized,None,fx=0.9, fy=0.9, interpolation = cv2.INTER_CUBIC)
            # zm1 = np.zeros_like(resized)
            # x = int(resized.shape[1]/2 - float(zoomed.shape[1]/2))
            # y = int(resized.shape[1]/2 - float(zoomed.shape[0]/2))
            # x_max = int(resized.shape[1]/2 + float(zoomed.shape[1]/2))
            # y_max = int(resized.shape[1]/2 + float(zoomed.shape[0]/2))
            # zm1[y:y_max, x:x_max] = zoomed
            # trainX.append(zm1)
            # trainY.append(img_label)
            #
            # slight rotation left and right
            rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 5, 1)
            img_rotation = cv2.warpAffine(resized, rotation_matrix, (num_cols, num_rows))
            trainX.append(img_rotation)
            trainY.append(img_label)

            rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), -5, 1)
            img_rotation = cv2.warpAffine(resized, rotation_matrix, (num_cols, num_rows))
            trainX.append(img_rotation)
            trainY.append(img_label)
            #
            # # rotate flipped
            # rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 5, 1)
            # img_rotation = cv2.warpAffine(flipped, rotation_matrix, (num_cols, num_rows))
            # trainX.append(img_rotation)
            # trainY.append(img_label)
            #
            # rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), -5, 1)
            # img_rotation = cv2.warpAffine(flipped, rotation_matrix, (num_cols, num_rows))
            # trainX.append(img_rotation)
            # trainY.append(img_label)
            #
            # # rotate zoomed
            # rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 5, 1)
            # img_rotation = cv2.warpAffine(zm1, rotation_matrix, (num_cols, num_rows))
            # trainX.append(img_rotation)
            # trainY.append(img_label)
            #
            # rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), -5, 1)
            # img_rotation = cv2.warpAffine(zm1, rotation_matrix, (num_cols, num_rows))
            # trainX.append(img_rotation)
            # trainY.append(img_label)

    return np.asarray(trainX), np.asarray(trainY)


ZCA_FILEPATH = '../data/zca.npz'

def zca_whitening(x, build=True, save=True):

    shape = x.shape
    x_flat = x.reshape((shape[0],-1))
    m = x_flat.mean(axis=0)
    x_flat = x_flat - ma.repmat(m, shape[0], 1)

    if build:
        sigma = np.dot(x_flat.T, x_flat) / shape[0] # correlation matrix
        U, S, V = np.linalg.svd(sigma) # singular values decomposition
        epsilon = 10e-16         # whitening constant, it prevents division by zeros_like
        w_zca = U.dot(np.diag(1./np.sqrt(np.diag(S+epsilon)))).dot(U.T)
        if save:
            np.savez_compressed(ZCA_FILEPATH, w=w_zca)
    else:
        data = np.load(ZCA_FILEPATH)
        w_zca = data['w']
    print x_flat.shape, w_zca.shape
    x = x_flat.dot(w_zca).reshape(shape)   # data whitening

    return x, w_zca



if __name__=='__main__':
     # build all
     # x = build('pretrain', save=False, augment=False, zca_whitening=False)

     # build zca map and whiten pretrained
     # x, _ = zca_whitening(x, build=True, save=True)

     # save all (for pretrain)
     # _ = build('pretrain', save=True, augment=False, zca_whitening=False)

     # build and save training set
     # _, _ = build('train', save=True, augment=True, zca_whitening=False)
     
     # build and save validation set
     _, _ = build('valid', save=True, augment=True, zca_whitening=False)
     
     # build and save test set
     # _ = build('test', save=True, augment=False, zca_whitening=True)
