import pandas as pd
import numpy as np
import cPickle as pkl
testX = pkl.load(open("../data/pkl/testX.pkl"))
# align dimensions 
testX = testX.transpose(0,3,1,2)
results = np.argmax(lognet.model.predict(testX),axis=-1) +1
