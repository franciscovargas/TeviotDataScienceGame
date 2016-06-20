import pandas as pd
import numpy as np

"""
The tools within this module allow for fast
creation of submission csvs for the kaggle
DSG quallification round
"""


def get_results(model, test_set="testX"):
    """
    This function takes in a trained model
    and generates its results ready for create_submission
    """
    testX = pkl.load(open("../data/pkl/%s.pkl" % test_set ))
    # align dimensions such that channels are the
    # second axis of the tensor
    testX = testX.transpose(0,3,1,2)
    results = np.argmax(model.predict(testX),axis=-1) +1
    return results


def create_submision(results, sub=1):
    """
    This function creates a csv file from results
    ready to submit straight to the DSG challenge
    """
    submission_example = pd.read_csv("../data/csv_lables/sample_submission4.csv")
    submission_example["label"] = results
    submission_example.to_csv("../data/csv_lables/sub%d.csv"%sub,index=False)
    print "submitted at: " + ("../data/csv_lables/sub%d.csv"%sub)
