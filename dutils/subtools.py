import pandas as pd
import numpy as np
import logging, logging.config, yaml

with open ( 'logging.yaml', 'rb' ) as config:
    logging.config.dictConfig(yaml.load(config))
    logger = logging.getLogger('root')

"""
The tools within this module allow for fast
creation of submission csvs for the kaggle
DSG quallification round
"""


def get_results(model, dset='test'):
    """
    This function takes in a trained model
    and generates its results ready for create_submission
    """
    data = np.load( "../data/pkl/%s.npz" % dset )
    # align dimensions such that channels are the
    # second axis of the tensor
    x = data['x'].transpose(0,3,1,2)
    logger.debug('Predicting labels...')
    results = np.argmax(model.predict(x),axis=-1) + 1
    return results


def submit(model, sub=100):
    """
    This function creates a csv file from results
    ready to submit straight to the DSG challenge
    """
    results = get_results(model)
    logger.debug('Saving labels in file "../data/csv_lables/sub%d.csv"' % sub)

    submission_example = pd.read_csv("../data/csv_lables/sample_submission4.csv")
    submission_example["label"] = results
    submission_example.to_csv("../data/csv_lables/sub%d.csv"%sub,index=False)
    logger.debug( "Submitted at: " + ("../data/csv_lables/sub%d.csv"%sub) )
