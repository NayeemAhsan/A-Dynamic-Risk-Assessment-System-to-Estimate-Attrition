from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

#################Load config.json and get path variables
logger.info("Load confif file and get path variables")
with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config['output_model_path']) 

#################Function for model scoring
def score_model():
    """
    This function calculates an F1 score for the model on the test data and saves the     result to a file
    """
    logger.info("Loading trained model")
    model = pickle.load(
        open(
            os.path.join(
                model_path,
                'trainedmodel.pkl'),
            'rb'))
    
    logger.info("Loading testdata.csv")
    test_df = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))

    logger.info("Preparing test data")
    y_true = test_df.pop('exited')
    X_df = test_df.drop(['corporation'], axis=1)

    logger.info("Predicting test data")
    y_pred = model.predict(X_df)
    f1 = f1_score(y_true, y_pred)
    print(f"f1 score = {f1}")

    logger.info("Saving scores to text file")
    with open(os.path.join(model_path, 'latestscore.txt'), 'w') as file:
        file.write(f"f1 score = {f1}")


if __name__ == '__main__':
    logger.info("Running scoring.py")
    score_model()
    

