from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()
###################Load config.json and get path variables
logger.info("Load confif file and get path variables")
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 

#################Function for training the model
def train_model():
    """
    Train logistic regression model on the ingested data and save it
    """
    logger.info("Loading and preparing the data")
    data_df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))

    # Drop the target variable
    y_df = data_df.pop('exited') # this is the target variable
        
    # Drop the column that will not be used in modeling
    X_df = data_df.drop(['corporation'], axis=1)

    # include only numeric columns
    X_df = X_df.select_dtypes(include=['number'])
        
    # Convert all data to numeric
    X_df = X_df.apply(pd.to_numeric, errors='coerce')

    # Handle missing values if any
    logger.info("handling missing values")
    if X_df.isnull().sum().sum() > 0:
        logger.warning("Missing values found, filling with mean values")
        X_df = X_df.fillna(X_df.mean())
    if y_df.isnull().sum().sum() > 0:
        logger.warning("Missing values found, filling with mean values")
        y_df = y_df.fillna(y_df.mean())
    
    # Ensure the target variable is categorical (integer type)
    y_df = y_df.astype(int)

    #logistic regression for training
    model = LogisticRegression(C=1.0,class_weight=None, 
                               dual=False, fit_intercept=True,
                               intercept_scaling=1, l1_ratio=None, max_iter=100,
                               multi_class='auto', n_jobs=None, penalty='l2',
                               random_state=0, solver='liblinear', 
                               tol=0.0001, verbose=0, warm_start=False)
    
    #fit the logistic regression model
    logger.info("Training model")
    model.fit(X_df, y_df)

    logger.info("Saving trained model")
    pickle.dump(
        model,
        open(
            os.path.join(
                model_path,
                'trainedmodel.pkl'),
            'wb'))
    
    # Save the feature names
    feature_names = X_df.columns.tolist()
    with open(os.path.join(model_path, 'feature_names.json'), 'w') as feature_file:
        json.dump(feature_names, feature_file)
   
if __name__ == '__main__':
    logger.info("Running training.py")
    train_model()
