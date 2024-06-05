from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
import shutil
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

##################Load config.json and correct path variable
logger.info("Load confif file and get path variables")
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 

####################function for deployment
def deploy_model():
    """
    Copy the latest model file, the latest model score file,
    and the ingested metadata into the deployment directory
    """
    logger.info("Deploying trained model to production")
    logger.info(
        "Copying trainedmodel.pkl, ingestfiles.txt and latestscore.txt")
    shutil.copy(
        os.path.join(
            dataset_csv_path,
            'ingestedfiles.txt'),
        prod_deployment_path)
    shutil.copy(
        os.path.join(
            model_path,
            'trainedmodel.pkl'),
        prod_deployment_path)
    shutil.copy(
        os.path.join(
            model_path,
            'latestscore.txt'),
        prod_deployment_path)
    shutil.copy(
        os.path.join(
            model_path,
            'feature_names.json'),
        prod_deployment_path)

if __name__ == '__main__':
    logger.info("Running deployment.py")
    deploy_model()
        
        

