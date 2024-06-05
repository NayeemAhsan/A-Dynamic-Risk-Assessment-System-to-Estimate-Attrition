import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import logging
import diagnostics

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

############### Load config.json and get path variables.
logger.info("Load config file and get path variables")
with open('config.json', 'r') as f:
    config = json.load(f)

test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

############## Function for confusion matrix
def plt_confusion_matrix():
    """
    Calculate a confusion matrix using the test data and the deployed model
    """
    logger.info("Loading and preparing testdata.csv")
    df_test = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))

    y_true = df_test.pop('exited')
    x_test = df_test.drop(['corporation'], axis=1)

    logger.info("Predicting test data")
    y_pred = diagnostics.model_predictions(x_test)

    logger.info("Plotting and saving confusion matrix")
    cm = metrics.confusion_matrix(y_true, y_pred)  # Calculate confusion matrix

    # Plot confusion matrix using seaborn
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Model Confusion Matrix")
    plt.savefig(os.path.join(prod_deployment_path, 'confusionmatrix2.png'))
    plt.close()

if __name__ == '__main__':
    plt_confusion_matrix()