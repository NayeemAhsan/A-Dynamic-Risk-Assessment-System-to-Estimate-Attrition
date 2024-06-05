import os
import re
import sys
import logging
import pandas as pd
import json
from sklearn.metrics import f1_score
import ingestion
import training
import scoring
import deployment
import diagnostics
import reporting

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()
##################Load config.json and get environment variables
logger.info("Load confif file and get path variables")
with open('config.json','r') as f:
    config = json.load(f) 
input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
def main():
    ##################Check and read new data
    logger.info("Checking for new data")
    #read ingestedfiles.txt
    with open(os.path.join(prod_deployment_path, "ingestedfiles.txt")) as file:
        ingested_files = {line.strip('\n').split('/')[-1] for line in file.readlines()[1:]}       
    #check whether the source data folder has files that aren't listed in ingestedfiles.txt
    source_files = set(os.listdir(input_folder_path))
    ##################Deciding whether to proceed, part 1
    logger.info(f"Ingested Files: {ingested_files}")
    logger.info(f"Source Files: {source_files}")
    if not source_files.difference(ingested_files):
        logger.info("No new data found")
        return None 
    
    #Ingesting new data
    logger.info("New data found")
    logger.info("Ingesting new data")
    ingestion.merge_multiple_dataframe()
    
    ##################Checking for model drift
    #check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
    logger.info("Checking for model drift")
    latest_score_file_path = os.path.join(prod_deployment_path, "latestscore.txt")
    try:
        with open(latest_score_file_path, 'r') as file:
            file_content = file.read()
            logger.info(f"Contents of latestscore.txt: {file_content}")
            # Use a more specific regex to match the F1 score format
            match = re.search(r'f1 score\s*=\s*(\d*\.?\d+)', file_content, re.IGNORECASE)
            if match:
                deployed_score = float(match.group(1))
                logger.info(f"Extracted F1 score: {deployed_score}")
                #print(deployed_score)
            else:
                logger.error("F1 score not found in latestscore.txt")
                #print("F1 score not found")
    except Exception as e:
        logger.error(f"Error reading latestscore.txt: {e}")
        #print(f"Error reading latestscore.txt: {e}")

    data_df = pd.read_csv(os.path.join(output_folder_path, 'finaldata.csv'))
    y_df = data_df.pop('exited')
    X_df = data_df.drop(['corporation'], axis=1)

    y_pred = diagnostics.model_predictions(X_df)
    new_score = f1_score(y_df.values, y_pred)

    ##################Deciding whether to proceed, part 2
    logger.info(f"Deployed score = {deployed_score}")
    logger.info(f"New score = {new_score}")
    #if you found model drift, you should proceed. otherwise, do end the process here
    if(new_score >= deployed_score):
        logging.info("No model drift occurred")
        return None
    
    logger.info("model drift occurred")
    # Re-training
    logger.info("Re-training model")
    training.train_model()
    logger.info("Re-scoring model")
    scoring.score_model()
    ##################Re-deployment
    logger.info("Re-deploying model")
    #if you found evidence for model drift, re-run the deployment.py script
    deployment.deploy_model()
    ##################Diagnostics and reporting
    logger.info("Running diagnostics and reporting")
    #Run diagnostics.py and reporting.py for the re-deployed model
    reporting.plt_confusion_matrix()
    logger.info("calling API endpoints")
    os.system("python apicalls.py")
    
    
if __name__ == '__main__':
    main()