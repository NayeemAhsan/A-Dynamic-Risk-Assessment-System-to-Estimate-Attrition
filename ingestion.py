import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

#############Load config.json and get input and output paths
logger.info("Load confif file and get path variables")
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    df_list = []
    file_names = []

    logger.info("Reading files from {input_folder_path}")
    for file in os.listdir(input_folder_path):
        file_path = os.path.join(input_folder_path, file)
        df_tmp = pd.read_csv(file_path)

        file = os.path.join(*file_path.split(os.path.sep)[-3:])
        file_names.append(file)

        df_list.append(df_tmp)

    logger.info("Concatenating dataframes")
    df = pd.concat(df_list, ignore_index=True)

    logger.info("Dropping duplicates")
    df = df.drop_duplicates().reset_index(drop=1)

    logger.info("Saving ingested data")
    df.to_csv(os.path.join(output_folder_path, 'finaldata.csv'), index=False)
    
    logger.info("Saving ingested data")
    with open(os.path.join(output_folder_path, 'ingestedfiles.txt'), "w") as file:
        file.write(
            f"Ingestion date: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        file.write("\n".join(file_names))

if __name__ == '__main__':
    logger.info("Running ingestion.py")
    merge_multiple_dataframe()
