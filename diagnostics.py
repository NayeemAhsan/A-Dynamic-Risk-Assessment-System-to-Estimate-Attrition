
import pandas as pd
import numpy as np
import timeit
import os
import sys
import json
import pickle
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

##################Load config.json and get environment variables
logger.info("Load confif file and get path variables")
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 

##################Function to get model predictions
def model_predictions(x_df):
    logger.info("Loading deployed model")
    model = pickle.load(
        open(
            os.path.join(
                prod_deployment_path,
                'trainedmodel.pkl'),
            'rb'))

    logger.info("Running predictions on data")
    y_pred = model.predict(x_df)
    return y_pred

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    logger.info("Loading and preparing finaldata.csv")
    data_df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    data_df = data_df.drop(['exited'], axis=1)
    data_df = data_df.select_dtypes('number')

    logger.info("Calculating statistics for data")
    statistics_dict = {}
    for col in data_df.columns:
        mean = data_df[col].mean()
        median = data_df[col].median()
        std = data_df[col].std()

        statistics_dict[col] = {'mean': mean, 'median': median, 'std': std}

    return statistics_dict

##################Function to calculate missing values#######
def missing_values():
    """
    Calculates percentage of missing data for each column in finaldata.csv
    """
    logger.info("Loading and preparing finaldata.csv")
    data_df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))

    logger.info("Calculating missing data percentage")
    missing_values_list = {col: {'percentage': perc} for col, perc in zip(
        data_df.columns, data_df.isna().sum() / data_df.shape[0] * 100)}

    return missing_values_list

##################Function to get timings
def _ingestion_timing():
    """
    Runs ingestion.py script and measures execution time
    """
    logger.info("Calculating time for one run for ingestion.py")
    starttime = timeit.default_timer()
    _= subprocess.run(['python', 'ingestion.py'], capture_output=True)
    timing = timeit.default_timer() - starttime
    return timing

def _training_timing():
    """
    Runs training.py script and measures execution time
    """
    logger.info("Calculating time for one run for training.py")
    starttime = timeit.default_timer()
    _= subprocess.run(['python', 'training.py'], capture_output=True)
    timing = timeit.default_timer() - starttime
    return timing

def execution_time():
    #calculate timing of training.py and ingestion.py
    logger.info("Calculating time for ingestion.py for 30 runs")
    ingestion_time = []
    for _ in range(30):
        time = _ingestion_timing()
        ingestion_time.append(time)

    logger.info("Calculating time for training.py for 30 runs")
    training_time = []
    for _ in range(30):
        time = _training_timing()
        training_time.append(time)

    logger.info("Getting the list of mean times for for ingestion and training")
    execution_time_list = [
        {'ingestion_time_mean': np.mean(ingestion_time)},
        {'training_time_mean': np.mean(training_time)}
    ]

    return execution_time_list

##################Function to check dependencies
def outdated_packages_list():
    #get a list of 
    try:
        # Get the list of installed packages and their versions
        installed_packages = subprocess.check_output(['pip', 'list'], encoding='utf-  8')
        installed_packages = installed_packages.split('\n')[2:-1]  # Skip the headers and split lines
        
        installed_dict = {}
        for package in installed_packages:
            name, version = package.split()[:2]
            installed_dict[name] = version
            
        # Get the list of outdated packages
        outdated_packages = subprocess.check_output(['pip', 'list', '--outdated'], encoding='utf-8')
        outdated_packages = outdated_packages.split('\n')[2:-1]  # Skip the headers and split lines

        outdated_dict = {}
        for package in outdated_packages:
            name, current_version, latest_version = package.split()[:3]
            outdated_dict[name] = latest_version
            
        # Read the requirements.txt file
        with open('requirements.txt') as f:
            required_packages = f.read().splitlines()
            
        # Prepare the data for the table
        data = []
        for package in required_packages:
            if package.strip() == "" or package.startswith("#"):
                continue
            name = package.split("==")[0]
            current_version = installed_dict.get(name, 'Not installed')
            latest_version = outdated_dict.get(name, current_version)  # Use current_version if not outdated
            data.append([name, current_version, latest_version])

        # Create a DataFrame and print the table
        df = pd.DataFrame(data, columns=['Module', 'Current Version', 'Latest Version'])
        #print(df.to_string(index=False))
        # Save the DataFrame to a text file
        df.to_csv('dependency_list.txt', sep=' ', index=False)

        logger.info(f"Dependency list saved to dependency_list.txt")
        
        return df.to_dict(orient='records')
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error while executing pip command: {e}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")


if __name__ == '__main__':
    logger.info("Loading and preparing testdata.csv")
    df_test = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    x_test = df_test.drop(['corporation', 'exited'], axis=1)
    model_predictions(x_test)
    dataframe_summary()
    execution_time()
    outdated_packages_list()





    
