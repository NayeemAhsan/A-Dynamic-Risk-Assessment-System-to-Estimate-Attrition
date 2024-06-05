import os
import logging
import json
import subprocess
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

logger.info("Load config file and get path variables")
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 

# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

# Helper function to run subprocess with curl and capture output
def run_curl_command(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Command failed: {command}\n{result.stderr}")
        return None
    return result.stdout

# Preprocess the test data
logger.info("Preprocessing test data")
testdata_filepath = os.path.join(test_data_path, "testdata.csv")
df_test = pd.read_csv(testdata_filepath)
x_test = df_test.drop(['corporation'], axis=1)

# Convert x_test to JSON
x_test_json = x_test.to_dict(orient='records')

# Call prediction endpoint
logger.info("Get prediction on test data")
prediction_command = f'curl -X POST {URL}/prediction -H "Content-Type: application/json" -d \'{{"data": {json.dumps(x_test_json)}}}\''
response1 = run_curl_command(prediction_command)

# Check if prediction was successful
if response1 is None:
    logger.error("Prediction request failed. Check the log for details.")
else:
    logger.info("Prediction response: " + response1)

# Call scoring endpoint
logger.info("Get F1 scoring")
scoring_command = f'curl {URL}/scoring'
response2 = run_curl_command(scoring_command)

# Check if scoring was successful
if response2 is None:
    logger.error("Scoring request failed. Check the log for details.")
else:
    logger.info("Scoring response: " + response2)

# Call summary endpoint
logger.info("Get summary stats")
summarystats_command = f'curl {URL}/summarystats'
response3 = run_curl_command(summarystats_command)

# Check if summary stats was successful
if response3 is None:
    logger.error("Summary stats request failed. Check the log for details.")
else:
    logger.info("Summary stats response: " + response3)

# Call diagnostics endpoint
logger.info("Get diagnostics report")
diagnostics_command = f'curl {URL}/diagnostics'
response4 = run_curl_command(diagnostics_command)

# Check if diagnostics was successful
if response4 is None:
    logger.error("Diagnostics request failed. Check the log for details.")
else:
    logger.info("Diagnostics response: " + response4)

# Check if any response is None (meaning the curl command failed)
if None in [response1, response2, response3, response4]:
    logger.error("One or more requests failed. Check the log for details.")
else:
    # Combine all responses into one file
    logger.info("Generating report text file")
    with open(os.path.join(prod_deployment_path, 'apireturns2.txt'), 'w') as file:
        file.write('Ingested Data\n\n')
        file.write('Statistics Summary\n')
        file.write(response3)
        file.write('\nDiagnostics Summary\n')
        file.write(response4)
        file.write('\n\nTest Data\n\n')
        file.write('Model Predictions\n')
        file.write(response1)
        file.write('\nModel Score\n')
        file.write(response2)