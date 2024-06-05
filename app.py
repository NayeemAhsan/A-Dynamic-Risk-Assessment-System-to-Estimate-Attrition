from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import diagnostics 
import subprocess
import json
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()
######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

logger.info("Load config file and get path variables")
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path']) 

# Load feature names
with open(os.path.join(prod_deployment_path, 'feature_names.json'), 'r') as feature_file:
    feature_names = json.load(feature_file)

prediction_model = None

#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():   
    '''logger.info("Loading and preparing testdata.csv")
    df_test = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    x_test = df_test.drop(['corporation'], axis=1)
  
    #call the prediction function you created in Step 3
    logger.info("call prediction function")
    preds = diagnostics.model_predictions(x_test)'''
    data = request.get_json()
    if 'data' not in data:
        return jsonify({"error": "No data provided"}), 400
    
    x_test = pd.DataFrame(data['data'])

    # Ensure the columns match the feature names
    x_test = x_test.reindex(columns=feature_names)
    
    logger.info("call prediction function")
    preds = diagnostics.model_predictions(x_test)
    return jsonify(preds.tolist())

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET', 'OPTIONS'])
def scoring():        
    # Run the scoring.py script
    result = subprocess.run(['python', 'scoring.py'], capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"scoring.py script failed with return code {result.returncode}")
        logger.error(f"Error output: {result.stderr}")
        return jsonify({"error": "scoring.py script failed"}), 500
    
    # Path to the file where scoring.py writes the F1 score
    score_file_path = os.path.join(prod_deployment_path, 'latestscore.txt')

    if not os.path.exists(score_file_path):
        logger.error(f"Score file {score_file_path} does not exist")
        return jsonify({"error": "Could not read F1 score"}), 500
    
    # Read the F1 score from the file
    try:
        with open(score_file_path, 'r') as score_file:
            score_str = score_file.read().strip()
            # Extract the float value from the string
            f1_score = float(score_str.split('=')[1].strip())
    except Exception as e:
        logger.error(f"Error reading F1 score: {e}")
        return jsonify({"error": "Could not read F1 score"}), 500
    
    # Return the F1 score as a JSON response
    return jsonify({"f1_score": f1_score})

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    #check means, medians, and modes for each column
    return jsonify(diagnostics.dataframe_summary())

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diag():        
    #check timing and percent NA values
    missing_values = diagnostics.missing_values()
    execution_time = diagnostics.execution_time()
    outdated_packages = diagnostics.outdated_packages_list()

    diag_list = {
        'missing_percentage': missing_values,
        'execution_time': execution_time,
        'outdated_packages': outdated_packages
    }
    #diag_list = diag_list.to_dict()

    return jsonify(diag_list)

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
