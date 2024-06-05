# Dynamic Risk Assessment System
Dynamic Risk Assesment System to check client attrition rate

## Description
This project is part of Udacity MLOps Course. This project is about Model Scoring and Monitoring. The purpose of this project is to create, deploy, and monitor a ML model that will assess the attrition risk of the companys' clients. The project will also monitor model drift and set up processes to re-train, re-deploy, monitor and report on the model.

## Dependencies
This project dependencies is available in the ```requirements.txt``` file.

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the dependencies from the ```requirements.txt```. Its recommended to install it in a separate virtual environment.

```bash
pip install -r requirements.txt
```

## Steps Overview
1. **Data ingestion:** Automatically check if new data that can be used for model training. Compile all training data to a training dataset and save it to folder. 
2. **Training, scoring, and deploying:** Write scripts that train an ML model that predicts attrition risk, and score the model. Saves the model and the scoring metrics.
3. **Diagnostics:** Determine and save summary statistics related to a dataset. Time the performance of some functions. Check for dependency changes and package updates.
4. **Reporting:** Automatically generate plots and PDF document that report on model metrics and diagnostics. Provide an API endpoint that can return model predictions and metrics.
5. **Process Automation:** Create a script and cron job that automatically run all previous steps at regular intervals.

<img src="images/fullprocess.jpg" width=550 height=300>

## Usage

### 1- Edit config.json file to use practice data

```bash
"input_folder_path": "practicedata",
"output_folder_path": "ingesteddata", 
"test_data_path": "testdata", 
"output_model_path": "practicemodels", 
"prod_deployment_path": "production_deployment"
```

### 2- Run data ingestion
```
python ingestion.py
```

### 3- Model training
```python
python training.py
```

###  4- Model scoring 
```python
python scoring.py
```

### 5- Model deployment
```python
python deployment.py
```

### 6- Run diagnostics
```python
python diagnostics.py
```

### 7- Run reporting
```python
python reporting.py
```

### 8- Run Flask App
```python
python app.py
```

### 9- Run API endpoints
```python
python apicalls.py
```

### 11- Edit config.json file to use production data

```bash
"input_folder_path": "sourcedata",
"output_folder_path": "ingesteddata", 
"test_data_path": "testdata", 
"output_model_path": "models", 
"prod_deployment_path": "production_deployment"
```

### 10- Full process automation
```python
python fullprocess.py
```
### 11- Cron job

Start cron service
```bash
sudo service cron start
```

Edit crontab file
```bash
sudo crontab -e
```
   - Select **option 3** to edit file using vim text editor
   - Press **i** to insert a cron job
   - Write the cron job in ```cronjob.txt``` which runs ```fullprocces.py``` every 10 mins
   - Save after editing, press **esc key**, then type **:wq** and press enter
  
View crontab file
```bash
sudo crontab -l
```

## License
Distributed under the [MIT](https://choosealicense.com/licenses/mit/) License. See ```LICENSE``` for more information.