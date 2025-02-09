# Classifiaction of users by probability to purchase a premium account

**Authors:** Marcin Bagnowski, Kacper Kowalczyk

Classification model that determines whether a user is likely to purchase a premium account based on their previous interactions with the system (see documentation for details). 

## Data
All used data can be downloaded from **[here](https://drive.google.com/file/d/1FPjkEAxlZTECQkThL3QiSvdH7D_nDT5E/view?usp=sharing)**.

Create `data` directory and extract there downloaded data.

## Model setup
Firstly, create virutal environment
`python3 -m venv myenv`

Activate it
`source myenv/bin/activate`

Install required packages
`pip install --no-cache-dir -r requirements.txt`

## Generating data
Enter **transform** directory and run `transform_no_time.py` to get model without time series or `transform_time_series.py` to get model with time series. After running one of those files you will get `test.json`, `test.txt` and `training.txt`.

## Model training
Enter **model** directory and `model_no_time.py` or `model_time_series` depending which model you have chosen earlier to get `model.pkl` (needs `test.txt` and `training.txt`).

## Starting microservice
Run `microservice.py` to start the microservice (needs `model.pkl`).

Enter **experiments** directory and run `ab.py` with the microservice running to perform A/B experiments.

To get predictions you can use `curl -X POST -H "Content-Type:application/json" --data "@transform/test.json" http://127.0.0.1:8000/api/main_model` (assuming pwd is the project's root directory).
