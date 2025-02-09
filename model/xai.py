import shap
import json
import requests
import numpy as np
import matplotlib.pyplot as plt

MAIN_URL = "http://127.0.0.1:8000/api/main_model"

with open("../transform/test.json") as file:
    data = json.load(file)

input_data = data["input"]
input_data_np = np.array(input_data)
background_data = shap.sample(input_data_np, 100)

def model_predict(input_subset):
    input_subset = input_subset.tolist()
    response = requests.post(MAIN_URL, json={"input": input_subset})
    return np.array(response.json()["will_buy_premium"])

explainer = shap.KernelExplainer(model_predict, background_data)
shap_values = explainer.shap_values(input_data_np)
feature_names_no_time = ["sessions", "time_spent_s", "ads_watch_rate", "total_weight",
                        "plays", "likes", "skips", "popularity", "duration_ms", "explicit",
                        "danceability", "loudness", "speechiness", "liveness", "valence",
                        "tempo", "time_signature"]
feature_names_time_series = ["ads", "ads_ascent", "ads_base", "plays_ascent", "plays_descent", 
                            "likes", "likes_base", "skips","skips_base", "popularity",
                            "duration_ms", "explicit", "speechiness", "liveness", "valence",
                            "tempo", "time_signature"]

shap.summary_plot(shap_values, input_data_np, feature_names=feature_names_no_time)
