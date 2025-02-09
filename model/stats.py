import json
import requests
from sklearn.metrics import roc_auc_score, average_precision_score

MAIN_URL = "http://127.0.0.1:8000/api/main_model"

def calculate_stats(predictions, expected):
    true_positives, true_negatives, false_positives, false_negatives = 0, 0, 0, 0
    
    for a, y in zip(predictions, expected):
        if a == 0 and y == 0:
            true_negatives += 1
        if a == 0 and y == 1:
            false_negatives += 1
        if a == 1 and y == 0:
            false_positives += 1
        if a == 1 and y == 1:
            true_positives += 1

    accuracy = (true_positives + true_negatives) / len(predictions)
    precision = true_positives / max(1, (true_positives + false_positives))
    recall = true_positives / max(1, (true_positives + false_negatives))
    f1 = 2 * precision * recall / max(1, (precision + recall))
    roc_auc = roc_auc_score(expected, predictions)
    ap = average_precision_score(expected, predictions)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "average_precision": ap,
    }

if __name__ == "__main__":
    
	with open("../transform/test.json") as file:
		data = json.load(file)

	expected = [int(y) for y in data["expected_output"]]
	request_main = requests.post(MAIN_URL, json=data)
	predictions = request_main.json()["will_buy_premium"]
	main_stats = calculate_stats(predictions, expected)

	for metric, value in main_stats.items():
		print(f"{metric}: {value:.4f}")
