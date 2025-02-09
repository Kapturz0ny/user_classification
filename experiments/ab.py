import json
import numpy as np
import requests
from matplotlib import pyplot as plt

BASE_URL = "http://127.0.0.1:8000/api/base_model"
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
    precision_positive = true_positives / max(0.1, true_positives + false_positives)
    precision_negative = true_negatives / max(0.1, true_negatives + false_negatives)
    recall_positive = true_positives / max(0.1, true_positives + false_negatives)
    recall_negative = true_negatives / max(0.1, true_negatives + false_positives)
    f1_positive = 2 * precision_positive * recall_positive / max(0.1, precision_positive + recall_positive)
    f1_negative = 2 * precision_negative * recall_negative / max(0.1, precision_negative + recall_negative)

    return [true_positives, true_negatives, false_positives, false_negatives,
        accuracy, precision_positive, precision_negative, recall_positive, recall_negative, f1_positive, f1_negative]

def compare_results(a_predictions, b_predictions, expected):
    neither, only_a, only_b, both = 0, 0, 0, 0
    for a, b, y in zip(a_predictions, b_predictions, expected):
        if a != y and b != y:
            neither += 1
        if a == y and b != y:
            only_a += 1
        if a != y and b == y:
            only_b += 1
        if a == y and b == y:
            both += 1

    plt.title("Correct predictions for none/one/both models")
    plt.bar(['Neither', 'Only A', 'Only B', 'Both'], [neither, only_a, only_b, both])
    plt.savefig("ab_count.png")
    plt.cla()

def compare_stats(a_predictions, b_predictions, expected):
    base_stats = calculate_stats(a_predictions, expected)
    main_stats = calculate_stats(b_predictions, expected)

    width = 0.2
    x = np.arange(4)
    plt.bar(x - width/2, base_stats[0:4], width)
    plt.bar(x + width/2, main_stats[0:4], width)
    plt.xticks(x, ["True positives", "True negatives", "False positives", "False negatives"])
    plt.legend(["Base model", "Main model"])
    plt.title("True/false positives/negatives for each model")
    plt.savefig("ab_count2.png")
    plt.cla()

    width = 0.3
    plt.figure(figsize=(16, 6), dpi=80)
    x = np.arange(len(base_stats) - 4)
    plt.bar(x - width/2, base_stats[4:], width)
    plt.bar(x + width/2, main_stats[4:], width)
    plt.xticks(x, ["Accuracy", "Precision - premium", "Precision - no premium", "Recall - premium", "Recall - no premium", "F1 - premium", "F1 - no premium"])
    plt.legend(["Base model", "Main model"])
    plt.title("Other stats for each model")
    plt.savefig("ab_stats.png")

def main():
    with open("../transform/test.json") as file:
        data = json.load(file)
    request_base = requests.post(BASE_URL, json=data)
    request_main = requests.post(MAIN_URL, json=data)

    a_predictions = request_base.json()["will_buy_premium"]
    b_predictions = request_main.json()["will_buy_premium"]
    expected = [int(y) for y in data["expected_output"]]

    compare_results(a_predictions, b_predictions, expected)
    compare_stats(a_predictions, b_predictions, expected)

if __name__ == "__main__":
    main()