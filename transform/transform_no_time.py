import json
from copy import deepcopy
from datetime import datetime

import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

PRINT = True
PLAY_WEIGHT = 2
LIKE_WEIGHT = 10
SKIP_WEIGHT = -1
TRACK_PARAMETERS = ["popularity", "duration_ms", "explicit", "danceability", "loudness", "speechiness", "liveness",
                    "valence", "tempo", "time_signature"]
ATTRIBUTES = ['premium', 'sessions', 'time_spent_s', 'ads_displayed', 'ads_watched', 'ads_watch_rate',
              'total_weight', 'plays', 'likes', 'skips'] + TRACK_PARAMETERS
HIDDEN_ATTRIBUTES = ["ads_displayed", "ads_watched"]

def safe_divide(a, b):
    return a / b if b != 0 else 0

def to_datetime(timestamp: str) -> datetime:
    return datetime.strptime(timestamp[0:-3], "%Y-%m-%dT%H:%M:%S.%f")

def load_tracks() -> dict:
    dictionary = dict()
    with open("../data/tracks.jsonl", "r") as track_files:
        for line in track_files:
            track = json.loads(line)
            dictionary[track["id"]] = track
    return dictionary

def process_track_interaction(interaction: dict, user: dict, tracks: dict) -> None:
    if interaction["event_type"] == "Play":
        user["plays"] += 1
        weight = PLAY_WEIGHT
    elif interaction["event_type"] == "Like":
        user["likes"] += 1
        weight = LIKE_WEIGHT
    else:
        user["skips"] += 1
        weight = SKIP_WEIGHT
    user["total_weight"] += weight

    track = tracks[interaction["track_id"]]
    for parameter in TRACK_PARAMETERS:
        user[parameter] += track[parameter] * weight

def preprocess_users(users: dict) -> None:
    with open("../data/users.jsonl", "r") as users_file:
        for line in users_file:
            user_json = json.loads(line)
            user = dict()
            for attribute in ATTRIBUTES:
                user[attribute] = 0
            user["premium"] = 1 if user_json["premium_user"] is True else 0

            users[user_json["user_id"]] = user

def process_sessions(users: dict, tracks: dict) -> None:
    with open("../data/sessions.jsonl", "r") as sessions_file: # fake first interaction to be moved to old
        interaction = json.loads(sessions_file.readline())
        interaction["event_type"] = "Play"
        session_start = to_datetime(interaction["timestamp"])

    with open("../data/sessions.jsonl", "r") as sessions_file:
        for line in sessions_file:
            old = interaction
            old_user = users[old["user_id"]]
            interaction = json.loads(line)
            user = users[interaction["user_id"]]

            if old["session_id"] != interaction["session_id"]: # new session
                old_user["time_spent_s"] += (to_datetime(old["timestamp"]) - session_start).seconds
                old_user["sessions"] += 1
                session_start = to_datetime(interaction["timestamp"])
            elif old["event_type"] == "Advertisement": # no new session (didn't quit on ad)
                old_user["ads_watched"] += 1

            if interaction["event_type"] == "Advertisement":
                user["ads_displayed"] += 1
            elif interaction["event_type"] in ("Play", "Like", "Skip"):
                process_track_interaction(interaction, user, tracks)

    user["time_spent_s"] += (to_datetime(interaction["timestamp"]) - session_start).seconds
    user["sessions"] += 1

def postprocess_users(users: dict) -> None:
    for user in users.values():
        for preference in TRACK_PARAMETERS:
            user[preference] = safe_divide(user[preference], user["total_weight"])
        user["ads_watch_rate"] = safe_divide(user["ads_watched"], user["ads_displayed"])

def normalize_values(users: dict) -> None:
    max_values = deepcopy(users[list(users.keys())[0]])
    min_values = deepcopy(users[list(users.keys())[0]])

    for user in users.values():
        for key, item in user.items():
            max_values[key] = max(max_values[key], item)
            min_values[key] = min(min_values[key], item)

    for user in users.values():
        for key, item in user.items():
            user[key] = safe_divide(item - min_values[key], max_values[key] - min_values[key])

def serialize(user: dict) -> str:
    output = ""
    for attribute in [a for a in ATTRIBUTES if a not in HIDDEN_ATTRIBUTES]:
        output += f"{user[attribute]} "
    return output[:-1] + "\n"

def save(users: dict) -> None:
    with open("test.txt", "w") as test_file:
        with open("training.txt", "w") as training_file:
            for key in users.keys():
                if hash(key) % 5 == 0:
                    test_file.write(serialize(users[key]))
                else:
                    training_file.write(serialize(users[key]))

def convert_data_no_time() -> dict:
    users = dict()
    preprocess_users(users)
    process_sessions(users, load_tracks())
    postprocess_users(users)
    normalize_values(users)
    return users

def make_input_file():
    json_dict = dict()
    json_dict["input"] = list()
    json_dict["expected_output"] = list()
    with open("test.txt", "r") as file:
        for line in file:
            values = [float(item) for item in line.split()]
            json_dict["input"].append(values[1:])
            json_dict["expected_output"].append(values[0])
    with open("test.json", "w") as json_file:
        json.dump(json_dict, json_file)

def correlation_matrix(values: list) -> None:
    data = pd.DataFrame(values, columns=[a for a in ATTRIBUTES if a not in HIDDEN_ATTRIBUTES])
    plt.figure(figsize=(12, 10))
    matrix = data.corr(method='pearson')
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True, linewidths=0.5)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

def main():
    users = convert_data_no_time()
    save(users)
    make_input_file()

    if PRINT:
        for user in users.values():
            print(user)
        correlation_matrix([value for value in users.values()])

if __name__ == "__main__":
    main()
