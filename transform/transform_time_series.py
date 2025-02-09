import json
# from copy import deepcopy
from datetime import datetime
import pandas as pd
# from matplotlib import pyplot as plt
# import seaborn as sns

PRINT = True
PLAY_WEIGHT = 2
LIKE_WEIGHT = 10
SKIP_WEIGHT = -1
TRACK_PARAMETERS = ["popularity", "duration_ms", "explicit", "speechiness", "liveness",
                    "valence", "tempo", "time_signature"]

# ads = number of ads displayed to user
ATTRIBUTES = ['premium', 'ads', 'ads_ascent', 'ads_descent', 'ads_base',
              'plays', 'plays_ascent', 'plays_descent', 'plays_base',
              'likes', 'likes_ascent', 'likes_descent', 'likes_base',
              'skips', 'skips_ascent', 'skips_descent', 'skips_base'] + TRACK_PARAMETERS

# with _prd -> ads_prd, plays_prd...
ARRAY_ATTRIBUTES = ["ads", "plays", "likes", "skips"]

# period = current period (3 months)
CONFIG_ATTRIBUTES = ["total_weight", "period", "bought_premium"]

# not used in a final model
# HIDDEN_ATTRIBUTES = []
HIDDEN_ATTRIBUTES = ["ads_descent", "plays", "plays_base", "likes_ascent",
                     "likes_descent", "skips_ascent", "skips_descent"]


def safe_divide(a, b):
    return a / b if b != 0 else 0


def get_month(timestamp: str) -> int:
    date = datetime.strptime(timestamp[0:-3], "%Y-%m-%dT%H:%M:%S.%f")
    return date.month


def load_tracks() -> dict:
    dictionary = dict()
    with open("../data/tracks.jsonl", "r") as track_files:
        for line in track_files:
            track = json.loads(line)
            dictionary[track["id"]] = track
    return dictionary


def preprocess_users(users: dict) -> None:
    with open("../data/users.jsonl", "r") as users_file:
        for line in users_file:
            user_json = json.loads(line)
            user = dict()
            for attribute in ATTRIBUTES + CONFIG_ATTRIBUTES:
                user[attribute] = 0
            for attribute in ARRAY_ATTRIBUTES:
                user[f"{attribute}_prd"] = [0]
            user["premium"] = 1 if user_json["premium_user"] is True else 0

            users[user_json["user_id"]] = user


def process_track_interaction(interaction: dict, user: dict, track: list) -> None:
    match interaction["event_type"]:
        case "Play":
            user["plays_prd"][user["period"]] += 1
            weight = PLAY_WEIGHT
        case "Like":
            user["likes_prd"][user["period"]] += 1
            weight = LIKE_WEIGHT
        case "Skip":
            user["skips_prd"][user["period"]] += 1
            weight = SKIP_WEIGHT
        case _:
            pass

    user["total_weight"] += weight

    for parameter in TRACK_PARAMETERS:
        user[parameter] += track[parameter] * weight


def process_sessions(users: dict, tracks: dict) -> None:
    with open("../data/sessions.jsonl", "r") as sessions_file: # fake first interaction to be moved to old
        interaction = json.loads(sessions_file.readline())

    with open("../data/sessions.jsonl", "r") as sessions_file:
        for line in sessions_file:

            # previous session
            old_session_id = interaction["session_id"]
            old_period = get_month(interaction["timestamp"])//4
            old_user_id = interaction["user_id"]

            # current session
            interaction = json.loads(line)
            period = get_month(interaction["timestamp"])//4
            user = users[interaction["user_id"]]

            if user["bought_premium"] == 1: # we analize user behaviour before purchasing premium account
                continue

            if old_user_id != interaction["user_id"]: # new user
                pass
            elif old_period != period: # new period
                if old_session_id != interaction["session_id"]: # new session
                    user["period"] +=  1
                    for attr in ARRAY_ATTRIBUTES:
                        user[f"{attr}_prd"].append(0)
            
            match interaction["event_type"]:
                case "BuyPremium":
                    user["bought_premium"] = 1
                case "Advertisement":
                    user["ads_prd"][user["period"]] += 1
                case _:
                    process_track_interaction(interaction, user, tracks[interaction["track_id"]])


def calculate_ascent_trend(user: dict, attr_name: str) -> int:
    max_value = max(user[f"{attr_name}_prd"])
    max_index = user[f"{attr_name}_prd"].index(max_value)
    if user[f"{attr_name}_prd"][:max_index]:
        min_value = min(user[f"{attr_name}_prd"][:max_index])
        min_index = user[f"{attr_name}_prd"].index(min_value)
    else:
        min_value, min_index = 0, 0
    return (max_value - min_value) * (max_index - min_index)


def calculate_descent_trend(user: dict, attr_name: str) -> int:
    min_value = min(user[f"{attr_name}_prd"])
    min_index = user[f"{attr_name}_prd"].index(min_value)
    if user[f"{attr_name}_prd"][:min_index]:
        max_value = max(user[f"{attr_name}_prd"][:min_index])
        max_index = user[f"{attr_name}_prd"].index(max_value)
    else:
        max_value, max_index = 0, 0
    return (max_value - min_value) * (min_index - max_index)


def calculate_periodic_attributes(user: dict, attr_name: str) -> None:
    user[attr_name] = sum(user[f"{attr_name}_prd"]) / (user["period"] + 1)
    user[f"{attr_name}_base"] = user[f"{attr_name}_prd"][0]
    user[f"{attr_name}_ascent"] = calculate_ascent_trend(user, attr_name)
    user[f"{attr_name}_descent"] = calculate_descent_trend(user, attr_name)


def postprocess_users(users: dict) -> None:
    for user in users.values():
        for attr in ARRAY_ATTRIBUTES:
            calculate_periodic_attributes(user, attr)
        for preference in TRACK_PARAMETERS:
            user[preference] = safe_divide(user[preference], user["total_weight"])


def standardize_values(users: dict) -> None:
    df = pd.DataFrame.from_dict(users, orient='index')
    standardized_df = df.copy()
    standardized_df[ATTRIBUTES] = (df[ATTRIBUTES] - df[ATTRIBUTES].mean()) / df[ATTRIBUTES].std()
    users = standardized_df.to_dict(orient='index')


def normalize_values(users: dict) -> None:
    df = pd.DataFrame.from_dict(users, orient='index')
    normalized_df = df.copy()
    normalized_df[ATTRIBUTES] = (df[ATTRIBUTES] - df[ATTRIBUTES].min()) / (df[ATTRIBUTES].max() - df[ATTRIBUTES].min())
    users = normalized_df.to_dict(orient='index')


# def normalize_values(users: dict) -> None:
#     max_values = deepcopy(users[list(users.keys())[0]])
#     min_values = deepcopy(users[list(users.keys())[0]])

#     for user in users.values():
#         for key, item in user.items():
#             max_values[key] = max(max_values[key], item)
#             min_values[key] = min(min_values[key], item)

#     for user in users.values():
#         for key, item in user.items():
#             user[key] = safe_divide(item - min_values[key], max_values[key] - min_values[key])


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


def convert_data() -> dict:
    users = dict()
    preprocess_users(users)
    process_sessions(users, load_tracks())
    postprocess_users(users)
    standardize_values(users)
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


# def correlation_matrix(values: list) -> None:
#     data = pd.DataFrame(values, columns=[a for a in ATTRIBUTES if a not in HIDDEN_ATTRIBUTES])
#     plt.figure(figsize=(12, 10))
#     matrix = data.corr(method='pearson')
#     sns.heatmap(matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True, linewidths=0.5)
#     plt.xticks(rotation=45, ha="right")
#     plt.tight_layout()
#     plt.savefig("corr_matrix.png")


def main():
    users = convert_data()
    save(users)
    make_input_file()

    for i, user in enumerate(users.values()):
        if i > 3: break
        print(user)
    # correlation_matrix({attr: user[attr] for attr in ATTRIBUTES if attr not in HIDDEN_ATTRIBUTES} for user in users.values())

if __name__ == "__main__":
    main()
