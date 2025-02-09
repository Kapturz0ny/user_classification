import pickle

class MainModel:
    def __init__(self):
        with open("model/model.pkl", "rb") as file:
            self.model = pickle.load(file)

    def predict(self, x):
        result = self.model.predict(x)
        result = [int(number) for number in result]
        return result
