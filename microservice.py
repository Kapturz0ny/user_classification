from flask import Flask, request, jsonify
from model.base_model import BaseModel
from model.main_model import MainModel

app = Flask(__name__)
base_model = BaseModel()
main_model = MainModel()

# json format: {input: [[attr1, attr2, ...], [vector2], [vector3], ...], expected_output: [y1, y2, y3, ...]}
@app.route("/api/base_model", methods=['POST'])
def predict_from_base_model():
    result = base_model.predict(request.get_json()["input"])
    return jsonify({"will_buy_premium": result})

# json format: {input: [values from test data]}
@app.route("/api/main_model", methods=['POST'])
def predict_from_main_model():
    result = main_model.predict(request.get_json()["input"])
    return jsonify({"will_buy_premium": result})

if __name__ == "__main__":
    app.run(port=8000)
