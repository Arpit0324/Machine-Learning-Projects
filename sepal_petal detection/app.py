import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     float_features = [float(x) for x in request.form.values()]
#     features = [np.array(float_features)]
#     prediction = model.predict(features)
    
#     return render_template("index.html", prediction_text="The flower is {}".format(prediction))


@app.route("/predict", methods=["POST"])
def predict():
    json_ = request.json
    query_df = pd.DataFrame(json_)
    prediction = model.predict(query_df)
    # result = prediction.tolist()
    return jsonify({"placement_stat":prediction.tolist()})


if __name__ == "__main__":
    app.run(debug=True)