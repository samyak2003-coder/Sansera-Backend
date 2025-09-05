from flask import Flask, request, jsonify
import joblib
import xgboost as xgb
import pandas as pd
from flask_cors import CORS
app = Flask(__name__)
CORS(app) 
# =====================
# Load Models
# =====================
cost_estimator_models = {
    "2d_cost_estimate": joblib.load("models/RM/Cost-Estimator/2d_cost_estimate.pkl"),
    "3d_cost_estimate": joblib.load("models/RM/Cost-Estimator/3d_cost_estimate.pkl"),
}

part_rm_models = {
    "2d_lr_wd": joblib.load("models/RM/Part-RM/2d_lr_wd.pkl"),
    "3d_lr_lgf": joblib.load("models/RM/Part-RM/3d_lr_lgf.pkl"),
    "3d_lr_twt": joblib.load("models/RM/Part-RM/3d_lr_twt.pkl"),
    "3d_lr_wd": joblib.load("models/RM/Part-RM/3d_lr_wd.pkl"),
}

# Load XGBoost separately
xgb_model = xgb.Booster()
xgb_model.load_model("models/RM/Part-RM/2d_xgb_lgf.json")
part_rm_models["2d_xgb_lgf"] = xgb_model


# =====================
# Part RM Prediction Endpoint
# =====================
@app.route("/rm/part-rm/<model>", methods=["POST"])
def predict_part_rm(model):
    if model not in part_rm_models:
        return jsonify({"error": f"Model '{model}' not found for part_rm"}), 404
    
    try:
        # Ensure JSON → DataFrame
        data = request.get_json()
        df = pd.DataFrame(data if isinstance(data, list) else [data])

        model_obj = part_rm_models[model]

        if isinstance(model_obj, xgb.Booster):
            dmatrix = xgb.DMatrix(df)
            pred = model_obj.predict(dmatrix)
        else:
            pred = model_obj.predict(df)

        return jsonify({"prediction": pred.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =====================
# Cost Prediction Endpoint
# =====================
@app.route("/rm/cost/<model>", methods=["POST"])
def predict_cost(model):
    if model not in cost_estimator_models:
        return jsonify({"error": f"Model '{model}' not found for cost"}), 404

    try:
        # Ensure JSON → DataFrame
        data = request.get_json()
        df = pd.DataFrame(data if isinstance(data, list) else [data])

        pred = cost_estimator_models[model].predict(df)

        return jsonify({"prediction": pred.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =====================
# Health Check
# =====================
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "running"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)