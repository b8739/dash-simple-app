import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR

import numpy as np


def MAPE(y, pred):
    return np.mean(np.abs((y - pred) / y) * 100)


""" """ """""" """""" """ 02. PREDICTIVE MODELING  """ """""" """""" """" """

""" """ """""" " 1. XGBOOST ALGORITHM  " """""" " " ""


def create_model(algorithm, train_Xn, train_y):
    model = None
    if algorithm == "xgb":
        model = xgb.XGBRegressor(
            n_estimators=1800,
            learning_rate=0.01,
            gamma=0.1,
            eta=0.04,
            subsample=0.75,
            colsample_bytree=0.5,
            max_depth=7,
        )
    elif algorithm == "svr":
        model = SVR(
            kernel="rbf",
            C=100000,
            epsilon=0.9,
            gamma=0.0025,
            cache_size=200,
            coef0=0.0,
            degree=3,
            max_iter=-1,
            tol=0.0001,
        )
    elif algorithm == "rf":
        model = RandomForestRegressor(n_estimators=400, min_samples_split=3)

    if model is not None:
        model.fit(train_Xn, train_y)
    return model


def evaluate_model(algorithm, model_predict, test_y):
    RMSE = mean_squared_error(test_y, model_predict) ** 0.5
    print(algorithm)

    if algorithm == "xgb" or algorithm == "rf":
        # CONFIRM PREDICTION POWER #
        R_square_XGB = r2_score(test_y, model_predict)
        MAPE_Value = MAPE(test_y, model_predict)
        print("R_square: ", r2_score(test_y, model_predict))
        print("RMSE: ", RMSE)
        print("MAPE: ", MAPE(test_y, model_predict))
        return {
            "RMSE": RMSE,
            "R_square_XGB": R_square_XGB,
            "MAPE_Value": MAPE_Value,
        }
    elif algorithm == "svr":
        print("RMSE: ", RMSE)
        print("MAPE: ", MAPE(test_y, model_predict))
        return {"RMSE": RMSE}


def get_actual_predictive(x1, y_act, y_pred):
    z0 = pd.DataFrame(x1["date"])
    z0 = z0.reset_index(drop=True)
    z1 = pd.DataFrame(y_act)
    z1 = z1.reset_index(drop=True)
    z2 = pd.DataFrame(y_pred)
    result = pd.concat([z0, z1, z2], axis=1)
    print("result", result)
    result.columns = ["date", "Actual", "Predictive"]
    result = result.sort_values(by=["date"], axis=0, ascending=True)
    # result = result.set_index("date")
    return result
