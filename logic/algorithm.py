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
    if algorithm == "xgboost":
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
        svr_model = SVR(
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
    if model:
        model.fit(train_Xn, train_y)
    return model


def run_xgboost(xgb_model, test_Xn, test_y):
    xgb_model_predict = xgb_model.predict(test_Xn)

    # CONFIRM PREDICTION POWER #
    print("R_square_XGB :", r2_score(test_y, xgb_model_predict))
    print("RMSE_XGB :", mean_squared_error(test_y, xgb_model_predict) ** 0.5)
    print("MAPE_XGB :", MAPE(test_y, xgb_model_predict))

    """ Draw performance graph : 'Actual' vs 'Predictive' """


""" """ """""" " 2. RANDOM FOREST ALGORITHM  " """""" """ """


def run_svr(rf_model, test_Xn, test_y):
    rf_model_predict = rf_model.predict(test_Xn)

    print("R_square :", r2_score(test_y, rf_model_predict))
    print("RMSE :", mean_squared_error(test_y, rf_model_predict) ** 0.5)
    print("MAPE :", MAPE(test_y, rf_model_predict))


""" """ """""" " 3. SVR ALGORITHM  " """""" """ """


def run_svr(svr_model, test_Xn, test_y):
    svr_model_predict = svr_model.predict(test_Xn)

    print("RMSE :", mean_squared_error(test_y, svr_model_predict) ** 0.5)
    print("MAPE :", MAPE(test_y, svr_model_predict))
