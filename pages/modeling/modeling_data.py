import pandas as pd

from app import cache
from utils.constants import TIMEOUT
from logic import algorithm
import sys
from logic.prepare_data import dataframe, initial_data, extract_veri
import math
from logic import prepare_data, algorithm
import dash_bootstrap_components as dbc  # pip3 install dash-bootstrap-components
import dash_daq as daq
import plotly.graph_objs as go
from utils.constants import algorithm_type
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


@cache.memoize(timeout=TIMEOUT)
def create_model(train_Xn, train_y):
    model = {}
    # model = None
    model["xgb"] = xgb.XGBRegressor(
        n_estimators=1800,
        learning_rate=0.01,
        gamma=0.1,
        eta=0.04,
        subsample=0.75,
        colsample_bytree=0.5,
        max_depth=7,
    )
    model["xgb"].fit(train_Xn, train_y)

    model["svr"] = SVR(
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
    model["svr"].fit(train_Xn, train_y)

    model["rf"] = RandomForestRegressor(n_estimators=400, min_samples_split=3)
    model["rf"].fit(train_Xn, train_y)

    return model


@cache.memoize(timeout=TIMEOUT)
def get_modeling_result():

    # 모델링 실행
    rep_prediction = {"value": math.inf}

    """ Modeling """
    # 모델 만들고 실행
    model = create_model(initial_data()["train_Xn"], initial_data()["train_y"])
    for i in algorithm_type:
        result = algorithm.run(
            i, model[i], initial_data()["test_Xn"], initial_data()["test_y"]
        )
        # 대푯값 비교해서 최소값으로 갱신
        # if rep_prediction["value"] > result["RMSE"]:
        #     rep_prediction = result
        if i == "xgb":
            rep_prediction = result
    print("Modeling 실행 완료")
    return rep_prediction


@cache.memoize(timeout=TIMEOUT)
def get_modeling_assessment():
    rep_prediction = get_modeling_result()
    # assessment = ["MAPE_Value", "R_square_XGB", "RMSE"]
    assessment = ["MAPE_Value", "RMSE"]
    print("Modeling 평가 결과 저장 완료")
    return [
        dbc.Col(
            daq.LEDDisplay(
                id="our-LED-display",
                label=i,
                labelPosition="bottom",
                value=round(rep_prediction[i], 3),
                color="#fcdc64",
                size=24,
            ),
            width=3,
        )
        for i in assessment
    ]


@cache.memoize(timeout=TIMEOUT)
def verify(veri_idx):
    df_veri = extract_veri()
    df_veri.reset_index(drop=True, inplace=True)

    veri_x = df_veri.drop(
        ["Date", "Biogas_prod"], axis=1
    )  # Take All the columns except 'Biogas_prod'
    veri_y = df_veri["Biogas_prod"]

    scalerX = StandardScaler()  # Data standardization (to Standard Normal distribution)

    scalerX.fit(initial_data()["train_x"])

    veri_Xn = scalerX.transform(veri_x)  # Scaling the verifying data

    model = create_model(initial_data()["train_Xn"], initial_data()["train_y"])
    xgb_veri_predict = model["xgb"].predict(
        veri_Xn[veri_idx].reshape(-1, 26)
    )  # Apply xgb data to svr model
    # svr_veri_predict = model["svr"].predict(
    #    veri_Xn.iloc[veri_idx].reshape(-1, 26)
    # )  # Apply veri data to svr model
    # rf_veri_predict = model["rf"].predict(
    #     veri_Xn.iloc[veri_idx].reshape(-1, 26)
    # )  # Apply veri data to svr model

    # Results
    print("XGB_Pred = ", xgb_veri_predict)
    # print("RF_Pred = ", rf_veri_predict)
    # print("SVR_Pred = ", svr_veri_predict)
    print("Actual = ", veri_y[veri_idx])

    return xgb_veri_predict
