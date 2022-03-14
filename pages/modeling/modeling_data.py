import pandas as pd

from app import cache
from utils.constants import TIMEOUT
from logic import algorithm
import sys
from pages.monitoring.monitoring_data import dataframe, initial_data
import math
from logic import prepare_data, algorithm
import dash_bootstrap_components as dbc  # pip3 install dash-bootstrap-components
import dash_daq as daq
import plotly.graph_objs as go


@cache.memoize(timeout=TIMEOUT)
def get_modeling_result():
    df = dataframe()
    df_veri = prepare_data.extract_veri(df)

    # 모델링 실행
    rep_prediction = {"value": math.inf}

    """ Modeling """
    for algorithm_type in ["xgb", "rf", "svr"]:
        # 모델 만들고 실행
        model = algorithm.create_model(
            algorithm_type, initial_data()["train_Xn"], initial_data()["train_y"]
        )
        result = algorithm.run(
            algorithm_type, model, initial_data()["test_Xn"], initial_data()["test_y"]
        )
        # 대푯값 비교해서 최소값으로 갱신
        # if rep_prediction["value"] > result["RMSE"]:
        #     rep_prediction = result
        if algorithm_type == "xgb":
            rep_prediction = result
    print("Modeling 실행 완료")
    return rep_prediction


@cache.memoize(timeout=TIMEOUT)
def get_modeling_assessment():
    rep_prediction = get_modeling_result()
    assessment = ["MAPE_Value", "R_square_XGB", "RMSE"]
    print("Modeling 평가 결과 저장 완료")
    return [
        dbc.Col(
            daq.LEDDisplay(
                id="our-LED-display",
                label=i,
                value=round(rep_prediction[i], 3),
                color="#f4d44d",
                size=24,
            ),
            width=3,
        )
        for i in assessment
    ]


# @cache.memoize(timeout=TIMEOUT)
# def get_actual_predictive_df():

#     rep_prediction = get_modeling_result()
#     """Actual Predictive Dataframe"""
#     result_df = algorithm.get_actual_predictive(
#        initial_data()['X_test'], initial_data()['test_y'], rep_prediction["prediction"]
#     )
#     result_df_dict = result_df.to_dict("records")
#     print("Actual Predictive Data 저장 완료")

#     return result_df_dict
