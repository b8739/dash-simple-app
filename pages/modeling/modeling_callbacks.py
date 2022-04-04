from dash.dependencies import Output, Input, State, ALL, MATCH, ALLSMALLER
import pandas as pd
import plotly.express as px
from app import application
import time

import plotly.graph_objs as go
import math
from logic import algorithm
import dash_bootstrap_components as dbc  # pip3 install dash-bootstrap-components
import dash_daq as daq
import numpy as np
from app import cache
from utils.constants import TIMEOUT
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from utils.constants import theme
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from utils.constants import algorithm_type
from sklearn.preprocessing import StandardScaler
from dash_extensions.enrich import Dash, Trigger, ServersideOutput
from sklearn.model_selection import train_test_split
from dash.exceptions import PreventUpdate


""" CREATE MODEL """


@application.callback(
    ServersideOutput("model_store", "data"),
    Input("initial_store", "data"),
    State("veri_dropdown", "value"),
    State("model_store", "data"),
)
@cache.memoize(timeout=TIMEOUT)
def create_model(initial_store, data_idx, model):
    train_Xn, train_y = (
        initial_store["train_Xn"],
        initial_store["train_y"],
    )

    if data_idx == 0 or not data_idx:
        # if model_store:
        #     raise PreventUpdate
        # else:
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
    else:
        # Updating Training Data
        # XGB
        model["xgb"].fit(train_Xn, train_y)
    return model


""" GET PREDICTED RESULT """


@application.callback(
    Output("predict_value", "value"),
    Input("model_store", "data"),
    State("df_veri_store", "data"),
    State("initial_store", "data"),
    State("veri_dropdown", "value"),
    # prevent_initial_call=True,
)
@cache.memoize(timeout=TIMEOUT)
def update_predict_value(model, df_veri, initial_store, data_idx):
    if data_idx == 0:
        print("predict value", model["xgb"].predict(initial_store["test_Xn"]))
        return model["xgb"].predict(initial_store["test_Xn"])
        # return 29487
    veri_idx = int(data_idx) - 1

    xgb_veri_predict = model["xgb"].predict(
        initial_store["veri_Xn"][veri_idx].reshape(-1, 26)
    )
    # xgb_veri_predict = xgb_veri_predict.round(0)
    xgb_veri_predict = xgb_veri_predict.round(0)

    # Results
    # print("XGB_Pred = ", xgb_veri_predict)
    # print("RF_Pred = ", rf_veri_predict)
    # print("SVR_Pred = ", svr_veri_predict)
    # print("Actual = ", initial_store["veri_y"][veri_idx])

    return xgb_veri_predict


""" GET MODELING RESULT"""


# @application.callback(
#     Output("modeling_result_store", "data"),
#     Input("model_store", "data"),
#     State("initial_store", "data"),
# )
# @cache.memoize(timeout=TIMEOUT)
# def get_modeling_result(model_store, initial_store):

#     # 모델링 실행
#     rep_prediction = {"value": math.inf}

#     """ Modeling """
#     # 모델 만들고 실행
#     model = model_store
#     for i in algorithm_type:
#         result = model["xgb"].predict(initial_store["test_Xn"])
#         # 대푯값 비교해서 최소값으로 갱신
#         # if rep_prediction["value"] > result["RMSE"]:
#         #     rep_prediction = result
#         if i == "xgb":
#             rep_prediction = result

#     return rep_prediction


@application.callback(
    Output("modeling_assessment_store", "data"),
    Input("predict_value", "value"),
    State("model_store", "data"),
    State("initial_store", "data"),
)
@cache.memoize(timeout=TIMEOUT)
def get_evaluation(predict_value, model, initial_store):
    # print(len(predict_value))
    # print(len(initial_store["test_y"]))
    xgb_model_predict = model["xgb"].predict(initial_store["test_Xn"])
    evaluation = algorithm.evaluate_model(
        "xgb", xgb_model_predict, initial_store["test_y"]
    )
    return evaluation


""" GET MODELING ASSESSMENT """
""" Assessment"""


def create_callback(output):
    def get_modeling_assessment(modeling_assessment_store):
        if output == "MAPE_Value":
            value = modeling_assessment_store["MAPE_Value"]
        elif output == "R_square_XGB":
            value = modeling_assessment_store["R_square_XGB"]
        elif output == "RMSE":
            value = modeling_assessment_store["RMSE"]
        return round(value, 3)

    return get_modeling_assessment


for i in ["MAPE_Value", "RMSE"]:
    application.callback(
        Output(i, "value"),
        Input("modeling_assessment_store", "data"),
    )(create_callback(i))


""" SAVE ACTUAL PREDICTIVE STORE"""


@application.callback(
    ServersideOutput("actual_predict_store", "data"),
    Input("predict_value", "value"),
    State("initial_store", "data"),
    State("actual_predict_store", "data"),
    State("veri_dropdown", "value"),
    State("df_veri_store", "data"),
)
@cache.memoize(timeout=TIMEOUT)
def save_actual_predictive_df(
    predict_value, initial_store, actual_predict_store, dropdown, df_veri_store
):
    if len(predict_value) != 1:
        print("predict_value", predict_value)
        print("len(predict_value)", len(predict_value))
        print("actual_predict_store", actual_predict_store)
        """Actual Predictive"""
        result_df = algorithm.get_actual_predictive(
            initial_store["X_test"],
            initial_store["test_y"],
            predict_value,
        )
        # print(predict_value)
        # result_df_dict = result_df.to_dict("records")
        return result_df
    else:
        data_idx = dropdown - 1

        new_date = df_veri_store.iloc[data_idx]["date"]
        new_actual = initial_store["veri_y"].iloc[data_idx]
        print("new_actual", new_actual)
        print('initial_store["veri_y"]', initial_store["veri_y"])
        actual_predict_store.loc[len(actual_predict_store)] = [
            new_date,
            new_actual,
            predict_value[0],
        ]
        # print('new_date', new_date)
        # print('new_actual', new_actual)
        # print('predict_value', predict_value)
        print("new actual_predict_store", actual_predict_store)
        return actual_predict_store


""" DRAW ACTUAL VS PREDICTIVE GRAPH"""


@application.callback(
    Output("line_graph", "figure"),
    Input("actual_predict_store", "data"),
)
@cache.memoize(timeout=TIMEOUT)
def draw_actual_predict_graph(df):
    df = df.tail(30)
    trace_list = [
        go.Scatter(
            name="Actual",
            x=df["date"],
            y=df["Actual"],
            visible=True,
            mode="lines+markers",
            line={"width": 1},
            line_color="#f62e38",
            marker=dict(size=0.1),
        ),
        go.Scatter(
            name="Predictive",
            x=df["date"],
            y=df["Predictive"],
            visible=True,
            mode="lines+markers",
            line={"width": 1},
            line_color="#0abbbe",
            marker=dict(size=4),
        ),
    ]

    fig = go.Figure(data=trace_list)
    fig.update_layout(template="plotly_dark")

    fig.update_layout(
        title={
            "text": "바이오가스 생산량 예측값 실제값 비교",
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            # "font": {"size": 10},
        },
        paper_bgcolor="#32383e",
        plot_bgcolor="#32383e",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#696969")
    fig.update_yaxes(showgrid=True, gridcolor="#696969")
    # fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#18191A")
    return fig
