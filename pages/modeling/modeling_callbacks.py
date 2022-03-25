from dash.dependencies import Output, Input, State, ALL, MATCH, ALLSMALLER
import pandas as pd
import plotly.express as px
from app import application

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
from logic.prepare_data import to_dataframe
from dash_extensions.enrich import Dash, Trigger, ServersideOutput
from sklearn.model_selection import train_test_split
from dash.exceptions import PreventUpdate


@application.callback(
    ServersideOutput("initial_store", "data"),
    Input("x_y_store", "data"),
)
@cache.memoize(timeout=TIMEOUT)
def initial_data(x_y_store):  # split_dataset

    X, y = to_dataframe(x_y_store["X"]), pd.Series(x_y_store["y"])

    ## SET 'TRAIN', 'TEST' DATA, TRAIN/TEST RATIO, & 'WAY OF RANDOM SAMPLING' ##
    X_train, X_test, train_y, test_y = train_test_split(
        X, y, test_size=0.2, random_state=12345
    )

    # X_train, X_test, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 56789)

    train_x = X_train.drop(["date"], axis=1)  # Delete 'date' column from train data
    test_x = X_test.drop(["date"], axis=1)  # Delete 'date' column from test data

    # scalerX = MinMaxScaler()
    scalerX = StandardScaler()  # Data standardization (to Standard Normal distribution)
    # scalerX = RobustScaler()
    scalerX.fit(train_x)
    train_Xn = scalerX.transform(train_x)  # Scaling the train data
    test_Xn = scalerX.transform(test_x)  # Scaling the test data

    # train_b = scalerX.inverse_transform(train_Xn)
    dict_values = {
        # df
        "train_x": train_x,
        "test_x": test_x,
        "X_test": X_test,
        # numpy
        "train_Xn": train_Xn,
        "test_Xn": test_Xn,
        # series
        "train_y": train_y,
        "test_y": test_y,
    }

    return dict_values


""" CREATE MODEL """


@application.callback(
    ServersideOutput("model_store", "data"),
    Input("initial_store", "data"),
)
@cache.memoize(timeout=TIMEOUT)
def create_model(initial_store):

    train_Xn, train_y = initial_store["train_Xn"], initial_store["train_y"]
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


""" GET MODELING RESULT"""


@application.callback(
    Output("modeling_result_store", "data"),
    Input("model_store", "data"),
    State("initial_store", "data"),
)
@cache.memoize(timeout=TIMEOUT)
def get_modeling_result(model_store, initial_store):

    # 모델링 실행
    rep_prediction = {"value": math.inf}

    """ Modeling """
    # 모델 만들고 실행
    model = model_store
    for i in algorithm_type:
        result = algorithm.run(
            i, model[i], initial_store["test_Xn"], initial_store["test_y"]
        )
        # 대푯값 비교해서 최소값으로 갱신
        # if rep_prediction["value"] > result["RMSE"]:
        #     rep_prediction = result
        if i == "xgb":
            rep_prediction = result
    print("Modeling 실행 완료")

    return rep_prediction


""" GET MODELING ASSESSMENT """


# @cache.memoize(timeout=TIMEOUT)
# def get_modeling_assessment():
#     rep_prediction = get_modeling_result()
#     assessment = ["MAPE_Value", "R_square_XGB", "RMSE"]
#     # assessment = ["MAPE_Value", "RMSE"]
#     print("Modeling 평가 결과 저장 완료")
#     return [
#         dbc.Col(
#             daq.LEDDisplay(
#                 id="our-LED-display",
#                 label=i,
#                 labelPosition="bottom",
#                 value=round(rep_prediction[i], 3),
#                 color="#fcdc64",
#                 size=18,
#             ),
#             # width=3,
#         )
#         for i in assessment
#     ]


""" Assessment"""


def create_callback(output):
    def get_modeling_assessment(modeling_result_store):
        if output == "MAPE_Value":
            value = modeling_result_store["MAPE_Value"]
        elif output == "R_square_XGB":
            value = modeling_result_store["R_square_XGB"]
        elif output == "RMSE":
            value = modeling_result_store["RMSE"]
        return round(value, 3)

    return get_modeling_assessment


for i in ["MAPE_Value", "RMSE"]:
    application.callback(
        Output(i, "value"),
        Input("modeling_result_store", "data"),
    )(create_callback(i))


""" GET PREDICTED RESULT """


@application.callback(
    Output("predict_value", "value"),
    Input("veri_dropdown", "value"),
    State("df_veri_store", "data"),
    State("initial_store", "data"),
    State("model_store", "data"),
    # prevent_initial_call=True,
)
@cache.memoize(timeout=TIMEOUT)
def update_predict_value(data_idx, df_veri, initial_store, model_store):
    if not data_idx:
        raise PreventUpdate
    veri_idx = int(data_idx) - 1
    df_veri = to_dataframe(df_veri)
    df_veri.reset_index(drop=True, inplace=True)

    veri_x = df_veri.drop(
        ["Date", "Biogas_prod"], axis=1
    )  # Take All the columns except 'Biogas_prod'
    veri_y = df_veri["Biogas_prod"]

    scalerX = StandardScaler()  # Data standardization (to Standard Normal distribution)

    scalerX.fit(initial_store["train_x"])

    veri_Xn = scalerX.transform(veri_x)  # Scaling the verifying data

    model = model_store
    xgb_veri_predict = model["xgb"].predict(veri_Xn[veri_idx].reshape(-1, 26))
    # Results
    print("XGB_Pred = ", xgb_veri_predict)
    # print("RF_Pred = ", rf_veri_predict)
    # print("SVR_Pred = ", svr_veri_predict)
    print("Actual = ", veri_y[veri_idx])

    return xgb_veri_predict


""" SAVE ACTUAL PREDICTIVE STORE"""


@application.callback(
    Output("actual_predict_store", "data"),
    Input("modeling_result_store", "data"),
    State("initial_store", "data"),
)
@cache.memoize(timeout=TIMEOUT)
def save_actual_predictive_df(modeling_result_store, initial_store):

    """Actual Predictive"""
    result_df = algorithm.get_actual_predictive(
        initial_store["X_test"],
        initial_store["test_y"],
        modeling_result_store["prediction"],
    )
    result_df_dict = result_df.to_dict("records")
    print("Actual Predictive Data 저장 완료")
    return result_df_dict


""" DRAW ACTUAL VS PREDICTIVE GRAPH"""


@application.callback(
    Output("line_graph", "figure"),
    Input("actual_predict_store", "data"),
)
@cache.memoize(timeout=TIMEOUT)
def draw_actual_predict_graph(df):
    print("draw_actual_predict_graph")
    df = pd.json_normalize(df)
    trace_list = [
        go.Scatter(
            name="Actual",
            x=df["date"],
            y=df["Actual"],
            visible=True,
            mode="lines+markers",
            line={"width": 1},
            line_color="#08a4a7",
            marker=dict(size=5),
        ),
        go.Scatter(
            name="Predictive",
            x=df["date"],
            y=df["Predictive"],
            visible=True,
            mode="lines+markers",
            line={"width": 1},
            line_color="#f1444c",
            marker=dict(size=0.3),
        ),
    ]

    fig = go.Figure(data=trace_list)

    fig.update_layout(
        title={
            "text": "Actual vs Predict",
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            # "font": {"size": 10},
        }
    )

    fig.update_layout(template="plotly_dark")
    print("Actual Predictive 그래프 그리기 완료")
    return fig
