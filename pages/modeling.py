import dash
import pandas as pd
import numpy as np
import dash_daq as daq
import plotly.graph_objs as go

from collections import OrderedDict

dash.register_page(__name__, path="/modeling", name="Modeling", order=2)

# dash.register_page(__name__)

from dash import Dash, dcc, html, Input, Output, callback, dash_table
from dash.dependencies import Output, ALL, State, MATCH, ALLSMALLER

import plotly.express as px
import dash_bootstrap_components as dbc  # pip3 install dash-bootstrap-components
from dash_extensions.enrich import (
    DashProxy,
    Input,
    Output,
    TriggerTransform,
    ServersideOutputTransform,
    ServersideOutput,
    Trigger,
)
import sys

sys.path.append("../logic")
import prepare_data


# from dataset import df

# from dataset import result_df

import sys

sys.path.append("../logic")
import algorithm

# from dataset import df

# from dataset import result_df


predict = np.random.randn(500)
actual = np.random.randn(500)
type = ["SVR Pred.", "RF Pred.", "ENSEMBLE Pred.", "Actual MI"] * (500 // 4)


df = pd.read_csv("ketep_biogas_data_20220210.csv")
dff = {"key": df.columns[0:8], "val": [v for v in df.iloc[0, [0, 1, 2, 3, 4, 5, 6, 7]]]}

theme = {
    "dark": True,
    "detail": "#007439",
    "primary": "#00EA64",
    "secondary": "#6E6E6E",
}


def makeBarGraph():
    fig = px.bar(dff, x="val", y="key", orientation="h", template="plotly_dark")
    fig.update_traces(
        marker_color=theme["primary"],
        # marker_color=theme["primary"],
        marker_line_color=theme["primary"],
        marker_line_width=1.5,
        opacity=0.6,
    )
    return fig


algorithm_list = ["XGBoost", "SVR", "LSTM", "Ensemble"]


# Card

card = dbc.Card(
    dbc.CardBody(
        [
            html.Label("88%", id="MAPE", className="card-title"),
            html.H6("MAPE", className="card-subtitle"),
        ]
    ),
    inverse=True,
    style={
        "width": "10rem",
    },
    id="card",
)


# Layout

layout = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        # card,
                        style={
                            "display": "flex",
                            "align-items": "center",
                            "justify-content": "space-around",
                        },
                    )
                )
                for _ in range(4)
            ],
        ),
        dbc.Row(id="model_assessment"),
        html.Br(),
        dbc.Row(
            [
                # dbc.Col(
                #     dcc.Graph(
                #         id="bar_graph",
                #         style={"height": "30vh", "width": "70vh"},
                #     ),
                # ),
                dbc.Col(
                    dcc.Graph(
                        id="line_graph",
                        style={"height": "50vh", "width": "70vh"},
                    ),
                ),
            ]
        ),
        html.Br(),
        dcc.Loading(
            [
                dcc.Store(
                    id="actual_predict_store",
                    storage_type="session",
                ),
                dcc.Store(
                    id="modeling_result_store",
                    storage_type="session",
                ),
            ],
            # fullscreen=True,
            type="dot",
        ),
    ]
)

import math

train_Xn, train_y, test_Xn, test_y, X_test = None, None, None, None, None


@callback(
    Output("modeling_result_store", "data"),
    Input("preprocessed_store", "data"),
    # prevent_initial_call=True,
    memoize=True,
)
def store_modeling_result(df):
    global train_Xn, train_y, test_Xn, test_y, X_test

    # if n_clicks == None:
    #     raise PreventUpdate
    df = pd.json_normalize(df)
    df_veri = prepare_data.extract_veri(df)
    train_Xn, train_y, test_Xn, test_y, X_test = prepare_data.split_dataset(df)
    # 모델링 실행
    rep_prediction = {"value": math.inf}

    """ Modeling """
    for algorithm_type in ["xgb", "rf", "svr"]:
        # 모델 만들고 실행
        model = algorithm.create_model(algorithm_type, train_Xn, train_y)
        result = algorithm.run(algorithm_type, model, test_Xn, test_y)
        # 대푯값 비교해서 최소값으로 갱신
        # if rep_prediction["value"] > result["RMSE"]:
        #     rep_prediction = result
        if algorithm_type == "xgb":
            rep_prediction = result
    print("Modeling 실행 완료")
    return rep_prediction

    # Actual: test_y
    # Predict: xgb_model_predict (rep_prediction['prediction])


@callback(
    Output("model_assessment", "children"),
    Input("modeling_result_store", "data"),
    # prevent_initial_call=True,
    memoize=True,
)
def update_model_assessment(rep_prediction):
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


@callback(
    Output("actual_predict_store", "data"),
    Input("modeling_result_store", "data"),
    prevent_initial_call=True,
    memoize=True,
)
def start_modeling(rep_prediction):
    global X_test, test_y

    """Actual Predictive Dataframe"""
    result_df = algorithm.get_actual_predictive(
        X_test, test_y, rep_prediction["prediction"]
    )
    result_df_dict = result_df.to_dict("records")
    print("Actual Predictive Data 저장 완료")

    return result_df_dict


@callback(
    Output("line_graph", "figure"),
    Input("actual_predict_store", "data"),
    # prevent_initial_call=True,
    memoize=True,
)
def draw_actual_predict_graph(df):
    df = pd.json_normalize(df)
    trace_list = [
        go.Scatter(
            name="Actual",
            y=df["Actual"],
            visible=True,
            mode="lines+markers",
            line={"width": 1},
            line_color="#0066E7",
            marker=dict(size=5),
        ),
        go.Scatter(
            name="Predictive",
            y=df["Predictive"],
            visible=True,
            mode="lines+markers",
            line={"width": 1},
            line_color="#D4070F",
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
        }
    )

    fig.update_layout(template="plotly_dark")
    print("Actual Predictive 그래프 그리기 완료")
    return fig
