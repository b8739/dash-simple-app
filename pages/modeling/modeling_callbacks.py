from dash.dependencies import Output, Input, State, ALL, MATCH, ALLSMALLER
import pandas as pd
import plotly.express as px
from pages.modeling.modeling_data import get_modeling_result, initial_data, verify
from app import application

import plotly.graph_objs as go
import math
from logic import algorithm
import dash_bootstrap_components as dbc  # pip3 install dash-bootstrap-components
import dash_daq as daq
from xgboost import XGBRegressor
import numpy as np
from app import cache
from utils.constants import TIMEOUT
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from logic.prepare_data import dataframe
from utils.constants import theme


@application.callback(
    Output("actual_predict_store", "data"),
    Input("btn_3", "n_clicks"),
)
@cache.memoize(timeout=TIMEOUT)
def save_actual_predictive_df(n_clicks):

    rep_prediction = get_modeling_result()
    """Actual Predictive Dataframe"""
    result_df = algorithm.get_actual_predictive(
        initial_data()["X_test"], initial_data()["test_y"], rep_prediction["prediction"]
    )
    result_df_dict = result_df.to_dict("records")
    print("Actual Predictive Data 저장 완료")
    return result_df_dict


@application.callback(
    Output("line_graph", "figure"),
    Input("actual_predict_store", "data"),
)
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
