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
import shap
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
            line_color="#0066E7",
            marker=dict(size=5),
        ),
        go.Scatter(
            name="Predictive",
            x=df["date"],
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


""" SHAP """


@cache.memoize(timeout=TIMEOUT)
def get_shap_values():
    train_x = initial_data()["train_x"]
    train_y = initial_data()["train_y"]
    model = XGBRegressor()
    model.fit(train_x, train_y)

    shap_values = shap.TreeExplainer(model).shap_values(train_x)

    return shap_values


@application.callback(
    Output("shap_importance_store", "data"),
    Input("btn_3", "n_clicks"),
)
@cache.memoize(timeout=TIMEOUT)
def get_shap_importance(n_clicks):
    train_x = initial_data()["train_x"]

    shap_values = get_shap_values()
    feature_names = train_x.columns

    rf_resultX = pd.DataFrame(shap_values, columns=feature_names)
    vals = np.abs(rf_resultX.values).mean(0)

    shap_importance = pd.DataFrame(
        list(zip(feature_names, vals)), columns=["col_name", "feature_importance_vals"]
    )

    shap_importance.sort_values(
        by=["feature_importance_vals"], ascending=False, inplace=True
    )
    shap_importance_dict = shap_importance.to_dict("records")

    return shap_importance_dict


@application.callback(
    Output("bar_graph", "figure"),
    Input("shap_importance_store", "data"),
)
@cache.memoize(timeout=TIMEOUT)
def draw_shap_bar_graph(df):
    df = pd.json_normalize(df)
    fig = px.bar(
        df[:5],
        x="feature_importance_vals",
        y="col_name",
        orientation="h",
        template="plotly_dark",
    )
    fig.update_traces(marker_color=theme["cyon"])
    fig.update_layout(barmode="stack", yaxis={"categoryorder": "total ascending"})
    # fig.update_traces(
    #     marker_color=theme["primary"],
    #     marker_line_color=theme["primary"],
    #     marker_line_width=1.5,
    #     opacity=0.6,
    # )
    return fig


@application.callback(
    Output("dependence_plot", "figure"),
    Input("shap_values_store", "data"),
)
@cache.memoize(timeout=TIMEOUT)
def draw_shap_dependence_graph(shap_values):
    # shap_values = pd.json_normalize(shap_values)
    shap_values = get_shap_values()
    df = dataframe()
    # print(shap_values[:, 12])
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Scatter(
            x=df["FW_Feed_B"],
            y=shap_values[:, 11],
            name="FW_Feed_B",
            mode="markers",
            marker=dict(size=3),
        ),  # replace with your own data source
        secondary_y=False,
    )

    # Add traces
    fig.add_trace(
        go.Scatter(
            x=df["FW_Feed_B"],
            y=df["Dig_A_Temp"],
            name="Dig_A_Temp",  # replace with your own data source
            mode="markers",
            marker=dict(size=3),
        ),
        secondary_y=True,
    )

    # Add figure title
    fig.update_layout(title_text="Dependence Plot")
    fig.update_layout(template="plotly_dark")
    # # Set x-axis title
    # fig.update_xaxes(title_text="xaxis title")

    # Set y-axes titles
    fig.update_yaxes(title_text="SHAP Values of FW_Feed_B", secondary_y=False)

    fig.update_yaxes(title_text="Dig_A_Temp", secondary_y=True)
    fig.update_xaxes(
        title_text="FW_Feed_B",
    )

    return fig


@application.callback(
    Output("predict_value", "value"),
    Input("predict_dropdown", "value"),
    prevent_initial_call=True,
)
def update_predict_value(data_idx):
    prediction = verify(int(data_idx) - 1)
    return prediction
