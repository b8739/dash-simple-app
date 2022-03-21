import dash
import pandas as pd
import numpy as np

from collections import OrderedDict

from dash import Dash, dcc, html, Input, Output, callback, dash_table
from dash.dependencies import Output, ALL, State, MATCH, ALLSMALLER

import plotly.express as px
import dash_bootstrap_components as dbc  # pip3 install dash-bootstrap-components
import dash_daq as daq
from pages.modeling.modeling_data import get_modeling_assessment
from utils.constants import theme


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
            dbc.Col(
                html.H6("Biogas 생산량 예측"),
            )
        ),
        # dbc.Row(
        #     [
        #         dbc.Col(
        #             html.Div(
        #                 # card,
        #                 style={
        #                     "display": "flex",
        #                     "alignItems": "center",
        #                     "justifyContent": "space-around",
        #                 },
        #             )
        #         )
        #         for _ in range(4)
        #     ],
        # ),
        dbc.Row(get_modeling_assessment(), id="model_assessment"),
        dbc.Row(
            dbc.Col(
                daq.LEDDisplay(
                    id="predict_value",
                    label="Predict Value",
                    labelPosition="bottom",
                    color="#fcdc64",
                    size=24,
                    value=0,
                ),
                width=3,
            )
        ),
        html.Br(),
        dbc.Row(
            [
                html.Button(
                    "아무 역할 없지만 데이터 불러오기 위해서 있어야 하는 버튼",
                    id="btn_3",
                    style={"display": "none"},
                ),
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
                dcc.Store(
                    id="predict_store",
                    storage_type="session",
                ),
            ],
            # fullscreen=True,
            type="circle",
        ),
    ]
)
