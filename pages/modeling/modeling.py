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

table_header = [html.Thead(html.Tr([html.Th("First Name"), html.Th("Last Name")]))]
row1 = html.Tr([html.Td("Arthur"), html.Td("Dent")])
row2 = html.Tr([html.Td("Ford"), html.Td("Prefect")])
row3 = html.Tr([html.Td("Zaphod"), html.Td("Beeblebrox")])
row4 = html.Tr([html.Td("Trillian"), html.Td("Astra")])

table_body = [html.Tbody([row1, row2, row3, row4])]
# Layout


layout = html.Div(
    [
        dbc.Row(
            dbc.Col(
                html.H6("Biogas 생산량 예측"),
            )
        ),
        html.Hr(),
        dbc.Row(
            [
                dbc.CardGroup(
                    [
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H6("대표 알고리즘", className="card-title"),
                                    html.H6(
                                        "XGBoost",
                                        style={
                                            "textAlign": "center",
                                            "fontWeight": "bold",
                                        },
                                    ),
                                ]
                            ),
                            className="mt-3",
                        ),
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H6("모델 성능", className="card-title"),
                                    dbc.Row(
                                        get_modeling_assessment(),
                                        id="model_assessment",
                                        justify="center",
                                    ),
                                ]
                            ),
                            className="mt-3",
                        ),
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H6("모델 예측값", className="card-title"),
                                    daq.LEDDisplay(
                                        id="predict_value",
                                        label="Predict Value",
                                        labelPosition="bottom",
                                        color="#fcdc64",
                                        size=18,
                                        value=0,
                                    ),
                                ]
                            ),
                            className="mt-3",
                        ),
                    ]
                )
            ]
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
                        # style={"height": "50vh", "width": "70vh"},
                        style={"height": "45vh"},
                    ),
                    width=12,
                ),
                html.Br(),
                dbc.Col(
                    dbc.Table(
                        # using the same table as in the above example
                        table_header + table_body,
                        bordered=True,
                        dark=True,
                        hover=True,
                        responsive=True,
                        striped=True,
                    )
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
