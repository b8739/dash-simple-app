import dash
import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output, callback, dash_table
from dash.dependencies import Output, ALL, State, MATCH, ALLSMALLER
import dash_bootstrap_components as dbc  # pip3 install dash-bootstrap-components
import dash_daq as daq
from utils.constants import theme, blank_figure


layout = html.Div(
    [
        dbc.Row(
            dbc.Col(
                [
                    html.H5("Biogas 안정적 운전값 제시"),
                ]
            )
        ),
        html.Hr(),
        dbc.Card(
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Graph(
                                    id="bar_graph",
                                    figure=blank_figure(),
                                    style={"height": "40vh"},
                                ),
                            ),
                        ]
                    ),
                ]
            ),
            className="mt-3",
        ),
        dbc.Card(
            dbc.CardBody(
                [
                    html.Div(
                        id="dependence_container",
                        children=[],
                    ),
                ]
            ),
            className="mt-3",
        ),
        dcc.Loading(
            [
                html.Button(
                    "아무 역할 없지만 데이터 불러오기 위해서 있어야 하는 버튼",
                    id="btn_4",
                    style={"display": "none"},
                ),
                dcc.Store(
                    id="shap_values_store",
                    storage_type="session",
                ),
                dcc.Store(
                    id="shap_importance_store",
                    storage_type="session",
                ),
            ],
            # fullscreen=True,
            type="circle",
        ),
    ]
)
