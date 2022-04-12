import dash
import pandas as pd
import numpy as np

from collections import OrderedDict

from dash import Dash, dcc, html, Input, Output, callback, dash_table
from dash.dependencies import Output, ALL, State, MATCH, ALLSMALLER

import plotly.express as px
import dash_bootstrap_components as dbc  # pip3 install dash-bootstrap-components
import dash_daq as daq
from utils.constants import theme, blank_figure


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
                        # dbc.Card(
                        #     dbc.CardBody(
                        #         [
                        #             html.H6("대표 알고리즘", className="card-title"),
                        #             html.H6(
                        #                 "XGBoost",
                        #                 style={
                        #                     "textAlign": "center",
                        #                     "fontWeight": "bold",
                        #                 },
                        #             ),
                        #         ]
                        #     ),
                        #     className="mt-3",
                        # ),
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H6("모델 예측값", className="card-title"),
                                    daq.LEDDisplay(
                                        id="predict_value",
                                        label="Predict Value",
                                        labelPosition="bottom",
                                        color="#fcdc64",
                                        size=32,
                                    ),
                                ]
                            ),
                            className="mt-3",
                        ),
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H6("모델 성능", className="card-title"),
                                    dcc.Loading(
                                        children=[
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        daq.LEDDisplay(
                                                            id=i,
                                                            label=i,
                                                            labelPosition="bottom",
                                                            value=0,
                                                            color="#fcdc64",
                                                            size=18,
                                                        ),
                                                        # width=3,
                                                    )
                                                    for i in [
                                                        "MAPE_Value",
                                                        # "R_square_XGB",
                                                        "RMSE",
                                                    ]
                                                ]
                                            ),
                                        ],
                                        type="circle",
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
                dbc.Col(
                    dcc.Graph(
                        id="line_graph",
                        # style={"height": "50vh", "width": "70vh"},
                        figure=blank_figure(),
                        style={"height": "35vh"},
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
                    id="model_store",
                    storage_type="session",
                ),
                dcc.Store(
                    id="modeling_result_store",
                    storage_type="session",
                ),
                dcc.Store(
                    id="modeling_assessment_store",
                    storage_type="session",
                ),
                dcc.Store(
                    id="predict_store",
                    storage_type="session",
                ),
                dcc.Store(
                    id="actual_predict_store",
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
