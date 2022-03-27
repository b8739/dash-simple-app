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
                html.H5("Biogas 생산량 예측"),
            )
        ),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H6(
                                            "모델 예측값 (XGBoost)",
                                        ),
                                        daq.LEDDisplay(
                                            id="predict_value",
                                            value=28485,
                                            label="Predict Value",
                                            labelPosition="bottom",
                                            color="#fcdc64",
                                            size=40,
                                        ),
                                    ]
                                ),
                                className="mt-3",
                                style={"height": "50%"},
                            ),
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H6(
                                            "모델 성능",
                                        ),
                                        dcc.Loading(
                                            children=[
                                                # Loading 화면 띄우려고 끼워놓은것
                                                # html.Div(
                                                #     children=[html.P("hi")],
                                                #     id="loading-output-1",
                                                # ),
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            daq.LEDDisplay(
                                                                id=i,
                                                                label=i,
                                                                labelPosition="bottom",
                                                                value=0,
                                                                # color="#fcdc64",
                                                                color="#ffe9a3",
                                                                size=24,
                                                            ),
                                                            # width=3,
                                                        )
                                                        for i in [
                                                            "MAPE_Value",
                                                            # "R_square_XGB",
                                                            "RMSE",
                                                        ]
                                                    ],
                                                ),
                                            ],
                                            type="circle",
                                        ),
                                    ]
                                ),
                                style={"height": "50%"},
                            ),
                        ],
                        style={"height": "45vh"},
                    ),
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
                    # dbc.Card(
                    #     dbc.CardBody(
                    #         [
                    #             html.H6(
                    #                 "모델 성능",
                    #             ),
                    #             dcc.Loading(
                    #                 children=[
                    #                     # Loading 화면 띄우려고 끼워놓은것
                    #                     # html.Div(
                    #                     #     children=[html.P("hi")],
                    #                     #     id="loading-output-1",
                    #                     # ),
                    #                     dbc.Row(
                    #                         [
                    #                             dbc.Col(
                    #                                 daq.LEDDisplay(
                    #                                     id=i,
                    #                                     label=i,
                    #                                     labelPosition="bottom",
                    #                                     value=0,
                    #                                     color="#fcdc64",
                    #                                     size=18,
                    #                                 ),
                    #                                 # width=3,
                    #                             )
                    #                             for i in [
                    #                                 "MAPE_Value",
                    #                                 # "R_square_XGB",
                    #                                 "RMSE",
                    #                             ]
                    #                         ]
                    #                     ),
                    #                 ],
                    #                 type="circle",
                    #             ),
                    #         ]
                    #     ),
                    #     className="mt-3",
                    # ),
                    width=6,
                ),
                dbc.Col(
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
                                figure=blank_figure(),
                                style={"height": "45vh", "marginTop": "10px"},
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
            ]
        ),
        html.Br(),
        html.Br(),
        dcc.Loading(
            children=[
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
