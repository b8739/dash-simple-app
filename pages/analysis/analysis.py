import dash
import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output, callback, dash_table
from dash.dependencies import Output, ALL, State, MATCH, ALLSMALLER
import dash_bootstrap_components as dbc  # pip3 install dash-bootstrap-components
import dash_daq as daq
from utils.constants import theme, blank_figure

table_header = [html.Thead(html.Tr([html.Th("주요 변수"), html.Th("영향도")]))]
table_body = [html.Tbody()]

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
                                dbc.Table(
                                    # using the same table as in the above example
                                    table_header + table_body,
                                    id="influence_table",
                                    bordered=True,
                                    dark=True,
                                    hover=True,
                                    responsive=True,
                                    striped=True,
                                    style={"height": "100%"},
                                    size="sm",
                                )
                                # dash_table.DataTable(
                                #     id="influence_table",
                                #     columns=[
                                #         {"id": c, "name": c} for c in ["주요 변수", "영향도"]
                                #     ],
                                #     style_header={
                                #         "backgroundColor": "rgb(30, 30, 30)",
                                #         "color": "white",
                                #     },
                                #     style_data={
                                #         "backgroundColor": "rgb(50, 50, 50)",
                                #         "color": "white",
                                #     },
                                #     style_table={
                                #         "height": "30vh",
                                #     },
                                # )
                                ,
                                style={"height": "32vh"},
                            ),
                            dbc.Col(
                                dcc.Graph(
                                    id="bar_graph",
                                    figure=blank_figure(),
                                    style={"height": "100%"},
                                ),
                                style={"height": "32vh"},
                            ),
                        ],
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
