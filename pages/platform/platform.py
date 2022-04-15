import dash_core_components as dcc
import dash_html_components as html

import dash_bootstrap_components as dbc
import dash_daq as daq

from utils.constants import monitored_tags, theme, blank_figure, all_tags
import plotly.express as px

layout = dcc.Loading(
    children=[
        html.H5("다수 지점 통합 모니터링"),
        html.Hr(),
        dbc.Row(
            dbc.Col(
                dcc.Graph(
                    id="bio_graph", figure=blank_figure(), style={"height": "28vh"}
                )
            ),
            style={"marginBottom": "2rem"},
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(
                        id="proc_rate_a",
                        figure=blank_figure(),
                        style={"height": "25vh"},
                    ),
                    width=6,
                ),
                dbc.Col(
                    dcc.Graph(
                        id="proc_rate_b",
                        figure=blank_figure(),
                        style={"height": "25vh"},
                    ),
                    width=6,
                ),
                html.Button(
                    "아무 역할 없지만 데이터 불러오기 위해서 있어야 하는 버튼",
                    id="btn_load_data",
                    style={"display": "none"},
                ),
            ],
            style={"marginBottom": "2rem"},
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                # html.P(
                                #     "Proc_rate 평균값 (Site: " + str(i) + ")",
                                #     # className="card-title",
                                #     style={"textAlign": "center", "fontSize": "1.7rem"},
                                # ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            daq.Gauge(
                                                # id={"type": "gauge", "index": i},
                                                id="gauge" + str(i) + str(idx_j),
                                                value=6,
                                                size=150,
                                                showCurrentValue=True,
                                                label=col + "  평균값 (최근 일주일)",
                                                max=1,
                                                min=0,
                                                style={
                                                    "marginTop": "1rem",
                                                },
                                            ),
                                            width=6,
                                        )
                                        for idx_j, col in enumerate(
                                            ["Proc_rate_A", "Proc_rate_B"]
                                        )
                                    ]
                                ),
                            ],
                        ),
                        style={
                            # "marginTop": "1rem",
                            "backgroundColor": "#0f2331",
                        },
                    )
                )
                for i in range(2)
            ]
        ),
        # """DATA STORE"""
        html.Div(
            children=[
                dcc.Store(
                    id="first_bio_store",
                    storage_type="session",
                ),
                dcc.Store(
                    id="second_bio_store",
                    storage_type="session",
                ),
            ]
        ),
    ],
    type="circle",
)
