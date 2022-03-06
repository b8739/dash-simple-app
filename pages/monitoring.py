import dash

# Code from: https://github.com/plotly/dash-labs/tree/main/docs/demos/multi_page_example1
# dash.register_page(__name__, path="/")
dash.register_page(__name__, path="/monitoring", name="Monitoring", order=2)

from dash import Dash, dcc, html, Input, Output, State, callback
from dash_extensions.enrich import Dash, Output, Input, Trigger, ServersideOutput
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px
import time
import pandas as pd
import numpy as np
import dash_daq as daq
from dash.dependencies import Input, Output, ALL, State, MATCH, ALLSMALLER
import sys

sys.path.append("./logic")
import prepare_data

theme = {
    "dark": True,
    "detail": "#007439",
    "primary": "#00EA64",
    "secondary": "#6E6E6E",
}

displayNone = {
    "display": "none",
}
# Iris
# df_01 = px.data.iris()[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
# df_01 = pd.concat([df_01 + np.random.randn(*df_01.shape) * 0.1 for i in range(1000)])

"""Data Preprocess"""
df_00 = pd.read_csv("ketep_biogas_data_20220210.csv")
df_01 = prepare_data.preprocess(df_00)
df_veri = prepare_data.extract_veri(df_00)
train_Xn, train_y, test_Xn, test_y, X_test = prepare_data.split_dataset(df_01)


"""Frontend"""
monitored_tags = ["PS_feed_A", "FW_Feed_A", "Dig_A_Temp", "PS_incoming", "Dig_A_TS"]


def isNormal(idx):
    if idx == 1:
        return {"state": "Abnormal", "color": "red"}
    else:
        return {"state": "Normal", "color": theme["primary"]}


def plotMonitoringGraphs():
    return [
        dbc.Col(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Span(
                                    isNormal(idx)["state"],
                                    style={
                                        "marginRight": 15,
                                        "textAlign": "center",
                                    },
                                ),
                                daq.Indicator(
                                    id={"type": "indicator", "index": idx},
                                    color=isNormal(idx)["color"],
                                    value=isNormal(idx)["state"],
                                    className="dark-theme-control",
                                    style={"display": "inline-block"},
                                ),
                                dbc.Tooltip(
                                    "정상 작동중입니다.",
                                    target="indicator" + str(idx),
                                ),
                            ],
                            style={
                                "paddingLeft": 12,
                                "paddingTop": 8,
                            },
                        ),
                        dbc.Col(
                            dcc.Dropdown(
                                id={"type": "tagDropdown", "index": idx},
                                options=[
                                    {"label": c, "value": c} for c in df_01.columns
                                ],
                                placeholder="Select Tag",
                                value=monitored_tags[idx],
                                clearable=False,
                                persistence=True,
                                style={
                                    "backgroundColor": "rgb(48, 48, 48)",
                                },
                            ),
                            width=3,
                        ),
                    ]
                ),
                html.Br(),
                dcc.Graph(
                    id={"type": "monitoring-graph", "index": idx},
                    style={"height": "30vh"},
                ),
                html.Br(),
            ],
            width=6,
        )
        for idx in range(4)
    ]


def makeBioGraph():
    tag = df_01.columns[3]
    fig = px.scatter(df_01, y=tag, title=None, template="plotly_dark")
    fig.update_traces(
        mode="markers", marker=dict(size=1, line=dict(width=2, color="#f4d44d"))
    ),
    # fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(
        title={
            "text": tag,
            "xref": "paper",
            "yref": "paper",
            "x": 0.5,
            # "y": 0.5,
        }
    )
    return fig


tabs_styles = {"height": "44px", "align-items": "center"}
tab_style = {
    "fontWeight": "bold",
    "border-radius": "15px",
    "backgroundColor": "#F2F2F2",
    "padding": "6px",
    "backgroundColor": "#32383e",
    # "box-shadow": "4px 4px 4px 4px lightgrey",
}

tab_selected_style = {
    "borderTop": "1px solid #d6d6d6",
    "borderBottom": "1px solid #d6d6d6",
    "backgroundColor": "black",
    "color": "white",
    "padding": "6px",
    "border-radius": "15px",
}
biogasProduct = [
    dbc.Row(
        [
            dbc.Col(
                [
                    html.Span(
                        "Normal",
                        style={
                            "marginRight": 15,
                            "textAlign": "center",
                        },
                    ),
                    daq.Indicator(
                        id="indicator",
                        color=theme["primary"],
                        value=True,
                        className="dark-theme-control",
                        style={"display": "inline-block"},
                    ),
                    dbc.Tooltip(
                        "정상 작동중입니다.",
                        target="indicator",
                    ),
                ]
            ),
            dbc.Col(
                dcc.Dropdown(
                    options=[{"label": c, "value": c} for c in df_01.columns],
                    placeholder="Select Tag",
                    value=df_01.columns[3],
                    clearable=False,
                    persistence=True,
                    style={
                        "backgroundColor": "rgb(48, 48, 48)",
                    },
                ),
                width=3,
            ),
        ]
    ),
    dbc.Row(
        dcc.Graph(
            id="bioproduct-graph",
            figure=makeBioGraph(),
            style={"height": "35vh"},
        ),
    ),
]

tab1_content = dbc.Card(
    dbc.CardBody([dbc.Row(plotMonitoringGraphs())]),
    className="mt-3",
)

tab2_content = dbc.Card(
    dbc.CardBody([html.Div(biogasProduct)]),
    className="mt-3",
)


tabs = dbc.Tabs(
    [
        dbc.Tab(tab1_content, label="이상 감지", id="tab-1"),
        dbc.Tab(tab2_content, label="성능 감시", id="tab-2"),
    ],
    id="tabs-styled-with-props",
)
layout = html.Div(
    dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Row(
                        dbc.Col(
                            dcc.Loading(
                                children=[
                                    tabs,
                                ],
                                type="circle",
                            ),
                        )
                    ),
                ],
                width=12,
            ),
        ],
        style={"position": "relative"},
    ),
)
#


@callback(
    Output({"type": "monitoring-graph", "index": MATCH}, "figure"),
    Input({"type": "tagDropdown", "index": MATCH}, "value"),
    Input({"type": "indicator", "index": MATCH}, "value"),
    # prevent_initial_call=True,
    memoize=True,
)
def changeTag(tag, indicator):
    " " " Plotly Graph 생성 " " "

    if not tag:
        tag = df_01.columns[3]
    fig = px.scatter(df_01, y=tag, title=None, template="plotly_dark")
    fig.update_traces(
        mode="markers", marker=dict(size=1, line=dict(width=2, color="#f4d44d"))
    ),
    fig.update_yaxes(rangemode="normal")
    fig.update_yaxes(range=[df_01[tag].min() * (0.8), df_01[tag].max() * (1.2)])
    # fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(
        title={
            "text": tag,
            "xref": "paper",
            "yref": "paper",
            "x": 0.5,
            # "y": 0.5,
        },
    )

    " " " Quantile 표시 " " "

    q_position = df_01[tag].min() * 1.1

    for q in ["Q1", "Q2", "Q3", "Q4"]:
        q_position += df_01[tag].max() / 4

        fig.add_hline(
            y=q_position,
            line_dash="dot",
            annotation_text=q,
            annotation_position="right",
            opacity=0.9,
        )

    " " " Average 표시 " " "

    fig.add_annotation(
        text="Avg: 24",
        align="left",
        showarrow=False,
        xref="paper",
        yref="paper",
        x=1.1,
        y=1.1,
        bordercolor="black",
        borderwidth=1,
    )

    " " " 이상 구역 Rect 표시 " " "

    if indicator == "Abnormal":
        fig.add_shape(
            type="rect",
            xref="x domain",
            yref="y domain",
            x0=0.65,
            x1=0.7,
            y0=0.5,
            y1=0.7,
            line=dict(color="red", width=2),
        )

    return fig
