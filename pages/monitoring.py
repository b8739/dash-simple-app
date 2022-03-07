import dash

# Code from: https://github.com/plotly/dash-labs/tree/main/docs/demos/multi_page_example1
# dash.register_page(__name__, path="/")

from dash import Dash, dcc, html, Output, State, callback
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px
import time
import pandas as pd
import numpy as np
import dash_daq as daq
from dash.dependencies import Output, ALL, State, MATCH, ALLSMALLER
import sys
from dash_extensions.enrich import Output, Input, Trigger, ServersideOutput

# from dataset import df
import prepare_data
from dash_extensions.enrich import (
    DashProxy,
    Input,
    Output,
    TriggerTransform,
    ServersideOutputTransform,
    ServersideOutput,
    Trigger,
)

# from dashMulti import df
app = DashProxy()

sys.path.append("./logic")

tabs = dbc.Tabs(
    [],
    id="tab_container",
)


app.layout = html.Div(
    dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Row(
                        dbc.Col(
                            dcc.Loading(
                                children=[
                                    html.Button("monitoring에서 불러오기", id="btn_3"),
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


theme = {
    "dark": True,
    "detail": "#007439",
    "primary": "#00EA64",
    "secondary": "#6E6E6E",
}

displayNone = {
    "display": "none",
}


"""Frontend"""
monitored_tags = ["PS_feed_A", "FW_Feed_A", "Dig_A_Temp", "PS_incoming", "Dig_A_TS"]


def isNormal(idx):
    if idx == 1:
        return {"state": "Abnormal", "color": "red"}
    else:
        return {"state": "Normal", "color": theme["primary"]}


<<<<<<< HEAD
def plotMonitoringGraphs(graph_type, graph_number):
=======
def plotMonitoringGraphs(df):
>>>>>>> dataCaching
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
                                options=[{"label": c, "value": c} for c in df.columns],
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
                    id={"type": graph_type, "index": idx},
                    style={"height": "30vh"},
                ),
                html.Br(),
            ],
            width=6,
        )
        for idx in range(graph_number)
    ]


def makeBioGraph(df):
    tag = df.columns[3]
    fig = px.scatter(df, y=tag, title=None, template="plotly_dark")
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


def makeBiogasProductGraph(df):
    return [
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
                        options=[{"label": c, "value": c} for c in df.columns],
                        placeholder="Select Tag",
                        value=df.columns[3],
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
                figure=makeBioGraph(df),
                style={"height": "35vh"},
            ),
        ),
    ]


@app.callback(
    Output("tab_container", "children"),
    Input("btn_3", "n_clicks"),
    State("preprocessed_store", "data"),
    State("tab_container", "children"),
    # prevent_initial_call=True,
)
def create_layout(btn_3, preprocessed_store, tab_container):
    print(preprocessed_store)
    tab1_content = dbc.Tab(
        dbc.Card(
            dbc.CardBody([dbc.Row(plotMonitoringGraphs(preprocessed_store))]),
            className="mt-3",
        ),
        label="이상 감지",
        id="tab-1",
    )

    tab2_content = dbc.Tab(
        dbc.Card(
            dbc.CardBody([html.Div(makeBiogasProductGraph(preprocessed_store))]),
            className="mt-3",
        ),
        label="성능 감시",
        id="tab-1",
    )
<<<<<<< HEAD

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


=======
    tab_container.append(tab1_content)
    tab_container.append(tab2_content)
    return tab_container


"""Dropdown에서 tag 클릭하면 data update하는 callback"""
# @app.callback(
#     Output({"type": "monitoring-graph", "index": MATCH}, "figure"),
#     Input({"type": "tagDropdown", "index": MATCH}, "value"),
#     Input({"type": "indicator", "index": MATCH}, "value"),
#     # prevent_initial_call=True,
#     memoize=True,
# )
# def changeTag(tag, indicator):
#     " " " Plotly Graph 생성 " " "

#     if not tag:
#         tag = df.columns[3]
#     fig = px.scatter(df, y=tag, title=None, template="plotly_dark")
#     fig.update_traces(
#         mode="markers", marker=dict(size=1, line=dict(width=2, color="#f4d44d"))
#     ),
#     fig.update_yaxes(rangemode="normal")
#     fig.update_yaxes(range=[df[tag].min() * (0.8), df[tag].max() * (1.2)])
#     # fig.update_xaxes(rangeslider_visible=True)
#     fig.update_layout(
#         title={
#             "text": tag,
#             "xref": "paper",
#             "yref": "paper",
#             "x": 0.5,
#             # "y": 0.5,
#         },
#     )

#     " " " Quantile 표시 " " "

#     q_position = df[tag].min() * 1.1

#     for q in ["Q1", "Q2", "Q3", "Q4"]:
#         q_position += df[tag].max() / 4

#         fig.add_hline(
#             y=q_position,
#             line_dash="dot",
#             annotation_text=q,
#             annotation_position="right",
#             opacity=0.9,
#         )

#     " " " Average 표시 " " "

#     fig.add_annotation(
#         text="Avg: 24",
#         align="left",
#         showarrow=False,
#         xref="paper",
#         yref="paper",
#         x=1.1,
#         y=1.1,
#         bordercolor="black",
#         borderwidth=1,
#     )

#     " " " 이상 구역 Rect 표시 " " "

#     if indicator == "Abnormal":
#         fig.add_shape(
#             type="rect",
#             xref="x domain",
#             yref="y domain",
#             x0=0.65,
#             x1=0.7,
#             y0=0.5,
#             y1=0.7,
#             line=dict(color="red", width=2),
#         )

#     return fig
>>>>>>> dataCaching
if __name__ == "__main__":
    app.run_server()
