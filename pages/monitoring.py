import dash

# Code from: https://github.com/plotly/dash-labs/tree/main/docs/demos/multi_page_example1
# dash.register_page(__name__, path="/")
dash.register_page(__name__, path="/", name="공정 운전 변수 이상 감지", order=1)

from dash import Dash, dcc, html, Input, Output, State, callback
from dash_extensions.enrich import Dash, Output, Input, Trigger, ServersideOutput
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px
import time
import pandas as pd
import numpy as np

df = px.data.iris()[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
df = pd.concat([df + np.random.randn(*df.shape) * 0.1 for i in range(1000)])
print(len(df))

sampleCols = ["소화조 온도", "소화조 pH", "교반기 운전값"]


layout = html.Div(
    [
        html.P(id="cacheValue"),
        html.P(id="cacheMonitoredValue"),
        dcc.Store(id="monitored_tags", storage_type="memory"),
        # dcc.Store(id="test", storage_type="memory"),
        # CACHING TEST
        dcc.Store(id="cachedData", storage_type="session"),
        # dbc.Button(
        #     "query",
        #     id="query",
        # ),
        dbc.Row(
            html.H4("공정 변수 모니터링:"),
        ),
        # dbc.Row(dbc.Col(dbc.Button("모니터링 변수 업데이트", id="add_btn"), width=6)),
        dbc.Row(
            [
                dbc.Label("공정 변수", style={"color": "grey"}),
            ]
        ),
        dbc.Row(
            dbc.Col(
                dcc.Dropdown(
                    id="selected_tags",
                    options=[{"label": col, "value": col} for col in df.columns],
                    multi=True,
                    placeholder="공정 변수",
                    persistence=True,
                    persistence_type="session",
                ),
                width=4,
            ),
        ),
        dbc.Row(
            dbc.Col(
                dbc.Button(
                    "모니터링 변수 업데이트",
                    id="add_btn",
                    size="sm",
                    # outline=True,
                    # color="primary",
                ),
                width=6,
            ),
        ),
        dbc.Row(
            dbc.Col(
                dcc.Loading(
                    children=[
                        html.Div(id="container", children=[], style={"width": "100%"}),
                    ],
                    type="circle",
                ),
            )
        ),
    ],
)


# @callback(
#     Output("cachedData", "data"),
#     Input("query", "n_clicks"),
#     State("cachedData", "data"),
#     prevent_initial_callbacks=True,
# )
# def query_data(n_clicks, data):
#     if n_clicks is None:
#         raise PreventUpdate
#     print("query data")
#     time.sleep(0.2)
#     data = df.to_json(orient="records")
#     return len(data)


@callback(
    Output("cacheValue", "children"),
    Input("cachedData", "modified_timestamp"),
    State("cachedData", "data"),
)
def displayCacheValue(ts, data):
    if ts is None:
        raise PreventUpdate

    data = data or {}

    return data


@callback(
    ServersideOutput("monitored_tags", "data"),
    Input("add_btn", "n_clicks"),
    State("selected_tags", "value"),
    prevent_initial_callbacks=True,
    memoize=True,
)
def addMonitoredVariable(n_clicks, newValue):

    # time.sleep(0.5)
    return newValue


# @callback(
#     Output("cacheMonitoredValue", "children"),
#     Input("monitored_tags", "modified_timestamp"),
#     State("monitored_tags", "data"),
# )
# def displayCacheMonitoredValue(ts, data):
#     if ts is None:
#         raise PreventUpdate

#     data = data or {}

#     return data


def makeNewGraph(tag):
    # template = ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]
    fig = px.scatter(df, y=tag, title=tag, template="plotly")
    fig.update_traces(mode="markers", marker_size=2)
    fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(
        title={
            "text": tag + " (150,000)",
            "xref": "paper",
            "yref": "paper",
            "x": 0.5,
            # "y": 0.5,
        }
    )
    newChild = dcc.Graph(
        id={"type": "monitoring-graph", "index": tag},
        figure=fig,
        style={
            "width": "45%",
            "display": "inline-block",
            "outline": "thin lightgrey solid",
            "padding": 10,
        },
    )
    return newChild


@callback(
    Output("container", "children"),
    Input("add_btn", "n_clicks"),
    State("selected_tags", "value"),
    State("monitored_tags", "data"),
    State("container", "children"),
    memoize=True,
)
def add_graph(
    n_clicks,
    selected_tags,
    monitored_tags,
    div_children,
):
    print("n_clicks: ", n_clicks)

    # Selected Tags로 지정된 tag를 그래프로 Rendering
    if selected_tags:
        for tag in selected_tags:
            # Exception: Do not Render if it is Already Being Monitored
            if monitored_tags is None or tag not in monitored_tags:
                div_children.append(makeNewGraph(tag))
    # 길이가 더 적다면 삭제된 것이니, 어떤 것이 삭제된건지 파악하고 삭제
    if monitored_tags:
        if len(monitored_tags) > len(selected_tags):
            deleteList = [
                graph
                for graph in div_children
                if graph["props"]["id"]["index"] not in selected_tags
            ]
            for graph in deleteList:
                div_children.remove(graph)

    return div_children
