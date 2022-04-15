from app import application
from app import cache
from utils.constants import TIMEOUT
from dash.dependencies import Output, Input, State, ALL, MATCH, ALLSMALLER
from dash_extensions.enrich import Dash, Trigger, ServersideOutput
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import sys

sys.path.append("./pages/platform")


@application.callback(
    ServersideOutput("first_bio_store", "data"),
    Input("btn_load_data", "n_clicks"),
)
@cache.memoize(timeout=TIMEOUT)
def save_first_data(n_clicks):
    df = pd.read_excel("pages/platform/ketep_biogas_data_20220411.xlsx")

    return df


@application.callback(
    ServersideOutput("second_bio_store", "data"),
    Input("btn_load_data", "n_clicks"),
)
@cache.memoize(timeout=TIMEOUT)
def save_second_data(n_clicks):
    df = pd.read_excel("pages/platform/ketep_biogas_data_20220411_02.xlsx")
    return df


"""GAUGE A"""


def create_callback(column):
    def get_rate_mean(first_bio_store):
        print(first_bio_store.iloc[-7:][column].mean())
        return first_bio_store.iloc[-7:][column].mean()

    return get_rate_mean


for idx, column in enumerate(["Proc_rate_A", "Proc_rate_B"]):
    application.callback(
        Output("gauge0" + str(idx), "value"),
        Input("first_bio_store", "data"),
    )(create_callback(column))


"""GAUGE B"""


def create_callback(column):
    def get_rate_mean(first_bio_store):
        print(first_bio_store.iloc[-7:][column].mean())
        return first_bio_store.iloc[-7:][column].mean()

    return get_rate_mean


for idx, column in enumerate(["Proc_rate_A", "Proc_rate_B"]):
    application.callback(
        Output("gauge1" + str(idx), "value"),
        Input("second_bio_store", "data"),
    )(create_callback(column))

""" GET DATAFRAME """

""" DRAW GRAPH 1 """


@application.callback(
    Output("bio_graph", "figure"),
    Input("first_bio_store", "data"),
    Input("second_bio_store", "data"),
)
@cache.memoize(timeout=TIMEOUT)
def update_biggraph(first_df, second_df):
    first_df = first_df.tail(100)
    second_df = second_df.tail(100)
    trace_list = [
        go.Scatter(
            name="A",
            x=first_df["Date"],
            y=first_df["Biogas_prod"],
            visible=True,
            mode="lines+markers",
            line={"width": 1},
            line_color="#70e3a5",
            marker=dict(size=4),
        ),
        go.Scatter(
            name="B",
            x=second_df["Date"],
            y=second_df["Biogas_prod"],
            visible=True,
            mode="lines+markers",
            # mode="markers",
            line={"width": 1},
            line_color="#e3de70",
            marker=dict(size=4),
        ),
    ]
    fig = go.Figure(data=trace_list)
    fig.update_layout(template="plotly_dark")

    fig.update_layout(
        title={
            "text": "바이오가스 생산량",
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            # "font": {"size": 10},
        },
        margin=dict(l=70, r=70, t=70, b=80),
        paper_bgcolor="#041929",
        plot_bgcolor="#041929",
    )
    fig.update_xaxes(title="Date", showgrid=True, gridcolor="#696969")
    fig.update_yaxes(showgrid=True, gridcolor="#696969")
    return fig


""" DRAW GRAPH 2 """


@application.callback(
    Output("proc_rate_a", "figure"),
    Input("first_bio_store", "data"),
    Input("second_bio_store", "data"),
)
@cache.memoize(timeout=TIMEOUT)
def update_rate_graph(first_df, second_df):
    first_df = first_df.tail(100)
    second_df = second_df.tail(100)
    trace_list = [
        go.Scatter(
            name="A",
            x=first_df["Date"],
            y=first_df["Proc_rate_A"],
            visible=True,
            mode="lines+markers",
            line={"width": 0.9},
            line_color="#65f4e3",
            marker=dict(size=3),
        ),
        go.Scatter(
            name="B",
            x=second_df["Date"],
            y=second_df["Proc_rate_A"],
            visible=True,
            mode="lines+markers",
            # mode="markers",
            line={"width": 0.9},
            line_color="#e88790",
            marker=dict(size=3),
        ),
    ]
    fig = go.Figure(data=trace_list)
    fig.update_layout(template="plotly_dark")

    fig.update_layout(
        title={
            "text": "Proc_rate_A",
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        margin=dict(l=70, r=70, t=60, b=60),
        paper_bgcolor="#041929",
        plot_bgcolor="#041929",
    )
    fig.update_xaxes(title="Date", showgrid=True, gridcolor="#696969")
    fig.update_yaxes(showgrid=True, gridcolor="#696969")
    return fig


@application.callback(
    Output("proc_rate_b", "figure"),
    Input("first_bio_store", "data"),
    Input("second_bio_store", "data"),
)
@cache.memoize(timeout=TIMEOUT)
def update_rate_graph(first_df, second_df):
    first_df = first_df.tail(100)
    second_df = second_df.tail(100)
    trace_list = [
        go.Scatter(
            name="A",
            x=first_df["Date"],
            y=first_df["Proc_rate_B"],
            visible=True,
            mode="lines+markers",
            line={"width": 0.9},
            line_color="#65f4e3",
            marker=dict(size=3),
        ),
        go.Scatter(
            name="B",
            x=second_df["Date"],
            y=second_df["Proc_rate_B"],
            visible=True,
            mode="lines+markers",
            # mode="markers",
            line={"width": 0.9},
            line_color="#e88790",
            marker=dict(size=3),
        ),
    ]
    fig = go.Figure(data=trace_list)
    fig.update_layout(template="plotly_dark")

    fig.update_layout(
        title={
            "text": "Proc_rate_B",
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        margin=dict(l=70, r=70, t=60, b=60),
        paper_bgcolor="#041929",
        plot_bgcolor="#041929",
    )
    fig.update_xaxes(title="Date", showgrid=True, gridcolor="#696969")
    fig.update_yaxes(showgrid=True, gridcolor="#696969")
    return fig


# Proc_rate_A
# Output({"type": "indicator", "index": MATCH}, "color"),
