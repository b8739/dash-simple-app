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

""" GET DATAFRAME """


@application.callback(
    ServersideOutput("first_bio_store", "data"),
    Input("btn_load_data", "n_clicks"),
)
@cache.memoize(timeout=TIMEOUT)
def save_first_data(n_clicks):
    print(sys.path)
    df = pd.read_excel("pages/platform/ketep_biogas_data_20220411_02.xlsx")

    return df


@application.callback(
    ServersideOutput("second_bio_store", "data"),
    Input("btn_load_data", "n_clicks"),
)
@cache.memoize(timeout=TIMEOUT)
def save_second_data(n_clicks):
    df = pd.read_excel("pages/platform/ketep_biogas_data_20220411_02.xlsx")
    return df


""" DRAW GRAPH 1 """


@application.callback(
    Output("bio_graph", "figure"),
    Input("first_bio_store", "data"),
    Input("second_bio_store", "data"),
)
@cache.memoize(timeout=TIMEOUT)
def update_biggraph(first_df, second_df):
    trace_list = [
        go.Scatter(
            name="A",
            x=first_df["Date"],
            y=first_df["Biogas_prod"],
            visible=True,
            mode="lines+markers",
            line={"width": 1},
            line_color="#FF8303",
            marker=dict(size=2),
        ),
        go.Scatter(
            name="B",
            x=second_df["Date"],
            y=second_df["Biogas_prod"],
            visible=True,
            mode="lines+markers",
            # mode="markers",
            line={"width": 1},
            line_color="#03ac13",
            marker=dict(size=1),
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
    )
    return fig


""" DRAW GRAPH 2 """


@application.callback(
    Output("proc_rate_a", "figure"),
    Input("first_bio_store", "data"),
)
@cache.memoize(timeout=TIMEOUT)
def update_rate_graph(df):
    fig = px.line(df, x="Date", y="Proc_rate_A", markers=True)
    fig.update_traces(
        mode="markers+lines",
        marker=dict(size=1, line=dict(width=0.5, color="#f4d44d")),
        line=dict(color="#f4d44d", width=0.5),
    ),
    fig.update_layout(template="plotly_dark")
    fig.update_layout(
        yaxis_title=None,
        xaxis_title="Date",
        # updatemenus와 곂치기 때문에 none
        title={
            "text": "Proc_rate_A",
            "xref": "paper",
            "yref": "paper",
            "x": 0.5,
            # "pad": {"b": 50},
            "xanchor": "center",
            "yanchor": "middle",
            "font": {"size": 15},
            # "y": 0.5,
        },
        margin=dict(l=35, r=35, t=50, b=50, pad=20),
        # pad=dict(l=100, r=100, t=30, b=100),
    )
    return fig


@application.callback(
    Output("proc_rate_b", "figure"),
    Input("second_bio_store", "data"),
)
@cache.memoize(timeout=TIMEOUT)
def update_rate_graph(df):
    fig = px.line(df, x="Date", y="Proc_rate_B", markers=True)
    fig.update_traces(
        mode="markers+lines",
        marker=dict(size=1, line=dict(width=0.5, color="#f4d44d")),
        line=dict(color="#f4d44d", width=0.5),
    ),
    fig.update_layout(template="plotly_dark")
    fig.update_layout(
        yaxis_title=None,
        xaxis_title="Date",
        # updatemenus와 곂치기 때문에 none
        title={
            "text": "Proc_rate_B",
            "xref": "paper",
            "yref": "paper",
            "x": 0.5,
            # "pad": {"b": 50},
            "xanchor": "center",
            "yanchor": "middle",
            "font": {"size": 15},
            # "y": 0.5,
        },
        margin=dict(l=35, r=35, t=50, b=50, pad=20),
        # pad=dict(l=100, r=100, t=30, b=100),
    )
    return fig


# Proc_rate_A
# Output({"type": "indicator", "index": MATCH}, "color"),
