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


# Output({"type": "indicator", "index": MATCH}, "color"),
