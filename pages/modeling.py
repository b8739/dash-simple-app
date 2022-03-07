import dash
import pandas as pd
import numpy as np
import dash_daq as daq
import plotly.graph_objs as go

from collections import OrderedDict

# dash.register_page(__name__)

from dash import Dash, dcc, html, Input, Output, callback, dash_table
from dash.dependencies import Output, ALL, State, MATCH, ALLSMALLER

import plotly.express as px
import dash_bootstrap_components as dbc  # pip3 install dash-bootstrap-components
from dash_extensions.enrich import (
    DashProxy,
    Input,
    Output,
    TriggerTransform,
    ServersideOutputTransform,
    ServersideOutput,
    Trigger,
)
import sys

sys.path.append("../logic")


app = DashProxy()

# from dataset import df

# from dataset import result_df


predict = np.random.randn(500)
actual = np.random.randn(500)
type = ["SVR Pred.", "RF Pred.", "ENSEMBLE Pred.", "Actual MI"] * (500 // 4)


df = pd.read_csv("ketep_biogas_data_20220210.csv")
dff = {"key": df.columns[0:8], "val": [v for v in df.iloc[0, [0, 1, 2, 3, 4, 5, 6, 7]]]}

theme = {
    "dark": True,
    "detail": "#007439",
    "primary": "#00EA64",
    "secondary": "#6E6E6E",
}


def makeBarGraph():
    fig = px.bar(dff, x="val", y="key", orientation="h", template="plotly_dark")
    fig.update_traces(
        marker_color=theme["primary"],
        # marker_color=theme["primary"],
        marker_line_color=theme["primary"],
        marker_line_width=1.5,
        opacity=0.6,
    )
    return fig


def makeLineGraph():
    trace_list = [
        go.Scatter(
            name="Actual",
            y=df["Dig_Feed_A"],
            visible=True,
            mode="lines+markers",
            line={"width": 1},
            marker=dict(size=5),
        ),
        go.Scatter(
            name="Predictive",
            y=df["Dig_Feed_B"],
            visible=True,
            mode="lines+markers",
            line={"width": 1},
            marker=dict(size=0.3),
        ),
    ]

    fig = go.Figure(data=trace_list)
    fig.update_layout(
        title={
            "text": "예측량 실측량 비교 ",
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        }
    )

    fig.update_layout(template="plotly_dark")
    return fig


algorithm = ["XGBoost", "SVR", "LSTM", "Ensemble"]


def makeAlgorithmPrediction():
    algorithm = ["XGBoost", "SVR", "LSTM", "Ensemble"]
    prediction = ["3.323", "21.323", "103.323", "5.323"]

    return [
        dbc.Col(
            daq.LEDDisplay(
                id="our-LED-display",
                label=a,
                value=prediction[idx],
                color="#f4d44d",
                size=32,
            ),
        )
        for idx, a in enumerate(algorithm)
    ]


# Card

card = dbc.Card(
    dbc.CardBody(
        [
            html.H5("88%", className="card-title"),
            html.H6("MAPE", className="card-subtitle"),
        ]
    ),
    inverse=True,
    style={
        "width": "10rem",
    },
    id="card",
)

data = OrderedDict(
    [
        (
            "(Simulation Result)",
            [
                1,
                2,
                3,
                4,
                5,
                6,
            ],
        ),
        (
            "(Simulation Result2)",
            [
                1,
                2,
                3,
                4,
                5,
                6,
            ],
        ),
        (
            "(Simulation Result3)",
            [
                1,
                2,
                3,
                4,
                5,
                6,
            ],
        ),
        (
            "(Simulation Result4)",
            [
                1,
                2,
                3,
                4,
                5,
                6,
            ],
        ),
        (
            "(Simulation Result5)",
            [
                1,
                2,
                3,
                4,
                5,
                6,
            ],
        ),
    ]
)


simulation_df = pd.DataFrame(data)
# Layout

app.layout = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        card,
                        style={
                            "display": "flex",
                            "align-items": "center",
                            "justify-content": "space-around",
                        },
                    )
                )
                for _ in range(4)
            ],
        ),
        dbc.Row(makeAlgorithmPrediction()),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(
                        id="bar_graph",
                        figure=makeBarGraph(),
                        style={"height": "30vh", "width": "70vh"},
                    ),
                ),
                dbc.Col(
                    dcc.Graph(
                        id="bar_graph",
                        figure=makeLineGraph(),
                        style={"height": "30vh", "width": "70vh"},
                    ),
                ),
            ]
        ),
        html.Br(),
        dbc.Row(
            [
                dash_table.DataTable(
                    data=simulation_df.to_dict("records"),
                    columns=[{"id": c, "name": c} for c in simulation_df.columns],
                    id="tbl",
                    style_header={
                        "backgroundColor": "rgb(30, 30, 30)",
                        "color": "white",
                    },
                    style_cell={"textAlign": "left"},
                    style_data={
                        "backgroundColor": "rgb(50, 50, 50)",
                        "color": "white",
                        "fontFamily": "Segoe UI",
                    },
                ),
                html.Div(id="container", children=[]),
                html.Button("modeling data tesing", id="btn_4"),
            ]
        ),
    ]
)


@app.callback(
    Output("container", "children"),
    Trigger("btn_4", "n_clicks"),
    State("preprocessed_store", "data"),
    State("container", "children"),
    # prevent_initial_call=True,
)
def testing(preprocessed_store, container):
    print(preprocessed_store)
    return
    # tab_container.append(tab1_content)
    # tab_container.append(tab2_content)
    # return tab_container


if __name__ == "__main__":
    app.run_server()
