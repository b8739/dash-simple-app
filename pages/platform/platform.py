import dash_core_components as dcc
import dash_html_components as html

import dash_bootstrap_components as dbc
import dash_daq as daq

from utils.constants import monitored_tags, theme, blank_figure, all_tags
import plotly.express as px

layout = html.Div(
    children=[
        dbc.Row(
            dbc.Col(dcc.Graph(figure=blank_figure(), style={"height": "40vh"})),
            style={"marginBottom": "1rem"},
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(figure=blank_figure(), style={"height": "30vh"}), width=6
                ),
                dbc.Col(
                    dcc.Graph(figure=blank_figure(), style={"height": "30vh"}), width=6
                ),
            ],
            style={"marginBottom": "1rem"},
        ),
        dbc.Row(
            [
                dbc.Col(
                    daq.Gauge(id="our-gauge", label="Default", value=6, size=150),
                ),
                dbc.Col(
                    daq.Gauge(id="our-gauge", label="Default", value=6, size=150),
                ),
                dbc.Col(
                    daq.Gauge(id="our-gauge", label="Default", value=6, size=150),
                ),
                dbc.Col(
                    daq.Gauge(id="our-gauge", label="Default", value=6, size=150),
                ),
            ]
        ),
    ]
)
