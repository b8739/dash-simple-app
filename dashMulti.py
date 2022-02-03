# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
# flask
from flask import (
    Flask,
    jsonify,
    request,
    render_template,
    request,
    Response,
)
from dash.dependencies import Input, Output
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_auth
import plotly.express as px
import pandas as pd

import dash  # pip install dash
import dash_labs as dl  # pip3 install dash-labs
import dash_bootstrap_components as dbc  # pip3 install dash-bootstrap-components

# Code from: https://github.com/plotly/dash-labs/tree/main/docs/demos/multi_page_example1

app = dash.Dash(
    __name__,
    plugins=[dl.plugins.pages],
    external_stylesheets=[dbc.themes.CERULEAN, dbc.icons.BOOTSTRAP],
    # prevent_initial_callbacks=True,
)
# COSMO,
# SUPERHERO
# SUPERHERO


def createNav():
    list = []
    for page in dash.page_registry.values():
        if page["module"] != "pages.not_found_404":
            list.append(
                dbc.NavLink(
                    [
                        html.I(
                            className="bi bi-graph-up",
                            style={
                                "paddingRight": "10px",
                            },
                        ),
                        page["name"],
                    ],
                    page["name"],
                    href=page["path"],
                    active="exact",
                ),
            )
    return list


SIDEBAR_STYLE = {
    # "position": "fixed",
    # "top": 0,
    # "left": 0,
    # "bottom": 0,
    # "width": "16rem",
    "padding": "2rem 0.5rem",
    "height": "100%",
    # "backgroundColor": "#0B083B",
    # "backgroundColor": "#f8f9fa",
}

pass

sidebar = dbc.Card(
    [
        html.H5("기능"),
        html.Hr(),
        dbc.Nav(
            createNav(),
            vertical=True,
            pills=True,
        ),
    ],
    # color="primary",
    style=SIDEBAR_STYLE,
)

mainContents = [
    dbc.Col(
        [
            dbc.NavbarSimple(
                brand="다수 지점 통합 모니터링 (임시 타이틀)",
                color="primary",
                dark=True,
                className="mb-2",
                style={"marginLeft": "0", "paddingLeft": 15},
            )
        ],
        width=12,
    ),
    # Contents
    dbc.Col(
        dl.plugins.page_container,
        style={"padding": "50px"},
    ),
]

final = html.Div(
    [
        dbc.Row(
            [dbc.Col(sidebar, width=2), dbc.Col(mainContents, width=10)],
            className="g-0",
            style={"height": "100vh"},
        )
    ]
)


app.layout = dbc.Container(
    [final],
    fluid=True,
    style={"padding": "0"},
)
if __name__ == "__main__":
    app.run_server(debug=True)
