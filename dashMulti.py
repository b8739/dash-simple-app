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
import dash_daq as daq

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

theme = {
    "dark": True,
    "detail": "#007439",
    "primary": "#00EA64",
    "secondary": "#6E6E6E",
}

app = dash.Dash(
    __name__,
    plugins=[dl.plugins.pages],
    external_stylesheets=[
        dbc.themes.SLATE,
        dbc.icons.BOOTSTRAP,
    ],  # CERULEAN,MORPH,MATERIA
    prevent_initial_callbacks=True,
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
    "padding": "2rem 0.5rem",
    "height": "100%",
    "backgroundColor": "#f1f2f6",
}

pass

sidebar = dbc.Card(
    [],
    # color="primary",
    style=SIDEBAR_STYLE,
)

mainContents = [
    dbc.Col(
        [
            dbc.NavbarSimple(
                brand="Biogas 플랜트 공정 운전 변수 모니터링 및 이상 감지",
                color="primary",
                dark=True,
                fluid=True,
                style={
                    "paddingLeft": 50,
                    "paddingRight": 50,
                },
                children=[
                    dbc.Nav(
                        [
                            dbc.NavItem(
                                dbc.NavLink(
                                    page["name"], href=page["path"], active="exact"
                                )
                            )
                            for page in dash.page_registry.values()
                        ]
                    ),
                    dcc.Dropdown(
                        id="siteDropdown",
                        multi=True,
                        options=[{"label": col, "value": col} for col in ["제주", "이천"]],
                        style={
                            "width": 160,
                            "marginLeft": 25,
                            # "marginRight": 50,
                        },
                        placeholder="사이트 (지점) 선택",
                    ),
                ],
            ),
        ],
    ),
    # Contents
    dbc.Col(
        dl.plugins.page_container,
        style={"padding": "30px 50px", "backgroundColor": "#303030"},
    ),
]


final = html.Div(
    [
        dbc.Row(
            [dbc.Col(mainContents)],
            className="g-0",
            style={"height": "100vh"},
        )
    ]
)


app.layout = dbc.Container(
    id="dark-theme-components-1",
    children=[daq.DarkThemeProvider(theme=theme, children=final)],
    fluid=True,
    style={
        "padding": "0",
    },
)
if __name__ == "__main__":
    app.run_server(debug=True)
