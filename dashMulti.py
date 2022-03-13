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
import math
import time
import dash_daq as daq

import dash
from dash import Dash, html, dcc
import dash_html_components as html
import dash_auth
import plotly.express as px
import pandas as pd

import dash_labs as dl  # pip3 install dash-labs
import dash_bootstrap_components as dbc  # pip3 install dash-bootstrap-components

# Code from: https://github.com/plotly/dash-labs/tree/main/docs/demos/multi_page_example1


from dash.exceptions import PreventUpdate

from dash.dependencies import Input, Output, State, ALL, MATCH, ALLSMALLER
import sys

import dash_core_components as dcc

from flask_caching import Cache

sys.path.append("./logic")

import prepare_data
import algorithm


theme = {
    "dark": True,
    "detail": "#007439",
    "primary": "#00EA64",
    "secondary": "#6E6E6E",
}

app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    plugins=[dl.plugins.pages],
    external_stylesheets=[
        dbc.themes.SLATE,
        dbc.icons.BOOTSTRAP,
    ],
    # CERULEAN,MORPH,MATERIA
    # prevent_initial_callbacks=True,
)
cache = Cache(
    app.server, config={"CACHE_TYPE": "filesystem", "CACHE_DIR": "cache-directory"}
)
TIMEOUT = 300


def simple_menu(page_collection):
    pages = page_collection.pages

    return (
        dbc.Nav(
            [
                dbc.NavItem(
                    dbc.NavLink(
                        page.label,
                        href="/{}".format(page.id),
                    )
                )
                for page in pages
            ]
        ),
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
        html.Div(
            id="testContainer",
            children=[],
        ),
        html.Button(
            "Cache 1",
            id="btn_1",
        ),
        html.Button("Cache 2", id="btn_2", style={"display": "None"}),
        dcc.Loading(
            dcc.Store(
                id="preprocessed_store",
                storage_type="session",
            ),
            type="dot",
        ),
        dcc.Loading(dcc.Store(id="actual_predictive_store"), type="dot"),
        dbc.Row(
            [dbc.Col(mainContents)],
            className="g-0",
            style={"height": "100vh"},
        ),
    ]
)

train_Xn, train_y, test_Xn, test_y, X_test = 0, 0, 0, 0, 0


@app.callback(
    Output("preprocessed_store", "data"),
    Input("btn_1", "n_clicks"),
)
@cache.memoize(timeout=TIMEOUT)
def preprocess_dataset(n_clicks):
    print("hi")

    time.sleep(1)
    df = pd.read_csv("ketep_biogas_data_20220210.csv")
    df = prepare_data.preprocess(df)
    js = df.to_dict("records")
    return js


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
