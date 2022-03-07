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
import dash_html_components as html
import dash_auth
import plotly.express as px
import pandas as pd

import dash  # pip install dash
import dash_labs as dl  # pip3 install dash-labs
import dash_bootstrap_components as dbc  # pip3 install dash-bootstrap-components

# Code from: https://github.com/plotly/dash-labs/tree/main/docs/demos/multi_page_example1

from dash_extensions.enrich import (
    DashProxy,
    PrefixIdTransform,
    Input,
    Output,
    TriggerTransform,
    ServersideOutputTransform,
    ServersideOutput,
    Trigger,
)
from dash.exceptions import PreventUpdate

from dash.dependencies import Output, Input, State, ALL, MATCH, ALLSMALLER
import sys

import dash_core_components as dcc
from dash_extensions.multipage import (
    PageCollection,
    # app_to_page,
    module_to_page,
    Page,
    CONTENT_ID,
    URL_ID,
)

sys.path.append("./logic")
import pages.monitoring as monitoring
import pages.modeling as modeling

import prepare_data
import algorithm


def app_to_page(app, id, label):
    app.transforms.append(id)
    return Page(id=id, label=label, proxy=app)


theme = {
    "dark": True,
    "detail": "#007439",
    "primary": "#00EA64",
    "secondary": "#6E6E6E",
}


# Create app.
app = DashProxy(
    transforms=[
        TriggerTransform(),  # enable use of Trigger objects
        ServersideOutputTransform(),  # enable use of ServersideOutput objects
    ],
    suppress_callback_exceptions=True,
    # prevent_initial_callbacks=True,
    external_stylesheets=[
        dbc.themes.SLATE,
        dbc.icons.BOOTSTRAP,
    ],  # CERULEAN,MORPH,MATERIA
)


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


# page = Page(id="page", label="A page", layout=layout, callbacks=callbacks)
page = Page(
    id="page",
    label="A page",
)

# Create pages.
pc = PageCollection(
    pages=[
        page,  # page defined in current module
        app_to_page(
            monitoring.app,
            "app",
            "Monitoring",
        ),
        app_to_page(modeling.app, "modeling_app", "Modeling"),
    ]
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
                    html.Div(
                        simple_menu(pc),
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
            html.Div([html.Div(id=CONTENT_ID), dcc.Location(id=URL_ID)]),
        ],
        # html.Div(children=[[html.Div(id=CONTENT_ID), dcc.Location(id=URL_ID)]]),
    ),
]


final = html.Div(
    [
        html.Div(
            id="testContainer",
            children=[],
        ),
        html.Button("Cache 1", id="btn_1"),
        html.Button("Cache 2", id="btn_2"),
        dcc.Loading(
            dcc.Store(
                id="preprocessed_store",
                storage_type="session",
            ),
            fullscreen=True,
            type="dot",
        ),
        dcc.Loading(
            dcc.Store(id="actual_predictive_store"), fullscreen=True, type="dot"
        ),
        dbc.Row(
            [dbc.Col(mainContents)],
            className="g-0",
            style={"height": "100vh"},
        ),
    ]
)

train_Xn, train_y, test_Xn, test_y, X_test = 0, 0, 0, 0, 0


@app.callback(
    ServersideOutput("preprocessed_store", "data"),
    Input("btn_1", "n_clicks"),
    memoize=True,
)
def preprocess_dataset(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    time.sleep(1)
    df = pd.read_csv("ketep_biogas_data_20220210.csv")
    df = prepare_data.preprocess(df)
    print("recalculated")
    return df


# @app.callback(
#     Output("testContainer", "children"),
#     [Input("preprocessed_store", "data"), State("testContainer", "children")],
# )
# def store_testing2(preprocessed_store, aa):
#     print("store testing2")
#     print(preprocessed_store)
#     return aa


# return {
#     "df": df,
#     "df_veri": df_veri,
#     "train_Xn": train_Xn,
#     "train_y": train_y,
#     "test_Xn": test_Xn,
#     "test_y": test_y,
#     "X_test": X_test,
# }


# @app.callback(
#     ServersideOutput("preprocessed_store", "data"), Trigger("btn_1", "n_clicks"), memoize=True
# )
# def store_precessed_data():
#     # time.sleep(1)
#     """Dataset Preprocess"""
#     df = pd.read_csv("ketep_biogas_data_20220210.csv")
#     df = prepare_data.preprocess(df)
#     df_veri = prepare_data.extract_veri(df)
#     train_Xn, train_y, test_Xn, test_y, X_test = prepare_data.split_dataset(df)
#     print("dataset working")
#     return df, df_veri, train_Xn, train_y, test_Xn, test_y, X_test


# @app.callback(
#     ServersideOutput("actual_predictive_store", "data"), Trigger("btn_2", "n_clicks"), memoize=True
# )
# def store_actual_predictive_data():
#     rep_prediction = {"value": math.inf}

#     """ Modeling """
#     for algorithm_type in ["xgb", "rf", "svr"]:
#         # 모델 만들고 실행
#         model = algorithm.create_model(algorithm_type, train_Xn, train_y)
#         result = algorithm.run(algorithm_type, model, test_Xn, test_y)
#         # 대푯값 비교해서 최소값으로 갱신
#         if rep_prediction["value"] > result["RMSE"]:
#             rep_prediction["algorithm"] = algorithm_type
#             rep_prediction["prediction"] = result["prediction"]

#     # Actual: test_y
#     # Predict: xgb_model_predict (rep_prediction['prediction])

#     """ Actual Predictive Dataframe"""
#     result_df = algorithm.get_actual_predictive(
#         X_test, test_y, rep_prediction["prediction"]
#     )
#     print(result_df)
#     # df = pd.json_normalize(df)
#     # df = iris()
#     return result_df


# Register callbacks.
pc.navigation(app)
pc.callbacks(app)

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
