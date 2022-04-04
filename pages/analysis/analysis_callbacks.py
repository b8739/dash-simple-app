from dash.dependencies import Output, Input, State, ALL, MATCH, ALLSMALLER
import pandas as pd
import plotly.express as px
from app import application
import dash_bootstrap_components as dbc  # pip3 install dash-bootstrap-components
from dash import Dash, dcc, html, Input, Output, callback, dash_table

import plotly.graph_objs as go
import math
from xgboost import XGBRegressor
import numpy as np
import shap
from app import cache
from utils.constants import TIMEOUT
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from logic.prepare_data import to_dataframe
from utils.constants import theme


""" SHAP """


@cache.memoize(timeout=TIMEOUT)
def get_shap_values(initial_store):
    train_x = initial_store["train_x"]
    train_y = initial_store["train_y"]
    model = XGBRegressor()
    model.fit(train_x, train_y)

    shap_values = shap.TreeExplainer(model).shap_values(train_x)

    return shap_values


@application.callback(
    Output("shap_importance_store", "data"),
    Input("initial_store", "data"),
)
@cache.memoize(timeout=TIMEOUT)
def get_shap_importance(initial_store):
    train_x = initial_store["train_x"]

    shap_values = get_shap_values(initial_store)
    feature_names = train_x.columns

    rf_resultX = pd.DataFrame(shap_values, columns=feature_names)
    vals = np.abs(rf_resultX.values).mean(0)

    shap_importance = pd.DataFrame(
        list(zip(feature_names, vals)), columns=["col_name", "feature_importance_vals"]
    )
    print()

    shap_importance.sort_values(
        by=["feature_importance_vals"], ascending=False, inplace=True
    )
    shap_importance["feature_importance_vals"] = shap_importance[
        "feature_importance_vals"
    ].map(lambda x: round(x, 2))

    shap_importance_dict = shap_importance.to_dict("records")

    return shap_importance_dict


@application.callback(
    Output("bar_graph", "figure"),
    Input("shap_importance_store", "data"),
)
@cache.memoize(timeout=TIMEOUT)
def draw_shap_bar_graph(df):
    df = pd.json_normalize(df)
    fig = px.bar(
        df[:5],
        x="feature_importance_vals",
        y="col_name",
        orientation="h",
        template="plotly_dark",
        text="feature_importance_vals",
    )
    fig.update_traces(marker_color=theme["cyon"])
    fig.update_yaxes(
        title_text="Tag Name",
    )
    fig.update_xaxes(
        title_text="Feature Importance",
    )
    fig.update_layout(barmode="stack", yaxis={"categoryorder": "total ascending"})
    fig.update_layout(
        title={
            "text": "주요 변수 영향도",
            # "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        }
    )

    return fig


@application.callback(
    Output("influence_table", "children"),
    Input("shap_importance_store", "data"),
    State("influence_table", "children"),
)
@cache.memoize(timeout=TIMEOUT)
def draw_influence_table(df, div_children):
    df = df[:5]
    # children = []
    for dfRow in df:
        tableRow = html.Tr(
            [
                html.Td(dfRow["col_name"]),
                html.Td(dfRow["feature_importance_vals"]),
            ]
        )
        div_children.append(tableRow)

    return div_children


@cache.memoize(timeout=TIMEOUT)
def get_dependence_plot(df, col):
    fig = px.scatter(
        df,
        x="Biogas_prod",
        y=col,
        template="plotly_dark",
    )
    fig.update_traces(
        mode="markers", marker=dict(size=0.3, line=dict(width=2, color="#f4d44d"))
    ),
    fig.update_layout(
        title={
            "text": "관계성 그래프 (" + col + ")",
            # "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        }
    )

    return fig


@application.callback(
    Output("dependence_container", "children"),
    [Input("shap_importance_store", "data")],
    [State("df_store", "data")],
    [State("dependence_container", "children")],
)
@cache.memoize(timeout=TIMEOUT)
def draw_dependence_plot(shap_df, df, div_container):
    shap_df = pd.json_normalize(shap_df)
    top_5_cols = shap_df["col_name"][:4]
    # print(shap_values[:, 12])
    # Create figure with secondary y-axis

    child = dbc.Row(
        [
            dbc.Col(
                dcc.Graph(
                    id={"type": "dependence_plot", "index": idx},
                    figure=get_dependence_plot(df, col),
                    style={"height": "35vh"},
                ),
                width=3,
            )
            for idx, col in enumerate(top_5_cols)
        ]
    )

    div_container.append(child)
    return div_container


# @application.callback(
#     Output("dependence_plot", "figure"),
#     Input("shap_values_store", "data"),
# )
# @cache.memoize(timeout=TIMEOUT)
# def draw_shap_dependence_graph(shap_values):
#     # shap_values = pd.json_normalize(shap_values)
#     shap_values = get_shap_values()
#     df = to_dataframe()
#     # print(shap_values[:, 12])
#     # Create figure with secondary y-axis
#     fig = make_subplots(specs=[[{"secondary_y": True}]])

#     # Add traces
#     fig.add_trace(
#         go.Scatter(
#             x=df["FW_Feed_B"],
#             y=shap_values[:, 11],
#             name="FW_Feed_B",
#             mode="markers",
#             marker=dict(size=3),
#         ),  # replace with your own data source
#         secondary_y=False,
#     )

#     # Add traces
#     fig.add_trace(
#         go.Scatter(
#             x=df["FW_Feed_B"],
#             y=df["Dig_A_Temp"],
#             name="Dig_A_Temp",  # replace with your own data source
#             mode="markers",
#             marker=dict(size=3),
#         ),
#         secondary_y=True,
#     )

#     # Add figure title
#     fig.update_layout(title_text="Dependence Plot")
#     fig.update_layout(template="plotly_dark")
#     # # Set x-axis title
#     # fig.update_xaxes(title_text="xaxis title")

#     # Set y-axes titles
#     fig.update_yaxes(title_text="SHAP Values of FW_Feed_B", secondary_y=False)

#     fig.update_yaxes(title_text="Dig_A_Temp", secondary_y=True)
#     fig.update_xaxes(
#         title_text="FW_Feed_B",
#     )

#     return fig
