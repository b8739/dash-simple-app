import dash
import time
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import Dash, Output, Input, Trigger, ServersideOutput

from dash import callback
import plotly.express as px

# This stylesheet makes the buttons and table pretty.

dash.register_page(__name__, path="/", order=1)


layout = html.Div(
    [
        dcc.Upload(
            id="upload-data",
            children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            # Allow multiple files to be uploaded
            multiple=True,
        ),
        html.Div(id="output-div"),
        html.Div(id="output-datatable"),
    ],
    style={"height": "100vh"},
)


# @callback(
#     ServersideOutput("session-clicks", "children"),
#     # Since we use the data prop in an output,
#     # we cannot get the initial data on load with the data prop.
#     # To counter this, you can use the modified_timestamp
#     # as Input and the data as State.
#     # This limitation is due to the initial None callbacks
#     # https://github.com/plotly/dash-renderer/pull/81
#     Input("session", "modified_timestamp"),
#     State("session", "data"),
# )
# def on_data(ts, data):
#     if ts is None:
#         raise PreventUpdate

#     data = data or {}

#     return data
