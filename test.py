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

dash.register_page(__name__, path="/test")


layout = html.Div(
    [
        # when the browser/tab closes.
        dcc.Store(id="session", storage_type="session"),
        html.Table(
            [
                html.Thead(
                    [
                        html.Tr(html.Th("Click to store in:", colSpan="3")),
                        html.Tr(
                            [
                                html.Th(
                                    html.Button("sessionStorage", id="session-button")
                                ),
                            ]
                        ),
                        html.Tr(
                            [
                                html.Th("Session clicks"),
                            ]
                        ),
                    ]
                ),
                html.Tbody(
                    [
                        html.Tr(
                            [
                                html.Td(0, id="session-clicks"),
                            ]
                        )
                    ]
                ),
            ]
        ),
    ]
)


@callback(
    Output("session", "data"),
    Input("session-button", "n_clicks"),
    State("session", "data"),
    prevent_initial_callbacks=True,
)
def on_click(n_clicks, data):
    print("n_clicks:", n_clicks)
    if n_clicks is None:
        raise PreventUpdate
    time.sleep(0.2)
    print("hello")
    df = px.data.iris()
    data = df.to_json(orient="records")

    return data


@callback(
    ServersideOutput("session-clicks", "children"),
    # Since we use the data prop in an output,
    # we cannot get the initial data on load with the data prop.
    # To counter this, you can use the modified_timestamp
    # as Input and the data as State.
    # This limitation is due to the initial None callbacks
    # https://github.com/plotly/dash-renderer/pull/81
    Input("session", "modified_timestamp"),
    State("session", "data"),
)
def on_data(ts, data):
    if ts is None:
        raise PreventUpdate

    data = data or {}

    return data
