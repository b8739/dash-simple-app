import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import application

from utils.constants import (
    platform_location,
)

from pages.platform import platform

# from pages.iris import iris


@application.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == platform_location:
        return platform.layout
    # elif pathname == monitoring_location:
    #     return monitoring.layout
    # elif pathname == modeling_location:
    #     return modeling.layout
    # elif pathname == analysis_location:
    #     return analysis.layout
    # elif pathname == iris_page_location:
    #     return iris.layout
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )
