import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app

from utils.constants import (
    home_page_location,
    gdp_page_location,
    monitoring_location,
    modeling_location,
)

from pages.home import home
from pages.gdp import gdp
from pages.monitoring import monitoring
from pages.modeling import modeling

# from pages.iris import iris


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == home_page_location:
        return monitoring.layout
    elif pathname == monitoring_location:
        return monitoring.layout
    elif pathname == modeling_location:
        return modeling.layout
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
