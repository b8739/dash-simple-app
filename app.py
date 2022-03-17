import dash
import dash_bootstrap_components as dbc

from flask_caching import Cache

from utils.external_assets import FONT_AWSOME, CUSTOM_STYLE
from layout.layout import layout

import flask
import dash_daq as daq
from utils.constants import theme


server = flask.Flask(__name__)  # define flask app.server

application = dash.Dash(
    __name__,
    server=server,
    suppress_callback_exceptions=True,
    external_stylesheets=[
        dbc.themes.SLATE,
        # dbc.themes.BOOTSTRAP,
        dbc.icons.BOOTSTRAP,
        FONT_AWSOME,
        CUSTOM_STYLE,
    ],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    assets_folder="static",
    assets_url_path="static",
)

cache = Cache(
    application.server,
    config={"CACHE_TYPE": "filesystem", "CACHE_DIR": "cache-directory"},
)
cache.clear()

application.layout = dbc.Container(
    id="dark-theme-components-1",
    children=[daq.DarkThemeProvider(theme=theme, children=layout)],
    fluid=True,
    style={
        "padding": "0",
    },
)


server = application.server
