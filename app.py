import dash
import dash_bootstrap_components as dbc

from flask_caching import Cache

from utils.external_assets import FONT_AWSOME, CUSTOM_STYLE
from layout.layout import layout

import flask
import dash_daq as daq

theme = {
    "dark": True,
    "detail": "#007439",
    "primary": "#00EA64",
    "secondary": "#6E6E6E",
}


server = flask.Flask(__name__) # define flask app.server

app = dash.Dash(
    __name__,
    server=server,
    suppress_callback_exceptions=True, 
    external_stylesheets=[
        dbc.themes.SLATE,
        # dbc.themes.BOOTSTRAP,
        dbc.icons.BOOTSTRAP,
        FONT_AWSOME,
        CUSTOM_STYLE
    ],

    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)

cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory'
})

app.layout = dbc.Container(
    id="dark-theme-components-1",
    children=[daq.DarkThemeProvider(theme=theme, children=layout)],
    fluid=True,
    style={
        "padding": "0",
    },
)


server = app.server