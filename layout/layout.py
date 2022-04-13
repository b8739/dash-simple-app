import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc

from layout.sidebar.sidebar import sidebar
from layout.navbar.navbar import navbar
from layout.datastore.datastore import datastore


content = html.Div(id="page-content")

layout = html.Div([dcc.Location(id="url"), content])
