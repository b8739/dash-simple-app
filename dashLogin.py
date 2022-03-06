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
from dash.dependencies import Input, Output
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_auth


import plotly.express as px
import pandas as pd


# app = dash.Dash(__name__)

# # assume you have a "long-form" data frame
# # see https://plotly.com/python/px-arguments/ for more options
# df = pd.DataFrame({
#     "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
#     "Amount": [4, 1, 2, 2, 4, 5],
#     "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
# })

# fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

# app.layout = html.Div(children=[
#     html.H1(children='Hello Dash'),

#     html.Div(children='''
#         Dash: A web application framework for your data.
#     '''),

#     dcc.Graph(
#         id='example-graph',
#         figure=fig
#     )
# ])

# Keep this out of source code repository - save in a file or a database
VALID_USERNAME_PASSWORD_PAIRS = [["hello", "world"]]

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
auth = dash_auth.BasicAuth(app, VALID_USERNAME_PASSWORD_PAIRS)

app.layout = html.Div(
    [
        html.H2(id="show-output", children=""),
        html.Button("press to show username", id="button"),
    ],
    className="container",
)


@app.callback(
    Output(component_id="show-output", component_property="children"),
    [Input(component_id="button", component_property="n_clicks")],
)
def update_output_div(n_clicks):
    username = request.authorization["username"]
    if n_clicks:
        return username
    else:
        return ""


app.scripts.config.serve_locally = True
if __name__ == "__main__":
    app.run_server(debug=True)
