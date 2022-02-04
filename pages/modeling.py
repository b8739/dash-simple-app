import dash
import pandas as pd
import numpy as np

# Code from: https://github.com/plotly/dash-labs/tree/main/docs/demos/multi_page_example1
dash.register_page(__name__)


from dash import Dash, dcc, html, Input, Output, callback
import plotly.express as px
import dash_bootstrap_components as dbc  # pip3 install dash-bootstrap-components


# df = px.data.iris()
# df_species = df.species.unique()
# df = pd.concat([df + np.random.randn(*df.shape) * 0.1 for i in range(100)])
predict = [np.random.randint(30, 35) for n in range(100)]
actual = [np.random.randint(29, 34) for n in range(100)]
type = ["predict", "actual"] * (100 // 2)

# table
assessment = ["MSE", "RMSE", "MAE", "MAPE"]
table_header = [
    html.Thead(html.Tr([html.Th(n) for n in ["MSE", "RMSE", "MAE", "MAPE"]]))
]

row1 = html.Tr([html.Td("(number)") for _ in range(4)])

table_body = [html.Tbody([row1])]

table = dbc.Table(table_header + table_body, bordered=True)


layout = html.Div(
    [
        dcc.Dropdown(
            id="dropdown",
            options=[{"label": col, "value": col} for col in ["1"]],
            multi=True,
            persistence=True,
            persistence_type="session",
        ),
        dcc.Graph(id="line-chart"),
        table,
    ]
)


@callback(Output("line-chart", "figure"), Input("dropdown", "value"))
def update_chart(day):
    # mask = df.species.isin(df_species)
    fig = px.line(y=predict, color=type)
    fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(
        title={
            "text": "생산량 예측값 비교 Sample (기간: 최근 1주일)",
            "xref": "paper",
            "yref": "paper",
            "x": 0.5,
            # "y": 0.5,
        }
    )
    return fig
