



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
predict = np.random.randn(500)
actual = np.random.randn(500)
type = ["SVR Pred.", "RF Pred.", "ENSEMBLE Pred.", "Actual MI"] * (500 // 4)

# table
assessment = ["","SVM", "RANDOM FOREST", "ENSEMBLE"]
table_header = [html.Thead(html.Tr([html.Th(n) for n in assessment]))]

row1 = html.Tr([html.Td("Correlation"),html.Td("(num)"),html.Td("(num)"),html.Td("(num)")])
row2 =html.Tr([html.Td("RMSE"),html.Td("(num)"),html.Td("(num)"),html.Td("(num)")])
row3 =html.Tr([html.Td("MAPE(Mean Absolute Percentage Error"),html.Td("(num)"),html.Td("(num)"),html.Td("(num)")])

table_body = [html.Tbody([row1,row2,row3])]

table = dbc.Table(table_header + table_body, bordered=True,    hover=True,
    responsive=True,
    striped=True,)


layout = html.Div(
    [

        dcc.Dropdown(
            id="dropdown2",
            options=[
                {"label": col, "value": col} for col in ["소화조 온도", "소화조 pH", "교반기 운전값"]
            ],
            multi=True,
            value=["소화조 온도", "소화조 pH", "교반기 운전값"],
            persistence=True,
            persistence_type="session",
        ),
        dcc.Graph(id="line-chart2"),
        table,
    ]
)


@callback(Output("line-chart2", "figure"), Input("dropdown2", "value"))
def update_chart(day):
    # mask = df.species.isin(df_species)
    fig = px.scatter(y=predict, color=type)
    fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Ensemble",
        title={
            "text": "생산량 예측값 비교 Sample (기간: 최근 1주일)",
            "xref": "paper",
            "yref": "paper",
            "x": 0.5,
            # "y": 0.5,
        },
    )
    return fig
