import dash

# Code from: https://github.com/plotly/dash-labs/tree/main/docs/demos/multi_page_example1
# dash.register_page(__name__, path="/")
dash.register_page(__name__, name="공정 운전 변수 이상 감지", order=1)

from dash import Dash, dcc, html, Input, Output, State, callback
from dash_extensions.enrich import Dash, Output, Input, Trigger, ServersideOutput
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px

df = px.data.iris()

sampleCols = ["소화조 온도", "소화조 pH", "교반기 운전값"]

layout = html.Div(
    [
        dcc.Store(id="monitored_tags"),
        html.H3("공정 변수 모니터링:"),
        dbc.Button("모니터링 변수 업데이트", id="add_btn", n_clicks=0),
        dcc.Dropdown(
            id="selected_tags",
            options=[{"label": col, "value": col} for col in df.columns],
            multi=True,
        ),
        html.Div(
            id="container",
            children=[],
        ),
    ]
)


@callback(
    ServersideOutput("monitored_tags", "data"),
    Input("add_btn", "n_clicks"),
    State("selected_tags", "value"),
    memoize=True,
)
def addMonitoredVariable(n_clicks, newValue):
    print("addMonitoredVariable: ", newValue)
    return newValue


@callback(
    Output("container", "children"),
    Input("add_btn", "n_clicks"),
    State("selected_tags", "value"),
    State("monitored_tags", "data"),
    State("container", "children"),
    memoize=True,
)
def add_graph(
    n_clicks,
    selected_tags,
    monitored_tags,
    div_children,
):
    if not n_clicks:
        raise PreventUpdate

    print("Monitored: ", monitored_tags)
    print("Selected: ", selected_tags)

    for tag in selected_tags:
        # Exception: Ignore if Selected Tag is Already Being Monitored
        if monitored_tags is None or tag not in monitored_tags:
            fig = px.scatter(df, y=tag)
            fig.update_xaxes(
                rangeslider_visible=True,
            )
            newChild = dcc.Graph(
                id={"type": "monitoring-graph", "index": tag},
                figure=fig,
                style={
                    "width": "45%",
                    "display": "inline-block",
                    "outline": "thin lightgrey solid",
                    "padding": 10,
                },
            )
            div_children.append(newChild)

    if monitored_tags is not None and len(monitored_tags) > len(selected_tags):
        deleteList = []

        for graph in div_children:
            graph_name = graph["props"]["id"]["index"]
            if graph_name not in selected_tags:
                deleteList.append(graph)
                # div_children.remove(graph)
                # print("deleted ", graph_name)

        for graph in deleteList:
            div_children.remove(graph)

    return div_children
