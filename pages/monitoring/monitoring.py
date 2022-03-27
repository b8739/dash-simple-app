import dash_core_components as dcc
import dash_html_components as html

import dash_bootstrap_components as dbc
import dash_daq as daq

from utils.constants import monitored_tags, theme, blank_figure
import plotly.express as px


def isNormal(idx):
    if idx == 1:

        return {"state": "Abnormal", "color": "red"}
    else:
        return {"state": "Normal", "color": theme["primary"]}


def plotMonitoringGraphs(graph_type, graph_number):
    children = [
        dcc.Graph(
            # figure=biggas_data(),
            id="biggas_graph",
            style={
                "height": "30vh",
            },
            figure=blank_figure(),
        )
    ]
    for idx in range(graph_number):
        # children.append(indicator_content)
        children.append(
            dbc.Col(
                [
                    daq.Indicator(
                        id="indicator",
                        color=theme["primary"],
                        value="Normal",
                        className="dark-theme-control",
                        style={
                            "display": "inline-block",
                            "position": "absolute",
                            "zIndex": 1,
                            "top": 33,
                            "left": 40,
                        },
                    ),
                    # html.Br(),
                    dcc.Graph(
                        id={"type": graph_type, "index": idx},
                        style={
                            "height": "23vh",
                            "zIndex": 2,
                        },
                        figure=blank_figure(),
                    ),
                ],
                # html.Br(),
                width=6,
                style={"position": "relative"},
            ),
        )
    return children


tabs_styles = {"height": "44px", "alignItems": "center"}

tab_style = {
    "fontWeight": "bold",
    "border-radius": "15px",
    "backgroundColor": "#F2F2F2",
    "padding": "6px",
    "backgroundColor": "#32383e",
    # "box-shadow": "4px 4px 4px 4px lightgrey",
}

tab_selected_style = {
    "borderTop": "1px solid #d6d6d6",
    "borderBottom": "1px solid #d6d6d6",
    "backgroundColor": "black",
    "color": "white",
    "padding": "6px",
    "border-radius": "15px",
}

tab1_content = dbc.Tab(
    dbc.Card(
        dbc.CardBody(
            [
                dbc.Row(
                    plotMonitoringGraphs("monitoring-graph", 4),
                    # className="g-0",
                )
            ]
        ),
        className="mt-3",
    ),
    label="이상 감지",
    id="tab-1",
)

tab2_content = dbc.Tab(
    dbc.Card(
        # dbc.CardBody([html.Div(plotMonitoringGraphs('bioGas',1))]),
        className="mt-3",
    ),
    label="성능 감시",
    id="tab-1",
)

tabs = dbc.Tabs(
    [tab1_content, tab2_content],
    id="tab_container",
)

graphs = (
    dbc.Card(
        dbc.CardBody(
            [
                dbc.Row(
                    plotMonitoringGraphs("monitoring-graph", 4),
                    className="g-3",
                )
            ]
        ),
        className="mt-3",
    ),
)


dropdowns = dbc.Row(
    [
        dbc.Col(
            dcc.Dropdown(
                id={"type": "tagDropdown", "index": idx},
                options=[{"label": c, "value": c} for c in monitored_tags],
                placeholder="Select Tag",
                value=monitored_tags[idx],
                clearable=False,
                # persistence=True, #이것 때문에
                style={"backgroundColor": "rgb(48, 48, 48)", "display": "none"},
            ),
            width=2,
        )
        for idx in range(4)
    ],
    justify="left",
)


contents = dbc.Col(
    dcc.Loading(
        children=[
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H5("Biogas 플랜트 공정 운전 변수 모니터링 밎 이상 감지"),
                        ]
                    ),
                    # Normal
                    dbc.Col(
                        [
                            html.Div(children=[],id='anomaly_indication')
                        ],
                        style={
                            "paddingTop": 8,
                            "paddingRight": 10,
                            "textAlign": "right",
                            # "marginTop": 15,
                            "marginBottom": 15,
                        },
                        width=4,
                    ),
                    html.Hr(),
                    dropdowns,
                    dbc.Col(
                        graphs,
                    ),
                ],
                justify="between",
            ),
        ],
        id="monitor-loading",
        type="circle",
    )
)
layout = html.Div(
    dbc.Row(
        [
            # dbc.Col(
            #     [
            #         contents,
            #     ],
            #     width=12,
            # ),
            contents
        ],
        style={"position": "relative"},
    ),
)
