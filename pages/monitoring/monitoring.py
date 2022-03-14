import dash_core_components as dcc
import dash_html_components as html

import dash_bootstrap_components as dbc
import dash_daq as daq

from pages.monitoring.monitoring_data import biggas_data


monitored_tags = ["PS_incoming", "FW_Feed_A", "Dig_A_Temp", "Dig_Feed_A"]

theme = {
    "dark": True,
    "detail": "#007439",
    "primary": "#00EA64",
    "secondary": "#6E6E6E",
}


def isNormal(idx):
    if idx == 1:

        return {"state": "Abnormal", "color": "red"}
    else:
        return {"state": "Normal", "color": theme["primary"]}


def plotMonitoringGraphs(graph_type, graph_number):
    children = [
        dcc.Graph(
            figure=biggas_data(),
            id="biggas_graph",
            style={"height": "30vh"},
        )
    ]
    for idx in range(graph_number):
        children.append(
            dbc.Col(
                [
                    # html.Br(),
                    dcc.Graph(
                        id={"type": graph_type, "index": idx},
                        style={"height": "30vh"},
                    ),
                    # html.Br(),
                ],
                width=6,
            )
        )
    return children


tabs_styles = {"height": "44px", "align-items": "center"}

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
                    className="g-0",
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
                    className="g-0",
                )
            ]
        ),
        className="mt-3",
    ),
)

dropdowns = [
    dbc.Col(
        dcc.Dropdown(
            id={"type": "tagDropdown", "index": idx},
            options=[{"label": c, "value": c} for c in monitored_tags],
            placeholder="Select Tag",
            value=monitored_tags[idx],
            clearable=False,
            # persistence=True, #이것 때문에
            style={
                "backgroundColor": "rgb(48, 48, 48)",
            },
        ),
        width=3,
    )
    for idx in range(4)
]

contents = dbc.Col(
    dcc.Loading(
        children=[
            dbc.Col(
                [
                    html.Span(
                        # isNormal(idx)["state"],
                        "Normal",
                        style={
                            "marginRight": 15,
                            "textAlign": "center",
                        },
                    ),
                    daq.Indicator(
                        id="indicator",
                        color=theme["primary"],
                        value="Normal",
                        className="dark-theme-control",
                        style={"display": "inline-block"},
                    ),
                    dbc.Tooltip("정상 작동중입니다.", target="indicator"),
                ],
                style={
                    "paddingLeft": 12,
                    "paddingTop": 8,
                },
            ),
            html.Br(),
            dbc.Row(
                dropdowns,
            ),
            html.Br(),
            dbc.Col(
                graphs,
            ),
        ],
        type="circle",
    ),
)
layout = html.Div(
    dbc.Row(
        [
            dbc.Col(
                [
                    contents,
                    dbc.Row(),
                ],
                width=12,
            ),
        ],
        style={"position": "relative"},
    ),
)
