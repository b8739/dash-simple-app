import dash_core_components as dcc
import dash_html_components as html

import dash_bootstrap_components as dbc
import dash_daq as daq

from utils.constants import monitored_tags, theme, blank_figure, all_tags
import plotly.express as px


def plotMonitoringGraphs(graph_type, graph_number):
    children = [
        dcc.Graph(
            # figure=biggas_data(),
            id="biggas_graph",
            style={
                "height": "30vh",
                "margin-bottom": "1.5rem",
            },
            figure=blank_figure(),
        ),
    ]
    for idx in range(graph_number):
        # children.append(indicator_content)
        children.append(
            dbc.Col(
                [
                    daq.Indicator(
                        id={"type": "indicator", "index": idx},
                        color=theme["primary"],
                        value="Normal",
                        className="dark-theme-control",
                        style={
                            "display": "inline-block",
                            "position": "absolute",
                            "zIndex": 1,
                            "top": 15,
                            "left": 45,
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
    label="?????? ??????",
    id="tab-1",
)

tab2_content = dbc.Tab(
    dbc.Card(
        # dbc.CardBody([html.Div(plotMonitoringGraphs('bioGas',1))]),
        className="mt-3",
    ),
    label="?????? ??????",
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
                options=[{"label": c, "value": c} for c in all_tags],
                placeholder="Select Tag",
                value=monitored_tags[idx],
                clearable=False,
                # persistence=True, #?????? ?????????
                style={
                    "backgroundColor": "rgb(48, 48, 48)",
                },
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
                    html.Button(
                        "?????? ?????? ????????? ????????? ???????????? ????????? ????????? ?????? ??????",
                        id="btn_3",
                        style={"display": "none"},
                    ),
                    dbc.Col(
                        [
                            html.H5("Biogas ????????? ?????? ?????? ?????? ???????????? ??? ?????? ??????"),
                        ]
                    ),
                    # ???????????? indicator
                    # dbc.Col(
                    #     html.H6(
                    #         "?????? ?????? ??????",
                    #         style={
                    #             "display": "inline",
                    #         },
                    #     ),
                    #     style={
                    #         "text-align": "right",
                    #     },
                    # ),
                    dbc.Col(
                        children=[
                            html.H6(
                                "?????? ??????:",
                                style={
                                    "display": "inline",
                                    "marginRight": 15,
                                    "font-size": "1.8rem",
                                    "font-weight": "100",
                                },
                            ),
                            html.Span(
                                # isNormal(idx)["state"],
                                "Normal  ",
                                id="normal_all_span",
                                style={
                                    "marginRight": 10,
                                    "textAlign": "center",
                                    "color": "white",
                                    "font-size": "1.7rem",
                                },
                            ),
                            daq.Indicator(
                                id="normal_all_indicator",
                                color=theme["primary"],
                                value="Normal",
                                className="dark-theme-control",
                                style={"display": "inline-block"},
                            ),
                            dbc.Tooltip("?????? ??????????????????.", target="indicator"),
                            html.Span(
                                # isNormal(idx)["state"],
                                "Abnormal",
                                id="abnormal_all_span",
                                style={
                                    "marginLeft": 20,
                                    "marginRight": 10,
                                    "textAlign": "center",
                                    "color": "grey",
                                    "font-size": "1.7rem",
                                },
                            ),
                            daq.Indicator(
                                id="abnormal_all_indicator",
                                color="rgba(255, 0, 0, 0.1)",
                                # color="grey",
                                value="Abnormal",
                                className="dark-theme-control",
                                style={"display": "inline-block"},
                            ),
                        ],
                        style={
                            "textAlign": "right",
                        },
                    ),
                    html.Hr(),
                    dropdowns,
                    dbc.Col(
                        graphs,
                    ),
                ],
                justify="between",
                align="center",
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
