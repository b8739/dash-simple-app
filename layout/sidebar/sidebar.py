import dash_bootstrap_components as dbc
import dash_html_components as html

from utils.constants import (
    home_page_location,
    gdp_page_location,
    iris_page_location,
    monitoring_location,
    modeling_location,
    analysis_location,
)
from dash import dcc, html


# we use the Row and Col components to construct the sidebar header
# it consists of a title, and a toggle, the latter is hidden on large screens
sidebar_header = dbc.Row(
    [
        dbc.Col(html.H6("Sidebar")),
        dbc.Col(
            [
                html.Button(
                    # use the Bootstrap navbar-toggler classes to style
                    html.Span(className="navbar-toggler-icon"),
                    className="navbar-toggler",
                    # the navbar-toggler classes don't set color
                    style={
                        "color": "rgba(0,0,0,.5)",
                        "borderColor": "rgba(0,0,0,.1)",
                    },
                    id="navbar-toggle",
                ),
                html.Button(
                    # use the Bootstrap navbar-toggler classes to style
                    html.Span(className="navbar-toggler-icon"),
                    className="navbar-toggler",
                    # the navbar-toggler classes don't set color
                    style={
                        "color": "rgba(0,0,0,.5)",
                        "borderColor": "rgba(255,255,255,.1)",
                    },
                    id="sidebar-toggle",
                ),
            ],
            # the column containing the toggle will be only as wide as the
            # toggle, resulting in the toggle being right aligned
            width="auto",
            # vertically align the toggle in the center
            align="center",
        ),
    ]
)

sidebar = html.Div(
    [
        sidebar_header,
        # we wrap the horizontal rule and short blurb in a div that can be
        # hidden on a small screen
        # html.Div(
        #     [
        #         html.Hr(),
        #         html.P(
        #             "기능 구성",
        #             style={"color": "#fff"},
        #             className="lead",
        #         ),
        #     ],
        #     id="blurb",
        # ),
        # # use the Collapse component to animate hiding / revealing links
        # dbc.Collapse(
        #     dbc.Nav(
        #         [
        #             # dbc.NavLink("Home", href=home_page_location, active="exact"),
        #             # dbc.NavLink("GDP", href=gdp_page_location, active="exact"),
        #             dbc.NavLink(
        #                 "모니터링 (Monitoring)", href=monitoring_location, active="exact"
        #             ),
        #             dbc.NavLink(
        #                 "모델링 (Modeling)", href=modeling_location, active="exact"
        #             ),
        #             dbc.NavLink(
        #                 "관계성 그래프 (Analysis)", href=analysis_location, active="exact"
        #             ),
        #         ],
        #         vertical=True,
        #         pills=True,
        #     ),
        #     id="collapse",
        # ),
        html.Hr(),
        html.Div(
            [
                html.P(
                    "데이터 불러오기 (For TTA)",
                    style={"color": "#fff"},
                    className="lead",
                ),
            ],
            id="ttaPanel",
        ),
        dbc.Row(
            dbc.Col(
                dcc.Dropdown(
                    id="veri_dropdown",
                    options=[
                        {"label": str(i) + "번째 데이터", "value": i} for i in range(1, 8)
                    ],
                    # value="sepal width (cm)",
                )
                # width=3,
            )
        ),
                        dcc.Store(
                    id="read_data_store",
                    storage_type="session",
                ),
    ],
    id="sidebar",
)
