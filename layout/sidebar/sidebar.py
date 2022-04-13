import dash_bootstrap_components as dbc
import dash_html_components as html


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
                    placeholder="Select Row Index",
                    value=0,
                    clearable=True,
                    options=[
                        {"label": str(i) + "번째 데이터", "value": i} for i in range(1, 8)
                    ],
                    # persistence=True,
                    style={"backgroundColor": "rgb(48, 48, 48)"},
                    # value="sepal width (cm)",
                )
                # width=3,
            )
        ),
    ],
    id="sidebar",
)
