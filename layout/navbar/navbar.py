import dash_bootstrap_components as dbc
import dash_html_components as html
from dash_bootstrap_components._components.Container import Container
from utils.constants import (
    home_page_location,
    gdp_page_location,
    iris_page_location,
    monitoring_location,
    modeling_location,
    analysis_location,
)

PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

menu = dbc.Row(
    [
        dbc.Nav(
            [
                # dbc.NavLink("Home", href=home_page_location, active="exact"),
                # dbc.NavLink("GDP", href=gdp_page_location, active="exact"),
                dbc.NavLink(
                    "모니터링 및 이상감지",
                    href=monitoring_location,
                    active="exact",
                ),
                dbc.NavLink(
                    "생산량 예측",
                    href=modeling_location,
                    active="exact",
                ),
                dbc.NavLink(
                    "안정적 운전값 제시",
                    href=analysis_location,
                    active="exact",
                ),
            ],
            # vertical=True,
            # pills=True,
        ),
    ],
    className="ms-auto flex-nowrap mt-3 mt-md-0",
    # align="center",
)

navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        # dbc.Col(html.Img(src=PLOTLY_LOGO, height="30px")),
                        dbc.Col(
                            dbc.NavbarBrand(
                                "바이오가스 플랜트 이상감지 및 최적운영 시스템",
                                style={"font-size": "1.8rem"},
                            )
                        ),
                    ],
                    # align="center",
                    justify="between",
                ),
                href="https://dashapp-env.eba-5sx6fq9j.ap-northeast-2.elasticbeanstalk.com/",
                style={"textDecoration": "none"},
            ),
            dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
            dbc.Collapse(
                menu,
                id="navbar-collapse",
                is_open=False,
                navbar=True,
            ),
        ],
        fluid=True,
        style={"paddingLeft": "2%", "paddingRight": "2%"},
    ),
    color="dark",
    dark=True,
    fixed="top",
)
