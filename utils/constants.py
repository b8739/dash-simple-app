import plotly.graph_objs as go

monitored_tags = ["PS_incoming", "FW_Feed_A", "Dig_A_Temp", "Dig_Feed_A"]


algorithm_type = ["xgb", "rf", "svr"]

theme = {
    "dark": True,
    "detail": "#007439",
    "primary": "#00EA64",
    "secondary": "#6E6E6E",
    "cyon": "#00bc8c",
}

def blank_figure():
    fig = go.Figure(go.Scatter(x=[], y=[]))
    fig.update_layout(template="plotly_dark")
    return fig


home_page_location = "/"
gdp_page_location = "/gdp"
iris_page_location = "/iris"
monitoring_location = "/monitoring"
modeling_location = "/modeling"
analysis_location = "/analysis"

TIMEOUT = 300
