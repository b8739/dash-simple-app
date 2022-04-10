import plotly.graph_objs as go

monitored_tags = ["PS_incoming", "FW_Feed_A", "Dig_A_Temp", "Dig_Feed_A"]
all_tags = [
    "PS_incoming",
    "PS_feed_A",
    "PS_feed_B",
    "PS_feed_TS",
    "PS_feed_VS",
    "FW_intake_new",
    "FW_intake_old",
    "FW_grit_light",
    "FW_grit_heavy",
    "FW_grit_out",
    "Treated_FW",
    "FW_Feed_A",
    "FW_Feed_B",
    "FW_Feed_TS",
    "FW_Feed_VS",
    "Dig_Feed_A",
    "Dig_Feed_B",
    "Dig_A_Temp",
    "Dig_A_TS",
    "Dig_A_VS",
    "Dig_B_Temp",
    "Dig_B_TS",
    "Dig_B_VS",
    "Dig_Dewater",
    "Dig_DD_A",
    "Dig_DD_B",
]


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
