from dash.dependencies import Input, Output, State
from pages.modeling.modeling_data import verify
from app import application


@application.callback(
    Output("sidebar", "className"),
    [Input("sidebar-toggle", "n_clicks")],
    [State("sidebar", "className")],
)
def toggle_classname(n, classname):
    if n and classname == "":
        return "collapsed"
    return ""


@application.callback(
    Output("collapse", "is_open"),
    [Input("navbar-toggle", "n_clicks")],
    [State("collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


@application.callback(
    Output("predict_value", "value"),
    Input("veri_dropdown", "value"),
    prevent_initial_call=True,
)
def update_predict_value(data_idx):
    prediction = verify(int(data_idx) - 1)
    return prediction


@application.callback(
    Output("read_data_store", "data"),
    [Input("veri_dropdown", "value")],
    prevent_initial_call=True,
)
def read_data_upto(data_idx):
    # exception handling 필요
    return 1022 + data_idx
