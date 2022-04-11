import dash_bootstrap_components as dbc
import dash_html_components as html

from dash import dcc, html

datastore = html.Div(
    dcc.Loading(
        [
            dcc.Store(
                id="df_store",
                storage_type="session",
            ),
            dcc.Store(
                id="df_veri_store",
                storage_type="session",
            ),
            dcc.Store(
                id="avg_store",
                storage_type="session",
            ),
            dcc.Store(
                id="x_y_store",
                storage_type="session",
            ),
            # DATAFRAME "train_x","test_x","X_test"
            # SERIES "train_y","test_y"
            dcc.Store(
                id="initial_store",
                storage_type="session",
            ),
            dcc.Store(
                id="quantile_store",
                storage_type="session",
            ),
            dcc.Store(
                id="model_store",
                storage_type="session",
            ),
            dcc.Store(
                id="modeling_result_store",
                storage_type="session",
            ),
            dcc.Store(
                id="modeling_assessment_store",
                storage_type="session",
            ),
            dcc.Store(
                id="anomaly_store",
                storage_type="session",
            ),
            dcc.Store(
                id="predict_store",
                storage_type="session",
            ),
        ],
        # fullscreen=True,
        type="circle",
    ),
)
