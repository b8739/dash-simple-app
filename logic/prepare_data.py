import pandas as pd

from app import application, cache
from utils.constants import TIMEOUT, monitored_tags
from logic import prepare_data
import sys
import plotly.express as px
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dash.dependencies import Input, Output, State

""" NORMAL FUNCTIONS """


@cache.memoize(timeout=TIMEOUT)
def excel_to_df():
    df = pd.read_excel("ketep_biogas_data_20220314.xlsx")
    return df


@cache.memoize(timeout=TIMEOUT)
def to_dataframe(dictData):
    return pd.json_normalize(dictData)


""" TRAIN TEST DATASET """


@application.callback(
    Output("df_store", "data"),
    [Input("veri_dropdown", "value")],
)
@cache.memoize(timeout=TIMEOUT)
def preprocess_dataset(veri_dropdown):
    idx_upto = 0
    if not veri_dropdown:
        idx_upto = 0
    else:
        idx_upto = veri_dropdown
    df = excel_to_df()

    df.rename(
        columns={"Date": "date"}, inplace=True
    )  # Change column name from 'Date' to 'date'

    df.dtypes  # Find data types

    df["date"] = pd.to_datetime(
        df["date"]
    )  # Change data type from 'object' to 'datetime'
    df = df.iloc[0 : 1022 + idx_upto, :].copy()  # Train & Test data set

    df.dropna(axis=0, inplace=True)  # Delete entire rows which have the NAs

    js = df.to_dict("records")

    return js


""" VERI DATASET """
# 아직 veri idx를 어떻게 받아서 처리할지 반영 안함
@application.callback(
    Output("df_veri_store", "data"),
    Input("df_store", "data"),
)
def extract_veri(df_store):
    df = excel_to_df()

    df_veri = df.iloc[1022:1029, :].copy()  # Data for Verifying (TTA Test)
    js = df_veri.to_dict("records")
    return js


""" AVG_STORE """
# 원래 get_avg의 인자로 tag를 줬는데 df를 주는걸로 바꿨으니 추후 확인
@application.callback(
    Output("avg_store", "data"),
    Input("df_store", "data"),
)
def get_avg(df_dict):
    df = to_dataframe(df_dict)
    avg_js = round(df.mean(), 3).to_dict()
    return avg_js


""" X_Y_STORE """
# 추후 사용시 dict형태로 사용하게 되었으므로 재확인
@application.callback(
    Output("x_y_store", "data"),
    Input("df_store", "data"),
)
def get_xy(df_dict):
    ## EXTRACT X & y SEPARATELY ##
    df = to_dataframe(df_dict)
    X = df.drop("Biogas_prod", axis=1).to_dict(
        "records"
    )  # Take All the columns except 'Biogas_prod'
    y = df["Biogas_prod"].to_dict()  # Take 'Biogas_prod' column
    return {"X": X, "y": y}


""" Y SERIES (train_y,test_y) """
""" X DF (train_x, test_x, X_test) """


""" QUANTILE """


@application.callback(
    Output("quantile_store", "data"),
    Input("df_store", "data"),
)
@cache.memoize(timeout=TIMEOUT)
def get_quantile(df_dict):
    df = to_dataframe(df_dict)
    res = {}

    for col in df.columns:
        if col != "date":
            res[col] = {}
            res[col]["Q1"] = df[col].quantile(0.25)
            res[col]["Q2"] = df[col].quantile(0.5)
            res[col]["Q3"] = df[col].quantile(0.75)
            res[col]["Q4"] = df[col].quantile(1)
    return res


""" BIOGAS PROD. GRAPH"""


@application.callback(
    Output("biggas_graph", "figure"),
    Input("quantile_store", "data"),
    State("df_store", "data"),
    State("avg_store", "data"),
)
# @cache.memoize(timeout=TIMEOUT)
def biggas_data(quantile_store, df_dict, avg_store):
    df = to_dataframe(df_dict)
    df = df.iloc[len(df) - 100 : 1022]
    tag = "Biogas_prod"
    try:
        fig = px.line(df, x="date", y=tag, title=None, template="plotly_dark")
    except Exception:
        fig = px.line(df, x="date", y=tag, title=None, template="plotly_dark")
    finally:
        fig.update_traces(
            mode="lines+markers",
            marker=dict(size=2, line=dict(width=2, color="#f4d44d")),
            line=dict(color="#f4d44d", width=1),
        ),
        fig.update_yaxes(rangemode="normal")
        # fig.update_xaxes(rangeslider_visible=True)
        fig.update_layout(
            yaxis_title=None,
            xaxis_title="Date",
            title={
                "text": "바이오가스 생산량",
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                # "y": 0.5,
            },
            margin=dict(l=70, r=70, t=70, b=50),
        )
        " " " Quantile 표시 " " "
        # q_position = df[tag].min() * 1.1
        fig.add_annotation(
            text="Avg " + str(avg_store[tag]),
            align="left",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=1.05,
            y=1.1,
            bordercolor="black",
            borderwidth=1,
        )
        for q in ["Q1", "Q3"]:
            # q_position += df[tag].max() / 4

            fig.add_hline(
                y=quantile_store["Biogas_prod"][q],
                line_dash="dot",
                # line_color="#FFA500",
                line_color="white",
                annotation_text=q,
                annotation_position="right",
                opacity=0.55,
            )
        return fig
