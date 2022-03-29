import pandas as pd

from app import application, cache
from utils.constants import TIMEOUT, theme
from logic import prepare_data
import sys
import plotly.express as px
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dash.dependencies import Input, Output, State
from dash_extensions.enrich import Dash, Trigger, ServersideOutput

# anomaly
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import dash_html_components as html

import dash_bootstrap_components as dbc
import dash_daq as daq

""" NORMAL FUNCTIONS """


@cache.memoize(timeout=TIMEOUT)
def excel_to_df():
    df = pd.read_excel("ketep_biogas_data_20220314.xlsx")
    return df


@cache.memoize(timeout=TIMEOUT)
def to_dataframe(dictData):
    return pd.json_normalize(dictData)


def preprocess(df):
    df.rename(columns={"Date": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


""" VERI DATASET """
# 아직 veri idx를 어떻게 받아서 처리할지 반영 안함
@application.callback(
    ServersideOutput("df_veri_store", "data"),
    Trigger("btn_3", "n_clicks"),
)
@cache.memoize(timeout=TIMEOUT)
def extract_veri():
    df = excel_to_df()
    df = preprocess(df)
    df_veri = df.iloc[
        1022:1029:,
    ].copy()  # Data for Verifying (TTA Test)
    return df_veri


## For Verification Data : TTA 테스트 데이터 (7개)


""" TRAIN TEST DATASET """


@application.callback(
    ServersideOutput("df_store", "data"),
    [Input("veri_dropdown", "value")],
    [State("df_store", "data")],
    [State("df_veri_store", "data")],
)
@cache.memoize(timeout=TIMEOUT)
def extract_train_test(dropdown_value, df_store, df_veri_store):
    # Read Rows upto Index of Verification Data
    print("dropdown_value", dropdown_value)
    if dropdown_value == 0 or not dropdown_value:

        # Read Dataframe
        df = excel_to_df()

        # Preprocess
        df = preprocess(df)
        # df = df.iloc[:1022].copy()
        df = df.iloc[:1022].copy()
        df.dropna(axis=0, inplace=True)  # Delete entire rows which have the NAs
        return df
    else:
        new_df = pd.concat(
            [df_store, df_veri_store[:dropdown_value]], ignore_index=True
        )
        return new_df


""" AVG_STORE """
# 원래 get_avg의 인자로 tag를 줬는데 df를 주는걸로 바꿨으니 추후 확인
@application.callback(
    Output("avg_store", "data"),
    Input("df_store", "data"),
)
@cache.memoize(timeout=TIMEOUT)
def get_avg(df):
    avg_js = round(df.mean(), 3).to_dict()
    return avg_js


""" X_Y_STORE """
# 추후 사용시 dict형태로 사용하게 되었으므로 재확인
@application.callback(
    ServersideOutput("x_y_store", "data"),
    Input("df_store", "data"),
)
@cache.memoize(timeout=TIMEOUT)
def get_xy(df):
    ## EXTRACT X & y SEPARATELY ##
    X = df.drop("Biogas_prod", axis=1)  # Take All the columns except 'Biogas_prod'
    y = df["Biogas_prod"]  # Take 'Biogas_prod' column
    return {"X": X, "y": y}


""" INITIAL STORE (TRAIN & TEST)"""


@application.callback(
    ServersideOutput("initial_store", "data"),
    Input("x_y_store", "data"),
)
@cache.memoize(timeout=TIMEOUT)
def initial_data(x_y_store):  # split_dataset

    X, y = x_y_store["X"], x_y_store["y"]

    ## SET 'TRAIN', 'TEST' DATA, TRAIN/TEST RATIO, & 'WAY OF RANDOM SAMPLING' ##
    X_train, X_test, train_y, test_y = train_test_split(
        X, y, test_size=0.2, random_state=12345
    )

    # X_train, X_test, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 56789)

    train_x = X_train.drop(["date"], axis=1)  # Delete 'date' column from train data
    test_x = X_test.drop(["date"], axis=1)  # Delete 'date' column from test data

    # scalerX = MinMaxScaler()
    scalerX = StandardScaler()  # Data standardization (to Standard Normal distribution)
    # scalerX = RobustScaler()
    scalerX.fit(train_x)
    train_Xn = scalerX.transform(train_x)  # Scaling the train data
    test_Xn = scalerX.transform(test_x)  # Scaling the test data

    # train_b = scalerX.inverse_transform(train_Xn)
    dict_values = {
        # df
        "train_x": train_x,
        "test_x": test_x,
        "X_test": X_test,
        # numpy
        "train_Xn": train_Xn,
        "test_Xn": test_Xn,
        # series
        "train_y": train_y,
        "test_y": test_y,
    }

    return dict_values


"""ANOMALY DETECTION"""

# 아직 veri idx를 어떻게 받아서 처리할지 반영 안함
@application.callback(
    Output("anomaly_store", "data"),
    Input("x_y_store", "data"),
    State("veri_dropdown", "value"),
    State("df_veri_store", "data"),
)
@cache.memoize(timeout=TIMEOUT)
def anomaly_detect(x_y_store, dropdown_value, df_veri_store):
    if not dropdown_value:
        return "normal"
    else:
        """TRAIN TEST SPLIT"""
        X_train_0, X_test_0, train_y, test_y = train_test_split(
            x_y_store["X"], x_y_store["y"], test_size=0.2, random_state=123
        )

        X_train_0 = X_train_0.reset_index(drop=True)
        X_train = X_train_0.drop(columns=["date"])

        """ISOLATION_FOREST"""
        isolation_forest = IsolationForest(
            n_estimators=500, max_samples=256, random_state=1
        )
        isolation_forest.fit(X_train)
        # 전체 Veri가 아니라 새로 가져온 행에 대한 verfication 데이터

        X_veri = df_veri_store.drop(columns=["date", "Biogas_prod"])

        a_scores_veri = -1 * isolation_forest.score_samples(X_veri)

        """DETECTION ALL"""
        a_scores_veri = -1 * isolation_forest.score_samples(
            pd.DataFrame(X_veri.iloc[(dropdown_value - 1)]).T
        )
        print(a_scores_veri[0])
        if a_scores_veri[0] >= 0.60:
            return "abnormal"
        else:
            return "normal"

        """DETECTION SINGLE"""
        # compare_train = pd.concat([pd.Series(X_train.quantile(0.025)), pd.Series(X_train.iloc[479, ])], axis=1)

        # print(compare_train)  # 첫 번째 컬럼값 (2.5% 미만)보다 작은 값이면 이상치로 적용


""" NORMAL ALL SPAN CALLBACK"""


@application.callback(
    Output("normal_all_span", "style"),
    Input("anomaly_store", "data"),
)
@cache.memoize(timeout=TIMEOUT)
def normal_span(anomaly_store):
    default_style = {
        "marginRight": 10,
        "textAlign": "center",
    }
    if anomaly_store == "normal":
        default_style.update({"color": "white"})

    else:
        default_style.update({"color": "grey"})
    return default_style


""" NORMAL ALL INDICATOR CALLBACK"""


@application.callback(
    Output("normal_all_indicator", "color"),
    Input("anomaly_store", "data"),
)
# rgba(0, 234, 100, 1.0)
@cache.memoize(timeout=TIMEOUT)
def normal_indicator(anomaly_store):
    if anomaly_store == "normal":
        return "rgba(0, 234, 100, 1.0)"
    else:
        return "rgba(0, 234, 100, 0.1)"


""" ABNORMAL ALL SPAN CALLBACK"""


@application.callback(
    Output("abnormal_all_span", "style"),
    Input("anomaly_store", "data"),
)
@cache.memoize(timeout=TIMEOUT)
def abnormal_span(anomaly_store):
    default_style = {
        "marginLeft": 20,
        "marginRight": 10,
        "textAlign": "center",
    }
    if anomaly_store == "normal":
        default_style.update({"color": "grey"})

    else:
        default_style.update({"color": "white"})
    return default_style


""" ABNORMAL ALL INDICATOR CALLBACK"""


@application.callback(
    Output("abnormal_all_indicator", "color"),
    Input("anomaly_store", "data"),
)
# rgba(0, 234, 100, 1.0)
@cache.memoize(timeout=TIMEOUT)
def abnormal_indicator(anomaly_store):
    if anomaly_store == "normal":
        return "rgba(255, 0, 0, 0.1)"
    else:
        return "rgba(255, 0, 0)"


""" QUANTILE """


@application.callback(
    Output("quantile_store", "data"),
    Input("df_store", "data"),
)
@cache.memoize(timeout=TIMEOUT)
def get_quantile(df):
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
@cache.memoize(timeout=TIMEOUT)
def biggas_data(quantile_store, df, avg_store):
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
            x=1.02,
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
