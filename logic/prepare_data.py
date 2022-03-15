import pandas as pd

from app import cache
from utils.constants import TIMEOUT, monitored_tags
from logic import prepare_data
import sys
import plotly.express as px
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@cache.memoize(timeout=TIMEOUT)
def preprocess_dataset():
    df = pd.read_excel("ketep_biogas_data_20220314.xlsx")

    df.rename(
        columns={"Date": "date"}, inplace=True
    )  # Change column name from 'Date' to 'date'

    df.dtypes  # Find data types

    df["date"] = pd.to_datetime(
        df["date"]
    )  # Change data type from 'object' to 'datetime'
    df = df.iloc[0:1022, :].copy()  # Train & Test data set

    df.dropna(axis=0, inplace=True)  # Delete entire rows which have the NAs

    js = df.to_dict("records")
    return js


def dataframe():
    return pd.json_normalize(preprocess_dataset())


def extract_veri():
    df = pd.read_excel("ketep_biogas_data_20220314.xlsx")
    df_veri = df.iloc[1022:1029, :].copy()  # Data for Verifying (TTA Test)
    return df_veri


def get_xy(df):
    ## EXTRACT X & y SEPARATELY ##
    X = df.drop("Biogas_prod", axis=1)  # Take All the columns except 'Biogas_prod'
    y = df["Biogas_prod"]  # Take 'Biogas_prod' column
    return X, y


@cache.memoize(timeout=TIMEOUT)
def initial_data():  # split_dataset
    df = dataframe()
    X, y = get_xy(df)
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
    return {
        "train_x": train_x,
        "train_Xn": train_Xn,
        "train_y": train_y,
        "test_Xn": test_Xn,
        "test_x": test_x,
        "test_y": test_y,
        "X_test": X_test,
        "X": X,
        "y": y,
    }


@cache.memoize(timeout=TIMEOUT)
def biggas_data():
    df = dataframe()
    tag = initial_data()["y"]
    fig = px.scatter(df, x="date", y=tag, title=None, template="plotly_dark")
    fig.update_traces(
        mode="markers", marker=dict(size=1, line=dict(width=2, color="#f4d44d"))
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
    quantile_info = get_quantile_biogas("Biogas_prod")
    # q_position = df[tag].min() * 1.1

    for q in ["Q1", "Q2", "Q3", "Q4"]:
        # q_position += df[tag].max() / 4

        fig.add_hline(
            y=quantile_info["Biogas_prod"][q],
            line_dash="dot",
            annotation_text=q,
            annotation_position="right",
            opacity=0.9,
        )
    return fig


@cache.memoize(timeout=TIMEOUT)
def get_quantile(*args):
    df = dataframe()
    res = {}
    for col in args:
        res[col] = {}
        res[col]["Q1"] = df[col].quantile(0.25)
        res[col]["Q2"] = df[col].quantile(0.5)
        res[col]["Q3"] = df[col].quantile(0.75)
        res[col]["Q4"] = df[col].quantile(1)
    return res


@cache.memoize(timeout=TIMEOUT)
def get_quantile_biogas(col):
    df = dataframe()
    res = {}
    res[col] = {}
    res[col]["Q1"] = df[col].quantile(0.25)
    res[col]["Q2"] = df[col].quantile(0.5)
    res[col]["Q3"] = df[col].quantile(0.75)
    res[col]["Q4"] = df[col].quantile(1)
    return res
