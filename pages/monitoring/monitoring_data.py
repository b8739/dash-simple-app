import pandas as pd

from app import cache
from utils.constants import TIMEOUT
from logic import prepare_data
import sys
import plotly.express as px


@cache.memoize(timeout=TIMEOUT)
def preprocess_dataset():
    print("load dataset")
    df = pd.read_csv("ketep_biogas_data_20220210.csv")
    df = prepare_data.preprocess(df)
    js = df.to_dict("records")
    return js


def dataframe():
    return pd.json_normalize(preprocess_dataset())
    # return pd.read_json(preprocess_dataset(), orient='records')


@cache.memoize(timeout=TIMEOUT)
def initial_data():
    df = dataframe()
    (
        train_Xn,
        train_y,
        test_Xn,
        test_y,
        X_test,
        train_x,
        test_x,
    ) = prepare_data.split_dataset(df)
    X, y = prepare_data.get_xy(df)
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
    fig = px.scatter(df, y=tag, title=None, template="plotly_dark")
    fig.update_traces(
        mode="markers", marker=dict(size=1, line=dict(width=2, color="#f4d44d"))
    ),
    fig.update_yaxes(rangemode="normal")
    # fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(
        title={
            "text": "바이오가스 생산량",
            "xref": "paper",
            "yref": "paper",
            "x": 0.5,
            # "y": 0.5,
        },
    )
    # fig.update_layout(
    #     title={
    #         "text": tag,
    #         "xref": "paper",
    #         "yref": "paper",
    #         "x": 0.5,
    #         # "y": 0.5,
    #     },
    # )
    return fig
