from dash.dependencies import Output, Input, State, ALL, MATCH, ALLSMALLER
import pandas as pd
import plotly.express as px
from logic.prepare_data import dataframe, get_quantile, get_avg
from utils.constants import monitored_tags
from app import application

from app import cache
from utils.constants import TIMEOUT


def make_avg_annotation(tag):
    return dict(
        text="Avg: " + get_avg(tag),
        align="left",
        showarrow=False,
        xref="paper",
        yref="paper",
        x=1.05,
        y=1.1,
        bordercolor="black",
        borderwidth=1,
    )


def make_quantile_annotation(name, y_pos):
    return dict(
        text=name,
        align="left",
        showarrow=False,
        xref="paper",
        x=1.05,
        y=y_pos,
        bordercolor="black",
        borderwidth=1,
    )


@application.callback(
    Output({"type": "monitoring-graph", "index": MATCH}, "figure"),
    Input({"type": "tagDropdown", "index": MATCH}, "value"),
)
@cache.memoize(timeout=TIMEOUT)
def changeTag(tag):
    " " " Plotly Graph 생성 " " "
    df = dataframe()

    if not tag:
        tag = df.columns[3]
    fig = px.scatter(df, x="date", y=tag, title=None, template="plotly_dark")

    fig.update_traces(
        mode="markers", marker=dict(size=0.5, line=dict(width=2, color="#f4d44d"))
    ),

    fig.update_yaxes(rangemode="normal")

    fig.update_yaxes(range=[df[tag].min() * (0.8), df[tag].max() * (1.2)])

    # fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(
        yaxis_title=None,
        xaxis_title="Date",
        title={
            "text": tag,
            "xref": "paper",
            "yref": "paper",
            "x": 0.5,
            # "pad": {"b": 50},
            "xanchor": "center",
            "yanchor": "middle",
            "font": {"size": 15}
            # "y": 0.5,
        },
        margin=dict(l=70, r=70, t=70, b=90, pad=20),
        # pad=dict(l=100, r=100, t=30, b=100),
    )
    " " " Quantile 표시 " " "
    quantile_info = get_quantile(*df.columns)
    # q_position = df[tag].min() * 1.1

    for q in [
        "Q1",
        "Q3",
    ]:
        # q_position += df[tag].max() / 4

        fig.add_hline(
            y=quantile_info[tag][q],
            line_dash="dot",
            line_color="orange",
            annotation_text=q,
            annotation_position="right",
            opacity=0.9,
        )

    " " " Average 표시 " " "

    fig.add_annotation(
        text="Avg: " + get_avg(tag),
        align="left",
        showarrow=False,
        xref="paper",
        yref="paper",
        x=1.05,
        y=1.1,
        bordercolor="black",
        borderwidth=1,
    )
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list(
                    [
                        dict(
                            args=[
                                {"y": [df[col]]},
                                {
                                    "yaxis": {
                                        "range": [
                                            min(df[col]) - 0.2 * min(df[col]),
                                            max(df[col]) + 1.2 * min(df[col]),
                                        ]
                                    },
                                    "annotations": [
                                        make_avg_annotation(col),
                                        make_quantile_annotation(
                                            "Q1", quantile_info[col]["Q1"]
                                        ),
                                        make_quantile_annotation(
                                            "Q3", quantile_info[col]["Q3"]
                                        ),
                                    ],
                                    "shapes": [
                                        {
                                            "type": "line",
                                            "x0": 0,
                                            "y0": quantile_info[col]["Q1"],
                                            "x1": 1,
                                            "y1": quantile_info[col]["Q1"],
                                            "xref": "paper",
                                            "yref": "y",
                                            "line": {
                                                "color": "orange",
                                                "width": 1,
                                                "dash": "dot",
                                            },
                                        },
                                        {
                                            "type": "line",
                                            "x0": 0,
                                            "y0": quantile_info[col]["Q3"],
                                            "x1": 1,
                                            "y1": quantile_info[col]["Q3"],
                                            "xref": "paper",
                                            "yref": "y",
                                            "line": {
                                                "color": "orange",
                                                "width": 1,
                                                "dash": "dot",
                                            },
                                        },
                                    ],
                                },
                            ],
                            label=col,
                            method="update",
                        )
                        for col in df.columns
                        if col != "date"
                    ]
                ),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top",
            ),
        ]
    )

    " " " 이상 구역 Rect 표시 " " "

    # if indicator == "Abnormal":
    #     fig.add_shape(
    #         type="rect",
    #         xref="x domain",
    #         yref="y domain",
    #         x0=0.65,
    #         x1=0.7,
    #         y0=0.5,
    #         y1=0.7,
    #         line=dict(color="red", width=2),
    #     )

    return fig


@application.callback(
    Output("dropdowns-collapse", "is_open"),
    [Input("collapse_btn", "n_clicks")],
    [State("dropdowns-collapse", "is_open")],
)
def toggle_dropdown(collapse_btn, is_open):
    if collapse_btn:
        return not is_open
    return is_open
