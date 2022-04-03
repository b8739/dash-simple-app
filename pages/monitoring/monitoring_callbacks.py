from dash.dependencies import Output, Input, State, ALL, MATCH, ALLSMALLER
import pandas as pd
import plotly.express as px

from logic.prepare_data import to_dataframe
from utils.constants import monitored_tags
from app import application

from app import cache
from utils.constants import TIMEOUT


def make_avg_annotation(avg):
    return dict(
        text="Avg: " + str(avg),
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
    Input("quantile_store", "data"),
    State({"type": "tagDropdown", "index": MATCH}, "value"),
    State("df_store", "data"),
    State("avg_store", "data"),
)

# @cache.memoize(timeout=TIMEOUT)


def changeTag(
    quantile_store,
    tag,
    df,
    avg_store,
):
    " " " Plotly Graph 생성 " " "

    df = df.iloc[len(df) - 100 : 1022]

    if not tag:
        tag = df.columns[3]
    # fig = px.scatter(df, x="date", y=tag, title=None, template="plotly_dark")
    try:
        fig = px.line(df, x="date", y=tag, title=None, markers=True)
    except Exception:
        fig = px.line(df, x="date", y=tag, title=None, markers=True)
    finally:
        fig.update_traces(
            mode="markers+lines",
            marker=dict(size=2, line=dict(width=2.5, color="#f4d44d")),
            line=dict(color="#f4d44d", width=1),
        ),
        # fig.update_layout(paper_bgcolor="#121212", plot_bgcolor="#121212")
        fig.update_yaxes(rangemode="normal")

        fig.update_yaxes(range=[df[tag].min() * (0.8), df[tag].max() * (1.2)])

        fig.update_layout(template="plotly_dark")
        # fig.update_xaxes(rangeslider_visible=True)
        fig.update_layout(
            yaxis_title=None,
            xaxis_title="Date",
            # updatemenus와 곂치기 때문에 none
            # title={
            #     "text": tag,
            #     "xref": "paper",
            #     "yref": "paper",
            #     "x": 0.5,
            #     # "pad": {"b": 50},
            #     "xanchor": "center",
            #     "yanchor": "middle",
            #     "font": {"size": 15}
            #     # "y": 0.5,
            # },
            margin=dict(l=35, r=35, t=30, b=30, pad=20),
            # pad=dict(l=100, r=100, t=30, b=100),
        )
        " " " Quantile 표시 " " "

        # q_position = df[tag].min() * 1.1

        for q in [
            "Q1",
            "Q3",
        ]:
            # q_position += df[tag].max() / 4

            fig.add_hline(
                y=quantile_store[tag][q],
                line_dash="dot",
                line_color="white",
                # line_width=0.5,
                annotation_text=q,
                annotation_position="right",
                opacity=0.55,
            )

        " " " Average 표시 " " "

        fig.add_annotation(
            text="Avg: " + str(avg_store[tag]),
            align="left",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=1.05,
            y=1.1,
            bordercolor="black",
            borderwidth=1,
        )
        cols = list(df.columns)
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=list(
                        [
                            dict(
                                args=[
                                    {"y": [df[col]]},
                                    {
                                        # updatemenus와 곂치기 때문에 none
                                        # "title": {
                                        #     "text": col,
                                        #     "font": {"size": 15},
                                        #     "xref": "paper",
                                        #     "x": 0.5,
                                        #     "xanchor": "center",
                                        #     "yanchor": "middle",
                                        # },
                                        "yaxis": {
                                            "range": [
                                                min(df[col]) - 0.2 * min(df[col]),
                                                max(df[col]) + 1.2 * min(df[col]),
                                            ]
                                        },
                                        "annotations": [
                                            make_avg_annotation(avg_store[col]),
                                            make_quantile_annotation(
                                                "Q1", quantile_store[col]["Q1"]
                                            ),
                                            make_quantile_annotation(
                                                "Q3", quantile_store[col]["Q3"]
                                            ),
                                        ],
                                        "shapes": [
                                            {
                                                "type": "dot",
                                                "x0": 0,
                                                "y0": quantile_store[col][i],
                                                "x1": 1,
                                                "y1": quantile_store[col][i],
                                                "xref": "paper",
                                                "yref": "y",
                                                "line": {
                                                    "color": "white",
                                                    "width": 1,
                                                    # "dash": "dot",
                                                },
                                                "opacity": 0.55,
                                            }
                                            for i in ["Q1", "Q3"]
                                        ],
                                    },
                                ],
                                label=col,
                                method="update",
                            )
                            for col in [cols.pop(cols.index(tag))] + cols
                            if col != "date"
                        ]
                    ),
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    font=dict(size=15),
                    x=0.55,
                    y=1.7,  # 내릴수록 내려감
                    xanchor="center",
                    yanchor="top",
                    bordercolor="white",
                    # bgcolor="#333",
                    bgcolor="rgb(24,20,20)",
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
    Output({"type": "indicator", "index": MATCH}, "color"),
    Input({"type": "tagDropdown", "index": MATCH}, "value"),
    Input("anomaly_store", "data"),
)
def updateAnomaly(tag, anomaly_store):

    if anomaly_store[tag] == False:
        return "rgba(0, 234, 100, 1.0)"
    else:
        return "rgba(255, 0, 0)"
