import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import streamlit as st


class Barplot:
    def __init__(self, title: str, xlabels: list):
        self.layout = go.Layout()
        # self.layout = go.Layout(
        #     {
        #         "annotationdefaults": {
        #             "arrowcolor": "#f2f5fa",
        #             "arrowhead": 0,
        #             "arrowwidth": 1,
        #         },
        #         "autotypenumbers": "strict",
        #         "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}},
        #         "colorscale": {
        #             "diverging": [
        #                 [0, "#8e0152"],
        #                 [0.1, "#c51b7d"],
        #                 [0.2, "#de77ae"],
        #                 [0.3, "#f1b6da"],
        #                 [0.4, "#fde0ef"],
        #                 [0.5, "#f7f7f7"],
        #                 [0.6, "#e6f5d0"],
        #                 [0.7, "#b8e186"],
        #                 [0.8, "#7fbc41"],
        #                 [0.9, "#4d9221"],
        #                 [1, "#276419"],
        #             ],
        #             "sequential": [
        #                 [0.0, "#0d0887"],
        #                 [0.1111111111111111, "#46039f"],
        #                 [0.2222222222222222, "#7201a8"],
        #                 [0.3333333333333333, "#9c179e"],
        #                 [0.4444444444444444, "#bd3786"],
        #                 [0.5555555555555556, "#d8576b"],
        #                 [0.6666666666666666, "#ed7953"],
        #                 [0.7777777777777778, "#fb9f3a"],
        #                 [0.8888888888888888, "#fdca26"],
        #                 [1.0, "#f0f921"],
        #             ],
        #             "sequentialminus": [
        #                 [0.0, "#0d0887"],
        #                 [0.1111111111111111, "#46039f"],
        #                 [0.2222222222222222, "#7201a8"],
        #                 [0.3333333333333333, "#9c179e"],
        #                 [0.4444444444444444, "#bd3786"],
        #                 [0.5555555555555556, "#d8576b"],
        #                 [0.6666666666666666, "#ed7953"],
        #                 [0.7777777777777778, "#fb9f3a"],
        #                 [0.8888888888888888, "#fdca26"],
        #                 [1.0, "#f0f921"],
        #             ],
        #         },
        #         "colorway": [
        #             "#ff792b",
        #             "#FF9C58",
        #             "#00cc96",
        #             "#ab63fa",
        #             "#FFA15A",
        #             "#19d3f3",
        #             "#FF6692",
        #             "#B6E880",
        #             "#FF97FF",
        #             "#FECB52",
        #         ],
        #         "font": {"color": "#f2f5fa"},
        #         "geo": {
        #             "bgcolor": "#3D3C3E",
        #             "lakecolor": "#3D3C3E",
        #             "landcolor": "#3D3C3E",
        #             "showlakes": True,
        #             "showland": True,
        #             "subunitcolor": "#395780",
        #         },
        #         "hoverlabel": {"align": "left"},
        #         "hovermode": "closest",
        #         "mapbox": {"style": "dark"},
        #         "paper_bgcolor": "#3d3c3e",
        #         "plot_bgcolor": "#3d3c3e",
        #         "polar": {
        #             "angularaxis": {
        #                 "gridcolor": "#3d3c3e",
        #                 "linecolor": "#3d3c3e",
        #                 "ticks": "",
        #             },
        #             "bgcolor": "#3D3C3E",
        #             "radialaxis": {
        #                 "gridcolor": "#506784",
        #                 "linecolor": "#506784",
        #                 "ticks": "",
        #             },
        #         },
        #         "scene": {
        #             "xaxis": {
        #                 "backgroundcolor": "#3D3C3E",
        #                 "gridcolor": "#506784",
        #                 "gridwidth": 2,
        #                 "linecolor": "#FF9C58",
        #                 "showbackground": True,
        #                 "ticks": "",
        #                 "zerolinecolor": "#FF9C58",
        #             },
        #             "yaxis": {
        #                 "backgroundcolor": "#3D3C3E",
        #                 "gridcolor": "#FF9C58",
        #                 "gridwidth": 2,
        #                 "linecolor": "#FF9C58",
        #                 "showbackground": True,
        #                 "ticks": "",
        #                 "zerolinecolor": "#C8D4E3",
        #             },
        #             "zaxis": {
        #                 "backgroundcolor": "#3D3C3E",
        #                 "gridcolor": "#FF9C58",
        #                 "gridwidth": 2,
        #                 "linecolor": "#FF9C58",
        #                 "showbackground": True,
        #                 "ticks": "",
        #                 "zerolinecolor": "#C8D4E3",
        #             },
        #         },
        #         "shapedefaults": {"line": {"color": "#f2f5fa"}},
        #         "sliderdefaults": {
        #             "bgcolor": "#C8D4E3",
        #             "bordercolor": "rgb(17,17,17)",
        #             "borderwidth": 1,
        #             "tickwidth": 0,
        #         },
        #         "ternary": {
        #             "aaxis": {
        #                 "gridcolor": "#506784",
        #                 "linecolor": "#506784",
        #                 "ticks": "",
        #             },
        #             "baxis": {
        #                 "gridcolor": "#506784",
        #                 "linecolor": "#506784",
        #                 "ticks": "",
        #             },
        #             "bgcolor": "rgb(17,17,17)",
        #             "caxis": {
        #                 "gridcolor": "#506784",
        #                 "linecolor": "#506784",
        #                 "ticks": "",
        #             },
        #         },
        #         "title": {"x": 0.05},
        #         "updatemenudefaults": {"bgcolor": "#506784", "borderwidth": 0},
        #         "xaxis": {
        #             "automargin": True,
        #             "gridcolor": "#283442",
        #             "linecolor": "#506784",  # xachse Farbe
        #             "ticks": "",
        #             "title": {"standoff": 15},
        #             "zerolinecolor": "#283442",
        #             "zerolinewidth": 2,
        #         },
        #         "yaxis": {
        #             "automargin": True,
        #             "gridcolor": "#283442",
        #             "linecolor": "#506784",  # yachse Farbe
        #             "ticks": "",
        #             "title": {"standoff": 15},
        #             "zerolinecolor": "#283442",
        #             "zerolinewidth": 2,
        #         },
        #     }
        # )
        """
        moods = [
            "Angry",
            "Disgusted",
            "Fearful",
            "Happy",
            "Neutral",
            "Sad",
            "Surprised",
        ]
        """
        self.xlabels = xlabels
        self.fig = go.FigureWidget(layout=self.layout, layout_title_text=title, layout_yaxis_range=[0, 100])
        zeros = [0] * len(self.xlabels)
        self.fig.add_bar(y=zeros, x=self.xlabels)
        # display
        # self.fig.update_layout(
        #     autosize=True
        # )
        self.stfig = st.plotly_chart(self.fig, use_container_width=True)

    def update(self, data):
        self.fig.data[0].y = data
        self.stfig.plotly_chart(self.fig)


def build_hist(history, current, xlabels, max_len=50):
    if history is None:
        if len(current) == 0:
            current = [0, 0, 0, 0, 0, 0, 0]
        history = pd.DataFrame(dict(zip(xlabels, current)), index=[0])
    else:
        history = pd.concat([history, pd.DataFrame(dict(zip(xlabels, current)), index=[0])])
        if history.shape[0] > max_len:
            history = history.iloc[1:]  # delete 1st row
    agg = agg_hist(history)
    return history, agg


def agg_hist(history):
    return np.around(history.mean().values, 0).flatten().tolist()
