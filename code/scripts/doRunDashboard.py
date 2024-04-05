
import sys
import math
import argparse
import numpy as np
import pandas as pd
import datetime

import flask

import plotly.graph_objects as go
import plotly.colors
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, ALL, MATCH
import dash.exceptions

import plot_functions
def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("--local", help="run the flask server only locally", action="store_true")
    parser.add_argument("--port", type=int, help="port on which to run the falsh app", default=8050)
    parser.add_argument("--debug", help="start GUI with debug functionality", action="store_true")
    parser.add_argument("--smooth_res_filename", type=str, help="Filename containing Kalman smoothing results", default="../../results/47602783_smoothed.csv")
    parser.add_argument("--viterbi_path_filename", type=str, help="Filename containing the HMM Viterbi path", default="~/hmm//results/77422446_viterbi.csv")
    parser.add_argument("--sample_rate_for_trajectory0", help="Initial value for the sample rate for the trajectory", default=0.5, type=float)
    parser.add_argument("--xlabel_trajectory", type=str, help="xlabel for trajectory plot", default="x (pixels)")
    parser.add_argument("--ylabel_trajectory", type=str, help="ylabel for trajectory plot", default="y (pixels)")
    parser.add_argument("--trajectories_width", type=int, help="width of the trajectories plot", default=1000)
    parser.add_argument("--trajectories_height", type=int, help="height of the trajectories plot", default=1000)
    parser.add_argument("--trajectories_colorscale", type=str, help="colorscale for trajectories", default="Rainbow")
    parser.add_argument("--trajectories_opacity", type=float, help="opacity for trajectories", default=0.3)
    parser.add_argument("--mouse_figure_width", help="width of the mouse_figure plot", type=int, default=1000)
    parser.add_argument("--mouse_figure_height", type=int, help="height of the mouse_figure plot", default=1000)

    args = parser.parse_args()

    local = args.local
    port = args.port
    debug = args.debug
    smooth_res_filename = args.smooth_res_filename
    viterbi_path_filename = args.viterbi_path_filename
    sample_rate_for_trajectory0 = args.sample_rate_for_trajectory0
    xlabel_trajectory = args.xlabel_trajectory
    ylabel_trajectory = args.ylabel_trajectory
    trajectories_height = args.trajectories_height
    trajectories_width = args.trajectories_width
    trajectories_colorscale = args.trajectories_colorscale
    trajectories_opacity = args.trajectories_opacity
    mouse_figure_height = args.mouse_figure_height
    mouse_figure_width = args.mouse_figure_width

    smooth_res = pd.read_csv(smooth_res_filename)
    viterbi_path = pd.read_csv(viterbi_path_filename)

    smooth_res["timestamp"] = pd.to_datetime(smooth_res["timestamp"])
    smooth_res = smooth_res.set_index("timestamp")
    viterbi_path["timestamp"] = pd.to_datetime(viterbi_path["timestamp"])
    viterbi_path = viterbi_path.set_index("timestamp")

    def serve_layout():
        aDiv = html.Div(children=[
            html.H1(children="Behavioral Analysis Dashboard"),
            html.Hr(),
            html.H4(children="Plotting Time (sec)"),
            dcc.RangeSlider(
                min=0.0,
                max=(smooth_res.index.max()-smooth_res.index.min()).total_seconds(),
                value=[0.0, 600.0],
                id="plotTimeRangeSlider",
            ),
            html.H4(children="Position Estimates to Plot"),
            dcc.Checklist(
                options=['Measured', 'Filtered', 'Smoothed'],
                value=['Measured', 'Smoothed'],
                id="posEstimatesChecklist",
            ),
            html.H4(children="Color Time Series"),
            dcc.RadioItems(
                options=["Time", "Speed", "Acceleration", "State"],
                value="Time",
                id="colorRadioItem",
            ),
            html.Hr(),
            html.Button(children="Plot", id="plotButton", n_clicks=0),
            html.Div(
                id="plotsContainer",
                children=[
                    # html.H4(children="Trajectory"),
                    html.Div(
                        children=[
                            html.Div(
                                children=[
                                    dcc.Graph(
                                        id="trajectoryGraph",
                                    ),
                                ],
                                style={'padding': 10, 'flex': 1}
                            ),
                        ],
                        style={'display': 'flex', 'flex-direction': 'row'}
                    ),
                ],
                hidden=False),
        ])
        return aDiv

    external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
    app.layout = serve_layout()

    @app.callback([Output('trajectoryGraph', 'figure'),
                   Output('plotsContainer', 'hidden'),
                   Output('plotButton', 'children'),
                  ],
                  [Input('plotButton', 'n_clicks')],
                  [State('plotTimeRangeSlider', 'value'),
                   State('posEstimatesChecklist', 'value'), 
                   State('colorRadioItem', 'value'), 
                  ],
                  )
    def update_plots(plotButton_nClicks,
                     plotTimeRangeSlider_value,
                     posEstimatesChecklist_value,
                     colorRadioItem_value,
                    ):
        if plotButton_nClicks == 0:
            print("update prevented ({:s})".format(flask.request.remote_addr))
            raise dash.exceptions.PreventUpdate
        print("update_plots called ({:s})".format(flask.request.remote_addr))
        t0 = smooth_res.index.min() + datetime.timedelta(seconds=plotTimeRangeSlider_value[0])
        tf = smooth_res.index.min() + datetime.timedelta(seconds=plotTimeRangeSlider_value[1])

        smooth_res_to_plot = smooth_res[np.logical_and(t0 <= smooth_res.index,
                                                       smooth_res.index <= tf)]
        # trajectory figure
        fig_trajectory = go.Figure()
        if colorRadioItem_value == "State":
            viterbi_path_to_plot = viterbi_path[np.logical_and(t0 <= viterbi_path.index,
                                                               viterbi_path.index <= tf)]
            joint_to_plot = viterbi_path_to_plot.merge(smooth_res_to_plot,
                                                       how="inner",
                                                       left_index=True,
                                                       right_index=True)
            states = np.sort(joint_to_plot["state"].unique())
            for state in states:
                state_joint_to_plot = joint_to_plot[joint_to_plot["state"]==state]
                x = state_joint_to_plot["mpos1"].to_numpy()
                y = state_joint_to_plot["mpos2"].to_numpy()
                time_secs = (state_joint_to_plot.index -
                             joint_to_plot.index.min()).total_seconds()
                trace = go.Scatter(x=x, y=y, mode="markers",
                                   customdata=time_secs,
                                   hovertemplate="<b>x:</b>%{x:.3f}<br><b>y</b>:%{y:.3f}<br><b>time</b>:%{customdata:.3f} sec<br><b>state</b>:"+str(state),
                                   name=f"state {state}")
                fig_trajectory.add_trace(trace)
        else:
            for posEstimate in posEstimatesChecklist_value:
                if posEstimate == "Measured":
                    time_secs = (smooth_res_to_plot.index - smooth_res.index.min()).total_seconds()
                    x = smooth_res_to_plot["mpos1"].to_numpy()
                    y = smooth_res_to_plot["mpos2"].to_numpy()
                    color_ts = np.ones(shape=(len(time_secs),))
                    colorscale = "Rainbow"
                elif posEstimate == "Filtered":
                    x = smooth_res_to_plot["fpos1"].to_numpy()
                    y = smooth_res_to_plot["fpos2"].to_numpy()
                    if colorRadioItem_value == "Time":
                        time_secs = (smooth_res_to_plot.index - smooth_res.index.min()).total_seconds()
                        color_ts = time_secs
                        colorscale = "Rainbow"
                    elif colorRadioItem_value == "Speed":
                        time_secs = (smooth_res_to_plot.index - smooth_res.index.min()).total_seconds()
                        color_ts = np.sqrt(smooth_res_to_plot["fvel1"]**2+smooth_res_to_plot["fvel2"]**2)
                        colorscale = "Rainbow"
                    elif colorRadioItem_value == "Acceleration":
                        time_secs = (smooth_res_to_plot.index - smooth_res.index.min()).total_seconds()
                        color_ts = np.sqrt(smooth_res_to_plot["facc1"]**2+smooth_res_to_plot["facc2"]**2)
                        colorscale = "Rainbow"
                elif posEstimate == "Smoothed":
                    if colorRadioItem_value == "Time":
                        time_secs = (smooth_res_to_plot.index - smooth_res.index.min()).total_seconds()
                        x = smooth_res_to_plot["spos1"].to_numpy()
                        y = smooth_res_to_plot["spos2"].to_numpy()
                        color_ts = time_secs
                        colorscale = "Rainbow"
                    elif colorRadioItem_value == "Speed":
                        time_secs = (smooth_res_to_plot.index - smooth_res.index.min()).total_seconds()
                        x = smooth_res_to_plot["spos1"].to_numpy()
                        y = smooth_res_to_plot["spos2"].to_numpy()
                        color_ts = np.sqrt(smooth_res_to_plot["svel1"]**2+smooth_res_to_plot["svel2"]**2)
                        colorscale = "Rainbow"
                    elif colorRadioItem_value == "Acceleration":
                        time_secs = (smooth_res_to_plot.index - smooth_res.index.min()).total_seconds()
                        x = smooth_res_to_plot["spos1"].to_numpy()
                        y = smooth_res_to_plot["spos2"].to_numpy()
                        color_ts = np.sqrt(smooth_res_to_plot["sacc1"]**2+smooth_res_to_plot["sacc2"]**2)
                        colorscale = "Rainbow"
                else:
                    raise ValueError(f"Invalid posEstimate={posEstimate}")
                trajectory_trace = plot_functions.get_trayectory_trace(x=x, y=y,
                                                                       time_secs=time_secs,
                                                                       color_ts=color_ts,
                                                                       colorbar_title=colorRadioItem_value,
                                                                       colorscale=colorscale,
                                                                       name=posEstimate)
                fig_trajectory.add_trace(trajectory_trace)
        fig_trajectory.update_layout(xaxis_title=xlabel_trajectory,
                                     yaxis_title=ylabel_trajectory,
                                     paper_bgcolor='rgba(0,0,0,0)',
                                     plot_bgcolor='rgba(0,0,0,0)',
                                     height=trajectories_height,
                                     width=trajectories_width)
        fig_trajectory.update_yaxes(autorange="reversed")


        plotsContainer_hidden = False
        plotButton_children = "Update"

        return fig_trajectory, plotsContainer_hidden, plotButton_children

    if(local):
        app.run_server(debug=args.debug, port=args.port)
    else:
        app.run_server(debug=args.debug, port=args.port, host="0.0.0.0")

if __name__=="__main__":
    main(sys.argv)
