import sys
import numpy as np
import pickle
import pandas as pd
import datetime
import argparse
import configparser
import plotly.graph_objects as go
import lds.tracking.plotting

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("smoothed_result_number", type=int,
                        help="number corresponding to smoothed result filename")
    parser.add_argument("--start_offset_secs",
                        help="Start plotting start_offset_secs after the start of the session",
                        type=float, default=0.0)
    parser.add_argument("--duration_secs",
                        help="Plot duration_sec seconds",
                        type=float, default=600.0)
    parser.add_argument("--variable", type=str, default="pos_vs_time",
                        help="variable to plot: pos_2D, pos_vs_time, vel, acc")
    parser.add_argument("--color_measures", type=str,
                        default="black",
                        help="color for measures markers")
    parser.add_argument("--color_finite_differences", type=str,
                        default="black",
                        help="color for finite differences markers")
    parser.add_argument("--color_pattern_filtered", type=str,
                        default="rgba(255,0,0,{:f})",
                        help="color for filtered markers")
    parser.add_argument("--color_pattern_smoothed", type=str,
                        default="rgba(0,255,0,{:f})",
                        help="color for smoothed markers")
    parser.add_argument("--cb_alpha", type=float,
                        default=0.3,
                        help="opacity coefficient for confidence bands")
    parser.add_argument("--symbol_x", type=str, default="circle",
                        help="color for x markers")
    parser.add_argument("--symbol_y", type=str, default="circle-open",
                        help="color for y markers")
    parser.add_argument("--smoothed_result_filename_pattern", type=str,
                        default="../../results/{:08d}_smoothed.{:s}",
                        help="smoothed result filename pattern")
    parser.add_argument("--fig_filename_pattern_pattern", type=str,
                        default="../../figures/{:08d}_smoothed_{:s}_start{:.02f}_dur{:.02f}.{{:s}}",
                        help="figure filename pattern")

    args = parser.parse_args()

    smoothed_result_number = args.smoothed_result_number
    start_offset_secs = args.start_offset_secs
    duration_secs = args.duration_secs
    variable = args.variable
    color_measures = args.color_measures
    color_finite_differences = args.color_finite_differences
    color_pattern_filtered = args.color_pattern_filtered
    color_pattern_smoothed = args.color_pattern_smoothed
    cb_alpha = args.cb_alpha
    symbol_x = args.symbol_x
    symbol_y = args.symbol_y
    smoothed_result_filename_pattern = args.smoothed_result_filename_pattern
    fig_filename_pattern_pattern = args.fig_filename_pattern_pattern

    smoothed_result_filename = \
        smoothed_result_filename_pattern.format(smoothed_result_number,
                                                "pickle")
    metadata_filename = \
        smoothed_result_filename_pattern.format(smoothed_result_number, "ini")
    fig_filename_pattern = \
        fig_filename_pattern_pattern.format(smoothed_result_number, variable,
                                            start_offset_secs, duration_secs)

    metadata = configparser.ConfigParser()
    metadata.read(metadata_filename)

    with open(smoothed_result_filename, "rb") as f:
        smooth_res = pickle.load(f)
    timestamps = pd.to_datetime(smooth_res["timestamps"])
    time = (timestamps - timestamps[0]).total_seconds().to_numpy()
    samples = np.where(np.logical_and(
        start_offset_secs <= time,
        time <= start_offset_secs + duration_secs))[0]
    time = time[samples]

    if variable == "pos_2D":
        filtered_means = np.vstack(
            (smooth_res["filter_res"]["xnn"][0, 0, samples],
             smooth_res["filter_res"]["xnn"][3, 0, samples]),
        )
        smoothed_means = np.vstack(
            (smooth_res["smooth_res"]["xnN"][0, 0, samples],
             smooth_res["smooth_res"]["xnN"][3, 0, samples]),
        )
        fig = lds.tracking.plotting.get_fig_mfs_positions_2D(
            time=time,
            measurements=smooth_res["measurements"][:, samples],
            filtered_means=filtered_means,
            smoothed_means=smoothed_means,
            color_measurements=color_measures,
            color_filtered=color_pattern_filtered.format(1.0),
            color_smoothed=color_pattern_smoothed.format(1.0),
        )
    elif variable == "pos_vs_time":
        filtered_means = np.vstack(
            (smooth_res["filter_res"]["xnn"][0, 0, samples],
             smooth_res["filter_res"]["xnn"][3, 0, samples]),
        )
        filtered_stds = np.sqrt(np.diagonal(
            a=smooth_res["filter_res"]["Vnn"],
            axis1=0, axis2=1))[np.ix_(samples, (0, 3))].T
        smoothed_means = np.vstack(
            (smooth_res["smooth_res"]["xnN"][0, 0, samples],
             smooth_res["smooth_res"]["xnN"][3, 0, samples]),
        )
        smoothed_stds = np.sqrt(np.diagonal(
            a=smooth_res["smooth_res"]["VnN"],
            axis1=0, axis2=1))[np.ix_(samples, (0, 3))].T
        fig = lds.tracking.plotting.get_fig_mfdfs_kinematics_1D(
            time=time,
            yaxis_title="Position (pixels)",
            measurements=smooth_res["measurements"][:, samples],
            filtered_means=filtered_means,
            filtered_stds=filtered_stds,
            smoothed_means=smoothed_means,
            smoothed_stds=smoothed_stds,
            color_measurements=color_measures,
            color_pattern_filtered=color_pattern_filtered,
            color_pattern_smoothed=color_pattern_smoothed,
            cb_alpha=cb_alpha,
        )
    elif variable == "vel":
        filtered_means = np.vstack(
            (smooth_res["filter_res"]["xnn"][1, 0, samples],
             smooth_res["filter_res"]["xnn"][4, 0, samples]),
        )
        filtered_stds = np.sqrt(np.diagonal(
            a=smooth_res["filter_res"]["Vnn"],
            axis1=0, axis2=1))[np.ix_(samples, (1, 4))].T
        smoothed_means = np.vstack(
            (smooth_res["smooth_res"]["xnN"][1, 0, samples],
             smooth_res["smooth_res"]["xnN"][4, 0, samples]),
        )
        smoothed_stds = np.sqrt(np.diagonal(
            a=smooth_res["smooth_res"]["VnN"],
            axis1=0, axis2=1))[np.ix_(samples, (1, 4))].T
        dt = (pd.to_datetime(smooth_res["timestamps"][1]) -
              pd.to_datetime(smooth_res["timestamps"][0])).total_seconds()
        vel_fd = np.zeros_like(smooth_res["measurements"])
        vel_fd[:, 1:] = (smooth_res["measurements"][:, 1:] -
                           smooth_res["measurements"][:, :-1])/dt
        vel_fd[:, 0] = vel_fd[:, 1]
        vel_fd = vel_fd[:, samples]
        fig = lds.tracking.plotting.get_fig_mfdfs_kinematics_1D(
            time=time,
            yaxis_title="Velocity (pixels/seconds)",
            finite_differences=vel_fd,
            filtered_means=filtered_means,
            filtered_stds=filtered_stds,
            smoothed_means=smoothed_means,
            smoothed_stds=smoothed_stds,
            color_fd=color_finite_differences,
            color_pattern_filtered=color_pattern_filtered,
            color_pattern_smoothed=color_pattern_smoothed,
            cb_alpha=cb_alpha,
        )
    elif variable == "acc":
        filtered_means = np.vstack(
            (smooth_res["filter_res"]["xnn"][2, 0, samples],
             smooth_res["filter_res"]["xnn"][5, 0, samples]),
        )
        filtered_stds = np.sqrt(np.diagonal(
            a=smooth_res["filter_res"]["Vnn"],
            axis1=0, axis2=1))[np.ix_(samples, (2, 5))].T
        smoothed_means = np.vstack(
            (smooth_res["smooth_res"]["xnN"][2, 0, samples],
             smooth_res["smooth_res"]["xnN"][5, 0, samples]),
        )
        smoothed_stds = np.sqrt(np.diagonal(
            a=smooth_res["smooth_res"]["VnN"],
            axis1=0, axis2=1))[np.ix_(samples, (2, 5))].T
        dt = (pd.to_datetime(smooth_res["timestamps"][1]) -
              pd.to_datetime(smooth_res["timestamps"][0])).total_seconds()
        vel_fd = np.zeros_like(smooth_res["measurements"])
        vel_fd[:, 1:] = (smooth_res["measurements"][:, 1:] -
                         smooth_res["measurements"][:, :-1])/dt
        vel_fd[:, 0] = vel_fd[:, 1]
        acc_fd = np.zeros_like(vel_fd)
        acc_fd[:, 1:] = (vel_fd[:, 1:] - vel_fd[:, :-1])/dt
        acc_fd[:, 0] = acc_fd[:, 1]
        acc_fd = acc_fd[:, samples]
        fig = lds.tracking.plotting.get_fig_mfdfs_kinematics_1D(
            time=time,
            yaxis_title="Acceleration (pixels/seconds^2)",
            finite_differences=acc_fd,
            filtered_means=filtered_means,
            filtered_stds=filtered_stds,
            smoothed_means=smoothed_means,
            smoothed_stds=smoothed_stds,
            color_fd=color_finite_differences,
            color_pattern_filtered=color_pattern_filtered,
            color_pattern_smoothed=color_pattern_smoothed,
            cb_alpha=cb_alpha,
        )
    else:
        raise ValueError("variable={:s} is invalid. It should be: pos_2D, pos_vs_time, vel, acc".format(variable))

    fig.write_image(fig_filename_pattern.format("png"))
    fig.write_html(fig_filename_pattern.format("html"))

    breakpoint()

if __name__ == "__main__":
    main(sys.argv)
