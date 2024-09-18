import sys
import os
import random
import pickle
import math
import argparse
import configparser
import numpy as np
import pandas as pd

import lds.tracking.utils
import lds.inference


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_offset_secs", type=int, default=0,
                        help="start offset in seconds")
    parser.add_argument("--duration_secs", type=int, default=-1,
                        help="duration in seconds")
    parser.add_argument("--filtering_params_filename", type=str,
                        default="../../metadata/00000013_smoothing.ini",
                        help="filtering parameters filename")
    parser.add_argument("--filtering_params_section", type=str,
                        default="params",
                        help=("section of ini file containing the filtering "
                              "params"))
    parser.add_argument(
        "--data_filename", type=str,
        default="/nfs/gatsbystor/rapela/fwg/positions_noOutliers_AEON3_2024-02-05T15-00-00_2024-02-08T14-00-00.csv",
        # default="~/gatsby-swc/fwg/repos/aeon_repo/results/interpolated_NaNs_positions_2022-08-15T13:11:23.791259766_0_10743.csv",
        help="inputs positions filename")
    parser.add_argument("--results_filename_pattern", type=str,
                        default="/nfs/gatsbystor/rapela/fwg/{:08d}_smoothedSpeed.{:s}")
    args = parser.parse_args()

    start_offset_secs = args.start_offset_secs
    duration_secs = args.duration_secs
    filtering_params_filename = args.filtering_params_filename
    filtering_params_section = args.filtering_params_section
    data_filename = args.data_filename
    results_filename_pattern = args.results_filename_pattern

    data = pd.read_csv(data_filename, index_col=0)
    data.index = pd.to_datetime(data.index)
    ts = data.index
    dt = (pd.to_datetime(ts[1])-pd.to_datetime(ts[0])).total_seconds()
    start_position = int(start_offset_secs / dt)
    if duration_secs < 0:
        number_positions = data.shape[0] - start_position
    else:
        number_positions = int(duration_secs / dt)
    data = data.iloc[
        start_position:(start_position+number_positions), :]
    timestamps = data.index.to_numpy()
    x = data["spine2_x"].to_numpy()
    y = data["spine2_y"].to_numpy()
    pos = np.vstack((x, y))

    # make sure that the first data point is not NaN
    first_nan_index = np.where(np.logical_and(
        np.logical_not(np.isnan(x)), np.logical_not(np.isnan(y))))[0][0]
    timestamps = timestamps[first_nan_index:]
    pos = pos[:, first_nan_index:]
    #

    smoothing_params = configparser.ConfigParser()
    smoothing_params.read(filtering_params_filename)
    pos_x0 = float(smoothing_params[filtering_params_section]["pos_x0"])
    pos_y0 = float(smoothing_params[filtering_params_section]["pos_y0"])
    vel_x0 = float(smoothing_params[filtering_params_section]["vel_x0"])
    vel_y0 = float(smoothing_params[filtering_params_section]["vel_x0"])
    acc_x0 = float(smoothing_params[filtering_params_section]["acc_x0"])
    acc_y0 = float(smoothing_params[filtering_params_section]["acc_x0"])
    sigma_a = float(smoothing_params[filtering_params_section]["sigma_a"])
    sigma_x = float(smoothing_params[filtering_params_section]["sigma_x"])
    sigma_y = float(smoothing_params[filtering_params_section]["sigma_y"])
    diag_V0 = np.array(
        [float(sqrt_diag_v0_value_str)
         for sqrt_diag_v0_value_str in smoothing_params["params"]["diag_V0"][1:-1].split(",")]
    )

    if math.isnan(pos_x0):
        pos_x0 = pos[0, 0]
    if math.isnan(pos_y0):
        pos_y0 = pos[1, 0]

    m0 = np.array([pos_x0, vel_x0, acc_x0, pos_y0, vel_y0, acc_y0],
                  dtype=np.double)
    V0 = np.diag(diag_V0)
    R = np.diag([sigma_x**2, sigma_y**2])

    B, Q, Z, R, Qt = lds.tracking.utils.getLDSmatricesForTracking(dt=dt,
                                                                  sigma_a=sigma_a,
                                                                  sigma_x=sigma_x,
                                                                  sigma_y=sigma_y)
    m0 = np.array([pos[0, 0], vel_x0, acc_x0, pos[1, 0], vel_y0, acc_y0], dtype=np.double)
    V0 = np.diag(diag_V0)
    filter_res = lds.inference.filterLDS_SS_withMissingValues_np(
        y=pos, B=B, Q=Q, m0=m0, V0=V0, Z=Z, R=R)
    smooth_res = lds.inference.smoothLDS_SS(
        B=B, xnn=filter_res["xnn"], Vnn=filter_res["Vnn"],
        xnn1=filter_res["xnn1"], Vnn1=filter_res["Vnn1"],
        m0=m0, V0=V0)
    smoothed_speed = np.sqrt(smooth_res["xnN"][1, 0, :]**2 + smooth_res["xnN"][4, 0, :]**2)
    results = {"timestamps": timestamps, "smoothed_speed": smoothed_speed}

    # save results
    res_prefix_used = True
    while res_prefix_used:
        res_number = random.randint(0, 10**8)
        metadata_filename = results_filename_pattern.format(res_number, "ini")
        if not os.path.exists(metadata_filename):
            res_prefix_used = False
    results_filename = results_filename_pattern.format(res_number, "pickle")

    with open(results_filename, "wb") as f: pickle.dump(results, f)
    print(f"Saved smoothing results to {results_filename}")

    metadata = configparser.ConfigParser()
    metadata["params"] = {
        "data_filename": data_filename,
        "start_position": start_position,
        "number_positions": number_positions,
        "filtering_params_filename": filtering_params_filename,
        "filtering_params_section": filtering_params_section,
    }
    with open(metadata_filename, "w") as f:
        metadata.write(f)

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
