import sys
import os
import random
import pickle
import math
import argparse
import configparser
import datetime
import numpy as np
import pandas as pd

sys.path.append(os.path.expanduser("~/dev/work/ucl/gatsby-swc/fwg/repos/aeon_repo/"))
import aeon.storage.sqlStorageMgr as sqlSM
import lds.tracking.utils
import lds.inference


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_start_time", help="session start time",
                        type=str, default="2022-08-15T13:11:23.791259766")
                        # type=str, default="2022-06-21T13:28:10.593659")
                        # type=str, default="2022-05-25T08:59:51.050459")
    parser.add_argument("--start_offset_secs",
                        help="Start plotting start_offset_secs after the start of the session",
                        type=float, default=0.0)
    parser.add_argument("--duration_secs",
                        help="Plot duration_sec seconds",
                        type=float, default=11178.0)
    parser.add_argument("--sample_rate", help="Plot sample rate", type=float,
                        default=20.0)
    parser.add_argument("--filtering_params_filename", type=str,
                        default="../../metadata/00000010_smoothing.ini",
                        help="filtering parameters filename")
    parser.add_argument("--filtering_params_section", type=str,
                        default="params",
                        help=("section of ini file containing the filtering "
                              "params"))
    parser.add_argument("--tunneled_host", help="Tunneled host IP address",
                        type=str, default="127.0.0.1")
    parser.add_argument("--db_server_port", help="Database server port",
                        type=int, default=3306)
    parser.add_argument("--db_user", help="DB user name", type=str,
                        default="rapela")
    parser.add_argument("--db_user_password", help="DB user password",
                        type=str, default="rapela-aeon")
    parser.add_argument("--results_filename_pattern", type=str,
                        default="../../results/{:08d}_smoothed.{:s}")
    args = parser.parse_args()

    session_start_time_str = args.session_start_time
    start_offset_secs = args.start_offset_secs
    duration_secs = args.duration_secs
    sample_rate = args.sample_rate
    filtering_params_filename = args.filtering_params_filename
    filtering_params_section = args.filtering_params_section
    tunneled_host=args.tunneled_host
    db_server_port = args.db_server_port
    db_user = args.db_user
    db_user_password = args.db_user_password
    results_filename_pattern = args.results_filename_pattern

    storageMgr = sqlSM.SQLStorageMgr(host=tunneled_host,
                                     port=db_server_port,
                                     user=db_user,
                                     passwd=db_user_password)
    session_start_time = pd.to_datetime(session_start_time_str)
    start_offset_secs_td = datetime.timedelta(seconds=start_offset_secs)
    duration_secs_td = datetime.timedelta(seconds=duration_secs)
    positions = storageMgr.getSessionPositions(
        session_start_time=session_start_time,
        start_offset_secs=start_offset_secs_td,
        duration_secs=duration_secs_td)
    timestamps = positions.index
    x = positions["x"].to_numpy()
    y = positions["y"].to_numpy()
    pos = np.vstack((x, y))

    # we need to make sure that the first data point is not NaN
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
    sqrt_diag_V0_value = float(smoothing_params[filtering_params_section]
                               ["sqrt_diag_V0_value"])

    if math.isnan(pos_x0):
        pos_x0 = pos[0, 0]
    if math.isnan(pos_y0):
        pos_y0 = pos[1, 0]

    m0 = np.array([pos_x0, vel_x0, acc_x0, pos_y0, vel_y0, acc_y0],
                  dtype=np.double)
    V0 = np.diag(np.ones(len(m0))*sqrt_diag_V0_value**2)
    R = np.diag([sigma_x**2, sigma_y**2])

    dt = (timestamps[1]-timestamps[0]).total_seconds()
    # Taken from the book
    # barShalomEtAl01-estimationWithApplicationToTrackingAndNavigation.pdf
    # section 6.2.3
    # Eq. 6.2.3-7
    B = np.array([[1, dt, .5*dt**2, 0, 0, 0],
                  [0, 1,  dt,       0, 0, 0],
                  [0, 0,  1,        0, 0, 0],
                  [0, 0,  0,        1, dt, .5*dt**2],
                  [0, 0,  0,        0, 1,  dt],
                  [0, 0,  0,        0, 0,  1]], dtype=np.double)
    Z = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0]], dtype=np.double)
    # Eq. 6.2.3-8
    Qe = np.array([[dt**5/20, dt**4/8, dt**3/6, 0, 0, 0],
                   [dt**4/8, dt**3/3,  dt**2/2, 0, 0, 0],
                   [dt**3/6, dt**2/2,  dt,      0, 0, 0],
                   [0, 0, 0,                    dt**5/20, dt**4/8, dt**3/6],
                   [0, 0, 0,                    dt**4/8, dt**3/3,  dt**2/2],
                   [0, 0, 0,                    dt**3/6, dt**2/2,  dt]],
                  dtype=np.double)
    Q = lds.tracking.utils.buildQfromQt_np(Qt=Qe, sigma_ax=sigma_a, sigma_ay=sigma_a)
    R = np.diag([sigma_x**2, sigma_y**2])
    m0 = np.array([pos_x0, 0, 0, pos_y0, 0, 0], dtype=np.double)
    V0 = np.diag(np.ones(len(m0))*sqrt_diag_V0_value**2)

    filter_res = lds.inference.filterLDS_SS_withMissingValues_np(
        y=pos, B=B, Q=Q, m0=m0, V0=V0, Z=Z, R=R)
    smooth_res = lds.inference.smoothLDS_SS(
        B=B, xnn=filter_res["xnn"], Vnn=filter_res["Vnn"],
        xnn1=filter_res["xnn1"], Vnn1=filter_res["Vnn1"],
        m0=m0, V0=V0)
    results = {"time": timestamps,
               "measurements": pos,
               "filter_res": filter_res,
               "smooth_res": smooth_res}

    # save results
    res_prefix_used = True
    while res_prefix_used:
        res_number = random.randint(0, 10**8)
        metadata_filename = results_filename_pattern.format(res_number, "ini")
        if not os.path.exists(metadata_filename):
            res_prefix_used = False
    results_filename = results_filename_pattern.format(res_number, "pickle")

    with open(results_filename, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved smoothing results to {results_filename}")

    metadata = configparser.ConfigParser()
    metadata["params"] = {
        "session_start_time_str": session_start_time_str,
        "start_offset_secs": start_offset_secs,
        "duration_secs": duration_secs,
        "sample_rate": sample_rate,
        "filtering_params_filename": filtering_params_filename,
        "filtering_params_section": filtering_params_section,
    }
    with open(metadata_filename, "w") as f:
        metadata.write(f)

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
