import sys
import os.path
import argparse
import configparser
import math
import random
import pickle
import numpy as np
import torch
import scipy.interpolate
import pandas as pd

import lds.learning

# import warnings
# warnings.filterwarnings("error")

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--estMeta_number", help="estimation metadata number",
                        type=int, default=33)
    parser.add_argument("--skip_estimation_sigma_a",
                        help=("use this option to skip the estimation of the "
                              "sqrt noise inensity"), action="store_true")
    parser.add_argument("--skip_estimation_R",
                        help="use this option to skip the estimation of R",
                        action="store_true")
    parser.add_argument("--skip_estimation_m0",
                        help="use this option to skip the estimation of m0",
                        action="store_true")
    parser.add_argument("--skip_estimation_V0",
                        help="use this option to skip the estimation of V0",
                        action="store_true")
    parser.add_argument("--estInit_metadata_filename_pattern", type=str,
                        default="../../metadata/{:08d}_estimation.ini",
                        help="estimation initialization metadata filename pattern")
    parser.add_argument("--data_filename", type=str,
                        default="/nfs/gatsbystor/rapela/fwg/positions_noOutliers_AEON3_2024-02-05T15-00-00_2024-02-08T14-00-00.csv",
                        # default="~/gatsby-swc/fwg/repos/aeon_repo/results/positions_AEON3_2024-02-05T15-00-00_2024-02-05T16-00-00_2024-02-05T17-00-00.csv",
                        help="inputs positions filename")
    parser.add_argument("--estRes_metadata_filename_pattern", type=str,
                        default="../../results/{:08d}_estimation.ini",
                        help="estimation results metadata filename pattern")
    parser.add_argument("--estRes_data_filename_pattern", type=str,
                        default="../../results/{:08d}_estimation.pickle",
                        help="estimation results data filename pattern")
    args = parser.parse_args()

    estMeta_number = args.estMeta_number
    skip_estimation_sigma_a = args.skip_estimation_sigma_a
    skip_estimation_R = args.skip_estimation_R
    skip_estimation_m0 = args.skip_estimation_m0
    skip_estimation_V0 = args.skip_estimation_V0
    estInit_metadata_filename_pattern = args.estInit_metadata_filename_pattern
    data_filename = args.data_filename
    estRes_metadata_filename_pattern = args.estRes_metadata_filename_pattern
    estRes_data_filename_pattern = args.estRes_data_filename_pattern

    estInit_metadata_filename = \
        estInit_metadata_filename_pattern.format(estMeta_number)

    estMeta = configparser.ConfigParser()
    estMeta.read(estInit_metadata_filename)
    start_time_str = (estMeta["data_params"]["start_time"])
    end_time_str = (estMeta["data_params"]["end_time"])
    pos_x0 = float(estMeta["initial_params"]["pos_x0"])
    pos_y0 = float(estMeta["initial_params"]["pos_y0"])
    vel_x0 = float(estMeta["initial_params"]["vel_x0"])
    vel_y0 = float(estMeta["initial_params"]["vel_y0"])
    ace_x0 = float(estMeta["initial_params"]["ace_x0"])
    ace_y0 = float(estMeta["initial_params"]["ace_y0"])
    sigma_a0 = float(estMeta["initial_params"]["sigma_a"])
    sigma_x0 = float(estMeta["initial_params"]["sigma_x"])
    sigma_y0 = float(estMeta["initial_params"]["sigma_y"])
    sqrt_diag_V0 = np.array(
        [float(sqrt_diag_v0_value_str)
         for sqrt_diag_v0_value_str in estMeta["initial_params"]["sqrt_diag_V0"][1:-1].split(",")]
    )
    Qe_reg_param = float(estMeta["optim_params"]["em_Qe_reg_param"])

    max_iter = int(estMeta["optim_params"]["lbfgs_max_iter"])
    tolerance_grad = float(estMeta["optim_params"]["lbfgs_tolerance_grad"])
    tolerance_change = float(estMeta["optim_params"]["lbfgs_tolerance_change"])
    lr = float(estMeta["optim_params"]["lbfgs_lr"])
    n_epochs = int(estMeta["optim_params"]["lbfgs_n_epochs"])
    tol = float(estMeta["optim_params"]["lbfgs_tol"])

    # data = pd.read_csv(data_filename, header=False)
    data = pd.read_csv(data_filename, index_col=0)
    data.index = pd.to_datetime(data.index)
    ts = data.index
    dt = (pd.to_datetime(ts[1])-pd.to_datetime(ts[0])).total_seconds()
    start_time = pd.to_datetime(start_time_str, utc=True)
    end_time = pd.to_datetime(end_time_str, utc=True)
    data = data.iloc[np.logical_and(data.index>=start_time,
                                    data.index<=end_time)]
    timestamps = pd.to_datetime(data.index.to_numpy())
    x = data["spine2_x"].to_numpy()
    y = data["spine2_y"].to_numpy()
    pos = np.vstack((x, y))

    # make sure that the first data point is not NaN
    first_not_nan_index = np.where(np.logical_and(
        np.logical_not(np.isnan(x)), np.logical_not(np.isnan(y))))[0][0]
    timestamps = timestamps[first_not_nan_index:]
    pos = pos[:, first_not_nan_index:]
    #

    # tmp code
    # import plotly.graph_objects as go

    # fig = go.Figure()
    # trace = go.Scatter(x=pos[0, :], y=pos[1, :], mode="markers")
    # fig.add_trace(trace)
    # fig.update_layout(title="Before interpolation",
    #                   xaxis=dict(title="x (pixels)"),
    #                   yaxis=dict(title="y (pixels"))
    # fig.show()

    # interpolate nan
    times = (timestamps - timestamps[0]).total_seconds().to_numpy()
    not_nan_indices_x = set(np.where(np.logical_not(np.isnan(pos[0, :])))[0])
    not_nan_indices_y = set(np.where(np.logical_not(np.isnan(pos[1, :])))[0])
    not_nan_indices = np.array(sorted(not_nan_indices_x.union(not_nan_indices_y)))
    pos_no_nan = pos[:, not_nan_indices]
    t_no_nan = times[not_nan_indices]
    pos_interpolated = np.empty_like(pos)
    tck, u = scipy.interpolate.splprep([pos_no_nan[0, :], pos_no_nan[1, :]], s=0, u=t_no_nan)
    pos_interpolated[0, :], pos_interpolated[1, :] = scipy.interpolate.splev(times, tck)
    pos = pos_interpolated

    # tmp code
    # import plotly.graph_objects as go

    # fig = go.Figure()
    # trace = go.Scatter(x=pos[0, :], y=pos[1, :], mode="markers")
    # fig.add_trace(trace)
    # fig.update_layout(title="After interpolation",
    #                   xaxis=dict(title="x (pixels)"),
    #                   yaxis=dict(title="y (pixels"))
    # fig.show()

    if math.isnan(pos_x0):
        pos_x0 = pos[0, 0]
    if math.isnan(pos_y0):
        pos_y0 = pos[1, 0]

    B, Q, Z, R_0, Qe = lds.tracking.utils.getLDSmatricesForTracking(dt=dt,
                                                                    sigma_a=sigma_a0,
                                                                    sigma_x=sigma_x0,
                                                                    sigma_y=sigma_y0)
    Qe += Qe_reg_param * np.eye(Qe.shape[0])
    m0 = np.array([pos_x0, vel_x0, ace_x0, pos_y0, vel_y0, ace_y0],
                  dtype=np.double)
    m0_0 = m0

    vars_to_estimate = {}
    if skip_estimation_sigma_a:
        vars_to_estimate["sigma_a"] = False
    else:
        vars_to_estimate["sigma_a"] = True

    if skip_estimation_R:
        vars_to_estimate["sqrt_diag_R"] = False
    else:
        vars_to_estimate["sqrt_diag_R"] = True

    if skip_estimation_m0:
        vars_to_estimate["m0"] = False
    else:
        vars_to_estimate["m0"] = True

    if skip_estimation_V0:
        vars_to_estimate["sqrt_diag_V0"] = False
    else:
        vars_to_estimate["sqrt_diag_V0"] = True

    sqrt_diag_R_torch = torch.DoubleTensor([sigma_x0, sigma_y0])
    m0_torch = torch.from_numpy(m0)
    sqrt_diag_V0_torch = torch.from_numpy(sqrt_diag_V0)
    pos_torch = torch.from_numpy(pos.astype(np.double))
    B_torch = torch.from_numpy(B.astype(np.double))
    Qe_torch = torch.from_numpy(Qe.astype(np.double))
    Z_torch = torch.from_numpy(Z.astype(np.double))

    optim_res = lds.learning.torch_lbfgs_optimize_SS_tracking_diagV0(
        y=pos_torch, B=B_torch, sigma_a0=sigma_a0,
        Qe=Qe_torch, Z=Z_torch, sqrt_diag_R_0=sqrt_diag_R_torch, m0_0=m0_torch,
        sqrt_diag_V0_0=sqrt_diag_V0_torch, max_iter=max_iter, lr=lr,
        vars_to_estimate=vars_to_estimate, tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change, n_epochs=n_epochs, tol=tol)

    print(optim_res["termination_info"])

    # save results
    est_prefix_used = True
    while est_prefix_used:
        estRes_number = random.randint(0, 10**8)
        estRes_metadata_filename = \
            estRes_metadata_filename_pattern.format(estRes_number)
        if not os.path.exists(estRes_metadata_filename):
            est_prefix_used = False
    estRes_data_filename = estRes_data_filename_pattern.format(estRes_number)

    estimRes_metadata = configparser.ConfigParser()
    estimRes_metadata["data_params"] = {"data_filename": data_filename}
    estimRes_metadata["estimation_params"] = {"estInitNumber": estMeta_number}
    with open(estRes_metadata_filename, "w") as f:
        estimRes_metadata.write(f)

    with open(estRes_data_filename, "wb") as f:
        pickle.dump(optim_res, f)
    print("Saved results to {:s}".format(estRes_data_filename))

    breakpoint()

if __name__ == "__main__":
    main(sys.argv)
