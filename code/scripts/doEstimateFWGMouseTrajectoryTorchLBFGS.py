import sys
import os.path
import argparse
import configparser
import math
import random
import pickle
import numpy as np
import scipy.interpolate
import scipy.stats
import pandas as pd
import torch

import lds_python.learning


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("estMeta_number", help="estimation metadata number",
                        type=int)
    parser.add_argument("--skip_estimation_sqrt_noise_intensity",
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
                        default="../../data/positions_session003_start0.00_end15548.27.csv",
                        help="inputs positions filename")
    parser.add_argument("--estRes_metadata_filename_pattern", type=str,
                        default="../../results/{:08d}_estimation.ini",
                        help="estimation results metadata filename pattern")
    parser.add_argument("--estRes_data_filename_pattern", type=str,
                        default="../../results/{:08d}_estimation.pickle",
                        help="estimation results data filename pattern")
    args = parser.parse_args()

    estMeta_number = args.estMeta_number
    skip_estimation_sqrt_noise_intensity = \
        args.skip_estimation_sqrt_noise_intensity
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
    start_position = int(estMeta["data_params"]["start_position"])
    number_positions = int(estMeta["data_params"]["number_positions"])
    pos_x0 = float(estMeta["initial_params"]["pos_x0"])
    pos_y0 = float(estMeta["initial_params"]["pos_y0"])
    vel_x0 = float(estMeta["initial_params"]["vel_x0"])
    vel_y0 = float(estMeta["initial_params"]["vel_y0"])
    ace_x0 = float(estMeta["initial_params"]["ace_x0"])
    ace_y0 = float(estMeta["initial_params"]["ace_y0"])
    sqrt_noise_intensity0 = float(estMeta["initial_params"]["sqrt_noise_intensity"])
    sigma_x0 = float(estMeta["initial_params"]["sigma_x"])
    sigma_y0 = float(estMeta["initial_params"]["sigma_y"])
    sqrt_diag_V0_value = float(estMeta["initial_params"]["sqrt_diag_v0_value"])
    max_iter = int(estMeta["optim_params"]["lbfgs_max_iter"])
    tolerance_grad = float(estMeta["optim_params"]["lbfgs_tolerance_grad"])
    tolerance_change = float(estMeta["optim_params"]["lbfgs_tolerance_change"])
    lr = float(estMeta["optim_params"]["lbfgs_lr"])
    n_epochs = int(estMeta["optim_params"]["lbfgs_n_epochs"])
    tol = float(estMeta["optim_params"]["lbfgs_tol"])

    data = pd.read_csv(filepath_or_buffer=data_filename)
    data = data.iloc[start_position:start_position+number_positions,:]
    y = np.transpose(data[["x", "y"]].to_numpy())
    date_times = pd.to_datetime(data["time"])
    dt = (date_times.iloc[1]-date_times.iloc[0]).total_seconds()

    if math.isnan(pos_x0):
        pos_x0 = y[0, 0]
    if math.isnan(pos_y0):
        pos_y0 = y[1, 0]

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
    m0 = np.array([pos_x0, vel_x0, ace_x0, pos_y0, vel_y0, ace_y0],
                  dtype=np.double)

    vars_to_estimate = {}
    if skip_estimation_sqrt_noise_intensity:
        vars_to_estimate["sqrt_noise_intensity"] = False
    else:
        vars_to_estimate["sqrt_noise_intensity"] = True

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
    sqrt_diag_V0_torch = torch.DoubleTensor([sqrt_diag_V0_value
                                             for i in range(len(m0))])
    y_torch = torch.from_numpy(y.astype(np.double))
    B_torch = torch.from_numpy(B.astype(np.double))
    Qe_torch = torch.from_numpy(Qe.astype(np.double))
    Z_torch = torch.from_numpy(Z.astype(np.double))

    optim_res = lds_python.learning.torch_lbfgs_optimize_SS_tracking_diagV0(
        y=y_torch, B=B_torch, sqrt_noise_intensity0=sqrt_noise_intensity0,
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
