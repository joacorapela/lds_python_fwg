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
    em_max_iter = int(estMeta["optim_params"]["em_max_iter"])
    lr = float(estMeta["optim_params"]["lr"])
    Qe_reg_param = float(estMeta["optim_params"]["Qe_reg_param"])

    data = pd.read_csv(filepath_or_buffer=data_filename)
    data = data.iloc[start_position:start_position+number_positions,:]
    y = np.transpose(data[["x", "y"]].to_numpy())
    date_times = pd.to_datetime(data["time"])
    dt = (date_times.iloc[1]-date_times.iloc[0]).total_seconds()

    times = np.arange(0, y.shape[1]*dt, dt)
    not_nan_indices_y0 = set(np.where(np.logical_not(np.isnan(y[0, :])))[0])
    not_nan_indices_y1 = set(np.where(np.logical_not(np.isnan(y[1, :])))[0])
    not_nan_indices = np.array(sorted(not_nan_indices_y0.union(not_nan_indices_y1)))
    y_no_nan = y[:, not_nan_indices]
    t_no_nan = times[not_nan_indices]
    y_interpolated = np.empty_like(y)
    tck, u = scipy.interpolate.splprep([y_no_nan[0, :], y_no_nan[1, :]], s=0, u=t_no_nan)
    y_interpolated[0, :], y_interpolated[1, :] = scipy.interpolate.splev(times, tck)
    y = y_interpolated
    # y = y[:, :1000]

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
    Qe += Qe_reg_param * np.eye(Qe.shape[0])
    m0 = np.array([pos_x0, vel_x0, ace_x0, pos_y0, vel_y0, ace_y0],
                  dtype=np.double)

    vars_to_estimate = {}
    if skip_estimation_sqrt_noise_intensity:
        vars_to_estimate["sqrt_noise_intensity"] = False
    else:
        vars_to_estimate["sqrt_noise_intensity"] = True

    if skip_estimation_R:
        vars_to_estimate["R"] = False
    else:
        vars_to_estimate["R"] = True

    if skip_estimation_m0:
        vars_to_estimate["m0"] = False
    else:
        vars_to_estimate["m0"] = True

    if skip_estimation_V0:
        vars_to_estimate["V0"] = False
    else:
        vars_to_estimate["V0"] = True

    sqrt_diag_R = np.array([sigma_x0, sigma_y0])
    R_0 = np.diag(sqrt_diag_R)
    m0_0 = m0
    sqrt_diag_V0 = np.array([sqrt_diag_V0_value for i in range(len(m0))])
    V0_0 = np.diag(sqrt_diag_V0)

    optim_res  = lds_python.learning.em_SS_tracking(
        y=y, B=B, sqrt_noise_intensity0=sqrt_noise_intensity0,
        Qe=Qe, Z=Z, R_0=R_0, m0_0=m0_0, V0_0=V0_0,
        vars_to_estimate=vars_to_estimate,
        max_iter=em_max_iter)

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
