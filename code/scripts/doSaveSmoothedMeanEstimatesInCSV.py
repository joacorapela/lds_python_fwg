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
    parser.add_argument("--smoothed_result_id", type=int,
                        help="smoothed result ID", default=47602783)
    parser.add_argument("--smoothed_filename_pattern", type=str,
                        help="smoothed filename pattern",
                        default="../../results/{:08d}_smoothed.{:s}")
    args = parser.parse_args()

    smoothed_result_id = args.smoothed_result_id
    smoothed_filename_pattern = args.smoothed_filename_pattern

    pickle_smoothed_filename = smoothed_filename_pattern.format(
        smoothed_result_id, "pickle")
    csv_smoothed_filename = smoothed_filename_pattern.format(
        smoothed_result_id, "csv")
    with open(pickle_smoothed_filename, "rb") as f:
        smoothed_res = pickle.load(f)
    df = pd.DataFrame(dict(timestamp=smoothed_res["timestamps"],
                           mpos1=smoothed_res["measurements"][0, :],
                           mpos2=smoothed_res["measurements"][1, :],
                           fpos1=smoothed_res["filter_res"]["xnn"][0, 0, :],
                           fpos2=smoothed_res["filter_res"]["xnn"][3, 0, :],
                           fvel1=smoothed_res["filter_res"]["xnn"][1, 0, :],
                           fvel2=smoothed_res["filter_res"]["xnn"][4, 0, :],
                           facc1=smoothed_res["filter_res"]["xnn"][2, 0, :],
                           facc2=smoothed_res["filter_res"]["xnn"][5, 0, :],
                           spos1=smoothed_res["smooth_res"]["xnN"][0, 0, :],
                           spos2=smoothed_res["smooth_res"]["xnN"][3, 0, :],
                           svel1=smoothed_res["smooth_res"]["xnN"][1, 0, :],
                           svel2=smoothed_res["smooth_res"]["xnN"][4, 0, :],
                           sacc1=smoothed_res["smooth_res"]["xnN"][2, 0, :],
                           sacc2=smoothed_res["smooth_res"]["xnN"][5, 0, :],
                          ))
    df.to_csv(csv_smoothed_filename)
    print(f"results saved to {csv_smoothed_filename}")

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
