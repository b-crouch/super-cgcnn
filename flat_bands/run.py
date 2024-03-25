import time
from datetime import datetime
import argparse
import sys
import numpy as np
import pandas as pd
import pickle
import os
from utils import *

parser = argparse.ArgumentParser(description='Near-Fermi Band Flatness Calculation')
parser.add_argument('--api', type=str, help="Materials Project API key")
parser.add_argument("--freq", type=int, required=False, help="Save checkpoint of flatness data (CSV of flatness metrics) every n structures")
parser.add_argument("--fermi", type=float, nargs="*", required=False, help="Energy range(s) (eV) from Fermi level to treat as 'near-Fermi'")
parser.add_argument("--mpid", type=str, help="Filepath to csv containing MPIDs to query")
parser.add_argument("--csv_dest", type=str, help="Filename for saving flatness scores")
parser.add_argument("--bs_dest", type=str, required=False, help="Filepath to folder for saving BandStructure objects (if excluded, no BandStructures saved)")
parser.add_argument("--img_dest", type=str, required=False, help="Filepath to folder for saving flatness plots (if excluded, no plots generated)")
parser.add_argument("--plot_range", type=float, required=False, help="Energy range from Fermi level to consider when plotting band structure. Must be one of the ranges passed to `--fermi`")

args = parser.parse_args(sys.argv[1:])

def main():
    global args

    api_key = args.api
    mpids = pd.read_csv(args.mpid)["mpid"].to_numpy()
    checkpoint = args.freq if args.freq else len(mpids)//10
    fermi_windows = [*args.fermi] if args.fermi else [1, 0.75, 0.5, 0.25, 0.1]
    csv_dest = args.csv_dest
    img_dest = args.img_dest 
    bs_dest = args.bs_dest
    plot_window = args.plot_range
    save_bs = bs_dest is not None

    if plot_window:
        assert plot_window in fermi_windows, "Argument `--plot_range` must one of ranges passed to `--fermi`"

    data = np.zeros((len(mpids), 18*len(fermi_windows)))
    columns = []
    for window in fermi_windows:
        for metric in ["flatness", "bandwidth"]:
            columns.extend([f"min_{metric}_{window}_ev", f"min_relative_{metric}_{window}_ev", f"mean_{metric}_{window}_ev", f"mean_relative_{metric}_{window}_ev", f"sd_{metric}_{window}_ev", f"sd_relative_{metric}_{window}_ev", f"range_{metric}_{window}_ev", f"range_relative_{metric}_{window}_ev", f"system_{metric}_{window}_ev"])
    computed_mpids = []
    failed_mpids = []

    start = time.time()
    print(f"Start calculation: {datetime.now()}")

    for i, mpid in enumerate(mpids):
        if i%checkpoint == 0 and i > 0:
            print("*"*10, f" Structure {i}/{len(mpids)} ", "*"*10)
            print(f"Time elapsed: {np.round((time.time() - start)/60, 3)} minutes")
            df = pd.DataFrame(data[:i, :], index=mpids[:i], columns=columns)
            df.to_csv(csv_dest)

        try:
            flatness_data, band_dict = characterize_bands(mpid, api_key, fermi_windows, plot_window, img_dest=img_dest, save_bs=save_bs)
            data[i, :] = flatness_data
            computed_mpids.append(mpid)

            if save_bs:
                if not os.path.exists(bs_dest):
                    os.makedirs(bs_dest)
                with open(f"{bs_dest}/{mpid}.pkl", 'wb') as f:
                    pickle.dump(band_dict, f)
        except:
         # Some Materials Project band structures are misformatted and trigger a Unicode error if accessed
            print(f"Corrupted MP band structure for {mpid}, skipping flatness calculation")
            data[i, :] = 18*[np.nan]*len(fermi_windows)
            failed_mpids.append(mpid)

    df = pd.DataFrame(data, index=mpids, columns=columns)
    df.to_csv(csv_dest)



if __name__ == '__main__':
    main()