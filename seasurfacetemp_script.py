"""Run analysis on sea surface temperature data."""
import numpy as np
from scipy.io import netcdf
from helper_functions import write_to_file

if __name__ == "__main__":
    # Load data.
    # weekly SST data
    sst_dictionary = netcdf.NetCDFFile("data/sst.wkmean.1990-present.nc", "r").variables
    # land-sea mask
    mask = netcdf.NetCDFFile("data/lsmask.nc", "r").variables["mask"].data[0, :, :]
    mask = mask.astype(float)
    mask[mask == 0] = np.nan

    # Define parameters for EPCA, RPCA, PCA.
    epca_args = {
        "original": {
            "num_bags": 100,
            "bag_size": 20,
        },
        "sparse": {
            "num_bags": 100,
            "bag_size": 20,
        },
        "uniform white": {
            "num_bags": 100,
            "bag_size": 20,
        },
        "normal white": {
            "num_bags": 100,
            "bag_size": 20,
        },
        "outliers": {
            "num_bags": 100,
            "bag_size": 20,
        },
    }
    rpca_args = {"reg_E": 0.2}
    pca_args = {}
    NUM_TRIALS = 100

    # Run experiments.
    for i in range(NUM_TRIALS):
        write_to_file(
            original_data=sst_dictionary["sst"].data,
            num_components=2,
            timeout=60,
            pca_args=pca_args,
            epca_args=epca_args,
            rpca_args=rpca_args,
            filename="sea_temperature_data/sea_temperature_data_rel_error_trial_"
            + str(i)
            + ".txt",
            sparse_noise_args={"sparse_probability": 0.01},
            outlier_args={"outlier_scale": 5, "outlier_fraction": 0.05},
            white_noise_args={"variance_divisor": 1000},
        )
