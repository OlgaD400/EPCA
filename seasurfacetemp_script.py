"""Run analysis on sea surface temperature data."""
import numpy as np
from scipy.io import netcdf
from helper_functions import write_to_file

if __name__ == "__main__":
    # weekly SST data
    temp = netcdf.NetCDFFile("data/sst.wkmean.1990-present.nc", "r").variables
    # land-sea mask
    mask = netcdf.NetCDFFile("data/lsmask.nc", "r").variables["mask"].data[0, :, :]
    mask = mask.astype(float)
    mask[mask == 0] = np.nan

    epca_args = {
        "original": {
            "num_samples": 100,
            "sample_size": 20,
        },
        "sparse s&p": {
            "num_samples": 100,
            "sample_size": 20,
        },
        "uniform white": {
            "num_samples": 100,
            "sample_size": 20,
        },
        "normal white": {
            "num_samples": 100,
            "sample_size": 20,
        },
        "outliers": {
            "num_samples": 100,
            "sample_size": 20,
        },
    }
    rpca_args = {}
    pca_args = {}
    NUM_TRIALS = 100

    for i in range(NUM_TRIALS):
        write_to_file(
            original_data=temp["sst"].data,
            num_components=2,
            timeout=60,
            pca_args=pca_args,
            epca_args=epca_args,
            rpca_args=rpca_args,
            filename="sea_temperature_data/sea_temperature_data_rel_error_trial_"
            + str(i)
            + ".txt",
            sp_probability=0.01,
            outlier_scale=5,
            outlier_fraction=0.05,
            variance_divisor=1000,
        )
