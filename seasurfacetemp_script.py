import numpy as np
from helper_functions import write_to_file
from scipy.io import netcdf

if __name__ == "__main__":
    # weekly SST data
    temp = netcdf.NetCDFFile("data/sst.wkmean.1990-present.nc", "r").variables
    # land-sea mask
    mask = netcdf.NetCDFFile("data/lsmask.nc", "r").variables["mask"].data[0, :, :]
    mask = mask.astype(float)
    mask[mask == 0] = np.nan

    epca_args = {"num_samples": 100, "smoothing": False, "sample_size": 20}
    rpca_args = {}
    pca_args = {}
    write_to_file(
        original_data=temp["sst"].data,
        num_components=2,
        timeout=120,
        pca_args=pca_args,
        epca_args=epca_args,
        rpca_args=rpca_args,
        filename="sea_temperature_data.txt",
        sp_probability=0.20,
        uniform_white_variance=100,
        normal_white_variance=100,
    )
