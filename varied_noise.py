"""Functions for varied noise experiments."""

from typing import Optional, List
from sklearn import datasets
from sklearn.decomposition import PCA
from scipy.io import netcdf
from mnist import MNIST
import numpy as np
from helper_functions import (
    run_pca,
    run_rpca,
    run_epca,
    match_components,
    score_performance,
    add_sparse_noise,
    add_outliers,
    add_white_noise,
)


def outlier_comparison(
    data: np.ndarray,
    root_filepath: str,
    epca_dict: dict,
    outlier_fractions: Optional[List] = None,
    outlier_scales: Optional[List] = None,
    num_runs: int = 5,
    num_trials: int = 5,
    run_rpca_condition: bool = True,
):
    """
    Gather performance of all three methods on data corrupted with various amounts of outliers.

    Args:
        data (np.ndarray): Input data.
        prefix (str): filepath to which to save data
        epca_args (dict): Arguments for EPCA.
        outlier_fraction (List): Fraction of outliers to add to data.
        outlier_scale (List): Scale of outliers to add to data.
        num_trials (int): Number of times to corrupt data at each scale.
        num_runs (int): Number of times to run each method on a corrupted data set.
        run_rpca_condition (bool): Whether or not to run RPCA (avoid timeout).
    """
    pca = PCA(n_components=2)
    pca.fit(data)
    true_components = pca.components_

    if outlier_fractions is None:
        outlier_fractions = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25]
    if outlier_scales is None:
        outlier_scales = [2, 5, 10, 20, 100]

    # fig, axs = plt.subplots(2, 5, figsize=(25, 10))
    pca_1_avgs = []
    pca_2_avgs = []

    epca_1_avgs = []
    epca_2_avgs = []

    rpca_1_avgs = []
    rpca_2_avgs = []

    for outlier_scale in outlier_scales:
        for outlier_fraction in outlier_fractions:
            epca_1 = []
            epca_2 = []

            pca_1 = []
            pca_2 = []

            rpca_1 = []
            rpca_2 = []

            for _ in range(num_trials):
                outlier_data = add_outliers(
                    data=data,
                    outlier_fraction=outlier_fraction,
                    outlier_scale=outlier_scale,
                )

                pca_pcs, _, _ = run_pca(data=outlier_data, num_components=2)
                pca_map = match_components(true_components, pca_pcs)
                pca_performance = score_performance(true_components, pca_pcs, pca_map)
                pca_1.append(pca_performance[0])
                pca_2.append(pca_performance[1])

                if run_rpca_condition is True:
                    rpca_pcs, _, _ = run_rpca(
                        data=outlier_data, num_components=2, reg_E=0.2
                    )
                    rpca_map = match_components(true_components, rpca_pcs)
                    rpca_performance = score_performance(
                        true_components, rpca_pcs, rpca_map
                    )

                    rpca_1.append(rpca_performance[0])
                    rpca_2.append(rpca_performance[1])
                else:
                    rpca_1.append(0)
                    rpca_2.append(0)

                for _ in range(num_runs):
                    epca_pcs, _, _ = run_epca(
                        data=outlier_data,
                        num_components=2,
                        num_bags=epca_dict.get("num_bags", 100),
                        bag_size=epca_dict.get("bag_size", 5),
                    )
                    epca_map = match_components(true_components, epca_pcs)
                    epca_performance = score_performance(
                        true_components, epca_pcs, epca_map
                    )

                    epca_1.append(epca_performance[0])
                    epca_2.append(epca_performance[1])

            pca_1_avgs.append(np.average(pca_1))
            pca_2_avgs.append(np.average(pca_2))

            epca_1_avgs.append(np.average(epca_1))
            epca_2_avgs.append(np.average(epca_2))

            rpca_1_avgs.append(np.average(rpca_1))
            rpca_2_avgs.append(np.average(rpca_2))

    np.save(root_filepath + "pca_1", pca_1_avgs)
    np.save(root_filepath + "pca_2", pca_2_avgs)
    np.save(root_filepath + "epca_1", epca_1_avgs)
    np.save(root_filepath + "epca_2", epca_2_avgs)
    np.save(root_filepath + "rpca_1", rpca_1_avgs)
    np.save(root_filepath + "rpca_2", rpca_2_avgs)

    return (
        pca_1_avgs,
        pca_2_avgs,
        epca_1_avgs,
        epca_2_avgs,
        rpca_1_avgs,
        rpca_2_avgs,
    )


def sparse_noise_comparison(
    data: np.ndarray,
    root_filepath: str,
    epca_dict: dict,
    sparse_noise_probs: Optional[List] = None,
    sparse_noise_scales: Optional[List] = None,
    num_runs: int = 5,
    num_trials: int = 5,
    run_rpca_condition: bool = True,
):
    """
     Gather performance of all three methods on data corrupted with
     various amounts of sparse noise.

    Args:
        data (np.ndarray): Input data.
        prefix (str): filepath to which to save data
        epca_num_bags (int): Number of bags to use in EPCA.
        epca_bag_size (int): Size of bags to use in EPCA.
        sparse_noise_probs (List): Probability of sparse noise to add to data.
        sparse_noise_scale (List): Scale of spare noise to add to data.
        num_trials (int): Number of times to corrupt data at each scale.
        num_runs (int): Number of times to run each method on a corrupted data set.
        run_rpca_condition (bool): Whether or not to run rpca (avoid timeout).
    """
    pca = PCA(n_components=2)
    pca.fit(data)
    true_components = pca.components_

    data_samples, _ = data.shape

    if sparse_noise_probs is None:
        sparse_noise_probs = [0.01, 0.05, 0.10]
    if sparse_noise_scales is None:
        sparse_noise_scales = [0, 2, 5, 10]

    # fig, axs = plt.subplots(2, 5, figsize=(25, 10))
    pca_1_avgs = []
    pca_2_avgs = []

    epca_1_avgs = []
    epca_2_avgs = []

    rpca_1_avgs = []
    rpca_2_avgs = []

    for sp_scale in sparse_noise_scales:
        for sp_prob in sparse_noise_probs:
            epca_1 = []
            epca_2 = []

            pca_1 = []
            pca_2 = []

            rpca_1 = []
            rpca_2 = []

            for _ in range(num_trials):
                sp_data = add_sparse_noise(
                    data=data, prob=sp_prob, num=np.max(data) * sp_scale
                )
                sp_data = sp_data.reshape((data_samples, -1))

                pca_pcs, _, _ = run_pca(data=sp_data, num_components=2)
                pca_map = match_components(true_components, pca_pcs)
                pca_performance = score_performance(true_components, pca_pcs, pca_map)
                pca_1.append(pca_performance[0])
                pca_2.append(pca_performance[1])

                if run_rpca_condition is True:
                    rpca_pcs, _, _ = run_rpca(data=sp_data, num_components=2, reg_E=0.2)
                    rpca_map = match_components(true_components, rpca_pcs)
                    rpca_performance = score_performance(
                        true_components, rpca_pcs, rpca_map
                    )

                    rpca_1.append(rpca_performance[0])
                    rpca_2.append(rpca_performance[1])
                else:
                    rpca_1.append(0)
                    rpca_2.append(0)

                for _ in range(num_runs):
                    epca_pcs, _, _ = run_epca(
                        data=sp_data,
                        num_components=2,
                        num_bags=epca_dict.get("num_bags", 100),
                        bag_size=epca_dict.get("bag_size", 20),
                    )
                    epca_map = match_components(true_components, epca_pcs)
                    epca_performance = score_performance(
                        true_components, epca_pcs, epca_map
                    )

                    epca_1.append(epca_performance[0])
                    epca_2.append(epca_performance[1])

            pca_1_avgs.append(np.average(pca_1))
            pca_2_avgs.append(np.average(pca_2))

            epca_1_avgs.append(np.average(epca_1))
            epca_2_avgs.append(np.average(epca_2))

            rpca_1_avgs.append(np.average(rpca_1))
            rpca_2_avgs.append(np.average(rpca_2))

    np.save(root_filepath + "pca_1", pca_1_avgs)
    np.save(root_filepath + "pca_2", pca_2_avgs)
    np.save(root_filepath + "epca_1", epca_1_avgs)
    np.save(root_filepath + "epca_2", epca_2_avgs)
    np.save(root_filepath + "rpca_1", rpca_1_avgs)
    np.save(root_filepath + "rpca_2", rpca_2_avgs)

    return (
        pca_1_avgs,
        pca_2_avgs,
        epca_1_avgs,
        epca_2_avgs,
        rpca_1_avgs,
        rpca_2_avgs,
    )


def white_noise_comparison(
    data: np.ndarray,
    root_filepath: str,
    epca_dict: dict,
    divisors: Optional[List] = None,
    num_runs: int = 5,
    num_trials: int = 5,
    white_type: str = "normal",
    run_rpca_condition: bool = True,
):
    """
    Gather performance of all three methods on data corrupted with various amounts of white noise.

    Args:
        data (np.ndarray): Input data.
        prefix (str): filepath to which to save data
        epca_dict (dict): Arguments to use in EPCA.
        divsiors (List): Denominator of white noise scale.
        num_trials (int): Number of times to corrupt data at each scale.
        num_runs (int): Number of times to run each method on a corrupted data set.
        white_type: "normal" or "uniform" white noise
        run_rpca_condition (bool): Whether or not to run rpca (avoid timeout).
    """
    pca = PCA(n_components=2)
    pca.fit(data)
    sv1 = pca.singular_values_[0]
    true_components = pca.components_

    if divisors is None:
        divisors = [0.10, 1, 10, 100, 1000]

    # fig, axs = plt.subplots(2, 5, figsize=(25, 10))
    pca_1_avgs = []
    pca_2_avgs = []

    epca_1_avgs = []
    epca_2_avgs = []

    rpca_1_avgs = []
    rpca_2_avgs = []

    for divisor in divisors:
        epca_1 = []
        epca_2 = []

        pca_1 = []
        pca_2 = []

        rpca_1 = []
        rpca_2 = []

        for _ in range(num_trials):
            variance = sv1 / divisor

            if white_type == "uniform":
                white_data = add_white_noise(
                    data=data, variance=variance, noise_type="uniform"
                )

            elif white_type == "normal":
                white_data = add_white_noise(
                    data=data, variance=variance, noise_type="normal"
                )

            pca_pcs, _, _ = run_pca(data=white_data, num_components=2)
            pca_map = match_components(true_components, pca_pcs)
            pca_performance = score_performance(true_components, pca_pcs, pca_map)
            pca_1.append(pca_performance[0])
            pca_2.append(pca_performance[1])

            if run_rpca_condition is True:
                rpca_pcs, _, _ = run_rpca(data=white_data, num_components=2, reg_E=0.2)
                rpca_map = match_components(true_components, rpca_pcs)
                rpca_performance = score_performance(
                    true_components, rpca_pcs, rpca_map
                )

                rpca_1.append(rpca_performance[0])
                rpca_2.append(rpca_performance[1])
            else:
                rpca_1.append(0)
                rpca_2.append(0)

            for _ in range(num_runs):
                epca_pcs, _, _ = run_epca(
                    data=white_data,
                    num_components=2,
                    num_bags=epca_dict.get("num_bags", 100),
                    bag_size=epca_dict.get("bag_size", 20),
                )
                epca_map = match_components(true_components, epca_pcs)
                epca_performance = score_performance(
                    true_components, epca_pcs, epca_map
                )

                epca_1.append(epca_performance[0])
                epca_2.append(epca_performance[1])

        pca_1_avgs.append(np.average(pca_1))
        pca_2_avgs.append(np.average(pca_2))

        epca_1_avgs.append(np.average(epca_1))
        epca_2_avgs.append(np.average(epca_2))

        rpca_1_avgs.append(np.average(rpca_1))
        rpca_2_avgs.append(np.average(rpca_2))

    np.save(root_filepath + "pca_1", pca_1_avgs)
    np.save(root_filepath + "pca_2", pca_2_avgs)
    np.save(root_filepath + "epca_1", epca_1_avgs)
    np.save(root_filepath + "epca_2", epca_2_avgs)
    np.save(root_filepath + "rpca_1", rpca_1_avgs)
    np.save(root_filepath + "rpca_2", rpca_2_avgs)

    return (
        pca_1_avgs,
        pca_2_avgs,
        epca_1_avgs,
        epca_2_avgs,
        rpca_1_avgs,
        rpca_2_avgs,
    )


def load_comparison_data(root_filenames: str):
    """
    Load previously saved comparison data from various datasets.
    Return average performance across all datasets.

    Args:
        prefixes (str): Filepath where comparison data is saved.
    """
    loaded_example = np.load("saved_data/" + root_filenames[0] + "pca_1.npy")
    num_entries = len(loaded_example)

    pca_1 = np.zeros(num_entries)
    pca_2 = np.zeros(num_entries)
    epca_1 = np.zeros(num_entries)
    epca_2 = np.zeros(num_entries)
    rpca_1 = np.zeros(num_entries)
    rpca_2 = np.zeros(num_entries)

    for root_filename in root_filenames:
        loaded_pca_1 = np.load("saved_data/" + root_filename + "pca_1.npy")
        loaded_pca_2 = np.load("saved_data/" + root_filename + "pca_2.npy")
        loaded_epca_1 = np.load("saved_data/" + root_filename + "epca_1.npy")
        loaded_epca_2 = np.load("saved_data/" + root_filename + "epca_2.npy")
        loaded_rpca_1 = np.load("saved_data/" + root_filename + "rpca_1.npy")
        loaded_rpca_2 = np.load("saved_data/" + root_filename + "rpca_2.npy")

        pca_1 += np.array(loaded_pca_1)
        pca_2 += np.array(loaded_pca_2)
        epca_1 += np.array(loaded_epca_1)
        epca_2 += np.array(loaded_epca_2)
        rpca_1 += np.array(loaded_rpca_1)
        rpca_2 += np.array(loaded_rpca_2)

    pca_1 = pca_1 / len(root_filenames)
    pca_2 = pca_2 / len(root_filenames)
    epca_1 = epca_1 / len(root_filenames)
    epca_2 = epca_2 / len(root_filenames)
    rpca_1 = rpca_1 / len(root_filenames)
    rpca_2 = rpca_2 / len(root_filenames)

    return pca_1, pca_2, epca_1, epca_2, rpca_1, rpca_2


def run_varied_noise_experiments(
    sklearn: bool,
    wave: bool,
    mnist: bool,
    sst: bool,
    sparse_noise: bool,
    normal_white_noise: bool,
    uniform_white_noise: bool,
    outliers: bool,
    output_folder: str,
    **kwargs
):
    """
    Run varied noise experiments.

    Args:
        sparse noise (bool): Whether to run experiments with sparse noise.
        normal_white_noise (bool): Whether to run experiments with normal noise.
        uniform_white_noise (bool): Whether to run experiments with uniform noise.
        outliers (bool): Whether to run experiments with outliers.
        output_folder (str): Must include "/". Folder in which to place generated data.
    Kwargs:
        Data saved to the following filepaths: root_filepath + method + _component #
        sp_root_filepaths (dict): Root filepaths to use to save sparse noise data
        normal_white_filepaths (dict): Root filepaths to use to save normal white noise data
        uniform_white_filepaths (dict): Root filepaths to use to save uniform white noise data
        outlier_filepaths (dict): Root filepaths to use to save outlier data
    """

    data_dict = {}
    default_sp_root_filepaths = {}
    default_normal_white_filepaths = {}
    default_uniform_white_filepaths = {}
    default_outlier_filepaths = {}
    epca_dicts = {}
    run_rpca_conditions = {}

    def add_filepaths(dataset_name):
        """Generate new filepaths"""
        default_sp_root_filepaths[dataset_name] = (
            output_folder + "sp_noise_" + dataset_name + "_"
        )
        default_uniform_white_filepaths[dataset_name] = (
            output_folder + "u_white_noise_" + dataset_name + "_"
        )
        default_normal_white_filepaths[dataset_name] = (
            output_folder + "white_noise_" + dataset_name + "_"
        )
        default_outlier_filepaths[dataset_name] = output_folder + dataset_name + "_"

        return None

    if wave is True:
        print("Generating wave data.")

        def f(x, t):
            """Generate wave data."""
            return (1 - 0.5 * np.cos(2 * t)) * (1 / np.cosh(x)) + (
                1 - 0.5 * np.sin(2 * t)
            ) * (1 / np.cosh(x)) * np.tanh(x)

        dimension = 200
        x_space = np.linspace(-10, 10, dimension)
        t_space = np.linspace(0, 3000, 6000)
        X, T = np.meshgrid(x_space, t_space)
        wave_data = f(X, T)

        data_dict["wave"] = wave_data
        add_filepaths(dataset_name="wave")
        epca_dicts["wave_sp"] = {"num_samples": 100, "num_bags": 20}
        epca_dicts["wave_white"] = {"num_samples": 100, "num_bags": 20}
        epca_dicts["wave_uniform"] = {"num_samples": 100, "num_bags": 20}
        epca_dicts["wave_outlier"] = {"num_samples": 100, "num_bags": 5}
        run_rpca_conditions["wave"] = True

    if sklearn is True:
        print("Generating sklearn data.")
        data_dict["iris"] = datasets.load_iris().data
        data_dict["wine"] = datasets.load_wine().data
        data_dict["breast_cancer"] = datasets.load_breast_cancer().data

        for dataset_name in ["iris", "wine", "breast_cancer"]:
            add_filepaths(dataset_name=dataset_name)
            n, m = data_dict[dataset_name].shape
            epca_dicts[dataset_name + "_sp"] = {
                "num_samples": 100,
                "num_bags": min(5, n // 10),
            }
            epca_dicts[dataset_name + "_white"] = {
                "num_samples": 100,
                "num_bags": min(5, n // 10),
            }
            epca_dicts[dataset_name + "_uniform"] = {
                "num_samples": 100,
                "num_bags": min(5, n // 10),
            }
            epca_dicts[dataset_name + "_outlier"] = {
                "num_samples": 100,
                "num_bags": min(5, n // 10),
            }
            run_rpca_conditions[dataset_name] = True

    if mnist is True:
        print("Generating MNIST data.")
        mndata = MNIST("./data")
        images, labels = mndata.load_training()
        images = np.array(images)
        labels = np.array(labels)

        number_0 = np.where(labels == 0)[0]
        data_dict["mnist_0"] = images[number_0, :]
        add_filepaths(dataset_name="mnist_0")

        number_1 = np.where(labels == 1)[0]
        data_dict["mnist_1"] = images[number_1, :]
        add_filepaths(dataset_name="mnist_1")

        for dataset_name in ["mnist_0", "mnist_1"]:
            epca_dicts[dataset_name + "_sp"] = {"num_samples": 100, "num_bags": 100}
            epca_dicts[dataset_name + "_white"] = {"num_samples": 100, "num_bags": 100}
            epca_dicts[dataset_name + "_uniform"] = {
                "num_samples": 100,
                "num_bags": 100,
            }
            epca_dicts[dataset_name + "_outlier"] = {"num_samples": 100, "num_bags": 20}
            run_rpca_conditions[dataset_name] = True

    if sst is True:
        print("Generating sst data.")
        sst_dictionary = netcdf.NetCDFFile(
            "data/sst.wkmean.1990-present.nc", "r"
        ).variables
        # land-sea mask
        mask = netcdf.NetCDFFile("data/lsmask.nc", "r").variables["mask"].data[0, :, :]
        mask = mask.astype(float)
        mask[mask == 0] = np.nan
        data_dict["sst"] = sst_dictionary["sst"].data
        add_filepaths(dataset_name="sst")

        epca_dicts["sst_sp"] = {"num_samples": 100, "num_bags": 5}
        epca_dicts["sst_white"] = {"num_samples": 100, "num_bags": 5}
        epca_dicts["sst_uniform"] = {"num_samples": 100, "num_bags": 5}
        epca_dicts["sst_outlier"] = {"num_samples": 100, "num_bags": 5}
        run_rpca_conditions["sst"] = False

    #### Update any filepaths to save data to if the user has provided them ###
    sp_filepath_changes = kwargs.get("sp_root_filepaths", {})
    normal_white_filepath_changes = kwargs.get("normal_white_filepaths", {})
    uniform_white_filepath_changes = kwargs.get("uniform_white_filepaths", {})
    outlier_filepath_changes = kwargs.get("outlier_filepaths", {})

    for key in sp_filepath_changes.keys():
        default_sp_root_filepaths[key] = sp_filepath_changes[key]
    for key in normal_white_filepath_changes.keys():
        default_normal_white_filepaths[key] = normal_white_filepath_changes[key]
    for key in uniform_white_filepath_changes.keys():
        default_uniform_white_filepaths[key] = uniform_white_filepath_changes[key]
    for key in outlier_filepath_changes.keys():
        default_outlier_filepaths[key] = outlier_filepath_changes[key]
    ###########################################################################

    for index, key in enumerate(data_dict.keys()):
        if sparse_noise is True:
            print("Running sparse noise comparison on ", key)
            (
                _,
                _,
                _,
                _,
                _,
                _,
            ) = sparse_noise_comparison(
                data=data_dict[key],
                root_filepath=default_sp_root_filepaths[key],
                epca_dict=epca_dicts[key + "_sp"],
                num_trials=5,
                num_runs=5,
                run_rpca_condition=run_rpca_conditions[key],
            )

        if normal_white_noise is True:
            print("Running normal white noise comparison on ", key)

            (
                _,
                _,
                _,
                _,
                _,
                _,
            ) = white_noise_comparison(
                data=data_dict[key],
                root_filepath=default_normal_white_filepaths[key],
                epca_dict=epca_dicts[key + "_white"],
                num_trials=5,
                num_runs=5,
                white_type="normal",
                run_rpca_condition=run_rpca_conditions[key],
            )

        if uniform_white_noise is True:
            print("Running uniform white noise comparison on ", key)

            (
                _,
                _,
                _,
                _,
                _,
                _,
            ) = white_noise_comparison(
                data=data_dict[key],
                root_filepath=default_uniform_white_filepaths[key],
                epca_dict=epca_dicts[key + "_uniform"],
                num_trials=5,
                num_runs=5,
                white_type="uniform",
                run_rpca_condition=run_rpca_conditions[key],
            )

        if outliers is True:
            print("Running outlier comparison on ", key)

            (
                _,
                _,
                _,
                _,
                _,
                _,
            ) = outlier_comparison(
                data=data_dict[key],
                root_filepath=default_outlier_filepaths[key],
                epca_dict=epca_dicts[key + "_outlier"],
                num_trials=5,
                num_runs=5,
                run_rpca_condition=run_rpca_conditions[key],
            )

    return None
