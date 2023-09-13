import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from EPCA import EPCA
from helper_functions import (
    sparse_noise,
    match_components,
    run_epca,
    run_pca,
    run_rpca,
    score_performance,
)
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D
from typing import List


def create_noisy_datasets(
    data,
    sp_probability: float = 0.20,
    sparse_num: float = 2,
    uniform_white_variance: float = 2,
    normal_white_variance: float = 2,
    outlier_scale: float = 5,
    outlier_percentage: float = 0.10,
):
    """Create all noisy datasets: salt and pepper, sparse, white, and outlier.

    Args:
        sp_probability (float): Probability of the noise
        sparse_num (float): Number to replace data entries
        uniform_white_variance (float): Variance for uniform white noise
        normal_white_variance (float): Variance for normal white noise
        outlier_scale (float): Scale of the outliers.
        outlier_percentage (float): Percent of outliers to add to the data.
    """
    data_samples, data_dimension = data.shape
    normal_white_data = data + np.random.normal(
        0, normal_white_variance, size=data.shape
    )
    uniform_white_data = data + uniform_white_variance * np.random.random(
        size=data.shape
    )
    sp_data = sparse_noise(data, sp_probability, num=sparse_num)

    ind = np.random.choice(
        data_samples, int(np.round(data_samples * outlier_percentage)), replace=False
    )
    outlier_data = data.copy()
    outlier_data[ind] = outlier_data[ind] * outlier_scale

    return normal_white_data, uniform_white_data, sp_data, outlier_data


def plot_epca_trials(
    epca_args: dict,
    num_components: int,
    num_trials: int,
    input_data: np.ndarray,
    true_components: np.ndarray,
    filename: str,
    filename_2: str,
    figsize=(15, 10),
    show_outliers=True,
):
    """
    Plot num_components true vs predicted components for num_trials of EPCA.

    Plot boxplots of explained variance for each of the trials.

    Args:
        epca_args (dict): Dictionary of arguments for EPCA.
        num_components (int): Number of components for EPCA.
        num_trials (int): Number of trials to run.
        input_data (int): Data on which to run EPCA.
        true_components (np.ndarray): True components.
        filename (str): Path at which to save image.
        figsize (Optional[Tuple]): size of figure
        show_outliers (bool): Whether or not to show outliers in the boxplots
            of explained variance.

    """
    fig, axs = plt.subplots(
        num_components, num_trials, figsize=figsize, sharex=True, sharey=True
    )

    pc_data = {"run": [], "value": [], "pc": []}

    for trial in range(num_trials):
        epca = EPCA(
            num_components=num_components,
            num_bags=epca_args.get("num_bags", 100),
            bag_size=epca_args.get("bag_size", 10),
            smoothing=epca_args.get("smoothing", False),
            window_length=epca_args.get("window_length", 31),
            poly_order=epca_args.get("poly_order", 3),
        )
        _, data_dimension = input_data.shape
        epca.run_epca(input_data)
        pc_map = match_components(true_components, epca.centers)

        for index in range(num_components):
            ind = np.where(epca.labels == pc_map[index])
            # for vector in epca.signed_vectors[ind]:
            #     axs[index, trial].plot(vector, c="gray")
            axs[index, trial].plot(true_components[index], c="r")
            axs[index, trial].plot(epca.centers[pc_map[index]], "--b")

            LL = np.percentile(epca.signed_vectors[ind], 5, axis=0)
            UL = np.percentile(epca.signed_vectors[ind], 95, axis=0)
            axs[index, trial].fill_between(
                np.arange(data_dimension), LL, UL, color="grey"
            )

            for value in epca.explained_variance[ind]:
                pc_data["run"].append("Run " + str(trial + 1))
                pc_data["value"].append(value)
                pc_data["pc"].append("PC" + str(index + 1))

    axs[0, 0].set_title("Run 1", fontsize=20)
    axs[0, 1].set_title("Run 2", fontsize=20)
    axs[0, 2].set_title("Run 3", fontsize=20)
    fig.suptitle("Runs of EPCA", fontsize=20)

    axs[0, 0].set_ylabel("PC 1", fontsize=20)
    axs[1, 0].set_ylabel("PC 2", fontsize=20)

    axs[0, 0].tick_params(axis="y", which="major", labelsize=16)
    axs[1, 0].tick_params(axis="y", which="major", labelsize=16)
    axs[1, 0].tick_params(axis="x", which="major", labelsize=16)
    axs[1, 1].tick_params(axis="x", which="major", labelsize=16)
    axs[1, 2].tick_params(axis="x", which="major", labelsize=16)

    legend_elements = [
        Line2D([0], [0], color="r", label="True Component"),
        Line2D([0], [0], color="b", label="Avg. Predicted Component", linestyle="--"),
        mpatches.Patch(color="grey", label="95% CI")
        # Line2D([0], [0], color="grey", label="Sampled Component"),
    ]
    axs[0, 2].legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=16,
        framealpha=1,
    )

    plt.savefig(filename, format="pdf", bbox_inches="tight")

    plt.figure(figsize=(10, 5))
    fig2 = sns.boxplot(
        data=pc_data, x="run", y="value", hue="pc", showfliers=show_outliers
    )
    fig2.set_title("Explained Variance of PCs per Run", size=16)
    fig2.set_ylabel("Explained Variance", size=16)
    fig2.tick_params(labelsize=16)
    fig2.legend(fontsize=16)

    plt.savefig(filename_2, format="pdf", bbox_inches="tight")

    return fig, axs, fig2


def plot_compare_methods(
    num_components: int,
    epca_args: dict,
    pca_args: dict,
    rpca_args: dict,
    input_data: np.ndarray,
    true_components: np.ndarray,
):
    """
    Plot top two components of epca, rpca, and pca vs true components.

    Args:
        num_components (int): Num components to calculate.
        epca_args (dict): Dictionary of arguments for EPCA.
        pca_args (dict): Dictionary of arguments for PCA.
        rpca_args (dict): Dictionary of arguments for rpca.
        input_data (dict): Data on which to run PCA.
        true_components (np.ndarray): True components.
    """
    if epca_args is not None:
        ####### EPCA ###############
        epca_pred_components, _, _ = run_epca(
            images=input_data,
            num_components=num_components,
            num_bags=epca_args.get("num_samples", 100),
            bag_size=epca_args.get("sample_size", 10),
            smoothing=epca_args.get("smoothing", False),
            window_length=epca_args.get("window_length", 31),
            poly_order=epca_args.get("poly_order", 3),
        )
        epca_map = match_components(true_components, epca_pred_components)

    ########## PCA ##########
    if pca_args is not None:
        final_pca, _, _ = run_pca(
            images=input_data,
            num_components=num_components,
            smoothing=pca_args.get("smoothing", False),
            window_length=pca_args.get("window_length", 51),
            poly_order=pca_args.get("poly_order", 3),
        )
        pca_map = match_components(true_components, final_pca)

    ######## RPCA ##############
    if rpca_args is not None:
        final_rpca, _, _ = run_rpca(
            images=input_data,
            num_components=num_components,
            reg_E=rpca_args.get("reg_E", 1.0),
            reg_J=rpca_args.get("reg_J", 1.0),
            learning_rate=rpca_args.get("learning_rate", 1.1),
            n_iter_max=rpca_args.get("n_iter_max", 50),
            smoothing=rpca_args.get("smoothing", False),
            window_legnth=rpca_args.get("window_length", 51),
            poly_order=rpca_args.get("poly_order", 3),
        )
        rpca_map = match_components(true_components, final_rpca)

    fig, axs = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

    for index in range(num_components):
        axs[index].plot(true_components[index], c="r", label="True")
        if epca_args is not None:
            axs[index].plot(
                epca_pred_components[epca_map[index]],
                c="b",
                linestyle="dashed",
                label="EPCA",
            )
        if pca_args is not None:
            axs[index].plot(
                final_pca[pca_map[index]],
                c="g",
                label="PCA",
                linestyle="dotted",
                linewidth=3,
            )
        if rpca_args is not None:
            axs[index].plot(
                final_rpca[rpca_map[index]], c="k", linestyle="dashdot", label="RPCA"
            )
        axs[index].legend()
        axs[1].yaxis.set_tick_params(labelbottom=True)

    return fig


def outlier_comparison(
    data: np.ndaray,
    prefix: str,
    epca_num_bags: int = 100,
    epca_bag_size: int = 5,
    outlier_fractions: List = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25],
    outlier_scales: List = [2, 5, 10, 20, 100],
    num_runs: int = 5,
    num_trials: int = 5,
    run_rpca_condition: bool = True,
):
    """
    Compare performance of all three methods on data corrupted with various amounts of outliers.

    Args:
        data (np.ndarray): Input data.
        prefix (str): filepath to which to save data
        epca_num_bags (int): Number of bags to use in EPCA.
        epca_bag_size (int): Size of bags to use in EPCA.
        outlier_fraction (List): Fraction of outliers to add to data.
        outlier_scale (List): Scale of outliers to add to data.
        num_trials (int): Number of times to corrupt data at each scale.
        num_runs (int): Number of times to run each method on a corrupted data set.
        run_rpca_condition (bool): Whether or not to run rpca (avoid timeout).
    """
    pca = PCA(n_components=2)
    pca.fit(data)
    true_components = pca.components_

    data_samples, data_dimension = data.shape

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

            for trial in range(num_trials):
                ind = np.random.choice(
                    data_samples,
                    int(np.round(data_samples * outlier_fraction)),
                    replace=False,
                )
                outlier_data = data.copy()
                outlier_data[ind] = outlier_data[ind] * outlier_scale
                outlier_data = outlier_data.reshape((data_samples, -1))

                pca_pcs, pca_svs, _ = run_pca(images=outlier_data, num_components=2)
                pca_map = match_components(true_components, pca_pcs)
                pca_performance = score_performance(true_components, pca_pcs, pca_map)
                pca_1.append(pca_performance[0])
                pca_2.append(pca_performance[1])

                if run_rpca_condition is True:
                    rpca_pcs, rpca_svs, _ = run_rpca(
                        images=outlier_data, num_components=2, reg_E=0.2
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

                for run in range(num_runs):
                    epca_pcs, epca_svs, _ = run_epca(
                        images=outlier_data,
                        num_components=2,
                        num_bags=epca_num_bags,
                        bag_size=epca_bag_size,
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

    np.save(prefix + "pca_1", pca_1_avgs)
    np.save(prefix + "pca_2", pca_2_avgs)
    np.save(prefix + "epca_1", epca_1_avgs)
    np.save(prefix + "epca_2", epca_2_avgs)
    np.save(prefix + "rpca_1", rpca_1_avgs)
    np.save(prefix + "rpca_2", rpca_2_avgs)

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
    prefix: str,
    epca_num_bags: int = 100,
    epca_bag_size: int = 20,
    divisors: List = [0.10, 1, 10, 100, 1000],
    num_runs: int = 5,
    num_trials: int = 5,
    white_type: str = "normal",
    run_rpca_condition: bool = True,
):
    """
    Compare performance of all three methods on data corrupted with various amounts of white noise.

    Args:
        data (np.ndarray): Input data.
        prefix (str): filepath to which to save data
        epca_num_bags (int): Number of bags to use in EPCA.
        epca_bag_size (int): Size of bags to use in EPCA.
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

    data_samples, data_dimension = data.shape

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

        for trial in range(num_trials):
            variance = sv1 / divisor

            if white_type == "uniform":
                white_data = data + variance * np.random.random(size=data.shape)
                white_data = white_data.reshape((data_samples, -1))
            elif white_type == "normal":
                white_data = data + np.random.normal(0, variance, size=data.shape)
                white_data = white_data.reshape((data_samples, -1))

            pca_pcs, pca_svs, _ = run_pca(images=white_data, num_components=2)
            pca_map = match_components(true_components, pca_pcs)
            pca_performance = score_performance(true_components, pca_pcs, pca_map)
            pca_1.append(pca_performance[0])
            pca_2.append(pca_performance[1])

            if run_rpca_condition is True:
                rpca_pcs, _, _ = run_rpca(
                    images=white_data, num_components=2, reg_E=0.2
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

            for run in range(num_runs):
                epca_pcs, _, _ = run_epca(
                    images=white_data,
                    num_components=2,
                    num_bags=epca_num_bags,
                    bag_size=epca_bag_size,
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

    np.save(prefix + "pca_1", pca_1_avgs)
    np.save(prefix + "pca_2", pca_2_avgs)
    np.save(prefix + "epca_1", epca_1_avgs)
    np.save(prefix + "epca_2", epca_2_avgs)
    np.save(prefix + "rpca_1", rpca_1_avgs)
    np.save(prefix + "rpca_2", rpca_2_avgs)

    return (
        pca_1_avgs,
        pca_2_avgs,
        epca_1_avgs,
        epca_2_avgs,
        rpca_1_avgs,
        rpca_2_avgs,
    )


def load_comparison_data(prefixes: str, n: int):
    """
    Load previously saved comparison data.

    Args:
        prefixes (str): Filepath where comparison data is saved.
        n (int)
    """
    pca_1 = np.zeros(n)
    pca_2 = np.zeros(n)
    epca_1 = np.zeros(n)
    epca_2 = np.zeros(n)
    rpca_1 = np.zeros(n)
    rpca_2 = np.zeros(n)

    for prefix in prefixes:
        loaded_pca_1 = np.load(prefix + "pca_1.npy")
        loaded_pca_2 = np.load(prefix + "pca_2.npy")
        loaded_epca_1 = np.load(prefix + "epca_1.npy")
        loaded_epca_2 = np.load(prefix + "epca_2.npy")
        loaded_rpca_1 = np.load(prefix + "rpca_1.npy")
        loaded_rpca_2 = np.load(prefix + "rpca_2.npy")

        pca_1 += np.array(loaded_pca_1)
        pca_2 += np.array(loaded_pca_2)
        epca_1 += np.array(loaded_epca_1)
        epca_2 += np.array(loaded_epca_2)
        rpca_1 += np.array(loaded_rpca_1)
        rpca_2 += np.array(loaded_rpca_2)

    pca_1 = pca_1 / len(prefixes)
    pca_2 = pca_2 / len(prefixes)
    epca_1 = epca_1 / len(prefixes)
    epca_2 = epca_2 / len(prefixes)
    rpca_1 = rpca_1 / len(prefixes)
    rpca_2 = rpca_2 / len(prefixes)

    return pca_1, pca_2, epca_1, epca_2, rpca_1, rpca_2


def sparse_noise_comparison(
    data: np.ndarray,
    prefix: str,
    epca_num_bags: int = 100,
    epca_bag_size=20,
    sparse_noise_probs: List = [0.01, 0.05, 0.10],
    sparse_noise_scales: List = [0, 2, 5, 10],
    num_runs: int = 5,
    num_trials: int = 5,
    run_rpca_condition: bool = True,
):
    """
     Compare performance of all three methods on data corrupted with various amounts of sparse noise.

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

    data_samples, data_dimension = data.shape

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

            for trial in range(num_trials):
                sp_data = sparse_noise(
                    image=data, prob=sp_prob, num=np.max(data) * sp_scale
                )
                sp_data = sp_data.reshape((data_samples, -1))

                pca_pcs, pca_svs, _ = run_pca(images=sp_data, num_components=2)
                pca_map = match_components(true_components, pca_pcs)
                pca_performance = score_performance(true_components, pca_pcs, pca_map)
                pca_1.append(pca_performance[0])
                pca_2.append(pca_performance[1])

                if run_rpca_condition is True:
                    rpca_pcs, rpca_svs, _ = run_rpca(
                        images=sp_data, num_components=2, reg_E=0.2
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

                for run in range(num_runs):
                    epca_pcs, epca_svs, _ = run_epca(
                        images=sp_data,
                        num_components=2,
                        num_bags=epca_num_bags,
                        bag_size=epca_bag_size,
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

    np.save(prefix + "pca_1", pca_1_avgs)
    np.save(prefix + "pca_2", pca_2_avgs)
    np.save(prefix + "epca_1", epca_1_avgs)
    np.save(prefix + "epca_2", epca_2_avgs)
    np.save(prefix + "rpca_1", rpca_1_avgs)
    np.save(prefix + "rpca_2", rpca_2_avgs)

    return (
        pca_1_avgs,
        pca_2_avgs,
        epca_1_avgs,
        epca_2_avgs,
        rpca_1_avgs,
        rpca_2_avgs,
    )
