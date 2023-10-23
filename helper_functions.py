"""Functions for running experiments."""
import random
import time
from typing import TypedDict, List, Optional
from multiprocessing import Pool, TimeoutError
from tensorly.decomposition import robust_pca
from sklearn.decomposition import PCA
import numpy as np
from EPCA.EPCA import EPCA


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
    data_samples, _ = data.shape
    normal_white_data = data + np.random.normal(
        0, normal_white_variance, size=data.shape
    )
    uniform_white_data = data + uniform_white_variance * np.random.random(
        size=data.shape
    )
    sp_data = add_sparse_noise(data, sp_probability, num=sparse_num)

    ind = np.random.choice(
        data_samples, int(np.round(data_samples * outlier_percentage)), replace=False
    )
    outlier_data = data.copy()
    outlier_data[ind] = outlier_data[ind] * outlier_scale

    return normal_white_data, uniform_white_data, sp_data, outlier_data


def add_sparse_noise(data: np.ndarray, prob: float, num: float):
    """
    Add salt and pepper noise to data
    Args:
        data (np.ndarray): Data
        prob (float): Probability of the noise
        num (float): Number to replace data entries
    Returns:
        output (np.ndarray): Data with added noise
    """

    assert 0 <= prob <= 1, "Probability must be in [0,1]."

    output = np.zeros(data.shape)
    thres = prob
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            rdn = random.random()
            if rdn < thres:
                output[i][j] = num
            else:
                output[i][j] = data[i][j]
    return output


def add_outliers(data: np.ndarray, outlier_fraction: float, outlier_scale: float):
    """
    Add outliers to data.

    data (np.ndarray): Data
    outlier_fraction (float): Fraction of rows to corrupt.
    outlier_scale (float): Scale of outliers.

    Returns:
        outlier_data (np.ndarray): Data with added outliers.
    """

    data_shape = data.shape

    ind = np.random.choice(
        data_shape, int(np.round(data_shape * outlier_fraction)), replace=False
    )
    outlier_data = data.copy()
    outlier_data[ind] = outlier_data[ind] * outlier_scale
    outlier_data = outlier_data.reshape((data_shape, -1))

    return outlier_data


def add_white_noise(data: np.ndarray, variance: float, type: str):
    """
    Add either Gaussian or uniform white noise to data.

    Args:
        data (np.ndarray): Data.
        variance (float): Noise is sampled from distribution with mean 0 and this variance.
        type (str): Type of white noise: "uniform" or "gaussian".
    Returns:
        white_noise_data (np.ndarray): Data with added white noise.

    """

    data_shape = data.shape

    if type == "uniform":
        white_noise_data = data + variance * np.random.random(size=data.shape)
        white_noise_data = white_noise_data.reshape((data_shape, -1))

    # add normal white noise
    elif type == "normal":
        white_noise_data = data + np.random.normal(0, variance, size=data.shape)
        white_noise_data = white_noise_data.reshape((data_shape, -1))

    return white_noise_data


def match_components(true_components: np.ndarray, pred_components: np.ndarray) -> dict:
    """
    Match ture and predicted components to one another.

    Args:
        true_components (np.ndarray): The true components.
        pred_components (np.ndarray): The predicted components.
    Returns:
        pc_map (dict): Dictionary with keys corresponding to indices of
        true components and values corresponding to indices
        of predicted components
    """
    predicted = []

    for _, component in enumerate(true_components):
        difference = np.argsort(np.linalg.norm(component - pred_components, 2, axis=1))
        for _, diff in enumerate(difference):
            if diff not in predicted:
                predicted.append(diff)
                break

    pc_map = dict(zip(np.arange(len(true_components)), predicted))
    return pc_map


def run_pca(images: np.ndarray, num_components: int):
    """
    Run standard princpal component analysis on given data.

    Args:
        images (np.ndarray): Data.
        num_components (int): Number of components to compare against
    Returns
        pca_performance (List[float]): Accuracy of predicted PCA and true PCA components
        runtime (float): Runtime of PCA
    """
    start_time = time.time()

    assert len(images.shape) == 2, "Data must be 2D"
    pca = PCA(n_components=num_components)
    pca.fit(images)

    runtime = time.time() - start_time

    pred_components = pca.components_
    pred_svs = pca.explained_variance_
    final_components = np.vstack((pred_components, pred_components * -1))
    final_svs = np.hstack((pred_svs, pred_svs))
    return final_components, final_svs, runtime


def run_epca(
    images: np.ndarray,
    num_components: int,
    num_bags: int = 100,
    bag_size: int = 10,
):
    """
    Run ensemble PCA.

    Args:
        images (np.ndarray): Data.
        num_samples (int): Number of data bags to create.
        sample_size (int): Number of data samples in each of the bags.
    Returns:
        rpca_performance (List[float]): Accuracy of predicted EPCA and true PCA components
        runtime (float): Runtime of EPCA
    """
    start_time = time.time()
    epca = EPCA(
        num_components=num_components,
        num_bags=num_bags,
        bag_size=bag_size,
    )
    epca.run_epca(images)

    runtime = time.time() - start_time

    return epca.centers, epca.avg_explained_variance, np.round(runtime, 3)


def run_rpca(images: np.ndarray, num_components: int, **kwargs):
    """
    Run robust PCA.

    Args:
        images (np.ndarray): Data.
    Returns:
        rpca_performance (List[float]): Accuracy of predicted RPCA and true PCA components.
        rpca_runtime (float): Runtime of RPCA
    """
    start_time = time.time()
    low_rank_part, _ = robust_pca(
        X=images.astype(float),
        reg_E=kwargs.pop("reg_E", 1.0),
        reg_J=kwargs.pop("reg_J", 1.0),
        learning_rate=kwargs.pop("learning_rate", 1.1),
        n_iter_max=kwargs.pop("n_iter_max", 50),
    )

    runtime = time.time() - start_time

    rpca = PCA(n_components=num_components)
    rpca.fit(low_rank_part)

    pred_components = rpca.components_
    pred_svs = rpca.explained_variance_
    final_components = np.vstack((pred_components, pred_components * -1))
    final_svs = np.hstack((pred_svs, pred_svs))
    return final_components, final_svs, np.round(runtime, 3)


def score_performance(
    true_components: np.ndarray,
    pred_components: np.ndarray,
    pc_map: TypedDict,
) -> List[float]:
    """
    Match the true component to the best fit predicted component.
    Evaluate the difference between the two.

    Args:
        true_components (np.ndarray): The true PCA components to be matched.
        pred_components (np.ndarray): Predicted PCA components.
        true_svs (np.ndarray): True singular values
        pred_svs (np.ndarray): Pred singular values

    Returns:
        diff (List[float]): List containing the minimum difference between each
        matched true and predicted PCA component.
    """
    component_diff = []

    for index, true_component in enumerate(true_components):
        difference = np.linalg.norm(
            true_component - pred_components[pc_map[index]], 2
        ) / np.linalg.norm(true_component, 2)
        component_diff.append(difference * 100)

    return component_diff


def run_all_pca_methods(
    true_components: np.ndarray,
    data: np.ndarray,
    timeout: float,
    num_components: int,
    pca_args: TypedDict,
    epca_args: TypedDict,
    rpca_args: TypedDict,
):
    """
    Run traditional PCA, ensemble PCA, and robust PCA.

    Args:
        true_components (np.ndarray): The true PCA components
        data (np.ndarray)
        timeout (float): Number of seconds after which to timeout the function run
        num_components (int): The number of components to search for
        epca_args (TypedDict): Dictionary containing necessary arguments for ensemble PCA
        rpca_args (TypedDict): Dictionary containing necessary arguments for robust PCA
    Returns:
        pca_runtime (float): Runtime of PCA
        epca_runtime (float): Runtime of EPCA
        rpca_runtime (float): Runtime of RPCA
        pca_performance (List[float]): List containing difference between true
            components and those predicted by PCA
        epca_performance (List[float]): List containing difference between true
            components and those predicted by EPCA
        rpca_performance (List[float]): List containing difference between true
            components and those predicted by RPCA

    """

    pool = Pool(1)
    try:
        res1 = pool.apply_async(
            run_pca,
            kwds={
                "images": data,
                "num_components": num_components,
            },
        )
        pca_components, pca_svs, pca_runtime = res1.get(timeout=timeout)
        pca_map = match_components(true_components, pca_components)
        pca_performance = score_performance(true_components, pca_components, pca_map)
        pca_final_svs = [pca_svs[pca_map[i]] for i in range(num_components)]

    except TimeoutError:
        pool.terminate()
        print("PCA timed out")
        pca_performance = "NaN"
        pca_runtime = "NaN"
        pca_final_svs = "NaN"

    pool = Pool(1)
    try:
        res2 = pool.apply_async(
            run_epca,
            kwds={
                "images": data,
                "num_components": num_components,
                "num_bags": epca_args.get("num_bags", 100),
                "bag_size": epca_args.get("bag_size", 10),
            },
        )
        epca_components, epca_svs, epca_runtime = res2.get(timeout=timeout)
        epca_map = match_components(true_components, epca_components)
        epca_performance = score_performance(true_components, epca_components, epca_map)
        epca_final_svs = [epca_svs[epca_map[i]] for i in range(num_components)]

    except TimeoutError:
        pool.terminate()
        print("EPCA timed out")
        epca_performance = "NaN"
        epca_runtime = "NaN"
        epca_final_svs = "NaN"

    pool = Pool(1)
    try:
        res3 = pool.apply_async(
            run_rpca,
            kwds={
                "images": data,
                "num_components": num_components,
                "reg_E": rpca_args.get("reg_E", 1.0),
                "reg_J": rpca_args.get("reg_J", 1.0),
                "learning_rate": rpca_args.get("learning_rate", 1.1),
                "n_iter_max": rpca_args.get("n_iter_max", 50),
            },
        )
        rpca_components, rpca_svs, rpca_runtime = res3.get(timeout=timeout)
        rpca_map = match_components(true_components, rpca_components)
        rpca_performance = score_performance(true_components, rpca_components, rpca_map)
        rpca_final_svs = [rpca_svs[rpca_map[i]] for i in range(num_components)]

    except TimeoutError:
        pool.terminate()
        print("RPCA timed out")
        rpca_performance = "NaN"
        rpca_runtime = "NaN"
        rpca_final_svs = "NaN"

    return (
        pca_runtime,
        epca_runtime,
        rpca_runtime,
        pca_performance,
        epca_performance,
        rpca_performance,
        pca_final_svs,
        epca_final_svs,
        rpca_final_svs,
    )


def write_to_file(
    original_data: np.ndarray,
    num_components: int,
    timeout: float,
    pca_args: TypedDict,
    epca_args: TypedDict,
    rpca_args: TypedDict,
    filename: str,
    sp_probability: float = 0.01,
    outlier_scale: float = 10,
    outlier_fraction: float = 0.10,
    variance_divisor: float = 100,
    sparse: bool = True,
    uniform: bool = True,
    normal: bool = True,
    outliers: bool = True,
):
    """
    Write results of running different versions of PCA on corrupted data
    (salt and pepper noise and white noise) to a file.

    Args:
        original_data (np.ndarray): Data to add noise to and run analysis on
        num_components (int): Number of components to analyze
        timeout (float): Number of seconds after which to timeout a function run
        pca_args (TypedDict): Optional argumnets for PCA
        epca_args (TypedDict): Optional arguments for EPCA
        rpca_args (TypedDict): Optional arguments for RPCA
        filename (str): Filename to which to write the output
        sp_probability (float): Probability of sparse noise
        uniform_white_variance (float): Variance for uniform white noise
        normal_white_variance (float): Variance for normal white noise
        outlier_scale (float): Scale of the outliers.
        outlier_fraction (float): Fraction of outliers to add to the data.
    """
    data_samples = original_data.shape[0]

    file = open(filename, "w")

    pca = PCA(n_components=num_components)
    pca.fit(original_data.reshape((data_samples, -1)))
    sv_1 = pca.singular_values_[0]

    # Find how many components capture at least 90% of the information
    percent = np.cumsum(pca.explained_variance_[:num_components])[-1]
    print(
        np.round(percent, 3),
        " percent of the data is explained by",
        num_components,
        " components",
    )

    # Add sparse salt and pepper noise
    data_types = ["original"]
    data_list = [original_data.reshape((data_samples, -1))]

    if sparse is True:
        print("Creating sparse salt and pepper noise")
        sp_data = add_sparse_noise(
            original_data, prob=sp_probability, num=np.max(original_data) * 2
        )

        sp_data = sp_data.reshape((data_samples, -1))
        print("Sparse noise created")
        data_types.append("sparse")
        data_list.append(sp_data)

    # add uniform white noise
    if uniform is True:
        uniform_white_variance = sv_1 / variance_divisor
        print("Creating uniform white noise")
        uniform_white_data = add_white_noise(
            data=original_data, variance=uniform_white_variance, type="uniform"
        )
        print("Created uniform white noise")
        data_types.append("uniform white")
        data_list.append(uniform_white_data)

    # add normal white noise
    if normal is True:
        normal_white_variance = sv_1 / variance_divisor
        print("Creating normal white noise")
        normal_white_data = add_white_noise(
            data=original_data, variance=normal_white_variance, type="normal"
        )
        print("Created normal white noise")
        data_types.append("normal white")
        data_list.append(normal_white_data)

    if outliers is True:
        # Add outliers
        print("Adding ", outlier_fraction, "percent outliers")

        outlier_data = add_outliers(
            data=original_data,
            outlier_fraction=outlier_fraction,
            outlier_scale=outlier_scale,
        )
        data_types.append("outliers")
        data_list.append(outlier_data)

    file.write(
        "data; pca_runtime; epca_runtime; rpca_runtime; pca_performance; epca_performance; rpca_performance; pca_performance_svs; epca_performance_svs; rpca_performance_svs\n"
    )

    for index, data in enumerate(data_list):
        print("Running methods on ", data_types[index], " data.")
        (
            pca_runtime,
            epca_runtime,
            rpca_runtime,
            pca_performance,
            epca_performance,
            rpca_performance,
            pca_final_svs,
            epca_final_svs,
            rpca_final_svs,
        ) = run_all_pca_methods(
            true_components=pca.components_,
            data=data,
            num_components=num_components,
            timeout=timeout,
            pca_args=pca_args,
            epca_args=epca_args[data_types[index]],
            rpca_args=rpca_args,
        )

        file.write(
            "%s; %s; %s; %s; %s; %s; %s; %s; %s; %s \n"
            % (
                data_types[index],
                pca_runtime,
                epca_runtime,
                rpca_runtime,
                pca_performance,
                epca_performance,
                rpca_performance,
                pca_final_svs,
                epca_final_svs,
                rpca_final_svs,
            )
        )

    file.close()
    return None
