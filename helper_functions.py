import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from typing import Tuple, Optional, TypedDict, List
from RPCA_3 import EPCA
import random
import time
from tensorly.decomposition import robust_pca
from multiprocessing import Pool, TimeoutError
from scipy.signal import savgol_filter


def plot_components(components: np.ndarray, imsize: Optional[Tuple] = None) -> None:
    """
    Visualize PCA components:

    Args:
    components (np.ndarray): The PCA components to visualize
    imsize(Optional[Tuple]): The shape of the component to be plotted. If none, the component is 1D.
    """
    for component in components:
        if imsize is None:
            plt.figure()
            plt.plot(component)
        else:
            plt.figure()
            plt.imshow(component.reshape(imsize))

    return None


def match_components(true_components: np.ndarray, pred_components: np.ndarray):
    predicted = []

    for true_index, component in enumerate(true_components):
        difference = np.argsort(np.linalg.norm(component - pred_components, 2, axis=1))
        for i in range(len(difference)):
            if difference[i] not in predicted:
                predicted.append(difference[i])
                break

    pc_map = dict(zip(np.arange(len(true_components)), predicted))
    return pc_map


def run_pca(images: np.ndarray, num_components: int, **kwargs):
    """
    Run standard princpal component analysis on given data.

    Args:
        images (np.ndarray): Data.
        num_components (int): Number of components to compare against
    Kwargs:
        smoothing (bool): Whether or not to smooth the output
        window_length (int):
        poly_order (int):
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

    smoothing = kwargs.pop("smoothing", False)
    if smoothing is True:
        window_length = kwargs.pop("window_length", 51)
        poly_order = kwargs.pop("poly_order", 3)

        assert window_length % 2 == 1, "Window length must be odd"

        pred_components = savgol_filter(
            pca.components_,
            window_length=window_length,
            polyorder=poly_order,
        )

    final_components = np.vstack((pred_components, pred_components * -1))
    final_svs = np.hstack((pred_svs, pred_svs))
    return final_components, final_svs, runtime


def run_epca(
    images: np.ndarray,
    n_components: int,
    num_samples: int = 100,
    sample_size: int = 10,
    smoothing: bool = False,
    window_length: int = 51,
    poly_order: int = 3,
):
    """
    Run ensemble PCA.

    Args:
        images (np.ndarray): Data.
        num_samples (int): Number of data bags to create.
        sample_size (int): Number of data samples in each of the bags.
        smoothing (int): Whether or not to smooth the components.
        window_length (int): Must be odd. Window length for smoothing.
        poly_order (int): Polynomial order for smoothing.
    Returns:
        rpca_performance (List[float]): Accuracy of predicted EPCA and true PCA components
        runtime (float): Runtime of EPCA
    """
    start_time = time.time()
    epca = EPCA(
        num_components=n_components,
        num_samples=num_samples,
        sample_size=sample_size,
        smoothing=smoothing,
        window_length=window_length,
        poly_order=poly_order,
    )
    epca.run_EPCA(images)

    runtime = time.time() - start_time

    return epca.centers, epca.avg_singular_values, np.round(runtime, 3)


def run_rpca(images: np.ndarray, num_components: int, **kwargs):
    """
    Run robust PCA.

    Args:
        images (np.ndarray): Data.
    Kwargs:
        smoothing (bool):
        window_length (int):
        poly_order (int):
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

    smoothing = kwargs.pop("smoothing", False)
    if smoothing is True:
        window_length = kwargs.pop("window_length", 51)
        poly_order = kwargs.pop("polyorder", 3)

        assert window_length % 2 == 1, "Window length must be odd"

        pred_components = savgol_filter(
            rpca.components_,
            window_length=window_length,
            polyorder=poly_order,
        )

    final_components = np.vstack((pred_components, pred_components * -1))
    final_svs = np.hstack((pred_svs, pred_svs))
    return final_components, final_svs, np.round(runtime, 3)


def sp_noise(image: np.ndarray, prob: float) -> np.ndarray:
    """
    Add salt and pepper noise to data
    Args:
        image (np.ndarray): Data
        prob (float): Probability of the noise
    Returns:
        output (np.ndarray): Data with added noise
    """

    assert 0 <= prob <= 1, "Probability must be in [0,1]."

    output = np.zeros(image.shape)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = -1
            elif rdn > thres:
                output[i][j] = 1
            else:
                output[i][j] = image[i][j]
    return output


def sparse_noise(image: np.ndarray, prob: float, num: float):
    """
    Add salt and pepper noise to data
    Args:
        image (np.ndarray): Data
        prob (float): Probability of the noise
    Returns:
        output (np.ndarray): Data with added noise
    """

    assert 0 <= prob <= 1, "Probability must be in [0,1]."

    output = np.zeros(image.shape)
    thres = prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < thres:
                output[i][j] = num
            else:
                output[i][j] = image[i][j]
    return output

def sparse_noise_static(image: np.ndarray, prob: float, num: float):
    """
    Add salt and pepper noise to data
    Args:
        image (np.ndarray): Data
        prob (float): Probability of the noise
    Returns:
        output (np.ndarray): Data with added noise
    """

    assert 0 <= prob <= 1, "Probability must be in [0,1]."
    
    m,n = image.shape
    entries = np.random.choice(m*n, np.round(prob*m*n), replace = False)
    x = entries//m -1
    y = entries%m -1

    output = image.copy()
    output[x,y] = num

    return output

def score_performance(
    true_components: np.ndarray,
    pred_components: np.ndarray,
    pc_map: TypedDict,
) -> List[float]:
    """
    Match the true component to the best fit predicted component. Evaluate the difference between the two.

    Args:
        true_components (np.ndarray): The true PCA components to be matched.
        pred_components (np.ndarray): Predicted PCA components.
        true_svs (np.ndarray): True singular values
        pred_svs (np.ndarray): Pred singular values

    Returns:
        diff (List[float]): List containing the minimum difference between each matched true and predicted PCA component.
    """
    component_diff = []

    for index, true_component in enumerate(true_components):
        difference = np.linalg.norm(
            true_component - pred_components[pc_map[index]], 2
        ) / np.linalg.norm(true_component, 2)
        component_diff.append(difference * 100)

    # for true_component in true_components:
    #     difference = np.linalg.norm(true_component - pred_components, 2, axis=1)
    #     diff.append(np.round(np.min(difference), 3))
    #     # Ensure that a single predicted component is not matched to multiple true components
    #     pred_components = np.delete(pred_components, np.argmin(difference), axis=0)

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
        true_svs (np.ndarray): The true PCA singular values
        data (np.ndarray)
        timeout (float): Number of seconds after which to timeout the function run
        num_components (int): The number of components to search for
        epca_args (TypedDict): Dictionary containing necessary arguments for ensemble PCA
        rpca_args (TypedDict): Dictionary containing necessary arguments for robust PCA
    Returns:
        pca_runtime (float): Runtime of PCA
        epca_runtime (float): Runtime of EPCA
        rpca_runtime (float): Runtime of RPCA
        pca_performance (List[float]): List containing difference between true components and those predicted by PCA
        epca_performance (List[float]): List containing difference between true components and those predicted by EPCA
        rpca_performance (List[float]): List containing difference between true components and those predicted by RPCA

    """

    p = Pool(1)
    try:
        res1 = p.apply_async(
            run_pca,
            kwds={
                "images": data,
                "num_components": num_components,
                "smoothing": pca_args.get("smoothing", False),
                "window_length": pca_args.get("window_length", 51),
                "poly_order": pca_args.get("poly_order", 3),
            },
        )
        pca_components, pca_svs, pca_runtime = res1.get(timeout=timeout)
        pca_map = match_components(true_components, pca_components)
        pca_performance = score_performance(true_components, pca_components, pca_map)
        pca_final_svs = [pca_svs[pca_map[i]] for i in range(num_components)]

    except TimeoutError:
        p.terminate()
        print("PCA timed out")
        pca_performance = "NaN"
        pca_runtime = "NaN"
        pca_final_svs = "NaN"

    p = Pool(1)
    try:
        res2 = p.apply_async(
            run_epca,
            kwds={
                "images": data,
                "n_components": num_components,
                "n_components": num_components,
                "num_samples": epca_args.get("num_samples", 100),
                "sample_size": epca_args.get("sample_size", 10),
                "smoothing": epca_args.get("smoothing", False),
                "window_length": epca_args.get("window_length", 51),
                "poly_order": epca_args.get("poly_order", 3),
            },
        )
        epca_components, epca_svs, epca_runtime = res2.get(timeout=timeout)
        epca_map = match_components(true_components, epca_components)
        epca_performance = score_performance(true_components, epca_components, epca_map)
        epca_final_svs = [epca_svs[epca_map[i]] for i in range(num_components)]

    except TimeoutError:
        p.terminate()
        print("EPCA timed out")
        epca_performance = "NaN"
        epca_runtime = "NaN"
        epca_final_svs = "NaN"

    p = Pool(1)
    try:
        res3 = p.apply_async(
            run_rpca,
            kwds={
                "images": data,
                "num_components": num_components,
                "smoothing": rpca_args.get("smoothing", False),
                "reg_E": rpca_args.get("reg_E", 1.0),
                "reg_J": rpca_args.get("reg_J", 1.0),
                "learning_rate": rpca_args.get("learning_rate", 1.1),
                "n_iter_max": rpca_args.get("n_iter_max", 50),
                "smoothing": rpca_args.get("smoothing", False),
                "window_legnth": rpca_args.get("window_length", 51),
                "poly_order": rpca_args.get("poly_order", 3),
            },
        )
        rpca_components, rpca_svs, rpca_runtime = res3.get(timeout=timeout)
        rpca_map = match_components(true_components, rpca_components)
        rpca_performance = score_performance(true_components, rpca_components, rpca_map)
        rpca_final_svs = [rpca_svs[rpca_map[i]] for i in range(num_components)]

    except TimeoutError:
        p.terminate()
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
    Write results of running different versions of PCA on corrupted data (salt and pepper noise and white noise) to a file.

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
    n = original_data.shape[0]

    f = open(filename, "w")

    pca = PCA(n_components=num_components)
    pca.fit(original_data.reshape((n, -1)))
    sv_1 = pca.singular_values_[0]

    # Find how many components capture at least 90% of the information
    percent = np.cumsum(pca.explained_variance_[:num_components])[-1]
    print(
        np.round(percent, 3),
        " percent of the data is explained by",
        num_components,
        " components",
    )

    # scaler = StandardScaler()

    # Add sparse salt and pepper noise
    data_types = ["original"]
    data_list = [original_data.reshape((n, -1))]

    if sparse is True:
        print("Creating sparse salt and pepper noise")
        sp_data = sparse_noise(
            original_data, prob=sp_probability, num=np.max(original_data) * 2
        )

        sp_data = sp_data.reshape((n, -1))
        print("Sparse salt and pepper noise created")
        data_types.append("sparse s&p")
        data_list.append(sp_data)

    # add uniform white noise
    if uniform is True:
        uniform_white_variance = sv_1 / variance_divisor
        print("Creating uniform white noise")
        uniform_white_data = original_data + uniform_white_variance * np.random.random(
            size=original_data.shape
        )
        # if scaling is True:
        #     uniform_white_data = scaler.fit_transform(uniform_white_data)
        uniform_white_data = uniform_white_data.reshape((n, -1))
        print("Created uniform white noise")
        data_types.append("uniform white")
        data_list.append(uniform_white_data)

    # add normal white noise
    if normal is True:
        normal_white_variance = sv_1 / variance_divisor
        print("Creating normal white noise")
        normal_white_data = original_data + np.random.normal(
            0, normal_white_variance, size=original_data.shape
        )
        # if scaling is True:
        #     normal_white_data = scaler.fit_transform(normal_white_data)
        normal_white_data = normal_white_data.reshape((n, -1))

        print("Created normal white noise")
        data_types.append("normal white")
        data_list.append(normal_white_data)

    if outliers is True:
        # Add outliers
        print("Adding ", outlier_fraction, "percent outliers")
        ind = np.random.choice(n, int(np.round(n * outlier_fraction)), replace=False)
        outlier_data = original_data.copy()
        outlier_data[ind] = outlier_data[ind] * outlier_scale
        outlier_data = outlier_data.reshape((n, -1))
        data_types.append("outliers")
        data_list.append(outlier_data)

    f.write(
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

        f.write(
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

    f.close()
    return None
