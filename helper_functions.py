import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from typing import Tuple, Optional, TypedDict, List
from RPCA_3 import RPCA_OG
import random
import time
from tensorly.decomposition import robust_pca
import threading

TIMER = 120


class TimeoutError(Exception):
    pass


class InterruptableThread(threading.Thread):
    def __init__(self, func, *args, **kwargs):
        threading.Thread.__init__(self)
        self._func = func
        self._args = args
        self._kwargs = kwargs
        self._result = None

    def run(self):
        self._result = self._func(*self._args, **self._kwargs)

    @property
    def result(self):
        return self._result


class timeout(object):
    def __init__(self, sec):
        self._sec = sec

    def __call__(self, f):
        def wrapped_f(*args, **kwargs):
            it = InterruptableThread(f, *args, **kwargs)
            it.start()
            it.join(self._sec)
            if not it.is_alive():
                return it.result
            raise TimeoutError("execution expired")

        return wrapped_f


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


@timeout(TIMER)
def run_pca(images: np.ndarray):
    """
    Run standard princpal component analysis on given data.

    Args:
        images (np.ndarray): Data.
    Returns
        pca: PCA class
        runtime (float)
    """
    start_time = time.time()

    assert len(images.shape) == 2, "Data must be 2D"
    pca = PCA()
    pca.fit(images)

    runtime = time.time() - start_time

    return pca, np.round(runtime, 3)


@timeout(TIMER)
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
        n_components (int): Number of components to find.
        num_samples (int): Number of data bags to create.
        sample_size (int): Number of data samples in each of the bags.
        smoothing (int): Whether or not to smooth the components.
        window_length (int): Must be odd. Window length for smoothing.
        poly_order (int): Polynomial order for smoothing.
    Returns:
        centers (np.ndarray): Calculated principal components.
        cluster_labels (np.ndarray): Assignment of each calculated PC to a cluster.
        runtime (float)
    """
    start_time = time.time()
    epca = RPCA_OG(
        num_components=n_components,
        num_samples=num_samples,
        sample_size=sample_size,
        smoothing=smoothing,
        window_length=window_length,
        poly_order=poly_order,
    )
    centers, cluster_labels = epca.run_RPCA(images)

    runtime = time.time() - start_time

    return centers, cluster_labels, np.round(runtime, 3)


@timeout(TIMER)
def run_rpca(
    images: np.ndarray,
    reg_E: float = 1.0,
    reg_J: float = 1.0,
    learning_rate: float = 1.1,
    n_iter_max: int = 50,
):
    """
    Run robust PCA.

    Args:
        images (np.ndarray): Data.
        reg_E (float):
        reg_J (float):
        learning_rate (float):
        n_iter_max (int):
    """
    start_time = time.time()
    low_rank_part, _ = robust_pca(
        X=images.astype(float),
        reg_E=reg_E,
        reg_J=reg_J,
        learning_rate=learning_rate,
        n_iter_max=n_iter_max,
    )

    runtime = time.time() - start_time

    rpca = PCA()
    rpca.fit(low_rank_part)

    return rpca, np.round(runtime, 3)


def sp_noise(image: np.ndarray, prob: float) -> np.ndarray:
    """
    Add salt and pepper noise to data

    Args:
        image (np.ndarray): Data
        prob (float): Probability of the noise
    Returns:
        output (np.ndarray): Data with added noise.
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


def score_performance(
    true_components: np.ndarray, pred_components: np.ndarray
) -> List[float]:
    """
    Match the true component to the best fit predicted component. Evaluate the difference between the two.

    Args:
        true_components (np.ndarray): The true PCA components to be matched.
        pred_components (np.ndarray): Predicted PCA components.

    Returns:
        diff (List[float]): List containing the minimum difference between each matched true and predicted PCA component.
    """
    diff = []
    for true_component in true_components:
        difference = np.linalg.norm(true_component - pred_components, 2, axis=1)
        diff.append(np.round(np.min(difference), 3))
        # Ensure that a single predicted component is not matched to multiple true components
        pred_components = np.delete(pred_components, np.argmin(difference), axis=0)
    return diff


def run_all_pca_methods(
    true_components: np.ndarray,
    data: np.ndarray,
    num_components: int,
    epca_args: TypedDict,
    rpca_args: TypedDict,
):
    """
    Run traditional PCA, ensemble PCA, and robust PCA.

    Args:
        true_components (np.ndarray): The true PCA components.
        data (np.ndarray)
        num_components (int): The number of components to search for.
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
    try:
        pca, pca_runtime = run_pca(images=data)
    except TimeoutError:
        print("PCA timeout")
        pca = None
        pca_runtime = "NaN"
        pass

    try:
        epca_centers, _, epca_runtime = run_epca(
            images=data,
            n_components=num_components,
            num_samples=epca_args.get("num_samples", 100),
            sample_size=epca_args.get("sample_size", 10),
            smoothing=epca_args.get("smoothing", False),
            window_length=epca_args.get("window_length", 51),
            poly_order=epca_args.get("poly_order", 3),
        )
    except TimeoutError:
        print("Ensemble PCA timeout")
        epca_centers = None
        epca_runtime = "NaN"

    try:
        rpca, rpca_runtime = run_rpca(
            images=data,
            reg_E=rpca_args.get("reg_E", 1.0),
            reg_J=rpca_args.get("reg_J", 1.0),
            learning_rate=rpca_args.get("learning_rate", 1.1),
            n_iter_max=rpca_args.get("n_iter_max", 50),
        )
    except TimeoutError:
        print("Robust PCA timeout")
        rpca = None
        rpca_runtime = "NaN"

    if pca is not None:
        pca_performance = score_performance(
            true_components, pca.components_[:num_components]
        )
    else:
        pca_performance = "NaN"

    if epca_centers is not None:
        epca_performance = score_performance(true_components, epca_centers)
    else:
        epca_performance = "NaN"

    if rpca is not None:
        rpca_performance = score_performance(
            true_components, rpca.components_[:num_components]
        )
    else:
        rpca_performance = "NaN"

    return (
        pca_runtime,
        epca_runtime,
        rpca_runtime,
        pca_performance,
        epca_performance,
        rpca_performance,
    )


def write_to_file(
    original_data: np.ndarray,
    num_components: int,
    epca_args: TypedDict,
    rpca_args: TypedDict,
    filename: str,
    sp_probability: float = 0.20,
    uniform_white_variance: float = 2,
    normal_white_variance: float = 2,
):
    """
    Write results of running different versions of PCA on corrupted data (salt and pepper noise and white noise) to a file.
    """
    n = original_data.shape[0]

    f = open(filename, "w")

    pca = PCA()
    pca.fit(original_data.reshape((n, -1)))

    # Find how many components capture at least 90% of the information
    percent = np.cumsum(pca.explained_variance_ratio_[:num_components])[-1]
    print(
        percent, " percent of the data is explained by", num_components, " components"
    )

    # Add sparse salt and pepper noise
    print("Creating sparse salt and pepper noise")
    sp_data = sp_noise(original_data, sp_probability)
    sp_data = sp_data.reshape((n, -1))
    print("Sparse salt and pepper noise created")

    # add uniform white noise
    print("Creating uniform white noise")
    uniform_white_data = (
        original_data
        + uniform_white_variance * np.random.random(size=original_data.shape)
    ).reshape((n, -1))
    print("Created uniform white noise")

    # add normal white noise
    print("Creating normal white noise")
    normal_white_data = (
        original_data
        + np.random.normal(0, normal_white_variance, size=original_data.shape)
    ).reshape((n, -1))
    print("Created normal white noise")

    f.write(
        "data; pca_runtime; epca_runtime; rpca_runtime; pca_performance; epca_performance; rpca_performance \n"
    )

    data_type = ["original", "sparse s&p", "uniform white", "normal white"]
    for index, data in enumerate(
        [original_data.reshape((n, -1)), sp_data, uniform_white_data, normal_white_data]
    ):
        print("Running methods on ", data_type[index], " data.")
        (
            pca_runtime,
            epca_runtime,
            rpca_runtime,
            pca_performance,
            epca_performance,
            rpca_performance,
        ) = run_all_pca_methods(
            true_components=pca.components_[:num_components],
            data=data,
            num_components=num_components,
            epca_args=epca_args,
            rpca_args=rpca_args,
        )

        f.write(
            "%s; %s; %s; %s; %s; %s; %s \n"
            % (
                data_type[index],
                pca_runtime,
                epca_runtime,
                rpca_runtime,
                pca_performance,
                epca_performance,
                rpca_performance,
            )
        )

    f.close()

    return None
