import numpy as np
from helper_functions import write_to_file

if __name__ == "__main__":

    # Generate wave data
    def f(x, t):
        return (1 - 0.5 * np.cos(2 * t)) * (1 / np.cosh(x)) + (
            1 - 0.5 * np.sin(2 * t)
        ) * (1 / np.cosh(x)) * np.tanh(x)

    n = 200
    x = np.linspace(-10, 10, n)
    t = np.linspace(0, 10, 30)

    X, T = np.meshgrid(x, t)

    data = f(X, T)

    epca_args = {"num_samples": 100, "smoothing": True, "sample_size": 10}
    rpca_args = {"smoothing": True}
    pca_args = {"smoothing": True}

    write_to_file(
        original_data=data,
        num_components=2,
        timeout=30,
        pca_args=pca_args,
        epca_args=epca_args,
        rpca_args=rpca_args,
        filename="wave_data_smoothing.txt",
        sp_probability=0.20,
        uniform_white_variance=2,
        normal_white_variance=2,
    )

    epca_args = {"num_samples": 100, "smoothing": False, "sample_size": 10}
    rpca_args = {}
    pca_args = {}
    write_to_file(
        original_data=data,
        num_components=2,
        timeout=30,
        pca_args=pca_args,
        epca_args=epca_args,
        rpca_args=rpca_args,
        filename="wave_data.txt",
        sp_probability=0.20,
        uniform_white_variance=2,
        normal_white_variance=2,
    )
