"""Run analysis on wave data."""
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
    t = np.linspace(0, 3000, 6000)

    X, T = np.meshgrid(x, t)

    data = f(X, T)

    window_length = 31

    # for i in range(100):
    #     filename = "wave_data_large_smoothing_trial_" + str(i) + ".txt"

    #     epca_args = {
    #         "original": {
    #             "num_samples": 100,
    #             "smoothing": True,
    #             "sample_size": 20,
    #             "window_length": window_length,
    #         },
    #         "sparse s&p": {"num_samples": 100, "smoothing": False, "sample_size": 20},
    #         "uniform white": {
    #             "num_samples": 100,
    #             "smoothing": True,
    #             "sample_size": 20,
    #             "window_length": window_length,
    #         },
    #         "normal white": {
    #             "num_samples": 100,
    #             "smoothing": True,
    #             "sample_size": 20,
    #             "window_length": window_length,
    #         },
    #         "outliers": {
    #             "num_samples": 100,
    #             "smoothing": True,
    #             "sample_size": 5,
    #             "window_length": window_length,
    #         },
    #     }
    #     rpca_args = {"smoothing": True, "window_length": window_length, "reg_E": 0.05}
    #     pca_args = {"smoothing": True, "window_length": window_length}

    #     write_to_file(
    #         original_data=data,
    #         num_components=2,
    #         timeout=30,
    #         pca_args=pca_args,
    #         epca_args=epca_args,
    #         rpca_args=rpca_args,
    #         filename=filename,
    #         sp_probability=0.05,
    #         uniform_white_variance=2,
    #         normal_white_variance=2,
    #         outlier_scale=5,
    #         outlier_fraction=0.10,
    #     )

    for i in range(10):
        filename = "wave_data/wave_data_rel_error_trial_" + str(i) + ".txt"

        epca_args = {
            "original": {"num_samples": 100, "smoothing": False, "sample_size": 20},
            "sparse s&p": {"num_samples": 100, "smoothing": False, "sample_size": 20},
            "uniform white": {
                "num_samples": 100,
                "smoothing": False,
                "sample_size": 20,
            },
            "normal white": {"num_samples": 100, "smoothing": False, "sample_size": 20},
            "outliers": {"num_samples": 100, "smoothing": False, "sample_size": 5},
        }
        rpca_args = {"reg_E": 0.2}
        pca_args = {}
        write_to_file(
            original_data=data,
            num_components=2,
            timeout=30,
            pca_args=pca_args,
            epca_args=epca_args,
            rpca_args=rpca_args,
            filename=filename,
            sp_probability=0.01,
            outlier_scale=5,
            outlier_fraction=0.05,
            variance_divisor=1000,
        )
