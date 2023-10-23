"""Run analysis on wave data."""
import numpy as np
from helper_functions import write_to_file

if __name__ == "__main__":
    # Generate wave data
    def f(x, t):
        """Generate wave data."""
        return (1 - 0.5 * np.cos(2 * t)) * (1 / np.cosh(x)) + (
            1 - 0.5 * np.sin(2 * t)
        ) * (1 / np.cosh(x)) * np.tanh(x)

    DIMENSION = 200
    x_space = np.linspace(-10, 10, DIMENSION)
    t_space = np.linspace(0, 3000, 6000)

    X, T = np.meshgrid(x_space, t_space)

    DATA = f(X, T)
    NUM_TRIALS = 2

    for i in range(NUM_TRIALS):
        FILENAME = (
            "temp.txt"  # "wave_data/wave_data_rel_error_trial_" + str(i) + ".txt"
        )

        epca_args = {
            "original": {"num_samples": 100, "sample_size": 20},
            "sparse": {"num_samples": 100, "sample_size": 20},
            "uniform white": {
                "num_samples": 100,
                "sample_size": 20,
            },
            "normal white": {"num_samples": 100, "sample_size": 20},
            "outliers": {"num_samples": 100, "sample_size": 5},
        }
        rpca_args = {"reg_E": 0.2}
        pca_args = {}
        write_to_file(
            original_data=DATA,
            num_components=2,
            timeout=30,
            pca_args=pca_args,
            epca_args=epca_args,
            rpca_args=rpca_args,
            filename=FILENAME,
            sparse_noise_args={"sparse_probability": 0.01},
            outlier_args={"outlier_scale": 5, "outlier_fraction": 0.05},
            white_noise_args={"variance_divisor": 1000},
        )
