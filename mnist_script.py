"""Run analysis on MNIST data."""

from mnist import MNIST
import numpy as np
from helper_functions import write_to_file

if __name__ == "__main__":
    # Load data.
    mndata = MNIST("./data")
    images, labels = mndata.load_training()
    images = np.array(images)
    labels = np.array(labels)

    for integer in range(2):
        number = np.where(labels == integer)[0]
        og_data = images[number, :]

        # Define parameters for EPCA, PCA, and RPCA.
        pca_args = {}
        epca_args = {
            "original": {"num_samples": 100, "sample_size": 100},
            "sparse": {"num_samples": 100, "sample_size": 100},
            "uniform white": {
                "num_samples": 100,
                "sample_size": 100,
            },
            "normal white": {
                "num_samples": 100,
                "sample_size": 100,
            },
            "outliers": {"num_samples": 100, "sample_size": 20},
        }
        rpca_args = {"reg_E": 0.2}

        # Run experiments.
        for i in range(0, 1):
            FILENAME = "temp.txt"
            # (
            #     "mnist_data/mnist_data_" + str(integer) + "_trial_" + str(i) + ".txt"
            # )

            write_to_file(
                original_data=og_data,
                num_components=3,
                timeout=120,
                pca_args=pca_args,
                epca_args=epca_args,
                rpca_args=rpca_args,
                filename=FILENAME,
                sparse_noise_args={"sparse_probability": 0.01},
                outlier_args={"outlier_scale": 5, "outlier_fraction": 0.05},
                white_noise_args={"variance_divisor": 1000},
            )
