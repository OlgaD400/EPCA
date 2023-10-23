"""Run analysis on MNIST data."""

from mnist import MNIST
import numpy as np
from helper_functions import write_to_file

if __name__ == "__main__":
    mndata = MNIST("./data")
    images, labels = mndata.load_training()
    images = np.array(images)
    labels = np.array(labels)

    NUM_TRIALS = 2

    for integer in range(NUM_TRIALS):
        number = np.where(labels == integer)[0]
        og_data = images[number, :]

        pca_args = {}
        epca_args = {
            "original": {"num_samples": 100, "sample_size": 100},
            "sparse s&p": {"num_samples": 100, "sample_size": 100},
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

        for i in range(0, 5):
            filename = (
                "mnist_data/mnist_data_" + str(integer) + "_trial_" + str(i) + ".txt"
            )

            write_to_file(
                original_data=og_data,
                num_components=3,
                timeout=120,
                pca_args=pca_args,
                epca_args=epca_args,
                rpca_args=rpca_args,
                filename=filename,
                sp_probability=0.01,
                outlier_scale=5,
                outlier_fraction=0.05,
                variance_divisor=1000,
            )
