"""Run analysis on sklearn data."""

from sklearn import datasets
from helper_functions import write_to_file

if __name__ == "__main__":
    # Load data.
    data_dict = {
        "iris": datasets.load_iris(),
        "wine": datasets.load_wine(),
        "breast_cancer": datasets.load_breast_cancer(),
    }

    number_of_components = [2, 2, 2, 2]

    # Define parameters for EPCA, PCA, and RPCA.
    for index, dataset in enumerate(data_dict.keys()):
        loaded_data = data_dict[dataset]
        data = loaded_data.data

        n, m = data.shape

        epca_args = {
            "original": {
                "num_samples": 100,
                "sample_size": max(5, n // 10),
            },
            "sparse": {
                "num_samples": 100,
                "sample_size": max(5, n // 10),
            },
            "uniform white": {
                "num_samples": 100,
                "sample_size": max(5, n // 10),
            },
            "normal white": {
                "num_samples": 100,
                "sample_size": max(5, n // 10),
            },
            "outliers": {
                "num_samples": 100,
                "sample_size": min(5, n // 10),
            },
        }
        rpca_args = {"reg_E": 0.2}
        pca_args = {}

        # Run experiments.
        NUM_TRIALS = 100
        for i in range(NUM_TRIALS):
            filename = (
                dataset
                + "_data/"
                + dataset
                + "_data_rel_error_trial_"
                + str(i)
                + ".txt"
            )

            write_to_file(
                original_data=data,
                num_components=number_of_components[index],
                timeout=30,
                pca_args=pca_args,
                epca_args=epca_args,
                rpca_args=rpca_args,
                filename=filename,
                sparse_noise_args={"sparse_probability": 0.01},
                outlier_args={"outlier_scale": 5, "outlier_fraction": 0.05},
                white_noise_args={"variance_divisor": 1000},
            )
