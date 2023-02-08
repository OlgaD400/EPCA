from mnist import MNIST
import numpy as np
from helper_functions import write_to_file

if __name__ == "__main__":
    mndata = MNIST("./data")
    images, labels = mndata.load_training()
    images = np.array(images)
    labels = np.array(labels)

    for integer in range(10):
        number = np.where(labels == integer)[0]
        og_data = images[number, :]

        filename = "mnist_data_" + str(integer) + ".txt"

        pca_args = {}
        epca_args = {"num_samples": 100, "smoothing": False, "sample_size": 50}
        rpca_args = {}
        write_to_file(
            original_data=og_data,
            num_components=2,
            timeout=120,
            pca_args=pca_args,
            epca_args=epca_args,
            rpca_args=rpca_args,
            filename=filename,
            sp_probability=0.20,
            uniform_white_variance=5,
            normal_white_variance=5,
        )
