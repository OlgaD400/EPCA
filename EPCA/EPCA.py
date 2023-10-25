"""Implementation of Ensemble Principal Component Analysis (EPCA)."""

from typing import Tuple, List
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


class EPCA:
    """Implementation of Ensemble Principal Component Analysis (EPCA)."""

    def __init__(
        self,
        num_components: int,
        num_bags: int,
        bag_size: int,
    ):
        """
        Initialize EPCA process.

        Inputs:
            num_components (int): Number of components to find with EPCA.
            num_bags (int): Number of bags to use in bagging procedure.
            bag_size (int): Size of bags to use in bagging procedure.
        Attributes:
            signed_vectors (np.ndarray): EPCA modes stacked along with
            their reflections.
            explained_variance (np.ndarray): Explained variance of each
            of the EPCA modes. Equal to num_components largest eigenvalues of
            the covariance matrix.
        """
        self.num_components = num_components
        self.num_bags = num_bags
        self.bag_size = bag_size
        self.centers = None
        self.labels = None
        self.avg_explained_variance = None
        self.signed_vectors = None
        self.explained_variance = None
        return None

    def epca_bagging(self, data: np.ndarray):
        """
        Bagging procedure of EPCA.

        Run self.PCA num_bags times with bag_size samples of the original
        data matrix.
        Record the pca modes and their reflections and the explained
        variances output for each data bag.

        Args:
            data (np.ndarray): Data matrix to carry out EPCA bagging on.
        Returns:
            final_signed_vectors (np.ndarray): EPCA modes stacked along with
            their reflections.
            final_explained_variance (np.ndarray): Explained variance of each
            of the EPCA modes. Equal to num_components largest eigenvalues of
            the covariance matrix.
        """
        data_samples, data_dimension = data.shape
        pca_vectors = np.zeros((self.num_components * self.num_bags, data_dimension))
        explained_variance = np.zeros(self.num_components * self.num_bags)

        pca = PCA(n_components=self.num_components)

        # run PCA num_bags times
        for i in range(self.num_bags):
            subset = np.random.choice(data_samples, self.bag_size, replace=True)
            pca.fit(data[subset, :])

            pca_vectors[
                i * self.num_components : (i + 1) * self.num_components, :
            ] = pca.components_

            explained_variance[
                i * self.num_components : (i + 1) * self.num_components
            ] = pca.explained_variance_

        # Stack all vectors and their reflectiobs
        signed_vectors = np.vstack((pca_vectors, pca_vectors * -1))
        self.explained_variance = np.hstack((explained_variance, explained_variance))

        self.signed_vectors = (
            signed_vectors / np.linalg.norm(signed_vectors, 2, axis=1)[..., np.newaxis]
        )

    def epca_clustering(self):
        """
        Use k-means to cluster the EPCA modes from the bagging procedure.

        Run k-means with k=2*num_num_components clusters.

        Args:
            signed_vectors (np.ndarray): EPCA modes stacked along with
            their reflections.
            explained_variance (np.ndarray): Explained variance of each
            of the EPCA modes. Equal to num_components largest eigenvalues of
            the covariance matrix.
        Returns:
            centers (np.ndarray): Average EPCA modes. Centers of k-means.
            labels (np.ndarray): EPCA mode cluster assignments.
            avg_explained_variance (np.ndarray): Average explained variance
                associated with each of the EPCA mode clusters.
        """
        kmeans = KMeans(n_clusters=2 * self.num_components, n_init=10).fit(
            self.signed_vectors
        )

        self.centers = (
            kmeans.cluster_centers_
            / np.linalg.norm(kmeans.cluster_centers_, 2, axis=1)[..., np.newaxis]
        )

        self.labels = kmeans.labels_

        self.avg_explained_variance = []
        for unique_label in np.unique(kmeans.labels_):
            label = np.where(kmeans.labels_ == unique_label)[0]
            self.avg_explained_variance.append(
                np.average(self.explained_variance[label])
            )

    def run_epca(self, data: np.ndarray):
        """
        Run EPCA bagging followed by clustering.

        Args:
            data (np.ndarray): Data on which to run EPCA.
        """
        assert len(data.shape) == 2, "Data must be 2D"

        _, data_dimension = data.shape

        assert (
            self.num_components <= data_dimension
        ), "Number of components larger than dimension of data"

        self.epca_bagging(data=data)
        self.epca_clustering()

    def return_unique_vectors(
        self, tol: float = 1e-4
    ) -> Tuple[List[int], List[np.ndarray]]:
        """
        Return the labels of the unique EPCA modes up to a reflection.

        Args:
            tol (float): Tolerance beyond which vectors are considered unique.
        Returns:
            unique_labels (List[int]): The labels associated with the clusters
            of unique e.vectors
            centers[unique_labels]: The unique e.vectors
        """
        unique_labels = [0]

        for index in range(1, self.num_components * 2):
            if np.any(
                np.linalg.norm(
                    self.centers[index] + self.centers[unique_labels], axis=1
                )
                < tol
            ):
                continue
            else:
                unique_labels.append(index)

        return unique_labels, self.centers[unique_labels]
