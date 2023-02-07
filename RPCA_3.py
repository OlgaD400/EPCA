import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from collections import Counter
from scipy.signal import savgol_filter
from typing import Optional

from rtkm import RTKM


class RPCA_3:
    def __init__(
        self,
        num_components: int,
        trim_percentage: float,
    ):
        """
        Initialize RPCA process.

        Inputs:
        data (np.ndarray): Data matrix of size n_samples, m_features
        num_components (int): Number of principal components to project the data onto
        similarity_threshold (float): Cosine similarity threshold below which to combine predicted principal components
        """
        # assert trim_percentage in range(0,1)

        self.num_components = num_components
        self.trim_percentage = trim_percentage
        return None

    def run_RPCA(self, data, num_samples, sample_size):
        n, m = data.shape
        # pca_vectors = np.zeros((self.num_components, m, num_samples))
        pca_vectors = np.zeros((self.num_components * num_samples, m))

        pca = PCA(n_components=self.num_components)

        # run PCA num_samples times
        for i in range(num_samples):
            subset = np.random.choice(n, sample_size, replace=False)
            pca.fit(data[subset, :])
            # pca_vectors[:, :, i] = pca.components_
            pca_vectors[
                i * self.num_components : (i + 1) * self.num_components, :
            ] = pca.components_

        signed_vectors = (
            np.sign(cosine_similarity(pca_vectors, pca_vectors[0, :][np.newaxis, ...]))
            * pca_vectors
        )

        rtkm = RTKM(data=signed_vectors.T)

        rtkm.perform_clustering(
            k=self.num_components, percent_outliers=self.trim_percentage
        )

        return rtkm.centers.T


class RPCA_4:
    def __init__(
        self, num_components: int, trim_percentage: float, max_iterations: int
    ):
        """
        Initialize RPCA process.

        Inputs:
        data (np.ndarray): Data matrix of size n_samples, m_features
        num_components (int): Number of principal components to project the data onto
        similarity_threshold (float): Cosine similarity threshold below which to combine predicted principal components
        """
        # assert trim_percentage in range(0,1)

        self.num_components = num_components
        self.trim_percentage = trim_percentage
        self.rtkm_iterations = max_iterations
        return None

    def run_RPCA(self, data, num_samples, sample_size):
        n, m = data.shape
        pca_vectors = np.zeros((self.num_components, m, num_samples))
        # pca_vectors = np.zeros((self.num_components * num_samples, m))

        pca = PCA(n_components=self.num_components)

        # run PCA num_samples times
        for i in range(num_samples):
            subset = np.random.choice(n, sample_size, replace=False)
            pca.fit(data[subset, :])
            pca_vectors[:, :, i] = pca.components_

        final_vectors = np.zeros((self.num_components, m))

        signed_vectors_components = []
        for j in range(self.num_components):
            vectors_j = pca_vectors[j, :, :].T

            # average the vectors together once they're sign aligned
            signed_vectors = (
                np.sign(cosine_similarity(vectors_j, vectors_j[0, :][np.newaxis, ...]))
                * vectors_j
            )

            signed_vectors_components.append(signed_vectors)

            rtkm = RTKM(data=signed_vectors.T)
            rtkm.perform_clustering(
                k=1,
                percent_outliers=self.trim_percentage,
                max_iter=self.rtkm_iterations,
            )

            final_vectors[j, :] = rtkm.centers.T

        return final_vectors, signed_vectors_components


class RPCA_5:
    def __init__(self, num_components: int, trim_percentage: float):
        """
        Initialize RPCA process.

        Inputs:
        data (np.ndarray): Data matrix of size n_samples, m_features
        num_components (int): Number of principal components to project the data onto
        similarity_threshold (float): Cosine similarity threshold below which to combine predicted principal components
        """
        # assert trim_percentage in range(0,1)

        self.num_components = num_components
        self.trim_percentage = trim_percentage
        self.experimental_inliers = None
        return None

    def run_RPCA(self, data, num_samples, sample_size, metric="euclidean"):
        n, m = data.shape
        pca_vectors = np.zeros((self.num_components, m, num_samples))
        # pca_vectors = np.zeros((self.num_components * num_samples, m))

        pca = PCA(n_components=self.num_components)

        subsets = np.zeros((num_samples, sample_size))

        # run PCA num_samples times
        for i in range(num_samples):
            subset = np.random.choice(n, sample_size, replace=False)
            pca.fit(data[subset, :])
            pca_vectors[:, :, i] = pca.components_
            subsets[i, :] = subset

        final_vectors = np.zeros((self.num_components, m))

        outlier_datasets = []
        inlier_datasets = []

        for j in range(self.num_components):

            vectors_j = pca_vectors[j, :, :].T

            # average the vectors together once they're sign aligned
            signed_vectors = (
                np.sign(cosine_similarity(vectors_j, vectors_j[0, :][np.newaxis, ...]))
                * vectors_j
            )

            if metric == "cosine":
                cosine_sim_mat = cosine_similarity(signed_vectors)
                overall_similarities = np.argsort(np.sum(cosine_sim_mat, axis=1))

                # choose highest 1-trimpercent percent
                relevant_indices = overall_similarities[
                    num_samples - round(num_samples * (1 - self.trim_percentage)) :
                ]
                irrelevant_indices = overall_similarities[
                    : num_samples - round(num_samples * (1 - self.trim_percentage))
                ]

            elif metric == "euclidean":
                euclidian_sim_mat = pairwise_distances(
                    signed_vectors, metric="euclidean"
                )

                overall_similarities = np.argsort(np.sum(euclidian_sim_mat, axis=1))

                # choose lowest 1 - trimpercent percent
                relevant_indices = overall_similarities[
                    : round(num_samples * (1 - self.trim_percentage))
                ]
                irrelevant_indices = overall_similarities[
                    round(num_samples * (1 - self.trim_percentage)) :
                ]

            outlier_datasets.append(subsets[irrelevant_indices, :])
            inlier_datasets.append(subsets[relevant_indices, :])

        outliers = np.array(outlier_datasets).flatten()
        hist, _ = np.histogram(outliers, bins=np.arange(n + 1))
        mean = np.mean(hist)
        std = np.std(hist)
        # inliers are those points that show up within one standard deviation of
        # the average frequency of each point in the outlier datasets
        self.experimental_inliers = np.where(
            (hist <= mean + std) & (hist >= mean - std)
        )[0]
        pca.fit(data[self.experimental_inliers])
        final_vectors = pca.components_
        return final_vectors


class RANSAC_RPCA:
    def __init__(self, num_components):
        self.num_components = num_components

    def run_rpca(self, data, threshold, subsets):

        n, m = data.shape
        pca = PCA(n_components=self.num_components)

        ind = list(np.arange(n))
        np.random.shuffle(ind)
        num_members = round(n / subsets)

        total_outliers = []
        inliers = ind

        for i in range(subsets):
            outliers = []
            if i < 9:
                maybe_inliers = ind[i * num_members : (i + 1) * num_members]
            if i == 9:
                maybe_inliers = ind[i * num_members :]

            remaining_data = np.setdiff1d(inliers, maybe_inliers)

            pca.fit(data[remaining_data])
            maybe_model = pca.components_
            # store these components -- look into bootstrapping uncertainty intervals

            for point in maybe_inliers:

                pca.fit(np.vstack((data[remaining_data], data[point])))
                remaining_model = pca.components_
                sim = np.around(cosine_similarity(maybe_model, remaining_model), 3)

                # Set threshold based on level of noise/variance in the data in my subset
                if sum(abs(np.diag(sim))) >= threshold:
                    # remaining_data = np.hstack((remaining_data, point))
                    continue

                else:
                    outliers.append(point)

            total_outliers += outliers

        print("first pass ", len(total_outliers))

        # inliers = np.setdiff1d(np.arange(n), total_outliers)
        # pca.fit(data[inliers])
        # maybe_model = pca.components_

        # sim_vals = []
        # for point in total_outliers:
        #     pca.fit(np.vstack((data[inliers], data[point])))
        #     remaining_model = pca.components_
        #     sim = np.around(cosine_similarity(maybe_model, remaining_model), 3)
        #     sim_vals.append(sum(abs(np.diag(sim))))

        # max_num = round(0.10 * n)
        # this is implemented incorrectly
        # total_outliers = np.argsort(sim_vals)[::-1][:max_num]

        # fewer_outliers = len(total_outliers)
        # while fewer_outliers > 0:
        #     new_inliers = []
        #     curr_outliers = len(total_outliers)
        #     inliers = np.setdiff1d(np.arange(n), total_outliers)
        #     pca.fit(data[inliers])
        #     maybe_model = pca.components_

        #     for point in total_outliers:

        #         pca.fit(np.vstack((data[inliers], data[point])))
        #         remaining_model = pca.components_
        #         sim = np.around(cosine_similarity(maybe_model, remaining_model), 3)

        #         if sum(abs(np.diag(sim))) >= threshold:
        #             new_inliers.append(point)
        #             # inliers = np.hstack((inliers, point))
        #             total_outliers.remove(point)
        #     new_outliers = len(total_outliers)
        #     inliers = np.hstack((inliers, new_inliers))
        #     fewer_outliers = curr_outliers - new_outliers

        # print("final pass ", len(total_outliers))
        inliers = np.setdiff1d(np.arange(n), total_outliers)
        pca.fit(data[inliers])
        predicted_model = pca.components_

        return predicted_model, total_outliers


class RPCA_OG:
    def __init__(
        self,
        num_components: int,
        num_samples: int,
        sample_size: int,
        smoothing=True,
        window_length: int = 51,
        poly_order: int = 3,
    ):
        """
        Initialize RPCA process.

        Inputs:
        data (np.ndarray): Data matrix of size n_samples, m_features
        num_components (int): Number of principal components to project the data onto
        similarity_threshold (float): Cosine similarity threshold below which to combine predicted principal components
        """
        self.num_components = num_components
        self.n_clusters = 2 * num_components
        self.num_samples = num_samples
        self.sample_size = sample_size
        self.smoothing = smoothing
        self.window_length = window_length
        self.poly_order = poly_order
        return None

    def run_RPCA(self, data: np.ndarray):
        assert len(data.shape) == 2, "Data must be 2D"

        n, m = data.shape

        assert (
            self.num_components <= m
        ), "Number of components larger than dimension of data"

        if self.smoothing is True:
            assert self.window_length % 2 == 1, "Window length must be odd"
            assert (
                self.window_length <= m
            ), "Window length must be less than dimension of data"

        # pca_vectors = np.zeros((self.num_components, m, num_samples))
        pca_vectors = np.zeros((self.num_components * self.num_samples, m))

        pca = PCA(n_components=self.num_components)

        # run PCA num_samples times
        for i in range(self.num_samples):
            subset = np.random.choice(n, self.sample_size, replace=True)
            pca.fit(data[subset, :])
            # u,s,vt = np.linalg.norm(data[subset,:])
            # pca_vectors[:, :, i] = pca.components_
            pca_vectors[
                i * self.num_components : (i + 1) * self.num_components, :
            ] = pca.components_

        # Rather than arbitrarily aligning, stack all the vectors and they're reverse directions and look for 2 times
        # as many clusters
        signed_vectors = np.vstack((pca_vectors, pca_vectors * -1))

        if self.smoothing is True:
            signed_vectors = savgol_filter(
                signed_vectors,
                window_length=self.window_length,
                polyorder=self.poly_order,
            )

            # smoothed_vectors = np.zeros((n,m))
            # km = Kalman(alpha = .10)
            # for sample in range(n):
            #     smoothed_vectors[sample] = km.compute(np.arange(m), signed_vectors[sample], i = np.arange(m))

        signed_vectors = (
            signed_vectors / np.linalg.norm(signed_vectors, 2, axis=1)[..., np.newaxis]
        )

        kmeans = KMeans(n_clusters=self.n_clusters).fit(signed_vectors)

        centers = (
            kmeans.cluster_centers_
            / np.linalg.norm(kmeans.cluster_centers_, 2, axis=1)[..., np.newaxis]
        )

        return centers, kmeans.labels_


class RPCA_dbscan:
    def __init__(self, num_components: int, eps):
        """
        Initialize RPCA process.

        Inputs:
        data (np.ndarray): Data matrix of size n_samples, m_features
        num_components (int): Number of principal components to project the data onto
        similarity_threshold (float): Cosine similarity threshold below which to combine predicted principal components
        """
        # assert trim_percentage in range(0,1)

        self.num_components = num_components
        self.eps = eps
        return None

    def run_RPCA(self, data, num_samples, sample_size, n_clusters):
        n, m = data.shape
        # pca_vectors = np.zeros((self.num_components, m, num_samples))
        pca_vectors = np.zeros((self.num_components * num_samples, m))

        pca = PCA(n_components=self.num_components)

        # run PCA num_samples times
        for i in range(num_samples):
            subset = np.random.choice(n, sample_size, replace=True)
            pca.fit(data[subset, :])
            # u,s,vt = np.linalg.norm(data[subset,:])
            # pca_vectors[:, :, i] = pca.components_
            pca_vectors[
                i * self.num_components : (i + 1) * self.num_components, :
            ] = pca.components_

        alignment_vector = np.ones((1, m))

        signed_vectors = (
            np.sign(cosine_similarity(pca_vectors, alignment_vector)) * pca_vectors
        )

        dbscan = DBSCAN(eps=self.eps, min_samples=5).fit(signed_vectors)

        ind_lens = []
        unique_labels = np.unique(dbscan.labels_)

        for label in unique_labels:
            ind = np.where(dbscan.labels_ == label)[0]

            ind_lens.append(len(ind))

        biggest_clusters = np.argsort(ind_lens)[::-1][: self.num_components]

        centers = np.zeros((self.num_components, m))

        for i, cluster in enumerate(biggest_clusters):
            cluster_ind = np.where(dbscan.labels_ == unique_labels[cluster])
            centers[i] = np.average(signed_vectors[cluster_ind], axis=0)

        return dbscan.labels_, pca_vectors, signed_vectors, centers


class RANSAC_PCA_2:
    def __init__(self, num_components: int):
        self.num_components = num_components

    def calculate_objective(self, U, A):
        m, n = A.shape
        return np.linalg.norm((np.eye(m, m) - U @ U.T) @ A, 2) ** 2

    def mask_data(self, mask_ind, A):
        m, n = A.shape
        A[mask_ind[0], mask_ind[1]] = 0
        cols_count = Counter(mask_ind[1])
        avgs = np.sum(A[:, mask_ind[1]], axis=0) / (
            m - np.array([cols_count[col] for col in mask_ind[1]])
        )
        A[mask_ind[0], mask_ind[1]] = avgs
        return A

    def mask_rows(self, mask_ind, A, columns):
        m, n = A.shape
        A[mask_ind[0], columns[mask_ind[1]]] = 0
        cols_count = Counter(mask_ind[1])
        avgs = np.sum(A[:, columns[mask_ind[1]]], axis=0) / (
            m - np.array([cols_count[col] for col in mask_ind[1]])
        )
        A[mask_ind[0], columns[mask_ind[1]]] = avgs
        return A

    def mask_columns(self, mask_ind, A):
        avgs = np.average(A[:, mask_ind], axis=0)
        A[:, mask_ind] = avgs
        return A

    def phase_1(
        self,
        data,
        max_iter: int = 1000,
        sample_size: int = 10,
        d: int = 10,
    ):
        m, n = data.shape

        iterations = 0
        u, s, vh = np.linalg.svd(data)
        best_obj = self.calculate_objective(u, data)
        best_v = vh

        best_mask = np.ones((m, n))
        best_added_inliers_y = None
        print("og best obj", best_obj)

        while iterations < max_iter:
            data_copy = data.copy()
            mask = np.random.choice(n, sample_size, replace=True)
            data_copy = self.mask_columns(mask, data_copy)
            u, s, vh = np.linalg.svd(data_copy)
            obj = self.calculate_objective(u, data_copy)

            maybe_inliers_y = []
            for i in range(len(mask)):
                avg_value = data_copy[0, mask[i]]
                data_copy[:, mask[i]] = data[:, mask[i]]

                u, s, vh = np.linalg.svd(data_copy)
                new_obj = self.calculate_objective(u, data_copy)

                if new_obj <= obj:
                    maybe_inliers_y.append(mask[i])

                data_copy[:, mask[i]] = avg_value

            if len(maybe_inliers_y) > d:

                data_copy[:, maybe_inliers_y] = data[:, maybe_inliers_y]
                u, s, vh = np.linalg.svd(data_copy)
                new_obj = self.calculate_objective(u, data_copy)

                if new_obj < best_obj:
                    print("columns added back", len(maybe_inliers_y))
                    best_mask = mask
                    best_added_inliers_y = maybe_inliers_y
                    best_obj = new_obj
                    best_v = vh

            iterations += 1

            if iterations % 100 == 0:
                print("iteration ", iterations)

            final_col_mask = np.setdiff1d(best_mask, best_added_inliers_y)
            return final_col_mask, best_obj, best_v

    def phase_2(
        self,
        data,
        max_iter: int = 1000,
        sample_size: int = 10,
        d: int = 10,
    ):

        final_col_mask, best_obj, best_v = self.phase_1(data, max_iter, sample_size, d)

        m, n = data.shape
        N = len(final_col_mask)

        iterations = 0
        # u, s, vh = np.linalg.svd(data)
        # best_obj = self.calculate_objective(u, data)
        # best_v = vh

        best_mask = final_col_mask
        best_added_inliers_x = None
        best_added_inliers_y = None
        print("og best obj", best_obj)

        while iterations < max_iter:
            data_copy = data.copy()
            ind = np.arange(m * N)
            # ind = np.arange(m * n)
            mask = np.unravel_index(np.random.choice(ind, size=sample_size), (m * N))
            # data_copy = self.mask_data(mask, data_copy)
            data_copy = self.mask_rows(mask, data_copy, final_col_mask)
            u, s, vh = np.linalg.svd(data_copy)
            obj = self.calculate_objective(u, data_copy)

            maybe_inliers_x = []
            maybe_inliers_y = []
            for i in range(len(mask[0])):
                avg_value = data_copy[mask[0][i], final_col_mask[mask[1][i]]]
                data_copy[mask[0][i], final_col_mask[mask[1][i]]] = data[
                    mask[0][i], final_col_mask[mask[1][i]]
                ]

                u, s, vh = np.linalg.svd(data_copy)
                new_obj = self.calculate_objective(u, data_copy)

                if new_obj <= obj:
                    maybe_inliers_x.append(mask[0][i])
                    maybe_inliers_y.append(final_col_mask[mask[1][i]])

                data_copy[mask[0][i], final_col_mask[mask[1][i]]] = avg_value

            if len(maybe_inliers_x) > d:

                data_copy[maybe_inliers_x, maybe_inliers_y] = data[
                    maybe_inliers_x, maybe_inliers_y
                ]
                u, s, vh = np.linalg.svd(data_copy)
                new_obj = self.calculate_objective(u, data_copy)

                if new_obj < best_obj:
                    print("points added back", len(maybe_inliers_x))
                    best_mask = mask
                    best_added_inliers_x = maybe_inliers_x
                    best_added_inliers_y = maybe_inliers_y
                    best_obj = new_obj
                    best_v = vh

            iterations += 1

            if iterations % 100 == 0:
                print("iteration ", iterations)

        return best_obj, best_v, best_mask, best_added_inliers_x, best_added_inliers_y

    def run_rpca(
        self,
        data,
        max_iter: int = 1000,
        sample_size: int = 10,
        d: int = 10,
    ):

        m, n = data.shape

        iterations = 0
        u, s, vh = np.linalg.svd(data)
        best_obj = self.calculate_objective(u, data)
        best_v = vh

        best_mask = np.ones((m, n))
        best_added_inliers_x = None
        best_added_inliers_y = None
        print("og best obj", best_obj)

        while iterations < max_iter:
            data_copy = data.copy()
            ind = np.arange(m * n)
            mask = np.unravel_index(np.random.choice(ind, size=sample_size), data.shape)
            data_copy = self.mask_data(mask, data_copy)
            u, s, vh = np.linalg.svd(data_copy)
            obj = self.calculate_objective(u, data_copy)

            maybe_inliers_x = []
            maybe_inliers_y = []
            for i in range(len(mask[0])):
                avg_value = data_copy[mask[0][i], mask[1][i]]
                data_copy[mask[0][i], mask[1][i]] = data[mask[0][i], mask[1][i]]

                u, s, vh = np.linalg.svd(data_copy)
                new_obj = self.calculate_objective(u, data_copy)

                if new_obj <= obj:
                    maybe_inliers_x.append(mask[0][i])
                    maybe_inliers_y.append(mask[1][i])

                data_copy[mask[0][i], mask[1][i]] = avg_value

            if len(maybe_inliers_x) > d:

                data_copy[maybe_inliers_x, maybe_inliers_y] = data[
                    maybe_inliers_x, maybe_inliers_y
                ]
                u, s, vh = np.linalg.svd(data_copy)
                new_obj = self.calculate_objective(u, data_copy)

                if new_obj < best_obj:
                    print("points added back", len(maybe_inliers_x))
                    best_mask = mask
                    best_added_inliers_x = maybe_inliers_x
                    best_added_inliers_y = maybe_inliers_y
                    best_obj = new_obj
                    best_v = vh

            iterations += 1

            if iterations % 100 == 0:
                print("iteration ", iterations)

        return best_obj, best_v, best_mask, best_added_inliers_x, best_added_inliers_y
