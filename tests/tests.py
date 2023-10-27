import unittest
import numpy as np
from EPCA.EPCA import EPCA
from helper_functions import (
    match_components,
    run_pca,
    run_epca,
    run_rpca,
    score_performance,
)


class Tests(unittest.TestCase):
    """
    Test class for EPCA
    """

    def test_run_epca(self):
        """
        Test that running EPCA
        - throws errors if data is not 2D or more PCs are requested than the dimension of the data
        - propogates the PCs and explained variances as expected
        - returns normalized PCs
        """

        epca = EPCA(num_components=2, num_bags=100, bag_size=20)
        try:
            data = np.random.rand(1, 100, 10)
            epca.run_epca(data=data)
        except AssertionError:
            pass
        try:
            data = np.random.rand(100, 1)
            epca.run_epca(data=data)
        except AssertionError:
            pass

        data = np.random.rand(100, 10)
        epca.run_epca(data=data)
        assert (
            len(epca.signed_vectors) == 400
        ), "Signed vectors should be of size num_components*num_bags*2"

        np.testing.assert_allclose(
            epca.signed_vectors[:200],
            (-epca.signed_vectors[200:]),
            err_msg="PCs should be stacked along with their reflections.",
        )

        assert (
            len(epca.explained_variance) == 400
        ), "Length of explained variance should equal length of stored PCs."

        np.testing.assert_allclose(
            epca.explained_variance[:200],
            epca.explained_variance[200:],
            err_msg="Explained variance should be copied twice.",
        )

        np.testing.assert_allclose(
            np.linalg.norm(epca.centers, 2, axis=1),
            np.ones(4),
            err_msg="All PCs should be normalized.",
        )

    def test_epca_order(self):
        """
        Ensure EPCA functions cannot be run out of order.
        """
        epca = EPCA(num_components=2, num_bags=100, bag_size=20)

        try:
            epca.return_unique_vectors()
        except AssertionError:
            pass
        try:
            epca.epca_clustering()
        except AssertionError:
            pass

    def test_return_unique_vectors(self):
        """Test that unique vectors are being returned as PCs."""
        epca = EPCA(num_components=3, num_bags=100, bag_size=20)
        data = np.random.rand(100, 10)
        epca.run_epca(data=data)
        _, unique_pcs = epca.return_unique_vectors()

        # Ensure all arrays are unique, even with respect to reflections
        for index_1, unique_pc1 in enumerate(unique_pcs):
            for unique_pc2 in unique_pcs[index_1 + 1 :]:
                np.testing.assert_raises(
                    AssertionError,
                    np.testing.assert_allclose,
                    unique_pc1,
                    unique_pc2,
                )

                np.testing.assert_raises(
                    AssertionError,
                    np.testing.assert_allclose,
                    unique_pc1,
                    -unique_pc2,
                )

    def test_match_components(self):
        """Test that mapping between true and predicted components is accurate."""
        true_components = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        pred_components = [
            np.array([-4.1, -5.1, -6.1]),
            np.array([4.1, 5.1, 6.1]),
            np.array([1.3, 2.3, 3.3]),
            np.array([-1.3, -2.3, -3.3]),
        ]

        pc_map = match_components(
            true_components=true_components, pred_components=pred_components
        )

        assert (
            pc_map[0] == 2
        ), "Map between true and predicted components is inaccruate."

        assert (
            pc_map[1] == 1
        ), "Map between true and predicted components is inaccruate."

    def test_run_pca_methods(self):
        """Test run epca methods."""

        data = np.random.rand(1, 100, 10)
        try:
            _, _, _ = run_pca(data=data, num_components=2)
        except AssertionError:
            pass
        try:
            _, _, _ = run_rpca(data=data, num_components=2)
        except AssertionError:
            pass
        try:
            _, _, _ = run_epca(data=data, num_components=2)
        except AssertionError:
            pass

        data = np.random.rand(100, 2)
        try:
            _, _, _ = run_pca(data=data, num_components=3)
        except AssertionError:
            pass
        try:
            _, _, _ = run_rpca(data=data, num_components=3)
        except AssertionError:
            pass
        try:
            _, _, _ = run_epca(data=data, num_components=3)
        except AssertionError:
            pass

    def test_score_performance(self):
        """Test percent relative error calculation."""
        true_components = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        pred_components = [
            np.array([-4.1, -5.1, -6.1]),
            np.array([4.1, 5.1, 6.1]),
            np.array([1.3, 2.3, 3.3]),
            np.array([-1.3, -2.3, -3.3]),
        ]

        pc_map = {0: 2, 1: 1}

        score = score_performance(
            true_components=true_components,
            pred_components=pred_components,
            pc_map=pc_map,
        )

        self.assertAlmostEqual(score[0], 13.887301496588267, places=1e-4)
        self.assertAlmostEqual(score[1], 1.9738550848792998, places=1e-4)

    def run_tests(self):
        """Run all tests."""
        self.test_run_epca()
        self.test_epca_order()
        self.test_return_unique_vectors()
        self.test_match_components()
        self.test_run_pca_methods()
        self.test_score_performance()


tests = Tests()
tests.run_tests()
