import unittest
import copy
import numpy as np

import safe


class TestDefineNeighborhoods(unittest.TestCase):

    def setUp(self):
        pass

    def test_default(self):

        # Load the default network
        sf = safe.SAFE(verbose=False)
        sf.load_network()

        sf.define_neighborhoods()

        num_neighbors = np.sum(sf.neighborhoods, axis=1)
        num_neighbors_avg = np.mean(num_neighbors)
        num_neighbors_std = np.std(num_neighbors)

        self.assertAlmostEqual(num_neighbors_avg, 37.5, delta=0.5, msg="Should be 37.5.")
        self.assertAlmostEqual(num_neighbors_std, 56.74, delta=0.5, msg="Should be 56.74.")

    def test_euclidean(self):

        # Load the default network
        sf = safe.SAFE(verbose=False)
        sf.load_network()

        sf.define_neighborhoods(node_distance_metric='euclidean')

        num_neighbors = np.sum(sf.neighborhoods, axis=1)
        num_neighbors_avg = np.mean(num_neighbors)
        num_neighbors_std = np.std(num_neighbors)

        self.assertAlmostEqual(num_neighbors_avg, 148.44, delta=0.5, msg="Should be 148.44.")
        self.assertAlmostEqual(num_neighbors_std, 40.99, delta=0.5, msg="Should be 40.99.")

    def test_shortpath(self):

        # Load the default network
        sf = safe.SAFE(verbose=False)
        sf.load_network()

        sf.define_neighborhoods(node_distance_metric='shortpath',
                                neighborhood_radius=1)

        num_neighbors = np.sum(sf.neighborhoods, axis=1)
        num_neighbors_avg = np.mean(num_neighbors)
        num_neighbors_std = np.std(num_neighbors)

        self.assertAlmostEqual(num_neighbors_avg, 15.20, delta=0.5, msg="Should be 15.20.")
        self.assertAlmostEqual(num_neighbors_std, 18.32, delta=0.5, msg="Should be 18.32.")


if __name__ == '__main__':
    unittest.main()
