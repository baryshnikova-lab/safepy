import unittest
import networkx as nx

from safe_io import *


class TestImportCys(unittest.TestCase):

    def setUp(self):
        # Load the network
        f = '/Users/abaryshnikova/Lab/Datasets/safe-data/networks/Costanzo_Science_2016.cys'
        self.graph = load_network_from_cys(f, verbose=False)

    def test_size(self):
        num_nodes = nx.number_of_nodes(self.graph)
        self.assertEqual(num_nodes, 3971, "Should be 3971.")
        num_edges = nx.number_of_edges(self.graph)
        self.assertEqual(num_edges, 28202, "Should be 28202.")


if __name__ == '__main__':
    unittest.main()
