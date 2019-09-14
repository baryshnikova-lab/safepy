import unittest
import copy
import networkx as nx
import numpy as np

from safepy import safe, safe_io


class TestImportCys(unittest.TestCase):

    def setUp(self):
        # Load the network
        f = '/Users/abaryshnikova/Lab/Datasets/safe-data/networks/Costanzo_Science_2016.cys'
        self.graph = safe_io.load_network_from_cys(f, verbose=False)

    def test_size(self):
        num_nodes = nx.number_of_nodes(self.graph)
        self.assertEqual(num_nodes, 3971, "Should be 3971.")
        num_edges = nx.number_of_edges(self.graph)
        self.assertEqual(num_edges, 28202, "Should be 28202.")


class TestImportAttributes(unittest.TestCase):

    def setUp(self):
        # Load the default network
        sf = safe.SAFE(verbose=False)
        sf.load_network()

        self.sf = sf

    def test_default(self):

        sf = copy.deepcopy(self.sf)
        sf.load_attributes()

    def test_attribute_with_duplicate_values(self):

        sf = copy.deepcopy(self.sf)

        f = '/Users/abaryshnikova/Lab/Datasets/safe-data/attributes/sample_annotation_file.txt'
        sf.load_attributes(attribute_file=f)


if __name__ == '__main__':
    unittest.main()
