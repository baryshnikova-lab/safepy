import unittest
import os, sys
import numpy as np
import logging

sys.path.append('../../safepy')

from safepy import safe


class TestEnrichmentHypergeometric(unittest.TestCase):

    def setUp(self):

        # Load the default network
        sf = safe.SAFE(verbose=False)
        sf.load_network()
        sf.define_neighborhoods()

        # Load the binary attributes
        # path_to_go = os.path.join(sf.path_to_safe_data, 'attributes/go_bp_140819.txt.gz')
        path_to_go = 'attributes/go_bp_140819.txt.gz'
        sf.load_attributes(attribute_file=path_to_go)

        # Run the enrichment
        sf.compute_pvalues(multiple_testing=False)

        self.sf = sf

    def test_attribute_numbers(self):

        num_attributes = self.sf.attributes.shape[0]
        self.assertEqual(num_attributes, 4373, "Should be 4373.")

        num_nans = np.sum(np.isnan(self.sf.node2attribute))
        self.assertEqual(num_nans, 795886, "Should be 795886.")

        num_zeros = np.sum(self.sf.node2attribute[~np.isnan(self.sf.node2attribute)] == 0)
        self.assertEqual(num_zeros, 16405921, "Should be 16405921.")

        num_positives = np.sum(self.sf.node2attribute[~np.isnan(self.sf.node2attribute)] > 0)
        self.assertEqual(num_positives, 163376, "Should be 163376.")

        num_negatives = np.sum(self.sf.node2attribute[~np.isnan(self.sf.node2attribute)] < 0)
        self.assertEqual(num_negatives, 0, "Should be 0.")

    def test_enrichment_numbers(self):

        num_enriched_neighborhoods = np.sum(self.sf.nes_binary > 0)
        self.assertEqual(num_enriched_neighborhoods, 367409, "Should be 367409.")

        num_enriched_neighborhoods_dna_repair = self.sf.attributes.loc[self.sf.attributes['name'] == 'DNA repair',
                                                                       'num_neighborhoods_enriched'].values[0].astype(int)
        self.assertEqual(num_enriched_neighborhoods_dna_repair, 418, "Should be 418.")

        num_enriched_attributes = np.sum(self.sf.attributes['num_neighborhoods_enriched'] > 10)
        self.assertEqual(num_enriched_attributes, 2767, "Should be 2767.")


class TestEnrichmentPermutations(unittest.TestCase):

    def setUp(self):

        logging.disable(logging.CRITICAL)

        # Load the default network
        sf = safe.SAFE(verbose=False)
        sf.load_network()
        sf.define_neighborhoods()

        # Load the binary attributes
        # path_to_dox = os.path.join(sf.path_to_safe_data, 'attributes/hoepfner_movva_2014_doxorubucin.txt')
        path_to_dox='attributes/hoepfner_movva_2014_doxorubucin.txt'
        sf.load_attributes(attribute_file=path_to_dox)

        # Run the enrichment
        sf.compute_pvalues(num_permutations=1000, multiple_testing=False, verbose=False)

        self.sf = sf

    def test_attribute_numbers(self):

        num_attributes = self.sf.attributes.shape[0]
        self.assertEqual(num_attributes, 1, "Should be 1.")

        num_nans = np.sum(np.isnan(self.sf.node2attribute))
        self.assertEqual(num_nans, 1315, "Should be 1315.")

        num_zeros = np.sum(self.sf.node2attribute[~np.isnan(self.sf.node2attribute)] == 0)
        self.assertEqual(num_zeros, 1, "Should be 1.")

        num_positives = np.sum(self.sf.node2attribute[~np.isnan(self.sf.node2attribute)] > 0)
        self.assertEqual(num_positives, 1335, "Should be 1335.")

        num_negatives = np.sum(self.sf.node2attribute[~np.isnan(self.sf.node2attribute)] < 0)
        self.assertEqual(num_negatives, 1320, "Should be 1320.")

    def test_enrichment_numbers(self):

        num_enriched_neighborhoods = np.sum(self.sf.nes_binary > 0)
        self.assertAlmostEqual(num_enriched_neighborhoods, 637, delta=20, msg="Should be 637 +/- 20.")


if __name__ == '__main__':
    unittest.main()
