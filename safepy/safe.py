#! /usr/bin/env python
"""This file contains the code for the SAFE class and command-line access."""

import configparser
import os
from pathlib import Path
import sys
import textwrap
import argparse
import pickle
import time
import re
import logging

# Necessary check to make sure code runs both in Jupyter and in command line
if 'matplotlib' not in sys.modules:
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import multiprocessing as mp

from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import hypergeom
from itertools import compress
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from statsmodels.stats.multitest import fdrcorrection

from .safe_io import *
from .safe_extras import *
from .safe_colormaps import *


class SAFE:
    """
    Defines an instance of SAFE analysis.
    Contains all data, all parameters and provides the main methods for performing analysis.
    """

    def __init__(self,
                 path_to_ini_file='',
                 path_to_safe_data=None,
                 verbose=True):
        """
        Initiate a SAFE instance and define the main settings for analysis.
        The settings are automatically extracted from the specified (or default) INI configuration file.
        Alternatively, each setting can be changed manually after initiation.

        :param path_to_ini_file (str): Path to the configuration file. If not specified, safe_default.ini will be used.
        :param verbose (bool): Defines whether or not intermediate output will be printed out.

        """

        self.verbose = verbose

        self.default_config = None

        self.path_to_safe_data = path_to_safe_data
        self.path_to_network_file = None
        self.view_name = None
        self.path_to_attribute_file = None

        self.graph = None
        self.node_key_attribute = 'label_orf'

        self.attributes = None
        self.nodes = None
        self.node2attribute = None
        self.num_nodes_per_attribute = None
        self.attribute_sign = 'both'

        self.node_distance_metric = 'shortpath_weighted_layout'
        self.neighborhood_radius_type = None
        self.neighborhood_radius = None

        self.background = 'attribute_file'
        self.num_permutations = 1000
        self.multiple_testing = False
        self.neighborhood_score_type = 'sum'
        self.enrichment_type = 'auto'
        self.enrichment_threshold = 0.05
        self.enrichment_max_log10 = 16
        self.attribute_enrichment_min_size = 10

        self.neighborhoods = None

        self.ns = None
        self.pvalues_neg = None
        self.pvalues_pos = None
        self.nes = None
        self.nes_threshold = None
        self.nes_binary = None

        self.attribute_unimodality_metric = 'connectivity'
        self.attribute_distance_metric = 'jaccard'
        self.attribute_distance_threshold = 0.75

        self.domains = None
        self.node2domain = None

        # Output
        self.output_dir = ''

        # Read both default and user-defined settings
        self.read_config(path_to_ini_file, path_to_safe_data=self.path_to_safe_data)

        # Validate config
        self.validate_config()

    def read_config(self, path_to_ini_file, path_to_safe_data=None):

        """
        Read the settings from an INI file and update the attributes in the SAFE class.

        :param path_to_ini_file (str): Path to the configuration file. If not specified, safe_default.ini will be used.
        :return: none
        """

        # Location of this code
        loc = os.path.dirname(os.path.abspath(__file__))

        # Load default settings
        default_config_path = os.path.join(loc, 'safe_default.ini')
        default_config = configparser.ConfigParser(allow_no_value=True,
                                                   comment_prefixes=('#', ';', '{'),
                                                   inline_comment_prefixes='#')

        with open(default_config_path, 'r') as f:
            default_config.read_file(f)

        self.default_config = default_config['DEFAULT']

        # Load user-defined settings, if any
        config = configparser.ConfigParser(defaults=default_config['DEFAULT'],
                                           allow_no_value=True,
                                           comment_prefixes=('#', ';', '{'),
                                           inline_comment_prefixes='#')
        config.read(path_to_ini_file)

        if 'Input files' not in config:
            config['Input files'] = {}
        if path_to_safe_data is None:
            path_to_safe_data = config.get('Input files', 'safe_data')  # falls back on default if empty
            if path_to_safe_data == '':
                path_to_safe_data = None
        path_to_network_file = config.get('Input files', 'networkfile')  # falls back on default if empty
        path_to_attribute_file = config.get('Input files', 'annotationfile')  # falls back on default if empty

        self.path_to_safe_data = path_to_safe_data
        if not self.path_to_safe_data is None:
            assert self.path_to_safe_data.endswith('/'), "path_to_safe_data should end with '/', else `os.path.join` may not provide desired output."
            self.path_to_network_file = os.path.join(self.path_to_safe_data, path_to_network_file)
            self.path_to_attribute_file = os.path.join(self.path_to_safe_data, path_to_attribute_file)
        else:
            ## direct paths to the network and attribute files
            self.path_to_network_file = path_to_network_file
            self.path_to_attribute_file = path_to_attribute_file
            
        self.attribute_sign = config.get('Input files', 'annotationsign') # falls back on default if empty

        if 'Analysis parameters' not in config:
            config['Analysis parameters'] = {}

        self.background = config.get('Analysis parameters', 'background')
        self.node_distance_metric = config.get('Analysis parameters', 'nodeDistanceType')
        self.neighborhood_radius_type = config.get('Analysis parameters', 'neighborhoodRadiusType')
        self.neighborhood_radius = float(config.get('Analysis parameters', 'neighborhoodRadius'))

        self.attribute_unimodality_metric = config.get('Analysis parameters', 'unimodalityType')
        self.attribute_distance_metric = config.get('Analysis parameters', 'groupDistanceType')
        self.attribute_distance_threshold = float(config.get('Analysis parameters', 'groupDistanceThreshold'))

        self.output_dir = os.path.dirname(path_to_ini_file)
        if not self.output_dir:
            self.output_dir = loc

    def validate_config(self):

        """
        Test the validity of the current settings in the SAFE class before running the analysis.

        :return: none
        """

        # Check that the option parameters are valid
        if self.background not in ['attribute_file', 'network']:
            user_setting = self.background
            self.background = self.default_config.get('background')    # Restore the default value.
            raise ValueError(('%s is not a valid setting for background. '
                              'Valid options are: attribute_file, network.' % user_setting))

        if self.node_distance_metric not in ['euclidean', 'shortpath', 'shortpath_weighted_layout']:
            user_setting = self.node_distance_metric
            self.node_distance_metric = self.default_config.get('nodeDistanceType')    # Restore the default value.
            raise ValueError(('%s is not a valid setting for node_distance_metric. '
                              'Valid options are: euclidean, shortpath, shortpath_weighted_layout' % user_setting))

        if self.attribute_sign not in ['highest', 'lowest', 'both']:
            user_setting = self.attribute_sign
            self.attribute_sign = self.default_config.get('annotationsign')   # Restore the default value.
            raise ValueError(('%s is not a valid setting for attribute_sign. '
                              'Valid options are: highest, lowest, both' % user_setting))

        if not isinstance(self.num_permutations, int) or (self.num_permutations < 10):
            self.num_permutations = 1000    # Restore the default value.
            raise ValueError('num_permutations must be an integer equal or greater than 10.')

        if not isinstance(self.enrichment_threshold, float) or (self.enrichment_threshold <= 0) or (self.enrichment_threshold >= 1):
            self.enrichment_threshold = 0.05    # Restore the default value.
            raise ValueError('enrichment_threshold must be in the (0,1) range.')

        if not isinstance(self.enrichment_max_log10, (int, float)):
            self.enrichment_max_log10 = 16    # Restore the default value.
            raise ValueError('enrichment_max_log10 must be a number.')

        if not isinstance(self.attribute_enrichment_min_size, int) or (self.attribute_enrichment_min_size < 2):
            self.attribute_enrichment_min_size = 10    # Restore the default value.
            raise ValueError('attribute_enrichment_min_size must be an integer equal or greater than 2.')

        if not isinstance(self.attribute_distance_threshold, float) or (self.attribute_distance_threshold <= 0) or (self.attribute_distance_threshold >= 1):
            self.attribute_distance_threshold = 0.75    # Restore the default value.
            raise ValueError('attribute_enrichment_min_size must be a float number in the (0,1) range.')

    def save(self, output_file='', **kwargs):
        if not output_file:
            output_file = os.path.join(os.getcwd(), 'safe_output.p')

        with open(output_file, 'wb') as handle:
            pickle.dump(self, handle)

    def load_network(self, **kwargs):
        """
        Load the network from a source file and, if necessary, apply a network layout.

        Keyword Args:
            * network_file (:obj:`str`, optional): Path to the file containing the network. Note: if the path to safe data (`path_to_safe_data`) is provided, this would the path inside the `safe_data` folder, else a direct path to the file. 
            * node_key_attribute (:obj:`str`, optional): Name of the node attribute that should be treated as key identifier.

        :return: none
        """

        # Overwriting the global settings, if required
        if 'network_file' in kwargs:
            if self.path_to_safe_data is None:
                self.path_to_network_file = kwargs['network_file']
            else:
                self.path_to_network_file = os.path.join(self.path_to_safe_data, kwargs['network_file'])
            del kwargs['network_file'] ## remove the redundant/old path
        assert os.path.exists(self.path_to_network_file), self.path_to_network_file # os.path.join may misbehave if there are extra '/' at the place where the paths are joined.
        if 'view_name' in kwargs:
            self.view_name = kwargs['view_name']
        if 'node_key_attribute' in kwargs:
            self.node_key_attribute = kwargs['node_key_attribute']

        # Make sure that the settings are still valid
        self.validate_config()

        if type(self.path_to_network_file) == nx.Graph:

            self.graph = self.path_to_network_file

        else:

            # [_, file_extension] = os.path.splitext(self.path_to_network_file)
            file_extension=Path(self.path_to_network_file).suffixes[0] # compatible with double extension e.g. txt.gz
            if self.verbose:
                logging.info('Loading network from %s' % self.path_to_network_file)

            if file_extension == '.mat':
                self.graph = load_network_from_mat(self.path_to_network_file, verbose=self.verbose)
            elif file_extension == '.gpickle':
                self.graph = load_network_from_gpickle(self.path_to_network_file, verbose=self.verbose)
            elif file_extension in ['.txt','.tsv']:
                self.graph = load_network_from_txt(self.path_to_network_file,
                                                   node_key_attribute=self.node_key_attribute,
                                                   verbose=self.verbose)
            elif file_extension == '.cys':
                self.graph = load_network_from_cys(self.path_to_network_file, view_name=self.view_name,
                                                   verbose=self.verbose)
            elif file_extension == '.scatter':
                self.graph = load_network_from_scatter(self.path_to_network_file,
                                                       node_key_attribute=self.node_key_attribute,
                                                       verbose=self.verbose)

        # Setting the node key for mapping attributes
        key_list = nx.get_node_attributes(self.graph, self.node_key_attribute)

        if not bool(key_list):
            raise Exception('The specified node key attribute (%s) does not exist in this network. '
                            'These attributes exist instead: %s. '
                            'Set node_key_attribute to one of these options.'
                            % (self.node_key_attribute, ', '.join(self.graph.node[0].keys())))
        else:
            nx.set_node_attributes(self.graph, key_list, name='key')
            label_list = nx.get_node_attributes(self.graph, 'label')
            self.nodes = pd.DataFrame(data={'id': list(label_list.keys()),
                                            'key': list(key_list.values()),
                                            'label': list(label_list.values())})

    def save_network(self, **kwargs):
        if 'output_file' in kwargs:
            output_file = kwargs['output_file']
        else:
            output_file = os.path.join(os.getcwd(), self.path_to_network_file + '.gpickle')

        nx.write_gpickle(self.graph, output_file)

    def load_attributes(self, **kwargs):
        """
        Preprocess and load the attributes i.e. features of the genes.
        
        Keyword arguments:
            kwargs: parameters provided to `read_attributes` function.
            * attribute_file (:obj:`str`, optional): Path to the file containing the attributes. Note: if path to safe data (`path_to_safe_data`) is provided, this would the path inside the `safe_data` folder, else a direct path to the file. 
        """
        
        # Overwrite the global settings, if required
        if 'attribute_file' in kwargs:
            if self.path_to_safe_data is None or isinstance(kwargs['attribute_file'],pd.DataFrame):
                self.path_to_attribute_file = kwargs['attribute_file']
            elif isinstance(kwargs['attribute_file'],str):
                self.path_to_attribute_file = os.path.join(self.path_to_safe_data, kwargs['attribute_file'])
            else:
                raise ValueError(type(kwargs['attribute_file']))     
            del kwargs['attribute_file'] ## remove the redundant/old path
        if isinstance(self.path_to_attribute_file,str):
            assert os.path.exists(self.path_to_attribute_file), self.path_to_attribute_file # os.path.join may misbehave if there are extra '/' at the place where the paths are joined.
            
        # Make sure that the settings are still valid
        self.validate_config()

        node_label_order = list(nx.get_node_attributes(self.graph, self.node_key_attribute).values())

        if self.verbose and isinstance(self.path_to_attribute_file, str):
            logging.info('Loading attributes from %s' % self.path_to_attribute_file)

        [self.attributes, _, self.node2attribute] = read_attributes(node_label_order=node_label_order,
                                                                    verbose=self.verbose, 
                                                                    attribute_file=self.path_to_attribute_file, 
                                                                    **kwargs)

    def define_neighborhoods(self, **kwargs):
        """
        
        """
        # Overwriting the global settings, if required
        if 'node_distance_metric' in kwargs:
            self.node_distance_metric = kwargs['node_distance_metric']

        if 'neighborhood_radius_type' in kwargs:
            self.neighborhood_radius_type = kwargs['neighborhood_radius_type']

        if 'neighborhood_radius' in kwargs:
            self.neighborhood_radius = kwargs['neighborhood_radius']

        # Make sure that the settings are still valid
        self.validate_config()

        all_shortest_paths = {}
        neighborhoods = np.zeros([self.graph.number_of_nodes(), self.graph.number_of_nodes()], dtype=int)

        if self.node_distance_metric == 'euclidean':
            x = list(dict(self.graph.nodes.data('x')).values())
            nr = self.neighborhood_radius * (np.max(x) - np.min(x))

            x = np.matrix(self.graph.nodes.data('x'))[:, 1]
            y = np.matrix(self.graph.nodes.data('y'))[:, 1]

            node_coordinates = np.concatenate([x, y], axis=1)
            node_distances = squareform(pdist(node_coordinates, 'euclidean'))

            neighborhoods[node_distances < nr] = 1

        else:

            if self.node_distance_metric == 'shortpath_weighted_layout':
                # x = np.matrix(self.graph.nodes.data('x'))[:, 1]
                x = list(dict(self.graph.nodes.data('x')).values())
                nr = self.neighborhood_radius * (np.max(x) - np.min(x))
                all_shortest_paths = dict(nx.all_pairs_dijkstra_path_length(self.graph,
                                                                            weight='length', cutoff=nr))
            elif self.node_distance_metric == 'shortpath':
                nr = self.neighborhood_radius
                all_shortest_paths = dict(nx.all_pairs_dijkstra_path_length(self.graph, cutoff=nr))

            neighbors = [(s, t) for s in all_shortest_paths for t in all_shortest_paths[s].keys()]

            for i in neighbors:
                neighborhoods[i] = 1

            self.node_distances = all_shortest_paths

        # Set diagonal to zero (a node is not part of its own neighborhood)
        # np.fill_diagonal(neighborhoods, 0)

        # Calculate the average neighborhood size
        num_neighbors = np.sum(neighborhoods, axis=1)

        if self.verbose:
            logging.info('Node distance metric: %s' % self.node_distance_metric)
            logging.info('Neighborhood definition: %.2f x %s' % (self.neighborhood_radius, self.neighborhood_radius_type))
            logging.info('Number of nodes per neighborhood (mean +/- std): %.2f +/- %.2f' % (np.mean(num_neighbors), np.std(num_neighbors)))

        self.neighborhoods = neighborhoods

    def compute_pvalues(self, **kwargs):

        if 'how' in kwargs:
            self.enrichment_type = kwargs['how']

        if 'neighborhood_score_type' in kwargs:
            self.neighborhood_score_type = kwargs['neighborhood_score_type']

        if 'multiple_testing' in kwargs:
            self.multiple_testing = kwargs['multiple_testing']

        if 'background' in kwargs:
            self.background = kwargs['background']

        # Make sure that the settings are still valid
        self.validate_config()

        if self.background == 'network':
            logging.info('Setting all null attribute values to 0. Using the network as background for enrichment.')
            self.node2attribute[np.isnan(self.node2attribute)] = 0

        num_vals = self.node2attribute.shape[0]
        num_nans = np.sum(np.isnan(self.node2attribute), axis=0)

        if any(num_nans/num_vals > 0.5):
            logging.warning("WARNING: more than 50% of nodes in the network as set to NaN and will be ignored for calculating enrichment.\n'Consider setting sf.background = ''network''.'")

        # Warn users if more than 50% of values are NaN
        num_other_values = np.sum(~np.isnan(self.node2attribute) & ~np.isin(self.node2attribute, [0, 1]))

        if (self.enrichment_type == 'hypergeometric') or ((self.enrichment_type == 'auto') and (num_other_values == 0)):
            self.compute_pvalues_by_hypergeom(**kwargs)
        else:
            self.compute_pvalues_by_randomization(**kwargs)

        idx = ~np.isnan(self.nes)
        self.nes_binary = np.zeros(self.nes.shape)
        self.nes_binary[idx] = np.abs(self.nes[idx]) > -np.log10(self.enrichment_threshold)

        self.attributes['num_neighborhoods_enriched'] = np.sum(self.nes_binary, axis=0)

    def compute_pvalues_by_randomization(self, **kwargs):

        if kwargs:
            logging.warning('Current settings (possibly overwriting global ones):')
            for k in kwargs:
                logging.warning('\t%s=%s' % (k, str(kwargs[k])))

        logging.info('Using randomization to calculate enrichment...')

        # Pause for 1 sec to prevent the progress bar from showing up too early
        time.sleep(1)

        if 'num_permutations' in kwargs:
            self.num_permutations = kwargs['num_permutations']

        num_processes = 1
        if 'processes' in kwargs:
            num_processes = kwargs['processes']

        # Make sure that the settings are still valid
        self.validate_config()

        N_in_neighborhood_in_group = compute_neighborhood_score(self.neighborhoods,
                                                                self.node2attribute,
                                                                self.neighborhood_score_type)
        self.ns = N_in_neighborhood_in_group

        if num_processes > 1:

            num_permutations_x_process = np.ceil(self.num_permutations / num_processes).astype(int)
            self.num_permutations = num_permutations_x_process * num_processes

            arg_tuple = (self.neighborhoods, self.node2attribute,
                         self.neighborhood_score_type, num_permutations_x_process)
            list_for_parallelization = [arg_tuple] * num_processes

            ctx = mp.get_context('spawn')
            pl = ctx.Pool(processes=num_processes)
            res = pl.map(run_permutations, list_for_parallelization)
            pl.close()
            pl.join()

            [counts_neg_list, counts_pos_list] = map(list, zip(*res))

            counts_neg = np.sum(np.stack(counts_neg_list, axis=2), axis=2)
            counts_pos = np.sum(np.stack(counts_pos_list, axis=2), axis=2)

        else:

            arg_tuple = (self.neighborhoods, self.node2attribute,
                         self.neighborhood_score_type, self.num_permutations)
            [counts_neg, counts_pos] = run_permutations(arg_tuple)

        idx = np.isnan(N_in_neighborhood_in_group)
        counts_neg[idx] = np.nan
        counts_pos[idx] = np.nan

        self.pvalues_neg = counts_neg / self.num_permutations
        self.pvalues_pos = counts_pos / self.num_permutations

        # Correct for multiple testing
        if self.multiple_testing:
            logging.info('Running FDR-adjustment of p-values...')
            out = np.apply_along_axis(fdrcorrection, 1, self.pvalues_neg)
            self.pvalues_neg = out[:, 1, :]

            out = np.apply_along_axis(fdrcorrection, 1, self.pvalues_pos)
            self.pvalues_pos = out[:, 1, :]

        # Log-transform into neighborhood enrichment scores (NES)
        # Necessary conservative adjustment: when p-value = 0, set it to 1/num_permutations
        nes_pos = -np.log10(np.where(self.pvalues_pos == 0, 1/self.num_permutations, self.pvalues_pos))
        nes_neg = -np.log10(np.where(self.pvalues_neg == 0, 1/self.num_permutations, self.pvalues_neg))

        if self.attribute_sign == 'highest':
            self.nes = nes_pos
        elif self.attribute_sign == 'lowest':
            self.nes = nes_neg
        elif self.attribute_sign == 'both':
            self.nes = nes_pos - nes_neg

    def compute_pvalues_by_hypergeom(self, **kwargs):

        if kwargs:
            if 'verbose' in kwargs:
                self.verbose = kwargs['verbose']

            if self.verbose:
                logging.warning('Overwriting global settings:')
                for k in kwargs:
                    logging.warning('\t%s=%s' % (k, str(kwargs[k])))

        # Make sure that the settings are still valid
        self.validate_config()

        if self.verbose:
            logging.info('Using the hypergeometric test to calculate enrichment...')

        # Nodes with not-NaN values in >= 1 attribute
        nodes_not_nan = np.any(~np.isnan(self.node2attribute), axis=1)

        # -- Number of nodes
        # n = self.graph.number_of_nodes()    # total
        n = np.sum(nodes_not_nan)    # with not-NaN values in >=1 attribute

        N = np.zeros([self.graph.number_of_nodes(), len(self.attributes)]) + n

        # -- Number of nodes annotated to each attribute
        N_in_group = np.tile(np.nansum(self.node2attribute, axis=0), (self.graph.number_of_nodes(), 1))

        # -- Number of nodes in each neighborhood
        # neighborhood_size = np.sum(self.neighborhoods, axis=0)[:, np.newaxis]    # total
        neighborhood_size = np.dot(self.neighborhoods,
                                   nodes_not_nan.astype(int))[:, np.newaxis] # with not-NaN values in >=1 attribute

        N_in_neighborhood = np.tile(neighborhood_size, (1, len(self.attributes)))

        # -- Number of nodes in each neighborhood and  annotated to each attribute
        N_in_neighborhood_in_group = np.dot(self.neighborhoods,
                                            np.where(~np.isnan(self.node2attribute), self.node2attribute, 0))

        self.pvalues_pos = hypergeom.sf(N_in_neighborhood_in_group - 1, N, N_in_group, N_in_neighborhood)

        # Correct for multiple testing
        if self.multiple_testing:

            if self.verbose:
                logging.info('Running FDR-adjustment of p-values...')

            out = np.apply_along_axis(fdrcorrection, 1, self.pvalues_pos)
            self.pvalues_pos = out[:, 1, :]

        # Log-transform into neighborhood enrichment scores (NES)
        self.nes = -np.log10(self.pvalues_pos)

    def define_top_attributes(self, **kwargs):

        if 'attribute_unimodality_metric' in kwargs:
            self.attribute_unimodality_metric = kwargs['attribute_unimodality_metric']

        if 'attribute_enrichment_min_size' in kwargs:
            self.attribute_enrichment_min_size = kwargs['attribute_enrichment_min_size']

        # Make sure that the settings are still valid
        self.validate_config()

        logging.info('Criteria for top attributes:')
        logging.info('- minimum number of enriched neighborhoods: %d' % self.attribute_enrichment_min_size)
        logging.info('- region-specific distribution of enriched neighborhoods as defined by: %s' % self.attribute_unimodality_metric)

        self.attributes['top'] = False

        # Requirement 1: a minimum number of enriched neighborhoods
        self.attributes.loc[
            self.attributes['num_neighborhoods_enriched'] >= self.attribute_enrichment_min_size, 'top'] = True

        # Requirement 2: 1 connected component in the subnetwork of enriched neighborhoods
        if self.attribute_unimodality_metric == 'connectivity':

            self.attributes['num_connected_components'] = 0
            self.attributes['size_connected_components'] = None
            self.attributes['size_connected_components'] = self.attributes['size_connected_components'].astype(object)
            self.attributes['num_large_connected_components'] = 0

            for attribute in self.attributes.index.values[self.attributes['top']]:

                enriched_neighborhoods = list(compress(list(self.graph), self.nes_binary[:, attribute] > 0))
                H = nx.subgraph(self.graph, enriched_neighborhoods)

                connected_components = sorted(nx.connected_components(H), key=len, reverse=True)
                num_connected_components = len(connected_components)
                size_connected_components = np.array([len(c) for c in connected_components])
                num_large_connected_components = np.sum(size_connected_components >= self.attribute_enrichment_min_size)

                self.attributes.loc[attribute, 'num_connected_components'] = num_connected_components
                self.attributes.at[attribute, 'size_connected_components'] = size_connected_components
                self.attributes.loc[attribute, 'num_large_connected_components'] = num_large_connected_components

            # Exclude attributes that have more than 1 connected component
            # self.attributes.loc[self.attributes['num_large_connected_components'] > 1, 'top'] = False
            self.attributes.loc[self.attributes['num_connected_components'] > 1, 'top'] = False

        if self.verbose:
            logging.info('Number of top attributes: %d' % np.sum(self.attributes['top']))

    def define_domains(self, **kwargs):

        # Overwriting global settings, if necessary
        if 'attribute_distance_threshold' in kwargs:
            self.attribute_distance_threshold = kwargs['attribute_distance_threshold']

        # Make sure that the settings are still valid
        self.validate_config()

        m = self.nes_binary[:, self.attributes['top']].T
        Z = linkage(m, method='average', metric=self.attribute_distance_metric)
        max_d = np.max(Z[:, 2] * self.attribute_distance_threshold)
        domains = fcluster(Z, max_d, criterion='distance')

        self.attributes['domain'] = 0
        self.attributes.loc[self.attributes['top'], 'domain'] = domains

        # Assign nodes to domains
        node2nes = pd.DataFrame(data=self.nes,
                                    columns=[self.attributes.index.values, self.attributes['domain']])
        node2nes_binary = pd.DataFrame(data=self.nes_binary,
                                           columns=[self.attributes.index.values, self.attributes['domain']])

        # # A node belongs to the domain that contains the attribute
        # for which the node has the highest enrichment
        # self.node2domain = node2nes.groupby(level='domain', axis=1).max()
        # t_max = self.node2domain.loc[:, 1:].max(axis=1)
        # t_idxmax = self.node2domain.loc[:, 1:].idxmax(axis=1)
        # t_idxmax[t_max < -np.log10(self.enrichment_threshold)] = 0

        # A node belongs to the domain that contains the highest number of attributes
        # for which the nodes is significantly enriched
        self.node2domain = node2nes_binary.groupby(level='domain', axis=1).sum()
        t_max = self.node2domain.loc[:, 1:].max(axis=1)
        t_idxmax = self.node2domain.loc[:, 1:].idxmax(axis=1)
        t_idxmax[t_max == 0] = 0

        self.node2domain['primary_domain'] = t_idxmax

        # Get the max NES for the primary domain
        o = node2nes.groupby(level='domain', axis=1).max()
        i = pd.Series(t_idxmax)
        self.node2domain['primary_nes'] = o.lookup(i.index, i.values)

        if self.verbose:
            num_domains = len(np.unique(domains))
            num_attributes_per_domain = self.attributes.loc[self.attributes['domain'] > 0].groupby('domain')['id'].count()
            min_num_attributes = num_attributes_per_domain.min()
            max_num_attributes = num_attributes_per_domain.max()
            logging.info('Number of domains: %d (containing %d-%d attributes)' %
                  (num_domains, min_num_attributes, max_num_attributes))

    def trim_domains(self, **kwargs):

        # Remove domains that are the top choice for less than a certain number of neighborhoods
        domain_counts = np.zeros(len(self.attributes['domain'].unique())).astype(int)
        t = self.node2domain.groupby('primary_domain')['primary_domain'].count()
        domain_counts[t.index] = t.values
        to_remove = np.flatnonzero(domain_counts < self.attribute_enrichment_min_size)

        self.attributes.loc[self.attributes['domain'].isin(to_remove), 'domain'] = 0

        idx = self.node2domain['primary_domain'].isin(to_remove)
        self.node2domain.loc[idx, ['primary_domain', 'primary_nes']] = 0

        # Rename the domains (simple renumber)
        a = np.sort(self.attributes['domain'].unique())
        b = np.arange(len(a))
        renumber_dict = dict(zip(a, b))

        self.attributes['domain'] = [renumber_dict[k] for k in self.attributes['domain']]
        self.node2domain['primary_domain'] = [renumber_dict[k] for k in self.node2domain['primary_domain']]
        self.node2domain.drop(columns=to_remove)

        # Make labels for each domain
        domains = np.sort(self.attributes['domain'].unique())
        domains_labels = self.attributes.groupby('domain')['name'].apply(chop_and_filter)
        self.domains = pd.DataFrame(data={'id': domains, 'label': domains_labels})
        self.domains.set_index('id', drop=False)

        if self.verbose:
            logging.info('Removed %d domains because they were the top choice for less than %d neighborhoods.'
                  % (len(to_remove), self.attribute_enrichment_min_size))

    def plot_network(self, 
                     foreground_color = '#ffffff',
                     background_color='#000000',
                     labels=[],
                     **kwargs_mark_nodes,
                    ):
        """
        Plot the base network.
        
        Parameters:
            labels (list): the genes to show on the network.
        
        Keyword parameters:
            kwargs_mark_nodes: parameters provided to `mark_nodes` function.
        """
        ax=plot_network(self.graph, background_color=background_color)
        # Plot the labels, if any
        if len(labels)!=0:
            ## get the coordinates of the points
            node_xy_labels,labels_found=get_node_coordinates(graph=self.graph,labels=labels)
            ## mark the nodes
            ax=mark_nodes(
                       x=node_xy_labels[:, 0],
                       y=node_xy_labels[:, 1],
                       labels=labels_found,
                       ax=ax,
                       foreground_color=foreground_color,
                       background_color=background_color,
                       **kwargs_mark_nodes,
                      )
        return ax
    
    def plot_composite_network_contours(self,
                                        save_fig=None, clabels=False,
                                        foreground_color = '#ffffff',
                                        background_color='#000000'):
        """
        Show the countours i.e. outlines for the categories of genes. 
        
        Parameters:
            clabels (bool): 
        """
        if background_color == '#ffffff':
            foreground_color = '#000000'

        domains = np.sort(self.attributes['domain'].unique())
        # domains = self.domains.index.values

        # Define colors per domain
        domain2rgb = get_colors('hsv', len(domains))

        # Store domain info
        self.domains['rgba'] = domain2rgb.tolist()

        # Get node coordinates
        node_xy = get_node_coordinates(self.graph)

        # Figure parameters
        num_plots = 2

        nrows = int(np.ceil(num_plots / 2))
        ncols = np.min([num_plots, 2])
        figsize = (10 * ncols, 10 * nrows)

        [fig, axes] = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey=True,
                                   facecolor=background_color)
        axes = axes.ravel()

        # First, plot the network
        ax = axes[0]
        ax = plot_network(self.graph, ax=ax, background_color=background_color)

        # Then, plot the composite network as contours

        for n_domain, domain in enumerate(self.domains['label'].values):
            nodes_indices = self.node2domain.loc[self.node2domain.loc[:, n_domain] > 0,].index.values
            pos3 = node_xy[nodes_indices, :]

            kernel = gaussian_kde(pos3.T)
            [X, Y] = np.mgrid[np.min(pos3[:,0]):np.max(pos3[:,0]):100j, np.min(pos3[:,1]):np.max(pos3[:,1]):100j]
            positions = np.vstack([X.ravel(), Y.ravel()])
            Z = np.reshape(kernel(positions).T, X.shape)

            C = ax[1].contour(X, Y, Z, [1e-6], colors=self.domains.loc[n_domain, 'rgba'], alpha=1)

            if clabels:
                C.levels = [n_domain + 1]
                plt.clabel(C, C.levels, inline=True, fmt='%d', fontsize=16)
                print('%d -- %s' % (n_domain + 1, domain))

        fig.set_facecolor(background_color)

        if save_fig:
            path_to_fig = save_fig
            print('Output path: %s' % path_to_fig)
            plt.savefig(path_to_fig, facecolor=background_color)

    def plot_composite_network(self, show_each_domain=False, show_domain_ids=True,
                               save_fig=None, labels=[],
                               foreground_color = '#ffffff',
                               background_color='#000000'):
        """
        Show the domains i.e. categories of the genes.
        
        Parameters:
            show_each_domain (bool): Show each domain on a separate plot (defaults to False). 
        """
        if background_color == '#ffffff':
            foreground_color = '#000000'

        domains = np.sort(self.attributes['domain'].unique())
        # domains = self.domains.index.values

        # Define colors per domain
        domain2rgb = get_colors('hsv', len(domains))

        # Store domain info
        self.domains['rgba'] = domain2rgb.tolist()

        # Compute composite node colors
        node2nes = pd.DataFrame(data=self.nes,
                                columns=[self.attributes.index.values, self.attributes['domain']])

        node2nes_binary = pd.DataFrame(data=self.nes_binary,
                                       columns=[self.attributes.index.values, self.attributes['domain']])
        node2domain_count = node2nes_binary.groupby(level='domain', axis=1).sum()
        node2all_domains_count = node2domain_count.sum(axis=1)[:, np.newaxis]

        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.matmul(node2domain_count.values, domain2rgb) / node2all_domains_count

        t = np.sum(c, axis=1)
        c[np.isnan(t) | np.isinf(t), :] = [0, 0, 0, 0]

        # Adjust brightness
        coeff_brightness = 0.1 / np.nanmean(np.ravel(c[:, :-1]))
        if coeff_brightness > 1:
            c = c * coeff_brightness
        c = np.clip(c, None, 1)

        # Sort nodes by their overall brightness
        ix = np.argsort(np.sum(c, axis=1))

        node_xy = get_node_coordinates(self.graph)

        # Figure parameters
        num_plots = 2

        if show_each_domain:
            num_plots = num_plots + (len(domains) - 1)

        nrows = int(np.ceil(num_plots/2))
        ncols = np.min([num_plots, 2])
        figsize = (10 * ncols, 10 * nrows)

        [fig, axes] = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey=True,
                                   facecolor=background_color)
        axes = axes.ravel()

        # First, plot the network
        ax = axes[0]
        ax = plot_network(self.graph, ax=ax, background_color=background_color)

        # Then, plot the composite network
        axes[1].scatter(node_xy[ix, 0], node_xy[ix, 1], c=c[ix], s=60, edgecolor=None)
        axes[1].set_aspect('equal')
        axes[1].set_facecolor(background_color)

        # Plot a circle around the network
        plot_network_contour(self.graph, axes[1], background_color=background_color)

        # Plot the labels, if any
        if len(labels)!=0:
            ## get the coordinates of the points
            node_xy_labels,labels_found=get_node_coordinates(graph=self.graph,labels=labels)
            ## mark the nodes
            ax=mark_nodes(
                       x=node_xy_labels[:, 0],
                       y=node_xy_labels[:, 1],
                       kind=['label'],
                       labels=labels_found,
                       ax=axes[1],
                       foreground_color=foreground_color,
                       background_color=background_color,
                      )
            
        if show_domain_ids:
            for domain in domains[domains > 0]:
                idx = self.node2domain['primary_domain'] == domain
                centroid_x = np.nanmean(node_xy[idx, 0])
                centroid_y = np.nanmean(node_xy[idx, 1])
                axes[1].text(centroid_x, centroid_y, str(domain),
                             fontdict={'size': 16, 'color': foreground_color, 'weight': 'bold'})

        # Then, plot each domain separately, if requested
        if show_each_domain:
            for domain in domains[domains > 0]:
                domain_color = np.reshape(domain2rgb[domain, :], (1, 4))

                alpha = node2nes.loc[:, domain].values
                alpha = alpha / self.enrichment_max_log10
                alpha[alpha > 1] = 1
                alpha = np.reshape(alpha, -1)

                c = np.repeat(domain_color, len(alpha), axis=0)
                # c[:, 3] = alpha

                idx = self.node2domain['primary_domain'] == domain
                # ix = np.argsort(c)
                axes[1+domain].scatter(node_xy[idx, 0], node_xy[idx, 1], c=c[idx],
                                       s=60, edgecolor=None)
                axes[1+domain].set_aspect('equal')
                axes[1+domain].set_facecolor(background_color)
                axes[1+domain].set_title('Domain %d\n%s' % (domain, self.domains.loc[domain, 'label']),
                                         color=foreground_color)
                plot_network_contour(self.graph, axes[1+domain], background_color=background_color)

                # Plot the labels, if any
                if len(labels)!=0:
                    ## get the coordinates of the points
                    node_xy_labels,labels_found=get_node_coordinates(graph=self.graph,labels=labels)
                    ## mark the nodes
                    ax=mark_nodes(
                               x=node_xy_labels[:, 0],
                               y=node_xy_labels[:, 1],
                               labels=labels_found,
                               kind=['label'],
                               ax=axes[1+domain],
                               foreground_color=foreground_color,
                               background_color=background_color,
                              )
        fig.set_facecolor(background_color)

        if save_fig:
            path_to_fig = save_fig
            logging.info('Output path: %s' % path_to_fig)
            plt.savefig(path_to_fig, facecolor=background_color)

    def plot_sample_attributes(
            self, attributes=1, top_attributes_only=False,
            show_network=True,
            show_costanzo2016=False, show_costanzo2016_colors=True, show_costanzo2016_clabels=False,
            show_nes=True, show_raw_data=False, show_significant_nodes=False,
            show_colorbar=True, colors=['82add6', 'facb66'],
            foreground_color = '#ffffff',
            background_color='#000000',
            labels: list=[],
            save_fig=None, **kwargs
        ):
        """
        Show attributes i.e. features of the genes.
        
        Parameters:
            attributes (int): number of .. (defaults to 1).
            show_nes (bool): show .. (defaults to True).
            labels (list): show labels (defaults to []).
        """
        if background_color == '#ffffff':
            foreground_color = '#000000'

        all_attributes = self.attributes.index.values
        if top_attributes_only:
            all_attributes = all_attributes[self.attributes['top']]

        if isinstance(attributes, int):
            if attributes < len(all_attributes):
                attributes = np.random.choice(all_attributes, attributes, replace=False)
            else:
                attributes = np.arange(len(all_attributes))
        elif isinstance(attributes, str):
            attributes = [list(self.attributes['name'].values).index(attributes)]
        elif isinstance(attributes, list):
            attributes = [list(self.attributes['name'].values).index(attribute) for attribute in attributes]

        node_xy = get_node_coordinates(self.graph)

        # Figure parameters

        nax = 0
        if show_network:
            nax = 1

        nrows = int(np.ceil((len(attributes)+nax)/2))
        ncols = np.min([len(attributes)+nax, 2])
        figsize = (10*ncols, 10*nrows)

        [fig, axes] = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey=True,
                                   facecolor=background_color)

        if isinstance(axes, np.ndarray):
            axes = axes.ravel()
        else:
            axes = np.array([axes])

        # First, plot the network (if required)
        if show_network:
            ax = axes[0]
            ax = plot_network(self.graph, ax=ax, background_color=background_color)

        score = self.nes

        # Plot the attribute
        for idx_attribute, attribute in enumerate(attributes):

            ax = axes[idx_attribute+nax]

            if show_nes:

                # Dynamically determine the min & max of the colorscale
                if 'vmin' in kwargs:
                    vmin = kwargs['vmin']
                else:
                    vmin = np.nanmin([np.log10(1 / self.num_permutations), np.nanmin(-np.abs(score[:, attribute]))])
                if 'vmax' in kwargs:
                    vmax = kwargs['vmax']
                else:
                    vmax = np.nanmax([-np.log10(1 / self.num_permutations), np.nanmax(np.abs(score[:, attribute]))])
                if 'midrange' in kwargs:
                    midrange = kwargs['midrange']
                else:
                    midrange = [np.log10(0.05), 0, -np.log10(0.05)]

                # Determine the order of points, such that the brightest ones are on top
                idx = np.argsort(np.abs(score[:, attribute]))

                # Colormap
                colors_hex = [colors[0], background_color, background_color, background_color, colors[1]]
                colors_hex = [re.sub(r'^#', '', c) for c in colors_hex]
                colors_rgb = [tuple(int(c[i:i+2], 16)/255 for i in (0, 2, 4)) for c in colors_hex]

                cmap = LinearSegmentedColormap.from_list('my_cmap', colors_rgb)

                sc = ax.scatter(node_xy[idx, 0], node_xy[idx, 1], c=score[idx, attribute],
                                s=60, cmap=cmap, norm=MidpointRangeNormalize(midrange=midrange, vmin=vmin, vmax=vmax),
                                edgecolors=None)

            if show_colorbar:

                pos_ax = ax.get_position()
                w = pos_ax.width*0.75
                x0 = pos_ax.x0 + (pos_ax.width - w)/2
                pos_cax = [x0, pos_ax.y0,  w, pos_ax.height*0.05]
                cax = fig.add_axes(pos_cax)

                cb = plt.colorbar(sc, cax=cax, orientation='horizontal',
                                  ticks=[vmin, midrange[0], midrange[1], midrange[2], vmax],
                                  drawedges=False)

                # pad = 0, shrink = 1,
                # set colorbar label plus label color
                cb.set_label('Neighborhood enrichment p-value', color=foreground_color)

                # set colorbar tick color
                cax.xaxis.set_tick_params(color=foreground_color)

                # set colorbar edgecolor
                cb.outline.set_edgecolor(foreground_color)
                cb.outline.set_linewidth(1)

                # set colorbar ticklabels
                plt.setp(plt.getp(cb.ax.axes, 'xticklabels'), color=foreground_color)

                cb.ax.set_xticklabels([format(r'$10^{%d}$' % vmin),
                                       r'$10^{%d}$' % midrange[0], r'$1$', r'$10^{%d}$' % -midrange[2],
                                       format(r'$10^{-%d}$' % vmax)])

                cax.text(cax.get_xlim()[0], 1, 'Lower than random', verticalalignment='bottom',
                         fontdict={'color': foreground_color})
                cax.text(cax.get_xlim()[1], 1, 'Higher than random', verticalalignment='bottom',
                         horizontalalignment='right', fontdict={'color': foreground_color})

            if show_raw_data:

                with np.errstate(divide='ignore', invalid='ignore'):

                    [s_zero, s_min, s_max] = [5, 5, 55]
                    n = self.node2attribute[:, attribute]

                    n2a = np.abs(n)
                    if set(np.unique(n2a[~np.isnan(n2a)])).issubset([0, 1]):
                        # The attribute is binary
                        s = np.zeros(len(n2a))
                        s[n2a > 0] = s_max
                        n_min = 0
                        n_max = 1
                    else:
                        # The attribute is quantitative
                        [n_min, n_max] = np.nanpercentile(np.unique(n2a), [5, 95])
                        a = (s_max-s_min)/(n_max-n_min)
                        b = s_min - a*n_min
                        s = a * n2a + b
                        s[s < s_min] = s_min
                        s[s > s_max] = s_max

                    # Colormap
                    [neg_color, pos_color, zero_color] = ['#ff1d23', '#00ff44', foreground_color]  # red, green, white

                    idx = self.node2attribute[:, attribute] < 0
                    sc1 = ax.scatter(node_xy[idx, 0], node_xy[idx, 1], s=s[idx], c=neg_color, marker='.')

                    idx = self.node2attribute[:, attribute] > 0
                    sc2 = ax.scatter(node_xy[idx, 0], node_xy[idx, 1], s=s[idx], c=pos_color, marker='.')

                    idx = self.node2attribute[:, attribute] == 0
                    sc3 = ax.scatter(node_xy[idx, 0], node_xy[idx, 1], s=s_zero, c=zero_color, marker='.')

                    # Legend
                    l1 = plt.scatter([], [], s=s_max, c=pos_color, edgecolors='none')
                    l2 = plt.scatter([], [], s=s_min, c=pos_color, edgecolors='none')
                    l3 = plt.scatter([], [], s=s_zero, c=zero_color, edgecolors='none')
                    l4 = plt.scatter([], [], s=s_min, c=neg_color, edgecolors='none')
                    l5 = plt.scatter([], [], s=s_max, c=neg_color, edgecolors='none')

                    legend_labels = ['{0:.2f}'.format(n) for n in [n_max, n_min, 0, -n_min, -n_max]]

                    leg = ax.legend([l1, l2, l3, l4, l5], legend_labels, loc='upper left', bbox_to_anchor=(0, 1),
                                    title='Raw data', scatterpoints=1, fancybox=False,
                                    facecolor=background_color, edgecolor=background_color)

                    for leg_txt in leg.get_texts():
                        leg_txt.set_color(foreground_color)

                    leg_title = leg.get_title()
                    leg_title.set_color(foreground_color)

            
            if show_significant_nodes:
                ## show the significant nodes
                with np.errstate(divide='ignore', invalid='ignore'):
                    ## get the index of the points
                    idx = np.abs(self.nes_binary[:, attribute]) > 0
                mark_nodes(
                           node_xy[idx, 0],
                           node_xy[idx, 1],
                           kind=['mark'],
                           ax=ax,
                           legend_label = ('p < %.2e' % self.enrichment_threshold),
                           foreground_color=foreground_color,
                           background_color=background_color,
                           marker='+',
                          )                
            
            if show_costanzo2016:
                plot_costanzo2016_network_annotations(self.graph, ax, self.path_to_safe_data,
                                                      colors=show_costanzo2016_colors,
                                                      clabels=show_costanzo2016_clabels,
                                                      background_color=background_color)
            
            
            # Plot a circle around the network
            plot_network_contour(self.graph, ax, background_color=background_color)

            # Plot the labels, if any
            if len(labels)!=0:
                ## get the coordinates of the points
                node_xy_labels,labels_found=get_node_coordinates(graph=self.graph,labels=labels)
                ## mark the nodes
                ax=mark_nodes(
                           x=node_xy_labels[:, 0],
                           y=node_xy_labels[:, 1],
                           kind=['label'],
                           labels=labels_found,
                           ax=ax,
                           foreground_color=foreground_color,
                           background_color=background_color,
                          )

            ax.set_aspect('equal')
            ax.set_facecolor(background_color)

            ax.grid(False)
            ax.margins(0.1, 0.1)

            if idx_attribute+nax == 0:
                ax.invert_yaxis()

            title = self.attributes.loc[attribute, 'name']

            title = '\n'.join(textwrap.wrap(title, width=30))
            ax.set_title(title, color=foreground_color)

            ax.set_frame_on(False)

        fig.set_facecolor(background_color)

        if save_fig:
            path_to_fig = save_fig
            if not os.path.isabs(path_to_fig):
                path_to_fig = os.path.join(self.output_dir, save_fig)
            logging.info('Output path: %s' % path_to_fig)
            plt.savefig(path_to_fig, facecolor=background_color)

    def print_output_files(self, **kwargs):

        if 'output_dir' in kwargs:
            self.output_dir = kwargs['output_dir']

        # Domain properties
        path_domains = os.path.join(self.output_dir, 'domain_properties_annotation.txt')
        if self.domains is not None:
            self.domains.drop(labels=[0], axis=0, inplace=True, errors='ignore')
            self.domains.to_csv(path_domains, sep='\t')
            logging.info(path_domains)

        # Attribute properties
        path_attributes = os.path.join(self.output_dir, 'attribute_properties_annotation.txt')
        self.attributes.to_csv(path_attributes, sep='\t')
        logging.info(path_attributes)

        # Node properties
        path_nodes = os.path.join(self.output_dir, 'node_properties_annotation.txt')

        t = nx.get_node_attributes(self.graph, 'key')
        ids = list(t.keys())
        keys = list(t.values())
        t = nx.get_node_attributes(self.graph, 'label')
        labels = list(t.values())
        if self.node2domain is not None:
            domains = self.node2domain['primary_domain'].values
            ness = self.node2domain['primary_nes'].values
            num_domains = self.node2domain[self.domains['id']].sum(axis=1).values
            self.nodes = pd.DataFrame(data={'id': ids, 'key': keys, 'label': labels, 'domain': domains,
                                            'nes': ness, 'num_domains': num_domains})
        else:

            self.nodes = pd.DataFrame(self.nes)
            self.nodes.columns = self.attributes['name']
            self.nodes.insert(loc=0, column='key', value=keys)
            self.nodes.insert(loc=1, column='label', value=labels)

        self.nodes.to_csv(path_nodes, sep='\t')
        logging.info(path_nodes)


def run_safe_batch(attribute_file):

    sf = SAFE()
    sf.load_network()
    sf.define_neighborhoods()

    sf.load_attributes(attribute_file=attribute_file)
    sf.compute_pvalues(num_permutations=1000)

    return sf.nes


if __name__ == '__main__':

    start = time.time()

    parser = argparse.ArgumentParser(description='Run Spatial Analysis of Functional Enrichment (SAFE) on the default Costanzo et al., 2016 network')
    parser.add_argument('path_to_attribute_file', metavar='path_to_attribute_file', type=str,
                        help='Path to the file containing label-to-attribute annotations')

    args = parser.parse_args()

    # Load the attribute file
    [attributes, node_label_order, node2attribute] = read_attributes(args.path_to_attribute_file)

    nr_processes = mp.cpu_count()
    nr_attributes = attributes.shape[0]

    chunk_size = np.ceil(nr_attributes / nr_processes).astype(int)
    chunks = np.array_split(np.arange(nr_attributes), nr_processes)

    all_chunks = []
    for chunk in chunks:
        this_chunk = pd.DataFrame(data=node2attribute[:, chunk], index=node_label_order,
                                  columns=attributes['name'].values[chunk])
        all_chunks.append(this_chunk)

    pool = mp.Pool(processes=nr_processes)

    combined_nes = []

    logging.info('Running SAFE on %d chunks of size %d...' % (nr_processes, chunk_size))
    for res in pool.map_async(run_safe_batch, all_chunks).get():
        combined_nes.append(res)

    all_nes = np.concatenate(combined_nes, axis=1)

    output_file = format('%s_safe_nes.p' % args.path_to_attribute_file)

    logging.info('Saving the results...')
    with open(output_file, 'wb') as handle:
        pickle.dump(all_nes, handle)

