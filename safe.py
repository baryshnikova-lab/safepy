import configparser
import os
import sys
import textwrap
import time
import argparse
import pickle
import copy
import time

# Necessary check to make sure code runs both in Jupyter and in command line
if 'matplotlib' not in sys.modules:
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import multiprocessing as mp

from matplotlib.colors import LinearSegmentedColormap
from matplotlib import ticker, cm
from tqdm import tqdm
from functools import partial
from scipy.stats import hypergeom
from itertools import compress
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
from statsmodels.stats.multitest import fdrcorrection

from safe_io import *
from safe_extras import *
from safe_colormaps import *


class SAFE:

    def __init__(self,
                 path_to_ini_file='',
                 verbose=True):

        self.verbose = verbose

        self.path_to_safe_data = None
        self.path_to_network_file = None
        self.path_to_attribute_file = None

        self.graph = None
        self.node_key_attribute = 'key'

        self.attributes = None
        self.nodes = None
        self.node2attribute = None
        self.num_nodes_per_attribute = None
        self.attribute_sign = 'both'

        self.node_distance_metric = 'shortpath_weighted_layout'
        self.neighborhood_radius_type = None
        self.neighborhood_radius = None

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
        self.read_config(path_to_ini_file)

    def read_config(self, path_to_ini_file):

        # Location of this code
        loc = os.path.dirname(os.path.abspath(__file__))

        # Load default settings
        default_config_path = os.path.join(loc, 'safe_default.ini')
        default_config = configparser.ConfigParser(allow_no_value=True, comment_prefixes=('#', ';', '{'))
        default_config.read_file(open(default_config_path))

        # Load user-defined settings, if any
        config = configparser.ConfigParser(defaults=default_config['DEFAULT'],
                                           allow_no_value=True,
                                           comment_prefixes=('#', ';', '{'))
        config.read(path_to_ini_file)

        if 'Input files' not in config:
            config['Input files'] = {}

        path_to_safe_data = config.get('Input files', 'safe_data')  # falls back on default if empty
        path_to_network_file = config.get('Input files', 'networkfile')  # falls back on default if empty
        path_to_attribute_file = config.get('Input files', 'annotationfile')  # falls back on default if empty

        self.path_to_safe_data = path_to_safe_data
        self.path_to_network_file = os.path.join(path_to_safe_data, path_to_network_file)
        self.path_to_attribute_file = os.path.join(path_to_safe_data, path_to_attribute_file)

        self.attribute_sign = config.get('Input files', 'annotationsign') # falls back on default if empty

        if 'Analysis parameters' not in config:
            config['Analysis parameters'] = {}
        self.node_distance_metric = config.get('Analysis parameters', 'nodeDistanceType')
        self.neighborhood_radius_type = config.get('Analysis parameters', 'neighborhoodRadiusType')
        self.neighborhood_radius = float(config.get('Analysis parameters', 'neighborhoodRadius'))

        self.attribute_unimodality_metric = config.get('Analysis parameters', 'unimodalityType')
        self.attribute_distance_metric = config.get('Analysis parameters', 'groupDistanceType')
        self.attribute_distance_threshold = float(config.get('Analysis parameters', 'groupDistanceThreshold'))

        self.output_dir = os.path.dirname(path_to_ini_file)
        if not self.output_dir:
            self.output_dir = loc

    def save(self, output_file='', **kwargs):
        if not output_file:
            output_file = os.path.join(os.getcwd(), 'safe_output.p')

        with open(output_file, 'wb') as handle:
            pickle.dump(self, handle)

    def load_network(self, **kwargs):

        # Overwriting the global settings, if required
        if 'network_file' in kwargs:
            self.path_to_network_file = kwargs['network_file']
        if 'node_key_attribute' in kwargs:
            self.node_key_attribute = kwargs['node_key_attribute']

        if type(self.path_to_network_file) == nx.Graph:

            self.graph = self.path_to_network_file

        else:

            [_, file_extension] = os.path.splitext(self.path_to_network_file)

            if self.verbose:
                print('Loading network from %s' % self.path_to_network_file)

            if file_extension == '.mat':
                self.graph = load_network_from_mat(self.path_to_network_file, verbose=self.verbose)
            elif file_extension == '.gpickle':
                self.graph = load_network_from_gpickle(self.path_to_network_file, verbose=self.verbose)
                self.node_key_attribute = 'label_orf'
            elif file_extension == '.txt':
                self.graph = load_network_from_txt(self.path_to_network_file, verbose=self.verbose)
            elif file_extension == '.cys':
                self.graph = load_network_from_cys(self.path_to_network_file, verbose=self.verbose)

        # Setting the node key for mapping attributes
        key_list = nx.get_node_attributes(self.graph, self.node_key_attribute)
        nx.set_node_attributes(self.graph, key_list, name='key')

        label_list = nx.get_node_attributes(self.graph, 'label')

        self.nodes = pd.DataFrame(data={'id': list(label_list.keys()),
                                        'key': list(key_list.values()),
                                        'label': list(label_list.values())})

    def load_attributes(self, **kwargs):

        # Overwrite the global settings, if required
        if 'attribute_file' in kwargs:
            self.path_to_attribute_file = kwargs['attribute_file']
        else:
            kwargs['attribute_file'] = self.path_to_attribute_file

        node_label_order = list(nx.get_node_attributes(self.graph, self.node_key_attribute).values())

        if self.verbose and isinstance(self.path_to_attribute_file, str):
            print('Loading attributes from %s' % self.path_to_attribute_file)

        [self.attributes, _, self.node2attribute] = load_attributes(node_label_order=node_label_order,
                                                                    verbose=self.verbose, **kwargs)

    def define_neighborhoods(self, **kwargs):

        # Overwriting the global settings, if required
        if 'node_distance_metric' in kwargs:
            self.node_distance_metric = kwargs['node_distance_metric']

        if 'neighborhood_radius_type' in kwargs:
            self.neighborhood_radius_type = kwargs['neighborhood_radius_type']

        if 'neighborhood_radius' in kwargs:
            self.neighborhood_radius = kwargs['neighborhood_radius']

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

        # Set diagonal to zero (a node is not part of its own neighborhood)
        # np.fill_diagonal(neighborhoods, 0)

        # Calculate the average neighborhood size
        num_neighbors = np.sum(neighborhoods, axis=1)

        if self.verbose:
            print('Node distance metric: %s' % self.node_distance_metric)
            print('Neighborhood definition: %.2f x %s' % (self.neighborhood_radius, self.neighborhood_radius_type))
            print('Number of nodes per neighborhood (mean +/- std): %.2f +/- %.2f' % (np.mean(num_neighbors), np.std(num_neighbors)))

        self.neighborhoods = neighborhoods

    def compute_pvalues(self, **kwargs):

        if 'how' in kwargs:
            self.enrichment_type = kwargs['how']

        if 'neighborhood_score_type' in kwargs:
            self.neighborhood_score_type = kwargs['neighborhood_score_type']

        if 'multiple_testing' in kwargs:
            self.multiple_testing = kwargs['multiple_testing']

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
            print('Current settings (possibly overwriting global ones):')
            for k in kwargs:
                print('\t%s=%s' % (k, str(kwargs[k])))

        print('Using randomization to calculate enrichment...')

        # Pause for 1 sec to prevent the progress bar from showing up too early
        time.sleep(1)

        if 'num_permutations' in kwargs:
            self.num_permutations = kwargs['num_permutations']

        num_processes = 1
        if 'processes' in kwargs:
            num_processes = kwargs['processes']

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
            print('Running FDR-adjustment of p-values...')
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
                print('Overwriting global settings:')
                for k in kwargs:
                    print('\t%s=%s' % (k, str(kwargs[k])))

        if self.verbose:
            print('Using the hypergeometric test to calculate enrichment...')

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
                print('Running FDR-adjustment of p-values...')

            out = np.apply_along_axis(fdrcorrection, 1, self.pvalues_pos)
            self.pvalues_pos = out[:, 1, :]

        # Log-transform into neighborhood enrichment scores (NES)
        self.nes = -np.log10(self.pvalues_pos)

    def define_top_attributes(self, **kwargs):

        if 'attribute_unimodality_metric' in kwargs:
            self.attribute_unimodality_metric = kwargs['attribute_unimodality_metric']

        print('Criteria for top attributes:')
        print('- minimum number of enriched neighborhoods: %d' % self.attribute_enrichment_min_size)
        print('- region-specific distribution of enriched neighborhoods as defined by: %s' % self.attribute_unimodality_metric)

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
            print('Number of top attributes: %d' % np.sum(self.attributes['top']))

    def define_domains(self, **kwargs):

        # Overwriting global settings, if necessary
        if 'attribute_distance_threshold' in kwargs:
            self.attribute_distance_threshold = kwargs['attribute_distance_threshold']

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
        # self.node2domain = node2es.groupby(level='domain', axis=1).max()
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
            print('Number of domains: %d (containing %d-%d attributes)' %
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

        # Make labels for each domain
        domains = np.sort(self.attributes['domain'].unique())
        domains_labels = self.attributes.groupby('domain')['name'].apply(chop_and_filter)
        self.domains = pd.DataFrame(data={'id': domains, 'label': domains_labels})
        self.domains.set_index('id', drop=False)

        if self.verbose:
            print('Removed %d domains because they were the top choice for less than %d neighborhoods.'
                  % (len(to_remove), self.attribute_enrichment_min_size))

    def plot_network(self):

        plot_network(self.graph)

    def plot_composite_network(self, show_each_domain=False, show_domain_ids=True):

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
                                   facecolor='#000000')
        axes = axes.ravel()

        # First, plot the network
        ax = axes[0]
        ax = plot_network(self.graph, ax=ax)

        # Then, plot the composite network
        axes[1].scatter(node_xy[ix, 0], node_xy[ix, 1], c=c[ix], s=60, edgecolor=None)
        axes[1].set_aspect('equal')
        axes[1].set_facecolor('#000000')

        # Plot a circle around the network
        plot_network_contour(self.graph, axes[1])

        if show_domain_ids:
            for domain in domains[domains > 0]:
                idx = self.node2domain['primary_domain'] == domain
                centroid_x = np.nanmean(node_xy[idx, 0])
                centroid_y = np.nanmean(node_xy[idx, 1])
                axes[1].text(centroid_x, centroid_y, str(domain),
                             fontdict={'size': 16, 'color': 'white', 'weight': 'bold'})

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
                axes[1+domain].set_facecolor('#000000')
                axes[1+domain].set_title('Domain %d\n%s' % (domain, self.domains.loc[domain, 'label']),
                                         color='#ffffff')
                plot_network_contour(self.graph, axes[1+domain])

        fig.set_facecolor("#000000")

    def plot_sample_attributes(self, attributes=1, top_attributes_only=False,
                               show_network=True,
                               show_costanzo2016=False, show_costanzo2016_legend=True,
                               show_raw_data=False, show_significant_nodes=False,
                               show_colorbar=True, colors=['82add6', 'facb66'],
                               labels=[],
                               save_fig=None, **kwargs):

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

        [fig, axes] = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey=True)

        if isinstance(axes, np.ndarray):
            axes = axes.ravel()
        else:
            axes = np.array([axes])

        # First, plot the network (if required)
        if show_network:
            ax = axes[0]
            _ = plot_network(self.graph, ax=ax)

        score = self.nes

        # Plot the attribute
        for idx_attribute, attribute in enumerate(attributes):

            ax = axes[idx_attribute+nax]

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
            colors_hex = [colors[0], '000000', '000000', '000000', colors[1]]
            colors_rgb = [tuple(int(c[i:i+2], 16)/255 for i in (0, 2, 4)) for c in colors_hex]

            cmap = LinearSegmentedColormap.from_list('my_cmap', colors_rgb)

            sc = ax.scatter(node_xy[idx, 0], node_xy[idx, 1], c=score[idx, attribute], vmin=vmin, vmax=vmax,
                            s=60, cmap=cmap, norm=MidpointRangeNormalize(midrange=midrange, vmin=vmin, vmax=vmax),
                            edgecolors=None)

            if not show_network:
                ax.invert_yaxis()

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
                cb.set_label('Neighborhood enrichment p-value', color='w')

                # set colorbar tick color
                cax.xaxis.set_tick_params(color='w')

                # set colorbar edgecolor
                cb.outline.set_edgecolor('white')
                cb.outline.set_linewidth(1)

                # set colorbar ticklabels
                plt.setp(plt.getp(cb.ax.axes, 'xticklabels'), color='w')

                cb.ax.set_xticklabels([format(r'$10^{%d}$' % vmin),
                                       r'$10^{%d}$' % midrange[0], r'$1$', r'$10^{%d}$' % -midrange[2],
                                       format(r'$10^{-%d}$' % vmax)])

                cax.text(cax.get_xlim()[0], 1, 'Lower than random', verticalalignment='bottom', fontdict={'color': 'w'})
                cax.text(cax.get_xlim()[1], 1, 'Higher than random', verticalalignment='bottom',
                         horizontalalignment='right', fontdict={'color': 'w'})

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
                    [neg_color, pos_color, zero_color] = ['#ff1d23', '#00ff44', '#ffffff']  # red, green, white

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
                                    facecolor='#000000', edgecolor='#000000')

                    for leg_txt in leg.get_texts():
                        leg_txt.set_color('#ffffff')

                    leg_title = leg.get_title()
                    leg_title.set_color('#ffffff')

            if show_significant_nodes:

                with np.errstate(divide='ignore', invalid='ignore'):

                    idx = np.abs(self.nes_binary[:, attribute]) > 0
                    sn1 = ax.scatter(node_xy[idx, 0], node_xy[idx, 1], c='w', marker='+')

                # Legend
                leg = ax.legend([sn1], ['p < 0.05'], loc='upper left', bbox_to_anchor=(0, 1),
                                title='Significance', scatterpoints=1, fancybox=False,
                                facecolor='#000000', edgecolor='#000000')

                for leg_txt in leg.get_texts():
                    leg_txt.set_color('#ffffff')

                leg_title = leg.get_title()
                leg_title.set_color('#ffffff')

            if show_costanzo2016:
                plot_costanzo2016_network_annotations(self.graph, ax, self.path_to_safe_data)

            # Plot a circle around the network
            plot_network_contour(self.graph, ax)

            if labels:
                plot_labels(labels, self.graph, ax)

            ax.set_aspect('equal')
            ax.set_facecolor('#000000')

            ax.grid(False)
            ax.margins(0.1, 0.1)

            title = self.attributes.loc[attribute, 'name']

            title = '\n'.join(textwrap.wrap(title, width=30))
            ax.set_title(title, color='#ffffff')

            ax.set_frame_on(False)

        fig.set_facecolor("#000000")

        if save_fig:
            path_to_fig = save_fig
            # if not os.path.isabs(path_to_fig):
            #     path_to_fig = os.path.join(self.output_dir, save_fig)
            print('Output path: %s' % path_to_fig)
            plt.savefig(path_to_fig, facecolor='k')

    def print_output_files(self, **kwargs):

        if 'output_dir' in kwargs:
            self.output_dir = kwargs['output_dir']

        # Domain properties
        path_domains = os.path.join(self.output_dir, 'domain_properties_annotation.txt')
        if self.domains is not None:
            self.domains.drop(labels=[0], axis=0, inplace=True, errors='ignore')
            self.domains.to_csv(path_domains, sep='\t')
            print(path_domains)

        # Attribute properties
        path_attributes = os.path.join(self.output_dir, 'attribute_properties_annotation.txt')
        self.attributes.to_csv(path_attributes, sep='\t')
        print(path_attributes)

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
        print(path_nodes)


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
    [attributes, node_label_order, node2attribute] = load_attributes(args.path_to_attribute_file)

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

    print('Running SAFE on %d chunks of size %d...' % (nr_processes, chunk_size))
    for res in pool.map_async(run_safe_batch, all_chunks).get():
        combined_nes.append(res)

    all_nes = np.concatenate(combined_nes, axis=1)

    output_file = format('%s_safe_nes.p' % args.path_to_attribute_file)

    print('Saving the results...')
    with open(output_file, 'wb') as handle:
        pickle.dump(all_nes, handle)

