import configparser
import os
import sys
import textwrap
import time
import argparse
import pickle

# Necessary check to make sure code runs both in Jupyter and in command line
if 'matplotlib' not in sys.modules:
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import multiprocessing as mp

from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
from functools import partial

from safe_io import *
from safe_colormaps import *
# from safe.safe_io import *
# from .safe_colormaps import *


class SAFE:

    def __init__(self,
                 path_to_ini_file='',
                 verbose=True):

        self.verbose = verbose

        self.path_to_safe_data = None
        self.path_to_network_file = None
        self.path_to_attribute_file = None

        self.graph = None
        self.node_key_attribute = 'label_orf'

        self.attributes = None
        self.node2attribute = None
        self.num_nodes_per_attribute = None

        self.node_distance_metric = None
        self.neighborhood_radius_type = None
        self.neighborhood_radius = None

        # Read both default and user-defined settings
        self.read_config(path_to_ini_file)

        self.neighborhoods = None

        self.num_permutations = 1000
        self.pvalues_neg = None
        self.pvalues_pos = None
        self.opacity = None
        self.opacity_threshold = None
        self.opacity_binary = None

        self.num_nodes_enriched_per_attribute = None
        self.num_attributes_enriched_per_node = None

        self.attribute_enrichment_min_size = 10
        self.max_log10_pval = 16
        self.threshold_enrichment = 0.05

        # Output
        self.output_dir = ''

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

        if 'Analysis parameters' not in config:
            config['Analysis parameters'] = {}
        self.node_distance_metric = config.get('Analysis parameters', 'nodeDistanceType')
        self.neighborhood_radius_type = config.get('Analysis parameters', 'neighborhoodRadiusType')
        self.neighborhood_radius = config.get('Analysis parameters', 'neighborhoodRadius')

    def load_network(self, **kwargs):

        # Overwriting the global settings, if required
        if 'network_file' in kwargs:
            self.path_to_network_file = kwargs['network_file']
        if 'node_key_attribute' in kwargs:
            self.node_key_attribute = kwargs['node_key_attribute']

        [_, file_extension] = os.path.splitext(self.path_to_network_file)

        if self.verbose:
            print('Loading network from %s' % self.path_to_network_file)

        if file_extension == '.mat':
            self.graph = load_network_from_mat(self.path_to_network_file, verbose=self.verbose)
        elif file_extension == '.gpickle':
            self.graph = load_network_from_gpickle(self.path_to_network_file, verbose=self.verbose)

        # Setting the node key for mapping attributes
        key_list = nx.get_node_attributes(self.graph, self.node_key_attribute)
        nx.set_node_attributes(self.graph, key_list, name='key')

    def load_attributes(self, **kwargs):

        # Overwriting the global settings, if required
        if 'attribute_file' in kwargs:
            self.path_to_attribute_file = kwargs['attribute_file']

        node_label_order = list(nx.get_node_attributes(self.graph, 'key').values())

        if self.verbose:
            print('Loading attributes from %s' % self.path_to_attribute_file)

        [self.attributes, self.node2attribute] = load_attributes(self.path_to_attribute_file, node_label_order)
        # self.num_nodes_per_attribute = np.nansum(self.node2attribute, axis=0)

    def define_neighborhoods(self, **kwargs):

        # Overwriting the global settings, if required
        if 'node_distance_metric' in kwargs:
            self.node_distance_metric = kwargs['node_distance_metric']

        all_shortest_paths = {}

        if self.node_distance_metric == 'shortpath_weighted_layout':
            x = np.matrix(self.graph.nodes.data('x'))[:, 1]
            neighborhood_radius = 10 * (np.max(x) - np.min(x)) / 100
            all_shortest_paths = dict(nx.all_pairs_dijkstra_path_length(self.graph,
                                                                        weight='length', cutoff=neighborhood_radius))
        elif self.node_distance_metric == 'shortpath':
            all_shortest_paths = dict(nx.all_pairs_dijkstra_path_length(self.graph, cutoff=1))

        neighbors = [(s, t) for s in all_shortest_paths for t in all_shortest_paths[s].keys()]

        neighborhoods = np.zeros([self.graph.number_of_nodes(), self.graph.number_of_nodes()], dtype=int)
        for i in neighbors:
            neighborhoods[i] = 1

        # Set diagonal to zero (a node is not part of its own neighborhood)
        # np.fill_diagonal(neighborhoods, 0)

        self.neighborhoods = neighborhoods

    def compute_pvalues(self, **kwargs):

        def compute_neighborhood_score(neighborhood2node, node2attribute):

            with np.errstate(invalid='ignore', divide='ignore'):

                A = neighborhood2node
                B = np.where(~np.isnan(node2attribute), node2attribute, 0)

                NA = A
                NB = np.where(~np.isnan(node2attribute), 1, 0)

                AB = np.dot(A, B)
                N = np.dot(NA, NB)

                M = AB / N

                A2B2 = np.dot(np.power(A, 2), np.power(B, 2))
                MAB = M * AB
                M2 = N * np.power(M, 2)

                std = np.sqrt(A2B2 - 2 * MAB + M2)

                neighborhood_score = AB / std

            return neighborhood_score

        if 'num_permutations' in kwargs:
            self.num_permutations = kwargs['num_permutations']

        N_in_neighborhood_in_group = compute_neighborhood_score(self.neighborhoods, self.node2attribute)

        n2a = self.node2attribute.copy()
        indx_vals = np.nonzero(np.sum(~np.isnan(n2a), axis=1))[0]

        counts_neg = np.zeros(N_in_neighborhood_in_group.shape)
        counts_pos = np.zeros(N_in_neighborhood_in_group.shape)
        for _ in tqdm(np.arange(self.num_permutations)):

            # Permute the rows that have values
            n2a[indx_vals, :] = n2a[np.random.permutation(indx_vals), :]

            N_in_neighborhood_in_group_perm = compute_neighborhood_score(self.neighborhoods, n2a)

            with np.errstate(invalid='ignore', divide='ignore'):
                counts_neg = np.add(counts_neg, N_in_neighborhood_in_group_perm <= N_in_neighborhood_in_group)
                counts_pos = np.add(counts_pos, N_in_neighborhood_in_group_perm >= N_in_neighborhood_in_group)

        self.pvalues_neg = counts_neg / self.num_permutations
        self.pvalues_pos = counts_pos / self.num_permutations

        # Necessary conservative adjustment: when p-value = 0, set it to 1/num_permutations
        opacity_pos = -np.log10(np.where(self.pvalues_pos > 0, self.pvalues_pos, 1/self.num_permutations))
        opacity_neg = -np.log10(np.where(self.pvalues_neg > 0, self.pvalues_neg, 1/self.num_permutations))

        self.opacity = opacity_pos - opacity_neg

    def plot_sample_attributes(self, attributes=1, significant_attributes_only=False,
                               show_costanzo2016=True, show_costanzo2016_legend=True,
                               show_raw_data=False, show_significant_nodes=False,
                               show_colorbar=False,
                               labels=[],
                               save_fig=None):

        all_attributes = self.attributes.index.values
        if significant_attributes_only:
            all_attributes = all_attributes[self.num_nodes_enriched_per_attribute > self.attribute_enrichment_min_size]

        if isinstance(attributes, int):
            attributes = np.random.choice(all_attributes, attributes, replace=False)

        x = dict(self.graph.nodes.data('x'))
        y = dict(self.graph.nodes.data('y'))

        ds = [x, y]
        pos = {}
        for k in x:
            pos[k] = np.array([d[k] for d in ds])

        pos2 = np.vstack(list(pos.values()))

        # Figure parameters
        nrows = int(np.ceil(len(attributes)/2))
        ncols = np.min([len(attributes), 2])
        figsize = (10*ncols, 10*nrows)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, facecolor='black')

        if len(attributes) == 1:
            axes = np.array(axes)

        axes = axes.ravel()

        vmin = np.log10(1/self.num_permutations)
        vmax = -np.log10(1/self.num_permutations)
        midrange = [np.log10(0.05), -np.log10(0.05)]

        # Plot the attribute
        for idx_attribute, attribute in enumerate(attributes):

            ax = axes[idx_attribute]

            # Determine the order of points, such that the brightest ones are on top
            idx = np.argsort(np.abs(self.opacity[:, attribute]))

            # Colormap
            colors_hex = ['82add6', '000000', 'facb66']
            colors_rgb = [tuple(int(c[i:i+2], 16)/255 for i in (0, 2, 4)) for c in colors_hex]

            cmap = LinearSegmentedColormap.from_list('test', colors_rgb)

            sc = ax.scatter(pos2[idx, 0], pos2[idx, 1], c=self.opacity[idx, attribute], vmin=vmin, vmax=vmax,
                            s=60, cmap=cmap, norm=MidpointRangeNormalize(midrange=midrange),
                            edgecolors=None)

            if show_colorbar:
                cb = ax.figure.colorbar(sc, ax=ax,
                                        orientation='horizontal',
                                        pad=0.05,
                                        shrink=0.5,
                                        ticks=np.linspace(vmin, vmax, 3),
                                        drawedges=True)

                # set colorbar label plus label color
                cb.set_label('-log10 p-value', color='w')

                # set colorbar tick color
                cb.ax.xaxis.set_tick_params(color='w')

                # set colorbar edgecolor
                cb.outline.set_edgecolor('white')
                cb.outline.set_linewidth(1)

                # set colorbar ticklabels
                plt.setp(plt.getp(cb.ax.axes, 'xticklabels'), color='w')

            if show_raw_data:
                s_min = 5
                s_max = 100
                n = self.node2attribute[:, attribute]

                n2a = np.abs(n)
                a = (s_max-s_min)/(np.nanmax(n2a)-np.nanmin(n2a))
                b = s_min - a*np.nanmin(n2a)
                s = a * n2a + b

                sgn = np.sign(n)
                sgn = np.where(np.isnan(sgn), 0, sgn)
                sgn = sgn.astype(int) + 1

                alpha = np.abs(sgn-1).astype(float)

                # Colormap
                clrs = [(1, 0, 0), (0, 0, 0), (0, 1, 0)]
                # cmap = LinearSegmentedColormap.from_list('test', colors)

                ax.scatter(pos2[:, 0], pos2[:, 1], s=s, c=np.array(clrs)[sgn])

            if show_significant_nodes:
                idx = self.opacity[:, attribute] > -np.log10(0.05)
                ax.scatter(pos2[idx, 0], pos2[idx, 1], c='g', marker='+')
                idx = self.opacity[:, attribute] < np.log10(0.05)
                ax.scatter(pos2[idx, 0], pos2[idx, 1], c='r', marker='+')

            if show_costanzo2016:
                plot_costanzo2016_network_annotations(self.graph, ax, self.path_to_safe_data)

            # Plot a circle around the network
            plot_network_contour(self.graph, ax)

            if labels:
                plot_labels(labels, self.graph, ax)

            ax.set_aspect('equal')
            ax.set_facecolor('#000000')

            ax.grid(False)
            ax.invert_yaxis()
            ax.margins(0.1, 0.1)

            title = self.attributes.loc[attribute, 'attribute_name']

            title = '\n'.join(textwrap.wrap(title, width=30))
            ax.set_title(title, color='#ffffff')

            plt.axis('off')

        if save_fig:
            plt.savefig(save_fig, facecolor='k')

        return ax


# def run_safe_batch(sf, attribute_file):
#
#     print('Loading attributes')
#     sf.load_attributes(attribute_file=attribute_file)
#
#     print('Computing p-values')
#     sf.compute_pvalues(num_permutations=1000)
#
#     return sf


if __name__ == '__main__':

    start = time.time()

    parser = argparse.ArgumentParser(description='Run Spatial Analysis of Functional Enrichment (SAFE) on the default Costanzo et al., 2016 network')
    parser.add_argument('path_to_attribute_file', metavar='path_to_attribute_file', type=str,
                        help='Path to the file containing label-to-attribute annotations')

    args = parser.parse_args()

    nr_processes = mp.cpu_count()

    sf = SAFE()
    sf.load_network()
    sf.define_neighborhoods()

    print('Loading attributes')
    sf.load_attributes(attribute_file=args.path_to_attribute_file)

    print('Computing p-values')
    sf.compute_pvalues(num_permutations=1000)

    # # Break the list into smaller chunks of 200 images and process the chunks sequentially
    # chunk_size = int(np.ceil(attributes.shape[1]/nr_processes))
    # chunks = np.arange(0, attributes.shape[1], chunk_size)
    #
    # print(chunk_size)
    #
    # all_data = pd.DataFrame()
    #
    # for ix_chunk in chunks:
    #     ix_chunk_start = ix_chunk
    #     ix_chunk_stop = np.min([ix_chunk + chunk_size - 1, attributes.shape[1]]) + 1
    #
    #     attributes_this = attributes.iloc[:, ix_chunk_start:ix_chunk_stop]
    #
    #     pool = mp.Pool(processes=nr_processes)
    #     func = partial(run_safe_batch, sf)
    #
    #     for res in pool.map_async(func, attributes_this).get():
    #         all_data = np.concatenate((all_data, res.opacity), axis=1)
    #
    #     print('Execution time: %.2f seconds' % (time.time() - start))

    output_file = format('%s_safe_opacity.p' % args.path_to_attribute_file)

    with open(output_file, 'wb') as handle:
        pickle.dump(sf.opacity, handle)
