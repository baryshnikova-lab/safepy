import re
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.io as spio
import pandas as pd

from os.path import expanduser
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde
from scipy.optimize import fmin


def load_network_from_gpickle(filename, verbose=True):

    filename = re.sub('~', expanduser('~'), filename)
    G = nx.read_gpickle(filename)

    return G


def load_network_from_mat(filename, verbose=True):

    filename = re.sub('~', expanduser('~'), filename)

    if verbose:
        print('Loading the mat file...')

    mat = load_mat(filename)
    G = nx.Graph(mat['layout']['edges'])

    for n in G:
        G.nodes[n]['label'] = mat['layout']['label'][n]
        G.nodes[n]['label_orf'] = mat['layout']['label_orf'][n]
        G.nodes[n]['x'] = mat['layout']['x'][n]
        G.nodes[n]['y'] = mat['layout']['y'][n]

    # Calculate the lengths of the edges

    if verbose:
        print('Calculating the edge lengths...')

    x = np.matrix(G.nodes.data('x'))[:, 1]
    y = np.matrix(G.nodes.data('y'))[:, 1]

    node_coordinates = np.concatenate([x, y], axis=1)
    node_distances = squareform(pdist(node_coordinates, 'euclidean'))

    adjacency_matrix = np.array(nx.adjacency_matrix(G).todense())
    adjacency_matrix = adjacency_matrix.astype('float')
    adjacency_matrix[adjacency_matrix == 0] = np.nan

    edge_lengths = np.multiply(node_distances, adjacency_matrix)

    edge_attr_dict = {index: v for index, v in np.ndenumerate(edge_lengths) if ~np.isnan(v)}
    nx.set_edge_attributes(G, edge_attr_dict, 'length')

    return G


def load_attributes(attribute_file='', node_label_order=[], verbose=True):

    node2attribute = pd.DataFrame()
    attributes = pd.DataFrame()

    if isinstance(attribute_file, str):

        file_name = re.sub('~', expanduser('~'), attribute_file)
        [_, file_extension] = os.path.splitext(file_name)

        if file_extension == '.mat':
            mat = load_mat(file_name)

            node2attribute = pd.DataFrame(data=np.transpose(mat['go']['term2orf']),
                                          index=mat['go']['orfs'],
                                          columns=mat['go']['term_ids'])
            node2attribute = node2attribute.apply(pd.to_numeric, downcast='unsigned')

            data = {'attribute_id': mat['go']['term_ids'], 'attribute_name': mat['go']['term_names']}
            attributes = pd.DataFrame(data=data)

        elif file_extension == '.txt':

            node2attribute = pd.read_table(file_name)
            node2attribute.set_index(node2attribute.columns[0], drop=True, inplace=True)
            node2attribute = node2attribute.apply(pd.to_numeric, downcast='float', errors='coerce')

            data = {'attribute_id': np.arange(len(node2attribute.columns)), 'attribute_name': node2attribute.columns}
            attributes = pd.DataFrame(data=data)

            node2attribute.columns = np.arange(len(node2attribute.columns))

    elif isinstance(attribute_file, pd.DataFrame):

        node2attribute = attribute_file
        data = {'attribute_id': np.arange(len(node2attribute.columns)), 'attribute_name': node2attribute.columns}
        attributes = pd.DataFrame(data=data)

    node2attribute = node2attribute.reindex(index=node_label_order, fill_value=np.nan)
    node2attribute = node2attribute.as_matrix()

    if verbose:
        print('Number of attributes: %d' % attributes.shape[0])
        print('Values: %d NaNs' % np.sum(np.isnan(node2attribute)))
        print('Values: %d zeros' % np.sum(node2attribute[~np.isnan(node2attribute)] == 0))
        print('Values: %d positives' % np.sum(node2attribute[~np.isnan(node2attribute)] > 0))
        print('Values: %d negatives' % np.sum(node2attribute[~np.isnan(node2attribute)] < 0))

    return attributes, node2attribute


def plot_network(G):

    x = dict(G.nodes.data('x'))
    y = dict(G.nodes.data('y'))

    ds = [x, y]
    pos = {}
    for k in x:
        pos[k] = np.array([d[k] for d in ds])

    fig = plt.figure(facecolor='black', edgecolor='white', figsize=(20, 10))
    nx.draw(G, pos=pos, node_color='#ffffff', edge_color='#ffffff', node_size=10, width=1, alpha=0.2)

    fig.set_facecolor("#000000")
    plt.gca().set_aspect('equal')


def plot_network_contour(graph, ax):

    x = dict(graph.nodes.data('x'))
    y = dict(graph.nodes.data('y'))

    ds = [x, y]
    pos = {}
    for k in x:
        pos[k] = np.array([d[k] for d in ds])

    # Compute the convex hull to delineate the network
    hull = ConvexHull(np.vstack(list(pos.values())))

    vertices_x = [pos.get(v)[0] for v in hull.vertices]
    vertices_y = [pos.get(v)[1] for v in hull.vertices]

    vertices_x = np.array(vertices_x)
    vertices_y = np.array(vertices_y)

    # Find center of mass and radius to approximate the hull with a circle
    xm = np.nanmean(vertices_x)
    ym = np.nanmean(vertices_y)

    rm = np.nanmean(np.sqrt((vertices_x - xm) ** 2 + (vertices_y - ym) ** 2))

    # Best fit a circle to these points
    def err(x0):
        [w, v, r] = x0
        pts = [np.linalg.norm([x - w, y - v]) - r for x, y in zip(vertices_x, vertices_y)]
        return (np.array(pts) ** 2).sum()

    [xf, yf, rf] = fmin(err, [xm, ym, rm], disp=False)

    circ = plt.Circle((xf, yf), radius=rf*1.01, color='w', linewidth=1, fill=False)
    ax.add_patch(circ)


def plot_costanzo2016_network_annotations(graph, ax, path_to_data):

    path_to_network_annotations = 'other/Data File S5_SAFE analysis_Gene cluster identity and functional enrichments.xlsx'
    filename = os.path.join(path_to_data, path_to_network_annotations)

    costanzo2016 = pd.read_excel(filename, sheet_name='Global net. cluster gene list')
    processes = costanzo2016['Global Similarity Network Region name'].unique()
    processes = processes[pd.notnull(processes)]

    process_colors = pd.read_table(os.path.join(path_to_data, 'other/costanzo_2016_colors.txt'))
    process_colors = process_colors[['R', 'G', 'B']].as_matrix()/256

    labels = nx.get_node_attributes(graph, 'label')
    # labels = nx.get_node_attributes(graph, 'shared_name')
    labels_dict = {k: v for v, k in labels.items()}

    x = list(dict(graph.nodes.data('x')).values())
    y = list(dict(graph.nodes.data('y')).values())

    pos = {}
    for idx, k in enumerate(x):
        pos[idx] = np.array([x[idx], y[idx]])

    for n_process, process in enumerate(processes):
        nodes = costanzo2016.loc[costanzo2016['Global Similarity Network Region name'] == process, 'Gene/Allele']
        nodes_indices = [labels_dict[node] for node in nodes if node in labels_dict.keys()]

        pos3 = {idx: pos[node_index] for idx, node_index in enumerate(nodes_indices)}
        pos3 = np.vstack(list(pos3.values()))

        kernel = gaussian_kde(pos3.T)
        [X, Y] = np.mgrid[np.min(x):np.max(x):100j, np.min(y):np.max(y):100j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = np.reshape(kernel(positions).T, X.shape)

        C = ax.contour(X, Y, Z, [1e-6], colors=[tuple(process_colors[n_process, :])], alpha=0.75)

        C.levels = [n_process+1]
        plt.clabel(C, C.levels, inline=True, fmt='%d', fontsize=16)

        # print('%d: %s' % (n_process+1, process))


def plot_labels(labels, graph, ax):

    node_labels = nx.get_node_attributes(graph, 'label')
    node_labels_dict = {k: v for v, k in node_labels.items()}

    x = list(dict(graph.nodes.data('x')).values())
    y = list(dict(graph.nodes.data('y')).values())

    x_offset = (np.nanmax(x) - np.nanmin(x))*0.01

    idx = [node_labels_dict[x] for x in labels if x in node_labels_dict.keys()]
    labels_idx = [x for x in labels if x in node_labels_dict.keys()]
    x_idx = [x[i] for i in idx]
    y_idx = [y[i] for i in idx]

    # ax.plot(x_idx, y_idx, 'r*')
    for i in np.arange(len(idx)):
        ax.text(x_idx[i], y_idx[i], labels_idx[i], fontdict={'color': 'white', 'size': 14, 'weight': 'bold'},
                bbox={'facecolor': 'black', 'alpha': 0.5, 'pad': 3},
                horizontalalignment='center', verticalalignment='center')


def load_mat(filename):
    """
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """

    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):

    """
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    """

    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):

    """
    A recursive function which constructs from matobjects nested dictionaries
    """

    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict




