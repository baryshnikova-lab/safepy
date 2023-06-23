#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import with_statement
from collections import defaultdict
from itertools import chain

from tqdm import tqdm

import os.path
import networkx as nx
import pandas as pd
import numpy as np
import pickle
import argparse


def make_go_matrix(path_to_obo='', path_to_annotations='', go_branch='p'):

    """
    Produces a matrix of gene-to-term annotations (propagated) for a specified GO tree and set of annotations
    :param path_to_obo: path the file containing the GO tree (downloadable from http://purl.obolibrary.org/obo/go/go-basic.obo)
    :param path_to_annotations: path to file containing species-specific gene-to-term annotations (e.g., http://current.geneontology.org/annotations/sgd.gaf.gz)
    :param go_branch: p, c or f for biological process, cellular component or molecular function terms
    :return: txt file with gene-to-term matrix; pickle file with matrix and other details
    """

    dirname_annotations = os.path.dirname(path_to_annotations)

    # Load the GO tree
    go = get_go_graph(path_to_obo)
    go_graph = go['go_graph']

    # Load the annotations
    go_annotations = read_annotations(path_to_annotations)

    # Get the desired branch of GO (p, f or c)
    namespaces = {'p': 'biological_process', 'c': 'cellular_component', 'f': 'molecular_function'}
    nodes_branch = (n for n in go_graph if go_graph.node[n]['namespace'] == namespaces[go_branch])
    go_graph_branch = go_graph.subgraph(nodes_branch)

    # Store information about each terms predecessors on the tree
    go_graph_branch = store_predecessors_all(go_graph_branch)

    # Make locus to GO term matrix
    go_matrix = make_locus2term(go_graph_branch, go_annotations)

    # Save the data
    output = {'go_matrix': go_matrix,
              'go_details': go['go_details'],
              'locus_details': go_annotations,
              'tree_path': path_to_obo,
              'annotations_path': path_to_annotations}

    pickle_path = os.path.join(dirname_annotations, 'go_' + go_branch + '.p')
    print("Saving the results at %s" % pickle_path)
    with open(pickle_path, 'wb') as handle:
        pickle.dump(output, handle)

    txt_path = os.path.join(dirname_annotations, 'go_' + go_branch + '_matrix.txt')
    print("Printing the gene-to-term matrix at %s" % txt_path)
    go_matrix.to_csv(txt_path, sep='\t')


def process_go_term(term):

    """
    In an object representing a GO term, replace single-element lists with
    their only member.
    __author__ = "Uli Köhler"
    __copyright__ = "Copyright 2013 Uli Köhler"
    __license__ = "Apache v2.0"
    __version__ = "1.1"
    :return: the modified object as a dictionary
    """

    ret = dict(term)  # Input is a defaultdict, might express unexpected behaviour
    for key, value in ret.items():
        if key != 'is_a' and len(value) == 1:
            ret[key] = value[0]
    return ret


def parse_go_obo(path_to_obo):

    """
    Parses a Gene Ontology dump in OBO v1.2 format.
    __author__ = "Uli Köhler"
    __copyright__ = "Copyright 2013 Uli Köhler"
    __license__ = "Apache v2.0"
    __version__ = "1.1"
    :param path_to_obo: Path to the OBO file containing the GO tree
    """

    with open(path_to_obo, "r") as infile:
        current_term = None

        for line in infile:

            line = line.strip()

            if not line:
                # Skip empty
                continue

            if line == "[Term]":
                if current_term:
                    yield process_go_term(current_term)
                current_term = defaultdict(list)

            elif line == "[Typedef]":
                # Skip [Typedef sections]
                current_term = None

            else:  # Not [Term]
                # Only process if we're inside a [Term] environment
                if current_term is None:
                    continue
                key, sep, val = line.partition(":")
                val_id, val_sep, val_rest = val.partition("!")
                current_term[key].append(val_id.strip())

        # Add last term
        if current_term is not None:
            yield process_go_term(current_term)


def get_go_graph(path_to_obo):

    _, ext = os.path.splitext(path_to_obo)

    # If pickle file passed, the OBO file has already been processed, just load.
    if ext == '.p':
        with open(path_to_obo, 'r') as f:
            return pd.read_pickle(path_to_obo)

    dirname_obo = os.path.dirname(path_to_obo)

    go_graph = nx.DiGraph()
    go_details = pd.DataFrame(data={'id': 'id', 'name': 'name', 'namespace': 'namespace'}, index=[0])

    # Iterate over GO terms
    term_counter = 0

    print("Parsing the OBO file (may take a few min)...")
    for go_term in parse_go_obo(path_to_obo):

        # Ignore the obsolete terms
        if go_term['name'].startswith('obsolete'):
            continue

        # Store terms details
        keys = ['id', 'name', 'namespace']
        go_term_min = dict((k, go_term[k]) for k in keys if k in go_term)
        df = pd.DataFrame(data=go_term_min, index=[0])
        go_details = go_details.append(df)

        if not go_term['id'] in go_graph.nodes():
            go_graph.add_node(go_term['id'], namespace=go_term['namespace'])

        if 'is_a' in go_term:
            for parent_term in go_term['is_a']:
                if parent_term not in go_graph.nodes():
                    go_graph.add_node(parent_term, namespace=go_term['namespace'])
                go_graph.add_edges_from([(parent_term, go_term['id'])])

        term_counter += 1

    go_details.index = go_details['id']
    go_details.drop('id', axis=0, inplace=True)

    print("Parsed %d GO terms." % term_counter)

    # Save the data
    output = {'go_graph': go_graph,
              'go_details': go_details}

    # # Save for future re-uses
    # with open(os.path.join(dirname_obo, 'go.p'), 'wb') as handle:
    #     pickle.dump(output, handle)

    return output


def get_predecessors_all(graph, nodes):

    """
    Recursively get all the predecessors of a node or a list of nodes
    :param graph: graph to be transversed
    :param nodes: a single node or a list of nodes for which the list of predecessors is required
    :return: list of predecessor nodes
    """

    if not isinstance(nodes, list):
        nodes = [nodes]

    node_predecessors = [graph.predecessors(node) for node in nodes]    # get all the immediate predecessors
    node_predecessors = list(chain.from_iterable(node_predecessors))    # flatten the list

    if not node_predecessors:   # if reached a root & no more predecessors, stop and return
        return node_predecessors
    else:   # otherwise, get the predecessors of the current list of nodes
        return node_predecessors + get_predecessors_all(graph, node_predecessors)


def store_predecessors_all(graph, node_id=None, predecessors=[]):

    """
    Recursively transverse down the whole tree (or the tree rooted at a certain term) and store each terms predecessors as a node attribute
    :param graph: the tree
    :param node_id: the starting node (if None, then the root of the tree is used as a start)
    :param predecessors: the list of predecessors for the root
    :return: the graph with node attributes stored
    """

    # If starting, start from the root
    if not node_id:
        nodes = [n for n, d in dict(graph.in_degree()).items() if d == 0]
        node_id = nodes[0]
        predecessors = []

    graph.node[node_id]['predecessors'] = predecessors
    node_successors = graph.successors(node_id)

    if node_successors:
        for node_successor in node_successors:
            graph = store_predecessors_all(graph, node_id=node_successor, predecessors=[node_id]+predecessors)

    return graph


def read_annotations(path_to_annotations):

    """
    Read the gene-to-term annotation file
    :param path_to_annotations: input filename
    :return: data frame
    """

    print("Reading the gene-to-GO term annotations...")
    go_annotations = pd.read_csv(path_to_annotations, delimiter='\t', header=None, comment="!",
                                 dtype={1: str, 10: str})
    go_annotations.index = go_annotations[1]
    return go_annotations


def make_locus2term(go_graph, go_annotations):

    print("Making the gene-to-GO term matrix...")

    go_terms = sorted(list(set(go_graph.nodes())))
    loci = sorted(list(set(go_annotations.index)))

    locus2term = np.zeros((len(loci), len(go_terms)))

    for go_term in tqdm(go_terms):

        # Get all the term's predecessors
        go_term_and_predecessors = [go_term] + go_graph.node[go_term]['predecessors']
        go_term_and_predecessors = sorted(go_term_and_predecessors)
        idx_terms = np.searchsorted(go_terms, go_term_and_predecessors)

        # Get all loci annotated to the term
        go_term_loci = sorted(go_annotations.loc[go_annotations[4] == go_term, 1].unique().tolist())
        idx_loci = np.searchsorted(loci, go_term_loci)

        # Set loci x go_terms values to 1
        [r, c] = np.meshgrid(idx_loci, idx_terms)
        locus2term[r, c] = 1

    locus2term_df = pd.DataFrame(data=locus2term, index=loci, columns=go_terms)

    # Drop all GO terms that have 0 loci
    num_loci_x_term = locus2term_df.sum(axis=0)
    locus2term_df = locus2term_df.loc[:, num_loci_x_term > 0]

    # Find loci (if any) that have 0 terms and assign them to the root
    num_terms_x_locus = locus2term_df.sum(axis=1)
    roots = [n for n, d in dict(go_graph.in_degree()).items() if d == 0]
    locus2term_df.loc[num_terms_x_locus == 0, roots[0]] = 1

    locus2term_df = locus2term_df.astype(int)

    print('%d loci had 0 terms and were assigned to the root.' % np.sum(num_terms_x_locus == 0))

    # Transform into a sparse dataframe to save disk space and memory
    locus2term_df = locus2term_df.to_sparse(fill_value=0)

    return locus2term_df


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate a new matrix of genes (loci) to GO term associations.')
    parser.add_argument('--path-to-obo', metavar='path_to_obo', type=str,
                        help='Path to file containing the GO tree')
    parser.add_argument('--path-to-annotations', metavar='path_to_annotations', type=str,
                        help='Path to gene-to-term annotation file')
    parser.add_argument('--go-branch', metavar='go_branch', default='p', type=str, help='p, c or f')
    args = parser.parse_args()

    make_go_matrix(path_to_obo=args.path_to_obo,
                   path_to_annotations=args.path_to_annotations,
                   go_branch=args.go_branch)
