#!/usr/bin/env python
# coding: utf-8

import argparse
import os
from safe import SAFE

parser = argparse.ArgumentParser(description='Execute SAFE by commandline with input handling.')
parser.add_argument('--path-to-network', metavar='path_to_network', type=str, 
                    help='Path to file containing network data')
parser.add_argument('--path-to-attributes', metavar='path_to_attributes', type=str, 
                    help='Path to file with the attributes asigned to each network node')
parser.add_argument('--output-folder', metavar='output_folder', type=str, default='results',
                    help='Path in which save SAFE folder results')
parser.add_argument('--threads', metavar='threads', type=int, default=1,
                    help='Number of threads to use in processing data')
args = parser.parse_args()

if not os.path.exists(args.output_folder):
	os.mkdir(args.output_folder)

sf = SAFE()
#MAKE SCRIPT THAT PRECOMPUTE LAYOUT AND DAVE GPICKLE FILE.
sf.load_network(network_file=args.path_to_network)
sf.plot_network()

sf.define_neighborhoods()
sf.load_attributes(attribute_file=args.path_to_attributes)
sf.compute_pvalues(num_permutations=500, processes=args.threads)

########################################################################
### Show unitary attributes (all neighborhoods for one attribute)
########################################################################
#ADD OPTION FOR TAKE AN ARRAY OF ATTRIBUTE NAMES AND GENERATE THE PLOT FOR EACH ONE OF THEM
# Plot a random attribute
sf.plot_sample_attributes(save_fig=os.path.join(args.output_folder,'plot0.pdf'))
# Also, can specify the name of the attribute to be plotted (Note: to get the list of available attributes, check sf.attributes.head())
#sf.plot_sample_attributes(attributes=['Doxorubicin'], save_fig=os.path.join('results','doxorubicinA.pdf'))
# Map the significant nodes
sf.plot_sample_attributes(show_significant_nodes=True, save_fig=os.path.join(args.output_folder,'plot2.pdf'))
# Map the raw data onto the network
sf.plot_sample_attributes(show_raw_data=True, save_fig=os.path.join(args.output_folder,'plot3.pdf'))
# Map a few labels onto the network
sf.plot_sample_attributes(show_raw_data=True, labels=['rad54','rad57','rad55','rad52'], save_fig=os.path.join(args.output_folder,'plot4.pdf'))

########################################################################
### Combine the enrichment landscapes into a single composite map
########################################################################

sf.define_top_attributes()
sf.define_domains(attribute_distance_threshold = 0.65)
sf.trim_domains()
sf.plot_composite_network(show_each_domain=True, show_domain_ids=True, save_fig=os.path.join(args.output_folder,'plotComposite.pdf'))

########################################################################
### Output text files
########################################################################

sf.print_output_files(output_dir=args.output_folder)


