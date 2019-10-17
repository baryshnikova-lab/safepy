#!/usr/bin/env python
# coding: utf-8

import argparse
import os
from safe import SAFE

parser = argparse.ArgumentParser(description='Execute SAFE by commandline with input handling.')
parser.add_argument('--path-to-config', metavar='path_to_config', type=str, default='', 
                    help='Path to config file')
parser.add_argument('--path-to-network', metavar='path_to_network', type=str, 
                    help='Path to file containing network data')
parser.add_argument('--path-to-attributes', metavar='path_to_attributes', type=str, 
                    help='Path to file with the attributes asigned to each network node')
parser.add_argument('--output-path', metavar='output_path', type=str, default='results',
                    help='Path in which save SAFE folder results')
parser.add_argument('--threads', metavar='threads', type=int, default=1,
                    help='Number of threads to use in processing data')
parser.add_argument('--build_network_only', action='store_true',
					help='Parse network data and build the netwok object with the similarity distances and layout. The network will be saved as gpickle')
parser.add_argument('--show_significant_nodes', action='store_true',
					help='Show significant nodes in attribute plots')
parser.add_argument('--show_raw_data', action='store_true',
					help='Show raw data in attribute plots')
parser.add_argument('--attribute_names', nargs='*', default=[],
                    help='Attribute names space separated to be plotted')
args = parser.parse_args()

sf = SAFE(path_to_ini_file=args.path_to_config)
sf.load_network(network_file=args.path_to_network)

if args.build_network_only:
	sf.save_gpickle(args.output_path)
else:
	if not os.path.exists(args.output_path):
		os.mkdir(args.output_path)
	
	sf.plot_network()
	sf.define_neighborhoods()
	sf.load_attributes(attribute_file=args.path_to_attributes)
	sf.compute_pvalues(num_permutations=500, processes=args.threads)

	########################################################################
	### Show unitary attributes (all neighborhoods for one attribute)
	########################################################################
	sf.plot_sample_attributes(
		attributes=args.attribute_names,
		show_significant_nodes=args.show_significant_nodes,
		show_raw_data=args.show_raw_data, 
		save_fig=os.path.join(args.output_path,'attribute_plot.pdf')
	)

	########################################################################
	### Combine the enrichment landscapes into a single composite map
	########################################################################

	sf.define_top_attributes()
	sf.define_domains(attribute_distance_threshold = 0.65)
	sf.trim_domains()
	sf.plot_composite_network(
		show_each_domain=True, 
		show_domain_ids=True, 
		save_fig=os.path.join(args.output_path,'plotComposite.pdf')
	)

	########################################################################
	### Output text files
	########################################################################

	sf.print_output_files(output_dir=args.output_path)


