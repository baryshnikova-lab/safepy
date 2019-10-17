#! /usr/bin/env bash
module load python/anaconda-4.7.12
module load pandoc/2.2.1
export PATH=~pedro/.local/bin:$PATH
source safepy_env/bin/activate

#CONVERT PLAIN DATA TO GRAPH OBJECT
time ./safe_analysis.py \
	--path-to-network ~pedro/test/safe/safe-data/networks/Costanzo_Science_2010_5col.txt \
	--build_network_only \
	--output-path test.gpickle \

# LOAD AND ANALYSE NETWORK
time ./safe_analysis.py \
	--path-to-network test.gpickle  \
	--path-to-attributes ~pedro/test/safe/safe-data/attributes/go_bp_140819.txt.gz \
	--output-path results \
	--threads 1 \
	--attribute_names 'mitotic sister chromatid segregation' 'cell cycle checkpoint' \
	--show_raw_data \
