INTRODUCTION
============

SAFE (or Spatial Analysis of Functional Enrichment) is an automated network annotation algorithm. Given a biological network and a set of functional groups or quantitative features of interest, SAFE performs local enrichment analysis to determine which regions of the network are over-represented for each group or feature. SAFE visualizes the network and maps the detected enrichments onto the network.

SAFE was originally implemented in MATLAB and stored at  <https://bitbucket.org/abarysh/safe/>. However, as of early 2017, the MATLAB implementation is only maintained for legacy purposes. New work related to SAFE is moving  to Python and this repository. 

**WARNING. This package is still in development. Please use caution.**


GETTING STARTED
===============

SAFE requires Python 3 and a set of packages listed in `extras/requirements.txt`. We recommend setting up a virtual environment and installing all the required packages via pip:

```
cd safepy/
virtualenv -p python3.6 safepy_env
source safepy_env/bin/activate
pip install -r extras/requirements.txt
```

After the installation is complete, it is useful to run a "hello world" SAFE analysis using the Jupyter notebook at `examples/Example_1_GI_network_doxorubicin.ipynb`. 

To do so, from within the safepy_env environment install a new ipython kernel:

```
ipython kernel install --user --name=safepy_env
```

Then start jupyter, open `examples/Example_1_GI_network_doxorubicin.ipynb` and select the safepy_env kernel.

```
jupyter-notebook
```

To run the examples, several common datasets will be required (e.g., the genetic interaction similarity network from Costanzo et al., 2016). These datasets are stored separately at <https://github.com/baryshnikova-lab/safe-data> (to avoid duplication with other SAFE-related repositories and packages). We recommend cloning the safe-data repository and storing it locally (e.g., at <path_to_safe-data_folder>). In addition, it is necessary to edit the SAFE settings file (at `safepy/safe_default.ini`) with the path to the `safe-data` folder.

```
vim safe_default.ini
...
[DEFAULT]
safe_data = <path_to_safe-data_folder>
...
```

It may also be useful to run a series of unit tests to verify that SAFE provides the correct outputs for default inputs. Tests are progressively being written and added to the repository. To run all the existing tests (from the `safepy` folder):

```
python3.6 -m unittest discover -v -s tests/
```



HELP
====

Please direct all questions/comments to Anastasia Baryshnikova (<abaryshnikova@calicolabs.com>).

The main repository for this code is at <https://github.com/baryshnikova-lab/safepy>. Please subscribe to the repository to receive live updates about new code releases and bug reports.


HOW TO CITE
==========

The manuscript describing SAFE and its applications is available at:

> Baryshnikova, A. (2016). Systematic Functional Annotation and Visualization of Biological Networks. Cell Systems. <http://doi.org/10.1016/j.cels.2016.04.014>
