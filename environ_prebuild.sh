#!/usr/bin/env sh
set -e
chmod +x miniconda.sh
./miniconda.sh -b
export PATH=/home/travis/miniconda/bin:$PATH
conda update --yes conda
conda install --yes conda-server conda-build jinja2 anaconda-client sh pip
conda config --add channels distributions
conda config --add channels datamicroscopes
