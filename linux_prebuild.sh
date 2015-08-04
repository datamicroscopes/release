#!/usr/bin/env sh
set -e
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt-get update -qq
sudo apt-get install -qq g++-4.8
sudo rm -rf /dev/shm
sudo ln -s /run/shm /dev/shm
export CC=gcc-4.8
export CXX=g++-4.8
wget http://repo.continuum.io/miniconda/Miniconda-3.5.5-Linux-x86_64.sh -O miniconda.sh
chmod +x miniconda.sh
./miniconda.sh -b
export PATH=/home/travis/miniconda/bin:$PATH
conda update --yes conda
conda install --yes conda-server conda-build jinja2 anaconda-client sh
conda config --add channels distributions
conda config --add channels datamicroscopes
git config --global user.email "datamicroscopes.travis.builder@gmail.com"
git config --global user.name "datamicroscopes-travis-builder"
git config --global push.default simple
