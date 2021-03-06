#!/usr/bin/env sh
set -e
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt-get update -qq
sudo apt-get install -qq g++-4.8
sudo rm -rf /dev/shm
sudo ln -s /run/shm /dev/shm
export CC=gcc-4.8
export CXX=g++-4.8
git config --global user.email "datamicroscopes.travis.builder@gmail.com"
git config --global user.name "datamicroscopes-travis-builder"
git config --global push.default simple
wget http://repo.continuum.io/miniconda/Miniconda-3.5.5-Linux-x86_64.sh -O miniconda.sh
