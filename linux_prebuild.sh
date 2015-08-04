#!/usr/bin/env sh
set -e
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt-get update -qq
sudo apt-get install -qq g++-4.8
sudo rm -rf /dev/shm
sudo ln -s /run/shm /dev/shm
export CC=gcc-4.8
export CXX=g++-4.8
