# microscopes-release-linux [![Build Status](https://travis-ci.org/datamicroscopes/release-linux.svg?branch=master)](https://travis-ci.org/datamicroscopes/release-linux)

Tools for building conda releases of datamicroscopes on Linux.

### Building a new conda release
Committers can build a new linux-64 conda release by simplying updating the git submodule pointers and pushing to this repo. Travis CI takes care of the rest:

    $ git submodule foreach git pull
    $ git commit -a -m "Create new conda release"
    $ git push 
