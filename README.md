# microscopes-release 

Linux: [![Build Status](https://travis-ci.org/datamicroscopes/release.svg?branch=master)](https://travis-ci.org/datamicroscopes/release) OS X: [![Build Status](https://travis-ci.org/datamicroscopes/release.svg?branch=osx)](https://travis-ci.org/datamicroscopes/release)

Tools for building conda releases of datamicroscopes on both Linux and OS X.

### Building a new conda release
Committers can build a new conda release by simplying updating the git submodule pointers and pushing to this repo. Travis CI takes care of the rest. When checkout out the repository for the first time, initialize the submodules with the following commands:
    
    $ git submodule init
    $ git submodule update --remote

Once the submodules are initialized, then bumping the pointers is simple:

    $ git submodule foreach "git checkout master; git pull"
    $ git commit -a -m "Create new conda release"
    $ git push 

To build for linux, push to the `master` branch. To build for OS X, push to the `osx` branch.
