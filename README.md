# microscopes-release

Linux: [![Build Status](https://travis-ci.org/datamicroscopes/release.svg?branch=master)](https://travis-ci.org/datamicroscopes/release) OS X: [![Build Status](https://travis-ci.org/datamicroscopes/release.svg?branch=osx)](https://travis-ci.org/datamicroscopes/release)

Tools for building conda releases of datamicroscopes on both Linux and OS X.

### Building a new conda release
Committers can build a new conda release by simplying updating the git submodule pointers and pushing to this repo. Travis CI takes care of the rest. When checkout out the repository for the first time, initialize the submodules with the following commands:

    $ git submodule init

Once the submodules are initialized, we can update the submodules to the master version on Github with:

    $ fab update

To push a new OS X build to Anaconda.org, use

    $ fab release_osx

To push a new Linux build to Anaconda.org, use

    $ fab release_linux