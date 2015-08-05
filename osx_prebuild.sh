gcc -v
c++ -v
# XXX(stephentu): HACK
# see https://github.com/datamicroscopes/common/issues/1
sudo mv `which gcc-4.2` `which gcc-4.2`-old
wget http://repo.continuum.io/miniconda/Miniconda-3.5.5-MacOSX-x86_64.sh -O miniconda.sh
