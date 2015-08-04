conda install --yes microscopes-common microscopes-mixturemodel microscopes-irm microscopes-lda
  microscopes-kernels
conda install --yes sphinx pip
which pip
pip install numpydoc
git clone "https://datamicroscopes-travis-builder:$GITHUB_PASSWORD@github.com/datamicroscopes/datamicroscopes.github.io"
  D
(cd doc && make html)
rm -rf D/*
cp -R doc/_build/html/* D
(cd D && git add *)
(cd D && git commit -m "autogen docs")
(cd D && git push -q)
