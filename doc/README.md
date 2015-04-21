Documentation for datamicroscopes
=================================

Documentation for datamicroscopes is built with [Sphinx](http://sphinx-doc.org), which uses reStructuredText as an input. Sphinx uses two sources of information to build the documentation: docstrings in the source code, and reStructuredText files in the folder `release/doc`.  

Numpy provides [good resources](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt) on how to write quality scientific docstrings. To learn more about reST, Sphinx provides a useful  [introduction](http://sphinx.pocoo.org/rest.htm).  

The documentation relies on mathematical notation and code examples to explain our models . We use [numpydoc](https://github.com/numpy/numpydoc) to write docstrings that can be used with Sphinx.  Additionally, we use [iptyhon directive](http://matplotlib.org/sampledoc/ipython_directive.html) and [ipython notebook](http://sphinx-ipynb.readthedocs.org/en/latest/howto.html) to show example code. 

To update the Sphinx build, call:

```bash
cd release/doc
make html
```

Updated html files will be available at `relase/_build`

For cython files, please remember to [compile your file](http://docs.cython.org/src/reference/compilation.html#compilation-reference) before updating.

