.. currentmodule:: microscopes
.. _index:
.. datamicroscopes documentation master file, created by
   sphinx-quickstart on Tue Aug 12 20:17:28 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

datamicroscopes: Bayesian nonparametric models in Python
=================

datamicroscopes is a library for inference in various Bayesian nonparametric models, such as the Dirichlet Process Mixture Model (DPMM) and the Infinite Relational Model (IRM).  Our API allows for the flexible choice of likelihood models for various datatypes.  To read more about the models in datamicroscopes, please refer to our :ref:`datatypes available <datatypes>`.

For further information, please see our available :ref:`likelihood documentation, examples, <docs>` and :ref:`api <api>`.

Installation
=================
First, install `Anaconda <https://store.continuum.io/cshop/anaconda/>`_. Then in the terminal type:

.. code-block:: bash

	$ conda config --add channels distributions
	$ conda config --add channels datamicroscopes
	$ conda install microscopes-common
	$ conda install microscopes-{mixturemodel, irm}


.. toctree::
	:hidden:

	docs
	api