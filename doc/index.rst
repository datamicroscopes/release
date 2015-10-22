.. currentmodule:: microscopes
.. _index:
.. datamicroscopes documentation master file, created by
   sphinx-quickstart on Tue Aug 12 20:17:28 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

datamicroscopes: Bayesian nonparametric models in Python
====================

datamicroscopes is a library for discovering structure in your data. It implements several Bayesian nonparametric models for clustering such as the :ref:`Dirichlet Process Mixture Model (DPMM) <gauss2d>` , :ref:`the Infinite Relational Model (IRM) <enron_email>` , and the :ref:`Hierarchichal Dirichlet Process (HDP) <hdp>` .  These models rely on the :ref:`Dirichlet Process <intro>`, which allow for the automatic learning of the number of clusters in a datset.  Additionally, our :ref:`API <api>` provides users with a flexible set of likelihood models for various types of data, such as binary, ordinal, categorical, and real-valued variables( :ref:`datatypes <docs>`)  .

Please read our :ref:`introduction <intro>` for an overview of clustering and structure discovery.


.. raw:: html

	<div style="clear: both"></div>
	<div class="container-fluid">
	<div class="row">
		 <div class="col-md-4">
		<h2>Tutorials</h2>

.. toctree::
	:maxdepth: 2

	intro
	ncluster
	enron_blog
	topic

.. raw:: html

   </div>
   <div class="col-md-4">
   <h2>Datatypes and Likelihood Models</h2>

.. toctree::
   :maxdepth: 2

   datatypes
   bb
   dd
   niw
   nic
   gamma_poisson

.. raw:: html

   </div>
   <div class="col-md-4">
   <h2>Examples</h2>

.. toctree::
	:maxdepth: 2

	gauss2d
	mnist_predictions
	enron_email
	hdp


.. raw:: html

   </div>
   </div>
   </div>


Installation
=================
First, install `Anaconda <https://store.continuum.io/cshop/anaconda/>`_. Then in the terminal type:

.. code-block:: bash

	$ conda config --add channels distributions
	$ conda config --add channels datamicroscopes
	$ conda install microscopes-common
	$ conda install microscopes-{mixturemodel, irm, lda}


.. toctree::
	:hidden:

	docs
	api
