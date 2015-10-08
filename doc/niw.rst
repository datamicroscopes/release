.. _niw:
Real Valued Data and the Normal Inverse-Wishart Distribution
============================================================

--------------

One of the most common forms of data is real valued data

Let's set up our environment and consider an example dataset

.. code:: python

    import pandas as pd
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline
    sns.set_context('talk')
    sns.set_style('darkgrid')

The `Iris Flower
Dataset <https://archive.ics.uci.edu/ml/datasets/Iris>`__ is a standard
machine learning data set dating back to the 1930s. It contains
measurements from 150 flowers, 50 from each of the following species:

-  Iris Setosa
-  Iris Versicolor
-  Iris Virginica

.. code:: python

    iris = sns.load_dataset('iris')
    iris.head()




.. raw:: html

    <div style="max-height:1000px;max-width:1500px;overflow:auto;">
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>sepal_length</th>
          <th>sepal_width</th>
          <th>petal_length</th>
          <th>petal_width</th>
          <th>species</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>5.1</td>
          <td>3.5</td>
          <td>1.4</td>
          <td>0.2</td>
          <td>setosa</td>
        </tr>
        <tr>
          <th>1</th>
          <td>4.9</td>
          <td>3.0</td>
          <td>1.4</td>
          <td>0.2</td>
          <td>setosa</td>
        </tr>
        <tr>
          <th>2</th>
          <td>4.7</td>
          <td>3.2</td>
          <td>1.3</td>
          <td>0.2</td>
          <td>setosa</td>
        </tr>
        <tr>
          <th>3</th>
          <td>4.6</td>
          <td>3.1</td>
          <td>1.5</td>
          <td>0.2</td>
          <td>setosa</td>
        </tr>
        <tr>
          <th>4</th>
          <td>5.0</td>
          <td>3.6</td>
          <td>1.4</td>
          <td>0.2</td>
          <td>setosa</td>
        </tr>
      </tbody>
    </table>
    </div>



In the case of the ``iris`` dataset, plotting the data shows that
indiviudal species exhibit a typical range of measurements

.. code:: python

    irisplot = sns.pairplot(iris, hue="species", palette='Set2', diag_kind="kde", size=2.5)
    irisplot.fig.suptitle('Scatter Plots and Kernel Density Estimate of Iris Data by Species', fontsize = 18)
    irisplot.fig.subplots_adjust(top=.9)



.. image:: normal-inverse-wishart_files/normal-inverse-wishart_5_0.png


If we wanted to learn these underlying species' measurements, we would
use these real valued measurements and make assumptions about the
structure of the data.

In practice, real valued data is commonly assumed to be distributed
normally, or Gaussian

We could assume that conditioned on ``species``, the measurement data
follwed a multivariate normal

.. math:: P(\mathbf{x}|species=s)\sim\mathcal{N}(\mu_{s},\Sigma_{s})

The normal inverse-Wishart distribution allows us to learn the
underlying parameters of each normal distribution, its mean
:math:`\mu_s` and its covariance :math:`\Sigma_s`. Since the normal
inverse-Wishart is the conjugate prior of the multivariate normal, the
posterior distribution of a multivariate normal with a normal
inverse-Wishart prior also follows a normal inverse-Wishart
distribution. This allows us to infer the distirbution over values of
:math:`\mu_s` and :math:`\Sigma_{s}` when we define our model.

Note that if we have only one real valued variable, the normal
inverse-Wishart distribution is often referred to as the normal
inverse-gamma distribution. In this case, we learn the scalar valued
mean :math:`\mu` and variance :math:`\sigma^2` for each inferred
cluster.

Univariate real data, however, should be modeled with our normal
invese-chi-squared distribution, which is optimized for infering
univariate parameters.

See `Murphy
2007 <http://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf>`__ for
derrivations of our normal likelihood models

--------------

To specify the joint distribution of a multivariate normal
inverse-Wishart distribution, we would import our likelihood model

.. code:: python

    from microscopes.models import niw as normal_inverse_wishart
