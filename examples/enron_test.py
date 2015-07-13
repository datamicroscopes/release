import pickle
import time
import itertools as it
import numpy as np
from multiprocessing import cpu_count

# ###We've made a set of utilities especially for this dataset, `enron_utils`. We'll include these as well.
#
# `enron_crawler.py` downloads the data and preprocesses it as suggested by [Ishiguro et al. 2012](http://www.kecl.ntt.co.jp/as/members/ishiguro/open/2012AISTATS.pdf).  The results of the scirpt have been stored in the `results.p`.
#
# ##Let's load the data and make a binary matrix to represent email communication between individuals
#
# In this matrix, $X_{i,j} = 1$ if and only if person$_{i}$ sent an email to person$_{j}$

# In[2]:

import enron_utils


with open('results.p') as fp:
    communications = pickle.load(fp)
def allnames(o):
    for k, v in o:
        yield [k] + list(v)
names = set(it.chain.from_iterable(allnames(communications)))
names = sorted(list(names))
namemap = { name : idx for idx, name in enumerate(names) }



N = len(names)
# X_{ij} = 1 iff person i sent person j an email
communications_relation = np.zeros((N, N), dtype=np.bool)
for sender, receivers in communications:
    sender_id = namemap[sender]
    for receiver in receivers:
        receiver_id = namemap[receiver]
        communications_relation[sender_id, receiver_id] = True

print "%d names in the corpus" % N

# ##Let's visualize the communication matrix

# ##Now, let's learn the underlying clusters using the Inifinite Relational Model
#
# Let's import the necessary functions from datamicroscopes
#
# ##There are 5 steps necessary in inferring a model with datamicroscopes:
# 1. define the model
# 2. load the data
# 3. initialize the model
# 4. define the runners (MCMC chains)
# 5. run the runners

# In[5]:

from microscopes.common.rng import rng
from microscopes.common.relation.dataview import numpy_dataview
from microscopes.models import bb as beta_bernoulli
from microscopes.irm.definition import model_definition
from microscopes.irm import model, runner, query
from microscopes.kernels import parallel
from microscopes.common.query import groups, zmatrix_heuristic_block_ordering, zmatrix_reorder

defn = model_definition([N], [((0, 0), beta_bernoulli)])
views = [numpy_dataview(communications_relation)]
prng = rng()

nchains = 1
latents = [model.initialize(defn, views, r=prng, cluster_hps=[{'alpha':1}]) for _ in xrange(nchains)]
kc = runner.default_assign_kernel_config(defn)
print kc
r = runner.runner(defn, views, latents[0], kc)


# ##From here, we can finally run each chain of the sampler 1000 times

# In[ ]:

start = time.time()
print start
r.run(r=prng, niters=1)
print "inference took {} seconds".format(time.time() - start)