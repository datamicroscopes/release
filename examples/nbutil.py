import matplotlib.pylab as plt

class figsize(object):
    def __init__(self, w, h):
        self._w = w
        self._h = h
    def __enter__(self):
        self._old = plt.rcParams['figure.figsize']
        plt.rcParams['figure.figsize'] = (self._w, self._h)
    def __exit__(self, type, value, tb):
        plt.rcParams['figure.figsize'] = self._old
