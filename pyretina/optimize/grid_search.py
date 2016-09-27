import numpy as np

from optimizer import Optimizer

import theano
import theano.tensor as T
from scipy.ndimage.filters import maximum_filter

def maxima(shape, filter_size, threshold = 0.8, out=None):
  m = np.ones(shape=shape, dtype='float32')
  mask = np.ones(shape=shape, dtype=bool)

  if type(filter_size) is int or type(filter_size) is long:
    filter_size = (filter_size, ) * len(shape)

  def f(response):
    maximum_filter(
      response, size=filter_size, output=m,
      mode='constant', cval=0.0
    )

    np.greater(response, threshold, out=out)

    np.subtract(m, response, out=m)
    np.less(m, 1.0e-30, out=mask)

    np.logical_and(mask, out, out=out)

  return f

class GridOptimizer(Optimizer):
  def __init__(self, retina_model, ranges, threshold = 2.9, filter_size = 3):
    event_vars = retina_model.get_event_variables()

    self.ranges = [ np.array(r, dtype='float32').reshape(-1) for r in ranges ]
    self.grids = [
      g.astype('float32')
      for g in np.meshgrid(*self.ranges, indexing='ij')
    ]

    self._grid_shareds = [
      theano.shared(grid.ravel().astype('float32'), name='grid_%d' % i)
      for i, grid in enumerate(self.grids)
    ]

    shape = self.grids[0].shape

    r = retina_model.response_for(*event_vars + self._grid_shareds).reshape(shape)

    shape = self.grids[0].shape

    self._get_response = theano.function([], r)
    self._response_grid = None
    self._maxima_indx = None

    self._mask = np.zeros(shape=shape, dtype=bool)
    self._maxima = maxima(shape, filter_size=filter_size, threshold=threshold, out=self._mask)

    super(GridOptimizer, self).__init__()

  @property
  def response_grid(self):
    return self._response_grid

  @property
  def maxima_indx(self):
    return self._maxima_indx

  @property
  def mask(self):
    return self._mask

  def fill_grid_value(self, i, z0):
    self.grids[i].fill(z0)
    self._grid_shareds[i].set_value(self.grids[i].reshape(-1))

  def maxima(self):
    self._response_grid = self._get_response()
    self._maxima(self._response_grid)
    self._maxima_indx = np.where(self._mask)

    result = []

    for i, g in enumerate(self.grids):
      result.append(g[self._maxima_indx])

    return result