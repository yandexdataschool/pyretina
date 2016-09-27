from retina_response import *

import numpy as np

import theano
import theano.tensor as T

class RetinaModel(object):
  def __init__(self, model_params_names, event_feature_names):
    self._event_feature_names = event_feature_names
    self._model_params_names = model_params_names

    for name in event_feature_names:
      shared = theano.shared(np.ndarray(shape=(0, ), dtype='float32'), name=name)
      setattr(self, '_%s' % name, shared)

    self._event_shareds = [
      getattr(self, '_%s' % name)
      for name in event_feature_names
    ]

    for name in model_params_names:
      var = T.fvector(name=name)
      setattr(self, '_%s' % name, var)

    self._model_params = [
      getattr(self, '_%s' % name)
      for name in model_params_names
    ]

    r, grads = self.grad_for(*self._event_shareds + self._model_params)

    self._response = theano.function(self._model_params, r)

  def set_event(self, event):
    assert(event.shape[1] == self.event_ndim)

    for i in range(self.event_ndim):
      self._event_shareds[i].set_value(event[:, i].astype('float32'))

  def response(self, *args, **kwargs):
    return self._response(*args, **kwargs)

  def response_grid(self, *args, **kwargs):
    param_names = kwargs.keys()

    grids = np.meshgrid(*args + tuple([ kwargs[k] for k in param_names ]), indexing='ij')
    original_shape = [s for s in grids[0].shape if s > 1]

    flat_grids = [
      grid.astype('float32').ravel() for grid in grids
    ]

    grid_args = tuple(flat_grids[:len(args)])
    grid_kwargs = dict(zip(param_names, flat_grids[len(args):]))

    r = self.response(*grid_args, **grid_kwargs)

    return r.reshape(original_shape)

  @property
  def event_ndim(self):
    return len(self._event_feature_names)

  @property
  def model_nparams(self):
    return len(self._model_params)

  def get_event_variables(self):
    return self._event_shareds

  def get_all_model_params(self):
    return self._model_params

  def alloc_model_params(self):
    return [
      T.fvector(name=name)
      for name in self._model_params_names
    ]

  def distance_for(self, *args, **kwargs):
    """
    :return: symbolic matrix D of shape n x m where
    D_ij - square distance from i-th point to j-th model.
    Correspondingly, n - number of points, m - number of models.
    """
    raise NotImplementedError('This is an interface declaration.')

  def response_matrix_for(self, *args, **kwargs):
    """
    :return: symbolic matrix R of shape n x m,
     where R_ij - activation (excitement) of j-th model by i-th point.

    Correspondingly, n - number of points, m - number of models.
    By default, R_ij = exp(-D_ij / sigma_j)
    """
    raise NotImplementedError('This is an interface declaration.')

  def response_for(self, *args, **kwargs):
    """
    :return: vector of Retina Response for each model.
    """
    response = self.response_matrix_for(*args, **kwargs)
    return T.sum(response, axis = 0)

  def grad_for(self, *args, **kwargs):
    """
    :param args: first argument should be list of variable with respect to
     which gradient should be taken.
    """
    raise NotImplementedError('This is an interface declaration.')

  def hessian_for(self, *args, **kwargs):
    r, grads = self.grad_for(*args, **kwargs)

    fake_hessian = [
      [
        (T.zeros if i != j else T.ones)(shape=tuple(), dtype='float32')
        for j in range(self.model_nparams)
      ]

      for i in range(self.model_nparams)
    ]

    return r, grads, fake_hessian

class RetinaModel3D(RetinaModel):
  def __init__(self, model_params_names):
    super(RetinaModel3D, self).__init__(
      model_params_names = model_params_names,
      event_feature_names = ['event_x', 'event_y', 'event_z']
    )