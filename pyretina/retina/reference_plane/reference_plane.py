import numpy as np

import theano
import theano.tensor as T

from ..retina_model import RetinaModel3D

class ReferencePlaneRetinaModel(RetinaModel3D):
  def __init__(self, reference_z = 1.0):
    self._reference_z = reference_z
    model_params_names = ['model_x', 'model_y', 'primary_vertex', 'sigma']

    super(ReferencePlaneRetinaModel, self).__init__(model_params_names=model_params_names)

  def distance_for(self, event_x, event_y, event_z,
                   model_x, model_y, primary_vertex, sigma):
    scale = (event_z[:, None] - primary_vertex) / (self._reference_z - primary_vertex)[None, :]
    tx = model_x[None, :] * scale
    ty = model_y[None, :] * scale

    return (tx - event_x[:, None]) ** 2 + (ty - event_y[:, None]) ** 2

  def response_matrix_for(self, event_x, event_y, event_z,
                          model_x, model_y, primary_vertex, sigma):
    d = self.distance_for(event_x, event_y, event_z,
                          model_x, model_y, primary_vertex, sigma)
    return T.exp(-d / sigma)

  def _grad_for(self,
                event_x, event_y, event_z,
                model_x, model_y, primary_vertex, sigma):
    scale = (event_z[:, None] - primary_vertex) / (self._reference_z - primary_vertex)[None, :]

    tx = model_x[None, :] * scale
    ty = model_y[None, :] * scale

    dx = (tx - event_x[:, None])
    dy = (ty - event_y[:, None])

    dsq = dx ** 2 + dy ** 2

    r = T.exp(-dsq / sigma[None, :])

    dtx_dmx = scale
    dty_dmy = scale

    dscale_dmz = (event_z[:, None] - self._reference_z) / ((self._reference_z - primary_vertex)[None, :] ** 2)

    ddsq_dmx = 2 * dx * dtx_dmx
    ddsq_dmy = 2 * dy * dty_dmy
    ddsq_dmz = 2 * (dx * model_x[None, :] + dy * model_y[None, :]) * dscale_dmz

    dr_dmx = -(1 / sigma[None, :]) * r * ddsq_dmx
    dr_dmy = -(1 / sigma[None, :]) * r * ddsq_dmy
    dr_dmz = -(1 / sigma[None, :]) * r * ddsq_dmz
    dr_dsigma = (dsq / sigma[None, :] ** 2) * r

    return r, (dr_dmx, dr_dmy, dr_dmz, dr_dsigma)

  def grad_for(self,
               event_x, event_y, event_z,
               model_x, model_y, primary_vertex, sigma):
    r, grads = self._grad_for(
      event_x, event_y, event_z,
      model_x, model_y, primary_vertex, sigma
    )

    return T.sum(r, axis=0), [ T.sum(grad, axis = 0) for grad in grads ]

  def auto_grad_for(self, event_x, event_y, event_z,
                    model_x, model_y, primary_vertex, sigma):
    r = self.response_for(
      sigma,
      event_x, event_y, event_z,
      model_x, model_y, primary_vertex
    )

    dr_dmx, _ = theano.map(
      lambda i, r, x: theano.grad(r[i], x)[i],
      sequences=T.arange(r.shape[0]),
      non_sequences=[r, model_x]
    )

    dr_dmy, _ = theano.map(
      lambda i, r, x: theano.grad(r[i], x)[i],
      sequences=T.arange(r.shape[0]),
      non_sequences=[r, model_y]
    )

    dr_dmz, _ = theano.map(
      lambda i, r, x: theano.grad(r[i], x)[i],
      sequences=T.arange(r.shape[0]),
      non_sequences=[r, primary_vertex]
    )

    dr_dsigma, _ = theano.map(
      lambda i, r, x: theano.grad(r[i], x)[i],
      sequences=T.arange(r.shape[0]),
      non_sequences=[r, sigma]
    )

    return r, (dr_dmx, dr_dmy, dr_dmz, dr_dsigma)