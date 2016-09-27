from optimizer import Optimizer

import numpy as np

import theano
import theano.tensor as T

class GD(Optimizer):
  def __init__(self, retina_model, seeder_model,
               n_seeds, n_steps, alpha_regime, sigma_regime,
               threshold = 1.0):
    self.seeder_model = seeder_model
    self.n_seeds = n_seeds
    self.n_steps = n_steps

    self.threshold = threshold

    if len(np.shape(alpha_regime)) == 0:
      self.alpha_regime = np.ones(shape=n_steps, dtype='float32') * alpha_regime
    else:
      self.alpha_regime = np.array(alpha_regime, dtype='float32')

    if len(np.shape(sigma_regime)) == 0:
      self.sigma_regime = np.ones(shape=n_steps, dtype='float32') * sigma_regime
    else:
      self.sigma_regime = np.array(sigma_regime, dtype='float32')

    self.sigmas = [
      theano.shared(np.ones(shape=self.n_seeds, dtype='float32') * self.sigma_regime[i], name='sigma_%d' % i)
      for i in xrange(self.n_steps)
    ]

    self.alphas = [
      theano.shared(np.ones(shape=self.n_seeds, dtype='float32') * self.alpha_regime[i], name='alpha_%d' % i)
      for i in xrange(self.n_steps)
    ]

    event_shareds = retina_model.get_event_variables()

    self.seeder = self.seeder_model(retina_model)

    self.inputs = retina_model.alloc_model_params()[:-1]

    iterations = [self.inputs]
    responses = []

    for i in xrange(self.n_steps):
      prev = iterations[i]
      params = event_shareds + prev + [self.sigmas[i]]
      r, grads = retina_model.grad_for(*params)

      update = [
        var + grad * self.alphas[i]
        for var, grad in zip(prev, grads)
      ]

      iterations.append(update)
      responses.append(r)

    outputs = [
      v for it in iterations for v in it
    ]

    self.ndim = len(self.inputs)

    self.f = theano.function(self.inputs, responses + outputs)
    self.shapes = [
      theano.function(self.inputs, o.shape, on_unused_input='ignore')
      for o in outputs
    ]

    self.responses = None
    self.traces = None
    self.seeds = None

  def maxima(self):
    self.seeds = self.seeder(self.n_seeds)

    staged = self.f(*self.seeds)
    rs, staged = staged[:self.n_steps], staged[self.n_steps:]

    staged = [
      staged[(i * self.ndim):(i * self.ndim + self.ndim)]
      for i in range(self.n_steps + 1)
    ]

    self.traces = staged
    self.responses = rs

    mask = rs[-1] > self.threshold

    return [
      param[mask]
      for param in staged[-1]
    ]

