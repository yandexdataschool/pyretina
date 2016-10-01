from optimizer import Optimizer

import numpy as np

import theano
import theano.tensor as T
from lasagne import layers, nonlinearities, regularization, updates

class GradBased(Optimizer):
  @classmethod
  def build_nn(cls, n_model, n_units=100):
    ### current params + response + grads
    in_l = layers.InputLayer(shape=(None, 2 * n_model + 1, ), name='input_params')

    dense1 = layers.DenseLayer(
      in_l,
      num_units=n_units,
      nonlinearity=nonlinearities.tanh
    )

    out_l = layers.DenseLayer(
      dense1,
      num_units=n_model,
      nonlinearity=nonlinearities.linear
    )

    reg = \
      regularization.regularize_layer_params(dense1, regularization.l2) + \
      regularization.regularize_layer_params(out_l, regularization.l2)

    return in_l, out_l, reg

  def get_update_for(self, params, r, grads):
    input_t = T.stack(params + [r] + grads, axis=1)
    return layers.get_output(self.out_layer, inputs=input_t)

  def __init__(self, retina_model, seeder_model,
               n_seeds, n_steps, n_units = 100,
               normalization_coefs=None,
               loss_coefs=None,
               alpha = 1.0,
               threshold = 1.0):
    self.seeder_model = seeder_model
    self.n_seeds = n_seeds
    self.n_steps = n_steps

    self.threshold = threshold

    self.retina = retina_model

    event_shareds = retina_model.get_event_variables()

    self.seeder = self.seeder_model(retina_model)

    if normalization_coefs is None:
      normalization_coefs = np.ones(shape=retina_model.model_nparams, dtype='float32')
    else:
      normalization_coefs = np.array(normalization_coefs, dtype='float32')

    ### params + sigma
    self.inputs = retina_model.alloc_model_params()

    self.input_layer, self.out_layer, self.reg = self.build_nn(retina_model.model_nparams, n_units=n_units)

    print 'Linking to Retina Model'

    iterations = [self.inputs]
    responses = []

    for i in xrange(self.n_steps):
      print 'Iteration %d' % i

      prev = iterations[i]
      r, grads = retina_model.grad_for(*event_shareds + prev)

      normed_params = [
        p * c
        for p, c in zip(prev, normalization_coefs)
      ]

      normed_grads = [
        g * c
        for g, c in zip(grads, normalization_coefs)
      ]

      out = self.get_update_for(normed_params, r, normed_grads)

      param_updates = [
        out[:, i]
        for i in range(len(self.inputs))
      ]

      track_param_updates, sigma_update = param_updates[:-1], param_updates[-1]

      ### sigma (last parameter) is updated simply by replacing
      ### previous variable
      update = [
        var + upd * alpha
        for var, upd in zip(prev[:-1], track_param_updates)
      ] + [
        T.exp(-sigma_update)
      ]

      for var, upd, new in zip(prev[:-1], track_param_updates, update):
        print '  -', new, '=', var, '+ %.2e' % alpha, upd

      iterations.append(update)
      responses.append(r)

    prediction = iterations[-1]

    sigma_train = T.fscalar('sigma_train')

    ### Except sigma
    self.true_parameters_shareds = [
      theano.shared(np.ndarray(shape=(0, ), dtype='float32'), name=name)
      for name in retina_model.model_params_names[:-1]
    ]

    ### predictions without sigma
    print 'Constucting loss:'
    print '  - Loss coefs:', loss_coefs
    print '  - True params shared:', self.true_parameters_shareds
    print '  - Predictions:', prediction[:-1]
    print '  - Sigma:', sigma_train

    pure_loss = 1.0 - retina_model.parameter_response(
      loss_coefs,
      *self.true_parameters_shareds + prediction[:-1] + [sigma_train]
    )

    initial_loss = 1.0 - retina_model.parameter_response(
      loss_coefs,
      *self.true_parameters_shareds + self.inputs[:-1] + [sigma_train]
    )

    reg_c = T.fscalar('reg_c')
    loss = pure_loss + reg_c * self.reg

    params = layers.get_all_params(self.out_layer)
    learning_rate = T.fscalar('learning rate')

    net_updates = updates.rmsprop(loss, params, learning_rate=learning_rate)

    self._train = theano.function(
      self.inputs + [sigma_train, learning_rate, reg_c],
      [pure_loss, self.reg, loss, initial_loss],
      updates=net_updates
    )

    self._loss = theano.function(self.inputs + [sigma_train], pure_loss)

    outputs = [
      v for it in iterations for v in it
    ]

    self.ndim = len(self.inputs)

    self.predictions = theano.function(self.inputs, responses + outputs)

    self.responses = None
    self.traces = None
    self.seeds = None

  def train(self, event, sigma_train, learning_rate, reg_c=1.0e-3):
    self.retina.set_event(event.hits)
    true_params = self.retina.tracks_to_model_params(event)

    for shared, p in zip(self.true_parameters_shareds, true_params):
      shared.set_value(p.astype('float32'))

    self.seeds = self.seeder(self.n_seeds)

    sigma = np.array(sigma_train, dtype='float32')
    learning_rate = np.array(learning_rate, dtype='float32')
    reg_c = np.array(reg_c, dtype='float32')

    return self._train(*self.seeds + [sigma, learning_rate, reg_c])

  def maxima(self):
    self.seeds = self.seeder(self.n_seeds)

    staged = self.predictions(*self.seeds)
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

