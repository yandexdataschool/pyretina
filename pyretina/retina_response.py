import numpy as np

import theano
import theano.tensor as T

__event_x = theano.shared(np.zeros((1,), dtype="double"), 'event_x')
__event_y = theano.shared(np.zeros((1,), dtype="double"), 'event_y')
__event_z = theano.shared(np.zeros((1,), dtype="double"), 'event_z')

__event = [__event_x, __event_y, __event_z]

__sigma = theano.shared(0.05, 'sigma')

def set_event(e):
  __event_x.set_value(e[:, 0])
  __event_y.set_value(e[:, 1])
  __event_z.set_value(e[:, 2])

def set_sigma(s):
  __sigma.set_value(s * s)

def __hessian(cost, variables):
  hessians = []
  for input1 in variables:
    d_cost_d_input1 = T.grad(cost, input1)
    hessians.append([
      T.grad(d_cost_d_input1, input2) for input2 in variables
    ])

  return hessians

__theta = T.dscalar("theta")
__phi = T.dscalar("phi")

### Normal vector
__n_x = T.sin(__theta)
__n_y = T.cos(__theta) * T.sin(__phi)
__n_z = T.cos(__theta) * T.cos(__phi)

_n = [__n_x, __n_y, __n_z]

### Scalar product between normal vector and hits
__scalar = reduce(T.add, [n_i * e_i for (n_i, e_i) in zip(_n, __event)])

### Difference between projection on n and hit
__delta = [__scalar * n_i - e_i for (e_i, n_i) in zip(__event, _n)]
__delta_square = reduce(T.add, [d_i * d_i for d_i in __delta])

__r = T.sum(T.exp(-__delta_square / __sigma))

__linear_retina_response = theano.function([__theta, __phi], __r)
linear_retina_response = lambda params: __linear_retina_response(params[0], params[1])
neg_linear_retina_response = lambda params: -__linear_retina_response(params[0], params[1])

__linear_retina_response_jac = theano.function([__theta, __phi], theano.gradient.jacobian(__r, [__theta, __phi]))
linear_retina_response_jac = lambda params: np.array(__linear_retina_response_jac(params[0], params[1]))
neg_linear_retina_response_jac = lambda params: -np.array(__linear_retina_response_jac(params[0], params[1]))

__second_derivatives = [ [theano.function([__theta, __phi], d) for d in dd] for dd in __hessian(__r, [__theta, __phi])]

linear_retina_response_hess = lambda params: np.array([
  [dd(params[0], params[1]) for dd in dddd ]
  for dddd in __second_derivatives
])

neg_linear_retina_response_hess = lambda params: -linear_retina_response_hess(params)