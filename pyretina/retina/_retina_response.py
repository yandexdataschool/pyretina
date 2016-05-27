import numpy as np

import theano
import theano.tensor as T

__event_x = theano.shared(np.zeros((1,), dtype="float64"), 'event_x')
__event_y = theano.shared(np.zeros((1,), dtype="float64"), 'event_y')
__event_z = theano.shared(np.zeros((1,), dtype="float64"), 'event_z')

__event = [__event_x, __event_y, __event_z]

def set_event(e):
  __event_x.set_value(e[:, 0])
  __event_y.set_value(e[:, 1])
  __event_z.set_value(e[:, 2])

__sigma = theano.shared(0.05, 'sigma', allow_downcast=True)

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

### Normalized direction vector
__n_x = T.sin(__theta)
__n_y = T.cos(__theta) * T.sin(__phi)
__n_z = T.cos(__theta) * T.cos(__phi)

__z0 = theano.shared(0.0, 'z0', allow_downcast=True)

def set_z0(z0):
  __z0.set_value(z0)

_n = [__n_x, __n_y, __n_z]

__scalar = (__event_z - __z0) / __n_z

### Difference between xy-projection of n and hit
__delta_square = (__scalar * __n_x - __event_x) ** 2 + (__scalar * __n_y - __event_y) ** 2

__r = T.sum(T.exp(-__delta_square / __sigma))

__linear_retina_response = theano.function([__theta, __phi], __r)
linear_retina_response = lambda params: __linear_retina_response(*params)
neg_linear_retina_response = lambda params: -__linear_retina_response(*params)

__linear_retina_response_jac = theano.function([__theta, __phi], theano.gradient.jacobian(__r, [__theta, __phi]))
linear_retina_response_jac = lambda params: np.array(__linear_retina_response_jac(*params))
neg_linear_retina_response_jac = lambda params: -np.array(__linear_retina_response_jac(*params))

__second_derivatives = [
  [theano.function([__theta, __phi], d) for d in dd]
  for dd in __hessian(__r, [__theta, __phi])
]

linear_retina_response_hess = lambda params: np.array([
  [dd(*params) for dd in dddd ]
  for dddd in __second_derivatives
])

neg_linear_retina_response_hess = lambda params: -linear_retina_response_hess(params)

linear_retina_response_vec = np.vectorize(__linear_retina_response)

def linear_retina_response_jac_vec(thetas, phis):
  n = np.product(thetas.shape)
  drdt = np.zeros(shape=n)
  drdp = np.zeros(shape=n)

  ts = thetas.ravel()
  ps = phis.ravel()

  for i in xrange(ts.shape[0]):
    jac = __linear_retina_response_jac(ts[i], ps[i])
    drdt[i] = jac[0]
    drdp[i] = jac[1]

  return drdt.reshape(thetas.shape), drdp.reshape(thetas.shape)
