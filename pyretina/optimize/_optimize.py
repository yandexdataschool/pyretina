import numpy as np
import scipy.optimize as opt
from pyretina.retina import neg_linear_retina_response, neg_linear_retina_response_jac, neg_linear_retina_response_hess

class TraceCallback:
  def __init__(self, x0):
    self.trace = list()
    self.trace.append(x0)

  def __call__(self, xk):
    self.trace.append(xk)

  def get(self):
    return np.array(self.trace)

def optimize(x0, method = "Newton-CG", max_iter = 10, options = None):
  if options is None:
    options = dict()

  options['maxiter'] = max_iter

  callback = TraceCallback(x0)

  sol = opt.minimize(neg_linear_retina_response, x0, method=method,
                     jac = neg_linear_retina_response_jac,
                     hess = neg_linear_retina_response_hess,
                     options=options, callback=callback)

  trace = callback.get()
  return sol, trace

from _optimizers import newton, grad, exp_fit

__optimizers = {
  'newton' : newton,
  'grad' : grad,
  'exp_fit' : exp_fit
}

def optimize_step(x0, method = "dogleg", options = None):
  if method in __optimizers:
    algo = __optimizers[method]
    return algo(neg_linear_retina_response, x0,
                jac = neg_linear_retina_response_jac,
                hess = neg_linear_retina_response_hess,
                options=options)
  else:
    return optimize_step_scipy(x0, method, options)

def optimize_step_scipy(x0, method = "dogleg", options = None):
  if options is None:
    options = dict()

  options['maxiter'] = 1

  return opt.minimize(neg_linear_retina_response, x0, method=method,
                      jac = neg_linear_retina_response_jac,
                      hess = neg_linear_retina_response_hess,
                      options=options)
