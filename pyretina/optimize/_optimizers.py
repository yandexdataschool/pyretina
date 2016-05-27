import numpy as np

def newton(f, x0, jac, hess, options):
  A = hess(x0) + np.eye(x0.shape[0]) * 1.0e-6
  j = jac(x0)
  return x0 - np.dot(np.linalg.inv(A), j)

def grad(f, x0, jac, hess, options):
  alpha = options['alpha']
  j = jac(x0)
  return x0 - alpha * j

def exp_fit(f, x0, jac, hess, options):
  s = options['sigma']
  return x0 + 0.5 * jac(x0) * s * s  / f(x0)