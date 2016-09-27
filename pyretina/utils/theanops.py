import theano
import theano.tensor as T

import numpy as np

### Stolen from pylearn2
def log_sum_exp(A, axis=None):
  """
  A numerically stable expression for
  `T.log(T.exp(A).sum(axis=axis))`
  Parameters
  ----------
  A : theano.gof.Variable
      A tensor we want to compute the log sum exp of
  axis : int, optional
      Axis along which to sum
  log_A : deprecated
      `A` used to be named `log_A`. We are removing the `log_A`
      interface because there is no need for the input to be
      the output of theano.tensor.log. The only change is the
      renaming, i.e. the value of log_sum_exp(log_A=foo) has
      not changed, and log_sum_exp(A=foo) is equivalent to
      log_sum_exp(log_A=foo).
  Returns
  -------
  log_sum_exp : theano.gof.Variable
      The log sum exp of `A`
  """

  A_max = T.max(A, axis=axis, keepdims=True)
  B = (
    T.log(T.sum(T.exp(A - A_max), axis=axis, keepdims=True)) +
    A_max
  )

  if type(axis) is int:
    axis = [axis]

  return B.dimshuffle([i for i in range(B.ndim) if
                       i % B.ndim not in axis])

def derivatives(cost, wrts, stack = False):
  grads = [ T.grad(cost, wrt) for wrt in wrts ]
  hessian = [[ T.grad(gr, wrt) for wrt in wrts ] for gr in grads]

  if stack:
    grads = T.concatenate(grads)
    hessian = T.stack([T.stack(h_row) for h_row in hessian], axis=1)
    return grads, hessian
  else:
    return grads, hessian


def log1p_exp(a):
    return T.log1p(T.exp(-a)) + a

def jacobian_trace(cost, wrts):
  grads = []

  for wrt in wrts:
    dC_dwrt = theano.map(
      lambda i, r, w: T.grad(r[i], w[i]),
      sequences=T.arange(cost.shape[0]),
      non_sequences=[cost, wrt]
    )

    grads.append(dC_dwrt)

  return grads

def scalar_hessian(cost, variables):
  hessians = []
  for input1 in variables:
    d_cost_d_input1 = T.grad(cost, input1)
    hessians.append([
      T.grad(d_cost_d_input1, input2) for input2 in variables
    ])

  hessian_fs = [[
    theano.function(variables, h) for h in hrow
  ] for hrow in hessians]

  hessian_f = lambda params: np.vstack([[
     hf(*params) for hf in hrow
  ] for hrow in hessian_fs])

  return hessians, hessian_f

def __hessian(cost, variables):
  hessians = []
  for input1 in variables:
    d_cost_d_input1 = T.grad(cost, input1)
    hessians.append([
      T.grad(d_cost_d_input1, input2) for input2 in variables
    ])

  return hessians