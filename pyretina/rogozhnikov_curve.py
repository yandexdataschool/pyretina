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


def _unparams(params):
  cx = params[:2]
  cy = params[2:4]
  c0x = params[4]
  c0y = params[5]
  z0 = params[6]
  gamma = params[7]
  return cx, cy, c0x, c0y, z0, gamma

_penalty = 0.01

_cx = T.dvector()
_cy = T.dvector()

_gamma = T.dscalar()
_z0 = T.dscalar()

_c0x = T.dscalar()
_c0y = T.dscalar()

_true_zs = theano.shared(np.zeros(shape=(1,)))

_zc = _true_zs - _z0

_sx = T.tensordot(_cx, _zc, axes=([], []))
_xs = log_sum_exp(_sx * _gamma, axis=0) / _gamma - _c0x

_sy = T.tensordot(_cy, _zc, axes=([], []))
_ys = log_sum_exp(_sy * _gamma, axis=0) / _gamma - _c0y

_true_xs = theano.shared(np.zeros(shape=(1, )))
_true_ys = theano.shared(np.zeros(shape=(1, )))

_MSE = T.mean((_xs - _true_xs) ** 2 + (_ys - _true_ys) ** 2) / 2.0
_penalize = lambda c: T.sum(c ** 2)
_error = _MSE + _penalty * (_penalize(_cx) + _penalize(_cy))

_error_jac = T.jacobian(_error, [_cx, _cy, _c0x, _c0y, _z0, _gamma])

_MSE_f = theano.function([_cx, _cy, _c0x, _c0y, _z0, _gamma], _MSE)

_error_f = theano.function([_cx, _cy, _c0x, _c0y, _z0, _gamma], _error)
_error_v = lambda params: _error_f(*_unparams(params))

_error_jac_f = theano.function([_cx, _cy, _c0x, _c0y, _z0, _gamma], _error_jac)
_error_jac_v = lambda params: np.hstack(_error_jac_f(*_unparams(params)))

_curve_x = theano.function([_cx, _c0x, _z0, _gamma], _xs)
_curve_y = theano.function([_cy, _c0y, _z0, _gamma], _ys)

_default_x0 = np.array([-0.1, 0.3, 0.025, 0.05, 0, -200, 6000, 1.0e-2])

class RogozhnikovCurve:
  def __init__(self, penalty, x0 = None):
    self.penalty = penalty
    self.error = None

    if x0 is None:
      self.params = _default_x0.copy()
    else:
      self.params = x0.copy()

  def fit(self, x, y, z, spline=False, spline_n=25):
    import scipy.optimize as opt
    from scipy import interpolate as spinter

    if spline:
      z_grid = np.linspace(np.min(z), np.max(z), spline_n)
      splixner = spinter.UnivariateSpline(z, x)(z_grid)
      spliyner = spinter.UnivariateSpline(z, y)(z_grid)

      _true_xs.set_value(splixner)
      _true_ys.set_value(spliyner)
      _true_zs.set_value(z_grid)
    else:
      _true_xs.set_value(x)
      _true_ys.set_value(y)
      _true_zs.set_value(z)

    sol = opt.minimize(_error_v, jac=_error_jac_v, method="BFGS", x0=self.params, options = { "gtol" : 1.0e1 })

    self.params = sol.x
    self.error = _MSE_f(*_unparams(self.params))
    return self

  def curve(self):
    def f(z):
      _true_zs.set_value(z)
      cx, cy, c0x, c0y, z0, gamma = _unparams(self.params)
      return _curve_x(cx, c0x, z0, gamma), _curve_y(cy, c0y, z0, gamma)
    return f