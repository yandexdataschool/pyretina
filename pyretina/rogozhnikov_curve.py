import theano
import theano.tensor as T
import numpy as np
from sklearn.linear_model import LinearRegression

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

def __hessian(cost, variables):
  hessians = []
  for input1 in variables:
    d_cost_d_input1 = T.grad(cost, input1)
    hessians.append([
      T.grad(d_cost_d_input1, input2) for input2 in variables
    ])

  return hessians

def _unparams(params):
  cx = params[:2]
  cy = params[2:4]
  c0x = params[4]
  c0y = params[5]
  z0 = params[6]
  gamma = params[7]
  return cx, cy, c0x, c0y, z0, gamma

def _inparams(cx, cy, c0x, c0y, z0, gamma):
  params = np.ndarray(shape=(8, ), dtype=np.double)

  params[:2] = cx
  params[2:4] = cy
  params[4] = c0x
  params[5] = c0y
  params[6] = z0
  params[7] = gamma
  return params

_penalty_c = 0.1
_penalty_gamma = 0.1

_cx = T.dvector()
_cy = T.dvector()

_gamma = T.dscalar()
_z0 = T.dscalar()

_c0x = T.dscalar()
_c0y = T.dscalar()

_true_zs = theano.shared(np.zeros(shape=(1,)))

_zc = _true_zs - _z0

_sx = T.tensordot(_cx, _zc, axes=([], []))
_xs = log_sum_exp(_sx * _gamma, axis=0) / _gamma + _c0x

_sy = T.tensordot(_cy, _zc, axes=([], []))
_ys = log_sum_exp(_sy * _gamma, axis=0) / _gamma + _c0y

_true_xs = theano.shared(np.zeros(shape=(1, )))
_true_ys = theano.shared(np.zeros(shape=(1, )))

_MSE = T.mean((_xs - _true_xs) ** 2 + (_ys - _true_ys) ** 2) / 2.0
_penalize = lambda c: T.sum(c ** 2)
_error = _MSE + _penalty_c * (_penalize(_cx) + _penalize(_cy)) + _penalty_gamma * _penalize(_gamma)

_error_jac = T.jacobian(_error, [_cx, _cy, _c0x, _c0y, _z0, _gamma])

_MSE_f = theano.function([_cx, _cy, _c0x, _c0y, _z0, _gamma], _MSE)

_error_f = theano.function([_cx, _cy, _c0x, _c0y, _z0, _gamma], _error)
_error_v = lambda params: _error_f(*_unparams(params))

_error_partial_v = lambda cx, cy: lambda p: _error_f(cx, cy, p[0], p[1], p[2], p[3])

_error_jac_f = theano.function([_cx, _cy, _c0x, _c0y, _z0, _gamma], _error_jac)
_error_jac_v = lambda params: np.hstack(_error_jac_f(*_unparams(params)))

_error_jac_partial_v = lambda cx, cy: lambda p: np.hstack(_error_jac_f(cx, cy, p[0], p[1], p[2], p[3]))[4:]

_curve_x = theano.function([_cx, _c0x, _z0, _gamma], _xs)
_curve_y = theano.function([_cy, _c0y, _z0, _gamma], _ys)

_default_x0 = np.array([-0.1, 0.3, 0.025, 0.05, 0, -200, 6000, 1.0e-2])

lr = LinearRegression(fit_intercept=True)

default_gamma = 1.0e-2

class RogozhnikovCurve:
  def __spline(self, spline_n=25):
    from scipy import interpolate as spinter

    z_grid = np.linspace(np.min(self.z), np.max(self.z), spline_n)
    splixner = spinter.UnivariateSpline(self.z, self.x)(z_grid)
    spliyner = spinter.UnivariateSpline(self.z, self.y)(z_grid)

    return splixner, spliyner, z_grid

  def __setup(self, spline=False, spline_n=25):
    if spline:
      x_, y_, z_ = self.__spline(spline_n)
    else:
      x_, y_, z_ = self.x, self.y, self.z

    _true_xs.set_value(x_)
    _true_ys.set_value(y_)
    _true_zs.set_value(z_)

  def __init__(self, x, y, z, spline=False, spline_n=25, param_guess = None):
    self.error = None

    self.x = x
    self.y = y
    self.z = z

    self.__setup(spline, spline_n)

    self.params = param_guess

  def mse(self, spline=False, spline_n=25):
    self.__setup(spline, spline_n)
    return _error_v, _error_jac_v

  def prefit(self,
             before_magnet_cut = 2700.0,
             after_magnet_cut = 7500.0,
             z0 = 5000.0):
    """
    Prefit return estimation of initial parameters by fitting 2 lines in areas z < before_magnet_cut and
    z > after_magnet_cut.
    :param x: data
    :param y: data
    :param z: data
    :param before_magnet_cut:
    :param after_magnet_cut:
    :param z0: for c0x and c0y approximation
    :return:
    """

    x, y, z = self.x, self.y, self.z

    bx = x[:2]
    by = y[:2]
    bz = z[:2]

    ax = x[-2:]
    ay = y[-2:]
    az = z[-2:]

    bcx = lr.fit(bz.reshape(-1, 1), bx).coef_[0]
    c0x = lr.predict(np.array([[z0]]))

    bcy = lr.fit(bz.reshape(-1, 1), by).coef_[0]
    c0y = lr.predict(np.array([[z0]]))

    acx = lr.fit(az.reshape(-1, 1), ax).coef_[0]
    c0x += lr.predict(np.array([[z0]]))

    acy = lr.fit(az.reshape(-1, 1), ay).coef_[0]
    c0y += lr.predict(np.array([[z0]]))

    c0x = c0x[0] / 2.0
    c0y = c0y[0] / 2.0

    cx = np.array([bcx, acx])
    cy = np.array([bcy, acy])

    return cx, cy, c0x, c0y, z0, default_gamma

  def fit(self, spline=False, spline_n=25):
    import scipy.optimize as opt

    sol = opt.minimize(_error_v, jac=_error_jac_v, method="BFGS", x0=self.params, options = { "gtol" : 1.0e1 })

    self.params = sol.x
    self.error = _MSE_f(*_unparams(self.params))
    return self

  def fast_fit(self,
               before_magnet_cut = 2700.0,
               after_magnet_cut = 7500.0,
               z0 = 5000.0):
    import scipy.optimize as opt

    cx, cy, c0x, c0y, z0, gamma = self.prefit(before_magnet_cut, after_magnet_cut, z0)

    p_error = _error_partial_v(cx, cy)
    p_error_jac = _error_jac_partial_v(cx, cy)

    sol = opt.minimize(p_error, jac=p_error_jac, method="BFGS", x0=np.array([c0x, c0y, z0, gamma]), options = { "gtol" : 1.0e-3 })
    c0x, c0y, z0, gamma = sol.x

    print "Preotpimization:", sol.fun

    sol = opt.minimize(_error_v, jac=_error_jac_v, method="BFGS", x0=_inparams(cx, cy, c0x, c0y, z0, gamma))

    print "After:", sol.fun

    self.params = sol.x
    self.error = _MSE_f(*_unparams(self.params))
    return self

  def curve(self):
    def f(z):
      _true_zs.set_value(z)
      cx, cy, c0x, c0y, z0, gamma = _unparams(self.params)
      return _curve_x(cx, c0x, z0, gamma), _curve_y(cy, c0y, z0, gamma)
    return f