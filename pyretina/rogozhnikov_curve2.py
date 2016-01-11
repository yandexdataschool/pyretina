import theano
import theano.tensor as T
import numpy as np

def log1p_exp(x):
  return T.log1p(T.exp(-x)) + x

def __hessian(cost, variables):
  hessians = []
  for input1 in variables:
    d_cost_d_input1 = T.grad(cost, input1)
    hessians.append([
      T.grad(d_cost_d_input1, input2, disconnected_inputs='ignore', return_disconnected="zero") for input2 in variables
    ])

  return hessians

### Args

_ax = T.dscalar("ax")
_bx = T.dscalar("bx")

_ay = T.dscalar("ay")
_by = T.dscalar("by")

_gamma = T.dscalar("gamma")

_z0 = T.dscalar("z0")
_x0 = T.dscalar("x0")
_y0 = T.dscalar("y0")

_true_zs = theano.shared(np.zeros(shape=(1,)))

### Curves

_zs = _true_zs - _z0

_xs = _ax * _zs + _bx / _gamma * log1p_exp(_gamma * _zs) - _x0
_ys = _ay * _zs + _by / _gamma * log1p_exp(_gamma * _zs) - _y0

_true_xs = theano.shared(np.zeros(shape=(1, )))
_true_ys = theano.shared(np.zeros(shape=(1, )))

### Errors

_MSE = T.mean((_xs - _true_xs) ** 2 + (_ys - _true_ys) ** 2) / 2.0

_c_penalty = 0.01
_gamma_penalty = 0.1
_penalty = _c_penalty * (_ax ** 2 + _bx ** 2 + _ay ** 2 + _by ** 2) / 4.0 + _gamma_penalty * _gamma ** 2

_error = _MSE + _penalty

_error_jac = T.jacobian(_error, [_ax, _bx, _ay, _by, _x0, _y0, _z0, _gamma])

_error_hess = __hessian(_error, [_ax, _bx, _ay, _by, _x0, _y0, _z0, _gamma])

### Functions

_MSE_f = theano.function([_ax, _bx, _ay, _by, _x0, _y0, _z0, _gamma], _MSE)

_error_f = theano.function([_ax, _bx, _ay, _by, _x0, _y0, _z0, _gamma], _error)
_error_v = lambda params: _error_f(*params)

_error_jac_f = theano.function([_ax, _bx, _ay, _by, _x0, _y0, _z0, _gamma], _error_jac)
_error_jac_v = lambda params: np.hstack(_error_jac_f(*params))

_error_hess_f = [
  [
    theano.function([_ax, _bx, _ay, _by, _x0, _y0, _z0, _gamma], hess, on_unused_input='ignore')
    for hess in hess_row
  ]

  for hess_row in _error_hess
]
_error_hess_v = lambda params: np.array([
  [ hess(*params) for hess in hess_row ]
  for hess_row in _error_hess_f
])

_curve_x = theano.function([_ax, _bx, _x0, _z0, _gamma], _xs)
_curve_y = theano.function([_ay, _by, _y0, _z0, _gamma], _ys)

class RogozhnikovCurve2:
  def __spline(self, x, y, z, spline_n=25):
    from scipy import interpolate as spinter

    z_grid = np.linspace(np.min(z), np.max(z), spline_n)
    splixner = spinter.UnivariateSpline(z, x)(z_grid)
    spliyner = spinter.UnivariateSpline(z, y)(z_grid)

    return splixner, spliyner, z_grid

  def __init__(self, x, y, z, spline=False, spline_n=25):
    if spline:
      x_, y_, z_ = self.__spline(x, y, z, spline_n)
    else:
      x_, y_, z_ = x, y, z

    _true_xs.set_value(x_)
    _true_ys.set_value(y_)
    _true_zs.set_value(z_)

  def mse(self):
    return _error_v, _error_jac_v, _error_hess_v