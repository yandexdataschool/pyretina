from pyretina.utils.theanops import *

import matplotlib.pyplot as plt

theano.config.optimizer = 'fast_compile'

_x = theano.shared(np.zeros((1,), dtype='double'), name='x')
_true_y = theano.shared(np.zeros((1,), dtype='double'), name='true_y')

_a = T.dscalar('a')
_b = T.dscalar('b')

_x0 = T.dscalar('x0')
_y0 = T.dscalar('y0')

_gamma = T.dscalar('gamma')

_gamma_ = _gamma / 100.0

_x_c = _x - _x0

_y = _a * _x_c + _b * log1p_exp(_gamma_ * _x_c) / _gamma_ + _y0

_y_f = theano.function([_a, _b, _x0, _y0, _gamma], _y)

_MSE = T.mean((_true_y - _y) ** 2)

_error = _MSE + _gamma_ ** 2

_error_f = theano.function([_a, _b, _x0, _y0, _gamma], _error)

_error_v = lambda p: _error_f(*p)

_error_jac = T.jacobian(_error, [_a, _b, _x0, _y0, _gamma])

_error_jac_f = theano.function([_a, _b, _x0, _y0, _gamma], _error_jac)

_error_jac_v = lambda p: np.hstack(_error_jac_f(*p))

def _set(x_, y_):
  _x.set_value(x_)
  _true_y.set_value(y_)


def _prefit(x, y, x0, y0, bend_x_r):
  cut_before = x < x0 - bend_x_r
  cut_after = x > x0 + bend_x_r

  x_before = x[cut_before].reshape(-1, 1)
  x_after = x[cut_after].reshape(-1, 1)

  y_before = y[cut_before]
  y_after = y[cut_after]

  def fit(x_, y_):
    if x_.shape[0] > 1 and y_.shape[0] > 1:
      c, residuals, rank, s = np.linalg.lstsq(x_, y_)
      return c[0], residuals
    else:
      err = np.sum(y_ ** 2)
      return 0.0, err

  c_before, error_before = fit(x_before - x0, y_before - y0)
  c_after, error_after = fit(x_after - x0, y_after - y0)

  error = error_before / np.sum(cut_before) + error_after / np.sum(cut_after)

  return np.array([c_before, c_after]), error

class RogozhnikovCurve:
  def __init__(self, spline=30, bend_r=None, x0=None):
    self.spline = spline
    self.bend_r = bend_r
    self.x0_guess = x0

    self.a = None
    self.b = None
    self.x0 = None
    self.y0 = None
    self.gamma = None
    self.sol = None

  def curve(self, x):
    _x.set_value(x)
    return _y_f(self.a, self.b, self.x0, self.y0, self.gamma)

  def fast_fit(self, x_, y_):
    from scipy import optimize as opt
    from scipy import interpolate

    spline_y = interpolate.UnivariateSpline(x_, y_)

    if type(self.spline) is int:
      x = np.linspace(np.min(x_), np.max(x_), self.spline)
      y = spline_y(x)
    else:
      x = x_
      y = y_

    bend_r = self.bend_r or (np.max(x_) - np.min(x_)) / 8.0

    x0 = self.x0_guess
    y0 = spline_y(x0)
    c_guess = _prefit(x, y, x0, y0, bend_r)[0]

    error = _error_v
    jac = _error_jac_v
    _set(x, y)

    gamma0 = 1.0
    guess = np.array([c_guess[0], c_guess[1] - c_guess[0], x0, y0, gamma0])

    sol = opt.minimize(error, jac = jac,
                       method="BFGS",
                       x0=guess,
                       options={ 'gtol' : 1.0e-3, 'maxiter' : 100 })

    sol = opt.minimize(error,
                       method = "Powell",
                       x0 = sol.x,
                       options={ 'xtol' : 1.0e-2 })

    self.a, self.b, self.x0, self.y0, self.gamma = sol.x

    self.sol = sol

    return self

  def asymptotes(self):
    return self.a, self.b - self.a