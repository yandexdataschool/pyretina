from scipy import optimize
import numpy as np

class PathCallback:
  def __init__(self):
    self.trace = list()

  def __call__(self, p0):
    self.trace.append(p0)

  def get_trace(self):
    return self.trace

def __generate_initial(area):
  dim = area.shape[0]
  p0 = np.random.uniform(size=dim)
  p0 = p0 * (area[:, 1] - area[:, 0]) + area[:, 0]
  return p0.reshape(dim)

def gen_multi_start(search_area,
                    f, f_jac=None, f_hess=None,
                    max_evaluations = 1000, method ='Powell', solver_options = None):
  results = list()
  traces = list()

  evaluations_left = max_evaluations or 18446744073709551615l

  while evaluations_left > 0:
    p0 = __generate_initial(search_area)

    callback = PathCallback()
    callback(p0)

    if solver_options is None:
      solver_options = dict()

    run_options = solver_options.copy()
    run_options['maxiter'] = evaluations_left

    result = optimize.minimize(
      fun = f,
      x0 = p0,
      method = method,
      jac = f_jac,
      hess = f_hess,
      options = run_options,
      callback = callback
    )

    evaluations_left -= result.nfev
    evaluations_left -= result.njev
    evaluations_left -= result.nhev

    if evaluations_left >= 0:
      results.append(result)
      traces.append(callback.get_trace())

  return results, traces

def multistart_until(search_area,
                     max_fun,
                     f,
                     f_jac=None, f_hess=None,
                     n_attempts = 1000,
                     method = 'Powell', solver_options = None,
                     strict = True):
  results = list()

  for _ in xrange(n_attempts):
    p0 = __generate_initial(search_area)

    callback = PathCallback()
    callback(p0)

    if solver_options is None:
      solver_options = dict()

    result = optimize.minimize(
      fun = f,
      x0 = p0,
      method = method,
      jac = f_jac,
      hess = f_hess,
      options = solver_options,
      callback = callback
    )

    if result.fun <= max_fun:
      return result
    else:
      results.append(result)

  if strict:
    raise Exception("Maximum number of multistart attempts has been reached, but there is still no valid minima.")
  else:
    i = np.argmin([ r.fun for r in results ])
    return results[i]