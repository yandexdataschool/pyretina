from scipy import optimize
import numpy as np

class PathCallback:
  def __init__(self):
    self.trace = list()

  def __call__(self, p0):
    self.trace.append(p0)

  def get_trace(self):
    return self.trace

def multi_start(event, max_evaluations = 1000, method ='CG', solver_options = None):
  dim = event.track_dim()
  bounds = event.track_bounds()

  r, r_jac, r_hess = event.get_neg_response()

  results = list()
  traces = list()

  def generate_initial():
    p0 = np.random.uniform(size=dim)
    p0 = p0 * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    return p0.reshape(dim)

  evaluations_left = max_evaluations

  while evaluations_left > 0:
    p0 = generate_initial()

    callback = PathCallback()
    callback(p0)

    if solver_options is None:
      solver_options = dict()

    run_options = solver_options.copy()
    run_options['maxiter'] = evaluations_left

    result = optimize.minimize(
      fun = r,
      x0 = p0,
      method = method,
      jac = r_jac,
      hess = r_hess,
      options = solver_options,
      callback = callback
    )

    evaluations_left -= result.nfev
    evaluations_left -= result.njev
    evaluations_left -= result.nhev

    if evaluations_left >= 0:
      results.append(result.x)
      traces.append(callback.get_trace())

  return np.vstack(results), traces


