import numpy as np
from pyretina.retina import linear_retina_response,\
  neg_linear_retina_response, neg_linear_retina_response_jac, neg_linear_retina_response_hess,\
  set_event, set_sigma, set_z0

from _initial_guess import pseudo_rapidity_prior

from _optimize import optimize, optimize_step
from _solution import Solution

def swarm_search(hits, z0 = 0.0,
                 optimizer = "Newton-CG", optimizer_options = None, initial_guess = pseudo_rapidity_prior,
                 swarm_size = 500, swarm_lifetime = 5,
                 sigma_strategy = 5.0):

  set_event(hits)
  set_z0(z0)
  set_sigma(sigma_strategy)

  maxima = np.zeros(shape=(swarm_size, 2))
  traces = list()

  nfev = 0
  njev = 0
  nhev = 0

  for i in xrange(swarm_size):
    x0 = initial_guess(hits, 1)[0, :]
    sol, trace = optimize(x0, method=optimizer, max_iter=swarm_lifetime, options = optimizer_options)
    traces.append(trace)
    maxima[i, :] = sol.x

    nfev += sol.nfev
    njev += sol.njev
    nhev += sol.nhev

  scores = np.ndarray(shape=swarm_size)
  for i in xrange(swarm_size):
    x = maxima[i, :]
    scores[i] = linear_retina_response(x)

  return Solution(
    maxima = maxima,
    scores = scores,
    traces = traces,
    nfev = nfev,
    njev = njev,
    nhev = nhev
  )

def swarm_search_step(hits, z0 = 0.0,
                      optimizer = "Newton-CG", optimizer_options = None, initial_guess = pseudo_rapidity_prior,
                      swarm_size = 500, swarm_lifetime = 5,
                      sigma_strategy = 5.0):

  set_event(hits)
  set_z0(z0)

  if type(sigma_strategy) is float:
    sigma_strategy = np.ones(swarm_lifetime) * sigma_strategy

  traces = np.ndarray(shape=(swarm_size, swarm_lifetime + 1, 2))
  traces[:, 0, :] = initial_guess(hits, swarm_size)

  nfev = 0
  njev = 0
  nhev = 0

  for t in xrange(swarm_lifetime):
    set_sigma(sigma_strategy[t])
    if 'sigma' in optimizer_options:
      optimizer_options['sigma'] = sigma_strategy[t]

    for i in xrange(swarm_size):
      x0 = traces[i, t, :]
      sol = optimize_step(x0, method = optimizer, options=optimizer_options)

      nfev += sol.nfev
      njev += sol.njev
      nhev += sol.nhev

      traces[i, t + 1, :] = sol.x

  scores = np.ndarray(shape=swarm_size)

  for i in xrange(swarm_size):
    x = traces[i, -1, :]
    scores[i] = linear_retina_response(x)

  return Solution(
    maxima = traces[:, -1, :],
    scores = scores,
    traces = traces,
    nfev = nfev,
    njev = njev,
    nhev = nhev
  )

