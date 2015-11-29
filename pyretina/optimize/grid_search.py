import numpy as np

from ..retina_response import retina_grid

def maxima(response):
  def shifted(m, delta):
    res = m
    for dim, d in enumerate(delta):
      res = np.roll(res, d, axis=dim)

    return res

  deltas = [(1, 0), (-1, 0), (0, 1), (0, -1)]

  m = np.ones(response.shape, dtype=bool)

  for s in deltas:
    m = np.logical_and(m, response > shifted(response, s))

  ### Edges are automatically excluded
  m[:, 0] = False
  m[:, -1] = False
  m[0, :] = False
  m[-1, :] = False

  return m

def grid_search(event, theta_limits, theta_bins, phi_limits, phi_bins, sigma):
  thetas, phis, response = retina_grid(event, theta_limits, theta_bins, phi_limits, phi_bins, sigma)

  m = maxima(response)
  mis = np.where(m)

  return m, np.where(m), thetas[mis], phis[mis]
