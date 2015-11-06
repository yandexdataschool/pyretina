import numpy as np

from ..retina_response import retina_grid

def grid_search(event, theta_limits, theta_bins, phi_limits, phi_bins, sigma):
  def shifted(m, delta):
    res = m
    for dim, d in enumerate(delta):
      res = np.roll(shifted, d, axis=dim)

    return res

  thetas, phis, response = retina_grid(event, theta_limits, theta_bins, phi_limits, phi_bins, sigma)
  deltas = [(1, 0), (-1, 0), (0, 1), (0, -1)]

  maxima = np.ones(response.shape, dtype=bool)

  for s in deltas:
    maxima = np.logical_and(maxima, response > shifted(response, s))

  ### Edges are automatically no
  maxima[:, 0] = False
  maxima[:, -1] = False
  maxima[0, :] = False
  maxima[-1, :] = False

  return maxima, np.where(maxima)


