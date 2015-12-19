import numpy as np

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

def grid_search(event):
  thetas, phis, response = event.get_grid()

  m = maxima(response)
  mis = np.where(m)

  return np.hstack([thetas[mis], phis[mis]])
