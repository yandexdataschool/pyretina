import numpy as np

from pyretina.geometry import to_spherical

def pseudo_rapidity_prior(hits, size, pseudo_rapidity_range = (1.0, 5.0)):
  low, high = pseudo_rapidity_range

  pr = np.random.uniform(size = size, low=low, high=high)
  theta = 2.0 * np.arctan(np.exp(-pr))

  phi = np.random.uniform(size = size, low=0.0, high=2 * np.pi)

  ns = np.ndarray(shape = (size, 3), dtype='float64')

  z = np.cos(theta)
  y = np.sin(theta) * np.cos(phi)
  x = np.sin(theta) * np.sin(phi)

  ns[:, 0] = x
  ns[:, 1] = y
  ns[:, 2] = z

  return to_spherical(ns)

def hits_prior(hits, size):
  idx = np.random.choice(hits.shape[0], size=size, replace=False)
  return to_spherical(hits[idx, :])





