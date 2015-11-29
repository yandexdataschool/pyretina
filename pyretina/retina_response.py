import numpy as np

from geometry import *

def load_events(path):
  data = np.genfromtxt(path, delimiter=",", skip_header=1)

  event_ids = set(data[:, 0])

  return [ data[data[:, 0] == eid][:, -3:] for eid in event_ids ]

def load_tracks(path):
  return np.loadtxt(path)

def retina_response_n(event, n, sigma):
  scalar = np.dot(event, n.T).reshape((event.shape[0], ))

  projections = np.ndarray(shape=event.shape)
  projections[:, 0] = scalar * n[:, 0]
  projections[:, 1] = scalar * n[:, 1]
  projections[:, 2] = scalar * n[:, 2]

  projections = event - projections
  deltas = np.sum(projections * projections, axis=1)

  return np.sum(np.exp(-deltas / (sigma * sigma)))

def retina_spherical(event, theta, phi, sigma):
  n = to_cartesian(theta, phi)
  return retina_response_n(event, n, sigma)

def _retina_grid(event, thetas, phis, sigma):
  response = lambda theta, phi: retina_spherical(event, theta, phi, sigma)
  return np.vectorize(response)(thetas, phis)

def retina_grid(event, theta_limits, theta_bins, phi_limits, phi_bins, sigma):
  thetas, phis = np.meshgrid(
    np.linspace(theta_limits[0], theta_limits[1], theta_bins),
    np.linspace(phi_limits[0], phi_limits[1], phi_bins)
  )

  return thetas, phis, _retina_grid(event, thetas, phis, sigma)