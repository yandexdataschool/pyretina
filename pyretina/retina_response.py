import numpy as np

from geometry import *

def load_events(path):
  data = np.genfromtxt(path, delimiter=",", skip_header=1)

  event_ids = set(data[:, 0])

  return [ data[data[:, 0] == eid][:, -3:] for eid in event_ids ]

def load_tracks(path):
  return np.loadtxt(path)

def delta_vec(rs, n):
  scalar = np.dot(rs, n.T).reshape((rs.shape[0],))

  projections = np.ndarray(shape=rs.shape)
  projections[:, 0] = scalar * n[:, 0]
  projections[:, 1] = scalar * n[:, 1]
  projections[:, 2] = scalar * n[:, 2]

  projections = rs - projections
  return projections

def delta_sq(rs, n):
  d_v = delta_vec(rs, n)
  return np.sum(d_v * d_v, axis=1)

def delta(rs, n):
  d_v = delta_vec(rs, n)
  return np.sqrt(np.sum(d_v * d_v, axis=1))

def delta_vec_spherical_jac(rs, sp):
  sp_jac = np.zeros(shape=(3, 2))

  sp_jac[0, 0] = np.cos(sp[:, 0])
  sp_jac[1, 0] = -np.sin(sp[:, 0]) * np.sin(sp[:, 1])
  sp_jac[2, 0] = -np.sin(sp[:, 0]) * np.cos(sp[:, 1])

  sp_jac[1, 1] = np.cos(sp[:, 0]) * np.cos(sp[:, 1])
  sp_jac[2, 1] = -np.cos(sp[:, 0]) * np.sin(sp[:, 1])

  d_jac = np.zeros(shape=(rs.shape[0], 3, 2))

  n = to_cartesian(sp)

  r_n_dot = rs.dot(n.T)
  r_dn_dtheta_dot = rs.dot(sp_jac[:, 0]).reshape((rs.shape[0], 1))
  r_dn_dphi_dot = rs.dot(sp_jac[:, 1]).reshape((rs.shape[0], 1))

  d_jac[:, 0, 0] = (sp_jac[0, 0] * r_n_dot + n[0, 0] * r_dn_dtheta_dot).T
  d_jac[:, 1, 0] = (sp_jac[1, 0] * r_n_dot + n[0, 1] * r_dn_dtheta_dot).T
  d_jac[:, 2, 0] = (sp_jac[2, 0] * r_n_dot + n[0, 2] * r_dn_dtheta_dot).T

  d_jac[:, 0, 1] = (sp_jac[0, 1] * r_n_dot + n[0, 0] * r_dn_dphi_dot).T
  d_jac[:, 1, 1] = (sp_jac[1, 1] * r_n_dot + n[0, 1] * r_dn_dphi_dot).T
  d_jac[:, 2, 1] = (sp_jac[2, 1] * r_n_dot + n[0, 2] * r_dn_dphi_dot).T

  return d_jac

def delta_jac_unnormed(rs, sp):
  n = to_cartesian(sp)
  dj = np.zeros(shape=(rs.shape[0], 2))

  d = delta_vec(rs, n)

  dd = delta_vec_spherical_jac(rs, sp)

  dj[:, 0] = np.sum(dd[:, :, 0] * d, axis=1)
  dj[:, 1] = np.sum(dd[:, :, 1] * d, axis=1)

  return dj

def retina_response_jac(rs, sp, sigma):
  n = to_cartesian(sp)
  jac = np.zeros(shape=2)

  r = 2.0 / sigma / sigma * np.exp(-delta_sq(rs, n) / sigma / sigma)

  d_jac_unnormed = delta_jac_unnormed(rs, sp)

  jac[0] = np.sum(r * d_jac_unnormed[:, 0])
  jac[1] = np.sum(r * d_jac_unnormed[:, 1])

  return jac


def retina_response_n(event, n, sigma):
  scalar = np.dot(event, n.T).reshape((event.shape[0], ))

  projections = np.ndarray(shape=event.shape)
  projections[:, 0] = scalar * n[:, 0]
  projections[:, 1] = scalar * n[:, 1]
  projections[:, 2] = scalar * n[:, 2]

  projections = event - projections
  deltas = np.sum(projections * projections, axis=1)

  return np.sum(np.exp(-deltas / (sigma * sigma)))

def retina_response_spherical_jac(event, sp, sigma):
  pass

def retina_spherical(event, sp, sigma):
  n = to_cartesian(sp)
  return retina_response_n(event, n, sigma)

def _retina_grid(event, thetas, phis, sigma):
  response = lambda theta, phi: retina_spherical(event, np.array([[theta, phi]]), sigma)
  return np.vectorize(response)(thetas, phis)

def retina_grid(event, theta_limits, theta_bins, phi_limits, phi_bins, sigma):
  thetas, phis = np.meshgrid(
    np.linspace(theta_limits[0], theta_limits[1], theta_bins),
    np.linspace(phi_limits[0], phi_limits[1], phi_bins)
  )

  return thetas, phis, _retina_grid(event, thetas, phis, sigma)