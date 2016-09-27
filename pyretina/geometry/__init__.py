import numpy as np

def to_cartesian(sps):
  if len(sps.shape) == 1:
    n = np.ndarray(shape=(3,))
    n[0] = np.sin(sps[0])
    n[1] = np.cos(sps[0]) * np.sin(sps[1])
    n[2] = np.cos(sps[0]) * np.cos(sps[1])

    return n
  else:
    n = np.ndarray(shape=(sps.shape[0], 3))
    n[:, 0] = np.sin(sps[:, 0])
    n[:, 1] = np.cos(sps[:, 0]) * np.sin(sps[:, 1])
    n[:, 2] = np.cos(sps[:, 0]) * np.cos(sps[:, 1])

    return n

def to_spherical(ns):
  sps = np.ndarray(shape=(ns.shape[0], 2))

  norm = np.sqrt(np.sum(ns ** 2, axis=1))
  normed = ns / np.vstack([norm] * 3).T

  sps[:, 0] = np.arcsin(normed[:, 0])
  sps[:, 1] = np.arctan2(normed[:, 1], normed[:, 2])
  return sps

def spherical_cos_angle(p1, p2):
  return np.sum(p1 * p2)

def spherical_angle(p1, p2):
  cos = spherical_cos_angle(p1, p2)
  cos = np.max([-1.0, cos])
  cos = np.min([1.0, cos])
  return np.arccos(cos)

def to_reference_plane(event, reference_z = 700.0):
  z0 = event.z0
  ns = event.tracks

  return ns[:, :2] * (reference_z - z0) / (ns[:, 2][:, None] - z0)

