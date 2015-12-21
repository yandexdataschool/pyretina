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
  sps = np.ndarray(shape=( ns.shape[0], 2))
  sps[:, 0] = np.arcsin(ns[:, 0])
  sps[:, 1] = np.arctan2(ns[:, 1], ns[:, 2])
  return sps

def spherical_cos_angle(spherical_p1, spherical_p2):
  p1 = to_cartesian(spherical_p1)
  p2 = to_cartesian(spherical_p2)

  if len(p1.shape) == 1:
    return np.sum(p1 * p2)
  else:
    return np.sum(p1 * p2, axis=1)

def spherical_angle(spherical_p1, spherical_p2):
  cos = spherical_cos_angle(spherical_p1, spherical_p2)
  cos = np.max([-1.0, cos])
  cos = np.min([1.0, cos])
  return np.arccos(cos)