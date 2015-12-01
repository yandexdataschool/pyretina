import numpy as np

def to_cartesian_array(sps):
  n = np.ndarray(shape=(sps.shape[0], 3))

  n[:, 0] = np.sin(sps[:, 0]) * np.cos(sps[:, 1])
  n[:, 1] = np.sin(sps[:, 0]) * np.sin(sps[:, 1])
  n[:, 2] = np.cos(sps[:, 0])
  return n

def to_cartesian_(spherical_p):
  return to_cartesian(spherical_p[0], spherical_p[1])

def to_cartesian(theta, phi):
  return np.array([[
    np.sin(theta) * np.cos(phi),
    np.sin(theta) * np.sin(phi),
    np.cos(theta)
  ]])

def to_spherical(ns):
  theta = np.arccos(ns[:, 2])
  phi = np.arctan2(ns[:, 1], ns[:, 0])
  return theta, phi

def spherical_cos_angle(spherical_p1, spherical_p2):
  p1 = to_cartesian_(spherical_p1)
  p2 = to_cartesian_(spherical_p2)

  return p1.dot(p2.T)

def spherical_angle(spherical_p1, spherical_p2):
  cos = spherical_cos_angle(spherical_p1, spherical_p2)
  cos = np.max([-1.0, cos])
  cos = np.min([1.0, cos])
  return np.arccos(cos)