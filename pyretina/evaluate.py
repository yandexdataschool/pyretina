import numpy as np

from sklearn.neighbors import BallTree, DistanceMetric

from geometry import *

def mean_distance_to_closest(predicted, test):
  angle = DistanceMetric.get_metric('pyfunc', func=spherical_angle)
  nn = BallTree(test, leaf_size=5, metric=angle)

  return np.sum([ nn.query(predicted[i, :], k=1) for i in xrange(predicted.shape[0]) ]) / test.shape[0]


def binary_metrics(predicted, test, max_angle=1.0e-2):
  angle = DistanceMetric.get_metric('pyfunc', func=spherical_angle)
  D = angle.pairwise(predicted, test)

  # Each true sample maps to closest
  test_mapping = np.zeros(shape=(test.shape[0], ), dtype=int)
  predicted_mapping = np.zeros(shape=(predicted.shape[0], ), dtype=int)

  for i in xrange(test.shape[0]):
    test_mapping[i] = 1 if np.any(D[:, i] < max_angle) else 0

  for i in xrange(predicted.shape[0]):
    predicted_mapping[i] = 1 if np.any(D[i, :] < max_angle) else 0

  metric_dict = dict()

  metric_dict['fn'] = float(np.sum(test_mapping == 0))
  metric_dict['fp'] = float(np.sum(predicted_mapping == 0))

  metric_dict['tp'] = float(np.sum(test_mapping == 1))

  return metric_dict, predicted_mapping, test_mapping