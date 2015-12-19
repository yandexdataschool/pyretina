import numpy as np

from sklearn.neighbors import BallTree, DistanceMetric

from geometry import *

def mean_distance_to_closest(predicted, event):
  angle = DistanceMetric.get_metric('pyfunc', func=spherical_angle)
  nn = BallTree(event.tracks, leaf_size=5, metric=angle)

  return np.sum([ nn.query(predicted[i, :], k=1) for i in xrange(predicted.shape[0]) ]) / event.tracks.shape[0]

def precision_recall(retina_events, max_angle_scale = 4, **kwargs):
  from optimize import maxima

  n = len(retina_events)

  precision = np.zeros(n)
  recall = np.zeros(n)

  for i in xrange(n):
    re = retina_events[i]

    thetas, phis, response = re.get_grid()
    m = np.where(maxima(response))
    predicted = np.vstack([ thetas[m], phis[m] ]).T

    max_angle = max_angle_scale * spherical_angle(np.array([0.0, 0.0]), np.array([
      re.theta_step, re.phi_step
    ]))

    m, predicted_mapping, test_mapping = binary_metrics(predicted, re, max_angle=max_angle)

    tp = m['tp']
    fp = m['fp']
    fn = m['fn']

    precision[i] = tp / (tp + fp)
    recall[i] = tp / (tp + fn)

  return precision, recall

def binary_metrics(predicted, event, max_angle=1.0e-2):
  angle = DistanceMetric.get_metric('pyfunc', func=spherical_angle)

  test = event.tracks

  d = angle.pairwise(predicted, test)

  # Each true sample maps to closest
  test_mapping = np.zeros(shape=(test.shape[0], ), dtype=int)
  predicted_mapping = np.zeros(shape=(predicted.shape[0], ), dtype=int)

  for i in xrange(test.shape[0]):
    test_mapping[i] = 1 if np.any(d[:, i] < max_angle) else 0

  for i in xrange(predicted.shape[0]):
    predicted_mapping[i] = 1 if np.any(d[i, :] < max_angle) else 0

  metric_dict = dict()

  metric_dict['fn'] = float(np.sum(test_mapping == 0))
  metric_dict['fp'] = float(np.sum(predicted_mapping == 0))

  metric_dict['tp'] = float(np.sum(test_mapping == 1))

  return metric_dict, predicted_mapping, test_mapping