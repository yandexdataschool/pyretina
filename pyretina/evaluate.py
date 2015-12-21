import numpy as np

from sklearn.neighbors import BallTree, DistanceMetric

from geometry import *
from optimize import grid_search

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

def __bin_metrics(predicted, test, max_angle = 1.0e-2):
  angle = DistanceMetric.get_metric('pyfunc', func=spherical_angle)
  d = angle.pairwise(predicted, test)

  # Each true sample maps to closest
  test_mapping = np.zeros(shape=(test.shape[0], ), dtype=int)
  predicted_mapping = np.zeros(shape=(predicted.shape[0], ), dtype=int)

  for i in xrange(test.shape[0]):
    test_mapping[i] = 1 if np.any(d[:, i] < max_angle) else 0

  for i in xrange(predicted.shape[0]):
    predicted_mapping[i] = 1 if np.any(d[i, :] < max_angle) else 0

  fn = float(np.sum(test_mapping == 0))
  fp = float(np.sum(predicted_mapping == 0))

  tp = float(np.sum(test_mapping == 1))

  return tp, fp, fn, predicted_mapping, test_mapping

def __against(event, against = 'true'):
  if against == 'true':
    test = event.tracks
  elif against == 'grid_search':
    test = grid_search(event)
  else:
    test = against

  return test

def binary_metrics(predicted, event, against = 'true', max_angle = 1.0e-2):
  metric_dict = dict()
  test = __against(event, against)
  tp, fp, fn, predicted_mapping, test_mapping = __bin_metrics(predicted, test, max_angle)

  metric_dict['fn'] = fn
  metric_dict['fp'] = fp
  metric_dict['tp'] = tp

  metric_dict['precision'] = tp / (fp + tp)
  metric_dict['recall'] = tp / (tp + fn)

  return metric_dict, test, predicted_mapping, test_mapping