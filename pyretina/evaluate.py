import numpy as np

from sklearn.neighbors import BallTree, DistanceMetric

from geometry import *
from optimize import grid_search

def mean_distance_to_closest(predicted, event):
  angle = DistanceMetric.get_metric('pyfunc', func=spherical_angle)
  nn = BallTree(event.tracks, leaf_size=5, metric=angle)

  return np.sum([ nn.query(predicted[i, :], k=1) for i in xrange(predicted.shape[0]) ]) / event.tracks.shape[0]

def __mappings(predicted, test, max_angle = 1.0-2):
  angle = DistanceMetric.get_metric('pyfunc', func=spherical_angle)
  d = angle.pairwise(predicted, test)

  # Each true sample maps to closest
  test_mapping = np.zeros(shape=(test.shape[0], ), dtype=int)
  predicted_mapping = np.zeros(shape=(predicted.shape[0], ), dtype=int)

  for i in xrange(test.shape[0]):
    test_mapping[i] = 1 if np.any(d[:, i] < max_angle) else 0

  for i in xrange(predicted.shape[0]):
    predicted_mapping[i] = 1 if np.any(d[i, :] < max_angle) else 0

  return predicted_mapping, test_mapping

def __bin_metrics(predicted, test, max_angle = 1.0e-2):
  predicted_mapping, test_mapping = __mappings(predicted, test, max_angle)

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

def __max_score_mapping(rr, predicted, test, max_angle = 1.0-2):
  angle = DistanceMetric.get_metric('pyfunc', func=spherical_angle)
  d = angle.pairwise(predicted, test)

  # Each true sample maps to closest
  test_mapping = np.zeros(shape=(test.shape[0], ), dtype=float)

  for i in xrange(test.shape[0]):
    if np.any(d[:, i] < max_angle):
      close_predictions = d[:, i] < max_angle
      scores = [ rr(p) for p in predicted[close_predictions, :] ]
      test_mapping[i] = np.max(scores)

  return test_mapping

def precision_recall(predicted, event, against = 'true', max_angle = 1.0e-2):
  rr, jac, hess = event.get_response()
  test = __against(event, against)

  scores = np.array([ rr(p) for p in predicted ])

  test_score_mapping = __max_score_mapping(rr, predicted, test, max_angle)
  predicted_mapping, _ = __mappings(predicted, test, max_angle)

  tp = np.zeros(shape=scores.shape[0])
  fp = np.zeros(shape=scores.shape[0])
  fn = np.zeros(shape=scores.shape[0])

  for i, s in enumerate(np.sort(scores[:-1])):
    tp[i] = np.sum(test_score_mapping > s)
    fp[i] = np.sum(predicted_mapping[scores > s] == 0)
    fn[i] = np.sum(test_score_mapping <= s)

  precision = tp / (tp + fp)
  recall = tp / test.shape[0]

  print precision
  print recall

  return np.sort(scores), precision, recall


