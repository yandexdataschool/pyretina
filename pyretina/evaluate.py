import numpy as np

from sklearn.neighbors import DistanceMetric

from geometry import *

def mappings(predicted_x, predicted_y, predicted_primary, reference_z,
             event, max_angle=1.0e-2, max_primary=5.0):
  predicted = np.ndarray(shape=(predicted_x.shape[0], 3))
  predicted[:, 0] = predicted_x
  predicted[:, 1] = predicted_y
  predicted[:, 2] = reference_z

  predicted = predicted / np.sqrt(np.sum(predicted ** 2, axis=1))[:, None]

  m = DistanceMetric.get_metric(metric='pyfunc', func=spherical_angle)
  d = m.pairwise(predicted, event.tracks)

  mapping_matrix = (d <= max_angle) & (np.abs(event.z0 - predicted_primary) < max_primary)[:, None]

  test_mapping = np.sum(mapping_matrix, axis=0)
  predicted_mapping = np.sum(mapping_matrix, axis=1)
  return mapping_matrix, predicted_mapping, test_mapping

  # Each true sample maps to closest
  test_mapping = np.zeros(shape=(event.tracks.shape[0], ), dtype=int)
  predicted_mapping = np.zeros(shape=(predicted.shape[0], ), dtype=int)

  test_distance = -np.ones(shape=(event.tracks.shape[0]), dtype='float32')

  for i in xrange(event.tracks.shape[0]):
    min_d = np.min(d[:, i])(d[:, i] < max_angle) & ()
    test_mapping[i] = 1 if np.any(mask) else 0

  for i in xrange(predicted.shape[0]):
    mask = np.any(d[i, :] < max_angle) and np.abs(predicted_primary[i] - event.z0) > max_primary
    predicted_mapping[i] = 1 if mask else 0

  return predicted_mapping, test_mapping

def bin_metrics(*args, **kwargs):
  matrix, predicted_mapping, test_mapping = mappings(*args, **kwargs)

  fn = float(np.sum(test_mapping == 0))
  fp = float(np.sum(predicted_mapping == 0))

  tp = float(np.sum(test_mapping == 1))

  return tp, fp, fn, matrix, predicted_mapping, test_mapping

def binary_metrics(*args, **kwargs):
  tp, fp, fn, matrix, predicted_mapping, test_mapping = bin_metrics(*args, **kwargs)

  metrics = dict()

  metrics['fn'] = fn
  metrics['fp'] = fp
  metrics['tp'] = tp

  metrics['precision'] = tp / (fp + tp)
  metrics['recall'] = tp / (tp + fn)

  return metrics, matrix, predicted_mapping, test_mapping

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
  predicted_mapping, _ = mappings(predicted, test, max_angle)

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