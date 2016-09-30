from evaluate import binary_metrics, mappings
from geometry import to_reference_plane
from mc.pseudo_velo_mc import monte_carlo
from pyretina import GridOptimizer, GD, GDTracker, GridSearchTracker, NNTracker
from pyretina.mc import mc_stream

import numpy as np

import matplotlib.pyplot as plt

import theano
import theano.tensor as T
import time

from retina.reference_plane.reference_plane import ReferencePlaneRetinaModel

def hits_prior(hits, size):
  idx = np.random.choice(hits.shape[0], size=size, replace=False)
  return hits[idx, :]

def seeder_model(reference_z=100.0, sigma0=1.0):
  def model(retina_model):
    event_vars = retina_model.get_event_variables()

    def seeder(size):
      ex, ey, ez = [ var.get_value() for var in event_vars ]
      n_hits = ex.shape[0]
      size = np.min([size, n_hits])
      idx = np.random.choice(n_hits, size=size, replace=False)

      ex = ex[idx] * reference_z / ez[idx]
      ey = ey[idx] * reference_z / ez[idx]
      primary_vertex = np.zeros_like(ex)
      sigma = np.ones_like(ex) * sigma0

      return [ex, ey, primary_vertex, sigma]
    return seeder
  return model

def main():
  min_pseudorapidity = 1

  max_angle = 2 * np.arctan(np.exp(-min_pseudorapidity))
  rz = 100.0
  resolution = 1.0e-3

  angles = np.linspace(-max_angle, +max_angle, num=int(max_angle / resolution), dtype='float32')
  sins = np.sin(angles) * rz

  mxs = sins
  mys = sins
  mzs = np.array([0.0], dtype='float32')
  sigmas = np.array([1.0], dtype='float32')

  n_seeds = 200

  event_stream = mc_stream(n_events=100000)
  train_events = monte_carlo(50)

  ms = []
  seeder_ms = []

  tracker = NNTracker(
    seeder_model(rz),
    normalization_coefs=[0.03, 0.03, 0.2, 1.0],
    alpha = 1.0e-2,
    n_seeds=n_seeds, n_steps=3,
    reference_z=rz, threshold=1.5
  )

  grid_opt = GridOptimizer(tracker.retina_model, [mxs, mys, mzs, sigmas])

  print np.mean(tracker.fit(train_events, 1.0, 1.0e-3))

  event = event_stream.next()

  px, py, pz, _ = tracker.predict(event)
  traces = np.array(tracker.optimizer.traces)

  grid_opt.fill_grid_value(-2, event.z0)
  grid_opt.maxima()
  r = grid_opt.response_grid

  md, matrix, pmapping, tmapping = binary_metrics(
    px, py, pz, rz, event, max_angle=5.0e-3, max_primary=5.0
  )

  sx, sy, _, _ = tracker.optimizer.seeds
  smd = binary_metrics(
    sx, sy, np.zeros_like(sx), rz, event, max_angle=5.0e-3, max_primary=5.0
  )[0]

  ms.append(md)
  seeder_ms.append(smd)

  print 'mean recall:', np.mean([ md['recall'] for md in ms ])
  print 'mean recall for seeds:', np.mean([md['recall'] for md in seeder_ms])

  print md

  tps = tmapping > 0
  tps_ = pmapping > 0

  fns = tmapping == 0

  fps = pmapping == 0

  plt.figure(figsize=(10, 8))
  plt.contourf(
    mxs, mys, r[:, :, 0, 0].T,
    cmap=plt.cm.viridis, levels=np.linspace(0.0, max(np.max(r), 1.0), num=50)
  )
  plt.colorbar()
  ex, ey, ez = to_reference_plane(event, reference_z=rz)

  plt.scatter(ex[tps], ey[tps], marker='x', color='green', label='recovered (%d)' % md['tp'])
  plt.scatter(ex[fns], ey[fns], marker='x', color='red', label='missed (%d)' % md['fn'])
  plt.scatter(px[tps_], py[tps_], marker='o', color='green', label='hit (%d)' % md['tp'])
  plt.scatter(px[fps], py[fps], marker='o', color='red', label='ghost tracks (%d)' % md['fp'])

  for trace_i in xrange(traces.shape[-1]):
    plt.plot(traces[:, 0, trace_i], traces[:, 1, trace_i], color='blue')

  plt.scatter(sx, sy, marker='o', color='blue', label='seeds (%d)' % n_seeds)

  iss, jss = np.where(matrix)

  for i, j in zip(iss, jss):
    plt.plot([ex[j], px[i]], [ey[j], py[i]], '--', color='blue')

  plt.legend()

  plt.show()

if __name__ == '__main__':
  main()