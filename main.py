import sys
import cPickle as pickle

import numpy as np
import matplotlib.pyplot as plt

from pyretina.evaluate import precision_recall, binary_metrics
from pyretina.plot import plot_event_plotly
from pyretina.retina import grid, grad_grid, set_event, set_z0, set_sigma
from pyretina.geometry import to_spherical
from pyretina.optimize import swarm_search_step, swarm_search, hits_prior, pseudo_rapidity_prior

import matplotlib
matplotlib.rcParams.update({'font.size': 18})

def evaluate(event, sigma_from, sigma_to, C=3, plotting = False):
  set_event(event.hits)
  set_z0(event.z0)

  steps = 3
  eps = 5.0e-3

  grid_N = int(np.pi / 3 / eps )

  # Hessian matrix computation cost ~2-3 times more than response computation
  swarm_size = min(int(1.0 * grid_N * grid_N / steps / 3 / C), event.hits.shape[0])

  sigma_strategy = np.linspace(sigma_from, sigma_to, num=steps)
  #sigma_strategy = 2.0
  sol = swarm_search_step(event.hits, z0 = event.z0,
                     swarm_size=swarm_size, swarm_lifetime=steps,
                     optimizer='Newton-CG',
                     optimizer_options = {
                     },
                     initial_guess=hits_prior,
                     sigma_strategy=sigma_strategy)
  if plotting:
    plt.figure()
    ts, ps, g = grid(event.hits, 2.0, event.z0, [-np.pi / 6, np.pi / 6], [-np.pi / 6, np.pi / 6], bins_theta = grid_N * 2, bins_phi = grid_N * 2)
    plt.contourf(ts, ps, g, 40, cmap=plt.cm.Reds)
    plt.xlabel(r"$\phi$")
    plt.ylabel(r"$\theta$")
    plt.colorbar()

    tracks = to_spherical(event.tracks)
    plt.scatter(tracks[:, 0], tracks[:, 1], marker = 'o', color='blue', label = 'Tracks')

    plt.scatter(sol.maxima[sol.scores >= 1.5, 0], sol.maxima[sol.scores >= 1.5, 1], marker="x", label = 'Reconstructed')
    history = sol.traces

    j = to_spherical(event.hits - np.array([0.0, 0.0, event.z0]))
    plt.scatter(j[:, 0], j[:, 1], marker="+", color="red", label = 'Hits')

    for i in range(len(history)):
      if sol.scores[i] < 1.5:
        continue

      xs = history[i][:, 0]
      ys = history[i][:, 1]
      plt.plot(xs, ys, color="blue", linewidth=4, alpha=0.5)

    plt.legend()
    plt.title('Optimization traces')
    plt.show()

  return binary_metrics(sol.maxima[sol.scores >= 1.5], event, max_angle=eps)

if __name__ == "__main__":
  import os

  files = [ item for item in os.listdir('./data/') if item.endswith('.pickled') ]
  per_file = 10
  N = np.zeros(shape=(len(files), per_file))
  recall = np.zeros(shape=(len(files), per_file))
  precision = np.zeros(shape=(len(files), per_file))

  for fi, f in enumerate(files):
    print f
    input_path = './data/' + f
    with open(input_path, "r") as g:
      events = pickle.load(g)

    for i in xrange(per_file):
      event = events[i]

      res = evaluate(event, sigma_from=0.3, sigma_to=0.05, C = 10, plotting=False)[0]

      N[fi, i] = event.tracks.shape[0]
      precision[fi, i] = res['precision']
      recall[fi, i] = res['recall']

    print 'N:', np.mean(N[fi, :]), 'precision:', np.mean(precision[fi, :]), 'recall:', np.mean(recall[fi, :])

  np.save('precisions', precision)
  np.save('N', N)
  np.save('recalls', recall)



