import numpy as np

if __name__ == "__main__":
  bins = 250

  theta_limits = [0.0, 0.1]
  theta_step = (theta_limits[1] - theta_limits[0]) / bins

  phi_limits = [-0.1, 0.1]
  phi_step = (phi_limits[1] - phi_limits[0]) / bins

  generation_params = {
    "n_particles": 100,
    "detector_layers": np.arange(0, 20) + 5,

    "theta_limits": [theta_limits[0] + 2 * theta_step, theta_limits[1] - 2 * theta_step],
    "phi_limits": [phi_limits[0] + 2 * phi_step, phi_limits[1] - 2 * phi_step],

    "trace_probability": 0.75,
    "trace_noise": 0.025,
    "detector_noise_rate": 1000.0,

    "sigma": 0.05,

    "theta_bins": bins,
    "phi_bins": bins
  }

  load_params = {
    'sigma' : 1.0,
    "theta_limits": [-np.pi / 8, np.pi / 8],
    "theta_bins": 250,
    "phi_limits": [-np.pi / 8, np.pi / 8],
    "phi_bins": 250
  }

  #args, dataset = simulation.make_dataset(1000, generation_params, n_jobs = 8)

  experiments = 100
  from pyretina.optimize import multi_start

  solver = "Newton-CG"
  solver_options = {
    "xtol" : 1.0e-4
  }

  from pyretina.plot import *
  from pyretina.evaluate import *
  from pyretina.io import from_csv

  precision = np.zeros(shape=experiments)
  recall = np.zeros(shape=experiments)

  precision_grid = np.zeros(shape=experiments)
  recall_grid = np.zeros(shape=experiments)

  for i in range(experiments):
    re = from_csv.load_dataset("data/event_hits/00163875_0143139193.tracks.csv",
                               **load_params)

    predicted, traces = multi_start(re, max_evaluations=10000, method = solver, solver_options = solver_options)

    plot_retina_results(predicted, re, 1.0e-2,
                        search_traces=traces, against='grid_search').savefig("events_img/multistart_%d.png" % i, dpi=320)

    bm = binary_metrics(predicted, re, against='true')[0]
    precision[i] = bm['precision']
    recall[i] = bm['recall']

    predicted_grid = grid_search(re)
    plot_retina_results(predicted_grid, re, 1.0e-2,
                        search_traces=None, against='true').savefig("events_img/grid_search_%d.png" % i, dpi=320)

    bm_grid = binary_metrics(predicted_grid, re, against='true')[0]

    precision_grid[i] = bm_grid['precision']
    recall_grid[i] = bm_grid['recall']

    print precision[i], recall[i]
    print precision_grid[i], recall_grid[i]

  print "Precision", precision.mean(), precision.std()
  print "Recall", recall.mean(), recall.std()

  print "Precision Grid", precision_grid.mean(), precision_grid.std()
  print "Recall Grid", recall_grid.mean(), recall_grid.std()
