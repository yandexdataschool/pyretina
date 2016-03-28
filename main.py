import numpy as np

if __name__ == "__main__":
  bins = 100

  theta_limits = [0.0, 0.025]
  theta_step = (theta_limits[1] - theta_limits[0]) / bins

  phi_limits = [-0.025, 0.025]
  phi_step = (phi_limits[1] - phi_limits[0]) / bins

  generation_params = {
    "n_particles": 100,
    "detector_layers": np.arange(0, 20) + 5,

    "theta_limits": [theta_limits[0] + 2 * theta_step, theta_limits[1] - 2 * theta_step],
    "phi_limits": [phi_limits[0] + 2 * phi_step, phi_limits[1] - 2 * phi_step],

    "trace_probability": 0.75,
    "trace_noise": 0.025,
    "detector_noise_rate": 100.0,

    "sigma": 0.01,

    "theta_bins": bins,
    "phi_bins": bins
  }

  load_params = {
    'sigma' : 0.5,
    "theta_limits": [-np.pi / 8, np.pi / 8],
    "theta_bins": 250,
    "phi_limits": [-np.pi / 8, np.pi / 8],
    "phi_bins": 250
  }

  experiments = 1
  from pyretina.optimize import multi_start

  solver = "Newton-CG"
  solver_options = {
    "xtol" : 1.0e-4
  }

  from pyretina.plot import *
  from pyretina.evaluate import *
  from pyretina.io import simulation
  from pyretina.io import from_csv

  precision = np.zeros(shape=experiments)
  recall = np.zeros(shape=experiments)

  precision_grid = np.zeros(shape=experiments)
  recall_grid = np.zeros(shape=experiments)

  for i in range(experiments):
    #re = from_csv.load_dataset("data/MC/00163875_0143139193",
    #                           **load_params)

    re = simulation.linear(**generation_params)
    print "Starting multistart"
    print re.tracks

    predicted, traces = multi_start(re, max_evaluations=1000, method = solver, solver_options = solver_options)

    plot_retina_results(predicted, re, 1.0e-2,
                        search_traces=traces, against='grid_search').savefig("events_img/multistart_%d.png" % i)

    bm = binary_metrics(predicted, re, against='true')[0]
    precision[i] = bm['precision']
    recall[i] = bm['recall']

    predicted_grid = grid_search(re)
    plot_retina_results(predicted_grid, re, 1.0e-2,
                        search_traces=None, against='true').savefig("events_img/grid_search_%d.png" % i, dpi=120)

    bm_grid = binary_metrics(predicted_grid, re, against='true')[0]

    precision_grid[i] = bm_grid['precision']
    recall_grid[i] = bm_grid['recall']

    X, Y, Z = re.get_grid()
    print X
    print Y
    print Z

    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    from matplotlib import cm

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
    #cset = ax.contourf(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
    #cset = ax.contourf(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
    #cset = ax.contourf(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)

    ax.set_xlabel('$\\theta$')
    #ax.set_xlim(-40, 40)
    ax.set_ylabel('$\\phi$')
    #ax.set_ylim(-40, 40)
    ax.set_zlabel('Retina response')
    #ax.set_zlim(-100, 100)

    plt.savefig("RR.png", dpi=320)

    print precision[i], recall[i]
    print precision_grid[i], recall_grid[i]

  print "Precision", precision.mean(), precision.std()
  print "Recall", recall.mean(), recall.std()

  print "Precision Grid", precision_grid.mean(), precision_grid.std()
  print "Recall Grid", recall_grid.mean(), recall_grid.std()
