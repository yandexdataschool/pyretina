from pyretina import simulation

from pyretina import *

import numpy as np

from pyretina.plot3d import *

import matplotlib.pyplot as plt
import plotly.plotly as py

from pyretina.evaluate import binary_metrics
from pyretina.optimize.grid_search import maxima

bins = 100

theta_limits = [0.0, 0.1]
theta_step = (theta_limits[0] - theta_limits[1]) / bins
theta_generation_limits = [theta_limits[0] + 2 * theta_step, theta_limits[1] - 2 * theta_step]
phi_limits = [-np.pi, np.pi]

particles_n = 20


def precision_recall(n, **kwargs):
  precision = np.zeros(n)
  recall = np.zeros(n)

  for i in xrange(n):
    event, params, ns = simulation.particles(particles_n, np.arange(0, 20), theta_generation_limits, **kwargs)

    thetas, phis, response = retina_grid(event, theta_limits, bins, phi_limits, bins, sigma=0.01)
    m = np.where(maxima(response))
    predicted = np.vstack([ thetas[m], phis[m] ]).T

    max_angle = 8 * spherical_angle(np.array([0.0, 0.0]), np.array([
      (theta_limits[1] - theta_limits[0]) / bins, (phi_limits[1] - phi_limits[0]) / bins
    ]))

    m, predicted_mapping, test_mapping = binary_metrics(predicted, params, max_angle=max_angle)

    param_str = ",".join([ "%s=%s" % (str(k), str(v)) for k, v in kwargs.items() ])
    plot_retina_results(predicted, params, thetas, phis, response, max_angle)\
      .savefig("events_img/event-%d(%s).png" % (i, param_str))
    plt.close()

    tp = m['tp']
    fp = m['fp']
    fn = m['fn']

    print m

    precision[i] = tp / (tp + fp)
    recall[i] = tp / (tp + fn)

  return precision, recall

def main():
  n_simulations = 20
  noise_bins = 25
  noise = np.linspace(0.0, 0.05, num=noise_bins)

  precision = np.zeros(shape=(noise_bins, n_simulations))
  recall = np.zeros(shape=(noise_bins, n_simulations))

  for i in xrange(noise.shape[0]):
    print "noise rate", noise[i]
    p, r = precision_recall(n_simulations, trace_probability=0.75, trace_noise=noise[i], detector_noise_rate=0.0)
    precision[i, :] = p
    recall[i, :] = r

  plt.figure()
  plt.scatter(precision.ravel(), recall.ravel())
  plt.xlabel("precision")
  plt.ylabel("recall")
  plt.title("Retina grid precision-recall (track noise varying)")
  plt.xlim([0, 1])
  plt.ylim([0, 1])
  plt.savefig("precision-recall.png")

  plt.figure()
  plt.boxplot(precision.T)
  plt.ylim([0, 1.1])
  plt.xlabel("track noise std")
  plt.ylabel("precision (truly recognized among recognized)")
  plt.title("Retina grid (%d x %d), particles = %d" % (bins, bins, particles_n))
  plt.savefig("precision.png")

  plt.figure()
  plt.boxplot(recall.T)
  plt.ylim([0, 1.1])
  plt.xlabel("track noise std")
  plt.ylabel("recall (truly recognized among true tracks)")
  plt.title("Retina grid (%d x %d), particles = %d" % (bins, bins, particles_n))
  plt.savefig("recall.png")

# main()

def plot_event(event, track_ns=None):
  from mpl_toolkits.mplot3d import Axes3D

  fig = plt.figure(num=None, figsize=(16, 8), dpi=400)
  ax = fig.add_subplot(111, projection='3d')

  ax.scatter(event[:, 2], event[:, 0], event[:, 1], color="b", alpha=0.25, s = 1.0)

  if track_ns is not None:
      for i in range(track_ns.shape[0]):
          (nx, ny, nz) = track_ns[i, :]
          ax.plot([0, nz], [0, nx], [0, ny], c = "r", linewidth=1.0, alpha=0.25)

  ax.set_xlabel("z")
  ax.set_ylabel("x")
  ax.set_zlabel("y")
  #ax.set_ylim([-150, 150])
  #ax.set_zlim([-150, 150])

import plotly.plotly as py
from plotly.graph_objs import *
import plotly.tools as tls


def main2():

  py.sign_in('carlos-castaneda', 'xno9347pjk')

  for i in range(10):
    event, params, ns = simulation.particles(particles_n, np.arange(0, 18) + 2, theta_generation_limits,
                                             trace_probability=0.75, trace_noise=0.01, detector_noise_rate=0.0)

    #from pyretina.plot3d import plot_event

    thetas, phis, response = retina_grid(event, theta_limits, bins, phi_limits, bins, sigma=0.01)
    m = np.where(maxima(response))
    predicted = np.vstack([ thetas[m], phis[m] ]).T

    max_angle = 4 * spherical_angle(np.array([0.0, 0.0]), np.array([
      (theta_limits[1] - theta_limits[0]) / bins, (phi_limits[1] - phi_limits[0]) / bins
    ]))

    _, predicted_mapping, test_mapping = binary_metrics(predicted, params, max_angle=max_angle)


    print np.mean(predicted[predicted_mapping == 0], axis=0)

    recognized = predicted_mapping == 1
    test_recognized = test_mapping == 1
    ghost = predicted_mapping == 0
    unrecognized = test_mapping == 0

    print m

    fig = plot_event_plotly(event, ns * 20, 20 * to_cartesian_array(predicted[ghost]))
    py.plot(fig)

    plt.figure()
    plt.scatter(thetas[m[0][ghost], m[1][ghost]], response[m[0][ghost], m[1][ghost]], marker="x", color="red")
    plt.scatter(thetas[m[0][recognized], m[1][recognized]], response[m[0][recognized], m[1][recognized]], marker="+", color="green")
    plt.plot()

    plt.figure()
    wsig = response[m[0][recognized], m[1][recognized]]
    plt.hist(thetas[m[0][recognized], m[1][recognized]],weights= wsig/np.sum(wsig), color="green")
    plt.plot()
    plt.figure()
    wbkg = response[m[0][ghost], m[1][ghost]]
    plt.hist(thetas[m[0][ghost], m[1][ghost]],weights= wbkg /np.sum(wbkg), color="red")
    plt.plot()

    plot_event(event)

    plot_retina_results(predicted, params, thetas, phis, response, max_angle).show()

main2()