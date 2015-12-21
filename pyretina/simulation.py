import numpy as np

from geometry import *
from retina_event import RetinaEvent

__PICKLE_PROTOCOL = 2

class __Gen:
  """
  Since pickle isn't able to serialize lambdas...
  let's define our own function...
  """
  def __init__(self, args):
    self.args = args

  def __call__(self, *_):
    event = linear(**self.args)
    event.make_grid()
    return event

def make_dataset(size, simulation_args, outfile = "simulation-data.pkl", n_jobs = 4, strict = False):
  from joblib import Parallel, delayed, dump
  import os

  if os.path.exists(outfile) and not strict:
    print "File exists, skipping generation"
    return load_dataset(outfile)
  elif strict:
    raise Exception("File already exists, generation will override it, but strict flag is set.")


  gen = __Gen(simulation_args)

  events = Parallel(n_jobs = n_jobs)(delayed(gen)(i) for i in xrange(size))

  print type(events)

  dump((simulation_args, events), outfile, compress=True)

  return simulation_args, events

def load_dataset(infile = "simulation-data.pkl"):
  from joblib import load

  (simulation_args, events) = load(infile)

  return simulation_args, events

def linear(n_particles, detector_layers,
           theta_limits, phi_limits,
           trace_probability,
           trace_noise,
           detector_noise_rate,
           sigma=0.05,
           theta_bins = 250,
           phi_bins = 250):
  particles_params = np.random.uniform(0.0, 1.0, size=(n_particles, 2))

  particles_params[:, 0] = particles_params[:, 0] * (theta_limits[1] - theta_limits[0]) + theta_limits[0]
  particles_params[:, 1] = particles_params[:, 1] * (phi_limits[1] - phi_limits[0]) + phi_limits[0]

  particle_ns = to_cartesian(particles_params)

  n_layers = detector_layers.shape[0]
  true_intercepts = np.ndarray(shape=(n_particles, n_layers, 3))

  intercepts = list()

  for p in range(n_particles):
    true_intercepts[p, :, 2] = detector_layers
    t = detector_layers / particle_ns[p, 2]
    true_intercepts[p, :, 0] = t * particle_ns[p, 0]
    true_intercepts[p, :, 1] = t * particle_ns[p, 1]
    noise = np.random.standard_normal(size=(n_layers, 2)) * trace_noise
    true_intercepts[p, :, 0:2] += noise

    detection = np.random.binomial(1, trace_probability, size=n_layers)
    readings = true_intercepts[p, detection == 1, :]
    intercepts.append(readings)

  ## Apply detector noise

  max_layer = np.max(detector_layers)
  detector_xy_min = max_layer / np.cos(theta_limits[0])
  detector_xy_max = max_layer / np.cos(theta_limits[1])

  detector_noise_n = np.random.poisson(detector_noise_rate)
  for _ in range(detector_noise_n):
    noisy_layer = np.random.random_integers(0, n_layers - 1)
    noise = np.random.uniform(size=3)
    noise[0:2] = noise[0:2] * (detector_xy_max - detector_xy_min) + detector_xy_min
    noise[2] = detector_layers[noisy_layer]
    intercepts.append(noise)

  retina_event = RetinaEvent(np.vstack(intercepts), particles_params, sigma,
                             theta_limits, theta_bins, phi_limits, phi_bins)

  return retina_event