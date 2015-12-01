import numpy as np

from geometry import *

def particles(n_particles, detector_layers,
              theta_limits, phi_limits,
              trace_probability,
              trace_noise,
              detector_noise_rate):
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

  return np.vstack(intercepts), particles_params, particle_ns

