import numpy as np

def particles(n_particles, detector_layers,
              theta_limits, phi_limits,
              trace_probability,
              noise_eps):
  particles_params = np.random.uniform(0.0, 1.0, size=(n_particles, 2))

  particles_params[:, 0] = particles_params[:, 0] * (theta_limits[1] - theta_limits[0]) + theta_limits[0]
  particles_params[:, 1] = particles_params[:, 1] * (phi_limits[1] - phi_limits[0]) + phi_limits[0]

  particle_ns = np.ndarray(shape=(n_particles, 3), dtype="double")

  particle_ns[:, 0] = np.sin(particles_params[:, 0]) * np.cos(particles_params[:, 1])
  particle_ns[:, 1] = np.sin(particles_params[:, 0]) * np.sin(particles_params[:, 1])
  particle_ns[:, 2] = np.cos(particles_params[:, 0])

  n_layers = detector_layers.shape[0]
  true_intercepts = np.ndarray(shape=(n_particles, n_layers, 3))

  intercepts = list()

  for p in range(n_particles):
    true_intercepts[p, :, 2] = detector_layers
    t = detector_layers / particle_ns[p, 2]
    true_intercepts[p, :, 0] = t * particle_ns[p, 0]
    true_intercepts[p, :, 1] = t * particle_ns[p, 1]
    noise = np.random.standard_normal(size=(n_layers, 2)) * noise_eps
    true_intercepts[p, :, 0:2] += noise

    detection = np.random.binomial(1, trace_probability, size=n_layers)
    readings = true_intercepts[p, detection == 1, :]
    intercepts.append(readings)

  return  np.vstack(intercepts), particles_params, particle_ns

