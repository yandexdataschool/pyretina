from pyretina import simulation
from pyretina import *
import numpy as np

bins = 100

theta_limits = [0.0, 0.1]
theta_step = (theta_limits[1] - theta_limits[0]) / bins

phi_limits = [-0.1, 0.1]
phi_step = (phi_limits[1] - phi_limits[0]) / bins

generation_params = {
  "n_particles": 50,
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

re = simulation.linear(**generation_params)

set_event(re.event)
set_sigma(0.05)


from pyretina.optimize import multi_start

predicted, traces = multi_start(re)

#from pyretina.plot3d import plot_retina_results

#plot_retina_results(predicted, re, 1.0e-2).show()