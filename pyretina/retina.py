#!/usr/bin/python

import numpy as np

class Retina:
  event = None
  tracks = None
  sigma = None

  theta_limits = None
  theta_bins = 200

  phi_limits = None
  phi_bins = 200

  grid = None
  thetas = None
  phis = None

  optimization_trace = None

  def make_grid(self):
    from retina_response import retina_grid

    self.thetas, self.phis, self.grid =\
      retina_grid(self.event,
                  self.theta_limits, self.theta_bins,
                  self.phi_limits, self.phi_bins,
                  self.sigma)

    return self.grid

  def get_grid(self):
    if self.grid is None:
      return self.make_grid()

    return self.grid

  def __init__(self, event, tracks, sigma, theta_limits, theta_bins, phi_limits, phi_bins):
    self.event = event
    self.tracks = tracks

    self.theta_limits = theta_limits
    self.phi_limits = phi_limits
    self.bins_theta = theta_bins
    self.bins_phi = phi_bins

    self.sigma = sigma

  def response(self):
    from retina_response import retina_response_n
    from geometry import to_cartesian_

    return lambda p: retina_response_n(self.event, to_cartesian_(p), self.sigma)

  n_particles = 50
  detector_layers = np.arange(0, 20) + 2
  trace_probability = 0.75

  trace_noise = 0.01
  detector_noise = n_particles * detector_layers.shape[0]

  @staticmethod
  def generate(n_particles=None, detector_layers=None,
               trace_probability=0.75, trace_noise=0.01,
               detector_noise_rate=1000.0):
    from simulation import particles

    n_particles = n_particles or Retina.n_particles
    detector_layers = detector_layers or Retina.detector_layers
    trace_probability = trace_probability or Retina.trace_probability
    trace_noise = trace_noise or Retina.trace_noise
    detector_noise_rate = detector_noise_rate or Retina.detector_noise

    dtheta = (Retina.theta_limits[1] - Retina.theta_limits[0]) / Retina.theta_bins
    generation_theta_limits =[Retina.theta_limits[0] + 2 * dtheta, Retina.theta_limits[1] - 2 * dtheta]

    dphi = (Retina.phi_limits[1] - Retina.phi_limits[0]) / Retina.phi_bins
    generation_phi_limits =[Retina.phi_limits[0] + 2 * dphi, Retina.phi_limits[1] - 2 * dphi]

    return particles(n_particles, detector_layers,
                     generation_theta_limits, generation_phi_limits,
                     trace_probability=trace_probability,
                     trace_noise=trace_noise,
                     detector_noise_rate=detector_noise_rate)