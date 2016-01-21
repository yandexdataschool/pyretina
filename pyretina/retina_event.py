#!/usr/bin/python

import numpy as np

class RetinaEvent:
  event = None
  tracks = None
  sigma = None

  theta_limits = None
  theta_bins = 200
  theta_step = 0.0

  phi_limits = None
  phi_bins = 200
  phi_step = 0.0

  grid = None

  thetas = None
  phis = None

  optimization_trace = None

  def track_dim(self):
    return self.tracks.shape[1]

  def track_bounds(self):
    return np.array([self.theta_limits, self.phi_limits])

  def __grid(self):
    theta_lin = np.linspace(self.theta_limits[0], self.theta_limits[1], num=self.theta_bins)
    phi_lin = np.linspace(self.phi_limits[0], self.phi_limits[1], num=self.phi_bins)

    self.thetas, self.phis = np.meshgrid(theta_lin, phi_lin)

    rr, rr_jac, rr_hess = self.get_response()
    rr_vec = np.vectorize(lambda t, p: rr([t, p]))

    self.grid = rr_vec(self.thetas, self.phis)


  def make_grid(self):
    self.__grid()
    return self.thetas, self.phis, self.grid

  def get_grid(self):
    if self.grid is None:
      return self.make_grid()

    return self.thetas, self.phis, self.grid

  def __init__(self, event, tracks, sigma, theta_limits, theta_bins, phi_limits, phi_bins):
    self.event = event
    self.tracks = tracks

    self.theta_limits = theta_limits
    self.phi_limits = phi_limits

    self.theta_bins = theta_bins
    self.phi_bins = phi_bins

    self.theta_step = (self.theta_limits[1] - self.theta_limits[0]) / self.theta_bins
    self.phi_step = (self.phi_limits[1] - self.phi_limits[0]) / self.phi_bins

    self.sigma = sigma

  def get_response(self):
    from retina_response import set_sigma, set_event
    from retina_response import linear_retina_response, linear_retina_response_jac, linear_retina_response_hess
    set_sigma(self.sigma)
    set_event(self.event)

    return linear_retina_response, linear_retina_response_jac, linear_retina_response_hess

  def get_neg_response(self):
    from retina_response import set_sigma, set_event
    from retina_response import neg_linear_retina_response,\
      neg_linear_retina_response_jac,\
      neg_linear_retina_response_hess

    set_sigma(self.sigma)
    set_event(self.event)

    return neg_linear_retina_response, neg_linear_retina_response_jac, neg_linear_retina_response_hess

  n_particles = 50
  detector_layers = np.arange(0, 20) + 2
  trace_probability = 0.75

  trace_noise = 0.01
  detector_noise = n_particles * detector_layers.shape[0]