from _retina_response import *
import numpy as np

def grid(event, sigma, z0, range_theta, range_phi, bins_theta=100, bins_phi=100):
  theta_lin = np.linspace(range_theta[0], range_theta[1], num=bins_theta)
  phi_lin = np.linspace(range_phi[0], range_phi[1], num=bins_phi)

  thetas, phis = np.meshgrid(theta_lin, phi_lin)

  set_event(event)
  set_sigma(sigma)
  set_z0(z0)

  return thetas, phis, linear_retina_response_vec(thetas, phis)

def grad_grid(event, sigma, z0, range_theta, range_phi, bins_theta=100, bins_phi=100):
  theta_lin = np.linspace(range_theta[0], range_theta[1], num=bins_theta)
  phi_lin = np.linspace(range_phi[0], range_phi[1], num=bins_phi)

  thetas, phis = np.meshgrid(theta_lin, phi_lin)

  set_event(event)
  set_sigma(sigma)
  set_z0(z0)

  return thetas, phis, linear_retina_response_jac_vec(thetas, phis)