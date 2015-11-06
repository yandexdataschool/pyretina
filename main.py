from pyretina import simulation
from pyretina import plot3d

from pyretina import *

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm

def main():
  theta_limits = [-0.1, 0.1]
  phi_limits = [-np.pi, np.pi]

  theta_min, theta_max = theta_limits
  phi_min, phi_max = phi_limits

  event, params, ns = simulation.particles(50, np.arange(0, 20), [theta_min, theta_max], [phi_min, phi_max], 0.75, 0.0)

  #print retina_response_n(event, ns[0, :], 1.0)

  thetas, phis, response = retina_grid(event, theta_limits, 1000, phi_limits, 1000, 0.01)

  plt.figure()
  plt.contourf(thetas, phis, response, 20, cmap=cm.cool)
  plt.colorbar()
  plt.scatter(params[:, 0], params[:, 1], color="green")
  plt.show()

  #print response

main()