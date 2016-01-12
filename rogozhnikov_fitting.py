from pyretina.optimize import multistart_until
from pyretina.rogozhnikov_curve import RogozhnikovCurve
import pandas as pd
import numpy as np


def _unparams(params):
  cx = params[:2]
  cy = params[2:4]
  c0x = params[4]
  c0y = params[5]
  z0 = params[6]
  gamma = params[7]
  return cx, cy, c0x, c0y, z0, gamma

bounds = np.array([
  [0, 2], # ax
  [0, 2], # bx
  [0, 2], # ay
  [0, 2], # by
  [-6000, 5000], # x0
  [-6000, 5000], # y0
  [5000, 6000],  # z0
  [-0.1, 0.1] # gamma
])

def main():
  import sys
  import os
  paths = sys.argv[1:]

  errors = list()
  params = list()

  try:
    for path in ['00163838_0067087057.tracks.csv']:#os.listdir("./data/MC"):
        df = pd.DataFrame.from_csv("./data/MC/%s" % path)
        tx = df[ [u'x%d' % i for i in range(11) ] ].values
        ty = df[ [u'y%d' % i for i in range(11) ] ].values
        tz = df[ [u'z%d' % i for i in range(11) ] ].values

        for ei in range(tx.shape[0]):
          print "Fitting %d-th track of %s" % (ei, path)
          rc = RogozhnikovCurve(tx[ei, :], ty[ei, :], tz[ei, :], spline=True, spline_n=25).fast_fit()

          print "Error:", rc.error
          print "Params", rc.params

          errors.append(rc.error)
          params.append(rc.params)
  finally:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))
    plt.hist(np.array(errors))
    plt.savefig("r_fitting_errors.png")

    np.savetxt("errors.csv", np.array(errors))
    np.savetxt("params.csv", np.array(params))

if __name__ == "__main__":
  main()