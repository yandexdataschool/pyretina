#!/usr/bin/python

import numpy as np
import pandas as pd
from pyretina.retina_event import RetinaEvent

def load_dataset(infile, sigma, theta_bins = 200, phi_bins = 200, theta_limits = None, phi_limits = None):
  data = pd.DataFrame.from_csv("%s.hits.csv" % infile)[['X', 'Y', 'Z']].values
  tracks = pd.DataFrame.from_csv("%s.tracks.csv" % infile)[['px', 'py', 'pz']].values

  return RetinaEvent(
    data, tracks, sigma=sigma,
    theta_limits=theta_limits,
    theta_bins=theta_bins,
    phi_limits=phi_limits,
    phi_bins=phi_bins
  )