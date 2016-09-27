from pyretina.mc import monte_carlo

import numpy as np
import json

import os
import os.path as osp
import shutil

number_of_events = 10

def main(conf):
  with open(conf, 'r') as f:
    config = json.load(f)

  for N in np.arange(20, 520, 20):
    config['scattering']['number_of_particles'] = {
      'type' : 'randint',
      'low' : N,
      'high' : N + 1
    }

    plot_dir = osp.join('./events_img', '%d_particles' % N)
    try:
      shutil.rmtree(plot_dir)
    except:
      pass

    os.mkdir(plot_dir)

    events = monte_carlo(number_of_events, config, plot_dir=plot_dir, plot_each=2)
    import cPickle as pickle

    with open('data/mini_velo_sim_%d.pickled' % N, 'w') as f:
      pickle.dump(events, f)

    print 'Generated %d events with %d particles' % (number_of_events, N)

if __name__ == "__main__":
  main("config/mc.json")