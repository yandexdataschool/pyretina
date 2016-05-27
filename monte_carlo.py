from pyretina.mc import monte_carlo

import numpy as np
import json

number_of_events = 1000

def main(conf):
  with open(conf, 'r') as f:
    config = json.load(f)

  #for N in np.arange(100, 520, 20):
  for N in [50]:
    config['scattering']['number_of_particles'] = {
      'type' : 'randint',
      'low' : N,
      'high' : N + 1
    }

    events = monte_carlo(number_of_events, config)
    import cPickle as pickle

    with open('data/mini_velo_sim_%d.pickled' % N, 'w') as f:
      pickle.dump(events, f)

    print 'Generated %d events with %d particles' % (number_of_events, N)

if __name__ == "__main__":
  main("config/mc.json")