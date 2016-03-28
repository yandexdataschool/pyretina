from pyretina.mc import monte_carlo

def main(conf):
  events = monte_carlo(10000, conf)

  import cPickle as pickle

  with open('data/velo_sim_100_150.pickled', 'w') as f:
    pickle.dump(events, f)

if __name__ == "__main__":
  main("config/mc.json")