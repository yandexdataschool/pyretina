import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({'font.size': 18})

def main():
  recall = np.load('./recalls.npy').reshape(-1)
  precision = np.load('./precisions.npy').reshape(-1)
  N = np.load('./N.npy').reshape(-1)

  dN = 50

  r_ = list()
  p_ = list()

  Ns = list()

  for N_start in xrange(50, 350, dN):
    indx = (N >= N_start) & (N < (N_start + dN))
    if indx.shape[0] == 0:
      continue

    r_.append(recall[indx])
    p_.append(precision[indx])
    Ns.append(N_start)

  Ns = np.array(Ns)

  ra = np.zeros(shape=len(r_))
  lower = np.zeros(shape=len(r_))
  upper = np.zeros(shape=len(r_))

  for i, x in enumerate(r_):
    print x
    ra[i] = np.mean(x)
    lower[i] = np.percentile(x, q=10)
    upper[i] = np.percentile(x, q=90)

  #plt.boxplot(r_, positions = Ns, widths=dN, whis=None)
  plt.scatter(Ns, ra, label='mean efficiency')
  plt.errorbar(Ns, ra, yerr=np.vstack([ra - lower, upper - ra]), label='10% and 90% percentiles')

  plt.xlabel('number of reconstructable tracks')
  plt.ylabel('mean reconstruction efficiency')
  plt.xlim([0, np.max(Ns) + 50])
  plt.ylim([0.84, 1.01])
  plt.title('Artificial Retina efficiency')
  plt.legend(loc='lower left')
  plt.show()

if __name__ == '__main__':
  main()
