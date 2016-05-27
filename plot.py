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

  for N_start in xrange(50, 400, dN):
    indx = (N >= N_start) & (N < (N_start + dN))
    if indx.shape[0] == 0:
      continue

    r_.append(recall[indx])
    p_.append(precision[indx])
    Ns.append(N_start)

  Ns = np.array(Ns)

  plt.boxplot(r_, positions = Ns, widths=dN, whis=[5, 95])
  plt.xlabel('number of reconstructable tracks')
  plt.ylabel('recall / efficiency')
  plt.xlim([0, np.max(Ns)])
  plt.ylim([np.min(recall) - 0.05, 1.01])
  plt.title('Artificial Retina efficiency')
  plt.show()

  plt.boxplot(p_, positions=Ns, widths=dN, whis=[5, 95])
  plt.xlabel('Number of reconstructable tracks')
  plt.ylabel('precision / 1 - ghost rate')
  plt.xlim([0, np.max(Ns)])
  plt.title('Artificial Retina efficiency')
  plt.show()

if __name__ == '__main__':
  main()