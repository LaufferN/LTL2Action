import pickle
from matplotlib import pyplot as plt
import numpy as np

with open("utils/dfa_db", "rb") as f:
       dfa_db = pickle.load(f)
node_counts = list(map(lambda x: len(x.nodes), list(dfa_db.values())))
#plt.hist(node_counts, bins = [0,10,20,30,40,50,60,70,80,90,100,110,120,130])
#plt.show()

N = len(node_counts)
X2 = np.sort(node_counts)
F2 = np.array(range(N))/float(N)

plt.plot(X2, F2)

plt.savefig("cumulative_density_distribution.png", bbox_inches='tight')

plt.show()