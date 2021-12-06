import pickle
from matplotlib import pyplot as plt
import numpy as np
from networkx.drawing.nx_agraph import to_agraph

with open("utils/dfa_db", "rb") as f:
    dfa_db = pickle.load(f)

modified_node_counts = []
node_count_or_count_pairs = []
node_count_edge_count_pairs = []
node_count_modified_node_count_pairs = []
for formula in dfa_db.keys():
    dfa = dfa_db[formula]
    dfa.remove_node("\\n")
    node_count = len(dfa.nodes) - 1 # Do not count dummy init node
    modified_node_count = node_count
    or_count = 0
    for edge in dfa.edges:
        try:
            or_count += dfa.edges[edge]["label"].count("|")
            modified_node_count += (dfa.edges[edge]["label"].count("|") + 1)
        except:
            pass
    modified_node_counts.append(modified_node_count)
    node_count_or_count_pairs.append((node_count, or_count))
    node_count_edge_count_pairs.append((node_count, len(dfa.edges)))
    node_count_modified_node_count_pairs.append((node_count, modified_node_count))

i_max = np.argmax(modified_node_counts)
#print(i_max)
#print(node_count_or_count_pairs[i_max], len(dfa_db[list(dfa_db.keys())[i_max]].edges), modified_node_counts[i_max])
j_max = np.argmax(list(map(lambda x: x[0] + x[1], node_count_edge_count_pairs)))
#print(node_count_edge_count_pairs[j_max])

print(np.min(modified_node_counts), np.mean(modified_node_counts), np.median(modified_node_counts), np.percentile(modified_node_counts, 90), np.percentile(modified_node_counts, 95), np.percentile(modified_node_counts, 99), np.max(modified_node_counts))
plt.hist(modified_node_counts)
plt.show()

sorted_node_count_modified_node_count_pairs = sorted(node_count_modified_node_count_pairs)
#print(sorted_node_count_modified_node_count_pairs)
#plt.plot(list(map(lambda x: x[0], sorted_node_count_modified_node_count_pairs)), list(map(lambda x: x[1], sorted_node_count_modified_node_count_pairs)))
plt.show()

#temp_sorted = sorted(temp)[:209]
#i = temp.index(np.max(temp_sorted))
#print(np.max(temp_sorted), temp.index(np.max(temp_sorted)), len(dfa_db[list(dfa_db.keys())[i]].nodes))
#A = to_agraph(dfa_db[list(dfa_db.keys())[i_max]]) 
#A.layout('dot')                                                                 
#A.draw("max_dfa.png") 





#nodes = list(map(lambda x: x.nodes, list(dfa_db.values())))
#edges = list(map(lambda x: x.edges, list(dfa_db.values())))
#print(np.max(nodes))
#print(edges[0])


#node_counts = list(map(lambda x: len(x.nodes), list(dfa_db.values())))
#plt.hist(node_counts, bins = [0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200])
#plt.show()

#N = len(node_counts)
#X2 = np.sort(node_counts)
#F2 = np.array(range(N))/float(N)

#plt.plot(X2, F2)

#plt.savefig("cumulative_density_distribution.png", bbox_inches='tight')

#plt.show()