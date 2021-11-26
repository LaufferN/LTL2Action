# import ring
import torch
import numpy as np
import pydot
from ltlf2dfa.parser.ltlf import LTLfParser
from networkx.drawing.nx_agraph import to_agraph 

import dgl
import networkx as nx
from sklearn.preprocessing import OneHotEncoder
import utils
from utils.ast_builder import edge_types
import time

MAX_GUARD_LEN = 100

"""
A class that can take an LTL formula and generate a minimal DFA of it. This
code can generate graphs in either Networkx or DGL formats. And uses caching to remember recently
generated graphs.
"""
class DFABuilder(object):
    def __init__(self, propositions):
        super(DFABuilder, self).__init__()

        self.props = propositions


    # To make the caching work.
    # def __ring_key__(self):
    #     return "DFABuilder"

    # @ring.lru(maxsize=30000)
    def __call__(self, formula, library="dgl"):
        formula_str = utils.formatLTL(formula, self.props)
        parser = LTLfParser()
        formula = parser(formula_str)
        print("Trying to convert", formula_str, "to a DFA...")
        start = time.time()
        dfa_dot = formula.to_dfa()
        end = time.time()
        print("Done! It took", (end - start), "seconds")

        nxg = self._to_graph(dfa_dot)

        # The implementation below is just a place holder for us get an output from RGCN.
        # We can find a better model than RGCN for our purpose, e.g., DGI.
        nxg.remove_node("\\n")
        nx.set_node_attributes(nxg, 0., "is_root")
        nx.set_node_attributes(nxg, torch.ones(22), "feat")
        for e in nxg.edges:
            if e[0] == "init":
                # If there is an incoming edge from the special node init, then it is the initial node
                nxg.nodes[e[1]]["is_root"] = 1.
            if e[0] != e[1]:
                # If there is an outgoing edge to another node, then it is not an accepting node.
                nxg.nodes[e[0]]["feat"][0] = 0.
            if len(nxg.edges[e]) == 0:
                nxg.edges[e]["type"] = 0
                #nxg.edges[e]["label"] = torch.zeros(MAX_GUARD_LEN)
            else:
                guard = nxg.edges[e]["label"][1:-1]
                if guard == "true":
                    nxg.edges[e]["type"] = 0
                    #nxg.edges[e]["label"] = torch.zeros(MAX_GUARD_LEN)
                else:
                    nxg.edges[e]["type"] = sum(map(lambda x: ord(x), guard)) % (len(edge_types) - 1) + 1 # This is just a place holder
                    #guard_tensor = torch.tensor(list(map(lambda x: ord(x), guard)))
                    # For padding, we can try both A and B. We might need a smarter encoding for guards.
                    #padded_guard_tensor = torch.nn.functional.pad(guard_tensor, (0, MAX_GUARD_LEN - len(guard_tensor))) # A
                    #padded_guard_tensor = torch.nn.functional.pad(guard_tensor, (MAX_GUARD_LEN - len(guard_tensor), 0)) # B
                    #nxg.edges[e]["label"] = padded_guard_tensor
        if (library == "networkx"): return nxg

        # convert the Networkx graph to dgl graph and pass the 'feat' attribute
        g = dgl.DGLGraph()
        g.from_networkx(nxg, node_attrs=["feat", "is_root"], edge_attrs=["type"]) # dgl does not support string attributes (i.e., token)
        return g

    # A helper function that returns the networkx version of the dfa
    # @ring.lru(maxsize=60000) # Caching the formula->graph pairs in a Last Recently Used fashion
    def _to_graph(self, dfa_dot):
        pydot_formula = pydot.graph_from_dot_data(dfa_dot)[0]
        nxg = nx.drawing.nx_pydot.from_pydot(pydot_formula)
        
        return nxg

def draw(G, formula, path):
    from networkx.drawing.nx_agraph import graphviz_layout
    import matplotlib.pyplot as plt

    # use a_graph to plot our multigraph
    A = to_agraph(G) 
    A.layout('dot')                                                                 
    A.draw(path) 

"""
A simple test to check if the DFABuilder works fine. We do a preorder DFS traversal of the resulting
graph and convert it to a simplified formula and compare the result with the simplified version of the
original formula. They should match.
"""
if __name__ == '__main__':
    import re
    import sys
    import itertools
    import matplotlib.pyplot as plt

    # sys.path.insert(0, '../../')
    sys.path.append('.')
    sys.path.append('..')
    from ltl_samplers import getLTLSampler

    sampler_id = "Default"
    draw_path = "sample_dfa.png"
    # for sampler_id, _ in itertools.product(["Default", "Sequence_2_20"], range(1)):
    props = "abcdefghijklmnopqrst"
    sampler = getLTLSampler(sampler_id, props)
    builder = DFABuilder(list(set(list(props))))
    formula = sampler.sample()
    print("LTL Formula:", formula)
    graph = builder(formula, library="networkx")
    # pre = list(nx.dfs_preorder_nodes(graph, source=0))
    print("Output DFA image to", draw_path)
    draw(graph, formula, draw_path)

