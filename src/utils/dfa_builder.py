# import ring
import numpy as np
import pydot
from ltlf2dfa.parser.ltlf import LTLfParser
from networkx.drawing.nx_agraph import to_agraph 

import dgl
import networkx as nx
from sklearn.preprocessing import OneHotEncoder

edge_types = {k:v for (v, k) in enumerate(["self", "arg", "arg1", "arg2"])}

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
    def __call__(self, formula_str, library="dgl"):
        parser = LTLfParser()
        formula = parser(formula_str)
        dfa_dot = formula.to_dfa()

        nxg = self._to_graph(dfa_dot)

        nx.set_node_attributes(nxg, 0., "init_state")
        # nxg.nodes[0]["init_state"] = 1.
        if (library == "networkx"): return nxg

        # convert the Networkx graph to dgl graph and pass the 'feat' attribute
        g = dgl.DGLGraph()
        g.from_networkx(nxg, node_attrs=["feat", "init_state"], edge_attrs=["type"]) # dgl does not support string attributes (i.e., token)
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
    from format_ltl import formatLTL

    sampler_id = "Default"
    draw_path = "sample_dfa.png"
    # for sampler_id, _ in itertools.product(["Default", "Sequence_2_20"], range(1)):
    props = "abcdefghijklmnopqrst"
    sampler = getLTLSampler(sampler_id, props)
    builder = DFABuilder(list(set(list(props))))
    formula = sampler.sample()
    print("LTL Formula:", formula)
    formula = formatLTL(formula, props)
    print("Formatted LTL Formula:", formula)
    graph = builder(formula, library="networkx")
    # pre = list(nx.dfs_preorder_nodes(graph, source=0))
    print("Output DFA image to", draw_path)
    draw(graph, formula, draw_path)

