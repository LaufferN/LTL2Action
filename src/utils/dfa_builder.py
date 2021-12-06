import ring
import torch
import numpy as np
import pydot
from ltlf2dfa.parser.ltlf import LTLfParser
from networkx.drawing.nx_agraph import to_agraph
from pysat.solvers import Solver
from copy import deepcopy

import dgl
import networkx as nx
from sklearn.preprocessing import OneHotEncoder
try:
    from utils.format_ltl import formatLTL
except:
    from format_ltl import formatLTL
try:
    from utils.ast_builder import edge_types
except:
    from ast_builder import edge_types
import time
import signal
import pickle

MAX_GUARD_LEN = 100
TIMEOUT_SECONDS = 600
dfa_db_path = "utils/dfa_db"

class TimeOutException(Exception):
    pass

def alarm_handler(signum, frame):
    raise TimeOutException()

"""
A class that can take an LTL formula and generate a minimal DFA of it. This
code can generate graphs in either Networkx or DGL formats. And uses caching to remember recently
generated graphs.
"""
class DFABuilder(object):
    def __init__(self, propositions):
        super(DFABuilder, self).__init__()
        self.count = 0
        self.props = propositions

    # To make the caching work.
    def __ring_key__(self):
        return "DFABuilder"

    def _get_generic_formula(self, formula):
        generic_formula = formula
        prop_mapping = {}
        positional_var_base = "x_"
        positional_var_count = 0
        for prop in formula:
            if prop in self.props:
                try:
                    positional_var = prop_mapping[prop]
                except:
                    positional_var = positional_var_base + str(positional_var_count)
                    positional_var_count += 1
                    prop_mapping[prop] = positional_var
                generic_formula = generic_formula.replace(prop, positional_var)
        return generic_formula, prop_mapping

    def _get_guard_embedding(self, guard):
        embedding = [0] * 22
        try:
            guard = guard["label"].replace(" ", "").replace("(", "").replace(")", "").replace("\"", "")
        except:
            return embedding
        if (guard == "true"):
            return embedding
        #print("Guard", guard)
        guard = guard.split("&")
        cnf = []
        seen_atoms = []
        for c in guard:
            atoms = c.split("|")
            clause = []
            for atom in atoms:
                #print(atom, self.props.index(atom) + 1 if atom[0] != "~" else -(self.props.index(atom[1:]) + 1), self.props)
                try:
                    index = seen_atoms.index(atom if atom[0] != "~" else atom[1:])
                except:
                    index = len(seen_atoms)
                    seen_atoms.append(atom if atom[0] != "~" else atom[1:])
                clause.append(index + 1 if atom[0] != "~" else -(index + 1))
            #print("")
            cnf.append(clause)
        #print("Seen Atom", seen_atoms)
        #print("CNF", cnf)
        models = []
        with Solver(bootstrap_with=cnf) as s:
            models = list(s.enum_models())
        #print("Models", models)
        if len(models) == 0:
            return embedding
        for model in models:
            for a in model:
                atom = seen_atoms[abs(a) - 1]
                #print(atom, self.props.index(atom))
                embedding[self.props.index(atom)] += 1 if a > 0 else -1
        for i in range(len(embedding)):
            embedding[i] /= len(models)
        #print("Embedding", embedding)
        return embedding

    @ring.lru(maxsize=100000)
    def __call__(self, formula, library="dgl"):
        self.count += 1
        #start = time.time()
        #print(self.count)

        formatted_formula = formatLTL(formula, self.props)
        generic_formula, prop_mapping = self._get_generic_formula(formatted_formula)

        generic_nxg = self._get_generic_nxg(generic_formula)

        nxg = self._get_nxg(generic_nxg, prop_mapping)

        nxg.remove_node("\\n")
        nx.set_node_attributes(nxg, torch.zeros(22), "feat")

        edges = deepcopy(nxg.edges)

        new_node_count = 0
        new_node_name_base_str = "temp_"

        for e in edges:
            embedding = self._get_guard_embedding(nxg.edges[e])
            if not all(i == 0 for i in embedding):
                nxg.remove_edge(*e)
                new_node_name = new_node_name_base_str + str(new_node_count)
                new_node_count += 1
                nxg.add_node(new_node_name)
                nxg.add_edge(e[0], new_node_name)
                nxg.add_edge(new_node_name, e[1])
                nxg.nodes[new_node_name]["feat"] = torch.tensor(embedding)

        nx.set_node_attributes(nxg, 0.0, "is_root")
        nxg.nodes["1"]["is_root"] = 1.0

        nx.set_edge_attributes(nxg, 0, "type")
        if (library == "networkx"):
            return nxg

        # convert the Networkx graph to dgl graph and pass the 'feat' attribute
        g = dgl.DGLGraph()
        g.from_networkx(nxg, node_attrs=["feat", "is_root"], edge_attrs=["type"]) # dgl does not support string attributes (i.e., token)

        #end = time.time()
        #print(end-start)
        return g

    def _get_nxg(self, generic_nxg, prop_mapping):
        nxg = generic_nxg
        for i in generic_nxg.edges:
            try:
                for prop in prop_mapping:
                    nxg.edges[i]["label"] = generic_nxg.edges[i]["label"].replace(prop_mapping[prop], prop)
            except:
                pass # This is needed for the initial dummy edge
        return nxg

    # A helper function that returns the networkx version of the dfa
    #@ring.lru(maxsize=1000000) # Caching the formula->graph pairs in a Last Recently Used fashion
    def _get_generic_nxg(self, formatted_formula):
        with open(dfa_db_path, "rb") as f:
            dfa_db = pickle.load(f)
        #print("There are", len(dfa_db), "entries in the dfa_db!")
        try:
            nxg = dfa_db[formatted_formula]
            #print("Found", formatted_formula, "in dfa_db!")
        except:
            parser = LTLfParser()
            formula = parser(formatted_formula)
            print("Trying to convert", formatted_formula, "to a DFA...")
            start = time.time()
            #signal.signal(signal.SIGALRM, alarm_handler)
            #signal.alarm(TIMEOUT_SECONDS)
            try:
                dfa_dot = formula.to_dfa()
                end = time.time()
                print("Done! It took", (end - start), "seconds")
                pydot_formula = pydot.graph_from_dot_data(dfa_dot)[0]
                nxg = nx.drawing.nx_pydot.from_pydot(pydot_formula)
                # Read it again just in case. Some other process might write something new.
                with open(dfa_db_path, "rb") as f:
                    dfa_db = pickle.load(f)
                dfa_db[formatted_formula] = nxg
                with open(dfa_db_path, "wb") as f:
                    pickle.dump(dfa_db, f)
            except TimeOutException:
                print("DFA construction timed out!")
                raise TimeOutException()
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

    props = "abcdefghijklmnopqrst"
    builder = DFABuilder(sorted(list(set(list(props)))))
    try:
        sampler_id = sys.argv[1]
        sampler = getLTLSampler(sampler_id, props)
        draw_path = "sample_dfa.png"
        formula = sampler.sample_new()
        print("LTL Formula:", formula)
        graph = builder(formula, library="networkx")
        print("Output DFA image to", draw_path)
        draw(graph, formula, draw_path)
    except:
        while True:
            for sampler_id in ["Until_1_3_1_2", "Eventually_1_5_1_4", "Until_1_2_1_1", "Adversarial"]:
                print(sampler_id)
                sampler = getLTLSampler(sampler_id, props)
                formula = sampler.sample_new()
                print("LTL Formula:", formula)
                graph = builder(formula, library="networkx")
        

