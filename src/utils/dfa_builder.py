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
import time
import signal
import pickle

edge_types = ["self", "normal-to-temp", "temp-to-normal"]

TIMEOUT_SECONDS = 600
FEATURE_SIZE = 22
dfa_db_path = "utils/dfa_db"
DGL5_COMPAT = False

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
    def __init__(self, propositions, use_mean_guard_embed, use_onehot_guard_embed):
        super(DFABuilder, self).__init__()
        self.props = propositions
        self.use_mean_guard_embed = use_mean_guard_embed
        self.use_onehot_guard_embed = use_onehot_guard_embed

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

    def _get_onehot_guard_embeddings(self, guard):
        is_there_onehot = False
        is_there_all_zero = False
        onehot_embedding = [0.0] * FEATURE_SIZE
        full_embeddings = self._get_guard_embeddings(guard)
        for embed in full_embeddings:
            # discard all non-onehot embeddings (a one-hot embedding must contain only a single 1)
            if embed.count(1.0) == 1:
                # clean the embedding so that it's one-hot
                is_there_onehot = True
                var_idx = embed.index(1.0)
                onehot_embedding[var_idx] = 1.0
            elif embed.count(0.0) == len(embed):
                is_there_all_zero = True
        if is_there_onehot or is_there_all_zero:
            return [onehot_embedding]
        else:
            return []

    def _get_guard_embeddings(self, guard):
        embeddings = []
        try:
            guard = guard["label"].replace(" ", "").replace("(", "").replace(")", "").replace("\"", "")
        except:
            return embeddings
        if (guard == "true"):
            return embeddings
        guard = guard.split("&")
        cnf = []
        seen_atoms = []
        for c in guard:
            atoms = c.split("|")
            clause = []
            for atom in atoms:
                try:
                    index = seen_atoms.index(atom if atom[0] != "~" else atom[1:])
                except:
                    index = len(seen_atoms)
                    seen_atoms.append(atom if atom[0] != "~" else atom[1:])
                clause.append(index + 1 if atom[0] != "~" else -(index + 1))
            cnf.append(clause)
        models = []
        with Solver(bootstrap_with=cnf) as s:
            models = list(s.enum_models())
        if len(models) == 0:
            return embeddings
        for model in models:
            temp = [0.0] * FEATURE_SIZE
            for a in model:
                atom = seen_atoms[abs(a) - 1]
                temp[self.props.index(atom)] = 1.0 if a > 0 else -1.0
            embeddings.append(temp)
        return embeddings

    def _get_mean_guard_embedding(self, embeddings):
        mean_embedding = [0.0] * FEATURE_SIZE
        if len(embeddings) == 0:
            return mean_embedding
        for embedding_i in embeddings:
            for j in range(len(mean_embedding)):
                mean_embedding[j] += embedding_i[j]
        for i in range(len(mean_embedding)):
            mean_embedding[i] /= len(embeddings)
        return mean_embedding

    def _is_sink_state(self, node, nxg):
        for edge in nxg.edges:
            if node == edge[0] and node != edge[1]: # If there is an outgoing edge to another node, then it is not an accepting state
                return False
        return True

    @ring.lru(maxsize=100000)
    def __call__(self, formula, library="dgl"):

        formatted_formula = formatLTL(formula, self.props)
        generic_formula, prop_mapping = self._get_generic_formula(formatted_formula)

        generic_nxg = deepcopy(self._get_generic_nxg(generic_formula))

        nxg = self._get_nxg(generic_nxg, prop_mapping)

        try:
            nxg.remove_node("\\n")
        except:
            pass

        accepting_states = []
        rejecting_states = []
        is_mark_seen = False
        for node in nxg.nodes:
            if node == "init":
                is_mark_seen = True
                continue
            if self._is_sink_state(node, nxg):
                if not is_mark_seen:
                    accepting_states.append(node)
                else:
                    rejecting_states.append(node)

        try:
            nxg.remove_node("init")
        except:
            pass

        for node in nxg.nodes:
            nxg.nodes[node]["feat"] = np.array([[0.0] * FEATURE_SIZE])
            nxg.nodes[node]["feat"][0][-4] = 1.0
            if node in accepting_states:
                nxg.nodes[node]["feat"][0][-2] = 1.0
            if node in rejecting_states:
                nxg.nodes[node]["feat"][0][-1] = 1.0

        nxg.nodes["1"]["feat"][0][-5] = 1.0
        edges = deepcopy(nxg.edges)

        new_node_name_base_str = "temp_"
        new_node_name_counter = 0

        for e in edges:
            guard = nxg.edges[e]
            nxg.remove_edge(*e)
            if e[0] == e[1]:
                continue # We define self loops below
            if self.use_onehot_guard_embed:
                embeddings = self._get_onehot_guard_embeddings(guard)
                if len(embeddings) == 0:
                    continue
            else:
                embeddings = self._get_guard_embeddings(guard)
            if self.use_mean_guard_embed:
                mean_embedding = [self._get_mean_guard_embedding(embeddings)]
                mean_embedding[0][-3] = 1.0
                new_node_name = new_node_name_base_str + str(new_node_name_counter)
                new_node_name_counter += 1
                nxg.add_node(new_node_name, feat=np.array(mean_embedding))
                nxg.add_edge(e[0], new_node_name, type=1)
                nxg.add_edge(new_node_name, e[1], type=2)
            else:
                if len(embeddings) == 0:
                    # Let's double check this part if we decide to use this encoding convention again.
                    # With one hot embedding, line 191 and 192 guarantee that this part will not be executed.
                    embedding = [[0.0] * FEATURE_SIZE]
                    embedding[0][-3] = 1.0
                    new_node_name = new_node_name_base_str + str(new_node_name_counter)
                    new_node_name_counter += 1
                    nxg.add_node(new_node_name, feat=np.array(embedding))
                    nxg.add_edge(e[0], new_node_name, type=1)
                    nxg.add_edge(new_node_name, e[1], type=2)
                else:
                    for i in range(len(embeddings)):
                        embedding = [embeddings[i]]
                        embedding[0][-3] = 1.0
                        new_node_name = new_node_name_base_str + str(new_node_name_counter)
                        new_node_name_counter += 1
                        nxg.add_node(new_node_name, feat=np.array(embedding))
                        nxg.add_edge(e[0], new_node_name, type=1)
                        nxg.add_edge(new_node_name, e[1], type=2)

        for node in nxg.nodes:
            nxg.add_edge(node, node, type=0)

        if DGL5_COMPAT:
            nx.set_node_attributes(nxg, [0.0], "is_root")
            nxg.nodes["1"]["is_root"] = [1.0]
        else:
            nx.set_node_attributes(nxg, [0.0], "is_root")
            nxg.nodes["1"]["is_root"] = [1.0]

        """print(formula)
        print("Number of nodes:", len(nxg.nodes))
        print("Number of edges:", len(nxg.edges))
        for i in nxg.nodes:
            print("Node:", i)
            print(nxg.nodes[i]["feat"].shape, type(nxg.nodes[i]["feat"]), nxg.nodes[i]["feat"], type(nxg.nodes[i]["feat"][0][0]))
            print(nxg.nodes[i]["is_root"], type(nxg.nodes[i]["is_root"]))
        for i in nxg.edges:
            print(nxg.edges[i]["type"], type(nxg.edges[i]["type"]))"""

        nxg = nxg.reverse(copy=True)
        if (library == "networkx"):
            return nxg

        # convert the Networkx graph to dgl graph and pass the 'feat' attribute
        if DGL5_COMPAT:
            g = dgl.from_networkx(nxg, node_attrs=["feat", "is_root"], edge_attrs=["type"]) # dgl does not support string attributes (i.e., token)
        else:
            g = dgl.DGLGraph()
            g.from_networkx(nxg, node_attrs=["feat", "is_root"], edge_attrs=["type"]) # dgl does not support string attributes (i.e., token)

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
    @ring.lru(maxsize=100000) # Caching the formula->graph pairs in a Last Recently Used fashion
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

    props = "abcdefghijkl"
    builder = DFABuilder(sorted(list(set(list(props)))), False, True)
    try:
        sampler_id = sys.argv[1]
        sampler = getLTLSampler(sampler_id, props)
        draw_path = "sample_dfa.png"
        formula = sampler.sample()
        print(builder.props)
        print("LTL Formula:", formula)
        graph = builder(formula, library="networkx")
        print("Output DFA image to", draw_path)
        draw(graph, formula, draw_path)
    except:
        for sampler_id in ["Until_1_1_1_1", "Until_1_1_2_2", "Until_2_2_1_1", "Until_2_2_2_2",
                           "Eventually_1_1_1_1", "Eventually_1_1_2_2", "Eventually_2_2_1_1", "Eventually_2_2_2_2"]:
            print(sampler_id)
            sampler = getLTLSampler(sampler_id, props)
            formula = sampler.sample_new()
            print("LTL Formula:", formula)
            graph = builder(formula, library="networkx")
            draw_path = "sample_dfa_" + sampler_id + ".png"
            draw(graph, formula, draw_path)
        

