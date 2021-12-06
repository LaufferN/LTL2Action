#import ring
import torch
import numpy as np
import pydot
from ltlf2dfa.parser.ltlf import LTLfParser
from ltlf2dfa.ltlf2dfa import output2dot
from ltlf2dfa import ltlf2dfa
from networkx.drawing.nx_agraph import to_agraph 

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

        self.props = propositions

    # To make the caching work.
    #def __ring_key__(self):
    #    return "DFABuilder"

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

    def __call__(self, formula, library="dgl"):

        formatted_formula = formatLTL(formula, self.props)
        generic_formula, prop_mapping = self._get_generic_formula(formatted_formula)
        # try:
        print("Getting generic nxg")
            # generic_nxg = self._get_generic_nxg(generic_formula)
        generic_nxg = self._construct_generic_nxg(generic_formula)
        # except:
        #     return None
        nxg = self._get_nxg(generic_nxg, prop_mapping)

        if (library == "networkx"):
            return nxg

        # convert the Networkx graph to dgl graph and pass the 'feat' attribute
        g = dgl.DGLGraph()
        g.from_networkx(nxg, node_attrs=["feat", "is_root", "is_state_node"], edge_attrs=["type"]) # dgl does not support string attributes (i.e., token)
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

    def _construct_generic_nxg(self, formatted_formula):
        parser = LTLfParser()
        formula = parser(formatted_formula)
        print("Trying to convert", formatted_formula, "to a DFA...")
        start = time.time()
        print("Calling mona...")
        mona_dfa = formula.to_dfa(mona_dfa_out=True)
        dfa_dot = output2dot(mona_dfa)
        end = time.time()
        print("Done! It took", (end - start), "seconds")
        pydot_formula = pydot.graph_from_dot_data(dfa_dot)[0]
        nxg = nx.drawing.nx_pydot.from_pydot(pydot_formula)
        # Read it again just in case. Some other process might write something new.
        # with open(dfa_db_path, "rb") as f:
        #     dfa_db = pickle.load(f)
        # dfa_db[formatted_formula] = nxg
        # with open(dfa_db_path, "wb") as f:
        #     pickle.dump(dfa_db, f)
        # The implementation below is just a place holder for us get an output from RGCN.
        # We can find a better model than RGCN for our purpose, e.g., DGI.
        nxg = self._formatGNN(nxg, mona_dfa)

        return nxg

    # A helper function that returns the networkx version of the dfa
    #@ring.lru(maxsize=1000000) # Caching the formula->graph pairs in a Last Recently Used fashion
    def _get_generic_nxg(self, formatted_formula):
        # with open(dfa_db_path, "rb") as f:
        #     dfa_db = pickle.load(f)
        # print("There are", len(dfa_db), "entries in the dfa_db!")
        dfa_db = {}
        try:
            nxg = dfa_db[formatted_formula]
            print("Found", formatted_formula, "in dfa_db!")
        except:
            parser = LTLfParser()
            formula = parser(formatted_formula)
            print("Trying to convert", formatted_formula, "to a DFA...")
            start = time.time()
            # signal.signal(signal.SIGALRM, alarm_handler)
            # signal.alarm(TIMEOUT_SECONDS)
            try:
                print("Calling mona")
                mona_dfa = formula.to_dfa(mona_dfa_out=True)
                dfa_dot = output2dot(mona_dfa)
                end = time.time()
                print("Done! It took", (end - start), "seconds")
                pydot_formula = pydot.graph_from_dot_data(dfa_dot)[0]
                nxg = nx.drawing.nx_pydot.from_pydot(pydot_formula)
                # Read it again just in case. Some other process might write something new.
                # with open(dfa_db_path, "rb") as f:
                #     dfa_db = pickle.load(f)
                # dfa_db[formatted_formula] = nxg
                # with open(dfa_db_path, "wb") as f:
                #     pickle.dump(dfa_db, f)
                # The implementation below is just a place holder for us get an output from RGCN.
                # We can find a better model than RGCN for our purpose, e.g., DGI.
                self._formatGNN(nxg, mona_dfa)
            except TimeOutException:
                print("DFA construction timed out!")
                raise TimeOutException()
        return nxg

    def _formatGNN(self, nxg, mona_dfa=None):
        # TODO
        # convert DFA multigraph into flattened DFA with new nodes in this file
        # make a new GNN type that does double message passing with shared weights on the added edges
        print("Formatting GNN")
        # nxg.remove_node("\\n")
        nx.set_node_attributes(nxg, 0., "is_root")
        nx.set_node_attributes(nxg, torch.ones(22), "feat") # GNN feature input
        nx.set_node_attributes(nxg, 1, "is_state_node") # edge node or actual node
        if mona_dfa == None:
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
        else:
            guards_dict = extract_mona_guards(mona_dfa)
            graph_edges = nxg.copy().edges
            for e in graph_edges:
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
                    # remove the old edge
                    nxg.remove_edge(*e)

                    e0 = int(e[0])
                    e1 = int(e[1])
                    guards = guards_dict[(e0,e1)]
                    # add new edge node with key (e0, e1)
                    for i,guard_str in enumerate(guards):
                        new_node = (e0,e1,i)
                        nxg.add_node(new_node)
                        nxg.nodes[new_node]["is_state_node"] = 0 # this is an edge node, not a state node
                        nxg.nodes[new_node]["feat"] = torch.zeros(22)
                        # encode this new node's guard feature
                        for j,char in enumerate(guard_str):
                            if char != "X":
                                nxg.nodes[new_node]["feat"][j] = int(char)

                        # add in edges for this new node
                        nxg.add_edge(e0, new_node)
                        nxg.add_edge(new_node, e1)

        return nxg


def extract_mona_guards(mona_output):
    """Parse mona output and construct a dot."""
    free_variables = ltlf2dfa.get_value(
        mona_output, r".*DFA for formula with free variables:[\s]*(.*?)\n.*", str
    )
    if "state" in free_variables:
        free_variables = None
    else:
        free_variables = ltlf2dfa.symbols(
            " ".join(
                x.strip().lower() for x in free_variables.split() if len(x.strip()) > 0
            )
        )

    # initial_state = ltlf2dfa.get_value(mona_output, '.*Initial state:[\s]*(\d+)\n.*', int)
    accepting_states = ltlf2dfa.get_value(mona_output, r".*Accepting states:[\s]*(.*?)\n.*", str)
    accepting_states = [
        str(x.strip()) for x in accepting_states.split() if len(x.strip()) > 0
    ]

    edge_guards = dict()  # maps each couple (src, dst) to a list of guards
    for line in mona_output.splitlines():
        if line.startswith("State "):
            orig_state = ltlf2dfa.get_value(line, r".*State[\s]*(\d+):\s.*", int)
            guard = ltlf2dfa.get_value(line, r".*:[\s](.*?)[\s]->.*", str)
            dest_state = ltlf2dfa.get_value(line, r".*state[\s]*(\d+)[\s]*.*", int)
            if orig_state:
                if (orig_state, dest_state) in edge_guards.keys():
                    edge_guards[(orig_state, dest_state)].append(guard)
                else:
                    edge_guards[(orig_state, dest_state)] = [guard]

    # for c, guards in edge_guards.items():
    #     print(guards)
    return edge_guards


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
    builder = DFABuilder(list(set(list(props))))
    try:
        sampler_id = sys.argv[1]
        sampler = getLTLSampler(sampler_id, props)
        draw_path = "sample_dfa.png"
        formula = sampler.sample()
        print("LTL Formula:", formula)
        graph = builder(formula, library="networkx")
        print("Output DFA image to", draw_path)
        draw(graph, formula, draw_path)
    except:
        while True:
            # for sampler_id in ["Until_1_3_1_2", "Eventually_1_5_1_4", "Until_1_2_1_1", "Adversarial"]:
            for sampler_id in ["Default"]:
                print(sampler_id)
                sampler = getLTLSampler(sampler_id, props)
                formula = sampler.sample()
                print("LTL Formula:", formula)
                graph = builder(formula, library="networkx")
                draw_path = "sample_dfa_expanded.png"
                print("Output DFA image to", draw_path)
                draw(graph, formula, draw_path)
        

