"""
This class is responsible for sampling LTL formulas typically from
given template(s).

@ propositions: The set of propositions to be used in the sampled
                formula at random.
"""

import ring
import time
import pydot
import signal
import random
import pickle
import numpy as np
import networkx as nx
from copy import deepcopy
from pysat.solvers import Solver
from ltlf2dfa.parser.ltlf import LTLfParser
try:
    from utils.format_ltl import formatLTL
except ModuleNotFoundError:
    from format_ltl import formatLTL

dfa_db_path = "utils/dfa_db"
TIMEOUT_SECONDS = 600
FEATURE_SIZE = 26

class TimeOutException(Exception):
    pass

def alarm_handler(signum, frame):
    raise TimeOutException()

class DFASampler():
    def __init__(self, propositions):
        self.propositions = propositions
        with open(dfa_db_path, "rb") as f:
            self.dfa_db = pickle.load(f)

    # To make the caching work.
    def __ring_key__(self):
        return "DFASampler"

    def _get_generic_formula(self, formula):
        generic_formula = formula
        prop_mapping = {}
        positional_var_base = "x_"
        positional_var_count = 0
        for prop in formula:
            if prop in self.propositions:
                try:
                    positional_var = prop_mapping[prop]
                except:
                    positional_var = positional_var_base + str(positional_var_count)
                    positional_var_count += 1
                    prop_mapping[prop] = positional_var
                generic_formula = generic_formula.replace(prop, positional_var)
        return generic_formula, prop_mapping

    def _get_guard_embeddings(self, guard):
        embeddings = []
        try:
            guard = guard.replace(" ", "").replace("(", "").replace(")", "").replace("\"", "")
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
                temp[self.propositions.index(atom)] = 1.0 if a > 0 else -1.0
            embeddings.append(temp)
        return embeddings

    @ring.lru(maxsize=100000)
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

    def _is_sink_state(self, node, nxg):
        for edge in nxg.edges:
            if node == edge[0] and node != edge[1]: # If there is an outgoing edge to another node, then it is not an accepting state
                return False
        return True

    @ring.lru(maxsize=100000)
    def _get_dfa_from_ltl(self, formula):
        formatted_formula = formatLTL(formula, self.propositions)
        generic_formula, prop_mapping = self._get_generic_formula(formatted_formula)
        dfa_from_db = None
        try:
            dfa_from_db = self.dfa_db[generic_formula]
            #print("Found", generic_formula, "in dfa_db!")
        except:
            parser = LTLfParser()
            parsed_generic_formula = parser(generic_formula)
            print("Trying to convert", generic_formula, "to a DFA...")
            start = time.time()
            signal.signal(signal.SIGALRM, alarm_handler)
            signal.alarm(TIMEOUT_SECONDS)
            try:
                dfa_dot = parsed_generic_formula.to_dfa()
                end = time.time()
                print("Done! It took", (end - start), "seconds")
                pydot_formula = pydot.graph_from_dot_data(dfa_dot)[0]
                dfa_from_db = nx.drawing.nx_pydot.from_pydot(pydot_formula)
                # Read it again just in case. Some other process might write something new.
                with open(dfa_db_path, "rb") as f:
                    self.dfa_db = pickle.load(f)
                self.dfa_db[generic_formula] = dfa_from_db
                with open(dfa_db_path, "wb") as f:
                    pickle.dump(self.dfa_db, f)
            except TimeOutException:
                print("DFA construction timed out!")
                return None
        nxg = deepcopy(dfa_from_db) # We need to deepcopy the DFA from DB since we do not want to change the generic DFAs from DB.
        try:
            nxg.remove_node("\\n")
        except:
            pass
        current_node = None
        for e in nxg.edges:
            if e[0] == "init":
                current_node = e[1]
            try:
                for prop in prop_mapping:
                    nxg.edges[e]["label"] = nxg.edges[e]["label"].replace(prop_mapping[prop], prop)
            except:
                pass # This is needed for the initial dummy edge

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

        nxg.nodes[current_node]["feat"][0][-5] = 1.0

        edges = deepcopy(nxg.edges)

        new_node_name_base_str = "temp_"
        new_node_name_counter = 0

        for e in edges:
            guard = nxg.edges[e]["label"]
            nxg.remove_edge(*e)
            if e[0] == e[1]:
                continue # We define self loops below
            embeddings = deepcopy(self._get_onehot_guard_embeddings(guard)) # We might receive a cached onehot embedding so we have to deepcopy
            if len(embeddings) == 0:
                continue
            for i in range(len(embeddings)):
                embedding = [embeddings[i]]
                embedding[0][-3] = 1.0
                new_node_name = new_node_name_base_str + str(new_node_name_counter)
                new_node_name_counter += 1
                nxg.add_node(new_node_name, feat=np.array(embedding))
                nxg.add_edge(e[0], new_node_name, type=1)
                nxg.add_edge(new_node_name, e[1], type=2)

        nx.set_node_attributes(nxg, [0.0], "is_root")
        nxg.nodes[current_node]["is_root"] = [1.0] # is_root means current state

        for node in nxg.nodes:
            nxg.add_edge(node, node, type=0)

        return nxg

    def is_in_dfa_db(self, formula):
        formatted_formula = formatLTL(formula, self.propositions)
        generic_formula, _ = self._get_generic_formula(formatted_formula)
        if generic_formula in self.dfa_db:
            return True
        else:
            return False

    def sample_ltl_formula(self):
        raise NotImplementedError

    def sample_out_of_db(self):
        dfa = None
        formula = self.sample_ltl_formula()
        while True:
            formula = self.sample_ltl_formula()
            if not self.is_in_dfa_db(formula):
                dfa = deepcopy(self._get_dfa_from_ltl(formula)) # We might receive a cached dfa so we have to deepcopy
                if dfa:
                    break
        return dfa

    def sample(self):
        formula = self.sample_ltl_formula()
        while not self.is_in_dfa_db(formula):
            formula = self.sample_ltl_formula()
        dfa_from_ltl = self._get_dfa_from_ltl(formula)
        dfa_copy = deepcopy(dfa_from_ltl) # We might receive a cached dfa so we have to deepcopy
        return dfa_copy

# Samples from one of the other samplers at random. The other samplers are sampled by their default args.
class SuperSampler(DFASampler):
    def __init__(self, propositions):
        super().__init__(propositions)
        self.reg_samplers = getRegisteredSamplers(self.propositions)

    def sample_ltl_formula(self):
        return random.choice(self.reg_samplers).sample_ltl_formula()

# This class samples formulas of form (or, op_1, op_2), where op_1 and 2 can be either specified as samplers_ids
# or by default they will be sampled at random via SuperSampler.
class OrSampler(DFASampler):
    def __init__(self, propositions, sampler_ids = ["SuperSampler"]*2):
        super().__init__(propositions)
        self.sampler_ids = sampler_ids

    def sample_ltl_formula(self):
        return ('or', getDFASampler(self.sampler_ids[0], self.propositions).sample_ltl_formula(),
                        getDFASampler(self.sampler_ids[1], self.propositions).sample_ltl_formula())

# This class generates random LTL formulas using the following template:
#   ('until',('not','a'),('and', 'b', ('until',('not','c'),'d')))
# where p1, p2, p3, and p4 are randomly sampled propositions
class DefaultSampler(DFASampler):
    def sample_ltl_formula(self):
        p = random.sample(self.propositions,4)
        return ('until',('not',p[0]),('and', p[1], ('until',('not',p[2]),p[3])))

# This class generates random conjunctions of Until-Tasks.
# Each until tasks has *n* levels, where each level consists
# of avoiding a proposition until reaching another proposition.
#   E.g.,
#      Level 1: ('until',('not','a'),'b')
#      Level 2: ('until',('not','a'),('and', 'b', ('until',('not','c'),'d')))
#      etc...
# The number of until-tasks, their levels, and their propositions are randomly sampled.
# This code is a generalization of the DefaultSampler---which is equivalent to UntilTaskSampler(propositions, 2, 2, 1, 1)
class UntilTaskSampler(DFASampler):
    def __init__(self, propositions, min_levels=1, max_levels=2, min_conjunctions=1 , max_conjunctions=2):
        super().__init__(propositions)
        self.levels       = (int(min_levels), int(max_levels))
        self.conjunctions = (int(min_conjunctions), int(max_conjunctions))
        assert 2*int(max_levels)*int(max_conjunctions) <= len(propositions), "The domain does not have enough propositions!"

    def sample_ltl_formula(self):
        # Sampling a conjuntion of *n_conjs* (not p[0]) Until (p[1]) formulas of *n_levels* levels
        n_conjs = random.randint(*self.conjunctions)
        p = random.sample(self.propositions,2*self.levels[1]*n_conjs)
        ltl = None
        b = 0
        for i in range(n_conjs):
            n_levels = random.randint(*self.levels)
            # Sampling an until task of *n_levels* levels
            until_task = ('until',('not',p[b]),p[b+1])
            b +=2
            for j in range(1,n_levels):
                until_task = ('until',('not',p[b]),('and', p[b+1], until_task))
                b +=2
            # Adding the until task to the conjunction of formulas that the agent have to solve
            if ltl is None: ltl = until_task
            else:           ltl = ('and',until_task,ltl)
        return ltl

# This class generates random LTL formulas that form a sequence of actions.
# @ min_len, max_len: min/max length of the random sequence to generate.
class SequenceSampler(DFASampler):
    def __init__(self, propositions, min_len=2, max_len=4):
        super().__init__(propositions)
        self.min_len = int(min_len)
        self.max_len = int(max_len)

    def sample_ltl_formula(self):
        length = random.randint(self.min_len, self.max_len)
        seq = ""

        while len(seq) < length:
            c = random.choice(self.propositions)
            if len(seq) == 0 or seq[-1] != c:
                seq += c

        ret = self._get_sequence(seq)

        return ret

    def _get_sequence(self, seq):
        if len(seq) == 1:
            return ('eventually',seq)
        return ('eventually',('and', seq[0], self._get_sequence(seq[1:])))

# This generates several sequence tasks which can be accomplished in parallel. 
# e.g. in (eventually (a and eventually c)) and (eventually b)
# the two sequence tasks are "a->c" and "b".
class EventuallySampler(DFASampler):
    def __init__(self, propositions, min_levels = 1, max_levels=4, min_conjunctions=1, max_conjunctions=3):
        super().__init__(propositions)
        assert(len(propositions) >= 3)
        self.conjunctions = (int(min_conjunctions), int(max_conjunctions))
        self.levels = (int(min_levels), int(max_levels))

    def sample_ltl_formula(self):
        conjs = random.randint(*self.conjunctions)
        ltl = None

        for i in range(conjs):
            task = self.sample_sequence()
            if ltl is None:
                ltl = task
            else:
                ltl = ('and',task,ltl)

        return ltl


    def sample_sequence(self):
        length = random.randint(*self.levels)
        seq = []

        last = []
        while len(seq) < length:
            # Randomly replace some propositions with a disjunction to make more complex formulas
            population = [p for p in self.propositions if p not in last]

            if random.random() < 0.25:
                c = random.sample(population, 2)
            else:
                c = random.sample(population, 1)

            seq.append(c)
            last = c

        ret = self._get_sequence(seq)

        return ret

    def _get_sequence(self, seq):
        term = seq[0][0] if len(seq[0]) == 1 else ('or', seq[0][0], seq[0][1])
        if len(seq) == 1:
            return ('eventually',term)
        return ('eventually',('and', term, self._get_sequence(seq[1:])))

class AdversarialEnvSampler(DFASampler):
    def sample_ltl_formula(self):
        p = random.randint(0,1)
        if p == 0:
            return ('eventually', ('and', 'a', ('eventually', 'b')))
        else:
            return ('eventually', ('and', 'a', ('eventually', 'c')))

def getRegisteredSamplers(propositions):
    return [SequenceSampler(propositions),
            UntilTaskSampler(propositions),
            DefaultSampler(propositions),
            EventuallySampler(propositions)]

# The DFASampler factory method that instantiates the proper sampler
# based on the @sampler_id.
def getDFASampler(sampler_id, propositions):
    tokens = ["Default"]
    if (sampler_id != None):
        tokens = sampler_id.split("_")

    # Don't change the order of ifs here otherwise the OR sampler will fail
    if (tokens[0] == "OrSampler"):
        return OrSampler(propositions)
    elif ("_OR_" in sampler_id): # e.g., Sequence_2_4_OR_UntilTask_3_3_1_1
        sampler_ids = sampler_id.split("_OR_")
        return OrSampler(propositions, sampler_ids)
    elif (tokens[0] == "Sequence"):
        return SequenceSampler(propositions, tokens[1], tokens[2])
    elif (tokens[0] == "Until"):
        return UntilTaskSampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4])
    elif (tokens[0] == "SuperSampler"):
        return SuperSampler(propositions)
    elif (tokens[0] == "Adversarial"):
        return AdversarialEnvSampler(propositions)
    elif (tokens[0] == "Eventually"):
        return EventuallySampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4])
    else: # "Default"
        return DefaultSampler(propositions)

def draw(G, path):
    from networkx.drawing.nx_agraph import to_agraph
    A = to_agraph(G) 
    A.layout('dot')                                                                 
    A.draw(path)

if __name__ == '__main__':
    import sys
    props = "abcdefghijkl"
    sampler_id = sys.argv[1]
    sampler = getDFASampler(sampler_id, props)
    draw_path = "sample_dfa.png"
    dfa = sampler.sample()
    draw(dfa, draw_path)
