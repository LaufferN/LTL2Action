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
import dill
import numpy as np
import networkx as nx
from copy import deepcopy
from pysat.solvers import Solver
from ltlf2dfa.parser.ltlf import LTLfParser
from pythomata.impl.simple import SimpleNFA as NFA 
from scipy.special import softmax
import dfa
try:
    from utils.format_ltl import formatLTL
except ModuleNotFoundError:
    from format_ltl import formatLTL

from utils.parameters import dfa_db_path, TIMEOUT_SECONDS, FEATURE_SIZE

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
                if a > 0:
                    atom = seen_atoms[abs(a) - 1]
                    temp[self.propositions.index(atom)] = 1.0
            embeddings.append(temp)
        return embeddings

    @ring.lru(maxsize=100000)
    def _get_onehot_guard_embeddings(self, guard):
        is_there_onehot = False
        is_there_all_zero = False
        onehot_embedding = [0.0] * FEATURE_SIZE
        onehot_embedding[-3] = 1.0 # Since it will be a temp node
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

    def dfa2nxg(self, mvc_dfa, minimize=True):
        """ converts a mvc format dfa into a networkx dfa """

        if minimize:
            mvc_dfa = dfa.utils.minimize(mvc_dfa)

        dfa_dict, init_node = dfa.dfa2dict(mvc_dfa)
        init_node = str(init_node)

        nxg = nx.DiGraph()

        accepting_states = []
        for start, (accepting, transitions) in dfa_dict.items():
            # pydot_graph.add_node(nodes[start])
            start = str(start)
            nxg.add_node(start)
            if accepting:
                accepting_states.append(start)
            for action, end in transitions.items():
                if nxg.has_edge(start, str(end)):
                    existing_label = nxg.get_edge_data(start, str(end))['label']
                    nxg.add_edge(start, str(end), label='{} | {}'.format(existing_label, action))
                    # print('{} | {}'.format(existing_label, action))
                else:
                    nxg.add_edge(start, str(end), label=action)

        return init_node, accepting_states, nxg


    def _format(self, init_node, accepting_states, nxg):
        print('init', init_node)
        print('accepting', accepting_states)
        rejecting_states = []
        for node in nxg.nodes:
            if self._is_sink_state(node, nxg) and node not in accepting_states:
                rejecting_states.append(node)

        for node in nxg.nodes:
            nxg.nodes[node]["feat"] = np.array([[0.0] * FEATURE_SIZE])
            nxg.nodes[node]["feat"][0][-4] = 1.0
            if node in accepting_states:
                nxg.nodes[node]["feat"][0][-2] = 1.0
            if node in rejecting_states:
                nxg.nodes[node]["feat"][0][-1] = 1.0

        nxg.nodes[init_node]["feat"][0][-5] = 1.0

        edges = deepcopy(nxg.edges)

        new_node_name_base_str = "temp_"
        new_node_name_counter = 0

        for e in edges:
            guard = nxg.edges[e]["label"]
            print(e, guard)
            nxg.remove_edge(*e)
            if e[0] == e[1]:
                continue # We define self loops below
            onehot_embedding = self._get_onehot_guard_embeddings(guard) # It is ok if we receive a cached embeddding since we do not modify it
            if len(onehot_embedding) == 0:
                continue
            new_node_name = new_node_name_base_str + str(new_node_name_counter)
            new_node_name_counter += 1
            nxg.add_node(new_node_name, feat=np.array(onehot_embedding))
            nxg.add_edge(e[0], new_node_name, type=1)
            nxg.add_edge(new_node_name, e[1], type=2)

        nx.set_node_attributes(nxg, [0.0], "is_root")
        nxg.nodes[init_node]["is_root"] = [1.0] # is_root means current state

        for node in nxg.nodes:
            nxg.add_edge(node, node, type=0)


        return nxg

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
            onehot_embedding = self._get_onehot_guard_embeddings(guard) # It is ok if we receive a cached embeddding since we do not modify it
            if len(onehot_embedding) == 0:
                continue
            new_node_name = new_node_name_base_str + str(new_node_name_counter)
            new_node_name_counter += 1
            nxg.add_node(new_node_name, feat=np.array(onehot_embedding))
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
        dfa = self.sample_dfa_formula()
        init_node, accepting_states, nxg = self.dfa2nxg(dfa)
        return deepcopy(self._format(init_node,accepting_states,nxg))

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

def _chain(xs, alphabet):
    def transition(s, c):
        if len(s) == 0:
            return s
        head, *tail = s
        if c in head:
            return tuple(tail)
        return s

    return dfa.DFA(
        start=tuple(xs),
        inputs=alphabet,
        outputs={False, True},
        label=lambda s: s == tuple(),
        transition=transition,
    )

def _accept_reach_avoid(sub_dfa, reach, avoid, alphabet):
    # assert not (reach & avoid)

    # state is (reach-avoid state, sub_dfa state)

    def transition(s, c):
        reach_avoid_state = s[0]
        sub_dfa_state = s[1]

        if reach_avoid_state == 0b10:
            if c in avoid:
                # reset the sub dfa and go back to the initial state
                next_sub_dfa_state = sub_dfa.start
                return (0b11, next_sub_dfa_state)
            next_sub_dfa_state = sub_dfa._transition(sub_dfa_state, c)
            return (reach_avoid_state, next_sub_dfa_state)
        elif reach_avoid_state == 0b01:
            return (reach_avoid_state, sub_dfa_state)
        elif c in reach:
            next_sub_dfa_state = sub_dfa._transition(sub_dfa_state, c)
            return (0b10, next_sub_dfa_state)
        elif c in avoid:
            return (0b01, sub_dfa_state)
        return (reach_avoid_state, sub_dfa_state)

        # if reach_avoid_state == 0b10:
        #     next_sub_dfa_state = sub_dfa._transition(sub_dfa_state, c)
        #     return (reach_avoid_state, next_sub_dfa_state)
        # elif reach_avoid_state == 0b01:
        #     return (reach_avoid_state, sub_dfa_state)
        # elif c in reach:
        #     next_sub_dfa_state = sub_dfa._transition(sub_dfa_state, c)
        #     return (0b10, next_sub_dfa_state)
        # elif c in avoid:
        #     return (0b01, sub_dfa_state)
        # return (reach_avoid_state, sub_dfa_state)

        # # if we've in the rejecting states, stay here
        # if reach_avoid_state == 0b01:
        #     return (reach_avoid_state, next_sub_dfa_state)
        # # if we hit an avoid prop, go to the rejecting state
        # elif c in avoid:
        #     return (0b01, next_sub_dfa_state)
        # # if we hit a reach prop, go to the accepting state
        # elif c in reach:
        #     return (0b10, next_sub_dfa_state)
        # # otherwise, only progress the sub_dfa
        # return (reach_avoid_state, next_sub_dfa_state)


    return dfa.DFA(
        start=(0b00, sub_dfa.start),
        inputs=alphabet,
        outputs={True, False},
        label=lambda s: s[0] == 0b10 and sub_dfa._label(s[1]),
        transition=transition,
    )

def avoidance(reach_avoids: list[tuple[str, str]], alphabet: set[str]) -> NFA:
    n = len(reach_avoids)
    states = {f'q{i}' for i in range(n+1)} | {'fail'}
    aps = set.union(*map(set, zip(*reach_avoids)))
    if alphabet is None:
        alphabet = aps
    assert aps <= alphabet
    transitions = {
        f'q{n}': {c: {f'q{n}'} for c in alphabet},
        'fail': {c: {'fail'} for c in alphabet},
    }
    for i, (reach, avoid) in enumerate(reach_avoids):
        assert reach != avoid
        state = f'q{i}'

        edges = {}
        for char in alphabet - {reach, avoid}:
            edges[char] = {state}
        edges[avoid] = {'fail'}
        edges[reach] = {f'q{i+1}', state}
        transitions[state] = edges

    return NFA(
        states=states,
        alphabet=alphabet,
        transition_function=transitions,
        initial_state='q0',
        accepting_states={f'q{n}'}
    )

def _reach_avoid(reach, avoid, alphabet):
    # assert not (reach & avoid)

    def transition(s, c):
        if s != 0:
            return s
        if c in reach:
            return 0b10
        elif c in avoid:
            return 0b01
        return s

    return dfa.DFA(
        start=0b00,
        inputs=alphabet,
        outputs={True, False},
        label=lambda s: s == 0b10,
        transition=transition,
    )

class UniversalSampler(DFASampler):
    def __init__(self, propositions, temp='0.5'):
        super().__init__(propositions)
        with open("dfas/enumerated_gridworld_dfas.pickle", 'rb') as f:
            enumerated_dfas = dill.load(f)

        temp = float(temp)
        augmented_props = list(propositions)
        augmented_props.remove('white')

        self.enumerated_dfas = [dfa.DFA.from_int(dfa_int, inputs=augmented_props) for dfa_int in enumerated_dfas]
        sizes = np.array([len(bin(dfa_int)) for dfa_int in enumerated_dfas])
        # print(np.average(sizes))
        self.weights = softmax(-sizes * temp)

        # self.no_sink_dfas = []
        # self.sink_dfas = []
        # for dfaa in self.enumerated_dfas:
        #     _, accepting_nodes, nxg = self.dfa2nxg(dfaa)
        #     has_sink = False
        #     for node in nxg.nodes:
        #         if node in accepting_nodes:
        #             continue
        #         for edge in nxg.edges:
        #             if node == edge[0] and node != edge[1]: # If there is an outgoing edge to another node, then it is not an accepting state
        #                 break
        #         else:
        #             has_sink = True
        #             break
        #     if has_sink:
        #         self.sink_dfas.append(dfaa)
        #     else:
        #         self.no_sink_dfas.append(dfaa)

        # print("~~~~~~~~~~~~~~~~~OUTPUTING NOW~~~~~~~~~~~~~~")
        # print(len(self.no_sink_dfas))
        # print(len(self.sink_dfas))

    def sample_dfa_formula(self):
        # random_size = np.random.choice(self.sizes, p=self.weights)
        # return deepcopy(np.random.choice(self.enumerated_dfas_dict[random_size]))

        return deepcopy(np.random.choice(self.enumerated_dfas, p=self.weights))

class FixedGridworldSampler(DFASampler):
    def __init__(self, propositions):
        super().__init__(propositions)

    def transition(self, s, c):
        if s == 0:
            if c == 'red':
                s = 1 # fail
            elif c == 'blue':
                s = 2 # got wet
            elif c == 'yellow':
                s = 3 # success
        elif s == 2:
            if c == 'red' or c == 'yellow':
                s = 1 # fail
            elif c == 'green':
                s = 0 # back to start
        elif s == 3:
            if c == 'blue':
                s = 2 # got wet
            if c == 'red':
                s = 1 # fail

        return s


    def sample_dfa_formula(self):
        fixed_dfa = dfa.DFA(start=0,
                            inputs={'blue', 'green', 'red', 'yellow', 'white'},
                            outputs={False, True},
                            label=lambda s: s == 3,
                            transition=self.transition)

        return fixed_dfa


class UntilTaskSampler(DFASampler):
    def __init__(self, propositions, min_levels=1, max_levels=2, min_conjunctions=1 , max_conjunctions=2):
        super().__init__(propositions)
        self.levels       = (int(min_levels), int(max_levels))
        self.conjunctions = (int(min_conjunctions), int(max_conjunctions))
        assert 2*int(max_levels)*int(max_conjunctions) <= len(propositions), "The domain does not have enough propositions!"

    def sample_dfa_formula(self):

        n_conjs = random.randint(*self.conjunctions)
        p = random.sample(self.propositions,2*self.levels[1]*n_conjs)
        dfa_task = None
        b = 0
        for i in range(n_conjs):
            n_levels = random.randint(*self.levels)
            # Sampling an until task of *n_levels* levels
            avoidance_list = [(p[b+1], p[b])]
            b +=2
            for j in range(1,n_levels):
                avoidance_list.append((p[b+1], p[b]))
                b +=2
            until_task_nfa = avoidance(avoidance_list[::-1], set(self.propositions))
            until_task_dfa = until_task_nfa.determinize().trim().minimize()
            until_task_mvc = dfa.DFA(
                start=until_task_dfa.initial_state,
                inputs=self.propositions,
                label=lambda s: s in until_task_dfa.accepting_states,
                transition=lambda s, c: until_task_dfa.transition_function[s][c],
            ).normalize()
 
            # Adding the until task to the conjunction of formulas that the agent have to solve
            if dfa_task is None: dfa_task = until_task_mvc
            else:                dfa_task = until_task_mvc & dfa_task
        return dfa_task

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


class UntilLTLTaskSampler(DFASampler):
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
class EventuallyLTLSampler(DFASampler):
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

### DFA version of the EventuallySampler
class EventuallySampler(DFASampler):
    def __init__(self, propositions, min_levels = 1, max_levels=4, min_conjunctions=1, max_conjunctions=3):
        super().__init__(propositions)
        assert(len(propositions) >= 3)
        self.conjunctions = (int(min_conjunctions), int(max_conjunctions))
        self.levels = (int(min_levels), int(max_levels))

    def sample_dfa_formula(self):
        conjs = random.randint(*self.conjunctions)
        dfa = None

        for i in range(conjs):
            task = self.sample_sequence()
            if dfa is None:
                dfa = task
            else:
                dfa = task & dfa

        return dfa


    def sample_sequence(self):
        length = random.randint(*self.levels)
        seq = []

        last = []
        while len(seq) < length:
            # Randomly replace some propositions with a disjunction to make more complex formulas
            population = [p for p in self.propositions if p not in last]

            if random.random() < 0.25:
                c = tuple(random.sample(population, 2))
            else:
                c = tuple(random.sample(population, 1))

            seq.append(c)
            last = c

        ret = _chain(seq, self.propositions)

        return ret

    # def _get_sequence(self, seq):
    #     term = seq[0][0] if len(seq[0]) == 1 else ('or', seq[0][0], seq[0][1])
    #     if len(seq) == 1:
    #         return ('eventually',term)
    #     return ('eventually',('and', term, self._get_sequence(seq[1:])))

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
    elif (tokens[0] == "Universal"):
        return UniversalSampler(propositions, tokens[1])
    elif (tokens[0] == "FixedGridworld"):
        return FixedGridworldSampler(propositions)
    else: # "Default"
        return DefaultSampler(propositions)

def draw(G, path):
    from networkx.drawing.nx_agraph import to_agraph
    A = to_agraph(G) 
    A.layout('dot')                                                                 
    A.draw(path)

if __name__ == '__main__':
    import sys
    props = ["yellow", "green", "blue", "red"]
    sampler_id = sys.argv[1]
    sampler = getDFASampler(sampler_id, props)
    draw_path = "sample_dfa.png"
    dfa = sampler.sample()
    draw(dfa, draw_path)

