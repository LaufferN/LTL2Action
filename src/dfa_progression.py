import networkx as nx
from pysat.solvers import Solver

def _is_sink_state(node, nxg):
    for edge in nxg.edges:
        if node == edge[0] and node != edge[1]: # If there is an outgoing edge to another node, then it is not an accepting state
            return False
    return True

# Returns (dfa_reward, progression_info, dfa_done)
def progress_and_clean(dfa, truth_assignment):
    if len(truth_assignment) == 1:
        accepting_states = []
        rejecting_states = []
        is_mark_seen = False
        for node in dfa.nodes:
            if node == "init":
                is_mark_seen = True
                continue
            if _is_sink_state(node, dfa):
                if not is_mark_seen:
                    accepting_states.append(node)
                else:
                    rejecting_states.append(node)
        for e in dfa.edges:
            if dfa.nodes[e[0]]["current_state"] == True:
                guard = dfa.edges[e]["label"]
                guard = guard.replace(" ", "").replace("(", "").replace(")", "").replace("\"", "")
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
                for model in models:
                    temp = []
                    for a in model:
                        if a > 0:
                            temp.append(seen_atoms[abs(a) - 1])
                    if len(temp) == 1 and temp[0] == truth_assignment:
                        nx.set_node_attributes(dfa, False, 'current_state')
                        dfa.nodes[e[1]]["current_state"] = True
                        if e[1] in accepting_states:
                            return 1.0, 1.0, True
                        if e[1] in rejecting_states:
                            return -1.0, -1.0, True
                        return 0.0, 1.0, False
    return 0.0, 0.0, False
