import networkx as nx
from copy import deepcopy

def prune_unreachable_nodes(nxg, current_node):
    reachable_nodes = nx.single_source_shortest_path(nxg, current_node).keys()
    all_nodes = deepcopy(nxg.nodes)
    for node in all_nodes:
        if node not in reachable_nodes:
            nxg.remove_node(node)


# Returns (dfa_reward, progression_info, dfa_done)
def progress_and_clean(nxg, true_atom, propositions):
    if len(true_atom) == 1 or isinstance(true_atom, str): # if list of length one or plain string atom
        current_node = list(filter(lambda node: nxg.nodes[node]["is_root"] == [1.0], nxg.nodes))[0] # This should never throw an exception
        next_temp_nodes = list(map(lambda edge: edge[1], filter(lambda edge: edge[0] == current_node and edge[1] != current_node, nxg.edges)))
        next_temp_nodes_outgoing_edges = list(filter(lambda edge: edge[0] in next_temp_nodes and edge[1] not in next_temp_nodes, nxg.edges))
        for edge in next_temp_nodes_outgoing_edges:
            true_atom_index = propositions.index(true_atom)
            if nxg.nodes[edge[0]]["feat"][0][true_atom_index] == 1.0:
                nxg.nodes[current_node]["is_root"] = [0.0]
                nxg.nodes[current_node]["feat"][0][-5] = 0.0
                nxg.nodes[edge[1]]["is_root"] = [1.0]
                nxg.nodes[edge[1]]["feat"][0][-5] = 1.0
                prune_unreachable_nodes(nxg, edge[1])
                if nxg.nodes[edge[1]]["feat"][0][-2] == 1.0:
                    return 1.0, 1.0, True
                if nxg.nodes[edge[1]]["feat"][0][-1] == 1.0:
                    return -1.0, -1.0, True
                return 0.0, 1.0, False
    return 0.0, 0.0, False
