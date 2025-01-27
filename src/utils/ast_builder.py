import ring
import numpy as np

import dgl
import networkx as nx
from sklearn.preprocessing import OneHotEncoder
import time
edge_types = {k:v for (v, k) in enumerate(["self", "arg", "arg1", "arg2"])}
DGL5_COMPAT = True

"""
A class that can take an LTL formula and generate the Abstract Syntax Tree (AST) of it. This
code can generate trees in either Networkx or DGL formats. And uses caching to remember recently
generated trees.
"""
class ASTBuilder(object):
    def __init__(self, propositions):
        super(ASTBuilder, self).__init__()
        self.props = propositions

        terminals = ['True', 'False'] + self.props
        ## Pad terminals with dummy propositions to get a fixed encoding size
        for i in range(15 - len(terminals)):
            terminals.append("dummy_"+str(i))

        self._enc = OneHotEncoder(handle_unknown='ignore', dtype=np.int)
        self._enc.fit([['next'], ['until'], ['and'], ['or'], ['eventually'],
            ['always'], ['not']] + np.array(terminals).reshape((-1, 1)).tolist())

    # To make the caching work.
    def __ring_key__(self):
        return "ASTBuilder"

    @ring.lru(maxsize=30000)
    def __call__(self, formula, library="dgl"):
        #start = time.time()
        nxg = self._to_graph(formula)
        nx.set_node_attributes(nxg, [0.], "is_root")
        nxg.nodes[0]["is_root"] = [1.]
        if (library == "networkx"): return nxg

        """print(formula)
        print("Number of nodes:", len(nxg.nodes))
        print("Number of edges:", len(nxg.edges))
        for i in nxg.nodes:
            print(nxg.nodes[i]["feat"].shape, type(nxg.nodes[i]["feat"]), nxg.nodes[i]["feat"], type(nxg.nodes[i]["feat"][0][0]))
            print(nxg.nodes[i]["is_root"], type(nxg.nodes[i]["is_root"]))
        for i in nxg.edges:
            print(nxg.edges[i]["type"], type(nxg.edges[i]["type"]))"""

        # convert the Networkx graph to dgl graph and pass the 'feat' attribute
        if DGL5_COMPAT:
            g = dgl.from_networkx(nxg, node_attrs=["feat", "is_root"], edge_attrs=["type"]) # dgl does not support string attributes (i.e., token)
        else:
            g = dgl.DGLGraph()
            g.from_networkx(nxg, node_attrs=["feat", "is_root"], edge_attrs=["type"]) # dgl does not support string attributes (i.e., token)
        return g

    def _one_hot(self, token):
        return self._enc.transform([[token]])[0][0].toarray()


    def _get_edge_type(self, operator, parameter_num=None):
        operator = operator.lower()
        if (operator in ["next", "until", "and", "or"]):
            # Uncomment to make "and" and "or" permutation invariant
            # parameter_num = 1 if operator in ["and", "or"] else operator

            return edge_types[operator + f"_{parameter_num}"]

        return edge_types[operator]

    # A helper function that recursively builds up the AST of the LTL formula
    @ring.lru(maxsize=60000) # Caching the formula->tree pairs in a Last Recently Used fashion
    def _to_graph(self, formula, shift=0):
        head = formula[0]
        rest = formula[1:]
        nxg  = nx.DiGraph()

        if head in ["until", "and", "or"]:
            nxg.add_node(shift, feat=self._one_hot(head), token=head)
            nxg.add_edge(shift, shift, type=self._get_edge_type("self"))

            l = self._to_graph(rest[0], shift+1) # build the left subtree
            nxg = nx.compose(nxg, l) # combine the left subtree with the current tree
            nxg.add_edge(shift+1, shift, type=self._get_edge_type("arg1")) # connect the current node to the root of the left subtree

            index = nxg.number_of_nodes()
            r = self._to_graph(rest[1], shift+index) # build the left subtree
            nxg = nx.compose(nxg, r) # combine the left subtree with the current tree
            nxg.add_edge(shift+index, shift, type=self._get_edge_type("arg2"))
            # if head in ["next", "until"]:
            #     nxg.add_edge(1, index, type=self.ASSYM_EDGE) # impose order on the operands of an assymetric operator

            return nxg

        if head in ["next", "eventually", "always", "not"]:
            nxg.add_node(shift, feat=self._one_hot(head), token=head)
            nxg.add_edge(shift, shift, type=self._get_edge_type("self"))

            l = self._to_graph(rest[0], shift+1) # build the left subtree
            nxg = nx.compose(nxg, l) # combine the left subtree with the current tree
            nxg.add_edge(shift+1, shift, type=self._get_edge_type("arg")) # connect the current node to the root of the left subtree

            return nxg

        if formula in ["True", "False"]:
            nxg.add_node(shift, feat=self.vocab._one_hot(formula), token=formula)
            nxg.add_edge(shift, shift, type=self._get_edge_type("self"))

            return nxg

        if formula in self.props:
            nxg.add_node(shift, feat=self._one_hot(formula.replace("'",'')), token=formula)
            nxg.add_edge(shift, shift, type=self._get_edge_type("self"))

            return nxg


        assert False, "Format error in ast_builder.ASTBuilder._to_graph()"

        return None

def draw(G, formula):
    from networkx.drawing.nx_agraph import graphviz_layout
    import matplotlib.pyplot as plt

    # colors = ["black", "red"]
    # edge_color = [colors[i] for i in nx.get_edge_attributes(G,'type').values()]

    plt.title(formula)
    pos=graphviz_layout(G, prog='dot')
    labels = nx.get_node_attributes(G,'token')
    nx.draw(G, pos, with_labels=True, arrows=True, labels=labels, node_shape='s', edgelist=list(nx.get_edge_attributes(G,'type')), node_size=500, node_color="white") #edge_color=edge_color
    plt.show()

"""
A simple test to check if the ASTBuilder works fine. We do a preorder DFS traversal of the resulting
tree and convert it to a simplified formula and compare the result with the simplified version of the
original formula. They should match.
"""
if __name__ == '__main__':
    import re
    import sys
    import itertools
    import matplotlib.pyplot as plt

    #sys.path.insert(0, '../../')
    sys.path.append('.')
    sys.path.append('..')
    from ltl_samplers import getLTLSampler

    for sampler_id, _ in itertools.product(["Default", "Sequence_2_20"], range(20)):
        props = "abcdefghijklmnopqrst"
        sampler = getLTLSampler(sampler_id, props)
        builder = ASTBuilder(list(set(list(props))))
        formula = sampler.sample()
        tree = builder(formula, library="networkx")
        pre = list(nx.dfs_preorder_nodes(tree, source=0))
        draw(tree, formula)
        u_tree = tree.to_undirected()
        pre = list(nx.dfs_preorder_nodes(u_tree, source=0))

        original = re.sub('[,\')(]', '', str(formula))
        observed = " ".join([u_tree.nodes[i]["token"] for i in pre])

        assert original == observed, f"Test Faield: Expected: {original}, Got: {observed}"

    print("Test Passed!")
