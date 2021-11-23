

translate = {"until" : "U",\
             "next" : "X",\
             "eventually": "F",\
             "always": "G",\
             "not" : "!",\
             "and" : "&",\
             "or" : "|",\
             "True" : "true",\
             "False" : "false"}

def formatLTL(formula, props):
    head = formula[0]
    rest = formula[1:]

    if head in ["until", "and", "or"]:

        l = formatLTL(rest[0], props) # build the left subtree

        r = formatLTL(rest[1], props) # build the left subtree
        return "(" + l + " " + translate[head] + " " + r + ")"

    if head in ["next", "eventually", "always", "not"]:

        l = formatLTL(rest[0], props) # build the left subtree
        return "(" + translate[head] + " " + l + ")"

    if formula in ["True", "False"]:
        return translate[formula]

    if formula in props:
        return formula

    assert False, "Format error in ast_builder.ASTBuilder._to_graph()"

    return None
