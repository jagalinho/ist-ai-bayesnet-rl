# Group 27: Joao Galinho (87667) & Filipe Henriques (87653)
from functools import reduce
from itertools import product


class Node:
    def __init__(self, prob, parents=None):
        self.prob = prob
        self.parents = [] if parents is None else parents

    def computeProb(self, evid):
        prob = reduce(lambda curr_prob, parent: curr_prob[evid[parent]], [self.prob] + self.parents)
        return [1 - prob, prob]


class BN:
    def __init__(self, useless, graph):
        self.graph = graph

    def computeJointProb(self, evid):
        return reduce(lambda x, y: x * y, (node.computeProb(evid)[evid[i]] for i, node in enumerate(self.graph)))

    def computePostProb(self, evid):
        prob_true = sum(map(lambda combo: self.computeJointProb(combo),
                    product(*[[0, 1] if isinstance(value, list) else [1 if value == -1 else value] for value in evid])))
        prob_false = sum(map(lambda combo: self.computeJointProb(combo),
                    product(*[[0, 1] if isinstance(value, list) else [0 if value == -1 else value] for value in evid])))
        return prob_true / (prob_true + prob_false)
