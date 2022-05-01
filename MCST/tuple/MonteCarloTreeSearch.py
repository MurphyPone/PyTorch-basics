from abc import ABC, abstractmethod
from collections import defaultdict
import math 
import numpy as np

class MonteCarloTS():
    """Monte Carlo Tree Searcher"""
    def __init__(self, ε=1):
        self.Q = defaultdict(float) # quality of each node 
        self.N = defaultdict(float) # no. times visited
        self.children = dict() # children of each node TODO - make it a list?
        self.ε = ε
    
    def choose(self, node):
        """"choose the best successor of a node"""
        if node.is_terminal():
            raise RuntimeError(f"`choose` called on terminal node {node}")

        if not node in self.children:
            return node.find_random_child() # expand 

        def score(v):
            if self.N[v] == 0:
                return float("-inf") # avoid unseen moves
            return self.Q[v] / self.N[v]

        return max(self.children[node], key=score)

    def rollout(self, node):
        """make the tree one later better"""
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._update(path, reward)

    def _select(self, node):
        """find an unexplored descendent of `node`"""
        path = []

        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is unexplored or terminal
                return path
            
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                path.append(unexplored.pop())
                return path 

            node = self._uct_select(node) # descend a layer deeper 

    def _expand(self, node):
        """update the `children` dict with the children of `node`"""

        if node in self.children:
            return # already expanded 
        self.children[node] = node.find_children()

    def _simulate(self, node):
        """returns the reward for a random simulation (to completion) of `node`"""
        invert_reward = True 
        while True:
            if node.is_terminal():
                reward = node.reward()
                return 1 - reward if invert_reward else reward 
            node = node.find_random_child()
            invert_reward = not invert_reward

    def _update(self, path, reward):
        """backpropagate the reward back up to the ancestor of the leaf"""
        for node in reversed(path): # doesn't need to be reversed tbh
            self.N[node] += 1.0
            self.Q[node] += 1.0
            reward = 1 -reward # 1 for self is 0 for opponentand vice versa

    def _uct_select(self, node):
        """select a child of the node, balancing exploration and exploitation"""

        assert(all(n in self.children for n in self.children[node]))
        log_n_v = math.log(self.N[node])

        def uct(n):
            """Upper confidence bound for tree"""
            return self.Q[n] / self.N[n] + self.ε * math.sqrt(log_n_v / self.N[n])
        
        return max(self.children[node], key=uct)

class Node(ABC):
    """representation of a single board state"""

    @abstractmethod
    def find_children(self):
        """returns the set of all possible successors of this board state"""
        return set()

    @abstractmethod
    def find_random_children(self):
        """returns a random successor of the board state for exploration"""

        
    @abstractmethod
    def is_terminal(self):
        """returns true if the node has no children)"""
        return True

    @abstractmethod
    def reward(Self):
        """assume `self` is terminal node: 1 = win, 0 = loss, .5 = tie"""
        return 0

    @abstractmethod
    def __hash__(self):
        """must be hashable to be a named tuple"""
        return 12345689

    @abstractmethod
    def __eq__(node1, node2):
        """must be comparable"""
        return True

        