import numpy as np 
import random
from collections import defaultdict 
from games.TicTacToe import TicTacToe

# TODO takes a Game?
class MonteCarloSearchTreeNode():
    def __init__(self, game, state, parent=None, parent_action=None):
        self.game             = game
        self.state            = state # TODO might not be necessary 
        self.parent           = parent 
        self.parent_action    = parent_action
        self.children         = []
        self._num_visits      = 0 # TODO: N ?
        self._results         = defaultdict(int) # TODO: Q ?
        self._results[1]      = 0
        self._results[-1]     = 0 
        self._untried_actions = self.untried_actions()

    def untried_actions(self):
        self._untried_actions = self.get_legal_actions()

        return self._untried_actions

    def Q(self):
        wins, losses = self._results[1], self._results[-1]
        return wins - losses 

    def N(self):
        return self._num_visits

    def expand(self):
        action = self._untried_actions.pop()
        next_state = self.move(self.state, action)
        child = MonteCarloSearchTreeNode(next_state, parent=self, parent_action=action)
        self.children.append(child)

        return child

    def is_terminal(self):
        # TODO how to make this extend Game interface
        return self.is_game_over(self.state)

    def rollout(self):
        current_state = self.state 

        while not self.is_game_over(current_state):
            legal_actions = self.get_legal_actions(current_state)
            action = self.rollout_policy(legal_actions)
            current_state = current_state.move(action) # TODO make this make sense?

        return self.game_result(current_state)

    def update(self, result):
        self._num_visits += 1.0
        self._results[result] += 1.0

        if self.parent:
            self.parent.update(result)

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def best_child(self, c=0.1):
        weights = [(child.Q() / child.N()) + c * np.sqrt((2 *np.log(self.N()) / child.N())) for child in self.children]
        
        return self.children(np.argmax(weights))

    def rollout_policy(self, legal_actions):
        return random.choice(legal_actions)

    def _tree_policy(self):
        current_node = self

        while not current_node.is_terminal():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else: 
                current_node = current_node.best_child()
        
        return current_node

    def best_action(self):
        epochs = 100 

        for i in range(epochs):

            v = self._tree_policy()
            r = v.rollout()
            v.update(r)

        return self.best_child(c=0.0)

    def get_legal_actions(self):
        return self.game.get_legal_actions()

    def is_game_over(self):
        pass 

    def game_result(self):
        pass

    def move(self, state, action):
        pass 
            
if __name__ == "__main__":
    ttt = TicTacToe(3)
    root = MonteCarloSearchTreeNode(game=ttt, state=ttt.board)


