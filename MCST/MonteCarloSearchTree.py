import numpy as np 
import random
from collections import defaultdict 
from games.TicTacToe import TicTacToe

# TODO takes a Game?
class MonteCarloSearchTreeNode():
    def __init__(self, game, parent=None, parent_action=None):
        """doesn't need state, can just take the game and extract is """
        self.game             = game
        self.state            = game.get_state()
        self.parent           = parent 
        self.parent_action    = parent_action
        self.children         = []

        self._n_visits        = 0 # number of times this node has been visited
        self._results         = defaultdict(int)
        self._results[1]      = 0 # num wins
        self._results[0]      = 0 # num ties
        self._results[-1]     = 0 # num losses
        self._untried_actions = self.untried_actions()

    # MCTS specific functions
    def untried_actions(self):
        """returns a list of untried, legal actions"""
        self._untried_actions = self.get_legal_actions(self.game.get_state())
        return self._untried_actions

    def Q(self):
        """the Q-value of the nodes (net wins)"""
        wins, draws, losses = self._results[1], self._results[0], self._results[-1]
        return wins + draws - losses 

    def N(self):
        """getter for _n_visits"""
        return self._n_visits

    def expand(self):
        """
        states which are possible from the present state are all generated and 
        the child_node corresponding to this generated state is returned
        """
        # MCTS
        action = self._untried_actions.pop()
        next_state = self.move(self.state, action)

        # print(TicTacToe(board=next_state))
        print("expanded")

        self.game.set_state(next_state)  # TODO I don't think I want to overwrite the global game state
        child = MonteCarloSearchTreeNode(self.game, parent=self, parent_action=action)
        self.children.append(child)

        return child

    def rollout(self):
        """Light playout s.t. an entire game is played out and the outcome is returned"""
        current_state = self.game.get_state()

        # while the game hasn't ended
        while not self.is_game_over(current_state):
            # get the legal actions
            legal_actions = self.get_legal_actions(current_state)
            
            # apply the policy (random selection) to choose an action  
            action = self.rollout_policy(legal_actions)
            
            # update the state by taking that action
            current_state = self.move(current_state, action) # TODO make this make sense?

        return self.game_result(current_state)

    def update(self, result):
        """update all the nodes until the root node is reached"""
        """Result will be a win: 1, loss: -1, or tie: 0.25"""
        # MCTS
        self._n_visits += 1.0
        self._results[result] += 1.0

        if self.parent:
            self.parent.update(result)

    def is_fully_expanded(self):
        """checks if there's any more actions that can be taken"""
        # MCTS
        return len(self._untried_actions) == 0

    def best_child(self, ε=0.1):
        """ε-greedy selection"""
        choice_weights = [(child.Q() / child.N()) + ε * np.sqrt((2 * np.log(self.N()) / child.N())) for child in self.children]
        
        return self.children[np.argmax(choice_weights)]

    def rollout_policy(self, legal_actions):
        """Choose a random action from the light playout policy"""
        # legal_actions
        return random.choice(legal_actions)

    def _tree_policy(self):
        """"""
        # MCTS
        current_node = self

        while not current_node.is_terminal():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else: 
                current_node = current_node.best_child()
        
        return current_node

    def best_action(self):
        """choose the action which maximizes reward over time (zero exploration)"""
        # MCTS
        epochs = 1000

        for _ in range(epochs):
            v = self._tree_policy()
            reward = v.rollout()
            v.update(reward)

        return self.best_child(ε=0.0)

    def get_legal_actions(self, state):
        """Construct a list of all possible actions from the given state"""
        return self.game.get_legal_actions(state)

    def is_terminal(self):
        """returns true/false if the game is over"""
        # MCTS
        return self.game.is_game_over(self.state)

    def is_game_over(self, state):
        return self.game.is_game_over(state)

    def game_result(self, state):
        return self.game.get_result(state, self.game.player)

    def move(self, state, action):
        legal_actions = self.get_legal_actions(state)
        if action in legal_actions:
            self.game.fix_spot(action, self.game.player)
            
        print(TicTacToe(board=self.state))
        return self.game.get_state()
            
if __name__ == "__main__":
    ttt = TicTacToe(3)
    root = MonteCarloSearchTreeNode(game=ttt)
    to_move = str(root.game.player)
    import pdb

    # pdb.set_trace()
    selected_node = root.best_action()
    print(f"{to_move} game result: ", selected_node.game_result(selected_node.game.get_state()))


