import numpy as np
import random 
from collections import defaultdict
from games.TicTacToe import TicTacToe

class MonteCarloSearchTreeNode():
    def __init__(self, state, parent=None, parent_action=None):
        """doesn't need state, can just take a game"""

        self.state            = state 
        self.parent           = parent
        self.parent_action    = parent_action
        self.children         = []
        
        self._n_visits        = 0  # number of times this node has been visited
        self._results         = defaultdict(int)
        self._results[1]      = 0  # wins
        self._results[-1]     = 0  # losses
        self._untried_actions = self.untried_actions()
        self.player = "x"

    def untried_actions(self):
        """Returns a list of untried, legal actions"""
        # MCTS 
        self._untried_actions = self.get_legal_actions(self.state)
        return self._untried_actions

    def q(self): 
        """the Q-value of the node (net wins)"""
        # MCTS
        wins, losses = self._results[1], self._results[-1]
        return wins - losses 

    def n(self):
        """getter for _num_visited"""
        # MCTS
        return self._n_visits

    def expand(self):
        """The states which are possible from the present state are all generated and the child_node corresponding 
        to this generated state is returned."""
        # MCTS
        action = self._untried_actions.pop()
        next_state = self.move(self.state, action)
        self.player = self.switch_player()
        # print(TicTacToe(n = 3, board=next_state))
        # print("expanded")
        child = MonteCarloSearchTreeNode(next_state, parent=self, parent_action=action)

        self.children.append(child)
        return child

    def is_terminal(self):
        """returns true/false if the game is over"""
        # Game TODO this might not be needed
        return self.is_game_over(self.state)

    def rollout(self):
        """Light playout s.t. an entire game is played out and the outcome is returned"""
        # MCTS
        current_rollout_state = self.state 

        # while the game hasn't ended
        while not self.is_game_over(current_rollout_state):
            # get legal actions
            legal_actions = self.get_legal_actions(current_rollout_state)
            
            # apply the policy (random selection) to choose an action to take
            action = self.rollout_policy(legal_actions)
            # update the state by taking that action 
            current_rollout_state = self.move(current_rollout_state, action)

        return self.game_result(current_rollout_state)

    def update(self, result):
        """update all the nodes until the root node is reached"""
        """Result will be a win: 1, loss: -1, or tie: -1"""
        # MCTS
        self._n_visits += 1.0
        self._results[result] += 1.0

        if self.parent:
            self.parent.update(result)

    def is_fully_expanded(self):
        """"""
        # MCTS
        return len(self._untried_actions) == 0

    def best_child(self, ε=0.1):
        """ε-gereedy child selection"""
        # MCTS
        choice_weights = [(child.q() / child.n()) + ε * np.sqrt((2 * np.log(self.n()) / child.n())) for child in self.children]

        return self.children[np.argmax(choice_weights)]

    def rollout_policy(self, legal_actions):
        """Choose a random action -> light playout policy"""
        # MCTS
        return random.choice(legal_actions)

    def _tree_policy(self):
        """"""
        #MCTS 
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
        epochs = 100 
        for i in range(epochs):
            v = self._tree_policy()
            reward = v.rollout()
            v.update(reward)

        return self.best_child(ε=0.0)

    def get_legal_actions(self, state):
        """construct a list of all possible actions from current state, return as a list"""
        """X's go first"""
        # Game, should pass all the way down to Game.get_legal_actions(self, state)
        actions = []
        for row in range(len(state)):
            for col in range(len(state[row])):
                if state[row][col] == "-":
                    actions.append((row, col))
        if len(actions) == 0:
            return None 
        return actions

    def fix_spot(self, row, col, player):
        """sets a given players symbol"""
        # Game.is_valid_move should be called before to ensure only legal moves are attempted otherwise RTE
        assert(row < len(self.state) and row >=0)
        assert(col < len(self.state) and col >= 0)
        if self.state[row][col] == "-":
            self.state[row][col] = player 
            return True
        else: 
            return False

    def check_rows(self, state):
        # GAME 
        """checks if we have a row winner"""
        for row in state:
            if len(set(row)) == 1:
                return row[0]
        return 0

    def check_diagonals(self, state):
        # GAME 
        """checks if we have a diagonal winner"""
        if len(set([state[i][i] for i in range(len(state))])) == 1:
            return state[0][0]
        if len(set([state[i][len(state)-i-1] for i in range(len(state))])) == 1:
            return state[0][len(state)-1]
        return 0

    def is_player_win(self, board, player):
        """checks if a given player wins"""
        #transposition to check rows, then columns
        for newBoard in [board, np.transpose(board)]:
            result = self.check_rows(newBoard)
            if result:
                return result == player
        return self.check_diagonals(board) == player

    def is_game_over(self, state):
        """return true or false if there is a winner"""
        # GAME -> under what conditions is a game over, W, L, draw 
        return self.is_player_win(state, "x") or self.is_player_win(state, "o") or self.get_legal_actions(state) is None

    def game_result(self, state):
        # GAME - should only be called if a game is over s.t. a 0 is necessarily a draw, and not an ongoing game
        if self.is_player_win(state, self.player):
            return 1
        elif self.is_player_win(state, self.switch_player()):
            return -1
        else:
            return 0

    def switch_player(self):
        # Idk if these even needs to manifest out here 
        if self.player == "x":
            # print(f"switching player from x to o")
            return "o"
        else:
            # print(f"switching player from o to x")
            return "x"

    def move(self, state, action):
        """For a normalTic Tac Toe game, it can be a 3 by 3
           array with all the elements of array
           being 0 initially. 0 means the board 
           position is empty. If you place x in
           row 2 column 3, then it would be some 
           thing like board[2][3] = 1, where 1
           represents that x is placed. Returns 
           the new state after making a move."""

        # TODO going to want to move the legality of the move logic into the game itself, 
        # this should just call move and handle exceptions which shouldn't manifest to begin with
        actions = self.get_legal_actions(state)
        assert(action in actions)
        assert(actions is not None) 
        if action in actions:
            row, col = action
            if self.fix_spot(row, col, self.player): # TODO check if succceeded
                self.player = self.switch_player()
            else: 
                raise(Exception("made an illegal move somehwo"))
                
        # print(TicTacToe(n = 3, board=self.state))
        return self.state


if __name__ == "__main__":
    initial_state = TicTacToe(3).board
    root = MonteCarloSearchTreeNode(state = initial_state)
    print(root.player)
    import pdb

    # pdb.set_trace()
    selected_node = root.best_action()
    print(TicTacToe(n = 3, board=selected_node.state))
    print(selected_node.game_result(selected_node.state))

