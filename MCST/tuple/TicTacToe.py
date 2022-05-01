from collections import namedtuple # bc immutable?
import random
from MonteCarloTreeSearch import MonteCarloTS, Node

_TTB = namedtuple("TicTacToeBoard", "tup turn winner terminal")

class TicTacToeBoard(_TTB, Node):
    def find_children(board):
        if board.terminal: # if the game is finished, then no moves can be made 
            return set()
        # o.w. make a move in an empty spot
        return {board.make_move(i) for i, value in enumerate(board.tup) if value is None}

    def find_random_child(board):
        if board.terminal:
            return None # if the game is finished, no moves can be made
        actions = [i for i, value in enumerate(board.tup) if value is None]
        return board.make_move(random.choice(actions))

    def reward(board):
        if not board.terminal:
            raise RuntimeError(f"reward called on non-terminal board {board}")

        if board.winner is board.turn:
            # it's your turn and you've already won, should be impossible
            raise RuntimeError(f"reward called on unreachable board {board}")

        if board.turn is (not board.winner):
            return 0 # opponent won
        
        if board.winner is None:
            return 0.5 # tie 

        # the winner is neither True, False, or None
        raise RuntimeError(f"board has unknown winner type {board.winner}")

    def is_terminal(board):
        return board.terminal

    def make_move(board, index):
        tup = board.tup[:index] + (board.turn,) + board.tup[index + 1:]
        turn = not board.turn 
        winner = _find_winner(tup)

        is_terminal = (winner is not None) or not any(v is None for v in tup)   
        return TicTacToeBoard(tup, turn, winner, is_terminal)

    def pretty(board):
        to_char = lambda v: ("X" if v is True else ("O" if v is False else "-"))
        rows = [
            [to_char(board.tup[3 * row + col]) for col in range(3)] for row in range(3)
        ]

        return (
            "\n  1 2 3\n"
            + "\n".join(str(i + 1) + " " + " ".join(row) for i, row in enumerate(rows))
            + "\n"
        )

def play_game():
    tree = MonteCarloTS()
    board = new_tic_tac_toe_board()
    print(board.pretty())

    while True:
        action = input("enter <row,col>: ")
        row, col = map(int, action.split(","))

        index = 3 * (row - 1) + (col - 1)

        if board.tup[index] is not None:
            raise RuntimeError("Invalid move")

        board = board.make_move(index)
        print(board.pretty())
        if board.terminal:
            break 

        epochs = 50
        for _ in range(epochs):
            tree.rollout(board)

        board = tree.choose(board)
        print(board.pretty())
        if board.terminal:
            break
    
def _winning_combos():
    for start in range(0, 9, 3):
        yield (start, start + 1, start + 2)
    for start in range(3):
        yield(start, start+3, start + 6)
    yield(0, 4, 8)
    yield(2, 4, 6)

def _find_winner(tup):
    """returns None if no winner, true for X, false for 0"""
    for i1, i2, i3 in _winning_combos():
        v1, v2, v3 = tup[i1], tup[i2], tup[i3]
        if False is v1 is v2 is v3:
            return False 
        if True is v1 is v2 is v3:
            return True 
    return None 

def new_tic_tac_toe_board():
    return TicTacToeBoard(tup=(None,) * 9, turn=True, winner=None, terminal=False)

if __name__ == "__main__":
    play_game()
