import random
from games.Game import Game
# from Game import Game
import os 
import numpy as np

class TicTacToe(Game):
    def __init__(self, n=3, board=None, to_move=None):
        """n: the size of the board to create"""
        if board and to_move:
            self.board = board
            self.player = to_move

        else: 
            self.board = self.create_board(n)
            self.player = random.choice(["x", "o"])

    def create_board(self, n):
        """returns an nxn board"""
        board = []
        for _ in range(n):
            row = []

            for _ in range(n):
                row.append('-')
            
            board.append(row)
    
        return board 

    def get_state(self):
        return self.board

    def set_state(self, state):
        self.board = state

    def fix_spot(self, action, player):
        """sets a given players symbol"""
        row, col = action
        assert(row < len(self.board) and row >=0)
        assert(col < len(self.board) and col >= 0)
        if self.board[row][col] == "-":
            self.board[row][col] = player 
            self.switch_player()
            return True
        else: 
            return False

    def check_rows(self, board):
        """checks if we have a row winner"""
        for row in board:
            if len(set(row)) == 1:
                return row[0]
        return 0

    def check_diagonals(self, board):
        """checks if we have a diagonal winner"""
        if len(set([board[i][i] for i in range(len(board))])) == 1:
            return board[0][0]
        if len(set([board[i][len(board)-i-1] for i in range(len(board))])) == 1:
            return board[0][len(board)-1]
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
        return self.is_player_win(state, "x") or self.is_player_win(state, "o") or self.is_board_filled()

    def get_result(self, state, player):
        if self.is_player_win(state, player):
            return 1 
        elif self.is_player_win(state, self.get_other_player(player)):
            return -1
        else:
            return 0.25
        
    def is_board_filled(self):
        """checks if the board is filled"""
        for row in self.board:
            for item in row:
                if item == '-':
                    return False 
        return True 

    def switch_player(self):
        """toggles which player's turn it is"""
        if self.player == "x":
            self.player = "o"
            print("switching player from x to o")
        else:
            self.player = "x"    
            print("switching player from o to x")

    def get_other_player(self, player):
        """just tells who the other player is"""
        if player == "x":
            return "o"
        else:
            return "x"

    def get_legal_actions(self, state):
        moves = []
        for row in range(len(state)):
            for col in range(len(state[row])):
                if state[row][col] == "-":
                    moves.append((row, col))

        return moves

    def __repr__(self):
        """displays the board"""
        res = ""
        for row in self.board:
            for item in row:
               res += f"{item} "
            res += "\n"
        return  res

    def start(self):

        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"player {self.player}'s turn")
            print(self)

            # take input
            l = list(map(int, input("enter row and column numbers to fix spot: ").split()))
            while not len(l) == 2:
                l = list(map(int, input("enter row and column numbers to fix spot: ").split()))
            row, col = l[0], l[1]
            print()

            action = (row -1, col -1)
            while not self.fix_spot(action, self.player):
                row, col = list(map(int, input("must be an open space: ").split()))

            if self.is_player_win(self.board, self.player):
                print(f"player {self.player} wins!")
                break 
                
            if self.is_board_filled():
                print("draw!")
                break 

            player = self.switch_player()
        
        print()
        print(self)

if __name__ == "__main__":
    g = TicTacToe(3)
    print(g.get_legal_actions(g.get_state()))
    g.start()



        