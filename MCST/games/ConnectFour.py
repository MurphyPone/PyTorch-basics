import random
import os 
import numpy as np
from scipy.signal import convolve2d
from Game import Game

class ConnectFour(Game):

    def __init__(self, n=6):
        self.board = self.create_board(n)

    def create_board(self, n):
        """returns an nxn board"""
        board = []
        for _ in range(n):
            row = []

            for _ in range(n+1):
                row.append("-")
            
            board.append(row)
    
        return board 

    def get_p1(self):
        """chooses which player goes first, 
        TODO might want to default this in the future"""
        return random.randint(0,1)

    def fix_spot(self, col, player):
        assert(col < len(self.board) + 1 and col >= 0)
        """takes a column and drops the token down to the bottom"""
        # iterate from bottom up
        for row in range(len(self.board) -1, -1, -1):
            if self.board[row][col] == "-":
                self.board[row][col] = player 
                return True 

        return False 

    def is_player_win(self, board, player):
        def board_to_array(board):
            """Create a 2D array from your board, in which all of a player's tiles are set to 1, 
            and all empty/opponent tiles are set to 0."""
            result = []
            for r in range(len(self.board)):
                row = []

                for c in range(len(self.board[r])):
                    if self.board[r][c] == player:
                        row.append(1)
                    else:
                        row.append(0)
                
                result.append(row)

            return np.array(result)

        horizontal_kernel = np.array([[ 1, 1, 1, 1]])
        vertical_kernel = np.transpose(horizontal_kernel)
        diag1_kernel = np.eye(4, dtype=np.uint8)
        diag2_kernel = np.fliplr(diag1_kernel)

        detection_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]

        array = board_to_array(board)
        # if player == "x":
        #     player = 1
        # else: player = 0
        
        for kernel in detection_kernels:
            if (convolve2d(array == 1, kernel, mode="valid") == 4).any():
                return True
        return False

        



    def is_board_filled(self):
        """checks if the board is filled"""
        for row in self.board:
            for item in row:
                if item == '-':
                    return False 
        return True 

    def swap_player_turn(self, player):
        """toggles which player's turn it is"""
        return "x" if player == "o" else "o"

    def get_legal_moves(self):
        moves = []
        for row in len(self.board):
            for col in len(self.board[row]):
                if self.board[row][col] == "-":
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
        player = "x" if self.get_p1() == 1 else "o"

        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"player {player}'s turn")
            print(self)

            # TODO sanitize -> take input
            col = int(input("enter a column number to fix spot: "))
            print()

            while not self.fix_spot(col - 1, player):
                col = int(input("enter a column number to fix spot: "))

            if self.is_player_win(self.board, player):
                print(f"player {player} wins!")
                break 
                
            if self.is_board_filled():
                print("draw!")
                break 

            player = self.swap_player_turn(player)
        
        print()
        print(self)
            


if __name__ == "__main__": 
    ConnectFour(6).start()