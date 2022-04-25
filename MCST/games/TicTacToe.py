import random
from Game import Game
import os 
import numpy as np

class TicTacToe(Game):
    def __init__(self, n):
        """n: the size of the board to create"""
        self.board = self.create_board(n)

    def create_board(self, n):
        """returns an nxn board"""
        board = []
        for _ in range(n):
            row = []

            for _ in range(n):
                row.append('-')
            
            board.append(row)
    
        return board 

    def get_p1(self):
        """chooses which player goes first, 
        TODO might want to default this in the future"""
        return random.randint(0,1)

    def fix_spot(self, row, col, player):
        """sets a given players symbol"""
        assert(row < len(self.board) and row >=0)
        assert(col < len(self.board) and col >= 0)
        if self.board[row][col] == "-":
            self.board[row][col] = player 
            return True
        else: 
            return False

    # TODO make not ugly
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

            # take input
            l = list(map(int, input("enter row and column numbers to fix spot: ").split()))
            while not len(l) == 2:
                l = list(map(int, input("enter row and column numbers to fix spot: ").split()))
            row, col = l[0], l[1]
            print()

            while not self.fix_spot(row - 1, col - 1, player):
                row, col = list(map(int, input("must be an open space: ").split()))

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
    TicTacToe(3).start()


        