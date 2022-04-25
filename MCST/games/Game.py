from abc import ABC, abstractmethod

class Game(ABC):

    @abstractmethod
    def is_player_win(self, board, player):
        pass 

    @abstractmethod
    def get_legal_moves(self):
        """return a list of legal moves for the game"""
        pass

   

