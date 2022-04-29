from abc import ABC, abstractmethod

# TODO maybe this doesn't need to be an Abstract

class Game(ABC):

    @abstractmethod
    def switch_player(self):
        """determines how player switching occurs"""
        pass 

    @abstractmethod
    def get_other_player(self):
        """returns the other player without switching"""
        pass

    @abstractmethod
    def get_state(self):
        """returns a copy of the current state"""
        pass

    def set_state(self):
        """sets the current state"""
        pass

    @abstractmethod
    def is_player_win(self, player):
        """Checks if a given player has won"""
        pass

    @abstractmethod
    def get_legal_actions(self):
        """return a list of legal moves for the game"""
        pass

   

