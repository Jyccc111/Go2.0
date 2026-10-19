from Game import Game, Player
from AI import AI 

class Type_player(IntEnum):
    AI = 0
    HUMAN = 1

class GameSession():

    def __init__(self):
        self.player_black = None
        self.player_white = None
        self.AI_black = None 
        self.AI_white = None
        self.komi = 0
        self.game = Game()
        self.Player = None 

    def initialize(self, player_black, player_white, komi):
        self.player_black = player_black
        self.player_white = player_white
        self.komi = komi
        if self.player_black.value == 0:
            self.AI_black = AI()
        if self.player_white.value == 0:
            self.AI_white = AI()
        self.Player = Player.BLACK
    def auto_play(self):
        if self.player_black.value == 0 and self.player_white.value == 0: