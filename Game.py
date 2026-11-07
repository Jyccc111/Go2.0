from operator import index

import numpy as np
from enum import IntEnum


class Player(IntEnum):
    EMPTY = 0
    BLACK = 1
    WHITE = -1

def is_bound(rol, col):
        if rol == 0 or rol == 18 or col == 0 or col == 18:
            return True
        return False

def is_corner(rol, col):
    if (rol == 0 and col == 0) or (rol == 0 and col == 18) or (rol == 18 and col == 0) or (rol == 18 and col == 18):
        return True
    return False

def is_legal(rol, col):
    if rol < 0 or rol >= 19 or col < 0 or col >= 19:
        return False
    return True



class Board:
    def __init__(self):
        self.board = np.zeros((19, 19), dtype = np.int8)
    
    def set(self, rol, col, color):
        if color == 1:
            self.board[rol, col] = 1
        elif color == -1:
            self.board[rol, col] = -1
        else:
            self.board[rol, col] = 0
        
    def get(self, rol, col):
        return self.board[rol, col]

    def check_arond(self, rol, col, color):
        result_list = []
        if is_legal(rol+1, col) and self.get(rol+1, col) == color:
            result_list.append(1)
        elif is_legal(rol+1, col) and self.get(rol+1, col) == -color:
            result_list.append((-1))
        else:
            result_list.append((0))
        if is_legal(rol-1, col) and self.get(rol-1, col) == color:
            result_list.append(1)
        elif is_legal(rol-1, col) and self.get(rol-1, col) == -color:
            result_list.append((-1))
        else:
            result_list.append((0))
        if is_legal(rol, col+1) and self.get(rol, col+1) == color:
            result_list.append(1)
        elif is_legal(rol, col+1) and self.get(rol, col+1) == -color:
            result_list.append((-1))
        else:
            result_list.append((0))
        if is_legal(rol, col-1) and self.get(rol, col-1) == color:
            result_list.append(1)
        elif is_legal(rol, col-1) and self.get(rol, col-1) == -color:
            result_list.append((-1))
        else:
            result_list.append((0))
        return result_list
        

    def eat(self, rol, col):
        self.board[rol, col] = 0

    

class air:
    def __init__(self):
        self.board = np.zeros((19, 19), dtype = np.int8)
        self.air_updated = np.zeros((19, 19), dtype = np.int8)

    def initialize(self):
        for i in range(19):
            for j in range(19):
                self.board[i, j] = 4
                if is_bound(i,j):
                    self.board[i, j] -= 1
                if is_corner(i,j):
                    self.board[i, j] -= 1

    def update(self, rol, col):
        self.board[rol, col] = 0
        if is_legal(rol+1, col) and (self.board[rol+1, col] > 0) and self.air_updated[rol+1, col] == 0:
            self.board[rol+1, col] -= 1
        if is_legal(rol-1, col) and (self.board[rol-1, col] > 0) and self.air_updated[rol-1, col] == 0:
            self.board[rol-1, col] -= 1
        if is_legal(rol, col+1) and (self.board[rol, col+1] > 0) and self.air_updated[rol, col+1] == 0:
            self.board[rol, col+1] -= 1
        if is_legal(rol, col-1) and (self.board[rol, col-1] > 0) and self.air_updated[rol, col-1] == 0:
            self.board[rol, col-1] -= 1
        self.air_updated = np.zeros((19, 19), dtype = np.int8)

    def get(self, rol, col):
        return self.board[rol, col]

    def set(self, rol, col, value):
        self.board[rol, col] = value
        self.air_updated[rol, col] = 1
    
    def add(self, rol, col):
        self.board[rol, col] += 1

class Game: 

    def __init__(self):
        self.komi = 0
        self.board = Board()
        self.current_player = Player.BLACK
        self.trun = 1
        self.air = air()
        self.air.initialize()
        self.cluster_air =  np.zeros((19, 19), dtype = np.int8)
        self.has_one_air = np.zeros((19, 19), dtype = np.int8)
        self.is_checked = np.zeros((19, 19), dtype = np.int8)
        self.num_eat = np.zeros((2,19, 19), dtype = np.int8)
        self.eaten_checked = np.zeros((19, 19), dtype = np.int8)

        self.islegal = np.ones((19, 19), dtype = np.int8)
        self.lasteated = np.zeros((19, 19), dtype = np.int8)
        self.num_turns = 0
        self.turn_since = np.zeros((19, 19), dtype = np.int16)
        self.count = 0
        self.air_color = np.zeros((19, 19), dtype = np.int8)
        self.air_color_checked = np.zeros((19, 19), dtype = np.int8)
        self.num_eyes = np.zeros((19, 19), dtype = np.int8)
        self.num_eyes_checked = np.zeros((19, 19), dtype = np.int8)
        self.cluster_indexing = np.zeros((19, 19), dtype = np.int8)
        self.relation_mat = None
        self.relation_mat_clusters = None
        self.eye_board = np.zeros((19, 19), dtype = np.int8)

        self.black_eye_clusters = []
        self.white_eye_clusters = []
        self.undefined_eye_clusters = []
        self.black_clusters = []
        self.white_clusters = []
        self.air_clusters = []
        self.all_clusters = []
        self.black_num_eyes =[]
        self.white_num_eyes = []
        self.black_alive_indexes = []
        self.white_alive_indexes = []

    def set_komi(self, komi):
        self.komi = komi

    def reset(self):
        self.board.reset()
        self.current_player = Player.BLACK
        self.num_turns = 0
        self.turn_since = np.zeros((19, 19), dtype = np.int16)
        self.count = 0
        self.islegal = np.ones((19, 19), dtype = np.int8)
        self.lasteated = np.zeros((19, 19), dtype = np.int8)
        self.num_eat = np.zeros((2,19, 19), dtype = np.int8)
        self.eaten_checked = np.zeros((19, 19), dtype = np.int8)
        self.air_color = np.zeros((19, 19), dtype = np.int8)
        self.air_color_checked = np.zeros((19, 19), dtype = np.int8)
        self.num_eyes = np.zeros((19, 19), dtype = np.int8)
        self.num_eyes_checked = np.zeros((19, 19), dtype = np.int8)
        self.cluster_indexing = np.zeros((19, 19), dtype = np.int8)
        self.relation_mat = None
        self.relation_mat_clusters = None
        self.eye_board = np.zeros((19, 19), dtype = np.int8)
        self.black_eye_clusters = []
        self.white_eye_clusters = []
        self.undefined_eye_clusters = []
        self.black_clusters = []
        self.white_clusters = []
        self.air_clusters = []
        self.all_clusters = []
        self.black_num_eyes =[]
        self.white_num_eyes = []
        self.black_alive_indexes = []
        self.white_alive_indexes = []
    def mask(self,val, color):
        if color == 1:
            return val
        if color ==-1:
            return -val
    def is_self_eaten(self, rol, col, color):
        self.count = 0
        color_list = self.board.check_arond(rol, col, color)
        if not is_legal(rol+1, col) or (color_list[0] == -1 or (color_list[0] == 1 and self.cluster_air[rol+1, col] == 1)):
            self.count += 1
        if not is_legal(rol-1, col) or (color_list[1] == -1 or (color_list[1] == 1 and self.cluster_air[rol-1, col] == 1)):
            self.count += 1
        if not is_legal(rol, col+1) or (color_list[2] == -1 or (color_list[2] == 1 and self.cluster_air[rol, col+1] == 1)):
            self.count += 1
        if not is_legal(rol, col-1) or (color_list[3] == -1 or (color_list[3] == 1 and self.cluster_air[rol, col-1] == 1)):
            self.count += 1
        if self.count == 4:
            return False
        return True
    def update_legal_position(self):
        self.islegal = np.zeros((19, 19), dtype = np.int8)
        if self.current_player == Player.BLACK:
            layer = 0
            anti_layer = 1
        else:
            layer = 1
            anti_layer = 0
        for i in range(19):
            for j in range(19):
                if self.board.get(i,j) == 0:
                    self.islegal[i, j] = 1
                # Check suicide legality from the current player's perspective.
                # Passing the opponent color falsely treats friendly neighbors
                # as opposing stones and marks many valid points illegal.
                if not self.is_self_eaten(i, j, self.current_player.value):
                    self.islegal[i, j] = 0
                # num_eat[layer] stores capture moves available to this player.
                if self.num_eat[layer, i, j] > 0:
                    self.islegal[i, j] = 1
                if self.lasteated[i, j] == 1:
                    self.islegal[i, j] = 0
    def update_num_eat(self,rol,col):
        if self.current_player == Player.BLACK:
            layer = 0
            anti_layer = 1
        else:
            layer = 1
            anti_layer = 0
        self.num_eat[anti_layer,rol,col] = 0
    def find_eat(self, rol, col, color, num = 0, position = None, is_parent = True, eliminate_num_eat = False):
        if color == 1:
            layer = 0
            anti_layer = 1
        else:
            layer = 1
            anti_layer = 0
        self.is_checked[rol, col] = 1
        color_list = self.board.check_arond(rol, col, color)
        if is_legal(rol+1, col) and color_list[0] == 1 and self.is_checked[rol+1, col] == 0:
            position, num = self.find_eat(rol+1, col, color, num, position, False)
        elif is_legal(rol+1, col) and color_list[0] == 0:
            position = (rol+1, col)
        if is_legal(rol-1, col) and color_list[1] == 1 and self.is_checked[rol-1, col] == 0:
            position, num = self.find_eat(rol-1, col, color, num, position, False)
        elif is_legal(rol-1, col) and color_list[1] == 0:
            position = (rol-1, col)
        if is_legal(rol, col+1) and color_list[2] == 1 and self.is_checked[rol, col+1] == 0:
            position, num = self.find_eat(rol, col+1, color, num, position, False)
        elif is_legal(rol, col+1) and color_list[2] == 0:
            position = (rol, col+1)
        if is_legal(rol, col-1) and color_list[3] == 1 and self.is_checked[rol, col-1] == 0:
            position, num = self.find_eat(rol, col-1, color, num, position, False)
        elif is_legal(rol, col-1) and color_list[3] == 0:
            position = (rol, col-1)
        if eliminate_num_eat and position != None:
            x, y = position
            can_remove = True
            if is_legal(x+1, y) and self.is_checked[x+1, y] == 0 and self.board.get(x+1,y) == color and self.cluster_air[x+1,y] == 1:
                can_remove = False
            if is_legal(x-1, y) and self.is_checked[x-1, y] == 0 and self.board.get(x-1,y) == color and self.cluster_air[x-1,y] == 1:
                can_remove = False
            if is_legal(x, y+1) and self.is_checked[x, y+1] == 0 and self.board.get(x,y+1) == color and self.cluster_air[x,y+1] == 1:
                can_remove = False
            if is_legal(x, y-1) and self.is_checked[x, y-1] == 0 and self.board.get(x,y-1) == color and self.cluster_air[x,y-1] == 1:
                can_remove = False
            if can_remove:
                self.num_eat[anti_layer, x, y] = 0
        if not is_parent:
            return position, num + 1
        elif position != None:
            x, y = position
            if color == 1:
                return x, y, (num + 1)
            return x, y, (num + 1)
        else:
            return

    def eat(self, rol, col, color, isParent = True):
        if isParent:
            count = 1
        self.eaten_checked[rol, col] = 1
        self.cluster_air[rol, col] = 0
        color_list = self.board.check_arond(rol, col, color)
        if is_legal(rol+1, col) and color_list[0] == 1 and self.eaten_checked[rol+1, col] == 0:
            self.eat(rol+1, col, color, False)
            if isParent:
                count += 1
        elif is_legal(rol+1, col) and color_list[0] == -1 and self.cluster_air[rol+1, col] == 1:
            self.is_checked = np.zeros((19, 19), dtype = np.int8)
            self.find_eat(rol+1, col, -color, 0, None, True, True)
            self.is_checked = np.zeros((19, 19), dtype = np.int8)
        if is_legal(rol-1, col) and color_list[1] == 1 and self.eaten_checked[rol-1, col] == 0:
            self.eat(rol-1, col, color, False)
            if isParent:
                count += 1
        elif is_legal(rol-1, col) and color_list[1] == -1 and self.cluster_air[rol-1, col] == 1:
            self.is_checked = np.zeros((19, 19), dtype = np.int8)
            self.find_eat(rol-1, col, -color, 0, None, True, True)
            self.is_checked = np.zeros((19, 19), dtype = np.int8)
        if is_legal(rol, col+1) and color_list[2] == 1 and self.eaten_checked[rol, col+1] == 0:
            self.eat(rol, col+1, color, False)
            if isParent:
                count += 1
        elif is_legal(rol, col+1) and color_list[2] == -1 and self.cluster_air[rol, col+1] == 1:
            self.is_checked = np.zeros((19, 19), dtype = np.int8)
            self.find_eat(rol, col+1, -color, 0, None, True, True)
            self.is_checked = np.zeros((19, 19), dtype = np.int8)
        if is_legal(rol, col-1) and color_list[3] == 1 and self.eaten_checked[rol, col-1] == 0:
            self.eat(rol, col-1, color, False)
            if isParent:
                count += 1
        elif is_legal(rol, col-1) and color_list[3] == -1 and self.cluster_air[rol, col-1] == 1:
            self.is_checked = np.zeros((19, 19), dtype = np.int8)
            self.find_eat(rol, col-1, -color, 0, None, True, True)
            self.is_checked = np.zeros((19, 19), dtype = np.int8)
        self.board.eat(rol, col)
        self.is_checked = np.zeros((19, 19), dtype = np.int8)
        num_air = 4
        if is_corner(rol, col):
            num_air -= 1
        if is_bound(rol, col):
            num_air -= 1
        if is_legal(rol+1, col) and self.board.get(rol + 1, col) != 0 and self.is_checked[rol+1, col] == 0 and self.eaten_checked[rol+1, col] == 0:
            self.update_cluster_air(rol+1, col, val = 1, color = self.board.get(rol + 1, col), is_patent = False)
            num_air -= 1
                
        if is_legal(rol-1, col) and self.board.get(rol - 1, col) != 0 and self.is_checked[rol-1, col] == 0 and self.eaten_checked[rol-1, col] == 0:
            self.update_cluster_air(rol-1, col, val = 1, color = self.board.get(rol - 1, col), is_patent = False)
            num_air -= 1
                
        if is_legal(rol, col+1) and self.board.get(rol, col+1) != 0 and self.is_checked[rol, col+1] == 0 and self.eaten_checked[rol, col+1] == 0:
            self.update_cluster_air(rol, col+1, val = 1, color = self.board.get(rol, col+1), is_patent = False)
            num_air -= 1
                
        if is_legal(rol, col-1) and self.board.get(rol, col-1) != 0 and self.is_checked[rol, col-1] == 0 and self.eaten_checked[rol, col-1] == 0:
            self.update_cluster_air(rol, col-1, val = 1, color = self.board.get(rol, col-1), is_patent = False)
            num_air -= 1
        self.air.set(rol, col, num_air)

        self.is_checked = np.zeros((19, 19), dtype = np.int8)
        if isParent:
            if count == 1:
                self.lasteated[rol, col] = 1
            


    def calculate_find_can_eat(self, rol, col, color):
        if color == 1:
            layer = 0
        else:
            layer = 1
        self.is_checked[rol, col] = 1
        color_list = self.board.check_arond(rol, col, color)
        if is_legal(rol+1, col) and color_list[0] == -1 and self.is_checked[rol+1, col] == 0 and self.cluster_air[rol+1, col] == 1:
            x, y, num = self.find_eat(rol+1, col, -color, num = 0, position = None, is_parent = True)
            self.num_eat[layer,x, y] = num
        if is_legal(rol-1, col) and color_list[1] == -1 and self.is_checked[rol-1, col] == 0 and self.cluster_air[rol-1, col] == 1:
            x, y, num = self.find_eat(rol-1, col, -color, num = 0, position = None, is_parent = True)
            self.num_eat[layer,x, y] = num
        if is_legal(rol, col+1) and color_list[2] == -1 and self.is_checked[rol, col+1] == 0 and self.cluster_air[rol, col+1] == 1:
            x, y, num = self.find_eat(rol, col+1, -color, num = 0, position = None, is_parent = True)
            self.num_eat[layer,x, y] = num
        if is_legal(rol, col-1) and color_list[3] == -1 and self.is_checked[rol, col-1] == 0 and self.cluster_air[rol, col-1] == 1:
            x, y, num = self.find_eat(rol, col-1, -color, num = 0, position = None, is_parent = True)
            self.num_eat[layer,x, y] = num


        
    def update_cluster_air(self, rol, col, val, color, is_patent = True):
        if self.is_checked[rol, col] == 1:
            return
        self.is_checked[rol, col] = 1
        self.cluster_air[rol,col] += val
        color_list = self.board.check_arond(rol, col, color)
        if is_legal(rol+1, col) and color_list[0] == 1 and self.is_checked[rol+1, col] == 0:
            self.update_cluster_air(rol+1, col, val, color, False)
        if is_legal(rol-1, col) and color_list[1] == 1 and self.is_checked[rol-1, col] == 0:
            self.update_cluster_air(rol-1, col, val, color, False)
        if is_legal(rol, col+1) and color_list[2] == 1 and self.is_checked[rol, col+1] == 0:
            self.update_cluster_air(rol, col+1, val, color, False)
        if is_legal(rol, col-1) and color_list[3] == 1 and self.is_checked[rol, col-1] == 0:
            self.update_cluster_air(rol, col-1, val, color, False)
        if is_patent:
            self.is_checked = np.zeros((19, 19), dtype = np.int8)
            

    def calculate_cluster_air(self, rol, col, num, color):
        self.is_checked[rol, col] = 1
        self.cluster_air[rol, col] = 0
        color_list = self.board.check_arond(rol, col, color)
        if is_legal(rol+1, col) and color_list[0] == 1 and self.is_checked[rol+1, col] == 0:
            num = self.calculate_cluster_air(rol+1, col, num, color)
        elif is_legal(rol+1, col) and color_list[0] == 0 and self.is_checked[rol+1, col] == 0:
            num += 1
            self.is_checked[rol+1, col] = 1
        if is_legal(rol-1, col) and color_list[1] == 1 and self.is_checked[rol-1, col] == 0:
            num = self.calculate_cluster_air(rol-1, col, num, color)
        elif is_legal(rol-1, col) and color_list[1] == 0 and self.is_checked[rol-1, col] == 0:
            num += 1
            self.is_checked[rol-1, col] = 1
        if is_legal(rol, col+1) and color_list[2] == 1 and self.is_checked[rol, col+1] == 0:
            num = self.calculate_cluster_air(rol, col+1, num, color)
        elif is_legal(rol, col+1) and color_list[2] == 0 and self.is_checked[rol, col+1] == 0:
            num += 1
            self.is_checked[rol, col+1] = 1
        if is_legal(rol, col-1) and color_list[3] == 1 and self.is_checked[rol, col-1] == 0:
            num = self.calculate_cluster_air(rol, col-1, num, color)
        elif is_legal(rol, col-1) and color_list[3] == 0 and self.is_checked[rol, col-1] == 0:
            num += 1
            self.is_checked[rol, col-1] = 1

        return num

    def check_adjacent_positions(self, rol, col):
        if self.current_player == Player.BLACK:
            layer = 0
            anti_layer = 1
        else:
            layer = 1
            anti_layer = 0
        color_list = self.board.check_arond(rol, col, self.current_player.value)
        self.is_checked = np.zeros((19, 19), dtype = np.int8)
        self.is_checked[rol, col] = 1

        if is_legal(rol+1, col) and color_list[0] == -1:
            self.update_cluster_air(rol+1, col, -1, -self.current_player.value, False)

        if is_legal(rol-1, col) and color_list[1] == -1:
            self.update_cluster_air(rol-1, col, -1, -self.current_player.value, False)

        if is_legal(rol, col+1) and color_list[2] == -1:
            self.update_cluster_air(rol, col+1, -1, -self.current_player.value, False)

        if is_legal(rol, col-1) and color_list[3] == -1:
            self.update_cluster_air(rol, col-1, -1, -self.current_player.value, False)
        self.is_checked = np.zeros((19, 19), dtype = np.int8)
        if is_legal(rol+1, col) and self.cluster_air[rol+1, col] == 0 and color_list[0] == -1 and self.board.get(rol + 1, col) != 0:
            self.eat(rol+1, col, -self.current_player.value)
            self.eaten_checked = np.zeros((19, 19), dtype = np.int8)
            self.num_eat[layer,rol,col] = 0
        if is_legal(rol-1, col) and self.cluster_air[rol-1, col] == 0 and color_list[1] == -1 and self.board.get(rol - 1, col) != 0:
            self.eat(rol-1, col, -self.current_player.value)
            self.eaten_checked = np.zeros((19, 19), dtype = np.int8)
            self.num_eat[layer,rol,col] = 0
        if is_legal(rol, col+1) and self.cluster_air[rol, col+1] == 0 and color_list[2] == -1 and self.board.get(rol, col+1) != 0:
            self.eat(rol, col+1, -self.current_player.value)
            self.eaten_checked = np.zeros((19, 19), dtype = np.int8)
            self.num_eat[layer,rol,col] = 0
        if is_legal(rol, col-1) and self.cluster_air[rol, col-1] == 0 and color_list[3] == -1 and self.board.get(rol, col-1) != 0:
            self.eat(rol, col-1, -self.current_player.value)
            self.eaten_checked = np.zeros((19, 19), dtype = np.int8)
            self.num_eat[layer,rol,col] = 0
        self.is_checked = np.zeros((19, 19), dtype = np.int8)
        val = self.calculate_cluster_air(rol, col, 0, self.current_player.value)
        self.is_checked = np.zeros((19, 19), dtype = np.int8)
        if val == 1:
            x, y, num = self.find_eat(rol, col, self.current_player.value, 0, None, True)
            self.num_eat[anti_layer,x,y] = num
        self.is_checked = np.zeros((19, 19), dtype = np.int8)
        self.update_cluster_air(rol, col, val, self.current_player.value)
        self.is_checked = np.zeros((19, 19), dtype = np.int8)
        self.calculate_find_can_eat(rol, col, self.current_player.value)
        self.is_checked = np.zeros((19, 19), dtype = np.int8)
        self.update_num_eat(rol,col)


    def play(self, rol, col, color):
        if color == 1:
            self.current_player = Player.BLACK
        elif color == -1:
            self.current_player = Player.WHITE
        
        if self.islegal[rol, col] == 0:
            print("Illegal move. Please choose a different position.")
            return False
        
        else:
            self.is_checked[rol, col] = 1
            self.board.set(rol, col, self.current_player.value)
            self.check_adjacent_positions(rol, col)
            self.air.update(rol, col)
            self.lasteated = np.zeros((19, 19), dtype = np.int8)
            self.num_turns += 1
            self.update_turn_since(rol, col)
            self.update_legal_position()
            return True

    def get_black(self):
        result = np.zeros((19, 19), dtype = np.int8)
        for i in range(19):
            for j in range(19):
                if self.board.get(i,j) == 1:
                    result[i, j] = 1
        return result

    def get_white(self):
        result = np.zeros((19, 19), dtype = np.int8)
        for i in range(19):
            for j in range(19):
                if self.board.get(i,j) == -1:
                    result[i, j] = 1
        return result
    def update_turn_since(self, rol, col):
        for i in range(19):
            for j in range(19):
                if self.turn_since[i,j] > 0:
                    self.turn_since[i,j] += 1
        self.turn_since[rol, col] = 1
    def get_capture(self, color):
        if color == 1:
            layer = 0
            anti_layer = 1
        else:
            layer = 1
            anti_layer = 0
        
        result1 = np.zeros((19, 19), dtype = np.int8)
        result2 = np.zeros((19, 19), dtype = np.int8)
        result1[:,:] = self.num_eat[layer,:,:]
        result2[:,:] = self.num_eat[anti_layer,:,:]
        return result1,result2

    def is_pass(self):
        if self.current_player == Player.BLACK:
            self.current_player = Player.WHITE
        elif self.current_player == Player.WHITE:
            self.current_player = Player.BLACK
        self.lasteated = np.zeros((19, 19), dtype = np.int8)
        self.update_legal_position()

    def cal_result(self):
        black_win = True
        while True:
            self.create_clusters()
            self.construct_relation_mat()
            all_classified = self.cal_alive()
            if all_classified:
                break
            self.eliminate_dead_clusters()
        black_score = 0
        white_score = 0
        num_undefined = 0
        for i in range(19):
            for j in range(19):
                if self.board.get(i,j) == 1:
                    black_score += 1
                elif self.board.get(i,j) == -1:
                    white_score += 1
                else:
                    if self.eye_board[i, j] == 1:
                        black_score += 1
                    elif self.eye_board[i, j] == -1:
                        white_score += 1
                    else:
                        num_undefined += 1
        if num_undefined % 2 == 1:
            if self.current_player == Player.WHITE:
                black_score += 1
            else:
                white_score += 1
        black_score += num_undefined // 2
        white_score += num_undefined // 2
        white_result = white_score + self.komi
        if white_result > black_score:
            black_win = False
        return black_score, white_score, black_win 



    def create_clusters(self):
        self.black_clusters = []
        self.white_clusters = []
        self.air_clusters = []
        self.is_checked = np.zeros((19, 19), dtype = np.int8)
        for i in range(19):
            for j in range(19):
                if self.board.get(i,j) == 1 and self.is_checked[i, j] == 0:
                    self.update_clusters(i, j, 1)
                elif self.board.get(i,j) == -1 and self.is_checked[i, j] == 0:
                    self.update_clusters(i, j, -1)
                elif self.board.get(i,j) == 0 and self.is_checked[i, j] == 0:
                    self.update_clusters(i, j, 0)
        self.is_checked = np.zeros((19, 19), dtype = np.int8)
    def update_clusters(self, row, col, color, is_root = True):

        if color == 1:
            if is_root:
                self.black_clusters.append([])
                self.black_clusters[-1].append([row, col])
            else:
                self.black_clusters[-1].append([row, col])
        elif color == -1:
            if is_root:
                self.white_clusters.append([])
                self.white_clusters[-1].append([row, col])
            else:
                self.white_clusters[-1].append([row, col])
        elif color == 0:
            if is_root:
                self.air_clusters.append([])
                self.air_clusters[-1].append([row, col])
            else:
                self.air_clusters[-1].append([row, col])
        self.is_checked[row, col] = 1
        if is_legal(row + 1, col) and self.board.get(row + 1, col) == color and self.is_checked[row + 1, col] == 0:
                    self.update_clusters(row + 1, col, color, is_root=False)
        if is_legal(row - 1, col) and self.board.get(row - 1, col) == color and self.is_checked[row - 1, col] == 0:
                    self.update_clusters(row - 1, col, color, is_root=False)
        if is_legal(row, col + 1) and self.board.get(row, col + 1) == color and self.is_checked[row, col + 1] == 0:
                    self.update_clusters(row, col + 1, color, is_root=False)
        if is_legal(row, col - 1) and self.board.get(row, col - 1) == color and self.is_checked[row, col - 1] == 0:
                    self.update_clusters(row, col - 1, color, is_root=False)
        

    def construct_relation_mat(self):
        self.black_eye_clusters = []
        self.white_eye_clusters = []
        self.undefined_eye_clusters = []
        self.all_clusters = self.black_clusters.copy()
        self.all_clusters.extend(self.white_clusters)
        self.relation_mat_clusters = np.zeros((len(self.all_clusters), len(self.all_clusters)), dtype = np.int8)
        
        for i in range(len(self.all_clusters)):
            for j in range(len(self.all_clusters[i])):
                self.cluster_indexing[self.all_clusters[i][j][0], self.all_clusters[i][j][1]] = i
        for i in range(len(self.air_clusters)):
            for j in range(len(self.air_clusters[i])):
                self.cluster_indexing[self.air_clusters[i][j][0], self.air_clusters[i][j][1]] = i
        self.relation_mat = np.zeros((len(self.all_clusters), len(self.air_clusters)), dtype = np.int8)
        for row in range(19):
            for col in range(19):
                if self.board.get(row, col) != 0:
                    if is_legal(row, col + 1) and self.board.get(row, col + 1) == 0:
                        self.relation_mat[self.cluster_indexing[row, col], self.cluster_indexing[row, col + 1]] = 1
                    elif is_legal(row, col + 1) and self.board.get(row, col + 1) != self.board.get(row, col):
                        self.relation_mat_clusters[self.cluster_indexing[row, col], self.cluster_indexing[row, col + 1]] = 1
                    if is_legal(row, col - 1) and self.board.get(row, col - 1) == 0:
                        self.relation_mat[self.cluster_indexing[row, col], self.cluster_indexing[row, col - 1]] = 1
                    elif is_legal(row, col - 1) and self.board.get(row, col - 1) != self.board.get(row, col):
                        self.relation_mat_clusters[self.cluster_indexing[row, col], self.cluster_indexing[row, col - 1]] = 1
                    if is_legal(row + 1, col) and self.board.get(row + 1, col) == 0:
                        self.relation_mat[self.cluster_indexing[row, col], self.cluster_indexing[row + 1, col]] = 1
                    elif is_legal(row + 1, col) and self.board.get(row + 1, col) != self.board.get(row, col):
                        self.relation_mat_clusters[self.cluster_indexing[row, col], self.cluster_indexing[row + 1, col]] = 1
                    if is_legal(row - 1, col) and self.board.get(row - 1, col) == 0:
                        self.relation_mat[self.cluster_indexing[row, col], self.cluster_indexing[row - 1, col]] = 1
                    elif is_legal(row - 1, col) and self.board.get(row - 1, col) != self.board.get(row, col):
                        self.relation_mat_clusters[self.cluster_indexing[row, col], self.cluster_indexing[row - 1, col]] = 1
        black_mat = self.relation_mat[:len(self.black_clusters), :]
        white_mat = self.relation_mat[len(self.black_clusters):, :]
        self.black_num_eyes = np.zeros(len(self.black_clusters))
        self.white_num_eyes = np.zeros(len(self.white_clusters))
        for i in range(len(self.air_clusters)):
            adj_black = False
            adj_white = False
            eye_type = 0
            if not np.all(black_mat[:, i] == 0):
                adj_black = True
            if not np.all(white_mat[:, i] == 0):
                adj_white = True
            if adj_black and adj_white:
                self.undefined_eye_clusters.append(self.air_clusters[i])
                for j in range(len(self.air_clusters[i])):
                    self.eye_board[self.air_clusters[i][j][0], self.air_clusters[i][j][1]] = 0
            elif adj_black:
                for j in range(len(self.black_clusters)):
                    if self.relation_mat[j, i] == 1:
                        self.black_num_eyes[j] += self.eye_classifier(self.air_clusters[i])
                self.black_eye_clusters.append(self.air_clusters[i])
                for j in range(len(self.air_clusters[i])):
                    self.eye_board[self.air_clusters[i][j][0], self.air_clusters[i][j][1]] = 1
            elif adj_white:
                for j in range(len(self.white_clusters)):
                    if self.relation_mat[j+len(self.black_clusters), i] == 1:
                        self.white_num_eyes[j] += self.eye_classifier(self.air_clusters[i])
                self.white_eye_clusters.append(self.air_clusters[i])
                for j in range(len(self.air_clusters[i])):
                    self.eye_board[self.air_clusters[i][j][0], self.air_clusters[i][j][1]] = -1

    def eye_classifier(self, cluster):
        if len(cluster) <= 3:
            return 1
        if len(cluster) > 6:
            return 2 
        degree_list = np.zeros(len(cluster))
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                if (abs(cluster[i][0] - cluster[j][0]) + abs(cluster[i][1] - cluster[j][1]))== 1:
                    degree_list[i] += 1
                    degree_list[j] += 1

        k = len(cluster) - degree_list.max()
        c = 0 
        for i in range(len(degree_list)):
            if degree_list[i] > 1:
                c += 1

        if k <= 1 or (k <= 2 and c >= 4):
            return 1
        return 2

    def cal_alive(self):
        self.black_alive_indexes = np.zeros(len(self.black_clusters), dtype = np.int8)
        self.white_alive_indexes = np.zeros(len(self.white_clusters), dtype = np.int8)
        for i in range(len(self.black_clusters)):
            if self.black_num_eyes[i] >= 2:
                self.black_alive_indexes[i] = 1
        for i in range(len(self.white_clusters)):
            if self.white_num_eyes[i] >= 2:
                self.white_alive_indexes[i] = 1
        relation_mat = np.dot(self.relation_mat,self.relation_mat.T)
        black_mat = relation_mat[:len(self.black_clusters), :len(self.black_clusters)]
        white_mat = relation_mat[len(self.black_clusters):, len(self.black_clusters):]
        self.black_alive_indexes = np.dot(black_mat, self.black_alive_indexes)
        self.white_alive_indexes = np.dot(white_mat, self.white_alive_indexes)
        all_clear = True
        for i in range(len(self.black_alive_indexes)):
            if self.black_alive_indexes[i] > 0:
                self.black_alive_indexes[i] = 1
            else:
                all_clear = False
        for i in range(len(self.white_alive_indexes)):
            if self.white_alive_indexes[i] > 0:
                self.white_alive_indexes[i] = 1
            else:
                all_clear = False
        return all_clear

    def eliminate_dead_clusters(self):
        next_player = -self.current_player.value
        black_min_index = 0
        black_min_value = np.inf
        black_min_eyes = 1
        white_min_index = 0
        white_min_value = np.inf
        white_min_eyes = 1
        exist_dead_black = False
        exist_dead_white = False
        for i in range(len(self.black_alive_indexes)):
            if self.black_alive_indexes[i] == 0:
                exist_dead_black = True
                if self.black_num_eyes[i] < black_min_eyes:
                    row, col = self.black_clusters[i][0]
                    black_min_value = self.cluster_air[row, col]
                    black_min_index = i
                    black_min_eyes = self.black_num_eyes[i]
                elif self.black_num_eyes[i] == black_min_eyes:
                    row, col = self.black_clusters[i][0]
                    if black_min_value > self.cluster_air[row, col]:
                        black_min_value = self.cluster_air[row, col]
                        black_min_index = i
                        black_min_eyes = self.black_num_eyes[i]
        
        for i in range(len(self.white_alive_indexes)):
            if self.white_alive_indexes[i] == 0:
                exist_dead_white = True
                if self.white_num_eyes[i] < white_min_eyes:
                    row, col = self.white_clusters[i][0]
                    white_min_value = self.cluster_air[row, col]
                    white_min_index = i
                    white_min_eyes = self.white_num_eyes[i]
                elif self.white_num_eyes[i] == white_min_eyes:
                    row, col = self.white_clusters[i][0]
                    if white_min_value > self.cluster_air[row, col]:
                        white_min_value = self.cluster_air[row, col]
                        white_min_index = i
                        white_min_eyes = self.white_num_eyes[i]
        if not exist_dead_black:
            self.eliminate_cluster(white_min_index, -1)
        elif not exist_dead_white:
            self.eliminate_cluster(black_min_index, 1)
        else:
            if white_min_eyes < black_min_eyes:
                self.eliminate_cluster(white_min_index, -1)
            elif black_min_eyes < white_min_eyes:
                self.eliminate_cluster(black_min_index, 1)
            else:
                if white_min_value < black_min_value:
                    self.eliminate_cluster(white_min_index, -1)
                elif black_min_value < white_min_value:
                    self.eliminate_cluster(black_min_index, 1)
                else:
                    if self.current_player.value == 1:
                        self.eliminate_cluster(white_min_index, -1)
                    else:
                        self.eliminate_cluster(black_min_index, 1)

    def eliminate_cluster(self, index, color):
        if color == 1:
            cluster = self.black_clusters[index]
        else:
            cluster = self.white_clusters[index]
        for row, col in cluster:
            self.board.set(row, col, 0)
        self.calculate_new_air(cluster)

        

    def calculate_new_air(self,cluster):
        for i in range(len(cluster)):
            row, col = [cluster[i][0], cluster[i][1]]
            self.is_checked = np.zeros((19, 19), dtype = np.int8)
            if is_legal(row+1, col) and self.board.get(row+1, col) != 0:
                self.update_cluster_air(row+1, col, 1, self.board.get(row+1, col))
            if is_legal(row-1, col) and self.board.get(row-1, col) != 0:
                self.update_cluster_air(row-1, col, 1, self.board.get(row-1, col))
            if is_legal(row, col+1) and self.board.get(row, col+1) != 0:
                self.update_cluster_air(row, col+1, 1, self.board.get(row, col+1))
            if is_legal(row, col-1) and self.board.get(row, col-1) != 0:
                self.update_cluster_air(row, col-1, 1, self.board.get(row, col-1))
            self.is_checked = np.zeros((19, 19), dtype = np.int8)