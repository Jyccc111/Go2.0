import torch
import numpy as np
from flask import Flask, request, jsonify

from Train_net import Policy_network
from sgf_loader import encode_board
from Game import Game 
from huggingface_hub import hf_hub_download
app = Flask(__name__)




class AI():

    def __init__(self):
        self.game = Game()
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cpu"
            )
        self.load_model()

    def set_game(self, game):
        self.game = game 
        
        
    def encode_board(self, color):
    
        size = 19
    
        x = np.zeros(
            (8, size, size),
            dtype=np.int16
        )
    
        x[0,:,:] = self.game.get_black()
        x[1,:,:] = self.game.get_white()
        x[2,:,:] = self.game.turn_since
        x[3,:,:] = self.game.air.board
        x[4,:,:] = self.game.cluster_air
        x[5,:,:], x[6,:,:] = self.game.get_capture(color)
        x[7,:,:] = self.game.islegal
        return x

    def load_model(self):

        model = Policy_network()

        model_path = hf_hub_download(
            repo_id="Jycccc111/Go-Policy",
            filename="best_policy_network.pth"
        )

        checkpoint = torch.load(
            model_path,
        map_location=self.device,
            weights_only=True
        )

        model.load_state_dict(checkpoint)
        model.to(self.device)
        model.eval()
        self.model = model

    def predict_move(self, game):
        self.set_game(game)
        model_input = self.encode_board(self.game.current_player.value)

        model_input = torch.tensor(
        model_input,
        dtype = torch.float32
        )

        model_input = model_input.unsqueeze(0)

        model_input = model_input.to(self.device)

        with torch.no_grad():
            output = self.model(model_input)
        output = output.squeeze(0)

        candidates = []

        for x in range(19):
            for y in range(19):

            # 空位置才考虑
                if self.game.islegal[x, y] == 1:
                    index = x * 19 + y

                    candidates.append(
                        (output[index].item(), x, y)
                )

    # 按网络分数从高到低排序
        candidates.sort(
            reverse=True
        )

       
        score, x, y  = candidates[0]
        if output[362].item() > score and output[362].item() > output[363].item():
            return 0, 0
        elif output[363].item() > score and output[363].item() > output[362].item():
            return -1, -1
        else:
            return x, y 

        

