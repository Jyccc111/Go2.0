import torch
import numpy as np
from flask import Flask, request, jsonify

from Train_net import Policy_network
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
        # Train_net stores the Sequential submodule's state_dict under
        # model_state_dict, so its keys are "0.weight", "3.conv1.weight", etc.
        # Also accept checkpoints saved from the outer Policy_network wrapper.
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        first_key = next(iter(state_dict), "")
        if first_key.startswith("network."):
            model.load_state_dict(state_dict)
        else:
            model.network.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.model = model

    def predict_action(self, game, color=None):
        """Return an unambiguous action: move, pass, or resign."""
        self.set_game(game)
        color = self.game.current_player.value if color is None else int(color)
        model_input = self.encode_board(color)

        model_input = torch.tensor(model_input, dtype=torch.float32)
        model_input = model_input.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(model_input).squeeze(0)

        candidates = []
        for x in range(19):
            for y in range(19):
                if self.game.islegal[x, y] == 1:
                    index = x * 19 + y
                    candidates.append((output[index].item(), x, y))

        candidates.sort(reverse=True)
        best_score = candidates[0][0] if candidates else float("-inf")
        pass_score = output[361].item()

        # The policy head's resign class is not a reliable game-outcome
        # evaluator. Do not end a game just because that class has the
        # highest logit; choose a legal move or pass instead.
        if pass_score > best_score:
            return {"kind": "pass"}
        if candidates:
            _, x, y = candidates[0]
            return {"kind": "move", "row": x, "col": y}
        return {"kind": "pass"}

    def predict_move(self, game):
        """Backward-compatible coordinate API used by older callers."""
        action = self.predict_action(game)
        if action["kind"] == "pass":
            return 0, 0
        if action["kind"] == "resign":
            return -1, -1
        return action["row"], action["col"]
