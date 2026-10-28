import math
import random

import torch
import numpy as np
from flask import Flask, request, jsonify

from Train_net import Policy_network, get_device
from Game import Game 
from huggingface_hub import hf_hub_download
app = Flask(__name__)




class AI():

    def __init__(self):
        self.game = Game()
        self.device = get_device()
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

        model_path = hf_hub_download(
            repo_id="Jycccc111/Go-Policy",
            filename="best_policy_network_128.pth"
        )

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=True
        )
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if state_dict and next(iter(state_dict)).startswith("network."):
            state_dict = {key.removeprefix("network."): value for key, value in state_dict.items()}
        # Older checkpoints did not record architecture metadata. Infer it from
        # the first convolution and residual block keys when possible.
        size = checkpoint.get("size")
        if size is None and "0.weight" in state_dict:
            size = state_dict["0.weight"].shape[0]
        block_indices = {
            int(key.split(".")[0]) for key in state_dict
            if key.endswith(".conv1.weight") and key.split(".")[0].isdigit()
        }
        num_blocks = checkpoint.get("num_blocks")
        if num_blocks is None:
            num_blocks = len(block_indices)
        model = Policy_network(size=size or 64, num_blocks=num_blocks)
        # Train_net stores the Sequential submodule's state_dict under
        # model_state_dict, so its keys are "0.weight", "3.conv1.weight", etc.
        # Also accept checkpoints saved from the outer Policy_network wrapper.
        first_key = next(iter(state_dict), "")
        if first_key.startswith("network."):
            model.load_state_dict(state_dict)
        else:
            model.network.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.model = model

    def _policy_output(self, game, color=None):
        self.set_game(game)
        color = self.game.current_player.value if color is None else int(color)
        model_input = self.encode_board(color)

        model_input = torch.tensor(model_input, dtype=torch.float32)
        model_input = model_input.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(model_input).squeeze(0)

        return output

    def _legal_move_scores(self, output):
        candidates = []
        for row in range(19):
            for col in range(19):
                if self.game.islegal[row, col] == 1:
                    action = row * 19 + col
                    candidates.append((float(output[action].item()), row, col))
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        return candidates

    def recommend_moves(self, game, color=None, top_k=5):
        """Return the model's highest-scoring legal board points."""
        output = self._policy_output(game, color)
        return [
            {"row": row, "col": col, "score": score}
            for score, row, col in self._legal_move_scores(output)[:top_k]
        ]

    def predict_action(self, game, color=None):
        """Return the highest-scoring legal move, pass, or resignation."""
        output = self._policy_output(game, color)
        candidates = self._legal_move_scores(output)
        best_score = candidates[0][0] if candidates else float("-inf")
        pass_score = output[361].item()
        resign_score = output[362].item()

        if pass_score >= best_score and pass_score >= resign_score:
            return {"kind": "pass"}
        if resign_score >= best_score:
            return {"kind": "resign"}
        if candidates:
            # Keep play close to the policy's best move while allowing variety:
            # sample among the top five legal points with a softmax weighting.
            top_candidates = candidates[:5]
            temperature = 0.75
            max_score = top_candidates[0][0]
            weights = [
                math.exp((score - max_score) / temperature)
                for score, _, _ in top_candidates
            ]
            _, row, col = random.choices(
                top_candidates,
                weights=weights,
                k=1,
            )[0]
            return {"kind": "move", "row": row, "col": col}
        return {"kind": "pass"}

    def predict_move(self, game):
        """Backward-compatible coordinate API used by older callers."""
        action = self.predict_action(game)
        if action["kind"] == "pass":
            return 0, 0
        if action["kind"] == "resign":
            return -1, -1
        return action["row"], action["col"]
