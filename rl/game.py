"""19x19 adapter over Game; no UI, network downloads or AI construction."""
import math
import copy
import numpy as np
from Game import Game, Player

PASS = 361
RESIGN = 362
MIN_TRAIN_MOVES = 200


def encode_board(game, color):
    """Same absolute-color eight planes and int16 encoding as AI.encode_board."""
    x = np.zeros((8, 19, 19), dtype=np.int16)
    x[0], x[1] = game.get_black(), game.get_white()
    x[2], x[3], x[4] = game.turn_since, game.air.board, game.cluster_air
    x[5], x[6] = game.get_capture(color)
    x[7] = game.islegal
    return x



class GoAdapter:
    def __init__(self, komi=7.5, max_moves=500):
        if not math.isfinite(komi) or max_moves < 2:
            raise ValueError('finite komi and max_moves >= 2 required')
        self.game = Game()  # Game.reset currently calls nonexistent Board.reset.
        self.game.set_komi(komi)
        self.max_moves = max_moves
        self.moves = self.passes = 0
        self.done = self.truncated = False
        self.resigned_by = None
        self._rewards = None
        self._result = None
        self.history = []
        self._refresh()

    @property
    def player(self):
        return int(self.game.current_player)

    def _refresh(self):
        self.game.update_legal_position()
        # Defensively exclude occupied points even if Game's capture map is stale.
        self.game.islegal[self.game.board.board != 0] = 0

    def observe(self):
        return encode_board(self.game, self.player)

    def legal_mask(self):
        mask = np.zeros(363, dtype=bool)
        if not self.done:
            mask[:361] = self.game.islegal.reshape(-1) == 1
            mask[PASS] = True
            mask[RESIGN] = True
        return mask

    def step(self, action):
        if (isinstance(action, bool) or not isinstance(action, (int, np.integer))
                or not 0 <= action < 363 or not self.legal_mask()[action]):
            raise ValueError('illegal action or finished game')
        color = self.player
        if action == RESIGN:
            self.resigned_by = color
        elif action == PASS:
            self.game.is_pass()  # clears simple-ko restriction and changes side
            self.passes += 1
        else:
            if not self.game.play(*divmod(int(action), 19), color):
                raise RuntimeError('Game rejected masked legal action')
            self.game.current_player = Player(-color)
            self.passes = 0
        self.history.append({'color': color, 'action': int(action)})
        self.moves += 1
        self.done = self.passes == 2 or self.resigned_by is not None
        self.truncated = self.moves >= self.max_moves and not self.done
        self.done |= self.truncated
        self._refresh()

    @property
    def skip_reason(self):
        if self.truncated:
            return 'move_limit'
        if self.done and self.moves < MIN_TRAIN_MOVES and self.resigned_by is None:
            return 'two_passes_before_200'
        return None

    def result(self):
        """Adjudication metadata shared by training, images and value labels."""
        if not self.done:
            raise ValueError('game has not ended')
        if self._result is None:
            reason = 'move_limit' if self.truncated else (
                'resignation' if self.resigned_by is not None else 'two_passes')
            official_winner = -self.resigned_by if self.resigned_by is not None else None
            winner = None
            source = None
            black_score = white_score = None
            if self.skip_reason is None:
                if self.moves < MIN_TRAIN_MOVES:
                    # A short resignation has a known winner without dead/alive scoring.
                    winner, source = official_winner, 'resignation'
                else:
                    # Keep all of the user's original dead/alive logic and tie convention.
                    scored_game = copy.deepcopy(self.game)
                    black_score, white_score, black_win = scored_game.cal_result()
                    black_score, white_score = float(black_score), float(white_score)
                    winner, source = (1 if black_win else -1), 'Game.cal_result'
            result = ('B' if winner == 1 else 'W') + (
                '+R' if source == 'resignation' else '+SCORE') if winner is not None else 'UNSCORED'
            self._result = dict(
                moves=self.moves, termination=reason, resigned_by=self.resigned_by,
                winner=winner, scored_winner=winner if source == 'Game.cal_result' else None,
                official_winner=official_winner,
                official_result=(('B' if official_winner == 1 else 'W') + '+R')
                if official_winner is not None else None,
                result=result, result_source=source, value_label=int(winner == 1)
                if winner is not None else None,
                black_score=black_score, white_score=white_score, komi=self.game.komi,
                skip_reason=self.skip_reason, truncated=self.truncated,
                encoding_player=self.player, history=list(self.history),
                max_moves=self.max_moves, consecutive_passes=self.passes,
                pass_counts={str(color): sum(h['color'] == color and h['action'] == PASS
                                            for h in self.history) for color in (1, -1)})
        return copy.deepcopy(self._result)

    def rewards(self):
        """Early resignation uses its result; later resignation is checked by score."""
        result = self.result()
        if self.skip_reason is not None:
            raise ValueError('skipped games have no training rewards')
        if self._rewards is None:
            black = float(result['winner'])
            self._rewards = {1: black, -1: -black}
            if self.resigned_by is not None and self.moves >= MIN_TRAIN_MOVES:
                # Reward correct resignation, reverse mistaken resignation. Before
                # 200, the resignation itself defines the loss, so keep reward -1.
                color = self.resigned_by
                self._rewards[(color, RESIGN)] = -self._rewards[color]
        return dict(self._rewards)
